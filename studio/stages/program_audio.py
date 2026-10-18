"""Duration-preserving audio crossfades across program edit joins.

Each piece is decoded with a handle on both sides of its frame-quantized
span. Adjacent pieces overlap by two handles centred on the join and are
summed with complementary gains, so the program keeps its exact length and
video stays in sync. Joins that continue the same timeline instant (camera
changes inside one retained range) use a linear fade; real edit joins use a
constant-power fade because the material on each side is uncorrelated.
"""

from __future__ import annotations

import subprocess
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from studio.core.media import build_atempo_chain, seek_input

SAMPLE_RATE = 48000
CHANNELS = 2
_BLOCK = SAMPLE_RATE * 10


@dataclass(frozen=True, slots=True)
class PartAudio:
    """One program piece: ``samples`` output samples plus optional handles."""

    audio: Path | None
    source_in_s: float
    rate: float
    samples: int
    continuous_with_previous: bool = False


def join_gains(length: int, *, continuous: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (fade_out, fade_in) gains over an overlap of ``length`` samples."""
    if length <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    position = (np.arange(length, dtype=np.float64) + 0.5) / length
    if continuous:
        fade_in = position
    else:
        fade_in = np.sin(position * np.pi / 2)
    fade_out = np.sqrt(1 - fade_in**2) if not continuous else 1 - fade_in
    return fade_out.astype(np.float32), fade_in.astype(np.float32)


def handle_samples(parts: list[PartAudio], crossfade_ms: float) -> int:
    """Half the overlap; shrinks so every piece keeps a non-negative middle."""
    if crossfade_ms <= 0 or len(parts) < 2:
        return 0
    handle = round(crossfade_ms / 2000 * SAMPLE_RATE)
    shortest = min(part.samples for part in parts)
    return max(0, min(handle, shortest // 2))


def decode_part(part: PartAudio, pre: int, post: int, scratch: Path) -> np.ndarray:
    """Decode ``pre + samples + post`` stereo int16 frames into a memory map."""
    total = pre + part.samples + post
    raw = scratch / "part.s16"
    frame = 2 * CHANNELS
    missing = 0
    with raw.open("wb") as handle:
        if part.audio is not None and total:
            pre_s = pre / SAMPLE_RATE
            # Output seconds map to source seconds through the placement rate.
            source_start = part.source_in_s - pre_s / part.rate
            if source_start < 0:
                missing = min(pre, round(-source_start * part.rate * SAMPLE_RATE))
                source_start = 0.0
            handle.write(bytes(missing * frame))
            handle.flush()
            duration = (total - missing) / SAMPLE_RATE / part.rate
            seek, source_start = seek_input(source_start)
            filters = [
                f"atrim=start={source_start:.9f}:duration={duration + 0.05:.9f}",
                "asetpts=PTS-STARTPTS",
                *build_atempo_chain(1 / part.rate),
                f"aresample={SAMPLE_RATE}",
                "aformat=sample_fmts=s16:channel_layouts=stereo",
            ]
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", *seek, "-i", str(part.audio), "-vn",
                 "-af", ",".join(filters), "-f", "s16le", "-ac", str(CHANNELS),
                 "-ar", str(SAMPLE_RATE), "-"],
                stdout=handle, stderr=subprocess.PIPE,
            )
            if result.returncode:
                raise RuntimeError(
                    f"program audio decode failed: {result.stderr.decode()[-2000:]}"
                )
        # Short decodes are padded with silence; overlong ones are cut.
        handle.truncate(total * frame)
    if total == 0:
        return np.zeros((0, CHANNELS), dtype="<i2")
    return np.memmap(raw, dtype="<i2", mode="r", shape=(total, CHANNELS))


Decoder = Callable[[PartAudio, int, int, Path], np.ndarray]


def mix_parts(
    parts: list[PartAudio], output: Path, crossfade_ms: float, decode: Decoder = decode_part
) -> dict[str, object]:
    """Write a 48 kHz stereo WAV of exactly ``sum(part.samples)`` frames."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix=".audio-", dir=output.parent) as temporary:
        return _mix(parts, output, crossfade_ms, decode, Path(temporary))


def _floats(block: np.ndarray) -> np.ndarray:
    return np.asarray(block, dtype=np.float32) / 32768


def _mix(
    parts: list[PartAudio], output: Path, crossfade_ms: float, decode: Decoder, scratch: Path
) -> dict[str, object]:
    handle = handle_samples(parts, crossfade_ms)
    overlap = 2 * handle
    pending: np.ndarray | None = None
    written = 0
    joins = {"edit": 0, "continuous": 0}
    with wave.open(str(output), "wb") as target:
        target.setnchannels(CHANNELS)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)

        def emit(block: np.ndarray) -> None:
            nonlocal written
            clipped = np.clip(block, -1.0, 32767 / 32768)
            target.writeframes((clipped * 32768).astype("<i2").tobytes())
            written += len(block)

        for index, part in enumerate(parts):
            first, last = index == 0, index == len(parts) - 1
            pre = 0 if first else handle
            post = 0 if last else handle
            data = decode(part, pre, post, scratch)
            if len(data) != pre + part.samples + post:
                raise RuntimeError("program audio part has an unexpected length")
            cursor = 0
            if pending is not None:
                fade_out, fade_in = join_gains(
                    overlap, continuous=part.continuous_with_previous
                )
                joins["continuous" if part.continuous_with_previous else "edit"] += 1
                emit(pending * fade_out[:, None] + _floats(data[:overlap]) * fade_in[:, None])
                cursor = overlap
            stop = len(data) - (0 if last else overlap)
            for block in _blocks(cursor, stop):
                emit(_floats(data[block]))
            pending = None if last else _floats(data[stop:])
            del data
    expected = sum(part.samples for part in parts)
    if written != expected:
        raise RuntimeError("program audio length does not match the video")
    return {
        "audio_crossfade_ms": round(overlap / SAMPLE_RATE * 1000, 3),
        "audio_join_policy": "constant-power edit joins, linear continuous joins",
        "audio_joins": joins,
        "audio_samples": written,
    }


def _blocks(start: int, stop: int) -> Iterator[slice]:
    for position in range(start, stop, _BLOCK):
        yield slice(position, min(stop, position + _BLOCK))
