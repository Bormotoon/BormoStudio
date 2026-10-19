"""Camera sound where the recorder heard nothing.

A recorder whose input was muted or unplugged writes exact zeros, and a voice
rendered from it under a camera is silent there although the camera heard the
room. Playback (program, previews, reels) takes the camera's own sound for those
stretches, brought to the voice's level, and the timeline transcript takes the
cameras' words there. The sync voices stay untouched: they are what sync was
measured on, and exports reference them.
"""

from __future__ import annotations

import subprocess
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from studio.core.media import probe
from studio.core.project import AssetRole, Project
from studio.core.timeline import file_to_timeline

# Exact zeros this long are a dead input (or no recorder at all), not a dropout
# between two samples; shorter ones are left as they are.
DEAD_S = 1.0
# Zero runs this close together are one stretch (a stray sample between them).
JOIN_S = 0.05
JOIN_RUN_FRAMES = 64
# Fades of the camera sound into and out of a filled stretch.
FILL_FADE_S = 0.05
# Level matching: 1 s windows where both the voice and the camera are audible.
LEVEL_WINDOW_S = 1.0
LEVEL_RATE = 16000
LEVEL_MIN_WINDOWS = 10
VOICE_AUDIBLE_DB = -50.0
CAMERA_AUDIBLE_DB = -70.0
GAIN_LIMITS_DB = (-20.0, 40.0)
CHUNK_FRAMES = 1 << 18
PLAYBACK_PREFIX = "voice:"

Run = tuple[int, int]
Spans = list[tuple[float, float]]


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int
    codec: str


def audio_format(path: Path) -> AudioFormat:
    info = probe(path)
    codec = info.audio_codec or ""
    return AudioFormat(int(info.audio_sample_rate or 48000), int(info.audio_channels or 1),
                       codec if codec.startswith("pcm_") else "pcm_s24le")


def _map(stream: int | None) -> list[str]:
    return ["-map", f"0:{stream}" if stream is not None else "0:a:0", "-vn"]


def _frames(
    command: list[str], channels: int, frames: int = CHUNK_FRAMES,
) -> Generator[np.ndarray, None, None]:
    """Decoded ``(n, channels)`` float32 chunks of an ffmpeg ``f32le`` pipe."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    size = 4 * channels
    try:
        while True:
            raw = process.stdout.read(frames * size)
            if not raw:
                break
            usable = len(raw) - len(raw) % size
            yield np.frombuffer(raw[:usable], dtype="<f4").reshape(-1, channels)
    finally:
        process.stdout.close()
        error = process.stderr.read().decode(errors="replace") if process.stderr else ""
        process.wait()
    if process.returncode:
        raise RuntimeError(f"audio decoding failed: {error.strip()}")


def zero_runs(path: Path, *, min_s: float = DEAD_S, stream: int | None = None,
              audio: AudioFormat | None = None) -> list[Run]:
    """``[start, end)`` frame ranges where every channel of ``path`` is exactly zero."""
    audio = audio or audio_format(path)
    minimum = max(1, round(min_s * audio.sample_rate))
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), *_map(stream),
               "-f", "f32le", "-"]
    runs: list[Run] = []
    open_start: int | None = None
    position = 0
    for chunk in _frames(command, audio.channels):
        silent = ~chunk.any(axis=1)
        edges = np.diff(np.concatenate([[open_start is not None], silent, [False]])
                        .astype(np.int8))
        starts = (np.flatnonzero(edges == 1) + position).tolist()
        ends = (np.flatnonzero(edges == -1) + position).tolist()
        if open_start is not None:
            starts.insert(0, open_start)
        open_start = None
        if len(silent) and silent[-1]:
            # The last run goes on in the next chunk (its end here is artificial).
            open_start = starts.pop()
            ends.pop()
        # Short runs are kept until joining: two long ones may sit around a stray sample.
        runs.extend((start, end) for start, end in zip(starts, ends, strict=True)
                    if end - start >= min(minimum, JOIN_RUN_FRAMES))
        position += len(chunk)
    if open_start is not None:
        runs.append((open_start, position))
    return _joined(runs, round(JOIN_S * audio.sample_rate), minimum)


def _joined(runs: list[Run], gap: int, minimum: int) -> list[Run]:
    result: list[Run] = []
    for start, end in runs:
        if result and start - result[-1][1] <= gap:
            result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return [(start, end) for start, end in result if end - start >= minimum]


def _camera_command(camera: Path, *, stream: int | None, pan: str | None, rate: int) -> list[str]:
    return ["ffmpeg", "-nostdin", "-v", "error", "-i", str(camera), *_map(stream),
            *(["-af", pan] if pan else []), "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]


def _energy(window: np.ndarray) -> float:
    return float(np.mean(np.square(window, dtype=np.float64)))


def _db(energy: float) -> float:
    return 10.0 * np.log10(energy) if energy > 0 else -200.0


def level_gain_db(voice: Path, camera: Path, runs_s: Spans, *,
                  stream: int | None = None, pan: str | None = None) -> float | None:
    """Gain that brings the camera's sound to the voice's level, or ``None``.

    Compares the energy of 1 s windows outside ``runs_s`` (the stretches to
    fill) where both are audible, over the louder half of the voice's windows:
    speech, not pauses where the camera's room noise would pull the gain down.
    ``None`` when too few windows qualify.
    """
    window = round(LEVEL_WINDOW_S * LEVEL_RATE)
    voices = _frames(["ffmpeg", "-nostdin", "-v", "error", "-i", str(voice), "-map", "0:a:0",
                      "-ac", "1", "-ar", str(LEVEL_RATE), "-f", "f32le", "-"], 1, window)
    cameras = _frames(_camera_command(camera, stream=stream, pan=pan, rate=LEVEL_RATE), 1,
                      window)
    pairs: list[tuple[float, float]] = []
    index = 0
    for heard, own in zip(voices, cameras, strict=False):
        start, end = index * LEVEL_WINDOW_S, (index + 1) * LEVEL_WINDOW_S
        index += 1
        if len(heard) < window or len(own) < window or any(
                a < end and b > start for a, b in runs_s):
            continue
        voice_energy, camera_energy = _energy(heard), _energy(own)
        if _db(voice_energy) > VOICE_AUDIBLE_DB and _db(camera_energy) > CAMERA_AUDIBLE_DB:
            pairs.append((voice_energy, camera_energy))
    # Close both decoders (a generator left open keeps its ffmpeg running).
    voices.close()
    cameras.close()
    if len(pairs) < LEVEL_MIN_WINDOWS:
        return None
    middle = float(np.median([voice_energy for voice_energy, _ in pairs]))
    speech = [pair for pair in pairs if pair[0] >= middle]
    gain = _db(sum(a for a, _ in speech)) - _db(sum(b for _, b in speech))
    return float(np.clip(gain, *GAIN_LIMITS_DB))


def _envelope(runs: Sequence[Run], start: int, count: int, fade: int) -> np.ndarray:
    """Weights of the camera sound for frames ``[start, start + count)``."""
    envelope = np.zeros(count, np.float32)
    frames = np.arange(start, start + count)
    for first, last in runs:
        if last <= start or first >= start + count:
            continue
        ramp = max(1, min(fade, (last - first) // 2))
        inside = (frames >= first) & (frames < last)
        weights = np.minimum(1.0, np.minimum(frames - first + 1, last - frames) / ramp)
        envelope[inside] = weights[inside]
    return envelope


def fill_voice(voice: Path, camera: Path, output: Path, runs: Sequence[Run], *, gain_db: float,
               stream: int | None = None, pan: str | None = None,
               audio: AudioFormat | None = None) -> Path:
    """``voice`` with the camera's sound (``gain_db`` louder) in its zero ``runs``.

    The voice is in the camera's file time, so frame ``n`` of both is the same
    instant; outside ``runs`` the voice is copied unchanged.
    """
    audio = audio or audio_format(voice)
    output.parent.mkdir(parents=True, exist_ok=True)
    gain = np.float32(10.0 ** (gain_db / 20.0))
    fade = max(1, round(FILL_FADE_S * audio.sample_rate))
    voices = _frames(["ffmpeg", "-nostdin", "-v", "error", "-i", str(voice), "-map", "0:a:0",
                      "-f", "f32le", "-"], audio.channels)
    cameras = _frames(_camera_command(camera, stream=stream, pan=pan, rate=audio.sample_rate), 1)
    encoder = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(audio.sample_rate),
         "-ac", str(audio.channels), "-i", "-", "-c:a", audio.codec, "-rf64", "auto",
         str(output)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert encoder.stdin is not None
    pending = np.zeros(0, np.float32)
    position = 0
    try:
        for chunk in voices:
            count = len(chunk)
            while len(pending) < count:
                more = next(cameras, None)
                if more is None:
                    break
                pending = np.concatenate([pending, more[:, 0]])
            own = np.zeros(count, np.float32)
            own[:min(count, len(pending))] = pending[:count]
            pending = pending[count:]
            weights = _envelope(runs, position, count, fade)
            mixed = chunk + (own * weights * gain)[:, None]
            encoder.stdin.write(np.clip(mixed, -1.0, 1.0).astype("<f4").tobytes())
            position += count
    finally:
        voices.close()
        cameras.close()
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
        encoder.wait()
    if encoder.returncode:
        raise RuntimeError(f"writing {output.name} failed: {error.strip()}")
    return output


def runs_seconds(runs: Sequence[Run], audio: AudioFormat) -> Spans:
    return [(start / audio.sample_rate, end / audio.sample_rate) for start, end in runs]


def merge(spans: Sequence[tuple[float, float]]) -> Spans:
    result: Spans = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def subtract(spans: Sequence[tuple[float, float]], cuts: Sequence[tuple[float, float]]) -> Spans:
    result: Spans = []
    holes = merge(cuts)
    for start, end in merge(spans):
        cursor = start
        for left, right in holes:
            if right <= cursor or left >= end:
                continue
            if left > cursor:
                result.append((cursor, left))
            cursor = max(cursor, right)
        if end > cursor:
            result.append((cursor, end))
    return result


def playback_voice(project: Project, asset_id: str) -> Path | None:
    """A camera's voice for playback: filled where the recorder heard nothing."""
    filled = next(iter(project.outputs.get(f"{PLAYBACK_PREFIX}{asset_id}", [])), None)
    if filled is not None and filled.is_file():
        return filled
    return next(iter(project.outputs.get(f"sync:{asset_id}", [])), None)


def recorder_audio(project: Project, asset_id: str) -> tuple[Path, int | None] | None:
    """The recorder file its placement was measured on (repaired when sync repaired it)
    and the stream to read."""
    asset = next((item for item in project.assets if item.id == asset_id), None)
    if asset is None:
        return None
    repaired = next((path for path in project.outputs.get(f"recorder:{asset_id}", [])
                     if path.is_file()), None)
    if repaired is not None:
        return repaired, None
    source = project.source_dir / asset.path
    stream = asset.manual.get("media_info", {}).get("audio_stream_index")
    return (source, int(stream) if stream is not None else None) if source.is_file() else None


@dataclass(frozen=True, slots=True)
class RecorderSound:
    """Where the placed recorders have live sound, on the timeline."""

    alive: Spans
    #: Every recorder's placement, dead stretches included.
    extent: Spans
    #: Every placed camera.
    cameras: Spans

    def unheard(self, min_s: float = DEAD_S) -> Spans:
        """Camera time no recorder heard: a dead input, or before/after every recorder."""
        return [span for span in subtract(self.cameras, self.alive)
                if span[1] - span[0] >= min_s]

    def dead_air(self, min_s: float = DEAD_S) -> Spans:
        """Recorder time where nothing at all was recorded: no live recorder, no camera."""
        return [span for span in subtract(self.extent, [*self.alive, *self.cameras])
                if span[1] - span[0] >= min_s]


def recorder_sound(project: Project, *, min_s: float = DEAD_S) -> RecorderSound | None:
    """Live recorder sound on the timeline; ``None`` without placed recorders or
    when one can't be read."""
    placements = {item.asset_id: item for item in project.placements}
    recorders = [asset for asset in project.assets
                 if asset.role is AssetRole.RECORDER and asset.id in placements]
    if not recorders:
        return None
    alive: Spans = []
    extent: Spans = []
    for asset in recorders:
        placement = placements[asset.id]
        found = recorder_audio(project, asset.id)
        if found is None:
            return None
        path, stream = found
        info = asset.manual.get("media_info", {})
        try:
            audio = (audio_format(path) if stream is None else AudioFormat(
                int(info.get("audio_sample_rate") or 48000),
                int(info.get("audio_channels") or 1), "pcm_s24le"))
            runs = zero_runs(path, min_s=min_s, stream=stream, audio=audio)
        except (OSError, RuntimeError, ValueError):
            return None
        span = (placement.offset_s, placement.offset_s + placement.duration_s * placement.k)
        dead = [(file_to_timeline(start, placement), file_to_timeline(end, placement))
                for start, end in runs_seconds(runs, audio)]
        extent.append(span)
        alive.extend(subtract([span], dead))
    cameras = [(item.offset_s, item.offset_s + item.duration_s * item.k)
               for asset in project.assets if asset.role is AssetRole.CAMERA
               for item in [placements.get(asset.id)] if item is not None]
    return RecorderSound(merge(alive), merge(extent), merge(cameras))


def unheard_spans(project: Project, *, min_s: float = DEAD_S) -> Spans:
    """Timeline stretches a camera covers where no recorder has live sound."""
    sound = recorder_sound(project, min_s=min_s)
    return sound.unheard(min_s) if sound is not None else []
