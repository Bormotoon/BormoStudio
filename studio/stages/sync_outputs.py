"""Sample-accurate voice segmentation and nonduplicating timeline master."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from studio.core.timeline import SourcePlacement
from studio.stages.sync_render import cut_wav_segment, mix_clips_on_timeline


def segment_voice(source: Path, directory: Path, minutes: float) -> tuple[Path, ...]:
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("segment duration must be finite and positive")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams",
        "-of", "json", str(source),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    rate = int(stream["sample_rate"])
    from fractions import Fraction

    frames = round(int(stream["duration_ts"]) * Fraction(stream["time_base"]) * rate)
    codec = stream["codec_name"]
    if codec not in {"pcm_s16le", "pcm_s24le", "pcm_s32le"}:
        raise ValueError("segmentation requires integer PCM WAV")
    step = max(1, round(minutes * 60 * rate))
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    entries = []
    for index, start in enumerate(range(0, frames, step), 1):
        end = min(frames, start + step)
        output = directory / f"segment-{index:04d}.wav"
        cut_wav_segment(source, output, start / rate, end / rate, codec=codec)
        artifacts.append(output)
        entries.append({"path": output.name, "source_start_frame": start,
                        "frame_count": end - start, "rendered_start_s": start / rate,
                        "duration_s": (end - start) / rate})
    manifest = directory / "segments.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "time_domain": "rendered_audio", "sample_rate": rate,
        "source_name": source.name, "segments": entries,
    }, indent=2) + "\n", encoding="utf-8")
    return (*artifacts, manifest)


def render_voice_master(
    placements: list[SourcePlacement], voices: dict[str, Path], output: Path,
    *, crossfade_ms: float = 10.0,
) -> Path:
    if not math.isfinite(crossfade_ms) or crossfade_ms < 0:
        raise ValueError("master crossfade must be finite and non-negative")
    candidates = [item for item in placements if item.asset_id in voices]
    if not candidates:
        raise ValueError("no synchronized voice is available for the master")
    boundaries = sorted({value for item in candidates
                         for value in (item.offset_s, item.offset_s + item.duration_s * item.k)})
    import tempfile

    intervals: list[tuple[float, float, Any]] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        available = [item for item in candidates
                     if item.offset_s <= start
                     and item.offset_s + item.duration_s * item.k >= end - 1e-9]
        if not available:
            continue
        selected = available[0]
        if intervals and intervals[-1][2] == selected and intervals[-1][1] == start:
            intervals[-1] = (intervals[-1][0], end, selected)
        else:
            intervals.append((start, end, selected))
    fades = [0.0] * len(intervals)
    for index in range(1, len(intervals)):
        previous_start, previous_end, _previous = intervals[index - 1]
        start, end, selected = intervals[index]
        if abs(previous_end - start) < 1e-9:
            # Crossfade before the switch only where both source files exist.
            # Linear complementary gains avoid doubling correlated dialogue.
            fades[index] = max(0.0, min(
                crossfade_ms / 1000, start - selected.offset_s,
                (previous_end - previous_start) / 2, (end - start) / 2,
            ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".master-", dir=output.parent) as temporary:
        clips = []
        for index, (start, end, selected) in enumerate(intervals):
            start -= fades[index]
            cut = Path(temporary) / f"part-{index}.wav"
            source_start = (start - selected.offset_s) / selected.k
            source_end = (end - selected.offset_s) / selected.k
            if abs(selected.k - 1) < 1e-9:
                cut_wav_segment(voices[selected.asset_id], cut, source_start, source_end)
            else:
                factor = 1 / selected.k
                tempos = []
                while factor < 0.5:
                    tempos.append("atempo=0.5")
                    factor /= 0.5
                while factor > 2:
                    tempos.append("atempo=2")
                    factor /= 2
                tempos.append(f"atempo={factor}")
                result = subprocess.run([
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-i",
                    str(voices[selected.asset_id]), "-af",
                    f"atrim=start={source_start}:end={source_end},asetpts=PTS-STARTPTS,"
                    + ",".join(tempos) + f",apad,atrim=duration={end - start}",
                    "-c:a", "pcm_s24le", str(cut),
                ], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f"master retiming failed: {result.stderr.strip()}")
            fade_out = fades[index + 1] if index + 1 < len(fades) else 0
            if fades[index] or fade_out:
                filters = []
                if fades[index]:
                    filters.append(f"afade=t=in:st=0:d={fades[index]}:curve=tri")
                if fade_out:
                    filters.append(f"afade=t=out:st={end - start - fade_out}:"
                                   f"d={fade_out}:curve=tri")
                faded = cut.with_name(f"faded-{index}.wav")
                subprocess.run([
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(cut),
                    "-af", ",".join(filters), "-c:a", "pcm_s24le", str(faded),
                ], capture_output=True, text=True, check=True)
                cut = faded
            clips.append((cut, start))
        return mix_clips_on_timeline(clips, boundaries[-1], 48000, output, channels=2)
