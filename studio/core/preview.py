"""Short audio previews of rough-cut decisions without a rendered master.

Timeline time is resolved to the best available file: the timeline voice
master when one was rendered, otherwise a synchronized camera voice or the
timeline transcript's own source through its placement. A preview either
plays the original span or the span as it will sound with the cut applied.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from studio.core.media import build_atempo_chain
from studio.core.project import AssetRole, Project
from studio.core.timeline import SourcePlacement, timeline_to_file
from studio.core.transcript import Transcript
from studio.core.voice_fill import playback_voice


@dataclass(frozen=True, slots=True)
class AudioSource:
    path: Path
    placement: SourcePlacement | None
    kind: str

    def covers(self, start: float, end: float) -> bool:
        if self.placement is None:
            return True
        placement = self.placement
        return (placement.offset_s - 1e-6 <= start
                and end <= placement.offset_s + placement.duration_s * placement.k + 1e-6)

    def file_span(self, start: float, end: float) -> tuple[float, float]:
        if self.placement is None:
            return start, end
        return timeline_to_file(start, self.placement), timeline_to_file(end, self.placement)


def audio_sources(project: Project) -> list[AudioSource]:
    """Timeline-addressable audio, best first."""
    sources: list[AudioSource] = []
    master = next(iter(project.outputs.get("master_wav", [])), None)
    if master is not None and master.is_file():
        sources.append(AudioSource(master, None, "master"))
    placements = {item.asset_id: item for item in project.placements}
    for asset in project.assets:
        voice = playback_voice(project, asset.id)
        if (asset.role is AssetRole.CAMERA and voice is not None and voice.is_file()
                and asset.id in placements):
            sources.append(AudioSource(voice, placements[asset.id], "synced voice"))
    timeline = project.transcripts.get("timeline")
    if timeline is not None and timeline.is_file():
        transcript = Transcript.load(timeline)
        source_id = transcript.metadata.get("source_asset_id")
        if transcript.source_audio.is_file():
            sources.append(AudioSource(
                transcript.source_audio, placements.get(str(source_id)), "transcript source"
            ))
    return sources


def choose_source(project: Project, start: float, end: float) -> AudioSource:
    for source in audio_sources(project):
        if source.covers(start, end):
            return source
    raise ValueError("no audio covers this part of the timeline; run sync or timeline")


def preview_ranges(
    cut_start: float, cut_end: float, *, context_s: float, applied: bool, duration: float
) -> list[tuple[float, float]]:
    """Timeline spans to play around a cut, with or without the removed part."""
    before = (max(0.0, cut_start - context_s), cut_start)
    after = (cut_end, min(duration, cut_end + context_s))
    if not applied:
        return [(before[0], after[1])]
    return [span for span in (before, after) if span[1] - span[0] > 1e-3]


def render_preview(
    project: Project, ranges: list[tuple[float, float]], output: Path,
    *, crossfade_ms: float = 10.0,
) -> tuple[Path, str]:
    """Write a stereo WAV of ``ranges`` joined in order; return it and the source kind."""
    if not ranges:
        raise ValueError("nothing to preview")
    source = choose_source(project, ranges[0][0], ranges[-1][1])
    filters: list[str] = []
    labels: list[str] = []
    for index, (start, end) in enumerate(ranges):
        file_start, file_end = source.file_span(start, end)
        tempo = (file_end - file_start) / (end - start)
        filters.append(
            f"[0:a]atrim=start={max(0.0, file_start):.6f}:end={file_end:.6f},"
            "asetpts=PTS-STARTPTS,"
            + ",".join([*build_atempo_chain(tempo), "aresample=48000",
                        "aformat=channel_layouts=stereo"])
            + f"[p{index}]"
        )
        labels.append(f"p{index}")
    current = labels[0]
    for index, label in enumerate(labels[1:], start=1):
        fade = crossfade_ms / 1000
        filters.append(f"[{current}][{label}]acrossfade=d={fade:.4f}:c1=qsin:c2=qsin[x{index}]")
        current = f"x{index}"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source.path),
         "-filter_complex", ";".join(filters), "-map", f"[{current}]", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"preview render failed: {result.stderr.strip()[-1000:]}")
    return output, source.kind
