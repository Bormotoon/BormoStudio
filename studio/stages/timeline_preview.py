"""Watch the whole edit before rendering the program master.

The preview is a low-resolution proxy of exactly what the program would
render — kept rough-cut ranges, speaker camera switches and synchronized
voices — cached by the edit and its inputs. Its map converts preview time to
timeline time and back, so the editor's cut overview can follow playback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studio.core.av_calibration import calibration_view, picture_shifts
from studio.core.project import Project, stable_fingerprint
from studio.core.transcript import Transcript

PREVIEW_HEIGHT = 360


@dataclass(frozen=True, slots=True)
class TimelinePreview:
    video: Path
    spans: tuple[dict[str, float], ...]  # timeline_start/end ↔ edited_start/end
    reused: bool

    def to_timeline(self, edited_s: float) -> float:
        """Timeline time shown at ``edited_s`` of the preview."""
        for span in self.spans:
            if span["edited_start"] <= edited_s <= span["edited_end"]:
                return span["timeline_start"] + edited_s - span["edited_start"]
        return self.spans[-1]["timeline_end"] if self.spans else 0.0

    def to_edited(self, timeline_s: float) -> float:
        """Preview time of ``timeline_s``; a removed moment maps to the next kept one."""
        for span in self.spans:
            if timeline_s < span["timeline_start"]:
                return span["edited_start"]
            if timeline_s <= span["timeline_end"]:
                return span["edited_start"] + timeline_s - span["timeline_start"]
        return self.spans[-1]["edited_end"] if self.spans else 0.0


def _identity(path: Path) -> list[object]:
    # Size and mtime, not a digest: hashing hours of camera files per preview is too slow.
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def preview_key(project: Project, settings: dict[str, Any], edit: Any, height: int) -> str:
    inputs = [project.source_dir / asset.path for asset in project.assets]
    inputs += [path for key, paths in project.outputs.items() if key.startswith("sync:")
               for path in paths]
    return stable_fingerprint(
        "timeline-preview-v1", height, [(k.start_s, k.end_s, k.camera_id) for k in edit.keep],
        [str(item) for item in project.placements],
        [_identity(path) for path in inputs if path.is_file()],
        settings.get("program", {}).get("fps", "auto"),
        *filter(None, [calibration_view(settings)]),
    )


def render_timeline_preview(
    project: Project, settings: dict[str, Any], *, height: int = PREVIEW_HEIGHT
) -> TimelinePreview:
    from studio.stages.program import program_edit, render_project_program, rendered_timeline_map

    timeline_path = project.transcripts.get("timeline")
    if timeline_path is None or not timeline_path.is_file():
        raise ValueError("no timeline transcript yet; run the plan up to timeline")
    duration = Transcript.load(timeline_path).duration
    edit = program_edit(project, settings, duration)
    folder = project.work_dir / "cache" / "preview"
    key = preview_key(project, settings, edit, height)
    video, index = folder / f"timeline-{key[:16]}.mp4", folder / f"timeline-{key[:16]}.json"
    if video.is_file() and index.is_file():
        spans = json.loads(index.read_text(encoding="utf-8"))["spans"]
        return TimelinePreview(video, tuple(spans), True)
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("timeline-*"):
        stale.unlink()
    conf = settings.get("program", {})
    conf = conf if isinstance(conf, dict) else {}
    report = render_project_program(
        project, edit, video, encoder="cpu", fps=str(conf.get("fps", "auto")),
        crossfade_ms=float(conf.get("crossfade_ms", 10.0)), max_height=height,
        picture_shift=picture_shifts(project, settings),
    )
    spans = rendered_timeline_map(edit, report)
    index.write_text(json.dumps({"schema_version": 1, "key": key, "spans": spans}, indent=2)
                     + "\n", encoding="utf-8")
    return TimelinePreview(video, tuple(spans), False)
