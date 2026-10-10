"""YouTube "most replayed" heatmap as a reel ranking signal."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from studio.core.project import Project
from studio.core.timeline import TimeDomain, timeline_to_file
from studio.core.transcript import Transcript
from studio.reels.analysis.contracts import MomentRecord, replace_record

Span = tuple[float, float]


def load_heatmap(info_path: Path | None) -> list[tuple[float, float, float]]:
    """(start, end, value) markers from a yt-dlp info.json, sorted; empty if absent."""
    if info_path is None or not info_path.is_file():
        return []
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    markers = []
    for item in data.get("heatmap") or [] if isinstance(data, dict) else []:
        try:
            start, end, value = (float(item[key]) for key in ("start_time", "end_time", "value"))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(number) for number in (start, end, value)) and end > start:
            markers.append((start, end, max(0.0, value)))
    return sorted(markers)


def span_intensity(
    heatmap: list[tuple[float, float, float]], spans: Iterable[Span],
) -> float | None:
    """Duration-weighted mean heatmap value over spans, scaled by the episode peak."""
    peak = max((value for _, _, value in heatmap), default=0.0)
    total = weighted = 0.0
    for start, end in spans:
        for left, right, value in heatmap:
            overlap = min(end, right) - max(start, left)
            if overlap > 0:
                total += overlap
                weighted += overlap * value
    if peak <= 0 or total <= 0:
        return None
    return round(weighted / total / peak, 4)


def media_spans(
    project: Project, transcript: Transcript, media: Path, start: float, end: float,
) -> list[Span]:
    """Map a moment to spans of the downloaded media file; empty when unmappable."""
    if transcript.time_domain is TimeDomain.FILE:
        return [(start, end)] if transcript.source_audio.name == media.name else []
    asset = next((item for item in project.assets
                  if (project.source_dir / item.path).resolve() == media.resolve()), None)
    placement = next((item for item in project.placements
                      if asset is not None and item.asset_id == asset.id), None)
    if placement is None:
        return []
    timeline: list[Span] = []
    if transcript.time_domain is TimeDomain.TIMELINE:
        timeline = [(start, end)]
    elif transcript.time_domain is TimeDomain.EDITED:
        timeline = _edited_to_timeline(project, start, end)
    return [(timeline_to_file(a, placement), timeline_to_file(b, placement))
            for a, b in timeline]


def _edited_to_timeline(project: Project, start: float, end: float) -> list[Span]:
    report_path = next(iter(project.outputs.get("program_report", [])), None)
    if report_path is None or not report_path.is_file():
        return []
    mapping: Any = json.loads(report_path.read_text(encoding="utf-8")).get("timeline_to_rendered")
    spans = []
    for piece in mapping if isinstance(mapping, list) else []:
        a, b, c, d = (float(piece[key]) for key in
                      ("timeline_start", "timeline_end", "edited_start", "edited_end"))
        left, right = max(start, c), min(end, d)
        if right > left:
            spans.append((a + left - c, a + right - c))
    return spans


def annotate_replay(
    records: list[MomentRecord], project: Project, transcript: Transcript, info_path: Path | None,
) -> tuple[list[MomentRecord], dict[str, Any]]:
    heatmap = load_heatmap(info_path)
    if not heatmap:
        return [replace_record(record, replay_score=None) for record in records], {
            "status": "unavailable"}
    media = next((path for path in project.outputs.get("fetch", [])
                  if not path.name.endswith(".json")), None)
    if media is None:
        return records, {"status": "unavailable"}
    annotated, scored = [], 0
    for record in records:
        value = span_intensity(heatmap, media_spans(
            project, transcript, media, record.start, record.end))
        scored += value is not None
        annotated.append(replace_record(record, replay_score=value))
    return annotated, {"status": "applied", "markers": len(heatmap), "scored": scored}
