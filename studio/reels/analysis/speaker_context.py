"""Optional speaker-change evidence in the transcript's time domain."""

import json
import math
from pathlib import Path

from studio.core.project import Project, stable_fingerprint
from studio.core.timeline import TimeDomain
from studio.core.transcript import Transcript


def speaker_context_paths(project: Project) -> tuple[Path | None, Path | None]:
    paths = project.outputs.get("speakers", [])
    return (
        next((path for path in paths if path.name == "diarization.json"), None),
        next((path for path in paths if path.name == "report.json"), None),
    )


def program_report_path(project: Project) -> Path | None:
    return next(iter(project.outputs.get("program_report", [])), None)


def speaker_context_identity(project: Project) -> str:
    contents = []
    for path in (*speaker_context_paths(project), program_report_path(project)):
        if path is not None:
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, ValueError):
                content = "unavailable"
            contents.append((str(path), content))
    return stable_fingerprint(contents)


def _speaker_at_boundary(
    turns: list[tuple[float, float, str]], at: float, *, left: bool
) -> str | None:
    active = {
        speaker
        for start, end, speaker in turns
        if (start < at <= end if left else start <= at < end)
    }
    if len(active) != 1:
        return None
    speaker = next(iter(active))
    return None if speaker.lower() in {"unknown", "overlap"} else speaker


def speaker_change_starts(project: Project, transcript: Transcript) -> tuple[list[float], str]:
    path, report = speaker_context_paths(project)
    if path is None:
        return [], "unavailable"
    if transcript.time_domain not in {TimeDomain.TIMELINE, TimeDomain.EDITED}:
        return [], "incompatible_time_domain"
    if report is None:
        raise ValueError("speaker context requires a registered time-domain report")
    metadata = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("time_domain") != "timeline":
        return [], "incompatible_time_domain"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("speaker turns must be a list")
    turns = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("speaker turn must be an object")
        start, end = float(item["start"]), float(item["end"])
        speaker = item.get("speaker")
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or not isinstance(speaker, str)
            or not speaker.strip()
        ):
            raise ValueError("invalid speaker turn")
        turns.append((start, end, speaker.strip()))
    changes = []
    previous = None
    for start, _end, speaker in sorted(turns):
        if transcript.time_domain is TimeDomain.TIMELINE and start >= transcript.duration:
            continue
        if speaker.lower() in {"unknown", "overlap"}:
            previous = None
            continue
        if previous is not None and previous != speaker:
            changes.append(start)
        previous = speaker
    if transcript.time_domain is TimeDomain.EDITED:
        mapping_path = program_report_path(project)
        if mapping_path is None:
            return [], "missing_edit_mapping"
        mapping_report = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(mapping_report, dict):
            raise ValueError("program report must be an object")
        if mapping_report.get("edited_timing_fingerprint") != stable_fingerprint(
            transcript.duration, [(word.start, word.end) for word in transcript.words]
        ):
            return [], "stale_edit_mapping"
        mapping = mapping_report.get("timeline_to_rendered")
        if not isinstance(mapping, list):
            return [], "missing_edit_mapping"
        mapped: list[float] = []
        previous_end = 0.0
        previous_piece = None
        for piece in mapping:
            if not isinstance(piece, dict):
                raise ValueError("rendered timeline mapping piece must be an object")
            a, b, c, d = (
                float(piece[key])
                for key in ("timeline_start", "timeline_end", "edited_start", "edited_end")
            )
            if (
                not all(math.isfinite(value) for value in (a, b, c, d))
                or a < 0
                or b <= a
                or c < previous_end - 1e-6
                or d <= c
                or d > transcript.duration + 1e-6
                or abs((b - a) - (d - c)) > 1e-6
            ):
                raise ValueError("invalid rendered timeline mapping")
            previous_end = d
            mapped.extend(c + start - a for start in changes if a <= start < b)
            if previous_piece is not None:
                previous_timeline_end, previous_edited_end = previous_piece
                if abs(c - previous_edited_end) <= 1e-6 and abs(a - previous_timeline_end) > 1e-6:
                    before = _speaker_at_boundary(turns, previous_timeline_end, left=True)
                    after = _speaker_at_boundary(turns, a, left=False)
                    if before is not None and after is not None and before != after:
                        mapped.append(c)
            previous_piece = (b, d)
        return sorted(set(mapped)), "mapped_edited"
    return sorted(set(changes)), "ok"


def speaker_turns(project: Project, transcript: Transcript) -> list[tuple[float, float, str]]:
    """Named speaker turns in the transcript's time domain; [] when unavailable.

    Edited (program) time goes through the program's timeline mapping, so a
    turn cut by the rough cut is cut the same way.
    """
    path, report = speaker_context_paths(project)
    if path is None or report is None:
        return []
    metadata = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("time_domain") != "timeline":
        return []
    turns = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        speaker = str(item.get("speaker") or "").strip()
        start, end = float(item["start"]), float(item["end"])
        if speaker and speaker.lower() not in {"unknown", "overlap"} and end > start:
            turns.append((start, end, speaker))
    if transcript.time_domain is TimeDomain.TIMELINE:
        return sorted(turns)
    if transcript.time_domain is not TimeDomain.EDITED:
        return []
    mapping_path = program_report_path(project)
    if mapping_path is None:
        return []
    mapping_report = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping_report, dict) or mapping_report.get(
        "edited_timing_fingerprint"
    ) != stable_fingerprint(
        transcript.duration, [(word.start, word.end) for word in transcript.words]
    ):
        return []
    mapped = []
    for piece in mapping_report.get("timeline_to_rendered") or []:
        a, b, c = (float(piece[key]) for key in ("timeline_start", "timeline_end", "edited_start"))
        for start, end, speaker in turns:
            low, high = max(start, a), min(end, b)
            if high > low:
                mapped.append((c + low - a, c + high - a, speaker))
    return sorted(mapped)


def speaker_at(turns: list[tuple[float, float, str]], at: float) -> str:
    """The one speaker talking at ``at``; "" for silence or overlaps."""
    active = {speaker for start, end, speaker in turns if start <= at < end}
    return next(iter(active)) if len(active) == 1 else ""
