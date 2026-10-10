"""Suggest speaker ↔ camera bindings from face and active-speaker evidence.

Evidence comes from the vision worker (``studio.vision.camera_evidence``) on a
few windows of each speaker's longest turns. A close-up camera of speaker S
shows a face speaking during S's turns and a silent face otherwise; a camera
that keeps two or more people in frame is the wide shot. Suggestions never
replace bindings the user made.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from studio.core.project import AssetRole, Project
from studio.core.timeline import timeline_to_file
from studio.stages.camera_selection import WIDE_SHOT

#: Speaking-logit lead a camera needs during S's turns over other turns.
MIN_CONTRAST = 0.5
#: Share of windows with two or more people for a wide-shot camera.
WIDE_SHARE = 0.6


@dataclass(frozen=True, slots=True)
class Window:
    speaker: str
    start_s: float  # camera file time
    end_s: float


def evidence_windows(
    project: Project, turns: list[dict], asset_id: str,
    *, per_speaker: int = 3, length_s: float = 4.0,
) -> list[Window]:
    """The longest covered turns of each speaker, trimmed to ``length_s``."""
    placement = next((item for item in project.placements if item.asset_id == asset_id), None)
    if placement is None:
        return []
    covered_end = placement.offset_s + placement.duration_s * placement.k
    windows: list[Window] = []
    speakers = sorted({str(turn["speaker"]) for turn in turns} - {"unknown", "overlap"})
    for speaker in speakers:
        mine = sorted(
            (turn for turn in turns if str(turn["speaker"]) == speaker),
            key=lambda turn: float(turn["start"]) - float(turn["end"]),
        )
        for turn in mine:
            start = max(float(turn["start"]), placement.offset_s)
            end = min(float(turn["end"]), covered_end)
            if end - start < min(length_s, 1.5):
                continue
            middle = (start + end) / 2
            half = min(length_s, end - start) / 2
            windows.append(Window(
                speaker,
                timeline_to_file(middle - half, placement),
                timeline_to_file(middle + half, placement),
            ))
            if sum(window.speaker == speaker for window in windows) >= per_speaker:
                break
    return windows


def suggest_bindings(
    evidence: dict[str, list[dict[str, Any]]], speakers: list[str],
) -> dict[str, Any]:
    """Decide from per-camera window results (each tagged with its speaker)."""
    wide_share = {
        camera: sum(item["faces"] >= 2 for item in items) / len(items)
        for camera, items in evidence.items() if items
    }
    wide = max(wide_share, key=lambda camera: wide_share[camera], default=None)
    if wide is not None and wide_share[wide] < WIDE_SHARE:
        wide = None
    contrasts: dict[str, dict[str, float]] = {}
    for camera, items in evidence.items():
        if camera == wide:
            continue
        for speaker in speakers:
            own = [item["speaking"] for item in items
                   if item["speaker"] == speaker and item.get("speaking") is not None]
            other = [item["speaking"] for item in items
                     if item["speaker"] != speaker and item.get("speaking") is not None]
            if own and other:
                contrasts.setdefault(speaker, {})[camera] = mean(own) - mean(other)
    chosen: dict[str, str] = {}
    pairs = sorted(
        ((value, speaker, camera) for speaker, row in contrasts.items()
         for camera, value in row.items() if value >= MIN_CONTRAST),
        reverse=True,
    )
    used: set[str] = set()
    for _value, speaker, camera in pairs:
        if speaker not in chosen and camera not in used:
            chosen[speaker] = camera
            used.add(camera)
    if wide is not None:
        chosen[WIDE_SHOT] = wide
    unresolved = [speaker for speaker in speakers if speaker not in chosen]
    return {"speaker_cameras": chosen, "unresolved": unresolved,
            "contrast": contrasts, "wide_share": wide_share}


def collect_evidence(
    project: Project, turns: list[dict], *, device: str = "cuda",
) -> dict[str, list[dict[str, Any]]]:
    from studio.stages.reel_render import run_vision_worker

    evidence: dict[str, list[dict[str, Any]]] = {}
    for asset in project.assets:
        if asset.role is not AssetRole.CAMERA:
            continue
        windows = evidence_windows(project, turns, asset.id)
        if not windows:
            continue
        value = run_vision_worker({
            "mode": "camera_evidence",
            "video": str((project.source_dir / asset.path).resolve()),
            "windows": [[window.start_s, window.end_s] for window in windows],
            "device": device, "active_speaker": True,
        })
        if not value.get("active_speaker"):
            raise RuntimeError("camera binding needs the Light-ASD model; download light-asd")
        evidence[asset.id] = [
            {**item, "speaker": window.speaker}
            for window, item in zip(windows, value["windows"], strict=True)
        ]
    return evidence
