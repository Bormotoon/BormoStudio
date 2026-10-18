"""Per-camera lip-sync calibration: a camera's picture against its own sound.

Synchronization aligns every sound to the cameras' sound. A fixed delay inside a
camera's own audio path (ADC, encoder) is invisible to that measurement and is
calibrated once by eye, as in WhisperSync: ``sync.camera_av_offset_ms`` for every
camera and ``sync.camera_av_offset_ms_by_camera`` per camera (asset id, camera
group, sub-folder or file name). Positive values delay the sound against the
picture (the sound arrived early), negative values advance it.

Placements stay what synchronization measured: they place sound. Consumers that
show a camera's picture (exports, program, preview, reels) move only the
picture, by the calibration, so sound tracks, the sync check and transcripts are
untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import PurePath
from typing import Any

from studio.core.project import AssetRole, Project
from studio.core.timeline import SourcePlacement


def calibration_view(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The calibration settings for fingerprints; empty while nothing is calibrated."""
    sync = settings.get("sync", {})
    sync = sync if isinstance(sync, Mapping) else {}
    default = float(sync.get("camera_av_offset_ms", 0.0) or 0.0)
    by_camera = {str(key): float(value) for key, value in
                 dict(sync.get("camera_av_offset_ms_by_camera") or {}).items()}
    if not default and not any(by_camera.values()):
        return {}
    return {"camera_av_offset_ms": default, "camera_av_offset_ms_by_camera": by_camera}


def picture_shifts(project: Project, settings: Mapping[str, Any]) -> dict[str, float]:
    """Seconds each camera's picture moves earlier on the timeline (asset id → s)."""
    view = calibration_view(settings)
    if not view:
        return {}
    by_camera: dict[str, float] = view["camera_av_offset_ms_by_camera"]
    result: dict[str, float] = {}
    for asset in project.assets:
        if asset.role is not AssetRole.CAMERA:
            continue
        path = PurePath(asset.path)
        names = (asset.id, asset.group_id or "", path.parent.name, path.name, path.stem)
        value = next((by_camera[name] for name in names if name and name in by_camera),
                     view["camera_av_offset_ms"])
        if value:
            result[asset.id] = value / 1000.0
    return result


def picture_placement(placement: SourcePlacement, shift_s: float) -> SourcePlacement:
    """Where the camera's picture lies when its sound lies at ``placement``.

    The picture moves ``shift_s`` earlier. A picture moved before the timeline
    origin loses its first frames instead of moving the origin, so edits,
    markers and every other placement keep their times.
    """
    if not shift_s:
        return placement
    offset = placement.offset_s - shift_s
    trim = max(0.0, -offset) / placement.k
    if trim >= placement.duration_s:
        raise ValueError(f"calibration of {shift_s * 1000:+.0f} ms leaves no picture "
                         f"of {placement.asset_id}")
    return replace(placement, offset_s=max(0.0, offset), in_s=placement.in_s + trim,
                   duration_s=placement.duration_s - trim)
