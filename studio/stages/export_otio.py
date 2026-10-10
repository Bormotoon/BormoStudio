"""OpenTimelineIO JSON (for DaVinci Resolve) from the neutral export sequence.

Written directly against the OTIO JSON schema so no extra dependency ships.
Clip.1/Marker.1 keep older readers working; current OTIO upgrades them on load.
Times are whole sequence frames, as in the XML exports, so cuts land on the
same frames in every NLE.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from studio.stages.export import Sequence, SequenceClip


def write_otio(sequence: Sequence, output: Path) -> Path:
    rate = float(sequence.fps)
    tracks: list[dict[str, Any]] = []
    video_number = audio_number = 0
    for lane in sorted({clip.lane for clip in sequence.clips}):
        clips = sorted((clip for clip in sequence.clips if clip.lane == lane),
                       key=lambda clip: clip.timeline_start_s)
        if clips[0].has_video:
            video_number += 1
            tracks.append(_track(f"V{video_number}", "Video", clips, sequence.fps))
            if any(clip.has_audio for clip in clips):
                # Camera sound stays available (disabled where synced voice replaces it).
                audio_number += 1
                tracks.append(_track(f"A{audio_number} camera", "Audio",
                                     [clip for clip in clips if clip.has_audio], sequence.fps))
        else:
            audio_number += 1
            tracks.append(_track(f"A{audio_number}", "Audio", clips, sequence.fps))
    timeline = {
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {"ultimate-video-forge": {"width": sequence.width,
                                              "height": sequence.height}},
        "name": sequence.name,
        "global_start_time": _time(0, rate),
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "metadata": {},
            "name": "tracks",
            "source_range": None,
            "effects": [],
            "markers": [
                {
                    "OTIO_SCHEMA": "Marker.1",
                    "metadata": {"kind": marker.kind},
                    "name": marker.text,
                    "color": "RED",
                    "marked_range": _range(_frames(marker.at_s, sequence.fps), 0, rate),
                }
                for marker in sequence.markers
            ],
            "enabled": True,
            "children": tracks,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return output


def _track(name: str, kind: str, clips: list[SequenceClip], fps: Fraction) -> dict[str, Any]:
    rate = float(fps)
    children: list[dict[str, Any]] = []
    cursor = 0
    for index, clip in enumerate(clips, 1):
        start = _frames(clip.timeline_start_s, fps)
        end = _frames(clip.timeline_start_s + clip.duration_s, fps)
        if start > cursor:
            children.append({"OTIO_SCHEMA": "Gap.1", "metadata": {}, "name": "",
                             "source_range": _range(0, start - cursor, rate),
                             "effects": [], "markers": [], "enabled": True})
        elif start < cursor:
            raise ValueError(f"{name}: clips overlap at {clip.timeline_start_s:.3f}s")
        effects = []
        if abs(clip.rate - 1.0) > 1e-9:
            effects.append({"OTIO_SCHEMA": "LinearTimeWarp.1", "metadata": {}, "name": "",
                            "effect_name": "LinearTimeWarp", "time_scalar": 1.0 / clip.rate})
        children.append({
            "OTIO_SCHEMA": "Clip.1",
            "metadata": {"asset_id": clip.asset_id, "role": clip.audio_role},
            "name": f"{clip.path.stem} {index}",
            "source_range": _range(_frames(clip.source_in_s, fps), end - start, rate),
            "effects": effects,
            "markers": [],
            "enabled": clip.audio_enabled if kind == "Audio" and clip.has_video else True,
            "media_reference": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "metadata": {},
                "name": clip.path.name,
                "available_range": None,
                "target_url": clip.path.resolve().as_uri(),
            },
        })
        cursor = end
    return {"OTIO_SCHEMA": "Track.1", "metadata": {}, "name": name, "source_range": None,
            "effects": [], "markers": [], "enabled": True, "kind": kind,
            "children": children}


def _frames(seconds: float, fps: Fraction) -> int:
    return round(seconds * float(fps))


def _time(value: float, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": float(value)}


def _range(start: float, duration: float, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "TimeRange.1", "start_time": _time(start, rate),
            "duration": _time(duration, rate)}
