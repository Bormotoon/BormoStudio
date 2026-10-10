"""Coverage-aware speaker camera decisions in source timeline time."""

from dataclasses import dataclass
from typing import Any

from studio.core.project import AssetRole, Project
from studio.core.timeline import EditMap, KeepRange

#: Reserved speaker-camera key naming the wide (establishing) shot.
WIDE_SHOT = "@wide"


@dataclass(frozen=True, slots=True)
class CameraPolicy:
    min_shot_s: float = 1.0
    # Switch this much before a turn starts (the listener sees who will speak).
    cut_lead_s: float = 0.0
    # With a wide shot: this many speaker changes inside the window → wide.
    max_switches: int = 3
    switch_window_s: float = 6.0
    # With a wide shot: a speaker shot longer than this gets a wide insert (0 = off).
    max_shot_s: float = 0.0
    wide_insert_s: float = 3.0

    @classmethod
    def from_settings(cls, conf: dict[str, Any]) -> "CameraPolicy":
        return cls(
            min_shot_s=float(conf.get("min_shot_s", 1.0)),
            cut_lead_s=float(conf.get("cut_lead_s", 0.0)),
            max_switches=int(conf.get("max_switches", 3)),
            switch_window_s=float(conf.get("switch_window_s", 6.0)),
            max_shot_s=float(conf.get("max_shot_s", 0.0)),
            wide_insert_s=float(conf.get("wide_insert_s", 3.0)),
        )


def speaker_camera_edit(
    project: Project, edit: EditMap, turns: list[dict], mapping: dict[str, str],
    *, min_shot_s: float | None = None, policy: CameraPolicy | None = None,
) -> EditMap:
    policy = policy or CameraPolicy()
    if min_shot_s is not None:
        policy = CameraPolicy(**{**_fields(policy), "min_shot_s": min_shot_s})
    cameras = {asset.id: asset for asset in project.assets if asset.role is AssetRole.CAMERA}
    for target in mapping.values():
        if not any(asset.id == target or asset.group_id == target for asset in cameras.values()):
            raise ValueError(f"unknown speaker camera/group: {target}")
    wide = mapping.get(WIDE_SHOT)
    speakers = {key: value for key, value in mapping.items() if key != WIDE_SHOT}
    # Shifting whole turns (not only starts) keeps real overlaps and adds none.
    turns = [
        {**turn, "start": max(0.0, float(turn["start"]) - policy.cut_lead_s),
         "end": max(0.0, float(turn["end"]) - policy.cut_lead_s)}
        if policy.cut_lead_s > 0 else turn
        for turn in turns
    ]
    changes = _speaker_changes(turns)
    # The busy-exchange window opens and closes at these times.
    half = policy.switch_window_s / 2
    window_edges = {value for change in changes for value in (change - half, change + half)}
    placements = [item for item in project.placements if item.asset_id in cameras]

    def matches(asset_id: str, target: str | None) -> bool:
        return target is not None and (asset_id == target or cameras[asset_id].group_id == target)

    result: list[KeepRange] = []
    current = None
    shot_start = 0.0
    for keep in edit.keep:
        boundaries = sorted({keep.start_s, keep.end_s} | {
            value for turn in turns for value in (float(turn["start"]), float(turn["end"]))
            if keep.start_s < value < keep.end_s
        } | {
            value for item in placements
            for value in (item.offset_s, item.offset_s + item.duration_s * item.k)
            if keep.start_s < value < keep.end_s
        } | ({value for value in window_edges if keep.start_s < value < keep.end_s}
             if wide else set()))
        for start, end in zip(boundaries, boundaries[1:], strict=False):
            available = [item.asset_id for item in placements
                         if item.offset_s <= start + 1e-9
                         and item.offset_s + item.duration_s * item.k >= end - 1e-9]
            if not available:
                raise ValueError(f"no camera covers speaker selection {start:.3f}-{end:.3f}")
            active = {str(turn["speaker"]) for turn in turns
                      if float(turn["start"]) <= start and float(turn["end"]) > start}
            speaker = next(iter(active)) if len(active) == 1 else None
            chosen = (speakers.get(speaker)
                      if speaker is not None and speaker not in {"unknown", "overlap"} else None)
            wide_ids = [asset_id for asset_id in available if matches(asset_id, wide)]
            if wide_ids and (len(active) > 1 or "overlap" in active or _busy(
                changes, start, policy.switch_window_s, policy.max_switches,
            )):
                chosen = wide
            forced = keep.camera_id
            desired = forced or chosen
            preferred = [asset_id for asset_id in available if matches(asset_id, desired)]
            if forced and not preferred:
                raise ValueError(f"requested camera/group {forced} does not cover {start:.3f}")
            position = start
            while position < end - 1e-9:
                selected = current if current in available else available[0]
                if preferred and (forced or current not in available
                                  or position - shot_start >= policy.min_shot_s):
                    selected = current if current in preferred else preferred[0]
                if selected != current:
                    current, shot_start = selected, position
                piece_end = end
                if (not forced and wide_ids and policy.max_shot_s > 0
                        and current not in wide_ids):
                    limit = shot_start + policy.max_shot_s
                    if limit <= position + 1e-9:
                        current, shot_start = wide_ids[0], position
                        piece_end = min(end, position + policy.wide_insert_s)
                    elif limit < end:
                        piece_end = limit
                _append(result, position, piece_end, current)
                position = piece_end
    return EditMap(edit.id, tuple(result))


def _append(result: list[KeepRange], start: float, end: float, camera: str | None) -> None:
    if result and result[-1].end_s == start and result[-1].camera_id == camera:
        result[-1] = KeepRange(result[-1].start_s, end, camera)
    else:
        result.append(KeepRange(start, end, camera))


def _speaker_changes(turns: list[dict]) -> list[float]:
    known = sorted(
        (float(turn["start"]), str(turn["speaker"])) for turn in turns
        if str(turn["speaker"]) not in {"unknown", "overlap"}
    )
    return [start for (start, speaker), (_, previous) in zip(known[1:], known, strict=False)
            if speaker != previous]


def _busy(changes: list[float], at: float, window: float, limit: int) -> bool:
    if limit <= 0 or window <= 0:
        return False
    half = window / 2
    return sum(at - half <= value < at + half for value in changes) >= limit


def _fields(policy: CameraPolicy) -> dict[str, Any]:
    return {name: getattr(policy, name) for name in CameraPolicy.__dataclass_fields__}
