"""Coverage-aware speaker camera decisions in source timeline time."""

from studio.core.project import AssetRole, Project
from studio.core.timeline import EditMap, KeepRange


def speaker_camera_edit(
    project: Project, edit: EditMap, turns: list[dict], mapping: dict[str, str],
    *, min_shot_s: float = 1.0,
) -> EditMap:
    cameras = {asset.id: asset for asset in project.assets if asset.role is AssetRole.CAMERA}
    for target in mapping.values():
        if not any(asset.id == target or asset.group_id == target for asset in cameras.values()):
            raise ValueError(f"unknown speaker camera/group: {target}")
    placements = [item for item in project.placements if item.asset_id in cameras]
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
        })
        for start, end in zip(boundaries, boundaries[1:], strict=False):
            available = [item.asset_id for item in placements
                         if item.offset_s <= start + 1e-9
                         and item.offset_s + item.duration_s * item.k >= end - 1e-9]
            if not available:
                raise ValueError(f"no camera covers speaker selection {start:.3f}-{end:.3f}")
            active = {str(turn["speaker"]) for turn in turns
                      if float(turn["start"]) <= start and float(turn["end"]) > start}
            speaker = next(iter(active)) if len(active) == 1 else None
            chosen = (mapping.get(speaker)
                      if speaker is not None and speaker not in {"unknown", "overlap"} else None)
            forced = keep.camera_id
            desired = forced or chosen
            preferred = [asset_id for asset_id in available
                         if asset_id == desired or cameras[asset_id].group_id == desired]
            if forced and not preferred:
                raise ValueError(f"requested camera/group {forced} does not cover {start:.3f}")
            selected = current if current in available else available[0]
            if preferred and (forced or current not in available
                              or start - shot_start >= min_shot_s):
                selected = current if current in preferred else preferred[0]
            if selected != current:
                current, shot_start = selected, start
            if result and result[-1].end_s == start and result[-1].camera_id == selected:
                result[-1] = KeepRange(result[-1].start_s, end, selected)
            else:
                result.append(KeepRange(start, end, selected))
    return EditMap(edit.id, tuple(result))
