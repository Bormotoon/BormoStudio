"""Connect independently recorded files through accepted clock maps."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from studio.core.project import Asset, Project
from studio.core.timeline import SourcePlacement
from studio.core.transcript import Transcript
from studio.stages.sync_alignment import align_sources
from studio.stages.sync_models import AlignmentMap

# fit(target, source, source_range=None, acoustic_search=None): an accepted map or None.
Fit = Callable[..., AlignmentMap | None]


def plan_sessions(
    project: Project, cameras: list[Asset], recorders: list[Asset],
    transcripts: dict[str, Transcript], conf: dict[str, Any], *, acoustic_first: bool,
    check: Callable[[Asset, Asset, AlignmentMap], AlignmentMap] | None = None,
    search_ranges: dict[tuple[str, str], tuple[float, float]] | None = None,
    text_only: set[tuple[str, str]] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    skipped_search: str | None = None,
    second_pass: Callable[[Fit, dict[tuple[str, str], AlignmentMap], dict[str, str]], None]
    | None = None,
    matching: dict[str, Transcript] | None = None,
) -> tuple[list[SourcePlacement], dict[tuple[str, str], AlignmentMap], dict[str, str]]:
    """Fit every camera to the recorders and place all files on one timeline.

    ``check`` may replace an accepted camera–recorder map (the acoustic drift
    check) before placements are derived from it. ``second_pass`` gets the
    fitting function and the results of the first pass and may drop or add
    camera–recorder maps (neighbour search, a stronger transcription) before
    the cameras are placed. ``matching`` replaces a file's transcript for word
    matching only (a recorder transcribed in the cameras' Whisper mode).
    """
    fits: dict[tuple[str, str], AlignmentMap] = {}
    rejected: dict[str, str] = {}

    def fit(target: Asset, source: Asset, source_range: tuple[float, float] | None = None,
            acoustic_search: bool | None = None) -> AlignmentMap | None:
        extra = {"skipped_search": skipped_search} if skipped_search else {}
        try:
            result = align_sources(
                (matching or {}).get(target.id, transcripts[target.id]),
                (matching or {}).get(source.id, transcripts[source.id]),
                project.source_dir / target.path, project.source_dir / source.path,
                conf, acoustic_first=acoustic_first,
                source_range=source_range or (search_ranges or {}).get((target.id, source.id)),
                acoustic_search=((target.id, source.id) not in (text_only or set())
                                 if acoustic_search is None else acoustic_search),
                **extra,
            )
        except ValueError as exc:
            reason = str(exc)
            if transcripts[source.id].metadata.get("coverage"):
                reason += ("; the camera may be outside the transcribed window: "
                           "set transcribe.recorder_scope=full")
            rejected[f"{target.id}:{source.id}"] = reason
            return None
        if check is not None:
            result = check(target, source, result)
        # A fitted line must actually put some recorder material inside the camera.
        left = max(0.0, result.offset)
        right = min(transcripts[target.id].duration,
                    result.offset + result.k * transcripts[source.id].duration)
        if right <= left:
            rejected[f"{target.id}:{source.id}"] = "accepted map has no shared coverage"
            return None
        return result

    total = len(cameras) * len(recorders)
    for index, camera in enumerate(cameras):
        for offset, recorder in enumerate(recorders):
            if progress is not None:
                progress(index * len(recorders) + offset, total, Path(camera.path).name)
            result = fit(camera, recorder)
            if result is not None:
                fits[camera.id, recorder.id] = result
    if second_pass is not None:
        second_pass(fit, fits, rejected)
    # A clip no recorder matches (a one-second false start, a test shot) is left
    # out with its reasons; only a project where nothing matches fails.
    unplaced = [camera for camera in cameras
                if not any(target == camera.id for target, _ in fits)]
    if len(unplaced) == len(cameras):
        raise ValueError(f"no recorder overlaps any camera: {rejected}")
    for camera in unplaced:
        rejected[camera.id] = "left out: no recorder matches this camera"
    cameras = [camera for camera in cameras if camera not in unplaced]

    placed = {cameras[0].id: SourcePlacement(
        cameras[0].id, 0, 0, transcripts[cameras[0].id].duration, provenance="reference",
    )}
    assets = {asset.id: asset for asset in [*recorders, *cameras]}
    offsets = conf.get("session_offsets", {})
    for asset in assets.values():
        offset = offsets.get(asset.id, offsets.get(asset.group_id))
        if offset is not None:
            placed[asset.id] = SourcePlacement(
                asset.id, float(offset), 0, transcripts[asset.id].duration, provenance="manual",
            )

    def propagate() -> None:
        changed = True
        while changed:
            changed = False
            for (target, source), mapping in fits.items():
                if target in placed and source not in placed:
                    base = placed[target]
                    placed[source] = SourcePlacement(
                        source, base.offset_s + base.k * mapping.offset, 0,
                        transcripts[source].duration, base.k * mapping.k, mapping.provenance,
                        {"inliers": float(mapping.inliers), "residual_ms": mapping.residual_ms},
                    )
                    changed = True
                elif source in placed and target not in placed:
                    base = placed[source]
                    rate = base.k / mapping.k
                    placed[target] = SourcePlacement(
                        target, base.offset_s - rate * mapping.offset, 0,
                        transcripts[target].duration, rate, mapping.provenance,
                        {"inliers": float(mapping.inliers), "residual_ms": mapping.residual_ms},
                    )
                    changed = True

    propagate()
    # A recorder or camera spanning sessions can bridge otherwise disconnected pairs.
    for group in (recorders, cameras):
        for index, target in enumerate(group):
            for source in group[index + 1:]:
                if (target.id in placed) == (source.id in placed):
                    continue
                result = fit(target, source)
                if result is not None:
                    fits[target.id, source.id] = result
                    propagate()
    missing = set(assets) - set(placed)
    if missing:
        raise ValueError(
            f"disconnected recording sessions: {sorted(missing)}; "
            "set sync.session_offsets for a camera/recorder in each session "
            "(seconds on the reference camera timeline)"
        )
    origin = min(0.0, *(item.offset_s for item in placed.values()))
    return ([replace(placed[asset.id], offset_s=placed[asset.id].offset_s - origin)
             for asset in [*recorders, *cameras]], fits, rejected)
