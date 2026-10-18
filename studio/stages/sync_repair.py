"""Conservative repair candidates: publish only after content and lag checks."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from studio.core.timeline import AudioWarpMap
from studio.core.transcript import Transcript
from studio.stages.sync_alignment import align_sources
from studio.stages.sync_check_settings import SelfCheckSettings
from studio.stages.sync_check_step import combine_checks
from studio.stages.sync_engine import render_aligned_clip
from studio.stages.sync_geometry import PieceConfig
from studio.stages.sync_refine import RefineSettings
from studio.stages.sync_render import PauseDuck
from studio.stages.sync_self_check import check_rendered_clip
from studio.stages.sync_verify import measure, short_clip_grid
from studio.stages.transcribe import TranscribeSettings, WhisperEngine


def repair_voices(
    references: dict[str, Transcript], camera_paths: dict[str, Path],
    sources: dict[str, tuple[str, Path, Transcript, int]],
    directory: Path, check_report: Path, transcribe_settings: object,
    sync_settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Path], dict[str, AudioWarpMap], tuple[Path, ...]]:
    initial = json.loads(check_report.read_text(encoding="utf-8"))
    failed = {item["asset_id"] for item in initial["clips"] if item["status"] == "failed"}
    initial_spans = {item["asset_id"]: len(item.get("spans", [])) for item in initial["clips"]}
    conf = sync_settings or {}
    threshold = float(conf.get("max_drift_ms", 20.0))
    refine = (None if conf.get("acoustic_refine", "auto") == "off"
              else RefineSettings.from_sync(conf, PieceConfig().seam_snap_max_s))
    replacements = {}
    maps = {}
    artifacts = []
    reports = []
    directory.mkdir(parents=True, exist_ok=True)
    engine = WhisperEngine(TranscribeSettings.from_mapping(transcribe_settings))
    try:
        for asset_id in sorted(failed):
            reference = references[asset_id]
            source_id, source_path, source_transcript, channels = sources[asset_id]
            entry = {"asset_id": asset_id, "accepted": False}
            candidate = directory / f"{asset_id}-candidate.wav"
            try:
                # Re-derive a whole-clip map; a fit valid near one flagged span
                # cannot safely be extrapolated over the remaining recording.
                alignment = align_sources(
                    reference, source_transcript, camera_paths[asset_id], source_path,
                    conf, acoustic_first=True,
                )
                # One line for the clip, then the acoustic refinement splits it into
                # regimes and absorbs steps: a single line cannot fix a step.
                plan = render_aligned_clip(
                    alignment, PieceConfig(), source_path, candidate,
                    clip_duration_s=reference.duration,
                    recorder_duration_s=source_transcript.duration,
                    recorder_words=[(word.start, word.end) for word in source_transcript.words],
                    strategy=1, source_asset_id=source_id, target_asset_id=asset_id,
                    channels=channels, refine_audio=camera_paths[asset_id], refine=refine,
                    label=asset_id,
                    duck=PauseDuck.from_settings(
                        conf, [(word.start, word.end) for word in reference.words],
                        [(word.start, word.end) for word in source_transcript.words]),
                )
                rendered = engine.transcribe(candidate)
                outcome = check_rendered_clip(
                    rendered.words, reference.words, SelfCheckSettings(), reference.duration,
                )
                grid_s, window_s = short_clip_grid(reference.duration,
                                                   float(conf.get("refine_grid_s", 2.5)),
                                                   float(conf.get("refine_window_s", 4.0)))
                verification = measure(camera_paths[asset_id], candidate,
                                       grid_s=grid_s, window_s=window_s,
                                       source_duration_s=reference.duration)
                status, reason = verification.verdict(threshold)
                combined = combine_checks(outcome, verification, threshold)
                entry.update({"content": asdict(outcome), "lag_status": status,
                              "lag_reason": reason, "lag": verification.summary(threshold),
                              "refine": plan.refine})
                content_not_worse = len(combined["spans"]) <= initial_spans.get(asset_id, 0)
                if status == "passed" and content_not_worse:
                    final = directory / f"{asset_id}-repaired.wav"
                    candidate.replace(final)
                    replacements[asset_id] = final
                    maps[asset_id] = replace(plan.warp, evidence={
                        **(plan.warp.evidence or {}), "repair_verified": 1.0,
                    })
                    artifacts.append(final)
                    rendered.source_audio = final
                    transcript_path = directory / f"{asset_id}-repaired.json"
                    rendered.save(transcript_path)
                    artifacts.append(transcript_path)
                    entry["accepted"] = True
                else:
                    entry["reason"] = ("candidate failed the acoustic verdict or has more "
                                       "content defects than the original")
            except (RuntimeError, ValueError, OSError) as exc:
                entry["reason"] = str(exc)
            finally:
                candidate.unlink(missing_ok=True)
            reports.append(entry)
    finally:
        unload = getattr(engine.backend, "unload", None)
        if unload is not None:
            unload()
    report = directory / "report.json"
    report.write_text(json.dumps({"schema_version": 1, "mode": "repair",
                                 "policy": "verified whole-clip rerender with acoustic refinement",
                                 "clips": reports}, indent=2) + "\n", encoding="utf-8")
    return replacements, maps, (*artifacts, report)
