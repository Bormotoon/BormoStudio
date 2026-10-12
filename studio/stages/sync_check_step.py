"""Re-recognize rendered voices and publish honest content-check outcomes."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from studio.core.transcript import Transcript
from studio.stages.sync_check_settings import SelfCheckSettings
from studio.stages.sync_self_check import SelfCheckOutcome, check_rendered_clip
from studio.stages.sync_verify import VerifyReport
from studio.stages.transcribe import TranscribeSettings, WhisperEngine


def check_voices(
    references: dict[str, Transcript], voices: dict[str, Path], directory: Path,
    transcribe_settings: object, cameras: dict[str, Path] | None = None,
    sync_settings: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Content check by Whisper plus, when camera audio is given, the acoustic verdict.

    A clip fails on a failed acoustic verdict or a content span the acoustics
    do not explain. A content span where the acoustic windows are confident and
    within the threshold is speech Whisper missed in the camera audio, not a
    render defect: it is reported as a ``camera_transcript_gap`` warning.
    """
    engine = WhisperEngine(TranscribeSettings.from_mapping(transcribe_settings))
    conf = sync_settings or {}
    threshold = float(conf.get("max_drift_ms", 20.0))
    reports = []
    artifacts = []
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for asset_id, reference in references.items():
            rendered = engine.transcribe(voices[asset_id])
            transcript_path = directory / f"{asset_id}.json"
            rendered.save(transcript_path)
            artifacts.append(transcript_path)
            outcome = check_rendered_clip(
                rendered.words, reference.words, SelfCheckSettings(), reference.duration,
            )
            entry: dict[str, Any] = {"asset_id": asset_id, "time_domain": "rendered_audio"}
            camera = (cameras or {}).get(asset_id)
            if camera is None:
                entry.update(asdict(outcome))
            else:
                from studio.stages.sync_verify import measure, short_clip_grid

                grid_s = float(conf.get("refine_grid_s", 2.5))
                window_s = float(conf.get("refine_window_s", 4.0))
                if reference.duration:
                    grid_s, window_s = short_clip_grid(reference.duration, grid_s, window_s)
                try:
                    acoustic = measure(camera, voices[asset_id], grid_s=grid_s, window_s=window_s,
                                       source_duration_s=reference.duration or None)
                except (RuntimeError, OSError, ValueError) as exc:
                    entry.update(asdict(outcome))
                    entry["acoustic"] = {"status": "inconclusive", "reason": str(exc)}
                else:
                    entry.update(combine_checks(outcome, acoustic, threshold))
            reports.append(entry)
    finally:
        unload = getattr(engine.backend, "unload", None)
        if unload is not None:
            unload()
    report = directory / "report.json"
    report.write_text(json.dumps({"schema_version": 2, "mode": "warn", "clips": reports},
                                 indent=2) + "\n", encoding="utf-8")
    return (*artifacts, report)


def combine_checks(
    content: SelfCheckOutcome, acoustic: VerifyReport, threshold_ms: float
) -> dict[str, Any]:
    """One clip status from the Whisper content check and the acoustic verdict."""
    acoustic_status, acoustic_reason = acoustic.verdict(threshold_ms)
    confirmed = []
    warnings = []
    for span in content.spans:
        inside = [item for item in acoustic.confident if span.start <= item.t <= span.end]
        if (span.kind == "content" and inside
                and all(abs(item.lag_ms) <= threshold_ms for item in inside)):
            warnings.append({
                "kind": "camera_transcript_gap", "start": span.start, "end": span.end,
                "detail": f"{span.detail}; {len(inside)} acoustic window(s) agree within "
                          f"{threshold_ms:g} ms, so the speech is in place",
            })
        else:
            confirmed.append(span)
    if acoustic_status == "failed" or confirmed:
        status = "failed"
    elif content.status == "inconclusive" and acoustic_status == "inconclusive":
        status = "inconclusive"
    else:
        status = "passed"
    return {
        "status": status,
        "spans": [asdict(span) for span in confirmed],
        "detail": content.detail,
        "content_status": content.status,
        "warnings": warnings,
        "acoustic": {"status": acoustic_status, "reason": acoustic_reason,
                     **acoustic.summary(threshold_ms)},
    }
