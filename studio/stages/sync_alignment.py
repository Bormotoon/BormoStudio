"""Acoustic-first alignment with accepted text evidence as a fallback."""

from pathlib import Path
from typing import Any

from studio.core.transcript import Transcript
from studio.stages.sync_acoustic import acoustic_coarse_align
from studio.stages.sync_engine import matching_transcript
from studio.stages.sync_match_settings import MatchSettings
from studio.stages.sync_matcher import align, evaluate_alignment
from studio.stages.sync_models import AlignmentMap


def align_sources(
    target: Transcript, source: Transcript, target_path: Path, source_path: Path,
    conf: dict[str, Any], *, acoustic_first: bool,
    source_range: tuple[float, float] | None = None,
) -> AlignmentMap:
    """``target = offset + k * source``; ``source_range`` bounds the acoustic search."""
    settings = MatchSettings()
    reasons = []

    def acoustic() -> AlignmentMap | None:
        try:
            fit = acoustic_coarse_align(
                target_path, source_path, target.duration, source.duration,
                grid_s=float(conf.get("acoustic_grid_s", 30.0)),
                window_s=float(conf.get("acoustic_window_s", 8.0)),
                min_sharpness=float(conf.get("acoustic_min_sharpness", 50.0)),
                max_k_deviation=settings.alignment_max_k_deviation,
                min_points=settings.alignment_min_acoustic_points,
                rec_range=source_range,
            )
        except RuntimeError as exc:
            reasons.append(str(exc))
            return None
        if fit is None:
            reasons.append("waveform scan found no unambiguous fit")
            return None
        candidate = AlignmentMap([], fit.offset, fit.k, 0.0, "acoustic",
                                 fit.inliers, fit.span_s)
        verdict = evaluate_alignment(candidate, target.duration, settings)
        if verdict.accepted:
            return candidate
        reasons.append(verdict.reason_text)
        return None

    if acoustic_first or not target.words:
        candidate = acoustic()
        if candidate is not None:
            return candidate
    if not target.words:
        raise ValueError("no accepted source alignment: " + "; ".join(reasons))
    try:
        candidate = align(matching_transcript(target), matching_transcript(source), settings)
        verdict = evaluate_alignment(candidate, target.duration, settings)
        if verdict.accepted:
            return candidate
        reasons.append(verdict.reason_text)
    except ValueError as exc:
        reasons.append(str(exc))
    if not acoustic_first:
        candidate = acoustic()
        if candidate is not None:
            return candidate
    raise ValueError("no accepted source alignment: " + "; ".join(reasons))
