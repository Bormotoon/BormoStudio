"""Check a camera–recorder clock map on the acoustic grid (``complex`` drift).

Text anchors (Whisper words, ±50–100 ms) cannot tell a clock ratio error of
7·10⁻⁵ from noise: over five minutes that is 20 ms, about the anchors' own
jitter. The acoustic grid measures the same map to a millisecond, so when it
has enough confident windows over enough of the clip it decides ``offset`` and
``k`` and supplies the anchors for strategies 2/3; text remains the coarse
placement and the fallback for cameras without usable sound.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from studio.stages.sync_grid import RegimeModel, decode_range, fit_model, measure_grid
from studio.stages.sync_models import AlignmentMap, Anchor
from studio.stages.sync_refine import RefineSettings
from studio.stages.sync_verify import SHORT_CLIP_S, SHORT_GRID_WINDOWS, VerifyReport

MIN_WINDOWS = 8
MIN_COVERAGE = 0.5
# Text anchors are kept only where no acoustic anchor is this close.
TEXT_GAP_S = 10.0
# Clock drift accumulated over the clip below this is no drift.
DRIFT_MIN_MS = 5.0
# complex may be worse than the constant (auto) plan by at most this much.
COMPLEX_TOLERANCE_MS = 1.0
# A clip too short or too quiet for a model is still confirmed by a few windows
# that agree on one offset (sony_run: dji-0935, 15 s, 3 windows of 5).
PARTIAL_MIN_WINDOWS = 3
PARTIAL_AGREE_MS = 20.0
ACOUSTIC_TOKEN = "acoustic:"
# Anchors placed from the recorder jump map carry their regime the same way.
JUMP_TOKEN = "jump:"


@dataclass(slots=True)
class DriftCheck:
    alignment: AlignmentMap
    report: dict[str, Any]
    # Only the constant plan (strategy 1) is supported by the acoustics.
    constant_only: bool = False
    # The acoustic model and the clip times of its confident windows, for the
    # recorder jump map shared by all cameras.
    model: RegimeModel | None = None
    confident_s: list[float] = field(default_factory=list)


def check_alignment(
    alignment: AlignmentMap,
    camera_audio: Path,
    recorder_audio: Path,
    *,
    clip_duration_s: float,
    recorder_duration_s: float,
    settings: RefineSettings,
    camera_track: np.ndarray | None = None,
    recorder_track: np.ndarray | None = None,
) -> DriftCheck:
    """Re-derive ``offset``/``k`` and anchors from the acoustic grid along ``alignment``."""
    k = alignment.k or 1.0
    rec0 = max(0.0, (0.0 - alignment.offset) / k - 2.0)
    rec1 = min(recorder_duration_s, (clip_duration_s - alignment.offset) / k + 2.0)
    report: dict[str, Any] = {
        "k_text": alignment.k, "offset_text": alignment.offset,
        "provenance_text": alignment.provenance,
    }
    if rec1 - rec0 <= 1.0:
        report.update(used="text", reason="no recorder audio under the clip")
        return DriftCheck(alignment, report)
    if camera_track is None:
        camera_track = decode_range(camera_audio, 0.0, clip_duration_s)
    if recorder_track is None:
        recorder_track = decode_range(recorder_audio, rec0, rec1 - rec0)
    else:  # the whole recorder from its start, mostly for tests
        recorder_track = recorder_track[int(rec0 * 16000):int(rec1 * 16000)]

    def mapping(t: float) -> float:
        return (t - alignment.offset) / k

    grid_s = settings.grid_s
    if clip_duration_s < SHORT_CLIP_S:
        grid_s = min(grid_s, max(0.5, clip_duration_s / SHORT_GRID_WINDOWS))
    grid = measure_grid(
        camera_track, recorder_track, mapping, duration_s=clip_duration_s,
        other_origin_s=rec0, grid_s=grid_s, window_s=settings.window_s,
        max_lag_s=settings.max_lag_s, min_sharpness=settings.min_sharpness,
    )
    times = [item.t for item in grid.confident]
    coverage = (max(times) - min(times)) / clip_duration_s if len(times) > 1 else 0.0
    report.update(windows=grid.attempted, confident=len(times), coverage=round(coverage, 3))
    model = fit_model(grid, step_min_ms=settings.step_min_ms)
    if model is None or len(times) < MIN_WINDOWS or coverage < MIN_COVERAGE:
        reason = (f"{len(times)} confident window(s) over {coverage:.0%} of the clip "
                  f"(need {MIN_WINDOWS} over {MIN_COVERAGE:.0%})")
        partial = _agreed_offset(grid)
        if partial is None:
            report.update(used="text", reason=reason)
            return DriftCheck(alignment, report, confident_s=times)
        # Too little for drift or steps, enough for where the clip is: the
        # windows agree on one offset from the text line.
        lag_ms, agreeing = partial
        new_k = 1.0 if clip_duration_s < SHORT_CLIP_S else k
        middle = float(np.median([item.t for item in agreeing]))
        rec_middle = mapping(middle) + lag_ms / 1000.0
        offset = middle - new_k * rec_middle
        report.update(used="acoustic offset", reason=reason, offset=offset, k=new_k,
                      lag_ms=round(lag_ms, 2), agreeing=len(agreeing))
        return DriftCheck(replace(alignment, offset=offset, k=new_k, provenance="acoustic",
                                  inliers=max(alignment.inliers, len(agreeing))),
                          report, confident_s=times)
    offset, new_k = _line_from_model(model, alignment.offset, k)
    anchors = _anchors(grid, mapping, model, alignment)
    constant_p90 = _constant_p90(grid, model)
    # Drift is what the acoustic clock ratio accumulates over the clip, not
    # whether a slope was needed to undo an error of the text ratio.
    drift_ms = abs(new_k - 1.0) * clip_duration_s * 1000.0
    text_drift_ms = abs(alignment.k - 1.0) * clip_duration_s * 1000.0
    drift_confirmed = drift_ms >= DRIFT_MIN_MS
    report.update(
        used="acoustic", model=model.summary(), k=new_k, offset=offset,
        p90_text_ms=round(model.p90_before_ms, 2), p90_ms=round(model.p90_after_ms, 2),
        p90_constant_ms=round(constant_p90, 2),
        drift="confirmed" if drift_confirmed else "not confirmed",
        drift_ms=round(drift_ms, 2), text_drift_ms=round(text_drift_ms, 2),
    )
    checked = replace(
        alignment, anchors=anchors, offset=offset, k=new_k,
        residual_ms=model.p90_after_ms, provenance="acoustic",
        inliers=max(alignment.inliers, len(times)),
    )
    constant_only = not drift_confirmed and text_drift_ms >= COMPLEX_TOLERANCE_MS
    if constant_only:
        report["complex"] = (
            f"complex rejected: drift is not confirmed by the acoustics "
            f"(k_text = {alignment.k:.7f}; p90 {model.p90_before_ms:.0f} ms against "
            f"{constant_p90:.0f} ms)")
    return DriftCheck(checked, report, constant_only, model, times)


def _agreed_offset(grid: VerifyReport) -> tuple[float, list[Any]] | None:
    """The offset (ms) most confident windows agree on, with those windows."""
    confident = list(grid.confident)
    if len(confident) < PARTIAL_MIN_WINDOWS:
        return None
    middle = float(np.median([item.lag_ms for item in confident]))
    agreeing = [item for item in confident if abs(item.lag_ms - middle) <= PARTIAL_AGREE_MS]
    if len(agreeing) < PARTIAL_MIN_WINDOWS or len(agreeing) < 0.6 * len(confident):
        return None
    return float(np.median([item.lag_ms for item in agreeing])), agreeing


def _line_from_model(model: RegimeModel, offset: float, k: float) -> tuple[float, float]:
    """``cam = offset' + k' * rec`` after reading the recorder ``residual(t)`` later.

    rec'(t) = (t − offset)/k + (a + b·(t − t0))/1000 for the dominant regime's
    offset ``a`` and slope ``b`` (ms per second); the slope is the windows-
    weighted mean over all regimes so a step does not bend the clock ratio.
    """
    dominant = max(model.regimes, key=lambda regime: regime.windows)
    total = sum(regime.windows for regime in model.regimes) or 1
    slope = sum(regime.slope * regime.windows for regime in model.regimes) / total
    a, t0 = dominant.offset_ms, dominant.anchor_s
    inverse = 1.0 / k + slope / 1000.0
    new_k = 1.0 / inverse
    intercept = -offset / k + a / 1000.0 - slope * t0 / 1000.0
    return -new_k * intercept, new_k


def _constant_p90(grid: VerifyReport, model: RegimeModel) -> float:
    """Predicted p90 of the constant-per-regime (``auto``) plan."""
    from studio.stages.sync_grid import p90

    values = []
    for item in grid.confident:
        regime = model.regimes[model.regime_index(item.t)]
        values.append(item.lag_ms - regime.offset_ms)
    return p90(values)


def _anchors(
    grid: VerifyReport, mapping: Any, model: RegimeModel, alignment: AlignmentMap,
) -> list[Anchor]:
    """One anchor per confident window, tagged with its regime; text fills the gaps."""
    acoustic = [
        Anchor(item.t, mapping(item.t) + item.lag_ms / 1000.0,
               f"{ACOUSTIC_TOKEN}{model.regime_index(item.t)}", min(1.0, item.sharpness / 200))
        for item in sorted(grid.confident, key=lambda item: item.t)
        if not any(abs(item.t - outlier) < 1e-9 for outlier in model.outliers)
    ]
    times = np.array([anchor.cam_time for anchor in acoustic])
    text = [
        anchor for anchor in alignment.anchors
        if not times.size or float(np.min(np.abs(times - anchor.cam_time))) > TEXT_GAP_S
    ]
    return sorted([*acoustic, *text], key=lambda anchor: anchor.rec_time)


def anchor_regime(anchor: Anchor) -> int | None:
    token = getattr(anchor, "token", "")
    for prefix in (ACOUSTIC_TOKEN, JUMP_TOKEN):
        if isinstance(token, str) and token.startswith(prefix):
            return int(token[len(prefix):])
    return None
