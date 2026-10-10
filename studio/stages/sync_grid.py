"""Acoustic grid between camera audio and a recorder or rendered voice.

One component measures and models how far the recorder read position is off,
for the verdict (rendered voice), for the acoustic refinement of a plan before
rendering and for checking a drift fit (``complex``):

* ``measure_grid`` correlates a window of the camera's own audio at clip time
  ``t`` with the second track at ``mapping(t)`` (GCC-PHAT). ``mapping`` is the
  identity for a rendered voice and the piece map "clip time → recorder time"
  for a plan that is not rendered yet.
* The sign is the same everywhere: ``residual(t)`` (``LagSample.lag_ms``) is
  how much later (+) or earlier (−) the recorder must be read at ``t``.
* ``segment_regimes`` splits the residuals into stretches with one offset
  (optionally a small trend); ``fit_model`` picks the simplest of "constant",
  "line" or "regimes" whose predicted p90 is not beaten by ≥ 2 ms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from studio.stages.sync_acoustic import _REFINE_SR, _window_slice, gcc_phat, load_mono16k_track
from studio.stages.sync_verify import DEFAULT_GCC_EPS, LagSample, VerifyReport

GRID_S = 2.5
WINDOW_S = 4.0
MAX_LAG_S = 0.25
MIN_SHARPNESS = 50.0
STEP_MIN_MS = 10.0
MIN_WINDOWS = 3
MAX_MAD_MS = 5.0
# A more complex model (line over constant, regimes over line, a trend inside
# a regime) must improve the predicted p90 by at least this much.
MODEL_GAIN_MS = 2.0

Mapping = Callable[[float], "float | None"]
Piece = tuple[float, float, float]


def decode_range(path: Path, start_s: float, duration_s: float | None) -> np.ndarray:
    """Mono 16 kHz samples of ``[start_s, start_s + duration_s]`` of ``path``."""
    return load_mono16k_track(path, start_s=max(0.0, start_s) or None, duration_s=duration_s)


def piece_mapping(lead: float, pieces: Sequence[Piece]) -> Mapping:
    """Clip time → recorder time of a contiguous piece plan (``None`` before ``lead``)."""
    starts: list[float] = []
    position = lead
    for _start, duration, factor in pieces:
        starts.append(position)
        position += duration / factor
    end = position

    def mapping(t: float) -> float | None:
        if not pieces or t < lead or t > end:
            return None
        index = int(np.searchsorted(starts, t, side="right")) - 1
        index = min(max(index, 0), len(pieces) - 1)
        start, _duration, factor = pieces[index]
        return start + (t - starts[index]) * factor

    return mapping


def measure_grid(
    camera: np.ndarray,
    other: np.ndarray,
    mapping: Mapping | None = None,
    *,
    duration_s: float | None = None,
    other_origin_s: float = 0.0,
    grid_s: float = GRID_S,
    window_s: float = WINDOW_S,
    max_lag_s: float = MAX_LAG_S,
    min_sharpness: float = MIN_SHARPNESS,
    video: str = "",
    voice: str = "",
) -> VerifyReport:
    """Residual per grid window between ``camera`` (clip time 0 at index 0) and ``other``.

    ``other`` starts at file time ``other_origin_s``; ``mapping`` (default:
    identity) gives the file time that should sound at clip time ``t``.
    """
    if grid_s <= 0 or window_s <= 0:
        raise ValueError("grid_s and window_s must be positive")
    camera_s = len(camera) / _REFINE_SR
    other_s = len(other) / _REFINE_SR
    if mapping is None:
        duration = min(camera_s, other_s) if duration_s is None else min(duration_s, camera_s)
    else:
        duration = camera_s if duration_s is None else min(duration_s, camera_s)
    report = VerifyReport(video, voice, duration_s=duration, min_sharpness=min_sharpness,
                          grid_s=grid_s)
    half = window_s / 2.0
    t = half
    while t <= duration - half + 1e-9:
        target = t if mapping is None else mapping(t)
        position = None if target is None else target - other_origin_s
        if position is None or position - half < 0 or position + half > other_s:
            report.rejected.append((t, "outside the second track"))
        else:
            lag_s, sharpness = gcc_phat(
                _window_slice(camera, t, window_s, _REFINE_SR),
                _window_slice(other, position, window_s, _REFINE_SR),
                _REFINE_SR, max_lag_s, DEFAULT_GCC_EPS,
            )
            if sharpness >= min_sharpness:
                # gcc_phat reports a late query as a negative lag; the residual
                # says to read the recorder that much later.
                report.samples.append(LagSample(t, -float(lag_s) * 1000.0, float(sharpness)))
            else:
                report.rejected.append((t, f"correlation peak too weak ({sharpness:.1f})"))
        t += grid_s
    return report


@dataclass(frozen=True, slots=True)
class Regime:
    start_s: float
    end_s: float
    offset_ms: float
    slope: float = 0.0  # ms per second, around ``anchor_s``
    anchor_s: float = 0.0
    windows: int = 0
    mad_ms: float = 0.0

    def at(self, t: float) -> float:
        return self.offset_ms + self.slope * (t - self.anchor_s)


@dataclass(frozen=True, slots=True)
class Step:
    """A jump between two regimes, located between ``before_s`` and ``after_s``."""

    t_s: float
    before_s: float
    after_s: float
    delta_ms: float


@dataclass(slots=True)
class RegimeModel:
    kind: str  # constant | line | regimes
    regimes: list[Regime]
    steps: list[Step] = field(default_factory=list)
    p90_before_ms: float = 0.0
    p90_after_ms: float = 0.0
    windows: int = 0
    outliers: list[float] = field(default_factory=list)

    def residual_at(self, t: float) -> float:
        for regime in self.regimes:
            if t < regime.end_s:
                return regime.at(t)
        return self.regimes[-1].at(t) if self.regimes else 0.0

    def regime_index(self, t: float) -> int:
        for index, regime in enumerate(self.regimes):
            if t < regime.end_s:
                return index
        return len(self.regimes) - 1

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.kind,
            "windows": self.windows,
            "p90_before_ms": round(self.p90_before_ms, 2),
            "p90_after_ms": round(self.p90_after_ms, 2),
            "regimes": [
                {"start_s": round(item.start_s, 3), "end_s": round(item.end_s, 3),
                 "offset_ms": round(item.offset_ms, 2), "slope_ms_per_s": round(item.slope, 4),
                 "windows": item.windows, "mad_ms": round(item.mad_ms, 2)}
                for item in self.regimes
            ],
            "steps": [
                {"t_s": round(item.t_s, 3), "delta_ms": round(item.delta_ms, 2)}
                for item in self.steps
            ],
            "outliers": len(self.outliers),
        }


def p90(values: Sequence[float] | np.ndarray) -> float:
    data = np.sort(np.abs(np.asarray(values, dtype=np.float64)))
    if not data.size:
        return 0.0
    return float(data[min(data.size - 1, int(data.size * 0.9))])


def _mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values)))) if values.size else 0.0


def segment_regimes(
    samples: Sequence[LagSample],
    *,
    step_min_ms: float = STEP_MIN_MS,
    min_windows: int = MIN_WINDOWS,
    max_mad_ms: float = MAX_MAD_MS,
) -> list[tuple[int, int]]:
    """Index ranges of ``samples`` (sorted by time) that share one offset.

    Binary segmentation: a range is split where the medians of its two sides
    differ the most, if that jump is at least ``step_min_ms`` and each side has
    ``min_windows`` windows; the cut is then moved to where the medians of the
    neighbouring ``min_windows`` groups jump. Shorter groups stay inside their
    neighbour and are later treated as outliers.
    """
    ordered = sorted(samples, key=lambda item: item.t)
    values = np.array([item.lag_ms for item in ordered], dtype=np.float64)

    def split(lo: int, hi: int) -> list[tuple[int, int]]:
        if hi - lo < 2 * min_windows:
            return [(lo, hi)]
        stride = max(1, (hi - lo) // 200)
        cuts = list(range(lo + min_windows, hi - min_windows + 1, stride))
        jumps = [abs(float(np.median(values[lo:cut])) - float(np.median(values[cut:hi])))
                 for cut in cuts]
        best_jump = max(jumps, default=0.0)
        if best_jump < step_min_ms:
            return [(lo, hi)]

        def local(cut: int) -> float:
            return abs(float(np.median(values[max(lo, cut - min_windows):cut]))
                       - float(np.median(values[cut:min(hi, cut + min_windows)])))

        # Whole-side medians stay flat while the cut moves inside one regime;
        # among those near-maximal cuts the step is where neighbouring groups jump.
        plateau = [cut for cut, jump in zip(cuts, jumps, strict=True)
                   if jump >= best_jump - step_min_ms / 2]
        refined = max(
            (cut for near in plateau
             for cut in range(max(lo + min_windows, near - stride),
                              min(hi - min_windows, near + stride) + 1)),
            key=local,
        )
        return [*split(lo, refined), *split(refined, hi)]

    ranges = split(0, len(ordered)) if ordered else []
    merged: list[tuple[int, int]] = []
    for lo, hi in ranges:
        if merged and abs(float(np.median(values[merged[-1][0]:merged[-1][1]]))
                          - float(np.median(values[lo:hi]))) < step_min_ms:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    return merged


def _fit_regime(times: np.ndarray, values: np.ndarray, start: float, end: float,
                max_mad_ms: float) -> tuple[Regime, np.ndarray]:
    offset = float(np.median(values))
    constant = values - offset
    mad = _mad(values)
    keep = np.abs(constant) <= max(3.0 * 1.4826 * mad, STEP_MIN_MS / 2, 1.0)
    regime = Regime(start, end, offset, 0.0, float(np.median(times)), int(values.size), mad)
    residual = constant
    if keep.sum() >= max(3, MIN_WINDOWS) and float(np.ptp(times[keep])) > 0:
        anchor = float(np.median(times[keep]))
        slope, intercept = np.polyfit(times[keep] - anchor, values[keep], 1)
        trended = values - (intercept + slope * (times - anchor))
        if p90(trended[keep]) <= p90(constant[keep]) - MODEL_GAIN_MS:
            regime = Regime(start, end, float(intercept), float(slope), anchor,
                            int(values.size), _mad(trended))
            residual = trended
    return regime, residual


def fit_model(
    report: VerifyReport,
    *,
    step_min_ms: float = STEP_MIN_MS,
    min_windows: int = MIN_WINDOWS,
    max_mad_ms: float = MAX_MAD_MS,
) -> RegimeModel | None:
    """The simplest model of the residuals that predicts them well.

    ``None`` without at least ``min_windows`` confident windows.
    """
    ordered = sorted(report.confident, key=lambda item: item.t)
    if len(ordered) < min_windows:
        return None
    times = np.array([item.t for item in ordered], dtype=np.float64)
    values = np.array([item.lag_ms for item in ordered], dtype=np.float64)
    duration = max(report.duration_s, float(times[-1]))
    before = p90(values)

    def outliers_of(residual: np.ndarray, mad: float) -> np.ndarray:
        # Only short groups are outliers: a sustained run of large residuals is
        # exactly what the model has to explain.
        large = np.abs(residual) > max(3.0 * 1.4826 * mad, step_min_ms)
        result = np.zeros_like(large)
        index = 0
        while index < large.size:
            if not large[index]:
                index += 1
                continue
            end = index
            while end < large.size and large[end]:
                end += 1
            if end - index < min_windows:
                result[index:end] = True
            index = end
        return result

    constant_regime = Regime(0.0, duration + 1.0, float(np.median(values)), 0.0,
                             float(np.median(times)), len(values), _mad(values))
    constant_residual = values - constant_regime.offset_ms
    mask = ~outliers_of(constant_residual, constant_regime.mad_ms)
    best = RegimeModel("constant", [constant_regime], [], before,
                       p90(constant_residual[mask]), len(values),
                       [float(t) for t in times[~mask]])

    line_regime, line_residual = _fit_regime(times, values, 0.0, duration + 1.0, max_mad_ms)
    if line_regime.slope:
        mask = ~outliers_of(line_residual, line_regime.mad_ms)
        after = p90(line_residual[mask])
        if after <= best.p90_after_ms - MODEL_GAIN_MS:
            best = RegimeModel("line", [line_regime], [], before, after, len(values),
                               [float(t) for t in times[~mask]])

    ranges = segment_regimes(ordered, step_min_ms=step_min_ms, min_windows=min_windows,
                             max_mad_ms=max_mad_ms)
    if len(ranges) > 1:
        regimes: list[Regime] = []
        residuals: list[np.ndarray] = []
        steps: list[Step] = []
        for index, (lo, hi) in enumerate(ranges):
            start = 0.0 if index == 0 else float((times[lo - 1] + times[lo]) / 2)
            end = duration + 1.0 if index == len(ranges) - 1 else float(
                (times[hi - 1] + times[hi]) / 2)
            regime, residual = _fit_regime(times[lo:hi], values[lo:hi], start, end, max_mad_ms)
            regimes.append(regime)
            residuals.append(residual)
            if index:
                previous = regimes[index - 1]
                steps.append(Step(start, float(times[lo - 1]), float(times[lo]),
                                  regime.at(start) - previous.at(start)))
        residual = np.concatenate(residuals)
        mad = max(_mad(item) for item in residuals)
        mask = ~outliers_of(residual, mad)
        after = p90(residual[mask])
        if after <= best.p90_after_ms - MODEL_GAIN_MS:
            best = RegimeModel("regimes", regimes, steps, before, after, len(values),
                               [float(t) for t in times[~mask]])
    return best


def predicted_p90(report: VerifyReport, correction: Callable[[float], float]) -> float:
    """p90 of the residuals left after reading the recorder ``correction(t)`` ms later."""
    values = [item.lag_ms - correction(item.t) for item in report.confident]
    return p90(values)
