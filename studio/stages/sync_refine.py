"""Acoustic refinement of a piece plan before rendering (steps inside a clip).

Every strategy builds a continuous map from word timings, so a recorder that
drops a few dozen milliseconds mid-clip (a click, a buffer underrun) leaves one
part of the clip shifted. This pass measures the plan on the acoustic grid,
models the residual as regimes and re-targets the plan's read positions:

* output windows of the pieces do not move, only where they read;
* at a step between regimes A and B one short absorption piece of output
  length ``w`` reads ``w + Δ`` of the recorder, placed in the nearest pause
  between recorder words; without a pause nearby it is widened so the tempo
  change stays within 3 % and a warning is reported;
* the map stays continuous on both time scales (``AudioWarpMap`` contract), so
  export, speakers and raw warp replay are untouched by design.

The refinement is applied only when the predicted p90 is within the threshold
and at least 2 ms better than the plan as it was.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from studio.stages.sync_acoustic import _REFINE_SR, _window_slice, gcc_phat
from studio.stages.sync_grid import (
    MODEL_GAIN_MS,
    Piece,
    RegimeModel,
    Step,
    decode_range,
    fit_model,
    measure_grid,
    piece_mapping,
)
from studio.stages.sync_pieces import MAX_TEMPO_FACTOR, MIN_TEMPO_FACTOR, validate_pieces
from studio.stages.sync_verify import DEFAULT_GCC_EPS

# An offset this small is not worth touching the plan for.
NEGLIGIBLE_MS = 3.0
FINE_WINDOW_S = 1.0
FINE_STEP_S = 0.1
FINE_MIN_SHARPNESS = 20.0
CLICK_BAND_HZ = 2000.0
CLICK_RISE_DB = 20.0
CLICK_MATCH_S = 1.0
# Absorption: tempo change inside a pause (20 %) or, if no pause is near, in speech (3 %).
PAUSE_TEMPO = 0.2
SPEECH_TEMPO = 0.03
MIN_ABSORB_S = 0.3


@dataclass(frozen=True, slots=True)
class RefineSettings:
    grid_s: float = 2.5
    window_s: float = 4.0
    step_min_ms: float = 10.0
    threshold_ms: float = 20.0
    seam_snap_max_s: float = 0.4
    min_sharpness: float = 50.0
    # Jumps of the recorder reach 0.8 s; a narrower lag range does not see them.
    max_lag_s: float = 1.0

    @classmethod
    def from_sync(cls, sync: dict[str, Any], seam_snap_max_s: float = 0.4) -> RefineSettings:
        return cls(
            grid_s=float(sync.get("refine_grid_s", 2.5)),
            window_s=float(sync.get("refine_window_s", 4.0)),
            step_min_ms=float(sync.get("step_min_ms", 10.0)),
            threshold_ms=float(sync.get("max_drift_ms", 20.0)),
            seam_snap_max_s=seam_snap_max_s,
            min_sharpness=float(sync.get("acoustic_min_sharpness", 50.0)),
            max_lag_s=float(sync.get("refine_max_lag_s", 1.0)),
        )


@dataclass(slots=True)
class RefineResult:
    lead: float
    pieces: list[Piece]
    accepted: bool
    report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def piece_inverse(lead: float, pieces: Sequence[Piece]) -> Any:
    """Recorder time → clip time of a contiguous piece plan (``None`` outside)."""
    outputs: list[float] = []
    position = lead
    for _start, duration, factor in pieces:
        outputs.append(position)
        position += duration / factor

    def inverse(rec: float) -> float | None:
        for (start, duration, factor), out in zip(pieces, outputs, strict=True):
            if start <= rec <= start + duration:
                return out + (rec - start) / factor
        return None

    return inverse


def refine_plan(
    lead: float,
    pieces: list[Piece],
    camera_audio: Path,
    recorder_audio: Path,
    *,
    clip_duration_s: float,
    recorder_duration_s: float,
    recorder_words: Sequence[tuple[float, float]],
    settings: RefineSettings,
    label: str = "",
    camera_track: np.ndarray | None = None,
    recorder_track: np.ndarray | None = None,
    recorder_origin_s: float = 0.0,
) -> RefineResult:
    """Measure ``pieces`` acoustically and correct offsets and steps when that helps."""
    if not pieces:
        return RefineResult(lead, pieces, False, {"accepted": False, "reason": "empty plan"})
    if camera_track is None:
        camera_track = decode_range(camera_audio, 0.0, clip_duration_s)
    if recorder_track is None:
        recorder_origin_s = max(0.0, pieces[0][0] - 2.0)
        end = min(recorder_duration_s, pieces[-1][0] + pieces[-1][1] + 2.0)
        recorder_track = decode_range(recorder_audio, recorder_origin_s, end - recorder_origin_s)
    mapping = piece_mapping(lead, pieces)
    grid = measure_grid(
        camera_track, recorder_track, mapping, duration_s=clip_duration_s,
        other_origin_s=recorder_origin_s, grid_s=settings.grid_s, window_s=settings.window_s,
        max_lag_s=settings.max_lag_s, min_sharpness=settings.min_sharpness,
    )
    model = fit_model(grid, step_min_ms=settings.step_min_ms)
    report: dict[str, Any] = {
        "windows": grid.attempted, "confident": len(grid.confident),
    }
    if model is None:
        report.update(accepted=False, reason="too few confident windows")
        return RefineResult(lead, pieces, False, report)
    report.update(model.summary())
    if model.kind == "constant" and abs(model.regimes[0].offset_ms) <= NEGLIGIBLE_MS:
        report.update(accepted=False, reason=f"plan is within {NEGLIGIBLE_MS:g} ms")
        return RefineResult(lead, pieces, False, report)
    if (model.p90_after_ms > settings.threshold_ms
            or model.p90_after_ms > model.p90_before_ms - MODEL_GAIN_MS):
        report.update(accepted=False, reason=(
            f"refine rejected: predicted p90 {model.p90_after_ms:.1f} ms "
            f"(before {model.p90_before_ms:.1f} ms, threshold {settings.threshold_ms:g} ms)"))
        return RefineResult(lead, pieces, False, report)

    inverse = piece_inverse(lead, pieces)
    warnings: list[str] = []
    windows: list[tuple[float, float, int]] = []  # (left, right, step index)
    step_reports: list[dict[str, Any]] = []
    for index, step in enumerate(model.steps):
        position, located_by = locate_step(
            step, model, camera_track, recorder_track, mapping, recorder_origin_s, inverse,
            settings.window_s)
        recorder_s = mapping(position)
        if camera_pts_gap(camera_audio, position):
            report.update(accepted=False, reason=(
                f"camera audio is discontinuous near {position:.1f} s; step left as is"))
            return RefineResult(lead, pieces, False, report, [
                f"{label}: camera audio is discontinuous near {position:.1f} s of the clip; "
                "the sync step there was not corrected"])
        left, right, in_pause = absorption_window(
            position, step.delta_ms / 1000.0, recorder_words, inverse, lead, pieces,
            settings.seam_snap_max_s)
        windows.append((left, right, index))
        rec_text = f"{recorder_s:.1f} s of the recorder" if recorder_s is not None else (
            f"{position:.1f} s of the clip")
        step_reports.append({
            "camera_s": round(position, 3),
            "recorder_s": round(recorder_s, 3) if recorder_s is not None else None,
            "delta_ms": round(step.delta_ms, 2),
            "located_by": located_by,
            "absorb": [round(left, 3), round(right, 3)],
            "absorbed_in": "pause" if in_pause else "speech",
        })
        warnings.append(
            f"{label}: sync jumps by {step.delta_ms:+.0f} ms at {rec_text}; corrected"
            + ("" if in_pause else " (absorbed in speech, check by ear)"))
    try:
        new_lead, refined = retarget(lead, pieces, model, windows, recorder_duration_s)
    except ValueError as exc:
        report.update(accepted=False, reason=f"refine rejected: {exc}", steps=step_reports)
        return RefineResult(lead, pieces, False, report)
    problems = validate_pieces(new_lead, refined, clip_duration_s, recorder_duration_s)
    if problems:
        report.update(accepted=False, reason="refine rejected: " + "; ".join(problems),
                      steps=step_reports)
        return RefineResult(lead, pieces, False, report)
    report.update(accepted=True, reason="", steps=step_reports)
    return RefineResult(new_lead, refined, True, report, warnings)


def locate_step(
    step: Step, model: RegimeModel, camera: np.ndarray, recorder: np.ndarray,
    mapping: Any, recorder_origin_s: float, inverse: Any, window_s: float = 4.0,
) -> tuple[float, str]:
    """Clip time of a step: a recorder click, else the fine grid, else the coarse midpoint.

    A grid window straddling the step can be classified either way, so the
    search covers half a window beyond the last A and first B windows.
    """
    span = (step.before_s - window_s / 2, step.after_s + window_s / 2)
    fine = _fine_transition(step, model, camera, recorder, mapping, recorder_origin_s, span)
    click = _click(span, recorder, mapping, recorder_origin_s, inverse)
    if click is not None and (fine is None or abs(click - fine) <= CLICK_MATCH_S):
        return click, "click"
    if fine is not None:
        return fine, "fine"
    return step.t_s, "coarse"


def _fine_transition(
    step: Step, model: RegimeModel, camera: np.ndarray, recorder: np.ndarray,
    mapping: Any, recorder_origin_s: float, span: tuple[float, float],
) -> float | None:
    index = model.regime_index(step.before_s)
    if index + 1 >= len(model.regimes):
        return None
    before, after = model.regimes[index], model.regimes[index + 1]
    points: list[tuple[float, bool]] = []
    t = max(FINE_WINDOW_S / 2, span[0])
    while t <= span[1] + 1e-9:
        target = mapping(t)
        if target is not None:
            lag, sharpness = gcc_phat(
                _window_slice(camera, t, FINE_WINDOW_S, _REFINE_SR),
                _window_slice(recorder, target - recorder_origin_s, FINE_WINDOW_S, _REFINE_SR),
                _REFINE_SR, 0.25, DEFAULT_GCC_EPS)
            if sharpness >= FINE_MIN_SHARPNESS:
                residual = -float(lag) * 1000.0
                points.append((t, abs(residual - after.at(t)) < abs(residual - before.at(t))))
        t += FINE_STEP_S
    if len(points) < 4:
        return None
    # The cut that agrees with the most points: A before it, B after it.
    flags = [is_after for _t, is_after in points]
    scores = [flags[:cut].count(False) + flags[cut:].count(True)
              for cut in range(1, len(points))]
    cut = 1 + int(np.argmax(scores))
    if scores[cut - 1] < 0.8 * len(points):
        return None
    return (points[cut - 1][0] + points[cut][0]) / 2


def _click(
    span: tuple[float, float], recorder: np.ndarray, mapping: Any, recorder_origin_s: float,
    inverse: Any,
) -> float | None:
    start, end = mapping(max(span[0], 0.0)), mapping(span[1])
    if start is None or end is None or end <= start:
        return None
    lo = max(0, int((start - recorder_origin_s - 0.5) * _REFINE_SR))
    hi = min(len(recorder), int((end - recorder_origin_s + 0.5) * _REFINE_SR))
    segment = recorder[lo:hi].astype(np.float64)
    frame = int(0.01 * _REFINE_SR)
    if segment.size < frame * 20:
        return None
    spectrum = np.fft.rfft(segment)
    spectrum[: int(CLICK_BAND_HZ * segment.size / _REFINE_SR)] = 0
    high = np.fft.irfft(spectrum, segment.size)
    count = high.size // frame
    energy = 10 * np.log10(np.mean(high[: count * frame].reshape(count, frame) ** 2, axis=1)
                           + 1e-12)
    # A click is a transient: louder than the frames shortly before AND after it.
    # A word onset is loud afterwards too, so it never qualifies.
    def surroundings(i: int) -> float:
        before = energy[max(0, i - 8):max(0, i - 2)]
        after = energy[i + 3:i + 9]
        parts = [float(np.median(part)) for part in (before, after) if part.size]
        return max(parts) if parts else float(energy[i])

    excess = np.array([energy[i] - surroundings(i) for i in range(count)])
    peak = int(np.argmax(excess))
    if excess[peak] < CLICK_RISE_DB:
        return None
    rec_time = recorder_origin_s + (lo + (peak + 0.5) * frame) / _REFINE_SR
    clip_time = inverse(rec_time)
    if clip_time is None or not span[0] <= clip_time <= span[1]:
        return None
    return float(clip_time)


def camera_pts_gap(camera: Path, t: float, radius_s: float = 2.0) -> bool:
    """Whether the camera's audio packets jump near clip time ``t`` (camera-side loss)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-read_intervals", f"{max(0.0, t - radius_s)}%{t + radius_s}",
             "-show_entries", "packet=pts_time,duration_time", "-of", "json", str(camera)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        packets = json.loads(result.stdout).get("packets", [])
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    previous: tuple[float, float] | None = None
    for packet in packets:
        try:
            pts, duration = float(packet["pts_time"]), float(packet.get("duration_time", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if previous is not None and pts - (previous[0] + previous[1]) > max(
                0.5 * previous[1], 0.002):
            return True
        previous = (pts, duration)
    return False


def absorption_window(
    position: float, delta_s: float, recorder_words: Sequence[tuple[float, float]],
    inverse: Any, lead: float, pieces: Sequence[Piece], snap_s: float,
) -> tuple[float, float, bool]:
    """Output-time window that absorbs ``delta_s``: in a pause if one is near."""
    width = max(MIN_ABSORB_S, abs(delta_s) / PAUSE_TEMPO)
    spans = sorted(recorder_words)
    # The pause nearest to the step (≤ ``snap_s`` away): everything between the
    # step and the window is read with the wrong regime, so closer is better.
    best: tuple[float, float, float] | None = None  # (distance, left, right)
    for (_a0, a1), (b0, _b1) in zip(spans, spans[1:], strict=False):
        left, right = inverse(a1), inverse(b0)
        if left is None or right is None or right - left < width:
            continue
        distance = max(0.0, left - position, position - right)
        if distance <= snap_s and (best is None or distance < best[0]):
            best = (distance, left, right)
    end = lead + sum(duration / factor for _s, duration, factor in pieces)
    if best is not None:
        _distance, pause_left, pause_right = best
        center = min(max(position, pause_left + width / 2), pause_right - width / 2)
        in_pause = True
    else:
        width = max(width, abs(delta_s) / SPEECH_TEMPO)
        center, in_pause = position, False
    left = max(lead, center - width / 2)
    right = min(end, left + width)
    return left, right, in_pause


def retarget(
    lead: float, pieces: Sequence[Piece], model: RegimeModel,
    windows: Sequence[tuple[float, float, int]], recorder_duration_s: float,
) -> tuple[float, list[Piece]]:
    """Move every read boundary by the model's residual; windows absorb the steps."""
    mapping = piece_mapping(lead, pieces)
    outputs = [lead]
    for _start, duration, factor in pieces:
        outputs.append(outputs[-1] + duration / factor)
    end = outputs[-1]

    def inside(u: float) -> bool:
        return any(left + 1e-6 < u < right - 1e-6 for left, right, _i in windows)

    # (output time, source time before correction, correction ms)
    boundaries: dict[float, float] = {}
    for u in outputs:
        if not inside(u):
            boundaries[round(u, 9)] = model.residual_at(u)
    for left, right, index in windows:
        step_regime = model.regime_index(model.steps[index].before_s)
        boundaries[round(left, 9)] = model.regimes[step_regime].at(left)
        boundaries[round(right, 9)] = model.regimes[step_regime + 1].at(right)
    points = sorted(boundaries.items())
    sources: list[float] = []
    for u, correction in points:
        base = mapping(min(max(u, lead), end))
        if base is None:
            raise ValueError("boundary outside the plan")
        sources.append(base + correction / 1000.0)
    sources[0] = max(0.0, sources[0])
    sources[-1] = min(recorder_duration_s, sources[-1])
    refined: list[Piece] = []
    for (u0, _c0), (u1, _c1), s0, s1 in zip(points, points[1:], sources, sources[1:],
                                            strict=False):
        out, src = u1 - u0, s1 - s0
        if out <= 1e-6:
            continue
        if src <= 0:
            raise ValueError("correction reverses the recorder")
        factor = src / out
        if not MIN_TEMPO_FACTOR <= factor <= MAX_TEMPO_FACTOR:
            raise ValueError(f"tempo {factor:.3f} outside the atempo range")
        refined.append((s0, src, factor))
    return points[0][0], refined
