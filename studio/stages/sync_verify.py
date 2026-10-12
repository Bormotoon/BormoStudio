"""Realized lip-sync measurement between a source clip and a rendered voice.

Cross-correlate a short window of the video's own audio against the rendered
voice track at points across the clip (GCC-PHAT, the same function Boundary
Flex uses) and report the realized lag distribution. A sharp, confident peak
means "this is genuinely how far off the two tracks are" rather than
correlation noise, so unconfident windows are rejected.

This lives in ``whispersync.engine`` — not in ``tools/`` — because it is part
of the shipped product: ``--verify`` imports it at the END of a sync run. With
the implementation in ``tools/``, which package discovery does not include, a
wheel-installed WhisperSync raised ``ModuleNotFoundError`` at that point, after
the entire (expensive) synchronisation had already completed. ``tools/
verify_sync.py`` is now a thin CLI wrapper around this module.

Two things the reporting gets right that are easy to get wrong:

* **Coverage is not the same as accuracy.** A rejected window is a window the
  measurement could not evaluate, not one that measured zero. Rejections are
  counted, so "12 of 40 windows were usable" cannot be presented as "12/12
  confident".
* **A median over a handful of clustered windows is not a verdict.** ``passed``
  additionally requires enough independent measurements, spread over enough of
  the clip, to mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studio.stages.sync_acoustic import load_mono16k_track

__all__ = [
    "DEFAULT_GCC_EPS",
    "DEFAULT_GRID_S",
    "DEFAULT_MAX_LAG_S",
    "DEFAULT_MIN_SHARPNESS",
    "DEFAULT_WINDOW_S",
    "SHORT_CLIP_S",
    "SHORT_GRID_WINDOWS",
    "LagSample",
    "ShiftedInterval",
    "VerifyReport",
    "WholeClip",
    "measure",
    "short_clip_grid",
    "validate_parameters",
]

DEFAULT_GRID_S = 5.0
DEFAULT_WINDOW_S = 4.0
DEFAULT_MAX_LAG_S = 1.0
DEFAULT_MIN_SHARPNESS = 50.0
DEFAULT_GCC_EPS = 1e-8

# A pass needs more than a small median: at least this many usable windows,
# spread over at least this fraction of the measured span. Two well-correlated
# windows three seconds apart in a ten-minute clip say nothing about the
# other 597 seconds.
DEFAULT_MIN_CONFIDENT = 3
DEFAULT_MIN_SPAN_RATIO = 0.5

# A run of at least this many consecutive bad windows (|lag| above the
# threshold) is a shifted interval; shorter runs are outliers. A clip also
# fails when more than this fraction of its confident windows is bad.
DEFAULT_MIN_BAD_WINDOWS = 3
DEFAULT_MAX_BAD_FRACTION = 0.1

# Below this a clip gets a denser grid (the sync check also a shorter window),
# so that a 15 s clip is measured by more than two windows; the drift check
# takes its text slope for noise (k = 1).
SHORT_CLIP_S = 120.0
SHORT_GRID_WINDOWS = 20
# Digital silence this long in the voice (the recorder wrote zeros, or the
# clip lies outside it) has nothing to measure and does not count against
# coverage (sony_run mvi-0844: 45 of its 68 s fall on the recorder's zeros).
SILENT_MIN_S = 1.0
# A short clip is also measured the way the camera clock finds a false start:
# its whole sound against the voice within the lag range, and each window's
# peak against the best other one more than 50 ms away (sync_camera_clock).
# A window whose peak does not stand MIN_PEAK_RATIO above the rest is not a
# measurement: sony_run mvi-0848's first 4 s peak at +256 ms only 1.8 times
# above the true lag, while the whole clip peaks at -0.3 ms 7.8 times above
# everything else.
MIN_PEAK_RATIO = 2.5


@dataclass
class LagSample:
    """One window. ``lag_ms`` is the residual: how much later (+) or earlier (−)
    the recorder must be read at ``t`` for the voice to land under the picture."""

    t: float
    lag_ms: float
    sharpness: float
    # The peak over the best one more than 50 ms away (short clips only).
    ratio: float | None = None


@dataclass
class ShiftedInterval:
    start_s: float
    end_s: float
    median_lag_ms: float
    windows: int


@dataclass(frozen=True, slots=True)
class WholeClip:
    """The whole clip against the voice: residual of the best peak and how far it
    stands above the best other one."""

    lag_ms: float
    ratio: float

    @property
    def decisive(self) -> bool:
        return self.ratio >= MIN_PEAK_RATIO


@dataclass
class VerifyReport:
    video: str
    voice: str
    # Every window that was measured, confident or not.
    samples: list[LagSample] = field(default_factory=list)
    # Windows attempted but rejected, with the reason, so coverage is honest.
    rejected: list[tuple[float, str]] = field(default_factory=list)
    duration_s: float = 0.0
    min_sharpness: float = DEFAULT_MIN_SHARPNESS
    grid_s: float = DEFAULT_GRID_S
    window_s: float = DEFAULT_WINDOW_S
    # Seconds of the clip where the voice is digital silence.
    silent_s: float = 0.0
    # Short clips: the whole clip correlated against the voice.
    whole: WholeClip | None = None

    @property
    def short(self) -> bool:
        return self.duration_s < SHORT_CLIP_S

    @property
    def confident(self) -> list[LagSample]:
        return [s for s in self.samples if s.sharpness >= self.min_sharpness]

    @property
    def attempted(self) -> int:
        """Windows the measurement TRIED, including the ones it could not use.

        The old report dropped unconfident windows before counting, then
        reported ``n_confident/n_total`` from what survived — so it always read
        N/N. Three windows with sharpness [100, 1, 1] reported "1/1 confident"
        instead of 1/3, turning a clip that could barely be measured into a
        clip that measured perfectly.
        """
        return len(self.samples) + len(self.rejected)

    @property
    def coverage_span_s(self) -> float:
        """Time between the first and last CONFIDENT window."""
        times = [s.t for s in self.confident]
        return (max(times) - min(times)) if len(times) >= 2 else 0.0

    @property
    def coverage_ratio(self) -> float:
        """Share of the measurable clip (not digital silence) the confident windows span."""
        measurable = self.duration_s - self.silent_s
        return min(1.0, self.coverage_span_s / measurable) if measurable > 0 else 0.0

    def summary(
        self, threshold_ms: float | None = None, min_bad_windows: int = DEFAULT_MIN_BAD_WINDOWS,
    ) -> dict[str, Any]:
        confident = self.confident
        base: dict[str, Any] = {
            "n_confident": len(confident),
            "n_attempted": self.attempted,
            "n_rejected": len(self.rejected),
            "duration_s": round(self.duration_s, 3),
            "coverage_span_s": round(self.coverage_span_s, 3),
            "coverage_ratio": round(self.coverage_ratio, 3),
            **({"silent_s": round(self.silent_s, 3)} if self.silent_s else {}),
            **({"whole_clip": {"lag_ms": round(self.whole.lag_ms, 1),
                               "ratio": round(self.whole.ratio, 2)}} if self.whole else {}),
        }
        if not confident:
            return base
        lags = sorted(float(abs(s.lag_ms)) for s in confident)
        n = len(lags)
        base.update(
            {
                "median_abs_lag_ms": lags[n // 2],
                "p90_abs_lag_ms": lags[min(n - 1, int(n * 0.9))],
                "max_abs_lag_ms": lags[-1],
            }
        )
        if threshold_ms is not None:
            intervals, outliers, fraction = self.bad_windows(threshold_ms, min_bad_windows)
            base.update(
                {
                    "threshold_ms": threshold_ms,
                    "bad_fraction": round(fraction, 3),
                    "shifted_intervals": [
                        {
                            "start_s": round(item.start_s, 3),
                            "end_s": round(item.end_s, 3),
                            "median_lag_ms": round(item.median_lag_ms, 1),
                            "windows": item.windows,
                        }
                        for item in intervals
                    ],
                    "outliers": [
                        {"t": round(item.t, 3), "lag_ms": round(item.lag_ms, 1)}
                        for item in outliers
                    ],
                }
            )
        return base

    def bad_windows(
        self, threshold_ms: float, min_bad_windows: int = DEFAULT_MIN_BAD_WINDOWS,
    ) -> tuple[list[ShiftedInterval], list[LagSample], float]:
        """Shifted intervals, isolated bad windows, and the bad-window fraction.

        A window rejected for a weak peak does not break a run of bad windows
        as long as the neighbouring confident windows are closer than three
        grid steps: the measurement could not see there, which is no evidence
        that the shift stopped.

        In a short clip the windows overlap (a 4 s window every 0.6 s), so
        neighbours repeat one measurement: windows count by the non-overlapping
        stretches of one window length they fall in, both for the bad fraction
        and for the length of a run.
        """
        confident = sorted(self.confident, key=lambda item: item.t)
        if not confident:
            return [], [], 0.0
        bad = [abs(item.lag_ms) > threshold_ms for item in confident]
        max_gap = 3.0 * self.grid_s
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, flag in enumerate(bad):
            broken = start is not None and confident[index].t - confident[index - 1].t > max_gap
            if start is not None and (not flag or broken):
                runs.append((start, index))
                start = None
            if flag and start is None:
                start = index
        if start is not None:
            runs.append((start, len(confident)))
        stretch = self.window_s if self.short and self.window_s > self.grid_s else 0.0

        def independent(items: list[LagSample]) -> int:
            return len({int(item.t // stretch) for item in items}) if stretch else len(items)

        intervals: list[ShiftedInterval] = []
        outliers: list[LagSample] = []
        for first, end in runs:
            members = confident[first:end]
            if independent(members) < min_bad_windows:
                outliers.extend(members)
                continue
            before = confident[first - 1] if first > 0 else None
            after = confident[end] if end < len(confident) else None
            left = (before.t + members[0].t) / 2 if before is not None and (
                members[0].t - before.t <= max_gap) else (
                0.0 if before is None else members[0].t - self.grid_s / 2)
            right = (members[-1].t + after.t) / 2 if after is not None and (
                after.t - members[-1].t <= max_gap) else (
                self.duration_s if after is None else members[-1].t + self.grid_s / 2)
            lags = sorted(item.lag_ms for item in members)
            intervals.append(ShiftedInterval(left, right, lags[len(lags) // 2], len(members)))
        flagged = [item for item, flag in zip(confident, bad, strict=True) if flag]
        return intervals, outliers, independent(flagged) / independent(confident)

    def verdict(
        self,
        threshold_ms: float,
        min_confident: int = DEFAULT_MIN_CONFIDENT,
        min_span_ratio: float = DEFAULT_MIN_SPAN_RATIO,
        min_bad_windows: int = DEFAULT_MIN_BAD_WINDOWS,
        max_bad_fraction: float = DEFAULT_MAX_BAD_FRACTION,
    ) -> tuple[str, str]:
        """``(status, reason)`` where status is passed/failed/inconclusive.

        ``inconclusive`` exists because "we could not measure this" and "this
        measured well" are different answers, and only one of them justifies
        shipping the render. A median alone is not a verdict either: a third
        of a clip shifted by 88 ms leaves the median at zero, so the clip
        fails on any sustained run of bad windows or on too many bad windows
        overall, and the reason names the shifted intervals.

        A short clip has few windows; when its whole sound peaks decisively
        within the threshold, that stands for the windows it lacks, and a
        decisive peak beyond the threshold fails it.
        """
        summary = self.summary()
        confident = int(summary["n_confident"])
        silent = (f"; the recorder is silent for {self.silent_s:.0f} of its "
                  f"{self.duration_s:.0f} s" if self.silent_s else "")
        whole = self.whole if self.whole is not None and self.whole.decisive else None
        whole_note = (f"the whole clip peaks at {whole.lag_ms:+.1f} ms, "
                      f"{whole.ratio:.1f} times above any other lag" if whole else "")
        if whole is not None and abs(whole.lag_ms) > threshold_ms:
            return "failed", f"{whole_note} (beyond {threshold_ms:.1f} ms)"
        if confident == 0 and whole is not None:
            return "passed", whole_note
        if confident < min_confident and whole is None:
            return (
                "inconclusive",
                f"only {confident} usable window(s) out of {self.attempted} attempted "
                f"(need {min_confident}) — the two tracks correlate too weakly to measure"
                + silent,
            )
        ratio = float(summary["coverage_ratio"])
        if ratio < min_span_ratio and whole is None:
            where = "of the clip outside the recorder's silence" if self.silent_s else "of the clip"
            return (
                "inconclusive",
                f"usable windows span only {ratio:.0%} {where} "
                f"(need {min_span_ratio:.0%}) — the rest was never measured",
            )
        intervals, outliers, fraction = self.bad_windows(threshold_ms, min_bad_windows)
        median = float(summary["median_abs_lag_ms"])
        if intervals:
            described = "; ".join(
                f"{item.start_s:.1f}–{item.end_s:.1f} s: {item.median_lag_ms:+.0f} ms, "
                f"{item.windows} windows"
                for item in intervals
            )
            return "failed", f"shifted beyond {threshold_ms:.1f} ms: {described}"
        if fraction > max_bad_fraction:
            return (
                "failed",
                f"{fraction:.0%} of windows exceed {threshold_ms:.1f} ms "
                f"(allowed {max_bad_fraction:.0%})",
            )
        if median > threshold_ms:
            return "failed", f"median |lag| {median:.1f} ms exceeds {threshold_ms:.1f} ms"
        note = f", {len(outliers)} outlier window(s)" if outliers else ""
        note += f"; {whole_note}" if whole else ""
        return "passed", f"median |lag| {median:.1f} ms over {confident} window(s){note}"


def validate_parameters(
    grid_s: float,
    window_s: float,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    median_threshold_ms: float = 20.0,
) -> None:
    """Reject parameters that cannot produce a measurement.

    ``--grid-s 0`` left the sampling cursor where it was and a negative grid
    walked it backwards, so the loop's exit condition was never reached: the
    tool sampled the same instant forever, growing the sample list without
    bound. Every value here is checked for finiteness and range at the entry
    point, so an impossible request is an immediate error rather than a hang.
    """
    import math

    checks = (
        ("grid_s", grid_s, True),
        ("window_s", window_s, True),
        ("max_lag_s", max_lag_s, True),
        ("min_sharpness", min_sharpness, False),
        ("median_threshold_ms", median_threshold_ms, False),
    )
    for name, value, strictly_positive in checks:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        if strictly_positive and value <= 0:
            raise ValueError(f"{name} must be greater than 0, got {value!r}")
        if not strictly_positive and value < 0:
            raise ValueError(f"{name} must not be negative, got {value!r}")


def short_clip_grid(duration_s: float, grid_s: float, window_s: float) -> tuple[float, float]:
    """``(grid_s, window_s)`` for a clip: denser and shorter below ``SHORT_CLIP_S``."""
    if duration_s >= SHORT_CLIP_S:
        return grid_s, window_s
    return (min(grid_s, max(0.25, duration_s / SHORT_GRID_WINDOWS)),
            min(window_s, max(1.0, duration_s / 3.0)))


def measure(
    video_audio_path: Path,
    voice_path: Path,
    grid_s: float = DEFAULT_GRID_S,
    window_s: float = DEFAULT_WINDOW_S,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    source_start_s: float = 0.0,
    source_duration_s: float | None = None,
    voice_start_s: float = 0.0,
) -> VerifyReport:
    """Cross-correlate a source's audio against a rendered voice on a time grid.

    ``source_start_s``/``source_duration_s`` select the stretch of the SOURCE
    that ``voice_path`` corresponds to, and ``voice_start_s`` the offset within
    the voice file. They matter for segmented voice output: comparing the start
    of a second voice segment against the start of the video measures two
    unrelated minutes of audio and reports the disagreement as lag. Every
    measured window — confident or not — is recorded, so coverage can be
    reported honestly.
    """
    validate_parameters(grid_s, window_s, max_lag_s, min_sharpness)
    from studio.stages.sync_grid import measure_grid

    video_track = load_mono16k_track(
        video_audio_path, start_s=source_start_s or None, duration_s=source_duration_s
    )
    voice_track = load_mono16k_track(voice_path, start_s=voice_start_s or None)
    report = measure_grid(
        video_track, voice_track, grid_s=grid_s, window_s=window_s, max_lag_s=max_lag_s,
        min_sharpness=min_sharpness, video=str(video_audio_path), voice=str(voice_path),
    )
    from studio.core.audio_spans import digital_silence

    end = int(report.duration_s * 16000)
    spans = digital_silence(voice_track[:end], int(SILENT_MIN_S * 16000))
    report.silent_s = float((spans[:, 1] - spans[:, 0]).sum()) / 16000 if len(spans) else 0.0
    if report.short:
        _measure_peaks(report, video_track[:end], voice_track, max_lag_s)
    return report


def _measure_peaks(report: VerifyReport, camera: Any, voice: Any, max_lag_s: float) -> None:
    """The whole clip's peak, and each window's: a window without a decisive one
    is rejected as ambiguous."""
    import numpy as np

    from studio.stages.sync_camera_clock import _peak

    rate = 16000
    lag = int(max_lag_s * rate)
    padded = np.concatenate([np.zeros(lag, np.float32), voice[:len(camera)],
                             np.zeros(lag, np.float32)])

    def peak(start: int, end: int) -> tuple[float, float] | None:
        query = camera[start:end]
        if not np.any(query):
            return None
        found = _peak(query, padded[start:end + 2 * lag])
        return None if found is None else ((found[0] - lag) * 1000.0 / rate, found[2])

    whole = peak(0, len(camera))
    if whole is not None:
        report.whole = WholeClip(*whole)
    kept = []
    half = int(report.window_s * rate / 2)
    for sample in report.samples:
        centre = int(sample.t * rate)
        found = peak(max(0, centre - half), centre + half)
        sample.ratio = None if found is None else found[1]
        if sample.sharpness >= report.min_sharpness and (
                sample.ratio is None or sample.ratio < MIN_PEAK_RATIO):
            report.rejected.append((sample.t, f"ambiguous peak (ratio {sample.ratio or 0:.1f})"))
        else:
            kept.append(sample)
    report.samples = kept
