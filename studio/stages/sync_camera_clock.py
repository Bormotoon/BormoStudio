"""A camera's own clock: where an unplaced clip of it must be in the recorder.

``recorded_at`` (the QuickTime creation date) is the recording time, but
devices stamp different moments of it: on sony_run DJI writes the start of
each file and the Canon the end. Read as starts, the Canon clock disagreed
with the recorder by 9084-9746 s from clip to clip; read as ends, by
9812-9814 s over 3.5 hours. Which moment a camera stamps is decided from its
clips that text or acoustics already placed: the reading that keeps the clock
delta steady wins.

Between two placed clips the delta of a third is interpolated, so a false
start too short or too quiet for words (sony_run: 1-12 s Canon clips, none
of them placed) has a predicted start good to about a second. Its whole sound
is then correlated against a few seconds of the recorder around it; a peak
well above every other one there is the clip's place. The search stays
narrow: a false start is followed by its retake seconds later, and a wider
search found the retake's competing peak (mvi-0848 and 0849 at +-11 s).

Calibration on sony_run (732 searches of +-4 s at wrong places, queries of
1.5-12 s): the peak stood 2.5 times above the next one in 0.14 % of them;
the five false starts found at their predicted places stood 3.0-4.9 times
above it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from studio.core.audio_spans import DIGITAL_SILENCE_S, digital_silence, overlaps
from studio.core.project import Asset
from studio.core.recorded_at import epoch
from studio.stages.locate import SR, _group, decode

# A clock is trusted when its delta to the recorder (after a linear drift)
# stays within this; otherwise recorded_at is a copy time, not a recording time.
CLOCK_AGREE_S = 30.0
# A camera clock drifts against the recorder by at most this rate (sony_run:
# 0.1 %, mostly samples the recorder lost).
MAX_DRIFT = 0.01
# Two readings closer than this cannot be told apart (clips of equal length).
STAMP_DIFFERENCE_S = 1.0
# The search reaches this far around a prediction, plus how much the deltas
# of the clips it lies between differ (half the clock's spread outside them).
SEARCH_S = 3.0
# At most this much of a clip's sound is correlated.
QUERY_MAX_S = 60.0
# The peak must stand this far above the best one more than EXCLUDE_S away.
MIN_RATIO = 2.5
EXCLUDE_S = 0.05
# Longer clips confirm the peak with each half: they must agree within AGREE_S.
HALVES_MIN_S = 6.0
AGREE_S = 0.02


@dataclass(frozen=True, slots=True)
class CameraClock:
    """``(stamp, clock start, recorder start)`` of the placed clips of one camera."""

    group: str
    stamps: tuple[str, ...]  # ("start",), ("end",) or both when they cannot be told apart
    spread_s: float
    # Per stamp: (clock start, recorder start - clock start) of every placed clip.
    deltas: dict[str, tuple[tuple[float, float], ...]]

    def predict(self, camera: Asset) -> list[tuple[float, float]]:
        """``(recorder start, search radius)`` of ``camera`` for every reading of its stamp."""
        info = camera.manual.get("media_info", {})
        moment = epoch(info)
        if moment is None:
            return []
        duration = float(info.get("duration_s") or 0.0)
        result = []
        for stamp in self.stamps:
            start = moment - (duration if stamp == "end" else 0.0)
            delta, uncertainty = _interpolate(self.deltas[stamp], start, self.spread_s / 2)
            result.append((start + delta, SEARCH_S + min(uncertainty, CLOCK_AGREE_S)))
        return result

    def report(self) -> dict[str, object]:
        return {"stamp": "/".join(self.stamps), "spread_s": round(self.spread_s, 2),
                "clips": len(self.deltas[self.stamps[0]])}


def camera_clocks(cameras: Sequence[Asset], starts: dict[str, float]) -> dict[str, CameraClock]:
    """The clock of every camera group with at least two placed clips that agree."""
    groups: dict[str, list[tuple[float, float, float]]] = {}
    for camera in cameras:
        info = camera.manual.get("media_info", {})
        moment = epoch(info)
        if camera.id in starts and moment is not None:
            groups.setdefault(_group(camera), []).append(
                (moment, float(info.get("duration_s") or 0.0), starts[camera.id]))
    clocks = {}
    for group, clips in groups.items():
        if len(clips) < 2:
            continue
        deltas = {
            stamp: tuple(sorted((moment - (duration if stamp == "end" else 0.0),
                                 start - moment + (duration if stamp == "end" else 0.0))
                                for moment, duration, start in clips))
            for stamp in ("start", "end")
        }
        spreads = {stamp: _spread(values) for stamp, values in deltas.items()}
        best = min(spreads.values())
        if best > CLOCK_AGREE_S:
            continue
        stamps = tuple(stamp for stamp in ("start", "end")
                       if spreads[stamp] <= best + STAMP_DIFFERENCE_S)
        clocks[group] = CameraClock(group, stamps, best, deltas)
    return clocks


def _spread(points: Sequence[tuple[float, float]]) -> float:
    """Spread of the delta around a linear drift (around its mean for two clips).

    The drift is at most ``MAX_DRIFT``: copy times put in place of recording
    times can line up as well, at an impossible rate.
    """
    times = np.array([point[0] for point in points])
    values = np.array([point[1] for point in points])
    if len(points) >= 3 and float(np.ptp(times)) > 0:
        slope = float(np.polyfit(times - times[0], values, 1)[0])
        values = values - float(np.clip(slope, -MAX_DRIFT, MAX_DRIFT)) * (times - times[0])
    return float(np.ptp(values))


def _interpolate(points: Sequence[tuple[float, float]], at: float,
                 outside: float) -> tuple[float, float]:
    """``(delta, uncertainty)`` at clock time ``at``: between the nearest clips, else
    the nearest one's with ``outside`` uncertainty."""
    times = [point[0] for point in points]
    index = int(np.searchsorted(times, at))
    if index == 0:
        return points[0][1], outside
    if index == len(points):
        return points[-1][1], outside
    (t0, d0), (t1, d1) = points[index - 1], points[index]
    if t1 == t0:
        return d0, abs(d1 - d0)
    return d0 + (d1 - d0) * (at - t0) / (t1 - t0), abs(d1 - d0)


@dataclass(frozen=True, slots=True)
class ClockMatch:
    position_s: float  # recorder start of the clip
    sharpness: float
    ratio: float  # the peak over the best other one
    predicted_s: float


def clock_search(
    camera: Path, camera_duration_s: float, recorder: Path, recorder_duration_s: float,
    predictions: Sequence[tuple[float, float]], *, pan: str | None = None,
    stream: int | None = None,
) -> ClockMatch | None:
    """The clip's start around one of ``predictions`` (start, radius), or None when no
    peak stands out."""
    query = decode(camera, 0.0, min(camera_duration_s, QUERY_MAX_S))
    if query.size < SR // 2 or not np.any(query):
        return None
    best: ClockMatch | None = None
    for predicted, radius in predictions:
        low = max(0.0, predicted - radius)
        high = min(recorder_duration_s, predicted + radius + query.size / SR)
        if high - low < query.size / SR:
            continue
        reference = decode(recorder, low, high - low, pan=pan, stream=stream)
        found = _peak(query, reference)
        if found is None:
            continue
        lag, sharpness, ratio = found
        # Where the recorder wrote zeros it heard nothing to match.
        silent = digital_silence(reference, int(DIGITAL_SILENCE_S * SR))
        if ratio < MIN_RATIO or overlaps(silent, lag, lag + query.size) or (
                not _halves_agree(query, reference, lag)):
            continue
        match = ClockMatch(low + lag / SR, sharpness, ratio, predicted)
        if best is None or match.ratio > best.ratio:
            best = match
    return best


def _correlation(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """PHAT cross-correlation of ``query`` at every start inside ``reference``."""
    count = reference.size - query.size + 1
    if count < 1:
        return np.zeros(0)
    size = 1 << int(np.ceil(np.log2(reference.size + query.size)))
    spectrum = np.fft.rfft(reference - reference.mean(), size) * np.conj(
        np.fft.rfft(query - query.mean(), size))
    spectrum /= np.abs(spectrum) + 1e-12
    return np.abs(np.fft.irfft(spectrum, size)[:count])


def _peak(query: np.ndarray, reference: np.ndarray) -> tuple[int, float, float] | None:
    """``(start sample, sharpness, peak over the best other peak)``."""
    values = _correlation(query, reference)
    if values.size < 2:
        return None
    index = int(np.argmax(values))
    floor = float(np.median(values)) or 1e-12
    exclude = int(EXCLUDE_S * SR)
    others = np.concatenate([values[:max(0, index - exclude)], values[index + exclude + 1:]])
    runner_up = float(others.max()) if others.size else 0.0
    if values[index] <= 0 or runner_up <= 0:
        return None  # nothing to compare: digital silence, or no other place
    return index, float(values[index]) / floor, float(values[index]) / runner_up


def _halves_agree(query: np.ndarray, reference: np.ndarray, lag: int) -> bool:
    """Each half of a longer clip peaks where the whole does (silent halves say nothing)."""
    if query.size < HALVES_MIN_S * SR:
        return True
    middle = query.size // 2
    for offset, half in ((0, query[:middle]), (middle, query[middle:])):
        if not np.any(half):
            continue
        found = _peak(half, reference)
        if found is not None and found[2] >= MIN_RATIO and abs(
                found[0] - offset - lag) > AGREE_S * SR:
            return False
    return True
