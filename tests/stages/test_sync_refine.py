from pathlib import Path

import numpy as np
import pytest

from studio.stages import sync_refine
from studio.stages.sync_grid import measure_grid, piece_mapping
from studio.stages.sync_pieces import validate_pieces
from studio.stages.sync_refine import RefineSettings, refine_plan

SR = 16000
DROP_AT = 60.0  # original time where the recorder loses 88 ms
DROP = int(0.088 * SR)


def _speech(seconds: float, seed: int = 0) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Noise 'words' 0.2-0.6 s long separated by 0.1-0.25 s, with longer pauses."""
    rng = np.random.default_rng(seed)
    signal = np.zeros(int(seconds * SR), np.float32)
    words = []
    t = 0.3
    while t < seconds - 1:
        length = rng.uniform(0.2, 0.6)
        start, end = int(t * SR), int((t + length) * SR)
        signal[start:end] = rng.standard_normal(end - start) * 0.3
        words.append((t, t + length))
        t += length + (rng.uniform(0.8, 1.2) if rng.random() < 0.15 else rng.uniform(0.1, 0.25))
    signal += rng.standard_normal(signal.size).astype(np.float32) * 0.002
    return signal, words


def _material(click: bool = True, drop_in_word: bool = False):
    original, words = _speech(130.0)
    # A dropout in a pause between phrases, as on the real recorder.
    drop_at = next((a1 + b0) / 2 for (_a0, a1), (b0, _b1) in zip(words, words[1:], strict=False)
                   if a1 > DROP_AT - 15 and b0 - a1 > 0.6)
    if drop_in_word:
        drop_at = next((a + b) / 2 for a, b in words if a > DROP_AT)
    cut = int(drop_at * SR)
    recorder = np.concatenate([original[:cut], original[cut + DROP:]]).copy()
    if click:
        recorder[cut:cut + 16] += np.array([0.9, -0.9] * 8, np.float32)
    shift = DROP / SR
    rec_words = [(a if a < drop_at else a - shift, b if b < drop_at else b - shift)
                 for a, b in words]
    if drop_in_word:
        rec_words = [(a, b) for a, b in rec_words if not a < drop_at < b + 2]
        rec_words = []  # no transcript pauses near: speech absorption
    camera = original[int(10 * SR):int(110 * SR)]
    return camera, recorder, rec_words, drop_at


def _refine(camera, recorder, words, monkeypatch=None):
    return refine_plan(
        0.0, [(10.0, 99.9, 1.0)], Path("camera.mov"), Path("recorder.wav"),
        clip_duration_s=100.0, recorder_duration_s=len(recorder) / SR, recorder_words=words,
        settings=RefineSettings(), label="DJI_0909", camera_track=camera,
        recorder_track=recorder, recorder_origin_s=0.0,
    )


def test_recorder_dropout_with_click_is_located_and_absorbed() -> None:
    camera, recorder, words, drop_at = _material()
    result = _refine(camera, recorder, words)
    assert result.accepted, result.report
    step = result.report["steps"][0]
    assert step["located_by"] == "click"
    assert step["recorder_s"] == pytest.approx(drop_at, abs=0.25)
    assert step["delta_ms"] == pytest.approx(-88, abs=3)
    assert "sync jumps by -88 ms" in result.warnings[0]
    assert not validate_pieces(result.lead, result.pieces, 100.0, len(recorder) / SR)
    for (s0, d0, _f0), (s1, _d1, _f1) in zip(result.pieces, result.pieces[1:], strict=False):
        assert s0 + d0 == pytest.approx(s1, abs=1e-9)  # continuous source
    check = measure_grid(camera, recorder, piece_mapping(result.lead, result.pieces))
    window = (step["absorb"][0] - 2.5, step["absorb"][1] + 2.5)
    outside = [abs(item.lag_ms) for item in check.confident
               if not window[0] <= item.t <= window[1]]
    assert outside and max(outside) <= 5.0


def test_step_in_speech_gets_a_wide_window_and_a_warning() -> None:
    camera, recorder, words, _drop_at = _material(click=False, drop_in_word=True)
    result = _refine(camera, recorder, words)
    assert result.accepted, result.report
    step = result.report["steps"][0]
    assert step["absorbed_in"] == "speech"
    assert step["absorb"][1] - step["absorb"][0] == pytest.approx(0.088 / 0.03, rel=0.05)
    assert "absorbed in speech" in result.warnings[0]


def test_camera_discontinuity_is_not_corrected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync_refine, "camera_pts_gap", lambda *_args: True)
    camera, recorder, words, _drop_at = _material()
    result = _refine(camera, recorder, words)
    assert not result.accepted
    assert result.pieces == [(10.0, 99.9, 1.0)]
    assert "discontinuous" in result.warnings[0]


def test_clip_without_step_keeps_its_plan() -> None:
    original, words = _speech(130.0, seed=5)
    camera = original[int(10 * SR):int(110 * SR)]
    result = _refine(camera, original, words)
    assert not result.accepted
    assert result.pieces == [(10.0, 99.9, 1.0)]
    assert "within" in result.report["reason"]


def test_absorption_window_stays_next_to_the_step() -> None:
    from studio.stages.sync_refine import absorption_window

    def inverse(rec: float) -> float:
        return rec

    pieces = [(0.0, 100.0, 1.0)]
    # A long pause that holds the step: the window sits at the step, not mid-pause.
    left, right, in_pause = absorption_window(
        32.0, -0.088, [(10.0, 20.0), (33.0, 34.0)], inverse, 0.0, pieces, 0.4)
    assert in_pause and left <= 32.0 <= right and right - left == pytest.approx(0.44)
    # The nearest pause is 5 s away: absorb in speech at the step instead.
    left, right, in_pause = absorption_window(
        32.0, -0.088, [(20.0, 26.0), (27.0, 31.5), (31.6, 40.0)], inverse, 0.0, pieces, 0.4)
    assert not in_pause and left < 32.0 < right
