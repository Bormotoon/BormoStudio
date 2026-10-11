import numpy as np
import pytest

from studio.stages.sync_grid import fit_model, measure_grid, piece_mapping
from studio.stages.sync_verify import LagSample, VerifyReport

SR = 16000


def _report(lags: list[tuple[float, float]], duration: float = 290.0) -> VerifyReport:
    report = VerifyReport("cam", "voice", duration_s=duration, grid_s=2.5)
    report.samples = [LagSample(t, lag, 200.0) for t, lag in lags]
    return report


def _grid(duration: float = 290.0) -> list[float]:
    return [2.0 + 2.5 * index for index in range(int((duration - 4) / 2.5) + 1)]


def test_partially_shifted_clip_fails_with_the_shifted_interval() -> None:
    rng = np.random.default_rng(1)
    report = _report([(t, (-88.0 if t < 61 else 0.0) + rng.normal(0, 1)) for t in _grid()])
    status, reason = report.verdict(20.0)
    assert status == "failed"
    interval = report.summary(20.0)["shifted_intervals"][0]
    assert interval["start_s"] == pytest.approx(0.0)
    assert 60.0 <= interval["end_s"] <= 63.0
    assert interval["median_lag_ms"] == pytest.approx(-88, abs=3)
    assert "0.0–" in reason


def test_single_false_window_is_an_outlier() -> None:
    report = _report([(t, 120.0 if t == 102.0 else 0.5) for t in _grid()])
    status, reason = report.verdict(20.0)
    assert status == "passed"
    assert len(report.summary(20.0)["outliers"]) == 1
    assert "1 outlier" in reason


def test_scattered_bad_windows_fail_by_fraction() -> None:
    lags = [(t, 60.0 if index % 7 == 0 else 0.0) for index, t in enumerate(_grid())]
    status, reason = _report(lags).verdict(20.0)
    assert status == "failed"
    assert "of windows exceed" in reason


def test_rejected_windows_do_not_break_a_shifted_run() -> None:
    lags = [(t, -50.0) for t in (2.0, 4.5, 9.5, 12.0)] + [(t, 0.0) for t in _grid() if t > 14]
    report = _report(lags)
    assert report.verdict(20.0)[0] == "failed"
    assert report.summary(20.0)["shifted_intervals"][0]["windows"] == 4


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Speech-like: noise bursts so the correlation peak is sharp.
    envelope = (np.sin(np.linspace(0, seconds * 9, int(seconds * SR))) > 0).astype(np.float32)
    return (rng.standard_normal(int(seconds * SR)).astype(np.float32) * (0.2 + envelope))


def test_measure_grid_sign_is_read_the_recorder_later() -> None:
    camera = _noise(40)
    delay = int(0.05 * SR)
    voice = np.concatenate([np.zeros(delay, np.float32), camera[:-delay]])
    report = measure_grid(camera, voice)
    assert report.confident
    assert np.median([item.lag_ms for item in report.confident]) == pytest.approx(50, abs=1)


def test_measure_grid_follows_a_piece_mapping() -> None:
    recorder = _noise(120, seed=3)
    camera = recorder[int(30 * SR):int(70 * SR)]  # clip starts at recorder 30 s
    report = measure_grid(camera, recorder[int(20 * SR):], piece_mapping(0.0, [(30.0, 40.0, 1.0)]),
                          other_origin_s=20.0)
    lags = [item.lag_ms for item in report.confident]
    assert lags and max(abs(value) for value in lags) < 1.0


def test_fit_model_finds_one_step() -> None:
    rng = np.random.default_rng(2)
    report = _report([(t, (-88.0 if t < 61 else 0.0) + rng.normal(0, 1)) for t in _grid()])
    model = fit_model(report)
    assert model is not None and model.kind == "regimes"
    assert len(model.steps) == 1
    assert model.steps[0].delta_ms == pytest.approx(88, abs=3)
    assert 59 <= model.steps[0].t_s <= 63
    assert model.p90_after_ms <= 3


def test_fit_model_keeps_a_constant_for_noise() -> None:
    rng = np.random.default_rng(4)
    model = fit_model(_report([(t, 3.0 + rng.normal(0, 1.5)) for t in _grid()]))
    assert model is not None and model.kind == "constant"
    assert model.regimes[0].offset_ms == pytest.approx(3.0, abs=1)


def test_fit_model_prefers_a_line_for_real_drift() -> None:
    model = fit_model(_report([(t, 0.075 * t) for t in _grid()]))
    assert model is not None and model.kind == "line"
    assert model.regimes[0].slope == pytest.approx(0.075, rel=0.05)


def test_fit_model_does_not_cut_a_slope_into_steps() -> None:
    # sony_run dji-0926: read along a text line with k = 1.004, the clip drifts
    # 4 ms/s between two real recorder losses (-300 ms at 100 s, -500 ms at 200 s);
    # the model used to be a staircase with a "step" of 0 ms every 10 s.
    rng = np.random.default_rng(5)
    lags = [(t, 4.0 * t - (300.0 if t > 100 else 0.0) - (500.0 if t > 200 else 0.0)
             + rng.normal(0, 0.3)) for t in _grid()]
    model = fit_model(_report(lags))
    assert model is not None and model.kind == "regimes"
    assert [round(step.delta_ms / 100) for step in model.steps] == [-3, -5]
    assert all(regime.slope == pytest.approx(4.0, abs=0.1) for regime in model.regimes)
    assert model.p90_after_ms <= 2


def test_the_recorders_silence_does_not_count_against_coverage() -> None:
    # sony_run mvi-0844: 68 s, of which the first 45 s fall on the recorder's
    # digital zeros; the rest measured 1-3 ms.
    report = _report([(t, 1.5) for t in np.arange(47.0, 66.5, 2.5)], duration=68.0)
    status, reason = report.verdict(20.0)
    assert status == "inconclusive" and "26% of the clip" in reason
    report.silent_s = 45.0
    assert report.verdict(20.0)[0] == "passed"
    assert report.summary()["silent_s"] == 45.0


def test_short_clips_get_a_denser_grid() -> None:
    from studio.stages.sync_verify import short_clip_grid

    assert short_clip_grid(600.0, 2.5, 4.0) == (2.5, 4.0)
    grid_s, window_s = short_clip_grid(14.7, 2.5, 4.0)  # sony_run dji-0935: 2 windows of 5
    assert grid_s == pytest.approx(0.735) and window_s == 4.0
    assert short_clip_grid(5.5, 2.5, 4.0) == pytest.approx((0.275, 5.5 / 3))


def test_measure_counts_the_voices_digital_silence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import wave

    from studio.stages.sync_verify import measure

    def write(path, data):  # type: ignore[no-untyped-def]
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SR)
            handle.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
        return path

    camera = _noise(30.0, seed=6) * 0.3
    voice = camera.copy()
    voice[:int(12 * SR)] = 0.0
    report = measure(write(tmp_path / "cam.wav", camera), write(tmp_path / "voice.wav", voice),
                     grid_s=2.5, window_s=4.0)
    assert report.silent_s == pytest.approx(12.0, abs=0.01)
    assert report.verdict(20.0)[0] == "passed"
