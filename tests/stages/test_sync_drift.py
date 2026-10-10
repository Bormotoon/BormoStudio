from pathlib import Path

import numpy as np
import pytest

from studio.stages.sync_drift import anchor_regime, check_alignment
from studio.stages.sync_models import AlignmentMap, Anchor
from studio.stages.sync_refine import RefineSettings

SR = 16000


def _speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = np.zeros(int(seconds * SR), np.float32)
    t = 0.3
    while t < seconds - 1:
        length = rng.uniform(0.2, 0.6)
        signal[int(t * SR):int((t + length) * SR)] = rng.standard_normal(
            int((t + length) * SR) - int(t * SR)) * 0.3
        t += length + rng.uniform(0.1, 0.4)
    return signal


def _text_map(k: float, offset: float, noise_ms: float, seed: int = 1) -> AlignmentMap:
    rng = np.random.default_rng(seed)
    anchors = [Anchor(cam, (cam - offset) / k + rng.normal(0, noise_ms / 1000), f"w{i}", 1.0)
               for i, cam in enumerate(np.arange(5, 285, 7.0))]
    return AlignmentMap(anchors, offset, k, noise_ms, "text", len(anchors), 280.0)


def _check(camera: np.ndarray, recorder: np.ndarray, alignment: AlignmentMap):
    # Each microphone has its own noise floor across the whole band.
    rng = np.random.default_rng(9)
    camera = camera + rng.standard_normal(camera.size).astype(np.float32) * 0.003
    recorder = recorder + rng.standard_normal(recorder.size).astype(np.float32) * 0.003
    return check_alignment(
        alignment, Path("camera.mov"), Path("recorder.wav"),
        clip_duration_s=len(camera) / SR, recorder_duration_s=len(recorder) / SR,
        settings=RefineSettings(), camera_track=camera, recorder_track=recorder,
    )


def test_text_drift_without_acoustic_support_is_rejected() -> None:
    recorder = _speech(400)
    camera = recorder[int(50 * SR):int(342 * SR)]  # true map: cam = rec - 50, k = 1
    checked = _check(camera, recorder, _text_map(1.0000708, -50.0, 80))
    assert checked.report["used"] == "acoustic"
    assert checked.alignment.k == pytest.approx(1.0, abs=5e-6)
    assert checked.alignment.offset == pytest.approx(-50.0, abs=0.002)
    assert checked.report["p90_ms"] <= 2.0
    assert checked.alignment.residual_ms == pytest.approx(checked.report["p90_ms"], abs=0.01)
    assert checked.constant_only
    assert "complex rejected" in checked.report["complex"]
    assert any(anchor_regime(anchor) == 0 for anchor in checked.alignment.anchors)


def test_real_drift_is_confirmed_acoustically() -> None:
    k = 1.00075
    original = _speech(400, seed=2)
    # The camera clock runs k times the recorder: cam = k * (rec - 50). Built
    # from 2 s chunks read at the drifted position (no interpolation blur).
    chunk = 2 * SR
    camera = np.concatenate([
        original[start:start + chunk]
        for start in (int(round((50 + index * 2 / k) * SR)) for index in range(146))
    ])
    checked = _check(camera, original, _text_map(1.0, -50.0, 80))
    assert checked.report["drift"] == "confirmed"
    assert checked.alignment.k == pytest.approx(k, abs=2e-5)
    assert not checked.constant_only


def test_too_few_windows_keep_the_text_map() -> None:
    recorder = _speech(400)
    silent_camera = np.zeros(int(292 * SR), np.float32)
    alignment = _text_map(1.0, -50.0, 80)
    checked = _check(silent_camera, recorder, alignment)
    assert checked.report["used"] == "text"
    assert checked.alignment is alignment


def test_smoothing_does_not_mix_regimes_across_a_step() -> None:
    from studio.stages.sync_pieces import _smoothed_map_at

    anchors = [Anchor(rec - 50 + (0.0 if rec < 110 else 0.088), rec,
                      f"acoustic:{0 if rec < 110 else 1}", 1.0)
               for rec in np.arange(80, 140, 2.5)]
    alignment = AlignmentMap(anchors, -50.0, 1.0, 1.0, "acoustic", len(anchors), 60.0)
    before, _rate = _smoothed_map_at(alignment, 105.0)
    after, _rate = _smoothed_map_at(alignment, 115.0)
    assert before == pytest.approx(55.0, abs=1e-6)
    assert after == pytest.approx(65.088, abs=1e-6)
