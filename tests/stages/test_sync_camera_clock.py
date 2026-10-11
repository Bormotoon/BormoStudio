"""A camera's clock: which moment recorded_at stamps, and false starts found by it."""

import wave
from pathlib import Path

import numpy as np
import pytest

from studio.core.project import Asset, AssetKind, AssetRole
from studio.stages.sync_camera_clock import camera_clocks, clock_search
from studio.stages.sync_models import AlignmentMap
from studio.stages.sync_rescue import _clock_placement

SR = 16000


def _speech(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(int(seconds * SR)).astype(np.float32) * 0.003
    t = 0.3
    while t < seconds - 1:
        length = rng.uniform(0.2, 0.6)
        start, end = int(t * SR), int((t + length) * SR)
        signal[start:end] += rng.standard_normal(end - start).astype(np.float32) * 0.2
        t += length + rng.uniform(0.1, 0.5)
    return signal


def _wav(path: Path, data: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _canon(name: str, duration: float, stamp: float) -> Asset:
    return Asset(name, Path(f"{name}.wav"), AssetKind.VIDEO, AssetRole.CAMERA, group_id="mvi",
                 manual={"media_info": {"duration_s": duration, "audio_channels": 1,
                                        "recorded_at": {"epoch": stamp}}})


# Canon on sony_run: recorded_at is the end of each clip; the clock runs 9813 s
# ahead of the recorder. Clips of different lengths tell start from end.
DELTA = 9813.0


def _clip(name: str, start: float, duration: float) -> Asset:
    return _canon(name, duration, start + DELTA + duration)


def test_the_stamp_is_the_reading_that_keeps_the_delta_steady() -> None:
    cameras = [_clip("mvi-0845", 415.0, 486.1), _clip("mvi-0846", 974.3, 726.8),
               _clip("mvi-0850", 3074.8, 728.7), _clip("mvi-0854", 6487.7, 280.3)]
    clock = camera_clocks(cameras, {camera.id: start for camera, start in zip(
        cameras, (415.0, 974.3, 3074.8, 6487.7), strict=True)})["mvi"]
    assert clock.stamps == ("end",) and clock.spread_s < 0.01
    # A false start between two placed clips, predicted from its end stamp.
    (start, radius), = clock.predict(_clip("mvi-0849", 3061.9, 5.5))
    assert start == pytest.approx(3061.9) and radius == pytest.approx(3.0, abs=0.01)


def test_equal_lengths_leave_both_readings_and_copy_times_none() -> None:
    same = [_canon("a", 600.0, 1000.0), _canon("b", 600.0, 1700.0)]
    assert camera_clocks(same, {"a": 0.0, "b": 700.0})["mvi"].stamps == ("start", "end")
    copied = [_canon("a", 600.0, 90000.0), _canon("b", 300.0, 90040.0),
              _canon("c", 500.0, 90080.0)]
    assert camera_clocks(copied, {"a": 100.0, "b": 900.0, "c": 1700.0}) == {}


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    folder = tmp_path_factory.mktemp("clock")
    recorder = _speech(240.0, 11)
    return _wav(folder / "rec.wav", recorder), recorder


def test_a_false_start_is_found_around_its_predicted_place(recording, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, recorder = recording
    noise = np.random.default_rng(12).standard_normal(int(5.5 * SR)).astype(np.float32) * 0.01
    clip = _wav(tmp_path / "clip.wav", recorder[int(100.3 * SR):int(105.8 * SR)] * 0.3 + noise)
    match = clock_search(clip, 5.5, path, 240.0, [(99.7, 3.0)])
    assert match is not None and match.position_s == pytest.approx(100.3, abs=0.002)
    assert match.ratio >= 2.5
    # Far from its place nothing stands out, and a wrong place is not taken.
    assert clock_search(clip, 5.5, path, 240.0, [(160.0, 3.0)]) is None


def test_digital_silence_is_not_a_match(tmp_path: Path) -> None:
    recorder = _speech(60.0, 13)
    recorder[int(20 * SR):int(40 * SR)] = 0.0
    path = _wav(tmp_path / "rec.wav", recorder)
    clip = _wav(tmp_path / "clip.wav", _speech(4.0, 14))
    assert clock_search(clip, 4.0, path, 60.0, [(30.0, 3.0)]) is None


def test_the_second_pass_places_a_false_start_by_its_clock(recording, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, recorder = recording
    source = path.parent
    clip = recorder[int(100.3 * SR):int(105.8 * SR)]
    _wav(source / "mvi-0849.wav", clip * 0.5)
    cameras = [_clip("mvi-0847", 20.0, 60.0), _clip("mvi-0849", 100.3, 5.5),
               _clip("mvi-0850", 150.0, 80.0)]
    rec = Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER,
                manual={"media_info": {"duration_s": 240.0, "audio_channels": 1}})
    fits = {("mvi-0847", "rec"): AlignmentMap([], -20.0, 1.0, 0.0, "text", 40),
            ("mvi-0850", "rec"): AlignmentMap([], -150.0, 1.0, 0.0, "text", 40)}
    searched: dict[str, dict[str, object]] = {}
    clocks = _clock_placement(lambda *args, **kwargs: None, fits, {}, cameras, [cameras[1]],
                              rec, source, searched)
    assert clocks["mvi"]["stamp"] == "end"
    placed = fits["mvi-0849", "rec"]
    assert placed.provenance == "acoustic" and placed.k == 1.0
    assert -placed.offset == pytest.approx(100.3, abs=0.002)
    assert searched["mvi-0849"]["map"] == "clock" and "clock" in placed.notes[0]
