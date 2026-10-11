"""The recorder clock: losses found against cameras, put back, transcripts moved."""

import wave
from pathlib import Path

import numpy as np
import pytest

from studio.core.transcript import Segment, Transcript, Word
from studio.stages.sync_models import AlignmentMap
from studio.stages.sync_recorder_clock import (
    ClipTrack,
    Line,
    Loss,
    RecorderClock,
    build_clock,
    clip_losses,
    refit,
    repair_transcript,
    track_clip,
    write_repaired,
)

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


def _drop(signal: np.ndarray, losses: list[tuple[float, float]]) -> np.ndarray:
    """``signal`` with ``(true time, seconds)`` cut out, like a phone dropping buffers."""
    keep = np.ones(signal.size, dtype=bool)
    for at, length in losses:
        keep[int(at * SR):int((at + length) * SR)] = False
    return signal[keep]


@pytest.fixture(scope="module")
def session() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = _speech(320, 3)
    # The recorder loses 300 ms at real 100 s and 150 ms at real 180 s.
    recorder = _drop(truth, [(100.0, 0.3), (180.0, 0.15)])
    camera = truth[int(50 * SR):int(250 * SR)] + np.random.default_rng(4).standard_normal(
        int(200 * SR)).astype(np.float32) * 0.003
    return truth, recorder, camera


def _track(camera: np.ndarray, recorder: np.ndarray, camera_id: str = "cam",
           start: float = 50.0) -> ClipTrack:
    line = Line(-start)
    track = ClipTrack(camera_id, track_clip(camera, recorder, 0.0, line, camera.size / SR), line,
                      camera.size / SR)
    clip_losses(track, camera, recorder, 0.0)
    return track


def test_losses_are_found_sized_and_located(session) -> None:  # type: ignore[no-untyped-def]
    _truth, recorder, camera = session
    clock = build_clock("rec", [_track(camera, recorder)], [])
    applied = clock.applied
    assert [round(loss.lost_ms) for loss in applied] == [300, 150]
    # File times: the second loss sits 0.3 s earlier in the file than in reality.
    assert [loss.located_by for loss in applied] == ["fine", "fine"]
    assert applied[0].rec_s == pytest.approx(100.0, abs=0.08)
    assert applied[1].rec_s == pytest.approx(179.7, abs=0.08)
    assert clock.to_repaired(150.0) == pytest.approx(150.3)
    assert clock.to_file(clock.to_repaired(200.0)) == pytest.approx(200.0)


def test_a_camera_that_heard_across_without_a_step_contradicts(session) -> None:  # type: ignore[no-untyped-def]
    truth, recorder, camera = session
    # A second camera on a recorder that lost nothing at 100 s (its own glitch
    # is the first camera's): build both against the clean recording.
    clean = _track(truth[int(60 * SR):int(140 * SR)], truth, "other", 60.0)
    glitch = _track(camera, recorder)
    clock = build_clock("rec", [glitch, clean], [])
    first = next(loss for loss in clock.losses if abs(loss.rec_s - 100.0) < 1.0)
    assert first.contradicted_by == ["other"] and not first.applied


def test_repaired_recorder_runs_with_real_time(session, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    truth, recorder, camera = session
    source = tmp_path / "rec.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(recorder, -1, 1) * 32767).astype("<i2").tobytes())
    clock = build_clock("rec", [_track(camera, recorder)], [])
    repaired = write_repaired(source, tmp_path / "out.wav", clock, {}, sample_rate=SR,
                              channels=1, codec="pcm_s16le")
    with wave.open(str(repaired), "rb") as handle:
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2") / 32767.0
    assert data.size == pytest.approx(truth.size, abs=SR * 0.004)
    # After both losses the repaired file is where reality is (within a few ms).
    window = truth[int(250 * SR):int(254 * SR)]
    best = max(range(-160, 161),
               key=lambda lag: float(np.dot(window, data[int(250 * SR) + lag:
                                                          int(254 * SR) + lag])))
    assert abs(best) <= 48  # 3 ms


def test_clips_and_words_follow_the_repaired_clock(session) -> None:  # type: ignore[no-untyped-def]
    _truth, recorder, camera = session
    track = _track(camera, recorder)
    clock = build_clock("rec", [track], [])
    fit = refit(AlignmentMap([], 50.0 * 0.0, 1.0, 0.0, "text"), clock, track.points, 200.0)
    assert fit.provenance == "acoustic"
    assert fit.k == pytest.approx(1.0, abs=2e-5)
    assert fit.offset == pytest.approx(-50.0, abs=0.003)
    words = Transcript(Path("rec.wav"), "ru", 319.55, [
        Segment(10.0, 10.5, (Word("раз", 10.0, 10.5),)),
        Segment(150.0, 150.5, (Word("два", 150.0, 150.5),)),
    ])
    moved = repair_transcript(words, clock, Path("repaired.wav"))
    assert [round(word.start, 3) for word in moved.words] == [10.0, 150.3]
    assert moved.duration == pytest.approx(320.0)
    assert moved.source_audio == Path("repaired.wav")


def test_a_lone_false_peak_is_not_a_jump() -> None:
    clock = RecorderClock("rec", [Loss(10.0, -120.0, "fine", ["cam"])])
    assert clock.applied == []  # recorders lose samples; they do not gain them
