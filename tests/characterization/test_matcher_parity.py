from dataclasses import asdict
from pathlib import Path

import pytest
from whispersync.config import WhisperSyncConfig
from whispersync.engine.matcher import align as frozen_align
from whispersync.models import Segment as FrozenSegment
from whispersync.models import Transcript as FrozenTranscript
from whispersync.models import Word as FrozenWord

from studio.stages.sync_match_settings import MatchSettings
from studio.stages.sync_matcher import align
from studio.stages.sync_models import Segment, Transcript, Word


@pytest.mark.parametrize("offset,k", [(2.0, 1.001), (-3.0, 0.999), (0.0, 1.0)])
def test_migrated_matcher_matches_frozen_ransac(offset: float, k: float) -> None:
    tokens = [f"token{index}" for index in range(30)]
    rec_times = [float(index * 2 + 10) for index in range(30)]
    cam_times = [offset + k * value for value in rec_times]
    # A deliberately wrong ASR timestamp must not dictate the fitted clock.
    cam_times[14] += 4.0

    def old(times: list[float]) -> FrozenTranscript:
        return FrozenTranscript(Path("audio"), "en", 80.0, [FrozenSegment(
            0.0, 80.0, [FrozenWord(token, time, time + 0.2, 0.99)
                       for token, time in zip(tokens, times, strict=True)],
        )])

    def new(times: list[float]) -> Transcript:
        return Transcript(Path("audio"), "en", 80.0, [Segment(
            0.0, 80.0, [Word(token, time, time + 0.2, 0.99)
                       for token, time in zip(tokens, times, strict=True)],
        )])

    expected = frozen_align(old(cam_times), old(rec_times), WhisperSyncConfig())
    actual = align(new(cam_times), new(rec_times), MatchSettings())
    # ``notes`` is new: the frozen WhisperSync map has nothing to explain.
    assert {key: value for key, value in asdict(actual).items() if key != "notes"} == asdict(
        expected)
    assert actual.k == pytest.approx(k)
    assert actual.offset == pytest.approx(offset + 0.1 * (1 - k))
