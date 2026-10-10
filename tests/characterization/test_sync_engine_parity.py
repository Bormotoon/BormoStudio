import struct
import wave
from pathlib import Path

import pytest
from whispersync.config import WhisperSyncConfig
from whispersync.engine.pipeline import clip_pieces, recorder_word_gaps
from whispersync.engine.timestretch import assemble_continuous, render_piece
from whispersync.models import AlignmentMap, Anchor

from studio.stages.sync_engine import render_aligned_clip


@pytest.mark.parametrize("strategy", [1, 2, 3])
@pytest.mark.parametrize("offset,drift", [(0.1, 1.001), (-0.2, 0.999), (0.0, 1.0)])
def test_planning_through_render_matches_frozen_engine(
    tmp_path: Path, strategy: int, offset: float, drift: float,
) -> None:
    rate = 48000
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(b"".join(
            struct.pack("<h", index % 20000 - 10000) for index in range(rate * 4)
        ))
    alignment = AlignmentMap(
        anchors=[Anchor(cam_time=t * drift + offset, rec_time=t, token=str(t), confidence=0.9)
                 for t in (0.5, 1.0, 1.5, 2.0, 2.5)],
        offset=offset, k=drift, residual_ms=0.0,
    )
    words = [(0.5, 1.0), (2.0, 2.5)]
    settings = WhisperSyncConfig()
    lead, pieces = clip_pieces(
        alignment, 3.0, 4.0, strategy, settings,
        rec_word_gaps=recorder_word_gaps(words), rec_words=words,
    )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    segments = [render_piece(
        source, legacy, start, duration, factor, index, 10,
        sample_rate=rate, channels=1, codec="pcm_s16le",
    ) for index, (start, duration, factor) in enumerate(pieces)]
    reference = assemble_continuous(
        segments, lead, 3.0, rate, tmp_path / "reference.wav", channels=1, codec="pcm_s16le",
    )
    output = tmp_path / "studio.wav"
    plan = render_aligned_clip(
        alignment, settings, source, output, clip_duration_s=3.0, recorder_duration_s=4.0,
        recorder_words=words, strategy=strategy, source_asset_id="rec", target_asset_id="cam",
        codec="pcm_s16le",
    )
    with wave.open(str(reference), "rb") as old, wave.open(str(output), "rb") as new:
        assert old.getparams() == new.getparams()
        assert old.readframes(old.getnframes()) == new.readframes(new.getnframes())
    assert plan.duration_s == 3.0
    assert plan.warp.strategy == strategy
