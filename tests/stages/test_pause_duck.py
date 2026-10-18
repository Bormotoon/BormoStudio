import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from studio.core.timeline import AudioWarpMap, AudioWarpPiece, TimeDomain
from studio.stages.sync_render import (
    PauseDuck,
    intersect_spans,
    render_audio_plan,
    silence_spans,
)


def test_silence_and_intersection_follow_whispersync() -> None:
    words = [(1.0, 2.0), (2.3, 3.0), (4.0, 5.0)]
    assert silence_spans(words, 6.0, 0.6) == [(0.0, 1.0), (3.0, 4.0), (5.0, 6.0)]
    assert silence_spans([], 2.0, 0.6) == [(0.0, 2.0)]
    assert intersect_spans([(0, 2), (3, 5)], [(1, 4)]) == [(1, 2), (3, 4)]


def test_pauses_follow_the_rendered_pieces_not_a_straight_line() -> None:
    # Recorder 10-14 s plays at camera 1-5 s, then 14-18.5 s is stretched over 5-10 s.
    warp = AudioWarpMap("w", "rec", "cam", (
        AudioWarpPiece(10.0, 4.0, 1.0, 4.0, "copy"),
        AudioWarpPiece(14.0, 4.5, 5.0, 5.0, "atempo"),
    ), TimeDomain.FILE, 3)
    duck = PauseDuck(
        camera_words=((1.2, 3.0), (7.0, 8.0)),
        # At camera 1.2-3.0 and 6.67-8.33; one straight line would say 6.5-8.0.
        recorder_words=((10.2, 12.0), (15.5, 17.0)),
        min_pause_s=0.6,
    )
    pauses = duck.pauses(warp, 10.0)
    assert pauses[0] == (0.0, pytest.approx(1.2))
    assert pauses[1] == (pytest.approx(3.0), pytest.approx(5.0 + 1.5 / 0.9))
    assert pauses[2] == (pytest.approx(5.0 + 3.0 / 0.9), 10.0)
    # Speech the camera heard but the recorder lost is never ducked, and vice versa.
    lost = PauseDuck(camera_words=((3.5, 4.5),), recorder_words=(), min_pause_s=0.6)
    assert all(not (a < 4.0 < b) for a, b in lost.pauses(warp, 10.0))


def test_ducking_is_off_unless_enabled() -> None:
    assert PauseDuck.from_settings({}, [], []) is None
    duck = PauseDuck.from_settings({"pause_duck_enabled": True, "pause_duck_db": -30,
                                    "pause_duck_fade_ms": 20}, [(0, 1)], [(0, 1)])
    assert duck is not None and (duck.db, duck.fade_ms, duck.min_pause_s) == (-30.0, 20, 0.6)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_rendered_voice_is_lowered_in_pauses_only(tmp_path: Path) -> None:
    rate = 48000
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(6 * rate) / rate) * 32767).astype("<i2")
    source = tmp_path / "rec.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "PCM"))
        audio.writeframes(tone.tobytes())
    duck = PauseDuck(camera_words=((0.0, 2.0), (4.0, 6.0)),
                     recorder_words=((0.0, 2.0), (4.0, 6.0)), db=-18.0, fade_ms=50)
    output = tmp_path / "voice.wav"
    render_audio_plan(source, output, [(0.0, 6.0, 1.0)], lead_silence_s=0.0, duration_s=6.0,
                      map_id="w", source_asset_id="rec", target_asset_id="cam", strategy=1,
                      codec="pcm_s16le", duck=duck)
    with wave.open(str(output), "rb") as audio:
        samples = np.frombuffer(audio.readframes(audio.getnframes()), "<i2").astype(float)

    def level(start: float, end: float) -> float:
        part = samples[int(start * rate):int(end * rate)]
        return 20 * np.log10(np.sqrt(np.mean(part ** 2)))

    assert level(2.2, 3.8) - level(0.5, 1.5) == pytest.approx(-18.0, abs=0.5)
    assert level(4.5, 5.5) == pytest.approx(level(0.5, 1.5), abs=0.1)
    assert len(samples) == 6 * rate
