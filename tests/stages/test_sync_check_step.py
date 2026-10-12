import json
from pathlib import Path
from unittest.mock import patch

from studio.core.transcript import Segment, Transcript, Word
from studio.stages.sync_check_step import check_voices


def test_content_check_reports_missing_voice_and_unloads_backend(tmp_path: Path) -> None:
    voice = tmp_path / "voice.wav"
    reference = Transcript(voice, "en", 3, [Segment(0, 3, tuple(
        Word(f"word{i}", i * 0.3, i * 0.3 + 0.2) for i in range(6)
    ))])
    with patch("studio.stages.sync_check_step.WhisperEngine") as factory:
        engine = factory.return_value
        engine.transcribe.return_value = Transcript(voice, "en", 3, [])
        artifacts = check_voices({"cam": reference}, {"cam": voice}, tmp_path / "check", {})
        engine.backend.unload.assert_called_once()
    report = json.loads(artifacts[-1].read_text())
    assert report["clips"][0]["status"] == "failed"
    assert report["clips"][0]["spans"][0]["kind"] == "content"
    assert report["clips"][0]["time_domain"] == "rendered_audio"
    assert all(path.is_file() for path in artifacts)


def test_content_gap_with_good_acoustics_is_a_camera_transcript_gap() -> None:
    from studio.stages.sync_check_step import combine_checks
    from studio.stages.sync_self_check import SelfCheckOutcome, SelfCheckSpan
    from studio.stages.sync_verify import LagSample, VerifyReport

    acoustic = VerifyReport("cam", "voice", duration_s=290, grid_s=2.5)
    acoustic.samples = [LagSample(2 + 2.5 * i, 0.5, 200) for i in range(115)]
    content = SelfCheckOutcome("failed", [SelfCheckSpan(251, 262, "content", "5 vs 0")])
    result = combine_checks(content, acoustic, 20.0)
    assert result["status"] == "passed"
    assert result["warnings"][0]["kind"] == "camera_transcript_gap"
    assert result["acoustic"]["status"] == "passed"

    shifted = VerifyReport("cam", "voice", duration_s=290, grid_s=2.5)
    shifted.samples = [LagSample(2 + 2.5 * i, -88.0 if i < 24 else 0.5, 200) for i in range(115)]
    result = combine_checks(SelfCheckOutcome("passed"), shifted, 20.0)
    assert result["status"] == "failed"
    assert result["acoustic"]["shifted_intervals"]


def test_acoustic_check_of_a_short_clip_uses_the_short_grid(tmp_path: Path) -> None:
    from studio.stages.sync_verify import VerifyReport

    voice = tmp_path / "voice.wav"
    reference = Transcript(voice, "en", 12, [Segment(0, 3, tuple(
        Word(f"word{i}", i * 0.3, i * 0.3 + 0.2) for i in range(6)
    ))])
    with patch("studio.stages.sync_check_step.WhisperEngine") as factory, patch(
        "studio.stages.sync_verify.measure", return_value=VerifyReport("cam", "voice",
                                                                       duration_s=12),
    ) as measure:
        factory.return_value.transcribe.return_value = reference
        check_voices({"cam": reference}, {"cam": voice}, tmp_path / "check", {},
                     cameras={"cam": tmp_path / "cam.mov"})
    assert measure.call_args.kwargs["grid_s"] == 0.6
    assert measure.call_args.kwargs["window_s"] == 4.0
