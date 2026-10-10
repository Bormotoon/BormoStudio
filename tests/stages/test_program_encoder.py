import subprocess
from unittest.mock import patch

import pytest

from studio.stages.program_encoder import select_encoder


def test_cpu_does_not_probe_hardware() -> None:
    with patch("studio.stages.program_encoder.subprocess.run") as run:
        assert select_encoder("cpu").codec == "libx264"
        run.assert_not_called()


@pytest.mark.parametrize("requested", ["auto", "nvenc"])
def test_usable_nvenc_is_selected(requested: str) -> None:
    with patch("studio.stages.program_encoder.subprocess.run") as run:
        run.return_value.returncode = 0
        assert select_encoder(requested).codec == "h264_nvenc"
        assert "h264_nvenc" in run.call_args.args[0]


def test_auto_falls_back_but_explicit_nvenc_fails() -> None:
    with patch("studio.stages.program_encoder.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stderr = "driver unavailable"
        choice = select_encoder("auto")
        assert choice.codec == "libx264"
        assert "driver unavailable" in choice.reason
        with pytest.raises(RuntimeError, match="driver unavailable"):
            select_encoder("nvenc")


def test_probe_timeout_falls_back() -> None:
    with patch("studio.stages.program_encoder.subprocess.run",
               side_effect=subprocess.TimeoutExpired("ffmpeg", 20)):
        assert select_encoder("auto").codec == "libx264"


def test_unknown_encoder_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        select_encoder("typo")


def test_auto_prefers_videotoolbox_on_macos_with_a_bitrate() -> None:
    with patch("studio.stages.program_encoder.sys.platform", "darwin"), patch(
        "studio.stages.program_encoder.subprocess.run"
    ) as run:
        run.return_value.returncode = 0
        choice = select_encoder("auto")
        assert choice.codec == "h264_videotoolbox" and choice.options == ("-b:v", "12M")
        assert "h264_videotoolbox" in run.call_args.args[0]
        assert "-b:v" in run.call_args.args[0]


def test_explicit_videotoolbox_fails_loudly_and_auto_falls_back() -> None:
    with patch("studio.stages.program_encoder.sys.platform", "darwin"), patch(
        "studio.stages.program_encoder.subprocess.run"
    ) as run:
        run.return_value.returncode = 1
        run.return_value.stderr = "Error: cannot create compression session"
        assert select_encoder("auto").codec == "libx264"
        with pytest.raises(RuntimeError, match="VideoToolbox"):
            select_encoder("videotoolbox")
