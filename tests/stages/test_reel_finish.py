import shutil
import subprocess
from pathlib import Path

import pytest

from studio.stages.reel_finish import measure_loudness, normalize_loudness, render_cover

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


def _clip(path: Path, audio: str | None) -> Path:
    inputs = ["-f", "lavfi", "-i", "color=c=blue:s=64x64:d=3:r=25"]
    if audio:
        inputs += ["-f", "lavfi", "-i", audio]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", *(["-c:a", "aac"] if audio else []), str(path)],
                   check=True)
    return path


def test_quiet_reel_is_normalized_to_target(tmp_path: Path) -> None:
    video = _clip(tmp_path / "reel.mp4", "sine=f=440:d=3,volume=0.05")
    before = measure_loudness(video)
    assert before is not None and before["input_i"] < -30
    report = normalize_loudness(video, -14.0)
    after = measure_loudness(video)
    assert report["status"] == "normalized" and after is not None
    assert abs(after["input_i"] + 14) < 1.5
    render_cover(video, tmp_path / "cover.jpg", 1.0)
    assert (tmp_path / "cover.jpg").stat().st_size > 0


def test_reel_without_audio_is_left_alone(tmp_path: Path) -> None:
    video = _clip(tmp_path / "silent.mp4", None)
    data = video.read_bytes()
    assert normalize_loudness(video, -14.0) == {"status": "no-audio"}
    assert video.read_bytes() == data
