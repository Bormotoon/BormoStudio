import os
from pathlib import Path

import pytest

from studio.core import ffmpeg_select


def _build(folder: Path, filters: str, encoders: str, broken: bool = False) -> Path:
    folder.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        script = folder / name
        body = "exit 127" if broken else (
            'case "$2" in -filters) echo "' + filters + '";; *) echo "' + encoders + '";; esac')
        script.write_text("#!/bin/sh\n" + body + "\n")
        script.chmod(0o755)
    return folder


def test_the_build_with_libass_and_nvenc_goes_first(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _build(tmp_path / "broken", "", "", broken=True)
    plain = _build(tmp_path / "brew", " ... loudnorm ", " libx264 ")
    nvenc = _build(tmp_path / "local", " ... loudnorm ", " h264_nvenc ")
    full = _build(tmp_path / "usr", " ... subtitles ", " h264_nvenc ")
    path = os.pathsep.join(str(p) for p in (broken, plain, nvenc, full))
    monkeypatch.setenv("PATH", path)
    monkeypatch.delenv("BSTUDIO_FFMPEG_CHOSEN", raising=False)
    monkeypatch.delenv("BSTUDIO_FFMPEG_DIR", raising=False)
    monkeypatch.delenv("UVF_FFMPEG_DIR", raising=False)
    assert ffmpeg_select.choose_ffmpeg() == full.resolve()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(full.resolve())
    assert os.environ["BSTUDIO_FFMPEG_CHOSEN"] == str(full.resolve())


def test_an_explicit_folder_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = _build(tmp_path / "brew", "", " libx264 ")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("BSTUDIO_FFMPEG_DIR", str(plain))
    monkeypatch.delenv("BSTUDIO_FFMPEG_CHOSEN", raising=False)
    assert ffmpeg_select.choose_ffmpeg() == plain
    assert os.environ["PATH"].startswith(str(plain))
