from pathlib import Path

from studio.core.project import Project
from studio.stages import fetch
from studio.stages.fetch import FetchStage, fetch_to_directory, read_fetch_metadata


def test_fetch_decision_skips_local_projects(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "_studio")
    assert FetchStage().decide(project, {}).reason == "input is a local folder"
    assert FetchStage().decide(project, {"fetch": {"url": "https://youtu.be/id"}}).choices[
        "url"
    ] == "https://youtu.be/id"


def test_fetch_metadata_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "info.json"
    path.write_text('{"title":"Episode"}', encoding="utf-8")
    assert read_fetch_metadata(path)["title"] == "Episode"


def test_fetch_to_directory_uses_no_overwrite_and_returns_one_source(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    def run(command: list[str], *_: object, **__: object) -> object:
        assert "--no-overwrites" in command
        (tmp_path / "source.mp4").write_bytes(b"media")
        (tmp_path / "source.info.json").write_text("{}", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(fetch, "run_cancellable", run)

    media, info = fetch_to_directory("https://example.invalid/video", tmp_path)

    assert media.name == "source.mp4"
    assert info.name == "source.info.json"
    assert not (tmp_path / fetch.INCOMPLETE).exists()


FAKE_YT_DLP = """
import pathlib, sys, time
output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1]).parent
part = output / "source.mp4.part"
if part.exists():
    part.rename(output / "source.mp4")
    (output / "source.info.json").write_text("{}")
    sys.exit(0)
part.write_bytes(b"partial")
time.sleep(60)
"""


def test_cancel_mid_download_keeps_partial_and_same_url_resumes(tmp_path, monkeypatch):
    import sys
    import time

    import pytest

    from studio.cli.main import _fetch_project
    from studio.modules.cancellation import InstallationCancelled

    script = tmp_path / "fake_yt_dlp.py"
    script.write_text(FAKE_YT_DLP, encoding="utf-8")
    monkeypatch.setattr(fetch, "_yt_dlp_command", lambda: [sys.executable, str(script)])
    destination = tmp_path / "episode"
    started = time.monotonic()

    def cancel_once_partial_exists():
        return (destination / "source.mp4.part").exists()

    with pytest.raises(InstallationCancelled):
        fetch_to_directory("https://youtu.be/x", destination, cancel_once_partial_exists)
    assert time.monotonic() - started < 20
    assert (destination / "source.mp4.part").read_bytes() == b"partial"
    assert fetch.resumable_fetch(destination, "https://youtu.be/x")
    assert not fetch.resumable_fetch(destination, "https://youtu.be/other")
    with pytest.raises(ValueError, match="interrupted download"):
        _fetch_project("https://youtu.be/other", destination)
    media, info = fetch_to_directory("https://youtu.be/x", destination)
    assert media.name == "source.mp4" and info.name == "source.info.json"
    assert not (destination / fetch.INCOMPLETE).exists()
