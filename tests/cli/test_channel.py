import json
from types import SimpleNamespace

import studio.cli.channel as channel


def test_channel_discovery_is_bounded_and_deduplicated(monkeypatch):
    monkeypatch.setattr(
        channel,
        "run_cancellable",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"entries": [{"id": "abcdefghijk"}, {"id": "abcdefghijk"}, {"id": "../bad"}]}
            )
        ),
    )
    assert channel.discover_channel("https://youtube.com/@example", 2) == ["abcdefghijk"]


def test_channel_resumes_and_reports_failed_acquisition(tmp_path, monkeypatch):
    monkeypatch.setattr(channel, "discover_channel", lambda *args: ["abcdefghijk", "lmnopqrstuv"])
    project = tmp_path / "abcdefghijk" / "_studio"
    project.mkdir(parents=True)
    from studio.core.project import Project

    Project(tmp_path / "abcdefghijk", project).save(project / "project.json")

    def failed(*args):
        raise RuntimeError("offline")

    sources, results = channel.acquire_channel("url", tmp_path, 2, failed)
    assert sources == [tmp_path / "abcdefghijk"]
    assert [item["status"] for item in results] == ["reused", "failed"]
    assert json.loads((tmp_path / "channel-acquisition.json").read_text())["videos"] == results


def test_channel_filters_apply_before_the_limit(monkeypatch):
    calls = []

    def listing(command, *args, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"entries": [
            {"id": "excludedvid", "duration": 3000},
            {"id": "livestream1", "duration": 3000, "live_status": "is_live"},
            {"id": "shortvideo1", "duration": 30},
            {"id": "unknownlen1"},
            {"id": "goodvideo01", "duration": 3000},
            {"id": "goodvideo02", "duration": 3000},
        ]}))

    monkeypatch.setattr(channel, "run_cancellable", listing)
    found = channel.discover_channel("url", 2, filters={
        "exclude": ["excludedvid"], "skip_live": True, "min_duration_s": 60,
    })
    assert found == ["unknownlen1", "goodvideo01"]
    assert calls[0][calls[0].index("--playlist-end") + 1] == "10"


def test_fetch_settings_become_yt_dlp_options():
    from studio.stages.fetch import yt_dlp_options

    assert yt_dlp_options({"download": "audio", "retries": 5}) == [
        "--format", "bestaudio/best", "--retries", "5",
    ]
    assert yt_dlp_options({"max_height": 720, "cookies_file": "c.txt", "rate_limit": "5M"}) == [
        "--format", "bv*[height<=720]+ba/b[height<=720]/b", "--cookies", "c.txt",
        "--limit-rate", "5M",
    ]


def test_a_finished_download_without_a_project_is_taken_as_is(tmp_path, monkeypatch):
    monkeypatch.setattr(channel, "discover_channel", lambda *args: ["abcdefghijk"])
    folder = tmp_path / "abcdefghijk"
    folder.mkdir()
    (folder / "source.webm").write_bytes(b"media")
    (folder / "source.info.json").write_text('{"id": "abcdefghijk"}', encoding="utf-8")

    def fetch(url, folder, cancelled):
        raise AssertionError("must not download again")

    sources, results = channel.acquire_channel("url", tmp_path, 1, fetch)
    assert sources == [folder]
    assert results == [{"id": "abcdefghijk", "status": "downloaded", "source": str(folder)}]
    (folder / "source.info.json").write_text('{"id": "other000000"}', encoding="utf-8")
    calls = []
    channel.acquire_channel("url", tmp_path, 1, lambda *args: calls.append(args))
    assert calls  # another video's info: not this download
