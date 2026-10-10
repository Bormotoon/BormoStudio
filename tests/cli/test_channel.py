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
