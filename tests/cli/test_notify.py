import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import studio.cli.channel as channel
from studio.cli.main import main
from studio.cli.notify import NotifyOptions, send, summary
from studio.modules.cancellation import InstallationCancelled

RUN = {"status": "failed", "projects": [
    {"source": "/a", "status": "ok"}, {"source": "/b", "status": "failed", "error": "x"},
]}


class _Response(io.BytesIO):
    status = 204


def test_options_validate_and_select_outcomes() -> None:
    with pytest.raises(ValueError):
        NotifyOptions(webhook="file:///etc/passwd")
    with pytest.raises(ValueError):
        NotifyOptions(when="sometimes")
    assert not NotifyOptions().wanted("failed")
    assert NotifyOptions(webhook="https://h").wanted("failed")
    assert not NotifyOptions(webhook="https://h").wanted("ok")
    assert NotifyOptions(webhook="https://h", when="always").wanted("ok")
    assert summary(RUN) == "BormoStudio: run failed (1 failed, 1 ok); failed: /b"


def test_webhook_and_desktop_outcomes_never_raise(monkeypatch) -> None:
    sent = []

    def opener(request, timeout):
        sent.append(json.loads(request.data))
        return _Response()

    outcomes = send(NotifyOptions(webhook="https://hooks.example/x"), RUN, "/r.json",
                    opener=opener)
    assert outcomes == [{"target": "webhook", "status": "sent"}]
    assert sent[0]["text"] == sent[0]["content"] and sent[0]["report"] == "/r.json"

    def offline(request, timeout):
        raise OSError("unreachable")

    failed = send(NotifyOptions(webhook="https://hooks.example/x"), RUN, None, opener=offline)
    assert failed[0]["status"] == "failed"
    monkeypatch.setattr("studio.cli.notify.desktop_command", lambda text: ["notify", text])
    desktop = send(NotifyOptions(desktop=True), RUN, None,
                   runner=lambda command, **kw: SimpleNamespace(returncode=0, stderr=""))
    assert desktop == [{"target": "desktop", "status": "sent", "error": ""}]


def test_batch_report_records_notifications(tmp_path: Path, monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr("studio.cli.notify.urllib.request.urlopen",
                        lambda request, timeout: calls.append(request) or _Response())
    report = tmp_path / "report.json"
    monkeypatch.setattr("studio.cli.main._run_batch", lambda *args, **kwargs: (
        1, [{"source": "/b", "status": "failed", "error": "x"}]))
    code = main(["batch", str(tmp_path), "--report", str(report),
                 "--notify-webhook", "https://hooks.example/x"])
    assert code == 1 and len(calls) == 1
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["notifications"] == [{"target": "webhook", "status": "sent"}]
    capsys.readouterr()


def test_channel_cancel_during_download_reports_cancelled(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(channel, "discover_channel", lambda *args: ["abcdefghijk", "lmnopqrstuv"])

    def cancelled_fetch(url, folder, cancelled):
        raise InstallationCancelled("Installation cancelled.")

    with pytest.raises(InstallationCancelled):
        channel.acquire_channel("url", tmp_path, 2, cancelled_fetch)
    saved = json.loads((tmp_path / "channel-acquisition.json").read_text())["videos"]
    assert saved == [{"id": "abcdefghijk", "status": "cancelled",
                      "source": str(tmp_path / "abcdefghijk")}]
    monkeypatch.setattr("studio.cli.channel.acquire_channel",
                        lambda *args: (_ for _ in ()).throw(InstallationCancelled("stop")))
    report = tmp_path / "night.json"
    arguments = ["channel", "https://youtube.com/@x", str(tmp_path), "--report", str(report)]
    assert main(arguments) == 130
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "cancelled" and data["acquisition"][0]["status"] == "cancelled"
    capsys.readouterr()


def test_batch_settings_supply_notification_defaults(tmp_path: Path, monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr("studio.cli.notify.urllib.request.urlopen",
                        lambda request, timeout: calls.append(request) or _Response())
    monkeypatch.setattr("studio.cli.main._run_batch", lambda *args, **kwargs: (
        0, [{"source": "/b", "status": "ok"}]))
    code = main(["batch", str(tmp_path), "--set", "batch.notify_when=always",
                 "--set", "batch.notify_webhook=https://hooks.example/x"])
    assert code == 0 and len(calls) == 1
    capsys.readouterr()


def test_batch_stops_before_a_project_on_low_disk(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from studio.cli.main import _run_batch

    monkeypatch.setattr("studio.cli.main.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=2e9))
    status, results = _run_batch([tmp_path, tmp_path / "next"], [], only=None, skip=None,
                                 stream=False, min_free_gb=5)
    assert status == 1
    assert results[0]["status"] == "failed" and "2.0 GB free" in results[0]["error"]
    assert results[1]["status"] == "not_started"
