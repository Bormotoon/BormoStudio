from pathlib import Path

import pytest

import studio.llm.session as sessions
from studio.modules.cancellation import InstallationCancelled


def test_owned_session_reuses_and_stops(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    calls = []

    class Server:
        def __init__(self, *args, **kwargs):
            self.process = type("Process", (), {"poll": lambda self: None})()

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sessions, "LlamaServer", Server)
    monkeypatch.setattr(sessions.shutil, "which", lambda name: "/server")
    monkeypatch.setattr(sessions, "urlopen", lambda *a, **kw: Response())
    settings = {
        "managed": True,
        "model_path": str(model),
        "executable": "server",
        "port": 0,
        "startup_timeout_s": 1,
    }
    session = sessions.ManagedLlamaSession(settings, tmp_path)
    session.ensure()
    session.ensure()
    session.close()
    assert calls == ["start", "stop"]
    session = sessions.ManagedLlamaSession(settings, tmp_path, lambda: True)
    with pytest.raises(InstallationCancelled):
        session.ensure()
    assert calls[-2:] == ["start", "stop"]


def test_managed_endpoint_override_keeps_input_unchanged() -> None:
    original = {"llm": {"managed": True, "port": 9090}, "text": {"base_url": "old"}}
    result = sessions.effective_llm_settings(original)
    assert result["text"]["base_url"] == "http://127.0.0.1:9090"
    assert original["text"]["base_url"] == "old"


def test_model_identity_tracks_replaced_bytes(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"first")
    settings = {"llm": {"managed": True, "model_path": str(path)}}
    before = sessions.model_identity(settings)
    path.write_bytes(b"other")
    assert sessions.model_identity(settings) != before


def test_batch_owner_reuses_and_switches_model(tmp_path: Path, monkeypatch) -> None:
    closed = []

    class Session:
        def __init__(self, config, work, cancelled):
            self.work = work

        def close(self):
            closed.append(self.work)

    monkeypatch.setattr(sessions, "ManagedLlamaSession", Session)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"first")
    settings = {"llm": {"managed": True, "model_path": str(model)}}
    owner = sessions.BatchLlamaSession()
    first = owner.select(settings, tmp_path / "one")
    assert owner.select(settings, tmp_path / "two") is first
    model.write_bytes(b"other")
    assert owner.select(settings, tmp_path / "two") is not first
    assert closed == [tmp_path / "one"]
    owner.close()
    assert closed == [tmp_path / "one", tmp_path / "two"]
