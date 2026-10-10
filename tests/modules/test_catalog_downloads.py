import hashlib
import io
import re
from pathlib import Path

import pytest

from studio.modules.cancellation import InstallationCancelled
from studio.modules.catalog import CATALOG, CatalogModel, RemoteFile, whisper_entry
from studio.modules.downloads import CatalogInstaller, download_file, file_digest
from studio.modules.models import ModelManager, whisper_model_source


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, status: int = 200) -> None:
        super().__init__(payload)
        self.status = status


class _Server:
    """Serves payloads by URL and honours byte ranges like Hugging Face."""

    def __init__(self, files: dict[str, bytes], ranges: bool = True) -> None:
        self.files = files
        self.ranges = ranges
        self.requests: list[tuple[str, str | None]] = []

    def __call__(self, request, timeout=None):  # type: ignore[no-untyped-def]
        url = request.full_url
        header = request.get_header("Range")
        self.requests.append((url, header))
        payload = self.files[url]
        if header and self.ranges:
            start = int(header.split("=")[1].rstrip("-"))
            return _Response(payload[start:], 206)
        return _Response(payload)


def _remote(path: str, payload: bytes, algorithm: str = "sha256") -> RemoteFile:
    if algorithm == "sha256":
        digest = hashlib.sha256(payload).hexdigest()
    else:
        digest = hashlib.sha1(b"blob %d\0" % len(payload) + payload).hexdigest()
    return RemoteFile(path, len(payload), algorithm, digest)


def test_catalog_entries_are_pinned_and_complete() -> None:
    for model in CATALOG.values():
        assert re.fullmatch("[0-9a-f]{40}", model.revision)
        assert model.files and all(item.size > 0 for item in model.files)
        for item in model.files:
            assert re.fullmatch("[0-9a-f]{64}" if item.algorithm == "sha256" else "[0-9a-f]{40}",
                                item.digest)
        if model.kind == "whisper":
            assert {"config.json", "model.bin"} <= {item.path for item in model.files}
    assert whisper_entry("large-v3").name == "whisper-large-v3"
    assert whisper_entry("nonexistent") is None


def test_git_blob_digest_matches_git(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b'{"a":1}')
    assert file_digest(path, "gitsha1") == "daa5053ecf5f9a37b2de733d0751cc1ab53ac010"


def test_cancelled_download_resumes_with_range_and_verifies(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 12000  # several blocks
    expected = _remote("model.bin", payload)
    server = _Server({"https://x/model.bin": payload})
    target = tmp_path / "model.bin"
    seen = []

    def cancel_after_first_block() -> bool:
        return len(seen) > 1

    with pytest.raises(InstallationCancelled):
        download_file("https://x/model.bin", target, expected, progress=seen.append,
                      cancelled=cancel_after_first_block, opener=server)
    partial = target.with_name("model.bin.part")
    assert partial.is_file() and 0 < partial.stat().st_size < len(payload)
    assert not target.exists()
    resumed_from = partial.stat().st_size
    download_file("https://x/model.bin", target, expected, opener=server)
    assert server.requests[-1][1] == f"bytes={resumed_from}-"
    assert target.read_bytes() == payload and not partial.exists()


def test_ignored_range_restarts_and_bad_digest_is_discarded(tmp_path: Path) -> None:
    payload = b"weights" * 1000
    target = tmp_path / "w.bin"
    target.with_name("w.bin.part").write_bytes(payload[:100])
    server = _Server({"https://u": payload}, ranges=False)
    download_file("https://u", target, _remote("w.bin", payload), opener=server)
    assert target.read_bytes() == payload
    bad = RemoteFile("w2.bin", len(payload), "sha256", "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        download_file("https://u", tmp_path / "w2.bin", bad, opener=server)
    assert not (tmp_path / "w2.bin").exists() and not (tmp_path / "w2.bin.part").exists()
    short = RemoteFile("w3.bin", len(payload) + 5, "sha256", "0" * 64)
    with pytest.raises(OSError, match="ended early"):
        download_file("https://u", tmp_path / "w3.bin", short, opener=server)


def test_whisper_install_marks_directory_and_feeds_transcribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, weights = b'{"n": 1}', b"\x00" * 5000
    model = CatalogModel(
        "whisper-test", "whisper", "test", "org/repo", "a" * 40,
        (_remote("config.json", config, "gitsha1"), _remote("model.bin", weights)),
        "test-model",
    )
    server = _Server({
        model.url(model.files[0]): config, model.url(model.files[1]): weights,
    })
    installer = CatalogInstaller(tmp_path, opener=server)
    assert not installer.installed(model)
    events = []
    path = installer.install(model, progress=events.append)
    assert path == tmp_path / "whisper" / "whisper-test"
    assert installer.installed(model) and events[-1]["percent"] == 100
    assert any(event.get("file") == "model.bin" for event in events)
    requests = len(server.requests)
    installer.install(model)
    assert len(server.requests) == requests

    import studio.modules.catalog as catalog

    monkeypatch.setitem(catalog.CATALOG, model.name, model)
    monkeypatch.setattr(catalog, "_MODELS", (model,))
    manager = ModelManager(tmp_path)
    assert manager.describe("whisper-test")["installed"] is True
    assert whisper_model_source("test-model", manager) == str(path)
    assert whisper_model_source("large-v3", manager) == "large-v3"
