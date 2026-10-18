"""Resumable, verified, cancellable downloads of pinned model files.

Bytes go to ``<target>.part`` next to the target. Cancellation or a dropped
connection keeps the partial file so the next attempt resumes with an HTTP
range request. The target only appears, atomically, after its size and digest
match the pinned values; a digest mismatch discards the partial file.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studio.core.workspace import project_lock
from studio.modules.cancellation import check_cancelled
from studio.modules.catalog import CatalogModel, RemoteFile

_BLOCK = 1024 * 1024
MARKER = ".verified.json"

Opener = Callable[..., Any]


def file_digest(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        value = hashlib.sha256()
    elif algorithm == "gitsha1":
        value = hashlib.sha1(usedforsecurity=False)
        value.update(b"blob %d\0" % path.stat().st_size)
    else:
        raise ValueError(f"unsupported digest algorithm: {algorithm}")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK), b""):
            value.update(block)
    return value.hexdigest()


def download_file(
    url: str,
    target: Path,
    expected: RemoteFile,
    *,
    progress: Callable[[int], None] = lambda done: None,
    cancelled: Callable[[], bool] = lambda: False,
    opener: Opener = urllib.request.urlopen,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    done = partial.stat().st_size if partial.is_file() else 0
    if done > expected.size:
        partial.unlink()
        done = 0
    if done < expected.size:
        headers = {"User-Agent": "bormostudio"}
        if done:
            headers["Range"] = f"bytes={done}-"
        request = urllib.request.Request(url, headers=headers)
        with opener(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if done and status != 206:
                done = 0  # the server ignored the range; start over
            with partial.open("ab" if done else "wb") as output:
                progress(done)
                for block in iter(lambda: response.read(_BLOCK), b""):
                    check_cancelled(cancelled)
                    done += len(block)
                    if done > expected.size:
                        raise ValueError(f"{expected.path}: download exceeds the pinned size")
                    output.write(block)
                    progress(done)
                output.flush()
                os.fsync(output.fileno())
    if done != expected.size:
        raise OSError(f"{expected.path}: download ended early ({done} of {expected.size} bytes)")
    check_cancelled(cancelled)
    if file_digest(partial, expected.algorithm) != expected.digest:
        partial.unlink()
        raise ValueError(f"{expected.path}: downloaded file checksum mismatch")
    os.replace(partial, target)
    return target


class CatalogInstaller:
    """Installs catalog models under ``root/whisper/<name>`` and ``root/llm``."""

    def __init__(self, root: Path, opener: Opener = urllib.request.urlopen) -> None:
        self.root = root
        self.opener = opener

    def directory(self, model: CatalogModel) -> Path:
        if model.kind == "whisper":
            return self.root / "whisper" / model.name
        return self.root / "llm" / model.repo.replace("/", "--")

    def path(self, model: CatalogModel) -> Path:
        directory = self.directory(model)
        return directory if model.kind == "whisper" else directory / model.files[0].path

    def installed(self, model: CatalogModel) -> bool:
        directory = self.directory(model)
        try:
            marker = json.loads((directory / f"{MARKER}.{model.name}").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return marker == _marker(model) and all(
            (directory / item.path).is_file()
            and (directory / item.path).stat().st_size == item.size
            for item in model.files
        )

    def install(
        self,
        model: CatalogModel,
        *,
        progress: Callable[[dict], None] = lambda event: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Path:
        directory = self.directory(model)
        with project_lock(self.root):
            check_cancelled(cancelled)
            if self.installed(model):
                progress({"phase": "complete", "percent": 100, "name": model.name})
                return self.path(model)
            total = model.size
            finished = 0
            last = -1
            for item in model.files:
                target = directory / item.path
                if target.is_file() and target.stat().st_size == item.size and (
                    file_digest(target, item.algorithm) == item.digest
                ):
                    finished += item.size
                    continue

                def report(done: int, base: int = finished, name: str = item.path) -> None:
                    nonlocal last
                    percent = min(99, int((base + done) * 100 / total)) if total else None
                    if percent != last:
                        last = percent if percent is not None else last
                        progress({
                            "phase": "download", "percent": percent, "bytes": base + done,
                            "total": total, "name": model.name, "file": name,
                        })

                download_file(
                    model.url(item), target, item, progress=report, cancelled=cancelled,
                    opener=self.opener,
                )
                finished += item.size
            progress({"phase": "verify", "percent": None, "name": model.name})
            (directory / f"{MARKER}.{model.name}").write_text(
                json.dumps(_marker(model), indent=2) + "\n", encoding="utf-8"
            )
            progress({"phase": "complete", "percent": 100, "name": model.name})
        return self.path(model)


def _marker(model: CatalogModel) -> dict[str, object]:
    return {
        "name": model.name, "repo": model.repo, "revision": model.revision,
        "files": [[item.path, item.size, item.algorithm, item.digest] for item in model.files],
    }
