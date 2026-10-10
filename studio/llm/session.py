"""Owned local llama session shared across adjacent LLM workers."""

from __future__ import annotations

import shutil
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from studio.llm.server import LlamaServer, server_arguments
from studio.modules.cancellation import check_cancelled


class ManagedLlamaSession:
    def __init__(
        self,
        settings: dict[str, Any],
        work_dir: Path,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        self.settings = settings
        self.work_dir = work_dir
        self.cancelled = cancelled
        self.server: LlamaServer | None = None

    def ensure(self) -> None:
        if not self.settings.get("managed", False):
            return
        if (self.server is not None and self.server.process is not None
                and self.server.process.poll() is None):
            return
        model = Path(self.settings["model_path"]).expanduser()
        if not model.is_file():
            raise ValueError(f"llama model does not exist: {model}")
        executable = shutil.which(self.settings["executable"])
        if not executable:
            raise ValueError("llama-server executable is unavailable")
        port = int(self.settings["port"])
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError("managed llama port is occupied; choose another port") from exc
        log_name = str(self.settings.get("log_file", "llama-server.log") or "")
        server = LlamaServer(
            Path(executable), model, self.work_dir / "llama", port,
            arguments=server_arguments(self.settings, executable),
            log_file=(self.work_dir / "llama" / log_name) if log_name else None,
        )
        self.server = server
        try:
            server.start()
            deadline = time.monotonic() + float(self.settings["startup_timeout_s"])
            while time.monotonic() < deadline:
                check_cancelled(self.cancelled)
                if server.process is None or server.process.poll() is not None:
                    raise RuntimeError("llama-server exited before readiness")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            return
                except OSError:
                    pass
                time.sleep(0.1)
            raise RuntimeError("llama-server startup timed out")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None


def effective_llm_settings(settings: dict[str, Any]) -> dict[str, Any]:
    from copy import deepcopy

    settings = deepcopy(settings)
    if settings.get("llm", {}).get("managed"):
        for section in ("text", "reels", "roughcut"):
            settings.setdefault(section, {})["base_url"] = (
                f"http://127.0.0.1:{settings['llm']['port']}"
            )
    return settings


def model_identity(settings: dict[str, Any]) -> str:
    from studio.modules.models import digest

    config = settings.get("llm", {})
    if not config.get("managed", False):
        return ""
    path = Path(config.get("model_path", "")).expanduser()
    return digest(path) if path.is_file() else "missing-model"


class BatchLlamaSession:
    """One owner across project boundaries; non-LLM work still releases resources."""

    def __init__(self, cancelled: Callable[[], bool] = lambda: False) -> None:
        self.cancelled = cancelled
        self.session: ManagedLlamaSession | None = None
        self.identity: str | None = None

    def select(self, settings: dict[str, Any], work_dir: Path) -> ManagedLlamaSession:
        from studio.core.project import stable_fingerprint

        identity = stable_fingerprint(settings.get("llm", {}), model_identity(settings))
        if identity != self.identity or self.session is None:
            self.close()
            self.session = ManagedLlamaSession(settings.get("llm", {}), work_dir, self.cancelled)
            self.identity = identity
        return self.session

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
        self.session = None
        self.identity = None
