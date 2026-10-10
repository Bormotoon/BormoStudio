"""Owned llama-server process lifecycle."""

from __future__ import annotations

import functools
import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any


@dataclass(slots=True)
class LlamaServer:
    executable: Path
    model: Path
    work_dir: Path
    port: int = 8080
    process: subprocess.Popen[str] | None = None
    # Tuning flags from server_arguments(); the log shows the real GPU offload.
    arguments: list[str] = field(default_factory=list)
    log_file: Path | None = None
    _log: IO[str] | None = None

    def start(self) -> int:
        if self.process and self.process.poll() is None:
            return self.process.pid
        self.work_dir.mkdir(parents=True, exist_ok=True)
        command = [str(self.executable), "-m", str(self.model), "--host", "127.0.0.1",
                   "--port", str(self.port), *self.arguments]
        output: IO[str] | int = subprocess.DEVNULL
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_file.open("a", encoding="utf-8")
            self._log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"{' '.join(command)} =====\n")
            self._log.flush()
            output = self._log
        self.process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT if self._log else subprocess.DEVNULL,
            text=True,
            start_new_session=os.name != "nt",
        )
        marker = {
            "pid": self.process.pid,
            "started": time.time(),
            "executable": str(self.executable.resolve()),
            "model": str(self.model.resolve()),
            "port": self.port,
            "arguments": self.arguments,
        }
        self.marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        return self.process.pid

    @property
    def marker_path(self) -> Path:
        return self.work_dir / "llama-server.json"

    def stop(self, timeout_s: float = 5.0) -> None:
        try:
            self._stop(timeout_s)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None

    def _stop(self, timeout_s: float) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is not None:
            if self._owns_process(process.pid):
                self.marker_path.unlink(missing_ok=True)
            return
        if not self._owns_process(process.pid):
            raise RuntimeError("refusing to stop an unowned llama-server process")
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        self.marker_path.unlink(missing_ok=True)

    def _owns_process(self, pid: int) -> bool:
        try:
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return bool(
            marker.get("pid") == pid
            and marker.get("executable") == str(self.executable.resolve())
        )

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def available_ram_mb() -> int | None:
    """MemAvailable in MiB, or None where /proc/meminfo is unavailable."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            return int(parts[1]) // 1024 if len(parts) > 1 and parts[1].isdigit() else None
    return None


def resolve_cache_ram_mb(settings: dict[str, Any]) -> int | None:
    """The prompt-cache budget: a fixed size, none, or free memory minus a reserve.

    "auto" is decided once at start: it is a budget, not a live guard.
    """
    value = settings.get("cache_ram_mb")
    if value is None or value == "":
        return None
    if value != "auto":
        return max(0, int(value))
    low, high = int(settings.get("cache_ram_min_mb", 512)), int(
        settings.get("cache_ram_max_mb", 8192))
    available = available_ram_mb()
    if available is None:
        return low
    return max(low, min(available - int(settings.get("cache_ram_reserve_mb", 6144)), high))


@functools.lru_cache(maxsize=16)
def server_supports(executable: str, flag: str) -> bool:
    """Whether this llama-server build knows a flag; unknown flags stop it from starting."""
    try:
        result = subprocess.run([executable, "--help"], capture_output=True, text=True,
                                timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in (result.stdout or "") + (result.stderr or "")


def server_arguments(
    settings: dict[str, Any], executable: str,
    supports: Callable[[str, str], bool] = server_supports,
) -> list[str]:
    """llama-server flags for the configured tuning; unset values keep defaults."""
    arguments: list[str] = []
    for key, flag in (("n_gpu_layers", "--n-gpu-layers"), ("ctx_size", "--ctx-size"),
                      ("batch_size", "--batch-size"), ("ubatch_size", "--ubatch-size"),
                      ("threads", "--threads"), ("main_gpu", "--main-gpu"),
                      ("parallel", "--parallel")):
        if settings.get(key) is not None:
            arguments += [flag, str(int(settings[key]))]
    if settings.get("flash_attn", "auto") != "auto":
        # An explicit value: bare --flash-attn can swallow the next argument.
        arguments += ["--flash-attn", str(settings["flash_attn"])]
    for key, flag in (("cache_type_k", "--cache-type-k"), ("cache_type_v", "--cache-type-v")):
        if settings.get(key):
            arguments += [flag, str(settings[key])]
    # Newer flags are passed only when this build knows them.
    cache_ram = resolve_cache_ram_mb(settings)
    if cache_ram is not None and supports(executable, "--cache-ram"):
        arguments += ["--cache-ram", str(cache_ram)]
    if settings.get("ctx_checkpoints") is not None and supports(executable, "--ctx-checkpoints"):
        arguments += ["--ctx-checkpoints", str(int(settings["ctx_checkpoints"]))]
    arguments += [str(item) for item in settings.get("extra_args", [])]
    return arguments
