"""Portable project locking and atomic publication helpers."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def ordinary_mode(path: str | Path) -> None:
    """Give a ``mkstemp`` file the mode a plain write would (``0o666`` minus umask).

    ``mkstemp`` always creates 0600; published under the final name that would
    hide project files from other users of a shared folder (an editor on a Mac
    reading the exports over the network).
    """
    os.chmod(path, 0o666 & ~_umask())


def _umask() -> int:
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    current = os.umask(0o022)
    os.umask(current)
    return current


class ProjectLockedError(RuntimeError):
    """Another live Studio process owns this project workspace."""


@contextmanager
def published(final_path: Path) -> Iterator[Path]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=final_path.parent, prefix=f".{final_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    ordinary_mode(name)
    temporary = Path(name)
    try:
        yield temporary
        if not temporary.is_file():
            raise RuntimeError(f"nothing was written for {final_path}")
        os.replace(temporary, final_path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


@contextmanager
def project_lock(work_dir: Path, *, timeout_s: float = 0.0) -> Iterator[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / ".studio-run.lock"
    token = f"{os.getpid()}-{time.time_ns()}"
    payload = json.dumps({"pid": os.getpid(), "token": token, "started": time.time()})
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
            break
        except FileExistsError:
            if _remove_stale(path):
                continue
            if time.monotonic() >= deadline:
                raise ProjectLockedError(f"project is locked by another process: {path}") from None
            time.sleep(0.1)
    try:
        yield path
    finally:
        try:
            owner = json.loads(path.read_text(encoding="utf-8"))
            if owner.get("token") == token:
                path.unlink()
        except (OSError, ValueError):
            pass


def _remove_stale(path: Path) -> bool:
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        pid = int(owner.get("pid", 0))
    except (OSError, ValueError, TypeError):
        pid = 0
    if _pid_alive(pid):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
