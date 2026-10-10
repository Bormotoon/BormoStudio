"""Cooperative installation cancellation and owned child cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable


class InstallationCancelled(RuntimeError):
    pass


def check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise InstallationCancelled("Installation cancelled.")


def run_install(command: list[str], cancelled: Callable[[], bool] | None = None) -> None:
    if cancelled is None:
        subprocess.run(command, check=True)
        return
    check_cancelled(cancelled)
    process = subprocess.Popen(command, start_new_session=os.name != "nt")
    try:
        while process.poll() is None:
            check_cancelled(cancelled)
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        check_cancelled(cancelled)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        _terminate(process)


def run_cancellable(
    command: list[str],
    cancelled: Callable[[], bool],
    *,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run with captured text output, stopping the process tree on cancellation.

    Output goes to temporary files rather than pipes, so a chatty child can
    never block on a full pipe while the parent polls for cancellation.
    """
    check_cancelled(cancelled)
    deadline = None if timeout is None else time.monotonic() + timeout
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as stdout, \
            tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as stderr:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, text=True,
            start_new_session=os.name != "nt",
        )
        try:
            while process.poll() is None:
                check_cancelled(cancelled)
                if deadline is not None and time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(command, timeout or 0)
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            _terminate(process)
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(
            command, process.returncode, stdout.read(), stderr.read()
        )


def _terminate(process: subprocess.Popen) -> None:  # type: ignore[type-arg]
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
