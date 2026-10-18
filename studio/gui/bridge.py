"""QProcess bridge from GUI screens to the shared Studio CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QProcess, pyqtSignal

from studio.core.launch import python_command


class RunnerBridge(QObject):
    stage_event = pyqtSignal(dict)
    output = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.cancel_directory: tempfile.TemporaryDirectory[str] | None = None
        self.cancel_file: Path | None = None
        self.stdout_buffer = ""
        self.document_output = ""
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._stdout)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.finished.connect(self._finished)

    def start(self, command: str, source: Path, arguments: list[str] | None = None) -> None:
        if self.process.state() is not QProcess.ProcessState.NotRunning:
            raise RuntimeError("a Studio command is already running")
        self.stdout_buffer = ""
        self.document_output = ""
        self.document_mode = command in {"scan", "plan", "material", "tracks", "cameras"}
        command_line = [command, str(source), *(arguments or [])]
        if command in {"run", "scan", "plan"}:
            self.cancel_directory = tempfile.TemporaryDirectory(prefix="bstudio-run-")
            self.cancel_file = Path(self.cancel_directory.name) / "cancel"
            command_line.extend(["--cancel-file", str(self.cancel_file)])
        if command in {"scan", "plan", "run", "material"}:
            command_line.append("--json")
        executable, *command_line = python_command("studio.cli.main", command_line)
        self.process.start(executable, command_line)

    def cancel(self) -> None:
        if self.cancel_file is not None:
            self.cancel_file.touch()
            self.output.emit("Cancellation requested; waiting for worker cleanup.\n")
            return
        if self.process.state() is not QProcess.ProcessState.NotRunning:
            self.process.terminate()

    def _stdout(self) -> None:
        text = self.process.readAllStandardOutput().data().decode(errors="replace")
        self.output.emit(text)
        if getattr(self, "document_mode", False):
            self.document_output += text
            return
        self.stdout_buffer += text
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                self.stage_event.emit(value)

    def _stderr(self) -> None:
        text = self.process.readAllStandardError().data().decode(errors="replace")
        if text:
            self.failed.emit(text)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._stdout()
        try:
            value = json.loads(
                self.document_output
                if getattr(self, "document_mode", False)
                else self.stdout_buffer
            )
            if isinstance(value, dict):
                self.stage_event.emit(value)
        except ValueError:
            pass
        self.cancel_file = None
        if self.cancel_directory is not None:
            self.cancel_directory.cleanup()
            self.cancel_directory = None
        if status is QProcess.ExitStatus.CrashExit:
            self.failed.emit("Studio worker crashed")
        self.finished.emit(code)
