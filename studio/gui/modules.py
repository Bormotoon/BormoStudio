"""Desktop installation controls using asynchronous CLI subprocesses."""

import json
import tempfile
from pathlib import Path

from PyQt6.QtCore import QProcess, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from studio.core.launch import python_command


class ModulesPage(QWidget):
    model_installed = pyqtSignal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.process = QProcess(self)
        self.buffer = ""
        self.installed_result: dict | None = None
        self.cancel_directory = None
        self.cancel_file = None
        self.cancel_button = QPushButton("Cancel installation")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.progress.setAccessibleName("Installation progress")
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(lambda _error: self._error())
        self.module = QComboBox()
        self.module.addItems(["vision", "diarization", "youtube"])
        self.module.setAccessibleName("Optional module")
        from studio.modules.catalog import CATALOG

        self.model = QComboBox()
        self.model.addItem("YuNet face detector (vision)", "yunet")
        self.model.addItem("Light-ASD active speaker (vision)", "light-asd")
        for entry in CATALOG.values():
            if entry.name != "whisper-tiny":
                self.model.addItem(entry.title, entry.name)
        self.model.setAccessibleName("Model to download")
        install = QPushButton("Install selected module")
        install.clicked.connect(lambda: self.start("modules", "install", self.module.currentText()))
        download = QPushButton("Download selected model")
        download.clicked.connect(
            lambda: self.start("models", "install", str(self.model.currentData()))
        )
        inspect = QPushButton("Show installed modules")
        inspect.clicked.connect(lambda: self.start("modules", "list"))
        models = QPushButton("Check installed models")
        models.clicked.connect(lambda: self.start("models", "list"))
        self.actions = [install, download, inspect, models]
        self.status = QLabel(
            "Download Whisper and Gemma GGUF models here; downloads resume after "
            "cancellation and are verified against pinned checksums. Install vision, "
            "then YuNet and Light-ASD for automatic framing."
        )
        self.status.setWordWrap(True)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setAccessibleName("Installation log")
        layout = QVBoxLayout(self)
        for widget in (
            self.status,
            self.progress,
            self.cancel_button,
            self.module,
            install,
            self.model,
            download,
            inspect,
            models,
            self.log,
        ):
            layout.addWidget(widget)

    def start(self, command: str, action: str, name: str | None = None) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        self.log.clear()
        self.buffer = ""
        self.installed_result = None
        self.cancel_directory = tempfile.TemporaryDirectory(prefix="uvf-install-")
        self.cancel_file = Path(self.cancel_directory.name) / "cancel"
        self.cancel_button.setEnabled(action == "install")
        self.progress.setRange(0, 0)
        self.status.setText(f"Running {command} {action} {name or ''}...")
        for button in self.actions:
            button.setEnabled(False)
        arguments = [
            command,
            action,
            *([name] if name else []),
            "--events",
            "--cancel-file",
            str(self.cancel_file),
        ]
        executable, *arguments = python_command("studio.cli.main", arguments)
        self.process.start(executable, arguments)

    def _output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self.log.insertPlainText(text)
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and {"name", "path"} <= event.keys() and "type" not in event:
                self.installed_result = event
                continue
            if not isinstance(event, dict) or event.get("type") != "installation_progress":
                continue
            percent = event.get("percent")
            self.progress.setRange(0, 100 if percent is not None else 0)
            if percent is not None:
                self.progress.setValue(percent)
            self.status.setText(_describe(event))

    def _finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self._output()
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if code == 0 else 0)
        self.status.setText(
            "Completed."
            if code == 0
            else "Installation cancelled."
            if code == 130
            else "Installation failed; see log."
        )
        if code == 0 and self.installed_result is not None:
            self.model_installed.emit(
                str(self.installed_result["name"]), str(self.installed_result["path"])
            )
            self.status.setText(f"Completed: {self.installed_result['path']}")
        self._cleanup_cancel()
        for button in self.actions:
            button.setEnabled(True)

    def _error(self) -> None:
        self.status.setText(self.process.errorString())
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._cleanup_cancel()
        for button in self.actions:
            button.setEnabled(True)

    def cancel(self) -> None:
        if self.cancel_file is not None:
            self.cancel_file.touch()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling; waiting for the current operation to stop...")

    def _cleanup_cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        self.cancel_file = None
        if self.cancel_directory is not None:
            self.cancel_directory.cleanup()
            self.cancel_directory = None


def _describe(event: dict) -> str:
    text = f"{event.get('name', '')}: {event.get('phase', '')}"
    done, total = event.get("bytes"), event.get("total")
    if isinstance(done, int) and isinstance(total, int) and total:
        text += f" {done / 1e9:.2f} of {total / 1e9:.2f} GB"
    return text
