"""Explicit project setup using native Qt wizard pages."""

import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from studio.core.settings import load_settings
from studio.core.settings_store import save_settings


class SetupWizard(QWizard):
    def __init__(self, work_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.work_dir = work_dir
        self.setWindowTitle("Set up processing")
        self.resize(600, 450)
        environment = QWizardPage()
        environment.setTitle("Local tools")
        layout = QVBoxLayout(environment)
        tools = "\n".join(
            f"{name}: {shutil.which(name) or 'not found'}" for name in ("ffmpeg", "ffprobe")
        )
        layout.addWidget(QLabel(tools))
        note = QLabel(
            "Whisper requires the transcription dependency. Optional vision/diarization "
            "modules and models are installed from Modules and models. "
            "Local LLM outputs require a running llama endpoint."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.addPage(environment)
        choices = QWizardPage()
        choices.setTitle("Choose outputs")
        layout = QVBoxLayout(choices)
        self.sync = QComboBox()
        self.sync.addItems(["auto", "camera", "simple", "complex"])
        self.sync.setAccessibleName("Synchronization mode")
        layout.addWidget(QLabel("Synchronization mode"))
        layout.addWidget(self.sync)
        self.program = QCheckBox("Render review master")
        self.proofread = QCheckBox("Proofread transcript using local LLM")
        self.reels = QCheckBox("Select and render reels using local LLM")
        for control in (self.program, self.proofread, self.reels):
            layout.addWidget(control)
        self.status = QLabel("Finish saves these choices for the current project.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.addPage(choices)
        try:
            settings = load_settings([work_dir / "settings.yaml"])
            self.sync.setCurrentText(settings.sync.mode)
            self.program.setChecked(settings.program.enabled)
            self.proofread.setChecked(settings.text.proofread)
            self.reels.setChecked(settings.reels.enabled)
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def accept(self) -> None:
        try:
            save_settings(
                self.work_dir,
                {
                    "sync": {"mode": self.sync.currentText()},
                    "program": {"enabled": self.program.isChecked()},
                    "text": {"proofread": self.proofread.isChecked()},
                    "reels": {"enabled": self.reels.isChecked(), "render": self.reels.isChecked()},
                },
            )
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return
        super().accept()
