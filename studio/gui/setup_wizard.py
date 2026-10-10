"""Explicit project setup using native Qt wizard pages."""

import shutil
from pathlib import Path

from PyQt6.QtCore import QProcess
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
from studio.gui.modules import ModulesPage
from studio.modules.catalog import CATALOG
from studio.modules.models import ModelManager


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
        self.addPage(self._models_page())
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
            index = self.whisper.findData(settings.transcribe.model)
            if index >= 0:
                self.whisper.setCurrentIndex(index)
            self.sync.setCurrentText(settings.sync.mode)
            self.program.setChecked(settings.program.enabled)
            self.proofread.setChecked(settings.text.proofread)
            self.reels.setChecked(settings.reels.enabled)
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def _models_page(self) -> QWizardPage:
        page = QWizardPage()
        page.setTitle("Models")
        layout = QVBoxLayout(page)
        manager = ModelManager()
        self.installed_models = {
            name for name, entry in CATALOG.items() if manager.installed(name)
        }
        ready = [CATALOG[name].title for name in sorted(self.installed_models)]
        summary = QLabel(
            "Already downloaded: " + (", ".join(ready) if ready else "none") + ".\n"
            "Transcription needs one Whisper model; text and reels need a GGUF model "
            "or another local LLM endpoint. Downloads resume and are checksum-verified; "
            "you can also skip this and download later from Modules and models."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        self.whisper = QComboBox()
        self.whisper.setAccessibleName("Whisper model for this project")
        for entry in CATALOG.values():
            if entry.kind == "whisper" and entry.name != "whisper-tiny":
                mark = " — downloaded" if entry.name in self.installed_models else ""
                self.whisper.addItem(entry.title + mark, entry.whisper_name)
        layout.addWidget(QLabel("Whisper model for this project"))
        layout.addWidget(self.whisper)
        self.downloads = ModulesPage()
        index = self.downloads.model.findData("whisper-large-v3-turbo")
        if index >= 0:
            self.downloads.model.setCurrentIndex(index)
        self.downloads.model_installed.connect(self._model_ready)
        layout.addWidget(self.downloads)
        self.llm_model_path: str | None = None
        return page

    def _model_ready(self, name: str, path: str) -> None:
        entry = CATALOG.get(name)
        if entry is None:
            return
        self.installed_models.add(name)
        if entry.kind == "whisper":
            index = self.whisper.findData(entry.whisper_name)
            if index >= 0:
                self.whisper.setItemText(index, entry.title + " — downloaded")
                self.whisper.setCurrentIndex(index)
        else:
            self.llm_model_path = path
        self.downloads.status.setText(f"{entry.title} is ready; Finish saves it for this project.")

    def _stop_downloads(self) -> None:
        if self.downloads.process.state() != QProcess.ProcessState.NotRunning:
            self.downloads.cancel()
            # Cancellation keeps the partial file for a later resume.
            if not self.downloads.process.waitForFinished(10000):
                self.downloads.process.kill()
                self.downloads.process.waitForFinished(3000)

    def reject(self) -> None:
        self._stop_downloads()
        super().reject()

    def accept(self) -> None:
        if self.downloads.process.state() != QProcess.ProcessState.NotRunning:
            self.status.setText("Wait for the download to finish or cancel it first.")
            return
        changes: dict[str, dict[str, object]] = {
            "transcribe": {"model": str(self.whisper.currentData())},
        }
        if self.llm_model_path:
            changes["llm"] = {"model_path": self.llm_model_path}
        try:
            save_settings(
                self.work_dir,
                {
                    **changes,
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
