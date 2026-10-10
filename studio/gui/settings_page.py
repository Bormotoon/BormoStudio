"""Project processing controls over the shared validated settings layer."""

import json
from pathlib import Path
from typing import Any

import yaml
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from studio.core.caption_styles import preset_names
from studio.core.launch import python_command
from studio.core.presets import apply_preset, list_presets
from studio.core.settings import load_settings
from studio.core.settings_store import save_settings

Control = QCheckBox | QComboBox | QSpinBox | QLineEdit
State = tuple[str, tuple[object, ...]]


class SettingsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.work_dir: Path | None = None
        self.controls: dict[tuple[str, str], Control] = {}
        self._snapshot: State | None = None
        self.probe = QProcess(self)
        self.probe.finished.connect(self._compute_finished)
        self.probe.errorOccurred.connect(self._compute_error)
        self.recommendations: dict[str, Any] | None = None
        form = QFormLayout()
        for section, key, label, choices in (
            ("sync", "mode", "Synchronization", ["auto", "camera", "simple", "complex"]),
            ("transcribe", "device", "Whisper device", ["auto", "cpu", "cuda", "mlx"]),
            ("speakers", "method", "Speakers", ["auto", "mics", "pyannote", "off"]),
            ("program", "encoder", "Video encoder", ["auto", "cpu", "nvenc", "videotoolbox"]),
            ("reels", "layout", "Reel layout", ["video", "title", "audiogram"]),
            ("reels", "caption_preset", "Reel caption style", ["", *preset_names()]),
        ):
            combo = QComboBox()
            combo.addItems(choices)
            self.controls[(section, key)] = combo
            form.addRow(label, combo)
            combo.setAccessibleName(label)
        for section, key, label in (
            ("transcribe", "model", "Whisper model"),
            ("text", "base_url", "Local LLM endpoint"),
            ("text", "model", "Local LLM model"),
        ):
            field = QLineEdit()
            self.controls[(section, key)] = field
            form.addRow(label, field)
            field.setAccessibleName(label)
        for section, key, label in (
            ("roughcut", "enabled", "Remove pauses / retakes"),
            ("roughcut", "llm", "Suggest LLM edits (slips, start, delivery)"),
            ("program", "enabled", "Render review master"),
            ("text", "proofread", "Proofread transcript"),
            ("text", "article", "Generate article"),
            ("reels", "enabled", "Select reels"),
            ("reels", "render", "Render reels"),
            ("reels", "burn_subtitles", "Burn reel captions"),
        ):
            check = QCheckBox(label)
            self.controls[(section, key)] = check
            form.addRow(check)
        managed = QCheckBox("Launch owned llama-server for LLM processing")
        self.controls[("llm", "managed")] = managed
        form.addRow(managed)
        for key, label, caption in (
            ("executable", "llama-server executable", "Choose llama-server"),
            ("model_path", "GGUF model file", "Choose GGUF model"),
        ):
            field = QLineEdit()
            field.setAccessibleName(label)
            self.controls[("llm", key)] = field
            row = QHBoxLayout()
            row.addWidget(field)
            browse = QPushButton("Browse")
            browse.setAccessibleName(caption)
            browse.clicked.connect(
                lambda _checked=False, target=field, title=caption: self._browse_llm(target, title)
            )
            row.addWidget(browse)
            form.addRow(label, row)
        port = QSpinBox()
        port.setRange(1, 65535)
        port.setAccessibleName("Managed LLM port")
        self.controls[("llm", "port")] = port
        form.addRow("Managed LLM port", port)
        self.status = QLabel("Choose a material folder first.")
        self.status.setWordWrap(True)
        save = QPushButton("Save processing settings")
        save.clicked.connect(self.save)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        basic = QWidget()
        basic_layout = QVBoxLayout(basic)
        basic_layout.addLayout(form)
        basic_layout.addWidget(save)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(basic)
        tabs.addTab(scroll, "Processing")
        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.addWidget(
            QLabel("Full project YAML. Missing fields use application defaults.")
        )
        self.yaml_editor = QPlainTextEdit()
        self.yaml_editor.setAccessibleName("Advanced project settings YAML")
        advanced_layout.addWidget(self.yaml_editor)
        save_yaml = QPushButton("Validate and save YAML")
        save_yaml.clicked.connect(self.save_yaml)
        advanced_layout.addWidget(save_yaml)
        tabs.addTab(advanced, "Advanced")
        layout.addWidget(tabs)
        layout.addWidget(self.status)
        self.detect_button = QPushButton("Inspect compute backends")
        self.detect_button.clicked.connect(self._inspect_compute)
        self.apply_button = QPushButton("Apply compute recommendations")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply_compute)
        self.tune_button = QPushButton("Measure and tune Whisper / llama")
        self.tune_button.setToolTip(
            "Times real candidates on this machine; takes several minutes and needs the GPU free"
        )
        self.tune_button.clicked.connect(self._tune)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.tune_button)
        layout.addWidget(self.apply_button)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setAccessibleName("Settings preset")
        self.preset_combo.addItems(list_presets())
        self.preset_button = QPushButton("Apply preset")
        self.preset_button.clicked.connect(self._apply_preset)
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(self.preset_button)
        layout.addLayout(preset_row)
        layout.addStretch()

    def load_project(self, work_dir: Path) -> None:
        if self._snapshot is not None and self.dirty and not self.confirm_discard():
            return
        self.recommendations = None
        self.apply_button.setEnabled(False)
        try:
            settings = load_settings([work_dir / "settings.yaml"]).to_dict()
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
            return
        self.work_dir = work_dir
        self.yaml_editor.setPlainText(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
        for (section, key), field in self.controls.items():
            value = settings[section][key]
            if isinstance(field, QCheckBox):
                field.setChecked(value)
            elif isinstance(field, QComboBox):
                field.setCurrentText(value)
            elif isinstance(field, QSpinBox):
                field.setValue(value)
            else:
                field.setText(value)
        self.status.setText("Settings apply to this project. Save before building the plan.")
        self._snapshot = self._state()

    def _state(self) -> State:
        return (
            self.yaml_editor.toPlainText(),
            tuple(
                field.isChecked()
                if isinstance(field, QCheckBox)
                else field.currentText()
                if isinstance(field, QComboBox)
                else field.value()
                if isinstance(field, QSpinBox)
                else field.text()
                for field in self.controls.values()
            ),
        )

    @property
    def dirty(self) -> bool:
        return self._snapshot is not None and self._state() != self._snapshot

    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved settings",
            "Discard unsaved processing or YAML edits?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Discard:
            self._snapshot = None
            return True
        return False

    def save(self) -> bool:
        if self.work_dir is None:
            return False
        if self._snapshot is not None and self.yaml_editor.toPlainText() != self._snapshot[0]:
            self.status.setText(
                "Save Advanced YAML first; processing save would replace YAML edits."
            )
            return False
        changes: dict[str, dict[str, object]] = {}
        for (section, key), field in self.controls.items():
            value = (
                field.isChecked()
                if isinstance(field, QCheckBox)
                else field.currentText()
                if isinstance(field, QComboBox)
                else field.value()
                if isinstance(field, QSpinBox)
                else field.text()
            )
            changes.setdefault(section, {})[key] = value
        # One visible local endpoint serves both text and reels requests.
        changes["reels"].update({key: changes["text"][key] for key in ("base_url", "model")})
        try:
            save_settings(self.work_dir, changes)
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return False
        self.status.setText("Processing settings saved.")
        self.yaml_editor.setPlainText(
            yaml.safe_dump(
                load_settings([self.work_dir / "settings.yaml"]).to_dict(), sort_keys=False
            )
        )
        self._snapshot = self._state()
        return True

    def use_model(self, name: str, path: str) -> None:
        """Fill the matching field after a catalog download; saving stays explicit."""
        from studio.modules.catalog import CATALOG

        entry = CATALOG.get(name)
        if entry is None:
            return
        key = ("transcribe", "model") if entry.kind == "whisper" else ("llm", "model_path")
        field = self.controls[key]
        if isinstance(field, QLineEdit):
            field.setText(entry.whisper_name if entry.kind == "whisper" else path)
        self.status.setText(f"{entry.title} is ready; save settings to use it in this project.")

    def _browse_llm(self, field: QLineEdit, title: str) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            title,
            field.text(),
            "GGUF models (*.gguf)" if "GGUF" in title else "All files (*)",
        )
        if path:
            field.setText(path)

    def _inspect_compute(self) -> None:
        if self.probe.state() != QProcess.ProcessState.NotRunning:
            return
        self.detect_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status.setText("Probing Whisper compute types and actual NVENC encoding...")
        executable, *arguments = python_command("studio.cli.main", ["compute"])
        self.probe.start(executable, arguments)

    def _tune(self) -> None:
        if self.probe.state() != QProcess.ProcessState.NotRunning or self.work_dir is None:
            return
        self.detect_button.setEnabled(False)
        self.tune_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status.setText("Measuring Whisper and llama candidates; this takes a few minutes...")
        executable, *arguments = python_command(
            "studio.cli.main", ["tune", str(self.work_dir.parent)])
        self.probe.start(executable, arguments)

    def _compute_error(self, _error: QProcess.ProcessError) -> None:
        self.detect_button.setEnabled(True)
        self.tune_button.setEnabled(True)
        self.status.setText(self.probe.errorString())

    def _compute_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self.detect_button.setEnabled(True)
        self.tune_button.setEnabled(True)
        try:
            if code:
                raise ValueError(
                    self.probe.readAllStandardError().data().decode(errors="replace")
                )
            report = json.loads(self.probe.readAllStandardOutput().data())
            self.recommendations = report["recommendations"]
            self.status.setText(json.dumps(report, indent=2))
            self.apply_button.setEnabled(self.work_dir is not None)
        except (ValueError, KeyError) as exc:
            self.status.setText(str(exc))

    def _apply_preset(self) -> None:
        if self.work_dir is None or not self.confirm_discard():
            return
        name = self.preset_combo.currentText()
        try:
            apply_preset(self.work_dir, name)
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return
        self._snapshot = None
        self.load_project(self.work_dir)
        self.status.setText(f"Preset {name} saved; other settings are unchanged.")

    def _apply_compute(self) -> None:
        if self.work_dir is None or self.recommendations is None:
            return
        if not self.confirm_discard():
            return
        try:
            save_settings(self.work_dir, self.recommendations)
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return
        self._snapshot = None
        self.load_project(self.work_dir)
        self.status.setText("Compute recommendations saved.")

    def save_yaml(self) -> bool:
        if self.work_dir is None:
            return False
        try:
            data = yaml.safe_load(self.yaml_editor.toPlainText())
            if not isinstance(data, dict):
                raise ValueError("Settings YAML must be a mapping.")
            save_settings(self.work_dir, data, replace=True)
        except (ValueError, OSError, RuntimeError, yaml.YAMLError) as exc:
            self.status.setText(str(exc))
            return False
        self._snapshot = None
        self.load_project(self.work_dir)
        self.status.setText("Advanced settings validated and saved.")
        return True
