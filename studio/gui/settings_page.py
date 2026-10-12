"""Project processing controls over the shared validated settings layer.

The basic mode offers :class:`BasicOptions`, a handful of choices; everything
else keeps its project value or default. The advanced mode shows
:class:`SettingsPage`, a form with every setting, built from
:mod:`studio.gui.settings_schema` and the typed settings.
"""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml
from PyQt6.QtCore import QProcess, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from studio.core import caption_styles
from studio.core.launch import python_command
from studio.core.presets import apply_preset, list_presets
from studio.core.settings import CHOICES, Settings, load_settings
from studio.core.settings_store import save_settings
from studio.gui.settings_schema import LANGUAGES, LIST_CHOICES, RANGES, SECTIONS

State = tuple[str, tuple[object, ...]]
# Path fields get a Browse button: (dialog title, file filter).
BROWSE = {
    "llm.executable": ("Choose llama-server", "All files (*)"),
    "llm.model_path": ("Choose GGUF model", "GGUF models (*.gguf)"),
    "fetch.cookies_file": ("Choose cookies file", "All files (*)"),
}
# The text LLM endpoint carries these along while they hold the same value.
FOLLOWERS = {
    "text.base_url": ("roughcut.base_url", "reels.base_url"),
    "text.model": ("roughcut.model", "reels.model"),
}


def _choices() -> dict[str, tuple[str, ...]]:
    """Every enumerated text setting; empty caption choices keep the style's own."""
    return {
        **CHOICES,
        "reels.caption_preset": ("", *caption_styles.preset_names()),
        "reels.caption_highlight": ("", *sorted(caption_styles.HIGHLIGHTS)),
        "reels.caption_case": ("", *sorted(caption_styles.CASES)),
        "reels.caption_punctuation": ("", *sorted(caption_styles.PUNCTUATION)),
        "reels.caption_censor_style": tuple(sorted(caption_styles.CENSOR_STYLES)),
    }


class ChoiceList(QWidget):
    """Check boxes for a list of known values, such as the export formats."""

    changed = pyqtSignal()

    def __init__(self, choices: tuple[str, ...]) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.boxes = {choice: QCheckBox(choice) for choice in choices}
        for box in self.boxes.values():
            box.toggled.connect(self.changed)
            layout.addWidget(box)
        layout.addStretch()

    def value(self) -> list[str]:
        return [choice for choice, box in self.boxes.items() if box.isChecked()]

    def setValue(self, value: list[str]) -> None:  # noqa: N802 - Qt naming
        for choice, box in self.boxes.items():
            box.setChecked(choice in value)


class Number(QDoubleSpinBox):
    """A decimal field that shows only the digits its value needs."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt naming
        locale = self.locale()
        text = locale.toString(value, "f", self.decimals()).replace(locale.groupSeparator(), "")
        point = locale.decimalPoint()
        return text.rstrip("0").rstrip(point) if point in text else text


@dataclass(slots=True)
class Field:
    """One setting's widget; ``raw`` is the widget state, ``get`` the setting value."""

    widget: QWidget
    raw: Callable[[], object]
    get: Callable[[], object]
    set: Callable[[Any], None]
    changed: Any


def _optional_int(text: str) -> int | str | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _yaml_value(text: str, empty: object) -> object:
    if not text.strip():
        return empty
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid value {text!r}: {exc}") from exc


def make_field(dotted: str, kind: Any, default: object,
               choices: dict[str, tuple[str, ...]]) -> Field:
    """A widget matching the setting's type and allowed values."""
    args = set(get_args(kind)) if get_origin(kind) in (Union, types.UnionType) else set()
    if kind is bool:
        box = QCheckBox()
        return Field(box, box.isChecked, box.isChecked,
                     lambda value: box.setChecked(bool(value)), box.toggled)
    if dotted in choices:
        combo = QComboBox()
        combo.addItems(choices[dotted])

        def choose(value: Any) -> None:
            # A value the list lacks (set in YAML) is shown rather than lost.
            if combo.findText(str(value)) < 0:
                combo.addItem(str(value))
            combo.setCurrentText(str(value))
        return Field(combo, combo.currentText, combo.currentText, choose,
                     combo.currentTextChanged)
    if dotted in LIST_CHOICES:
        boxes = ChoiceList(LIST_CHOICES[dotted])
        return Field(boxes, lambda: tuple(boxes.value()), boxes.value,
                     lambda value: boxes.setValue(list(value or [])), boxes.changed)
    if kind is int or args == {int, type(None)}:
        spin = QSpinBox()
        low, high = RANGES.get(dotted, (-1_000_000_000, 1_000_000_000))
        optional = kind is not int
        if optional:
            # The lowest value stands for "not set": the server's own default.
            spin.setRange(-1, int(high))
            spin.setSpecialValueText("default")
        else:
            spin.setRange(int(low), int(high))
        return Field(spin, spin.value,
                     (lambda: None if spin.value() == -1 else spin.value()) if optional
                     else spin.value,
                     lambda value: spin.setValue(-1 if value is None else int(value)),
                     spin.valueChanged)
    if kind is float:
        number = Number()
        number.setDecimals(6)
        bottom, top = RANGES.get(dotted, (-1_000_000_000.0, 1_000_000_000.0))
        number.setRange(bottom, top)
        number.setSingleStep(1.0 if abs(float(default)) >= 10 else 0.1)  # type: ignore[arg-type]
        return Field(number, number.value, number.value,
                     lambda value: number.setValue(float(value)), number.valueChanged)
    line = QLineEdit()
    if kind is str:
        return Field(line, line.text, line.text, lambda value: line.setText(str(value)),
                     line.textChanged)
    if args == {str, type(None)}:
        line.setPlaceholderText("detect automatically")
        return Field(line, line.text, lambda: line.text().strip() or None,
                     lambda value: line.setText(value or ""), line.textChanged)
    if args == {int, str, type(None)}:
        line.setPlaceholderText("default")
        return Field(line, line.text, lambda: _optional_int(line.text()),
                     lambda value: line.setText("" if value is None else str(value)),
                     line.textChanged)
    # Lists and mappings are edited as one line of YAML: [a, b] or {key: value}.
    empty: object = [] if get_origin(kind) is list else {}
    line.setPlaceholderText("[]" if empty == [] else "{}")

    def show(value: Any) -> None:
        line.setText(yaml.safe_dump(value, default_flow_style=True, allow_unicode=True,
                                    width=10_000).strip() if value else "")
    return Field(line, line.text, lambda: _yaml_value(line.text(), empty), show,
                 line.textChanged)


class SettingsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.work_dir: Path | None = None
        self.fields: dict[tuple[str, str], Field] = {}
        self.controls: dict[tuple[str, str], QWidget] = {}
        self.labels: dict[tuple[str, str], QLabel] = {}
        self.defaults: dict[tuple[str, str], object] = {}
        self._initial: dict[tuple[str, str], object] = {}
        self._previous: dict[str, str] = {}
        self._loading = False
        self._snapshot: State | None = None
        self.probe = QProcess(self)
        self.probe.finished.connect(self._compute_finished)
        self.probe.errorOccurred.connect(self._compute_error)
        self.recommendations: dict[str, Any] | None = None
        self.status = QLabel("Choose a material folder first.")
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._form(), "All settings")
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
        tabs.addTab(advanced, "YAML")
        layout.addWidget(tabs, 1)
        layout.addWidget(self.status)
        compute = QHBoxLayout()
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
        for button in (self.detect_button, self.tune_button, self.apply_button):
            compute.addWidget(button)
        layout.addLayout(compute)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setAccessibleName("Settings preset")
        self.preset_combo.addItems(list_presets())
        self.preset_button = QPushButton("Apply preset")
        self.preset_button.clicked.connect(self._apply_preset)
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(self.preset_button)
        layout.addLayout(preset_row)

    def _form(self) -> QWidget:
        """Sections on the left, their settings in groups on the right."""
        defaults = Settings()
        choices = _choices()
        page = QWidget()
        layout = QVBoxLayout(page)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter settings by name or key")
        self.filter.setAccessibleName("Filter settings")
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)
        body = QHBoxLayout()
        self.sections = QListWidget()
        self.sections.setMaximumWidth(200)
        self.section_pages = QStackedWidget()
        self.sections.currentRowChanged.connect(self.section_pages.setCurrentIndex)
        self._rows: list[tuple[int, QGroupBox, QFormLayout, list[tuple[str, QWidget]]]] = []
        for index, section in enumerate(SECTIONS):
            values = getattr(defaults, section.name)
            hints = get_type_hints(type(values))
            self.sections.addItem(section.title)
            content = QWidget()
            column = QVBoxLayout(content)
            for group in section.groups:
                box = QGroupBox(group.title)
                form = QFormLayout(box)
                rows: list[tuple[str, QWidget]] = []
                for item in group.items:
                    dotted = f"{section.name}.{item.key}"
                    key = (section.name, item.key)
                    default = getattr(values, item.key)
                    field = make_field(dotted, hints[item.key], default, choices)
                    self.fields[key] = field
                    self.controls[key] = field.widget
                    self.defaults[key] = default
                    field.widget.setAccessibleName(item.label)
                    label = QLabel(item.label)
                    label.setToolTip("\n\n".join(filter(None, (
                        item.help, f"{dotted} · default: {_shown(default)}"))))
                    field.widget.setToolTip(label.toolTip())
                    self.labels[key] = label
                    field.changed.connect(
                        lambda *_args, target=key: self._field_changed(target))
                    if dotted in BROWSE:
                        title, pattern = BROWSE[dotted]
                        holder = QWidget()
                        line = QHBoxLayout(holder)
                        line.setContentsMargins(0, 0, 0, 0)
                        line.addWidget(field.widget)
                        browse = QPushButton("Browse")
                        browse.setAccessibleName(title)
                        browse.clicked.connect(
                            lambda _checked=False, target=field.widget, caption=title,
                            kinds=pattern: self._browse(target, caption, kinds))
                        line.addWidget(browse)
                        form.addRow(label, holder)
                        rows.append((f"{dotted} {item.label} {item.help}".casefold(), holder))
                    else:
                        form.addRow(label, field.widget)
                        rows.append((f"{dotted} {item.label} {item.help}".casefold(),
                                     field.widget))
                column.addWidget(box)
                self._rows.append((index, box, form, rows))
            column.addStretch()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content)
            self.section_pages.addWidget(scroll)
        missing = {f"{section.name}.{item.name}" for section in fields(Settings)
                   for item in fields(getattr(defaults, section.name))} - {
            f"{section}.{key}" for section, key in self.fields}
        if missing:
            raise RuntimeError(f"settings without a place in the form: {sorted(missing)}")
        for key, field in self.fields.items():
            field.set(self.defaults[key])
        self.sections.setCurrentRow(0)
        body.addWidget(self.sections)
        body.addWidget(self.section_pages, 1)
        layout.addLayout(body, 1)
        hint = QLabel("Bold names differ from the defaults. Hover a name for its "
                      "meaning and settings key.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        save = QPushButton("Save processing settings")
        save.clicked.connect(self.save)
        layout.addWidget(save)
        return page

    def _apply_filter(self, text: str) -> None:
        query = text.strip().casefold()
        shown: set[int] = set()
        for index, box, form, rows in self._rows:
            visible = 0
            for haystack, widget in rows:
                match = not query or query in haystack
                form.setRowVisible(widget, match)
                visible += match
            box.setVisible(bool(visible))
            if visible:
                shown.add(index)
        for index in range(self.sections.count()):
            item = self.sections.item(index)
            if item is not None:
                item.setHidden(index not in shown)
        if self.sections.currentRow() not in shown and shown:
            self.sections.setCurrentRow(min(shown))

    def _field_changed(self, key: tuple[str, str]) -> None:
        self._mark(key)
        dotted = ".".join(key)
        if dotted not in FOLLOWERS:
            return
        field = self.controls[key]
        assert isinstance(field, QLineEdit)
        previous = self._previous.get(dotted)
        self._previous[dotted] = field.text()
        if self._loading or previous is None:
            return
        for follower in FOLLOWERS[dotted]:
            target = self.controls[tuple(follower.split("."))]  # type: ignore[index]
            if isinstance(target, QLineEdit) and target.text() == previous:
                target.setText(field.text())

    def _mark(self, key: tuple[str, str]) -> None:
        try:
            changed = self.fields[key].get() != self.defaults[key]
        except ValueError:
            changed = True
        font = QFont(self.labels[key].font())
        font.setBold(changed)
        self.labels[key].setFont(font)

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
        self._loading = True
        try:
            for (section, key), field in self.fields.items():
                field.set(settings[section][key])
        finally:
            self._loading = False
        self._initial = {target: field.raw() for target, field in self.fields.items()}
        for target in self.fields:
            self._mark(target)
        self.status.setText("Settings apply to this project. Save before building the plan.")
        self._snapshot = self._state()

    def _state(self) -> State:
        return (
            self.yaml_editor.toPlainText(),
            tuple(field.raw() for field in self.fields.values()),
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
        # Only edited fields are written: untouched values stay exactly as stored.
        changes: dict[str, dict[str, object]] = {}
        try:
            for (section, key), field in self.fields.items():
                if field.raw() != self._initial.get((section, key)):
                    changes.setdefault(section, {})[key] = field.get()
            save_settings(self.work_dir, changes)
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return False
        self._snapshot = None
        self.load_project(self.work_dir)
        self.status.setText("Processing settings saved.")
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

    def _browse(self, field: QWidget, title: str, pattern: str) -> None:
        if not isinstance(field, QLineEdit):
            return
        path, _filter = QFileDialog.getOpenFileName(self, title, field.text(), pattern)
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


def _shown(value: object) -> str:
    if value is None:
        return "not set"
    if value == "":
        return "empty"
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    return str(value)


class BasicOptions(QWidget):
    """The basic mode's few choices; every other setting keeps its value."""

    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        self.language = QComboBox()
        self.language.setAccessibleName("Speech language")
        for title, code in LANGUAGES:
            self.language.addItem(title, code)
        form.addRow("Speech language", self.language)
        self.roughcut = QCheckBox("Remove pauses and retakes")
        self.markers = QCheckBox("Mark cuts in Final Cut / Premiere")
        self.review = QCheckBox("Render a review video")
        self.review.setToolTip("A finished MP4 of the edit with camera switches; slower.")
        for box in (self.roughcut, self.markers, self.review):
            form.addRow(box)
        self._markers = "cuts"
        self._loaded: tuple[object, ...] | None = None

    def load(self, work_dir: Path) -> None:
        settings = load_settings([work_dir / "settings.yaml"])
        language = settings.transcribe.language
        index = self.language.findData(language)
        if index < 0:
            # A language set in the advanced mode stays selectable as it is.
            self.language.addItem(str(language), language)
            index = self.language.count() - 1
        self.language.setCurrentIndex(index)
        self.roughcut.setChecked(settings.roughcut.enabled)
        self._markers = settings.export.markers
        self.markers.setChecked(self._markers != "none")
        self.review.setChecked(settings.program.enabled)
        self._loaded = self._state()

    def _state(self) -> tuple[object, ...]:
        return (self.language.currentData(), self.roughcut.isChecked(),
                self.markers.isChecked(), self.review.isChecked())

    @property
    def dirty(self) -> bool:
        return self._loaded is not None and self._state() != self._loaded

    def changes(self) -> dict[str, dict[str, object]]:
        # Checked keeps a wider advanced choice (all markers) as it is.
        markers = (self._markers if self._markers != "none" else "cuts") \
            if self.markers.isChecked() else "none"
        return {
            "transcribe": {"language": self.language.currentData()},
            "roughcut": {"enabled": self.roughcut.isChecked()},
            "export": {"markers": markers},
            "program": {"enabled": self.review.isChecked()},
        }

    def save(self, work_dir: Path) -> None:
        save_settings(work_dir, self.changes())
        self.load(work_dir)
