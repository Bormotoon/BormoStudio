import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from studio.core.settings import load_settings
from studio.gui.settings_page import SettingsPage
from studio.gui.setup_wizard import SetupWizard


def test_dirty_settings_cancel_preserves_edits(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    app = QApplication.instance() or QApplication([])
    page = SettingsPage()
    page.load_project(tmp_path)
    assert not page.dirty
    page.controls[("text", "model")].setText("unsaved")
    assert page.dirty
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel)
    page.load_project(tmp_path / "other")
    assert page.work_dir == tmp_path
    assert page.controls[("text", "model")].text() == "unsaved"
    assert page.save()
    assert not page.dirty
    page.close()
    app.processEvents()


def test_desktop_settings_preserve_hidden_options(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "settings.yaml"
    path.write_text("reels:\n  crop_x: 0.8\n")
    page = SettingsPage()
    page.load_project(tmp_path)
    page.controls[("program", "enabled")].setChecked(True)
    page.controls[("text", "base_url")].setText("http://localhost:9000")
    assert page.save()
    settings = load_settings([path])
    assert settings.program.enabled
    assert settings.reels.crop_x == 0.8
    assert settings.reels.base_url == "http://localhost:9000"
    page.controls[("llm", "managed")].setChecked(True)
    page.controls[("llm", "model_path")].setText(str(tmp_path / "test.gguf"))
    page.controls[("llm", "port")].setValue(9090)
    assert page.save()
    assert load_settings([path]).llm.port == 9090
    before = path.read_bytes()
    page.controls[("text", "base_url")].setText("")
    assert not page.save()
    assert path.read_bytes() == before
    page.yaml_editor.setPlainText("sync:\n  mode: camera\n")
    assert page.save_yaml()
    assert load_settings([path]).sync.mode == "camera"
    page.yaml_editor.setPlainText("unknown: true")
    before = path.read_bytes()
    assert not page.save_yaml()
    assert path.read_bytes() == before
    page.close()
    app.processEvents()


def test_setup_wizard_persists_selected_outputs(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    wizard = SetupWizard(tmp_path)
    wizard.sync.setCurrentText("camera")
    wizard.program.setChecked(True)
    wizard.accept()
    settings = load_settings([tmp_path / "settings.yaml"])
    assert settings.sync.mode == "camera"
    assert settings.program.enabled
    assert not settings.reels.enabled
    app.processEvents()


def test_downloaded_models_fill_settings_and_modules_list_catalog(tmp_path):
    from studio.gui.modules import ModulesPage, _describe

    app = QApplication.instance() or QApplication([])
    page = SettingsPage()
    page.load_project(tmp_path)
    page.use_model("whisper-large-v3-turbo", str(tmp_path / "w"))
    assert page.controls[("transcribe", "model")].text() == "large-v3-turbo"
    page.use_model("gguf-gemma-4-e4b-it-q4", str(tmp_path / "m.gguf"))
    assert page.controls[("llm", "model_path")].text() == str(tmp_path / "m.gguf")
    assert page.dirty
    modules = ModulesPage()
    names = [modules.model.itemData(index) for index in range(modules.model.count())]
    assert names[:2] == ["yunet", "light-asd"] and "whisper-large-v3" in names
    assert _describe({"name": "x", "phase": "download", "bytes": 10**9, "total": 4 * 10**9}) == (
        "x: download 1.00 of 4.00 GB"
    )
    assert app is not None


def test_settings_page_applies_preset(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    page = SettingsPage()
    page.load_project(tmp_path)
    page.preset_combo.setCurrentText("roughcut-tight")
    page.preset_button.click()
    assert load_settings([tmp_path / "settings.yaml"]).roughcut.pause_min_s == 0.7
    assert not page.dirty
    assert "roughcut-tight" in page.status.text()
    page.close()
    app.processEvents()


def test_wizard_downloads_models_into_project_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UVF_MODELS_DIR", str(tmp_path / "models"))
    app = QApplication.instance() or QApplication([])
    wizard = SetupWizard(tmp_path)
    assert wizard.downloads.model.currentData() == "whisper-large-v3-turbo"
    wizard.downloads.model_installed.emit("whisper-large-v3-turbo", str(tmp_path / "w"))
    wizard.downloads.model_installed.emit("gguf-gemma-4-e4b-it-q4", str(tmp_path / "g.gguf"))
    assert wizard.whisper.currentData() == "large-v3-turbo"
    wizard.accept()
    settings = load_settings([tmp_path / "settings.yaml"])
    assert settings.transcribe.model == "large-v3-turbo"
    assert settings.llm.model_path == str(tmp_path / "g.gguf")
    app.processEvents()


def test_every_setting_has_a_described_place_in_the_form(tmp_path: Path) -> None:
    from dataclasses import fields

    from PyQt6.QtWidgets import QComboBox

    from studio.core.settings import CHOICES, Settings
    from studio.gui.settings_schema import SECTIONS, items

    app = QApplication.instance() or QApplication([])
    keys = [f"{section.name}.{item.key}" for section in SECTIONS
            for group in section.groups for item in group.items]
    assert len(keys) == len(set(keys))
    expected = {f"{section.name}.{item.name}" for section in fields(Settings)
                for item in fields(getattr(Settings(), section.name))}
    assert set(keys) == expected
    assert all(item.label.strip() for item in items().values())
    page = SettingsPage()
    assert {".".join(key) for key in page.controls} == expected
    for dotted, choices in CHOICES.items():
        combo = page.controls[tuple(dotted.split("."))]
        assert isinstance(combo, QComboBox)
        assert [combo.itemText(index) for index in range(combo.count())] == list(choices)
    page.close()
    app.processEvents()


def test_form_saves_only_edited_fields_with_their_types(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "settings.yaml"
    # More decimals than the form shows, and a list it edits as YAML.
    path.write_text("roughcut:\n  silence_threshold_db: -40.123456\n"
                    "transcribe:\n  glossary: [Иван, 'yes']\n", encoding="utf-8")
    page = SettingsPage()
    page.load_project(tmp_path)
    assert page.controls[("transcribe", "glossary")].text() == "[Иван, 'yes']"
    page.controls[("export", "markers")].setCurrentText("none")
    page.controls[("export", "marker_min_s")].setValue(1.5)
    page.controls[("export", "targets")].boxes["xmeml"].setChecked(False)
    page.controls[("export", "name")].setText("Interview")
    page.controls[("transcribe", "language")].setText("")
    page.controls[("llm", "n_gpu_layers")].setValue(99)
    page.controls[("llm", "cache_ram_mb")].setText("auto")
    page.controls[("sync", "session_offsets")].setText("{cam-2: 120.5}")
    assert page.dirty
    assert page.save(), page.status.text()
    settings = load_settings([path])
    assert settings.export.markers == "none"
    assert settings.export.marker_min_s == 1.5
    assert settings.export.targets == ["multicam"]
    assert settings.export.name == "Interview"
    assert settings.transcribe.language is None
    assert settings.llm.n_gpu_layers == 99 and settings.llm.ctx_size is None
    assert settings.llm.cache_ram_mb == "auto"
    assert settings.sync.session_offsets == {"cam-2": 120.5}
    assert settings.roughcut.silence_threshold_db == -40.123456
    assert settings.transcribe.glossary == ["Иван", "yes"]
    # Untouched sections stay out of the file.
    assert "reels" not in path.read_text(encoding="utf-8")
    assert not page.dirty
    page.controls[("sync", "session_offsets")].setText("{unclosed")
    before = path.read_bytes()
    assert not page.save()
    assert path.read_bytes() == before
    page.close()
    app.processEvents()


def test_form_filter_and_changed_names(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    page = SettingsPage()
    page.load_project(tmp_path)
    label = page.labels[("export", "marker_min_s")]
    assert not label.font().bold()
    page.controls[("export", "marker_min_s")].setValue(2.0)
    assert label.font().bold()
    page.filter.setText("marker_min")
    export = next(index for index in range(page.sections.count())
                  if page.sections.item(index).text() == "Final Cut / Premiere")
    assert page.sections.currentRow() == export
    assert [index for index in range(page.sections.count())
            if not page.sections.item(index).isHidden()] == [export]
    page.filter.setText("")
    assert not any(page.sections.item(index).isHidden()
                   for index in range(page.sections.count()))
    page.close()
    app.processEvents()


def test_basic_options_keep_advanced_choices(tmp_path: Path) -> None:
    from studio.gui.settings_page import BasicOptions

    app = QApplication.instance() or QApplication([])
    options = BasicOptions()
    options.load(tmp_path)
    assert options.language.currentData() == "ru"
    assert options.roughcut.isChecked() and options.markers.isChecked()
    assert not options.review.isChecked() and not options.dirty
    options.markers.setChecked(False)
    options.review.setChecked(True)
    options.language.setCurrentIndex(options.language.findData("en"))
    assert options.dirty
    options.save(tmp_path)
    settings = load_settings([tmp_path / "settings.yaml"])
    assert settings.export.markers == "none" and settings.program.enabled
    assert settings.transcribe.language == "en"
    # Choices only the advanced mode offers survive the basic one.
    (tmp_path / "settings.yaml").write_text(
        "export:\n  markers: all\ntranscribe:\n  language: de\n", encoding="utf-8")
    options.load(tmp_path)
    assert options.language.currentData() == "de"
    options.save(tmp_path)
    settings = load_settings([tmp_path / "settings.yaml"])
    assert settings.export.markers == "all" and settings.transcribe.language == "de"
    app.processEvents()
