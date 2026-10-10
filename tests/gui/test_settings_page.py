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
