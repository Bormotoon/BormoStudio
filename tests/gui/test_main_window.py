import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QFileDialog

from studio.core.project import Project
from studio.core.settings import load_settings
from studio.gui.main_window import BASIC_HIDDEN, MainWindow


def _window(tmp_path: Path) -> MainWindow:
    preferences = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    return MainWindow(preferences)


def test_basic_mode_is_the_default_and_the_choice_is_remembered(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = _window(tmp_path)
    assert not window.advanced.isChecked()
    assert all(window.navigation.item(row).isHidden() for row in BASIC_HIDDEN)
    assert not window.basic_start.isHidden() and window.advanced_start.isHidden()
    window.advanced.setChecked(True)
    assert not any(window.navigation.item(row).isHidden() for row in BASIC_HIDDEN)
    window.navigation.setCurrentRow(8)
    window.preferences.sync()
    again = _window(tmp_path)
    assert again.advanced.isChecked()
    window.advanced.setChecked(False)
    # A page the basic mode hides is left for the start page.
    assert window.navigation.currentRow() == 0
    for item in (window, again):
        item.close()
    app.processEvents()


def test_basic_process_saves_choices_and_runs_everything(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "material"
    source.mkdir()
    window = _window(tmp_path)
    assert not window.process_button.isEnabled()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(source))
    window._choose_basic()
    assert window.process_button.isEnabled() and str(source) in window.folder_label.text()
    window.basic_options.review.setChecked(True)
    window.basic_options.markers.setChecked(False)
    calls = []
    monkeypatch.setattr(window.bridge, "start", lambda *args: calls.append(args))
    window._process()
    assert calls == [("run", source, ["--events"])]
    assert window.navigation.currentRow() == 4
    settings = load_settings([source / "_studio" / "settings.yaml"])
    assert settings.program.enabled and settings.export.markers == "none"
    window._plan_event({"t": "start", "stage": "sync"})
    window._plan_event({"t": "progress", "stage": "sync", "value": 0.4, "message": "fitting"})
    assert window.progress_label.text() == "Synchronization: fitting"
    assert window.progress_bar.value() == 40
    window._plan_event({"t": "warning", "stage": "sync", "message": "clip moved"})
    assert "sync: clip moved" in window.log.toPlainText()
    window.close()
    app.processEvents()


def test_result_names_the_project_files(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = _window(tmp_path)
    project = Project(tmp_path, tmp_path / "_studio")
    export = tmp_path / "_studio" / "export"
    project.outputs["export"] = [export / "camera-audio" / "a.wav", export / "Studio.fcpxml",
                                 export / "Studio.xml"]
    project.work_dir.mkdir()
    (project.work_dir / "report.json").write_text(json.dumps(
        {"status": "needs_review", "review": {"DJI_0921": "failed: drift 40 ms"}}))
    text = window._result_text(project)
    assert f"Final Cut Pro: {export / 'Studio.fcpxml'}" in text
    assert f"Premiere Pro / DaVinci Resolve: {export / 'Studio.xml'}" in text
    assert "DJI_0921: failed: drift 40 ms" in text
    assert "a.wav" not in text and "All outputs" not in text
    window.close()
    app.processEvents()
