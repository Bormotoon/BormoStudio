import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from studio.core.project import Project
from studio.core.settings import load_settings
from studio.core.subtitles import load_edits
from studio.core.transcript import Segment, Transcript, Word
from studio.gui.main_window import MainWindow
from studio.gui.subtitles import SubtitleEditor


def test_native_editor_save_and_invalid_times(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    video = tmp_path / "reel-001.mp4"
    video.touch()
    path = video.with_suffix(".json")
    transcript = Transcript(video, "en", 5, [Segment(1, 3, (Word("hello", 1, 3),))])
    transcript.save(path)
    project = Project(tmp_path, tmp_path / "work", outputs={"reel_render": [video, path]})
    editor = SubtitleEditor()
    editor.load_project(project)
    assert editor.table.rowCount() == 1
    editor.table.item(0, 2).setText("Updated subtitle")
    editor.font_size.setValue(60)
    editor.framing.setCurrentText("crop")
    editor.crop_x.setValue(80)
    assert editor.dirty
    assert editor.save()
    cues, style = load_edits(project.work_dir, "reel-001", transcript)
    assert cues[0].text == "Updated subtitle"
    assert style.size == 60
    assert load_settings([project.work_dir / "settings.yaml"]).reels.crop_x == 0.8
    editor.table.item(0, 1).setText("8")
    assert not editor.save()
    assert editor.dirty
    editor.dirty = False
    editor.close()
    app.processEvents()


def test_main_window_has_native_subtitle_page() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.navigation.item(6).text() == "Subtitles"
    assert window.pages.widget(6) is window.subtitles
    assert window.navigation.item(7).text() == "Modules and models"
    assert window.pages.widget(7) is window.modules_page
    window.close()
    app.processEvents()
