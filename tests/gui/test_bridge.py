import json
from pathlib import Path

from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication

from studio.core.project import Project
from studio.gui.bridge import RunnerBridge
from studio.gui.main_window import MainWindow


def test_bridge_constructs_without_starting_process() -> None:
    bridge = RunnerBridge()
    assert bridge.process.state().name == "NotRunning"


def test_review_page_displays_persisted_edit_decisions(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = Project(tmp_path, tmp_path / "_studio")
    edit = project.work_dir / "stages" / "roughcut" / "edit.json"
    edit.parent.mkdir(parents=True)
    edit.write_text(json.dumps({"cuts": [{
        "id": "cut-1", "start": 1.0, "end": 2.0, "reason": "pause", "accepted": False,
    }]}), encoding="utf-8")
    project.outputs["roughcut"] = [edit]

    window = MainWindow()
    window._show_review(project)

    assert app is not None
    assert window.review.topLevelItemCount() == 1
    assert window.review.topLevelItem(0).text(4) == "Not applied"
    assert window.cut_timeline.cuts[0].id == "cut-1"
    window.cut_timeline.cut_selected.emit("cut-1")
    assert window.review.currentItem() is window.review.topLevelItem(0)


def test_bridge_delivers_multiline_document(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    bridge = RunnerBridge()
    events, codes = [], []
    bridge.stage_event.connect(events.append)
    loop = QEventLoop()
    bridge.finished.connect(lambda code: (codes.append(code), loop.quit()))
    QTimer.singleShot(10000, loop.quit)
    bridge.start("scan", tmp_path)
    loop.exec()
    assert codes == [0]
    assert events and events[-1]["assets"] == []
    app.processEvents()
