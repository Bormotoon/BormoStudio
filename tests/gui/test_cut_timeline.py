from PyQt6.QtWidgets import QApplication

from studio.gui.cut_timeline import CutTimeline, TimelineCut


def test_timeline_reports_program_length_and_hit_tests_cuts() -> None:
    app = QApplication.instance() or QApplication([])
    widget = CutTimeline()
    widget.resize(1000, 56)
    widget.set_edit(100.0, [
        TimelineCut("a", 10, 20, "pause", "pause 10s", True),
        TimelineCut("b", 15, 30, "retake", "take", True),
        TimelineCut("c", 50, 52, "llm", "slips: x", False),
    ], [(5.0, "ну")])
    assert widget.kept_duration() == 80.0
    assert widget.cut_at(510).id == "c"
    assert widget.cut_at(800) is None
    selected = []
    widget.cut_selected.connect(selected.append)
    widget.select("a")
    widget.grab()
    assert app is not None and widget.selected == "a"
    widget.set_edit(100.0, [], [])
    assert widget.selected is None
