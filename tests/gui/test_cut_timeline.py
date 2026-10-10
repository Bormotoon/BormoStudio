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


def test_whole_edit_preview_renders_in_background_and_follows_clicks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import os
    import time

    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import json
    import subprocess
    from pathlib import Path

    from studio.core.project import Asset, AssetKind, AssetRole, Project
    from studio.core.timeline import SourcePlacement, TimeDomain
    from studio.core.transcript import Transcript
    from studio.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    camera = tmp_path / "camera.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=320x180:r=25:d=6", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=6", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(camera)], check=True)
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mp4"), AssetKind.VIDEO, AssetRole.CAMERA)]
    project.assets[0].manual["media_info"] = {"width": 320, "height": 180, "fps": 25}
    project.placements = [SourcePlacement("cam", 0, 0, 6)]
    project.transcripts["timeline"] = project.work_dir / "timeline.json"
    Transcript(camera, "en", 6, [], time_domain=TimeDomain.TIMELINE).save(
        project.transcripts["timeline"])
    edit = project.work_dir / "stages" / "roughcut" / "edit.json"
    edit.parent.mkdir(parents=True)
    edit.write_text(json.dumps({"keep": [{"start": 0, "end": 2}, {"start": 4, "end": 6}]}))
    project.save(project.work_dir / "project.json")
    (project.work_dir / "settings.yaml").write_text("{}\n")
    window = MainWindow()
    window.source = tmp_path
    window.cut_timeline.set_edit(6.0, [TimelineCut("x", 2, 4, "pause", "", True)], [])
    window._start_timeline_preview()
    deadline = time.monotonic() + 60
    while window.timeline_preview is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert window.timeline_preview is not None, window.review_status.text()
    assert window.timeline_preview.video.is_file()
    requested = []
    window.cut_timeline.position_requested.connect(requested.append)
    window.cut_timeline.resize(600, 56)
    click = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(500, 20),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    window.cut_timeline.mousePressEvent(click)
    assert requested == [5.0]
    window._timeline_position(2500)  # preview 2.5 s is timeline 4.5 s, after the cut
    assert window.cut_timeline.playhead == 4.5
    window._show_review(project)
    assert window.timeline_preview is None
    window.close()
    app.processEvents()
