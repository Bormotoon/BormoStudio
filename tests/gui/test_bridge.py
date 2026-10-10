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


def test_gui_and_cli_export_identical_outputs_and_repair_corruption(tmp_path: Path) -> None:
    import subprocess

    from studio.cli.main import _run_discovery, _run_pipeline, _settings
    from studio.core.project import ArtifactStatus, StageManifest, describe_artifact
    from studio.core.transcript import Segment, Transcript, Word
    from studio.stages.transcribe import TranscribeStage

    app = QApplication.instance() or QApplication([])
    video = tmp_path / "camera.mov"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=s=160x120:r=25:d=1", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=1", "-c:v", "libx264", "-c:a", "pcm_s16le",
        str(video),
    ], check=True)
    project = _run_discovery(tmp_path, include_prepare=True)
    transcript = project.work_dir / "fixture-transcript.json"
    Transcript(video, "en", 1, [Segment(0, 1, (Word("Hello.", 0, 1),))]).save(transcript)
    project.transcripts[project.assets[0].id] = transcript
    project.save(project.work_dir / "project.json")
    StageManifest(
        "transcribe", TranscribeStage().fingerprint(
            project, _settings(project.work_dir / "settings.yaml")
        ), {}, [describe_artifact(project.work_dir, transcript)], ArtifactStatus.OK, "fixture",
    ).save(project.work_dir / "manifests" / "transcribe.json")
    project = _run_pipeline(tmp_path, only={"timeline", "export"})
    originals = {path: path.read_bytes() for path in project.outputs["export"]}
    mtimes = {path: path.stat().st_mtime_ns for path in originals}

    def gui_run():
        bridge = RunnerBridge()
        codes = []
        loop = QEventLoop()
        bridge.finished.connect(lambda code: (codes.append(code), loop.quit()))
        QTimer.singleShot(20000, loop.quit)
        bridge.start("run", tmp_path, ["--only", "timeline", "--only", "export"])
        loop.exec()
        assert codes == [0]
        app.processEvents()

    gui_run()
    assert {path: path.stat().st_mtime_ns for path in originals} == mtimes
    damaged = next(iter(originals))
    damaged.write_text("broken export")
    gui_run()
    assert {path: path.read_bytes() for path in originals} == originals
