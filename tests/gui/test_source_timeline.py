from pathlib import Path

from PyQt6.QtWidgets import QApplication

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.gui.source_timeline import SourceTimeline


def test_source_overview_shows_clock_mapping_and_clears_previous_project(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = Project(tmp_path, tmp_path / "work", assets=[
        Asset("cam", Path("camera.mov"), AssetKind.VIDEO, AssetRole.CAMERA),
    ], placements=[SourcePlacement("cam", 10, 2, 5, 1.01)])
    overview = SourceTimeline()
    overview.load(project)
    item = overview.topLevelItem(0).child(0)
    assert item.text(1) == "10.000"
    assert item.text(2) == "15.050"
    assert item.text(3) == "1.010000"
    assert "source in: 2.000s" in item.toolTip(0)
    assert overview.accessibleName()
    overview.load(Project(tmp_path, tmp_path / "other"))
    assert overview.topLevelItemCount() == 0
    app.processEvents()
