from pathlib import Path

from PyQt6.QtWidgets import QApplication

from studio.core.project import Asset, AssetKind, AssetRole, Project, RecorderTrack
from studio.gui.material import ASSET_ROLE, MaterialPage


def _project(tmp_path: Path) -> Project:
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [
        Asset("cam-a", Path("a.mp4"), AssetKind.VIDEO, AssetRole.CAMERA, group_id="osmo"),
        Asset("cam-b", Path("b.mp4"), AssetKind.VIDEO, AssetRole.CAMERA),
        Asset("zoom", Path("zoom.wav"), AssetKind.AUDIO, AssetRole.RECORDER, group_id="zoom",
              manual={"media_info": {"audio_channels": 2}}),
    ]
    project.recorder_tracks = [RecorderTrack("zoom", (0,), "Аня"), RecorderTrack("zoom", (1,))]
    project.speaker_cameras = {"Аня": "osmo"}
    return project


def test_material_page_groups_assets_and_emits_cli_commands(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    page = MaterialPage()
    requests = []
    page.command_requested.connect(lambda command, args: requests.append((command, args)))
    page.load(_project(tmp_path))
    groups = [page.tree.topLevelItem(index).text(0)
              for index in range(page.tree.topLevelItemCount())]
    assert groups == ["osmo", "(no group)", "zoom"]
    loose = page.tree.topLevelItem(1).child(0)
    assert loose.data(0, ASSET_ROLE) == "cam-b"
    assert page.tree.drop_group(page.tree.topLevelItem(0).child(0)) == "osmo"
    assert page.tree.drop_group(page.tree.topLevelItem(1)) == ""
    page.tree.regroup_requested.emit(["cam-b"], "osmo")
    assert requests[-1] == ("material", ["--asset", "cam-b", "--group", "osmo"])

    page.tracks.setCurrentItem(page.tracks.topLevelItem(1))
    page.track_channels.setText("0, 1")
    page.track_speaker.setText("Борис")
    page._save_track()
    assert requests[-1] == ("tracks", [
        "--asset", "zoom", "--channels", "0,1", "--speaker", "Борис", "--enable",
    ])
    assert page.cameras.topLevelItem(0).text(1) == "osmo"
    assert [page.camera_choice.itemText(i) for i in range(page.camera_choice.count())] == [
        "osmo", "cam-a", "cam-b",
    ]
    page.camera_speaker.setCurrentText("Аня")
    page.camera_choice.setCurrentText("cam-b")
    page._assign_camera()
    assert requests[-1] == ("cameras", ["--speaker", "Аня", "--camera", "cam-b"])
    page.camera_speaker.setCurrentIndex(page.camera_speaker.count() - 1)
    page._assign_camera()
    assert requests[-1] == ("cameras", ["--speaker", "@wide", "--camera", "cam-b"])
    assert app is not None
