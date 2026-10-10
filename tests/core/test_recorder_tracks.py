import json
from pathlib import Path

import pytest

from studio.cli.main import main
from studio.core.project import Asset, AssetKind, AssetRole, Project, RecorderTrack
from studio.core.recorder_tracks import (
    effective_speaker_cameras,
    microphone_map,
    refresh_tracks,
    set_track,
)
from studio.stages.speakers import microphone_tracks
from tools.make_fixtures import generate


def _recorder(asset_id: str, channels: int) -> Asset:
    return Asset(asset_id, Path(f"{asset_id}.wav"), AssetKind.AUDIO, AssetRole.RECORDER,
                 manual={"media_info": {"audio_channels": channels}})


def test_track_ids_validate_and_round_trip(tmp_path: Path) -> None:
    assert RecorderTrack("zoom", (0, 1)).id == "zoom:0+1"
    with pytest.raises(ValueError):
        RecorderTrack("zoom", ())
    assert RecorderTrack("zoom", (2,), speaker="  ").speaker is None
    project = Project(tmp_path, tmp_path / "_studio")
    project.recorder_tracks = [RecorderTrack("zoom", (0, 1), "Аня", "lav", False)]
    project.speaker_cameras = {"Аня": "cam-a"}
    project.save(tmp_path / "project.json")
    loaded = Project.load(tmp_path / "project.json")
    assert loaded.recorder_tracks == project.recorder_tracks
    assert loaded.speaker_cameras == {"Аня": "cam-a"}


def test_merge_channels_and_rescan_keeps_manual_lanes(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [_recorder("zoom", 4)]
    project.recorder_tracks = refresh_tracks(project.assets, [])
    assert [track.id for track in project.recorder_tracks] == [
        "zoom:0", "zoom:1", "zoom:2", "zoom:3",
    ]
    set_track(project, "zoom", (1, 0), speaker="Аня")
    set_track(project, "zoom", (3,), enabled=False)
    assert [track.id for track in project.recorder_tracks] == ["zoom:0+1", "zoom:2", "zoom:3"]
    assert microphone_map(project) == {"zoom:0": "Аня", "zoom:1": "Аня"}
    with pytest.raises(ValueError):
        set_track(project, "zoom", (4,))
    rescanned = refresh_tracks([_recorder("zoom", 4), _recorder("h1", 1)],
                               project.recorder_tracks)
    assert [track.id for track in rescanned] == ["zoom:0+1", "zoom:2", "zoom:3", "h1:0"]
    shrunk = refresh_tracks([_recorder("zoom", 1)], project.recorder_tracks)
    assert [(track.id, track.speaker) for track in shrunk] == [("zoom:0", None)]


def test_speakers_use_named_lanes_and_skip_disabled_channels(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [_recorder("a", 1), _recorder("b", 1)]
    project.recorder_tracks = refresh_tracks(project.assets, [])
    set_track(project, "b", (0,), enabled=False)
    assert microphone_tracks(project, {"method": "mics"}) == {"a:0": "a-ch1"}
    set_track(project, "a", (0,), speaker="Host")
    assert microphone_tracks(project, {"tracks": {"b:0": "Guest"}}) == {
        "b:0": "Guest", "a:0": "Host",
    }
    project.speaker_cameras = {"Host": "cam-1"}
    assert effective_speaker_cameras(project, {"Host": "wide", "Guest": "cam-2"}) == {
        "Host": "cam-1", "Guest": "cam-2",
    }


def test_cli_edits_tracks_and_cameras(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = generate(tmp_path / "fixture")
    assert main(["scan", str(source), "--json"]) == 0
    capsys.readouterr()
    assert main(["tracks", str(source)]) == 0
    listed = json.loads(capsys.readouterr().out)
    track = listed["tracks"][0]
    assert main(["tracks", str(source), "--asset", track["asset_id"], "--channels",
                 str(track["channels"][0]), "--speaker", "Аня"]) == 0
    assert json.loads(capsys.readouterr().out)["tracks"][0]["speaker"] == "Аня"
    project = Project.load(source / "_studio" / "project.json")
    camera = next(asset for asset in project.assets if asset.role is AssetRole.CAMERA)
    assert main(["cameras", str(source), "--speaker", "Аня", "--camera", camera.id]) == 0
    assert json.loads(capsys.readouterr().out)["speaker_cameras"] == {"Аня": camera.id}
    assert main(["cameras", str(source), "--speaker", "Аня", "--camera", "nope"]) != 0
    capsys.readouterr()
    assert main(["scan", str(source), "--json"]) == 0
    project = Project.load(source / "_studio" / "project.json")
    assert project.recorder_tracks[0].speaker == "Аня"
    assert project.speaker_cameras == {"Аня": camera.id}
