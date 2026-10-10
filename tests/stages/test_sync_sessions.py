from pathlib import Path
from unittest.mock import patch

import pytest

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.transcript import Transcript
from studio.stages.sync_models import AlignmentMap
from studio.stages.sync_sessions import plan_sessions


def test_reference_camera_choice_reorders_only_camera_assets(tmp_path: Path) -> None:
    from studio.stages.sync import _ordered_cameras

    cameras = [Asset(f"cam{i}", Path(f"cam{i}.mov"), AssetKind.VIDEO, AssetRole.CAMERA,
                     group_id=f"group{i}") for i in range(2)]
    project = Project(tmp_path, tmp_path / "work", assets=cameras)
    assert _ordered_cameras(project, {"sync": {"reference_camera": "group1"}}) == cameras[::-1]
    with pytest.raises(ValueError, match="unknown sync.reference_camera"):
        _ordered_cameras(project, {"sync": {"reference_camera": "missing"}})


def test_disjoint_sessions_require_explicit_offset_and_keep_valid_pairs(tmp_path: Path) -> None:
    cameras = [Asset(f"cam{i}", Path(f"cam{i}.mov"), AssetKind.VIDEO, AssetRole.CAMERA)
               for i in range(2)]
    recorders = [Asset(f"rec{i}", Path(f"rec{i}.wav"), AssetKind.AUDIO, AssetRole.RECORDER)
                 for i in range(2)]
    project = Project(tmp_path, tmp_path / "work", assets=[*cameras, *recorders])
    transcripts = {a.id: Transcript(tmp_path / a.path, "en", 10, []) for a in project.assets}

    def align(target, source, *_args, **_kwargs):
        if target.source_audio.stem[-1] != source.source_audio.stem[-1]:
            raise ValueError("different recording")
        return AlignmentMap([], -2, 1, 0, "acoustic", 5, 8)

    with patch("studio.stages.sync_sessions.align_sources", side_effect=align):
        with pytest.raises(ValueError, match="disconnected recording sessions"):
            plan_sessions(project, cameras, recorders, transcripts, {}, acoustic_first=True)
        placements, fits, rejected = plan_sessions(
            project, cameras, recorders, transcripts,
            {"session_offsets": {"cam1": 100}}, acoustic_first=True,
        )
    positions = {p.asset_id: p.offset_s for p in placements}
    assert positions == {"rec0": 0, "cam0": 2, "rec1": 100, "cam1": 102}
    assert set(fits) == {("cam0", "rec0"), ("cam1", "rec1")}
    assert "cam0:rec1" in rejected


def test_shared_recorder_connects_camera_clocks(tmp_path: Path) -> None:
    cameras = [Asset(f"cam{i}", Path(f"cam{i}.mov"), AssetKind.VIDEO, AssetRole.CAMERA)
               for i in range(2)]
    recorder = Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER)
    project = Project(tmp_path, tmp_path / "work", assets=[*cameras, recorder])
    transcripts = {a.id: Transcript(tmp_path / a.path, "en", 20, []) for a in project.assets}
    with patch("studio.stages.sync_sessions.align_sources", side_effect=[
        AlignmentMap([], 2, 1.001, 0, "text", 25, 18),
        AlignmentMap([], -3, 1.001, 0, "text", 25, 18),
    ]):
        placements, _, _ = plan_sessions(
            project, cameras, [recorder], transcripts, {}, acoustic_first=False,
        )
    positions = {p.asset_id: p for p in placements}
    assert positions["cam1"].offset_s == pytest.approx(5)
    assert positions["cam1"].k == pytest.approx(1)
    assert positions["rec"].k == pytest.approx(1.001)
