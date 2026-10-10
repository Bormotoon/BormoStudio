import json
from pathlib import Path
from unittest.mock import patch

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.core.transcript import Transcript
from studio.stages.base import StageContext
from studio.stages.speaker_backend import Diarization, diarize
from studio.stages.speakers import run_pyannote


def test_pyannote_maps_source_time_and_preserves_overlap(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "work")
    project.assets = [Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER)]
    project.placements = [SourcePlacement("rec", 2, 1, 3, 2)]
    transcript = Transcript(tmp_path / "rec.wav", "en", 8, [],
                            metadata={"source_asset_id": "rec"})
    with patch("studio.stages.speaker_backend.diarize", return_value=Diarization([
        {"start": 0, "end": 2, "speaker": "S0"},
        {"start": 1.5, "end": 3, "speaker": "S1"},
    ])):
        result = run_pyannote(StageContext(project, {}, project.work_dir), transcript, {})
    turns = json.loads(result.project_changes["outputs"]["speakers"][0].read_text())
    assert [(item["start"], item["end"], item["speaker"]) for item in turns] == [
        (2, 3, "S0"), (3, 4, "overlap"), (4, 6, "S1"),
    ]
    assert all(path.is_file() for path in result.artifacts)


def test_missing_module_fails_before_decode(tmp_path: Path) -> None:
    import pytest

    with patch("studio.stages.speaker_backend.ModuleManager") as manager, patch(
        "studio.stages.speaker_backend.subprocess.run",
    ) as decode:
        manager.return_value.installed.return_value = False
        with pytest.raises(RuntimeError, match="module is missing"):
            diarize(tmp_path / "missing.wav", tmp_path, model="test", device="cpu")
        decode.assert_not_called()
