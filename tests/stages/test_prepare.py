import json
from pathlib import Path

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.stages.base import StageContext
from studio.stages.prepare import PrepareStage


def test_complex_prepare_declares_all_transcripts_before_execution(tmp_path: Path) -> None:
    project = Project(
        tmp_path, tmp_path / "_studio",
        assets=[
            Asset("cam", Path("cam.mov"), AssetKind.VIDEO, AssetRole.CAMERA),
            Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER),
        ],
    )
    context = StageContext(project, {"sync": {"mode": "complex"}}, project.work_dir)
    output = PrepareStage().run(context)
    data = json.loads(output.artifacts[0].read_text(encoding="utf-8"))
    assert data["transcripts"] == ["rec", "cam"]
    assert data["plan_revision"] == 1
    for mode, expected in [("simple", ["rec", "cam"]), ("camera", ["cam"])]:
        context = StageContext(project, {"sync": {"mode": mode}}, project.work_dir)
        output = PrepareStage().run(context)
        assert json.loads(output.artifacts[0].read_text())["transcripts"] == expected


def test_text_placement_transcribes_every_camera_fast_and_the_whole_recorder(
    tmp_path: Path,
) -> None:
    from studio.stages.locate import locate_project

    def camera(name: str) -> Asset:
        return Asset(name, Path(f"{name}.mov"), AssetKind.VIDEO, AssetRole.CAMERA,
                     manual={"media_info": {"duration_s": 300.0, "audio_channels": 2}})

    recorder = Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER,
                     manual={"media_info": {"duration_s": 9000.0}})
    project = Project(tmp_path, tmp_path / "_studio",
                      assets=[camera("cam1"), camera("cam2"), recorder])
    settings = {"transcribe": {"mode": "quality"}}
    assert "placed by their speech" in locate_project(project, settings)["skipped"]
    output = PrepareStage().run(StageContext(project, settings, project.work_dir))
    data = json.loads(output.artifacts[0].read_text(encoding="utf-8"))
    assert data["transcripts"] == ["rec", "cam1", "cam2"]
    assert data["transcribe_ranges"] == {}
    # The recorder is the content (transcribe.mode); cameras only place themselves.
    assert data["transcribe_modes"] == {"cam1": "fast", "cam2": "fast"}
    acoustic = {"sync": {"placement": "acoustic"}, "transcribe": {"mode": "quality"}}
    output = PrepareStage().run(StageContext(project, acoustic, project.work_dir))
    data = json.loads(output.artifacts[0].read_text(encoding="utf-8"))
    assert data["transcripts"] == ["rec", "cam1", "cam2"]  # nothing located yet
    assert "not located" in data["reasons"]["cam1"]


def test_a_quality_recorder_gets_a_fast_transcript_for_matching(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "_studio", assets=[
        Asset("cam", Path("cam.mov"), AssetKind.VIDEO, AssetRole.CAMERA),
        Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER),
    ])
    for mode, expected in (("quality", {"rec": "fast"}), ("fast", {})):
        output = PrepareStage().run(StageContext(
            project, {"transcribe": {"mode": mode}}, project.work_dir))
        data = json.loads(output.artifacts[0].read_text(encoding="utf-8"))
        assert data["match_transcripts"] == expected
