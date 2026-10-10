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
