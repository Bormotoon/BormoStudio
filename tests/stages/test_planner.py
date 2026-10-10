from pathlib import Path
from typing import Any

import pytest

from studio.core.project import ArtifactStatus, Project, StageManifest, describe_artifact
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput
from studio.stages.planner import build_plan


class _Stage:
    title = "Test"
    gpu = GpuUse.NONE

    def __init__(self, stage_id: str, after: tuple[str, ...] = ()) -> None:
        self.id = stage_id
        self.after = after

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return []

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        return Decision.run()

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        return self.id

    def run(self, context: StageContext) -> StageOutput:
        return StageOutput()


def _project(tmp_path: Path) -> Project:
    work = tmp_path / "_studio"
    (work / "manifests").mkdir(parents=True)
    for name in ("discover", "scan", "prepare"):
        artifact = work / f"{name}.json"
        artifact.write_text("{}")
        StageManifest(
            name, name, {}, [describe_artifact(work, artifact)], ArtifactStatus.OK, "test",
        ).save(work / "manifests" / f"{name}.json")
    return Project(tmp_path, work)


def test_topological_order_is_independent_of_registration_order(tmp_path: Path) -> None:
    stages = [_Stage("export", ("sync",)), _Stage("sync", ("prepare",)), _Stage("prepare")]
    plan = build_plan(stages, _project(tmp_path), {})
    assert [item.stage_id for item in plan.stages] == ["prepare", "sync", "export"]


def test_main_plan_is_blocked_until_discovery_manifests_exist(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "_studio")
    plan = build_plan([_Stage("scan"), _Stage("sync", ("scan",))], project, {})
    assert plan.stages[1].decision.reason.startswith("discovery inputs are not finalized")


def test_user_skip_propagates_to_required_dependency(tmp_path: Path) -> None:
    plan = build_plan(
        [_Stage("prepare"), _Stage("sync", ("prepare",)), _Stage("export", ("sync",))],
        _project(tmp_path), {}, skip={"sync"},
    )
    assert plan.stages[-1].decision.reason == "dependency sync did not run"


def test_only_consumes_verified_dependency_but_skip_still_blocks(tmp_path: Path) -> None:
    project = _project(tmp_path)
    artifact = project.work_dir / "sync.json"
    artifact.write_text("{}")
    StageManifest("sync", "sync", {}, [describe_artifact(project.work_dir, artifact)],
                  ArtifactStatus.OK, "test").save(project.work_dir / "manifests" / "sync.json")
    stages = [_Stage("prepare"), _Stage("sync", ("prepare",)), _Stage("export", ("sync",))]
    assert build_plan(stages, project, {}, only={"export"}).stages[-1].decision.kind.value == "run"
    skipped = build_plan(stages, project, {}, only={"export"}, skip={"sync"})
    assert skipped.stages[-1].decision.kind.value == "skip"
    artifact.write_text("damaged")
    assert build_plan(stages, project, {}, only={"export"}).stages[-1].decision.kind.value == "skip"


def test_cycles_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cycle"):
        build_plan([_Stage("a", ("b",)), _Stage("b", ("a",))], _project(tmp_path), {})


def test_optional_dependency_preserves_order_without_blocking(tmp_path: Path) -> None:
    export = _Stage("export", ("sync", "roughcut"))
    export.optional_after = ("roughcut",)
    plan = build_plan(
        [export, _Stage("roughcut", ("sync",)), _Stage("sync")],
        _project(tmp_path), {}, skip={"roughcut"},
    )
    assert [item.stage_id for item in plan.stages] == ["sync", "roughcut", "export"]
    assert plan.stages[-1].decision.kind.value == "run"


@pytest.mark.parametrize("damage", ["malformed", "stale", "wrong_stage", "failed", "artifact"])
def test_invalid_discovery_blocks_main_plan(tmp_path: Path, damage: str) -> None:
    project = _project(tmp_path)
    path = project.work_dir / "manifests" / "scan.json"
    if damage == "malformed":
        path.write_text("{}")
    elif damage == "artifact":
        (project.work_dir / "scan.json").write_text("changed")
    else:
        manifest = StageManifest.load(path)
        if damage == "stale":
            manifest.fingerprint = "old-inputs"
        elif damage == "wrong_stage":
            manifest.stage = "prepare"
        else:
            manifest.status = ArtifactStatus.FAILED
        manifest.save(path)
    plan = build_plan([_Stage("scan"), _Stage("sync", ("scan",))], project, {})
    assert not plan.stages[0].will_reuse
    assert plan.stages[1].decision.reason.startswith("discovery inputs are not finalized")


class _Needs(_Stage):
    def requirements(self, settings):  # type: ignore[no-untyped-def]
        return [Requirement("binary", "definitely-not-a-binary-uvf", "test")]


def test_plan_reports_missing_only_for_stages_that_will_run(tmp_path: Path) -> None:
    project = _project(tmp_path)
    plan = build_plan([_Stage("discover"), _Stage("scan"), _Stage("prepare"),
                       _Needs("needs")], project, {})
    entry = next(item for item in plan.stages if item.stage_id == "needs")
    assert entry.missing[0].value == "definitely-not-a-binary-uvf"
    skipped = build_plan([_Stage("discover"), _Stage("scan"), _Stage("prepare"),
                          _Needs("needs")], project, {}, skip={"needs"})
    assert not next(item for item in skipped.stages if item.stage_id == "needs").missing


def _save_manifest(project: Project, stage, settings: dict, artifact_name: str) -> None:
    from studio.stages.planner import fingerprint_parts

    artifact = project.work_dir / artifact_name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not artifact.exists():
        artifact.write_text("{}")
    StageManifest(
        stage.id, stage.fingerprint(project, settings),
        {"effective_settings": settings,
         "fingerprint_parts": fingerprint_parts(stage, project, settings)},
        [describe_artifact(project.work_dir, artifact)], ArtifactStatus.OK, "test",
    ).save(project.work_dir / "manifests" / f"{stage.id}.json")


def _real_discovery(tmp_path: Path, settings: dict):
    from studio.core.project import Asset, AssetKind, AssetRole
    from studio.stages.prepare import PrepareStage

    project = _project(tmp_path)
    project.assets = [Asset("cam", Path("cam.mov"), AssetKind.VIDEO, AssetRole.CAMERA),
                      Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER)]
    prepare = PrepareStage()
    _save_manifest(project, prepare, settings, "stages/prepare/requirements.json")
    return project, prepare


def test_unrelated_setting_does_not_make_prepare_stale(tmp_path: Path) -> None:
    before = {"sync": {"mode": "auto"}, "reels": {"layout": "video"}}
    project, prepare = _real_discovery(tmp_path, before)
    after = {"sync": {"mode": "auto"}, "reels": {"layout": "title"}}
    plan = build_plan([_Stage("scan"), prepare, _Stage("reels", ("prepare",))],
                      project, after, only={"reels"})
    by_id = {item.stage_id: item for item in plan.stages}
    assert by_id["prepare"].decision.reason == "disabled by --only"
    assert by_id["reels"].decision.kind.value == "run"


def test_stale_prepare_runs_automatically_with_named_setting(tmp_path: Path) -> None:
    project, prepare = _real_discovery(tmp_path, {"sync": {"mode": "auto"}})
    stages = [_Stage("scan"), prepare, _Stage("sync", ("prepare",))]
    plan = build_plan(stages, project, {"sync": {"mode": "complex"}}, only={"sync"})
    by_id = {item.stage_id: item for item in plan.stages}
    assert by_id["prepare"].decision.kind.value == "run"
    assert "sync.mode auto \u2192 complex" in by_id["prepare"].decision.reason
    assert by_id["prepare"].decision.reason.startswith("runs automatically")
    assert "sync.mode auto \u2192 complex" in by_id["sync"].decision.reason

    skipped = build_plan(stages, project, {"sync": {"mode": "complex"}},
                         only={"sync"}, skip={"prepare"})
    sync = next(item for item in skipped.stages if item.stage_id == "sync")
    assert sync.decision.kind.value == "blocked"
    assert "sync.mode auto \u2192 complex" in sync.decision.reason
    assert sync.decision.fix == "remove --skip prepare"


def test_dependency_outside_only_names_missing_transcript(tmp_path: Path) -> None:
    import json

    from studio.stages.transcribe import TranscribeStage

    settings = {"sync": {"mode": "auto"}}
    project, prepare = _real_discovery(tmp_path, settings)
    requirements = project.work_dir / "stages" / "prepare" / "requirements.json"
    requirements.write_text(json.dumps({"transcripts": ["rec"]}))
    _save_manifest(project, prepare, settings, "stages/prepare/requirements.json")
    transcribe = TranscribeStage()
    _save_manifest(project, transcribe, settings, "stages/transcribe/rec.json")
    requirements.write_text(json.dumps({"transcripts": ["rec", "cam"]}))
    _save_manifest(project, prepare, settings, "stages/prepare/requirements.json")
    stages = [_Stage("scan"), prepare, transcribe, _Stage("sync", ("transcribe",))]
    plan = build_plan(stages, project, settings, only={"sync"})
    sync = next(item for item in plan.stages if item.stage_id == "sync")
    assert sync.decision.kind.value == "skip"
    assert "transcribe is stale" in sync.decision.reason
    assert "needs transcript cam" in sync.decision.reason
    assert sync.decision.reason.endswith("add transcribe to --only")


def test_gui_subtitle_render_command_is_planned(tmp_path: Path) -> None:
    from studio.core.settings import Settings

    defaults = Settings().to_dict()
    project, prepare = _real_discovery(tmp_path, defaults)
    changed = Settings().to_dict()
    changed["program"]["burn_subtitles"] = True
    stages = [_Stage("scan"), prepare, _Stage("program_subtitles", ("prepare",))]
    plan = build_plan(stages, project, changed, only={"program_subtitles"})
    target = next(item for item in plan.stages if item.stage_id == "program_subtitles")
    assert target.decision.kind.value == "run"
