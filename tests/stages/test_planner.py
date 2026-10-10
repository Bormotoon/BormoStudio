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
    assert plan.stages[1].decision.reason == "discovery inputs are not finalized"


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
    assert plan.stages[1].decision.reason == "discovery inputs are not finalized"
