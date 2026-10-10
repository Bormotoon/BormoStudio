"""Dependency-aware two-phase stage planner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studio.core.project import Project, StageManifest
from studio.stages.base import Decision, DecisionKind, Stage

DISCOVERY_STAGES = ("discover", "scan", "prepare")


@dataclass(frozen=True, slots=True)
class PlannedStage:
    stage_id: str
    decision: Decision
    fingerprint: str
    will_reuse: bool


@dataclass(frozen=True, slots=True)
class Plan:
    revision: int
    stages: tuple[PlannedStage, ...]


def build_plan(
    stages: Iterable[Stage], project: Project, settings: dict[str, Any],
    *, only: set[str] | None = None, skip: set[str] | None = None,
) -> Plan:
    ordered = _topological_order(stages)
    registry = {stage.id: stage for stage in ordered}
    planned: list[PlannedStage] = []
    decisions: dict[str, Decision] = {}
    only = only or set()
    skip = skip or set()
    discovery_complete = all(
        _reusable_manifest(project, stage.id, stage.fingerprint(project, settings))
        for stage in ordered if stage.id in DISCOVERY_STAGES
    )

    for stage in ordered:
        decision = stage.decide(project, settings)
        if only and stage.id not in only:
            decision = Decision.skip("disabled by --only")
        elif stage.id in skip:
            decision = Decision.skip("disabled by user")
        elif stage.id not in DISCOVERY_STAGES and not discovery_complete:
            decision = Decision.blocked(
                "discovery inputs are not finalized", "run scan and prepare"
            )
        else:
            failed_dependency = next(
                (
                    dependency
                    for dependency in stage.after
                    if dependency not in getattr(stage, "optional_after", ())
                    if decisions[dependency].kind is not DecisionKind.RUN
                    if not (only and dependency not in only and dependency not in skip
                            and _reusable_manifest(project, dependency,
                                registry[dependency].fingerprint(project, settings)))
                ),
                None,
            )
            if failed_dependency is not None:
                decision = Decision.skip(f"dependency {failed_dependency} did not run")

        fingerprint = stage.fingerprint(project, settings)
        reuse = (
            decision.kind is DecisionKind.RUN
            and _reusable_manifest(project, stage.id, fingerprint)
        )
        planned.append(PlannedStage(stage.id, decision, fingerprint, reuse))
        decisions[stage.id] = decision
    return Plan(project.plan_revision, tuple(planned))


def _topological_order(stages: Iterable[Stage]) -> list[Stage]:
    by_id = {stage.id: stage for stage in stages}
    ordered: list[Stage] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage: Stage) -> None:
        if stage.id in visiting:
            raise ValueError(f"stage dependency cycle at {stage.id}")
        if stage.id in visited:
            return
        visiting.add(stage.id)
        for dependency in stage.after:
            if dependency not in by_id:
                raise ValueError(f"unknown dependency {dependency} for {stage.id}")
            visit(by_id[dependency])
        visiting.remove(stage.id)
        visited.add(stage.id)
        ordered.append(stage)

    for candidate in by_id.values():
        visit(candidate)
    return ordered


def _manifest_path(project: Project, stage_id: str) -> Path:
    return project.work_dir / "manifests" / f"{stage_id}.json"


def _load_manifest(path: Path) -> StageManifest | None:
    try:
        return StageManifest.load(path)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _reusable_manifest(project: Project, stage_id: str, fingerprint: str) -> bool:
    manifest = _load_manifest(_manifest_path(project, stage_id))
    return (
        manifest is not None
        and manifest.stage == stage_id
        and manifest.reusable(project.work_dir, fingerprint)
    )
