"""Dependency-aware two-phase stage planner."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studio.core.digest import file_digest
from studio.core.project import ArtifactStatus, Project, StageManifest, stable_fingerprint
from studio.stages.base import (
    Decision,
    DecisionKind,
    Stage,
    select_settings,
    stage_settings_keys,
)
from studio.stages.requirements import MissingRequirement, missing_requirements

DISCOVERY_STAGES = ("discover", "scan", "locate", "prepare")


@dataclass(frozen=True, slots=True)
class PlannedStage:
    stage_id: str
    decision: Decision
    fingerprint: str
    will_reuse: bool
    missing: tuple[MissingRequirement, ...] = ()
    # Why a stage runs again (or why a dependency blocks it); empty when fresh.
    stale_reason: str = ""


@dataclass(frozen=True, slots=True)
class Plan:
    revision: int
    stages: tuple[PlannedStage, ...]

    def stale_reasons(self) -> dict[str, str]:
        return {item.stage_id: item.stale_reason for item in self.stages if item.stale_reason}


def build_plan(
    stages: Iterable[Stage], project: Project, settings: dict[str, Any],
    *, only: set[str] | None = None, skip: set[str] | None = None,
) -> Plan:
    ordered = _topological_order(stages)
    registry = {stage.id: stage for stage in ordered}
    fingerprints: dict[str, str] = {}
    reasons: dict[str, str] = {}

    def fingerprint_of(stage: Stage) -> str:
        # Fingerprints read transcripts and digest sources: compute each once.
        if stage.id not in fingerprints:
            fingerprints[stage.id] = stage.fingerprint(project, settings)
        return fingerprints[stage.id]

    def reusable(stage: Stage) -> bool:
        return _reusable_manifest(project, stage.id, fingerprint_of(stage))

    def why(stage: Stage) -> str:
        if stage.id not in reasons:
            reasons[stage.id] = explain_stale(stage, project, settings, fingerprint_of(stage))
        return reasons[stage.id]

    planned: list[PlannedStage] = []
    decisions: dict[str, Decision] = {}
    only = only or set()
    skip = skip or set()
    stale_discovery = [
        stage for stage in ordered if stage.id in DISCOVERY_STAGES and not reusable(stage)
    ]

    for stage in ordered:
        decision = stage.decide(project, settings)
        discovery = stage.id in DISCOVERY_STAGES
        stale_reason = ""
        if stage.id in skip:
            decision = Decision.skip("disabled by user")
        elif only and stage.id not in only and not (discovery and stage in stale_discovery):
            decision = Decision.skip("disabled by --only")
        elif only and stage.id not in only and decision.kind is DecisionKind.RUN:
            # Discovery decides what every later stage consumes: a stale one
            # always runs, otherwise --only would block the stages it names.
            stale_reason = why(stage)
            decision = Decision(DecisionKind.RUN, f"runs automatically: {stale_reason}",
                                choices=decision.choices)
        elif not discovery and stale_discovery:
            first = stale_discovery[0]
            if first.id in skip:
                decision = Decision.blocked(
                    f"{why(first)}; discovery inputs are not finalized",
                    f"remove --skip {first.id}",
                )
            else:
                decision = Decision.blocked(
                    f"discovery inputs are not finalized: {why(first)}", f"run {first.id}"
                )
        else:
            failed_dependency = next(
                (
                    dependency
                    for dependency in stage.after
                    if dependency not in getattr(stage, "optional_after", ())
                    if decisions[dependency].kind is not DecisionKind.RUN
                    if not (only and dependency not in only and dependency not in skip
                            and reusable(registry[dependency]))
                ),
                None,
            )
            if failed_dependency is not None:
                dependency_stage = registry[failed_dependency]
                if (only and failed_dependency not in only and failed_dependency not in skip
                        and decisions[failed_dependency].reason == "disabled by --only"):
                    decision = Decision.skip(
                        f"{why(dependency_stage)}; add {failed_dependency} to --only"
                    )
                else:
                    decision = Decision.skip(f"dependency {failed_dependency} did not run")

        fingerprint = fingerprint_of(stage)
        reuse = decision.kind is DecisionKind.RUN and reusable(stage)
        if decision.kind is DecisionKind.RUN and not reuse and not stale_reason:
            stale_reason = why(stage)
        missing = (
            tuple(missing_requirements(stage.requirements(settings), settings))
            if decision.kind is DecisionKind.RUN and not reuse else ()
        )
        planned.append(PlannedStage(stage.id, decision, fingerprint, reuse, missing,
                                    stale_reason))
        decisions[stage.id] = decision
    return Plan(project.plan_revision, tuple(planned))


def fingerprint_parts(stage: Stage, project: Project, settings: dict[str, Any]) -> dict[str, Any]:
    """Hashes of the inputs a manifest records so a later plan can name what changed."""
    transcripts = {
        key: file_digest(path, project.work_dir)
        for key, path in sorted(project.transcripts.items()) if path.is_file()
    }
    upstream = {}
    for dependency in (*stage.after, *getattr(stage, "optional_after", ())):
        manifest = _load_manifest(_manifest_path(project, dependency))
        upstream[dependency] = manifest.fingerprint if manifest is not None else None
    return {
        "settings": stable_fingerprint(select_settings(settings, stage_settings_keys(stage))),
        "assets": stable_fingerprint(
            [(asset.id, asset.role.value, str(asset.path), asset.manual)
             for asset in project.assets]
        ),
        "transcripts": stable_fingerprint(transcripts),
        "upstream": upstream,
    }


def explain_stale(
    stage: Stage, project: Project, settings: dict[str, Any], fingerprint: str
) -> str:
    """Name what makes ``stage``'s published result unusable, e.g. a settings change."""
    manifest = _load_manifest(_manifest_path(project, stage.id))
    if manifest is None or manifest.stage != stage.id:
        return f"{stage.id} has no previous result"
    if manifest.status is not ArtifactStatus.OK:
        return f"{stage.id} did not finish last time"
    if manifest.fingerprint == fingerprint:
        return f"{stage.id} results were changed or removed"
    changes: list[str] = []
    keys = stage_settings_keys(stage)
    previous = manifest.inputs.get("effective_settings")
    if isinstance(previous, dict):
        changes.extend(settings_changes(
            select_settings(previous, keys), select_settings(settings, keys)
        ))
    explain = getattr(stage, "explain_stale", None)
    if callable(explain):
        detail = explain(project, settings, manifest)
        if detail:
            changes.append(str(detail))
    recorded = manifest.inputs.get("fingerprint_parts")
    if isinstance(recorded, dict):
        current = fingerprint_parts(stage, project, settings)
        upstream = recorded.get("upstream") or {}
        changes.extend(
            f"{dependency} changed"
            for dependency, value in current["upstream"].items()
            if dependency in upstream and upstream[dependency] != value
        )
        for part, label in (("assets", "assets changed"), ("transcripts", "transcripts changed")):
            if part in recorded and recorded[part] != current[part]:
                changes.append(label)
    if not changes:
        changes.append("inputs changed")
    return f"{stage.id} is stale: " + "; ".join(dict.fromkeys(changes))


def settings_changes(before: dict[str, Any], after: dict[str, Any], limit: int = 3) -> list[str]:
    old = _flatten(before)
    new = _flatten(after)
    changed = [
        f"{key} {_show(old.get(key, _UNSET))} \u2192 {_show(new.get(key, _UNSET))}"
        for key in sorted(old.keys() | new.keys())
        if old.get(key, _UNSET) != new.get(key, _UNSET)
    ]
    if len(changed) > limit:
        return [*changed[:limit], f"{len(changed) - limit} more settings"]
    return changed


_UNSET = object()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value and not prefix.endswith(("_streams", "_offsets")):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value}


def _show(value: Any) -> str:
    if value is _UNSET:
        return "(unset)"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 40 else text[:37] + "..."


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
