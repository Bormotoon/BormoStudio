"""Stage contracts shared by planning and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from studio.core.enums import StrEnum
from studio.core.project import Project


class GpuUse(StrEnum):
    NONE = "none"
    WHISPER = "whisper"
    TORCH = "torch"
    LLM = "llm"
    NVENC = "nvenc"


class DecisionKind(StrEnum):
    RUN = "run"
    SKIP = "skip"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class Requirement:
    kind: str
    value: str | float | bool
    why: str
    optional: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    kind: DecisionKind
    reason: str = ""
    fix: str = ""
    choices: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def run(cls, choices: dict[str, Any] | None = None) -> Decision:
        return cls(DecisionKind.RUN, choices=choices or {})

    @classmethod
    def skip(cls, reason: str) -> Decision:
        return cls(DecisionKind.SKIP, reason=reason)

    @classmethod
    def blocked(cls, reason: str, fix: str) -> Decision:
        return cls(DecisionKind.BLOCKED, reason=reason, fix=fix)


@dataclass(frozen=True, slots=True)
class StageOutput:
    artifacts: tuple[Path, ...] = ()
    project_changes: dict[str, Any] = field(default_factory=dict)
    # Published as ``warning`` events: things the user should look at.
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class StageContext:
    project: Project
    settings: dict[str, Any]
    work_dir: Path


class Stage(Protocol):
    id: str
    title: str
    after: tuple[str, ...]
    gpu: GpuUse

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]: ...
    def decide(self, project: Project, settings: dict[str, Any]) -> Decision: ...
    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str: ...
    def run(self, context: StageContext) -> StageOutput: ...


def select_settings(settings: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """The part of ``settings`` named by dotted ``keys`` ("sync" or "sync.mode").

    Missing keys are left out, so adding an unrelated default does not change
    a fingerprint built from the selection.
    """
    selected: dict[str, Any] = {}
    for key in keys:
        value: Any = settings
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            selected[key] = value
    return selected


def stage_settings_keys(stage: object) -> tuple[str, ...]:
    return tuple(getattr(stage, "settings_keys", ()))


def stage_uses_llm(stage: Stage, settings: dict[str, Any]) -> bool:
    """Whether a run needs the owned llama server; optional for non-LLM stages."""
    uses = getattr(stage, "uses_llm", None)
    return bool(uses(settings)) if callable(uses) else stage.gpu == GpuUse.LLM
