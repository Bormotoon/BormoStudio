"""Check stage requirements before running and name where to install them."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studio.stages.base import Requirement


@dataclass(frozen=True, slots=True)
class MissingRequirement:
    kind: str
    value: str
    why: str
    optional: bool
    hint: str
    # Modules page target: ("module", name) or ("model", catalog/recipe name).
    install: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "value": self.value, "why": self.why,
            "optional": self.optional, "hint": self.hint,
            "install": list(self.install) if self.install else None,
        }


_PACKAGES = {
    "faster-whisper": ("faster_whisper", "pip install 'bormostudio[whisper]'"),
    "mlx-whisper": ("mlx_whisper", "pip install 'bormostudio[mlx]' (Apple Silicon)"),
}


def missing_requirements(
    requirements: list[Requirement], settings: dict[str, Any],
) -> list[MissingRequirement]:
    missing: list[MissingRequirement] = []
    seen: set[tuple[str, str]] = set()
    for item in requirements:
        key = (item.kind, str(item.value))
        if key in seen:
            continue
        seen.add(key)
        found = _check(item, settings)
        if found is not None:
            missing.append(found)
    return missing


def _check(item: Requirement, settings: dict[str, Any]) -> MissingRequirement | None:
    value = str(item.value)

    def report(hint: str, install: tuple[str, str] | None = None) -> MissingRequirement:
        return MissingRequirement(item.kind, value, item.why, item.optional, hint, install)

    if item.kind == "binary":
        return None if shutil.which(value) else report(
            f"install {value} with the system package manager and put it on PATH"
        )
    if item.kind == "package":
        module, hint = _PACKAGES.get(value, (value.replace("-", "_"), f"pip install {value}"))
        return None if importlib.util.find_spec(module) is not None else report(hint)
    if item.kind == "module":
        from studio.modules.manager import ModuleManager

        return None if ModuleManager().installed(value) else report(
            f"install the {value} module in Modules and models (bstudio modules install {value})",
            ("module", value),
        )
    if item.kind == "model":
        from studio.modules.models import ModelManager

        return None if ModelManager().installed(value) else report(
            f"download {value} in Modules and models (bstudio models install {value})",
            ("model", value),
        )
    if item.kind == "llama":
        llm = settings.get("llm", {})
        if not isinstance(llm, dict) or not llm.get("managed", False):
            return None  # An external endpoint is checked when the stage connects.
        executable = str(llm.get("executable", "llama-server"))
        if not shutil.which(executable) and not Path(executable).is_file():
            return report(f"install llama.cpp and set llm.executable (now {executable!r})")
        model_path = str(llm.get("model_path", ""))
        if not model_path or not Path(model_path).is_file():
            return report(
                "download a GGUF model in Modules and models and save it as llm.model_path",
                ("model", "gguf-gemma-4-e4b-it-q4"),
            )
    return None
