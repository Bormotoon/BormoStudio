"""Install and inspect isolated optional feature environments."""

from __future__ import annotations

import json
import sys
import venv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_path

from studio.core.names import env
from studio.core.workspace import project_lock
from studio.modules.cancellation import check_cancelled, run_install


@dataclass(frozen=True, slots=True)
class ModuleRecipe:
    name: str
    requirements: tuple[str, ...]
    python: str = sys.executable


RECIPES = {
    "youtube": ModuleRecipe("youtube", ("yt-dlp>=2025.1.0",)),
    # pyannote.audio 3.x reads torchaudio.AudioMetaData, which torchaudio 2.9 removed,
    # and passes use_auth_token, which huggingface_hub 1.0 removed; torch 2.7/2.8
    # are also the first CUDA 12.8 builds that run on Blackwell GPUs.
    "diarization": ModuleRecipe("diarization", (
        "pyannote.audio>=3.1,<4", "torch>=2.7,<2.9", "torchaudio>=2.7,<2.9",
        "huggingface_hub>=0.24,<1",
    )),
    "vision": ModuleRecipe(
        "vision", ("torch>=2.7,<3", "opencv-python-headless>=4.8,<5", "PyYAML>=6,<7")
    ),
}


class ModuleManager:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or user_data_path("studio") / "modules"

    def interpreter(self, name: str) -> Path:
        folder = self.root / name
        return folder / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def installed(self, name: str) -> bool:
        """Installed from the current recipe; one built from an older recipe is not."""
        marker = self.root / name / "module.json"
        if not (self.interpreter(name).is_file() and marker.is_file()):
            return False
        recipe = RECIPES.get(name)
        if recipe is None:
            return True
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8")).get("requirements")
        except (OSError, ValueError, AttributeError):
            return False
        return isinstance(recorded, list) and tuple(recorded) == recipe.requirements

    def install(
        self,
        name: str,
        *,
        progress: Callable[[dict], None] = lambda event: None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Path:
        recipe = RECIPES.get(name)
        if recipe is None:
            raise ValueError(f"unknown Studio module: {name}")
        folder = self.root / name
        folder.parent.mkdir(parents=True, exist_ok=True)
        with project_lock(self.root / ".locks" / name):
            check_cancelled(cancelled or (lambda: False))
            if self.installed(name):
                progress({"phase": "complete", "percent": 100, "name": name})
                return self.interpreter(name)
            return self._install(recipe, progress, cancelled)

    def _install(
        self,
        recipe: ModuleRecipe,
        progress: Callable[[dict], None],
        cancelled: Callable[[], bool] | None,
    ) -> Path:
        name = recipe.name
        folder = self.root / name
        progress({"phase": "environment", "percent": None, "name": name})
        if getattr(sys, "frozen", False):
            python = env("MODULE_PYTHON")
            if not python:
                raise RuntimeError(
                    "Set BSTUDIO_MODULE_PYTHON to a compatible Python interpreter "
                    "for module installation."
                )
            run_install([python, "-m", "venv", "--clear", str(folder)], cancelled)
        else:
            if cancelled is None:
                venv.EnvBuilder(with_pip=True, clear=True).create(folder)
            else:
                run_install([sys.executable, "-m", "venv", "--clear", str(folder)], cancelled)
        interpreter = self.interpreter(name)
        progress({"phase": "dependencies", "percent": None, "name": name})
        run_install(
            [str(interpreter), "-m", "pip", "install", *recipe.requirements],
            cancelled,
        )
        check_cancelled(cancelled or (lambda: False))
        (folder / "module.json").write_text(
            json.dumps(
                {"schema_version": 1, "name": name, "requirements": recipe.requirements},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        progress({"phase": "complete", "percent": 100, "name": name})
        return interpreter


def module_names() -> tuple[str, ...]:
    return tuple(RECIPES)
