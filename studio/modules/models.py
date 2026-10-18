"""Verified optional vision model installation without importing ML packages."""

from __future__ import annotations

import ast
import hashlib
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from studio.core.launch import frozen_root
from studio.core.names import models_dir
from studio.core.workspace import project_lock, published
from studio.modules.cancellation import check_cancelled

if TYPE_CHECKING:
    from studio.modules.catalog import CatalogModel
    from studio.modules.downloads import CatalogInstaller


@dataclass(frozen=True)
class ModelRecipe:
    filename: str
    url: str
    sha256: str


def recipes() -> dict[str, ModelRecipe]:
    result = {}
    for name, module, filename in (
        ("yunet", "face_crop.py", "face_detection_yunet_2023mar.onnx"),
        ("light-asd", "active_speaker.py", "light_asd_talkset.model"),
    ):
        source = (
            frozen_root() / "managed" / "studio" / "vision" / module
            if getattr(sys, "frozen", False)
            else files("studio.vision").joinpath(module)
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        constants = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in {"MODEL_URL", "MODEL_SHA256"}
        }
        url = constants["MODEL_URL"]
        if name == "yunet":
            url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        result[name] = ModelRecipe(filename, url, constants["MODEL_SHA256"])
    return result


class ModelManager:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or models_dir()

    def names(self) -> list[str]:
        from studio.modules.catalog import CATALOG

        return [*recipes(), *CATALOG]

    def describe(self, name: str) -> dict[str, object]:
        from studio.modules.catalog import CATALOG

        entry = CATALOG.get(name)
        return {
            "name": name,
            "kind": entry.kind if entry else "vision",
            "title": entry.title if entry else name,
            "size": entry.size if entry else None,
            "installed": self.installed(name),
            "path": str(self.path(name)),
        }

    def _catalog(self, name: str) -> tuple[CatalogModel, CatalogInstaller] | None:
        from studio.modules.catalog import CATALOG
        from studio.modules.downloads import CatalogInstaller

        entry = CATALOG.get(name)
        return (entry, CatalogInstaller(self.root)) if entry else None

    def path(self, name: str) -> Path:
        found = self._catalog(name)
        if found is not None:
            return found[1].path(found[0])
        if name not in recipes():
            raise ValueError(f"unknown model: {name}")
        return self.root / recipes()[name].filename

    def installed(self, name: str) -> bool:
        found = self._catalog(name)
        if found is not None:
            return found[1].installed(found[0])
        path = self.path(name)
        return path.is_file() and digest(path) == recipes()[name].sha256

    def install(
        self,
        name: str,
        *,
        progress: Callable[[dict], None] = lambda event: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Path:
        found = self._catalog(name)
        if found is not None:
            return found[1].install(found[0], progress=progress, cancelled=cancelled)
        path = self.path(name)
        recipe = recipes()[name]
        with project_lock(self.root):
            check_cancelled(cancelled)
            if self.installed(name):
                progress({"phase": "complete", "percent": 100, "name": name})
                return path
            progress({"phase": "download", "percent": 0, "name": name})
            with published(path) as temporary:
                with (
                    urllib.request.urlopen(recipe.url, timeout=60) as response,
                    temporary.open("wb") as output,
                ):
                    size = 0
                    total = (
                        int(response.headers.get("Content-Length", 0))
                        if hasattr(response, "headers")
                        else 0
                    )
                    last: int | None = -1
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        check_cancelled(cancelled)
                        size += len(block)
                        if size > 128 * 1024 * 1024:
                            raise ValueError("model download exceeds size limit")
                        output.write(block)
                        percent = min(99, int(size * 100 / total)) if total else None
                        if percent != last:
                            progress(
                                {
                                    "phase": "download",
                                    "percent": percent,
                                    "bytes": size,
                                    "total": total,
                                    "name": name,
                                }
                            )
                            last = percent
                progress({"phase": "verify", "percent": None, "name": name})
                if digest(temporary) != recipe.sha256:
                    raise ValueError(f"{name}: downloaded model checksum mismatch")
                check_cancelled(cancelled)
            progress({"phase": "complete", "percent": 100, "name": name})
        return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def whisper_model_source(model: str, manager: ModelManager | None = None) -> str:
    """Local directory of a downloaded catalog Whisper model, else the name itself."""
    from studio.modules.catalog import whisper_entry

    entry = whisper_entry(model)
    manager = manager or ModelManager()
    if entry is not None and manager.installed(entry.name):
        return str(manager.path(entry.name))
    return model
