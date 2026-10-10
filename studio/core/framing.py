"""Persist desktop framing choices in the ordinary project settings layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from studio.core.publication import recover_publications
from studio.core.settings import load_settings
from studio.core.workspace import project_lock, published


def save_framing(
    work_dir: Path, mode: str, width: int, height: int, crop_x: float, *, tracking: bool = False
) -> None:
    changes = {"framing": mode, "width": width, "height": height, "crop_x": crop_x}
    path = work_dir / "settings.yaml"
    changes["tracking"] = tracking
    with project_lock(work_dir):
        recover_publications(work_dir)
        load_settings([path], [f"reels.{name}={value}" for name, value in changes.items()])
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data = data or {}
        data.setdefault("reels", {}).update(changes)
        with published(path) as temporary:
            temporary.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )


# Per-reel choices that differ from the project's reels settings. Keys are the
# moment's candidate_id (stable when moments are re-ranked) or the reel stem.
OVERRIDE_FIELDS = {"framing", "crop_x", "tracking", "title"}


def overrides_path(work_dir: Path) -> Path:
    return work_dir / "reels" / "overrides.json"


def load_reel_overrides(work_dir: Path) -> dict[str, dict[str, object]]:
    path = overrides_path(work_dir)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("reels"), dict):
        raise ValueError(f"{path}: unsupported reel overrides")
    return {str(key): dict(value) for key, value in data["reels"].items()}


def validate_reel_override(values: dict[str, object]) -> None:
    unknown = set(values) - OVERRIDE_FIELDS
    if unknown:
        raise ValueError(f"unknown reel override fields: {sorted(unknown)}")
    framing = values.get("framing")
    if framing is not None and framing not in {"source", "crop", "fit"}:
        raise ValueError("reel framing must be source, crop or fit")
    crop_x = values.get("crop_x")
    if crop_x is not None and (
        not isinstance(crop_x, (int, float)) or isinstance(crop_x, bool)
        or not math.isfinite(crop_x) or not 0 <= crop_x <= 1
    ):
        raise ValueError("reel crop_x must be between 0 and 1")
    if "tracking" in values and not isinstance(values["tracking"], bool):
        raise ValueError("reel tracking must be true or false")
    if "title" in values and not isinstance(values["title"], str):
        raise ValueError("reel title must be text")


def reel_settings(
    settings: dict[str, Any], override: dict[str, object] | None
) -> dict[str, Any]:
    """Project reels settings with one reel's override applied and checked."""
    merged = {**settings, **{k: v for k, v in (override or {}).items() if k != "title"}}
    if merged.get("tracking") and merged.get("framing") != "crop":
        raise ValueError("reel tracking requires crop framing")
    return merged


def save_reel_override(work_dir: Path, key: str, values: dict[str, object] | None) -> None:
    """Replace one reel's override; None or {} returns it to project settings."""
    if not key.strip():
        raise ValueError("reel override needs a reel key")
    if values:
        validate_reel_override(values)
        if values.get("tracking") and values.get("framing", "crop") != "crop":
            raise ValueError("reel tracking requires crop framing")
    path = overrides_path(work_dir)
    with project_lock(work_dir):
        recover_publications(work_dir)
        current = load_reel_overrides(work_dir)
        if values:
            current[key] = dict(values)
        else:
            current.pop(key, None)
        with published(path) as temporary:
            temporary.write_text(
                json.dumps({"schema_version": 1, "reels": current}, ensure_ascii=False,
                           indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
