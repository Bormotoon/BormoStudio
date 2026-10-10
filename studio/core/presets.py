"""Packaged settings presets that merge into a project's settings.yaml."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from studio.core.settings_store import save_settings


def list_presets() -> list[str]:
    root = files("studio.resources").joinpath("presets")
    return sorted(
        entry.name.removesuffix(".yaml") for entry in root.iterdir()
        if entry.name.endswith(".yaml")
    )


def load_preset(name: str) -> dict[str, Any]:
    if name not in list_presets():
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(list_presets())}")
    text = files("studio.resources").joinpath("presets", f"{name}.yaml").read_text(
        encoding="utf-8"
    )
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"preset {name!r} must be a mapping")
    return data


def apply_preset(work_dir: Path, name: str) -> dict[str, Any]:
    """Merge the preset into settings.yaml; keys it does not name stay untouched."""
    data = load_preset(name)
    save_settings(work_dir, data)
    return data
