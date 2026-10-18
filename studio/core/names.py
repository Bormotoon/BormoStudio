"""The application's names: user folders and environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# User cache/state folders are named after the distribution.
APP = "bormostudio"


def env(name: str, default: str | None = None) -> str | None:
    """``BSTUDIO_<name>``, else the pre-rename ``UVF_<name>``, else ``default``."""
    return os.environ.get(f"BSTUDIO_{name}") or os.environ.get(f"UVF_{name}") or default


def models_dir() -> Path:
    """Downloaded models: ``BSTUDIO_MODELS_DIR`` or ``~/.cache/bormostudio/models``."""
    return Path(env("MODELS_DIR") or Path.home() / ".cache" / APP / "models")
