"""Subprocess argv shared by source and frozen application launches."""

import sys
from pathlib import Path


def frozen_root() -> Path:
    """Unpacked PyInstaller bundle; only valid when ``sys.frozen`` is set."""
    root: str = sys._MEIPASS  # type: ignore[attr-defined]  # set by PyInstaller
    return Path(root)


def python_command(module: str, arguments: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        switches = {"studio.cli.main": "--cli", "studio.stages.worker": "--worker"}
        if module not in switches:
            raise ValueError(f"unsupported frozen subprocess module: {module}")
        return [sys.executable, switches[module], *arguments]
    return [sys.executable, "-m", module, *arguments]
