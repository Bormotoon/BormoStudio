"""Pick the most capable ffmpeg on PATH for this process and its children.

A machine can carry several builds: on the owner's, Homebrew's ffmpeg 9 comes
first in PATH but has neither libass (burned captions, title bands) nor NVENC,
while /usr/bin's has both. Studio calls plain "ffmpeg"/"ffprobe", so the chosen
build's folder is put first in PATH once, at startup.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from studio.core.names import env

# Children inherit the choice instead of probing again.
_CHOSEN = "BSTUDIO_FFMPEG_CHOSEN"


def _capabilities(ffmpeg: Path) -> tuple[int, int]:
    """(has libass, has NVENC) of one build; (-1, -1) when it does not run."""
    try:
        filters = subprocess.run([str(ffmpeg), "-hide_banner", "-filters"],
                                 capture_output=True, text=True, timeout=20)
        encoders = subprocess.run([str(ffmpeg), "-hide_banner", "-encoders"],
                                  capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return -1, -1
    if filters.returncode or encoders.returncode:
        return -1, -1
    return int(" subtitles " in filters.stdout), int("h264_nvenc" in encoders.stdout)


def candidates(path: str) -> list[Path]:
    """Folders of PATH holding both ffmpeg and ffprobe, in PATH order."""
    seen: list[Path] = []
    for entry in path.split(os.pathsep):
        folder = Path(entry or ".").resolve()
        if folder not in seen and all(
                os.access(folder / name, os.X_OK) for name in ("ffmpeg", "ffprobe")):
            seen.append(folder)
    return seen


def choose_ffmpeg() -> Path | None:
    """Put the best ffmpeg folder first in PATH; ``BSTUDIO_FFMPEG_DIR`` overrides.

    Best: it runs, then libass, then NVENC; ties keep PATH order.
    """
    path = os.environ.get("PATH", "")
    chosen = env("FFMPEG_DIR") or os.environ.get(_CHOSEN)
    if not chosen:
        scored = [(_capabilities(folder / "ffmpeg"), -order, folder)
                  for order, folder in enumerate(candidates(path))]
        working = [item for item in scored if item[0][0] >= 0]
        if not working:
            return None
        chosen = str(max(working)[2])
    os.environ[_CHOSEN] = chosen
    if path.split(os.pathsep)[0] != chosen:
        os.environ["PATH"] = os.pathsep.join([chosen, path])
    return Path(chosen)
