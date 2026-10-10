"""Reel finishing: loudness normalization and cover images."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from studio.core.proc import run_logged

TRUE_PEAK_DB = -1.0
LOUDNESS_RANGE = 11.0


def measure_loudness(video: Path) -> dict[str, float] | None:
    """EBU R128 loudness of the first audio stream, or None without audio."""
    result = run_logged(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(video), "-map", "0:a:0?",
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        timeout=600,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    found = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.S)
    if found is None:
        return None
    data = json.loads(found.group(0))
    values = {key: float(data[key]) for key in
              ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    # Digital silence measures as -inf; there is nothing to normalize.
    return None if values["input_i"] == float("-inf") else values


def normalize_loudness(video: Path, target_lufs: float) -> dict[str, Any]:
    """Two-pass loudnorm in place; video is copied, audio re-encoded once."""
    measured = measure_loudness(video)
    if measured is None:
        return {"status": "no-audio"}
    loudnorm = (
        f"loudnorm=I={target_lufs}:TP={TRUE_PEAK_DB}:LRA={LOUDNESS_RANGE}"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )
    with tempfile.TemporaryDirectory(prefix="uvf-loudness-") as folder:
        output = Path(folder) / video.name
        result = run_logged(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
             "-map", "0:v:0?", "-map", "0:a:0", "-c:v", "copy", "-af", loudnorm,
             "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
             str(output)],
            timeout=1200,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        shutil.move(str(output), video)
    return {"status": "normalized", "target_lufs": target_lufs,
            "true_peak_db": TRUE_PEAK_DB, "input_lufs": measured["input_i"]}


def render_cover(video: Path, output: Path, at_s: float) -> None:
    result = run_logged(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{max(0.0, at_s):.3f}",
         "-i", str(video), "-frames:v", "1", "-q:v", "2", str(output)],
        timeout=120,
    )
    if result.returncode or not output.is_file():
        raise RuntimeError(result.stderr or "ffmpeg wrote no cover frame")
