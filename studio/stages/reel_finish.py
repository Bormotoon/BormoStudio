"""Reel finishing: loudness, covers, QA, extra formats and the preview reel."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from studio.core.media import probe
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


def normalize_loudness(
    video: Path, target_lufs: float, audio_bitrate: str = "192k"
) -> dict[str, Any]:
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
    with tempfile.TemporaryDirectory(prefix="bstudio-loudness-") as folder:
        output = Path(folder) / video.name
        result = run_logged(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
             "-map", "0:v:0?", "-map", "0:a:0", "-c:v", "copy", "-af", loudnorm,
             "-ar", "48000", "-c:a", "aac", "-b:a", audio_bitrate, "-movflags", "+faststart",
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


_BLACK_DURATION = re.compile(r"black_duration:\s*(\d+(?:\.\d+)?)")


def black_seconds(video: Path) -> float:
    """Seconds of black picture (blackdetect decodes the whole clip)."""
    result = run_logged(
        ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", str(video),
         "-vf", "blackdetect=d=0.5:pix_th=0.10", "-an", "-f", "null", "-"],
        timeout=600,
    )
    return sum(float(value) for value in _BLACK_DURATION.findall(result.stderr or ""))


def check_clip(video: Path, *, expected_s: float, blackdetect: bool = False) -> list[str]:
    """Problems of a rendered reel (Forge media QA); [] when it is fine.

    A truncated, silent or black reel would otherwise ship like any other.
    """
    if not video.is_file() or video.stat().st_size <= 0:
        return ["qa: empty or missing file"]
    try:
        info = probe(video)
    except (OSError, RuntimeError, ValueError) as exc:
        return [f"qa: ffprobe cannot read the file ({exc})"]
    problems = []
    if not info.video_codec:
        problems.append("qa: no video stream")
    if not info.audio_codec:
        problems.append("qa: no audio stream")
    tolerance = max(1.5, 0.05 * expected_s)
    if expected_s > 0 and abs(info.container_duration_s - expected_s) > tolerance:
        problems.append(
            f"qa: duration {info.container_duration_s:.1f}s, expected {expected_s:.1f}s")
    if blackdetect and info.container_duration_s > 0:
        black = black_seconds(video)
        if black > 0.5 * info.container_duration_s:
            problems.append(f"qa: {black:.1f}s of {info.container_duration_s:.1f}s is black")
    return problems


# Extra formats of a finished reel: suffix and ffmpeg arguments after the input.
EXPORTS = {
    "webm": (".webm", ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-c:a", "libopus"]),
    "audio": (".m4a", ["-vn", "-c:a", "aac", "-b:a", "192k"]),
}


def export_format(video: Path, kind: str) -> Path:
    """``webm``, ``audio`` (m4a) or ``gif`` (480 px, 12 fps, own palette) beside the reel."""
    if kind == "gif":
        output = video.with_suffix(".gif")
        scale = "fps=12,scale=480:-1:flags=lanczos"
        with tempfile.TemporaryDirectory(prefix="bstudio-gif-") as folder:
            palette = Path(folder) / "palette.png"
            for command in (
                ["-vf", f"{scale},palettegen", str(palette)],
                ["-i", str(palette), "-lavfi", f"{scale}[x];[x][1:v]paletteuse", str(output)],
            ):
                result = run_logged(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
                                     *command], timeout=1200)
                if result.returncode:
                    raise RuntimeError(result.stderr)
        return output
    suffix, arguments = EXPORTS[kind]
    output = video.with_suffix(suffix)
    result = run_logged(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
                         *arguments, str(output)], timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return output


def concat_preview(videos: list[Path], output: Path) -> bool:
    """All finished reels back to back, without re-encoding (Forge reels_preview)."""
    if not videos:
        return False
    with tempfile.TemporaryDirectory(prefix="bstudio-preview-") as folder:
        listing = Path(folder) / "reels.txt"
        listing.write_text("".join(
            "file '" + str(video.resolve()).replace("'", "'\\''") + "'\n" for video in videos),
            encoding="utf-8")
        result = run_logged(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat",
                             "-safe", "0", "-i", str(listing), "-c", "copy", str(output)],
                            timeout=1800)
    return result.returncode == 0 and output.is_file()
