"""Probe usable hardware encoding rather than trusting compiled-in support."""

import subprocess
import sys
from dataclasses import dataclass

# Hardware encoders by setting value: ffmpeg codec and its rate control. NVENC
# keeps ffmpeg's defaults; VideoToolbox needs a bitrate (its default is low),
# and -q:v constant quality exists only on Apple Silicon.
HARDWARE = {
    "nvenc": ("h264_nvenc", ()),
    "videotoolbox": ("h264_videotoolbox", ("-b:v", "12M")),
}


@dataclass(frozen=True)
class EncoderChoice:
    requested: str
    codec: str
    reason: str
    options: tuple[str, ...] = ()


def _auto_hardware() -> str:
    return "videotoolbox" if sys.platform == "darwin" else "nvenc"


def select_encoder(requested: str) -> EncoderChoice:
    if requested not in {"auto", "cpu", "libx264", *HARDWARE}:
        raise ValueError(f"unknown program encoder: {requested}")
    if requested in {"cpu", "libx264"}:
        return EncoderChoice(requested, "libx264", "CPU explicitly requested")
    hardware = _auto_hardware() if requested == "auto" else requested
    codec, options = HARDWARE[hardware]
    label = "NVENC" if hardware == "nvenc" else "VideoToolbox"
    try:
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=black:s=256x256:r=25", "-frames:v", "1",
            "-c:v", codec, *options, "-pix_fmt", "yuv420p", "-f", "null", "-",
        ], capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return EncoderChoice(requested, codec, f"{label} encode probe passed", options)
        reason = result.stderr.strip()[-2000:] or f"ffmpeg exited {result.returncode}"
    except subprocess.TimeoutExpired:
        reason = f"{label} encode probe timed out after 20 seconds"
    if requested != "auto":
        raise RuntimeError(f"requested {label} is unavailable: {reason}")
    return EncoderChoice(requested, "libx264", f"{label} unavailable: {reason}")
