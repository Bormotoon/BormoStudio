"""Video and audio encoder arguments of every reel encode (Forge's video.* keys)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from studio.stages.program_encoder import EncoderChoice, select_encoder


@lru_cache(maxsize=8)
def _choice(requested: str) -> EncoderChoice:
    # One encode probe per encoder and process, not one per reel and step.
    return select_encoder(requested)


def video_codec_args(reels: dict[str, Any]) -> list[str]:
    """NVENC: quality-driven VBR (``nvenc_cq``) capped at ``video_bitrate``;
    libx264: ``preset`` at ``video_bitrate``; ``auto`` falls back to libx264."""
    bitrate = str(reels.get("video_bitrate", "8M"))
    codec = _choice(str(reels.get("encoder", "auto")))
    if codec.codec == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", str(reels.get("nvenc_preset", "p5")),
                "-tune", "hq", "-rc", "vbr", "-cq", str(int(reels.get("nvenc_cq", 21))),
                "-b:v", "0", "-maxrate", bitrate, "-bufsize", bitrate, "-pix_fmt", "yuv420p"]
    if codec.codec != "libx264":
        return ["-c:v", codec.codec, "-b:v", bitrate, "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", str(reels.get("preset", "fast")), "-b:v", bitrate,
            "-pix_fmt", "yuv420p"]


def encoder_report(reels: dict[str, Any]) -> dict[str, str]:
    codec = _choice(str(reels.get("encoder", "auto")))
    return {"requested": codec.requested, "codec": codec.codec, "reason": codec.reason}
