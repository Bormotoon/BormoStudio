"""Small ffprobe facade used by discovery stages."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

WAV_MUX_ARGS = ["-rf64", "auto"]


def build_atempo_chain(factor: float) -> list[str]:
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("tempo factor must be finite and positive")
    filters: list[str] = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return filters


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_s: float
    container_duration_s: float
    fps: str | None
    width: int | None
    height: int | None
    video_codec: str | None
    audio_codec: str | None
    audio_channels: int | None
    audio_sample_rate: int | None
    audio_stream_index: int | None
    is_cfr: bool
    tags: dict[str, str]
    audio_bits_per_sample: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe(path: Path, timeout_s: float = 30.0) -> MediaInfo:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode:
        detail = result.stderr.strip() or str(result.returncode)
        raise RuntimeError(f"ffprobe failed for {path}: {detail}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    audio_index = next(
        (
            index
            for index, stream in enumerate(audios)
            if stream.get("disposition", {}).get("default")
        ),
        0 if audios else None,
    )
    audio = audios[audio_index] if audio_index is not None else None
    container_duration = _float(data.get("format", {}).get("duration"))
    if container_duration is None:
        container_duration = _float((video or audio or {}).get("duration"))
    if container_duration is None:
        raise RuntimeError(f"could not determine media duration for {path}")
    fps, is_cfr = _video_rate(video)
    picture_duration = _float((video or {}).get("duration")) or container_duration
    tags = {
        str(key).lower(): str(value)
        for source in (data.get("format", {}).get("tags", {}), (video or {}).get("tags", {}))
        for key, value in source.items()
    }
    return MediaInfo(
        duration_s=picture_duration,
        container_duration_s=container_duration,
        fps=str(fps) if fps else None,
        width=_int((video or {}).get("width")),
        height=_int((video or {}).get("height")),
        video_codec=(video or {}).get("codec_name"),
        audio_codec=(audio or {}).get("codec_name"),
        audio_channels=_int((audio or {}).get("channels")),
        audio_sample_rate=_int((audio or {}).get("sample_rate")),
        audio_stream_index=int(audio.get("index", audio_index)) if audio is not None else None,
        is_cfr=is_cfr,
        tags=tags,
        audio_bits_per_sample=(
            _int((audio or {}).get("bits_per_raw_sample"))
            or _int((audio or {}).get("bits_per_sample"))
        ),
    )


def _video_rate(video: dict[str, Any] | None) -> tuple[Fraction | None, bool]:
    if not video:
        return None, True
    nominal = _fraction(video.get("r_frame_rate"))
    average = _fraction(video.get("avg_frame_rate"))
    cfr = nominal is not None and (average is None or abs(float(nominal) - float(average)) < 1e-3)
    return nominal, cfr


def _fraction(value: object) -> Fraction | None:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return result if result > 0 else None


def _float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    try:
        result = int(str(value))
    except (TypeError, ValueError):
        return None
    return result or None
