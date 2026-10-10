"""Adapt legacy piece geometry to the rendered-audio contract.

Legacy factors are source seconds per output second. Assembly may trim the
last piece or pad silence; silence has no invertible source-time mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from studio.core.timeline import AudioWarpMap, AudioWarpPiece, TimeDomain


@dataclass(frozen=True, slots=True)
class RenderedAudioPlan:
    warp: AudioWarpMap
    duration_s: float
    lead_silence_s: float
    tail_silence_s: float
    # Acoustic refinement report (``sync_refine``) and the warnings it raised.
    refine: dict[str, Any] | None = None
    warnings: tuple[str, ...] = field(default=())


def adapt_piece_plan(
    pieces: list[tuple[float, float, float]],
    *,
    lead_silence_s: float,
    duration_s: float,
    map_id: str,
    source_asset_id: str,
    target_asset_id: str | None,
    strategy: int,
    stretch_method: str = "auto",
) -> RenderedAudioPlan:
    """Describe the assembled interval without stretching trimmed speech again."""
    if not all(math.isfinite(value) for value in (lead_silence_s, duration_s)):
        raise ValueError("render durations must be finite")
    if lead_silence_s < 0 or duration_s <= 0 or lead_silence_s >= duration_s:
        raise ValueError("render requires a positive mapped interval after lead silence")
    if stretch_method not in {"auto", "resample", "atempo"}:
        raise ValueError("unknown stretch method")
    converted: list[AudioWarpPiece] = []
    cursor = lead_silence_s
    previous_end: float | None = None
    for start, duration, factor in pieces:
        if not all(math.isfinite(value) for value in (start, duration, factor)):
            raise ValueError("piece geometry must be finite")
        if start < 0 or duration <= 0 or factor <= 0:
            raise ValueError("piece starts/durations/factors are invalid")
        if previous_end is not None and abs(start - previous_end) > 1e-6:
            raise ValueError("source pieces must be continuous")
        previous_end = start + duration
        if cursor >= duration_s:
            continue
        rendered_duration = min(duration / factor, duration_s - cursor)
        method = stretch_method
        if abs(factor - 1.0) <= 1e-6:
            method = "copy"
        elif method == "auto":
            method = "resample" if abs(factor - 1.0) <= 0.005 else "atempo"
        converted.append(AudioWarpPiece(
            start, rendered_duration * factor, cursor, rendered_duration, method,
        ))
        cursor += rendered_duration
    if not converted:
        raise ValueError("render has no source audio pieces")
    return RenderedAudioPlan(
        AudioWarpMap(
            map_id, source_asset_id, target_asset_id, tuple(converted), TimeDomain.FILE, strategy,
        ),
        duration_s, lead_silence_s, max(0.0, duration_s - cursor),
    )
