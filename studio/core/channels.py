"""What the channels of a multichannel recorder actually contain.

A lavalier duplicated into both channels (dual mono), a safety copy at
−6/−12 dB (Zoom, Tascam, DJI Mic "linked"), a dead second input and two real
microphones all look like "2 channels". Scan compares short excerpts so a
dual-mono file is shown, synchronized and exported as one mono lane instead
of two speakers.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from studio.core.project import Asset

EXCERPTS = (0.10, 0.35, 0.60, 0.85)
EXCERPT_S = 5.0
SAMPLE_RATE = 8000
SILENT_DB = -60.0
DUAL_CORRELATION = 0.999
DUAL_LEVEL_DB = 0.5
LINKED_CORRELATION = 0.98
CLIP_LEVEL = 0.999


def analyze_channels(
    path: Path, *, channels: int, duration_s: float, stream_index: int | None = None,
) -> dict[str, Any] | None:
    """``media_info.channel_analysis`` for a file with ≥ 2 channels, or ``None``."""
    if channels < 2 or duration_s <= 0:
        return None
    excerpts = []
    for fraction in EXCERPTS:
        start = max(0.0, min(duration_s - EXCERPT_S, duration_s * fraction))
        samples = _decode(path, start, min(EXCERPT_S, duration_s), channels, stream_index)
        if samples is not None and samples.shape[0] >= SAMPLE_RATE // 10:
            excerpts.append(samples)
    return classify(excerpts, channels)


def classify(excerpts: list[np.ndarray], channels: int) -> dict[str, Any]:
    """Classify channel relations from decoded excerpts (``samples × channels``)."""
    voiced = [item for item in excerpts
              if max(_db(item[:, c]) for c in range(channels)) > SILENT_DB]
    levels = [[_db(item[:, c]) for c in range(channels)] for item in voiced]
    silent = [c for c in range(channels)
              if voiced and all(level[c] <= SILENT_DB for level in levels)]
    pairs: list[dict[str, Any]] = []
    parent = list(range(channels))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    relation: dict[tuple[int, int], str] = {}
    for a in range(channels):
        for b in range(a + 1, channels):
            if a in silent or b in silent or not voiced:
                continue
            correlations = [_correlation(item[:, a], item[:, b]) for item in voiced]
            differences = [level[a] - level[b] for level in levels]
            kind = "independent"
            if all(value >= DUAL_CORRELATION for value in correlations) and all(
                    abs(value) <= DUAL_LEVEL_DB for value in differences):
                kind = "dual_mono"
            elif all(value >= LINKED_CORRELATION for value in correlations):
                kind = "linked"
            relation[a, b] = kind
            pairs.append({"channels": [a, b], "relation": kind,
                          "correlation": round(min(correlations), 5),
                          "level_difference_db": round(float(np.median(differences)), 2)})
            if kind != "independent":
                parent[find(b)] = find(a)
    groups: dict[int, list[int]] = {}
    for channel in range(channels):
        if channel not in silent:
            groups.setdefault(find(channel), []).append(channel)
    result_groups: list[dict[str, Any]] = []
    for members in groups.values():
        kinds = {relation.get((a, b)) for a in members for b in members if a < b}
        kind = "single" if len(members) == 1 else "dual_mono" if kinds == {"dual_mono"} else (
            "linked")
        result_groups.append({"channels": members, "relation": kind,
                              "primary": _primary(members, voiced)})
    if not voiced:
        overall = "unknown"
    elif silent and len(result_groups) == 1:
        overall = "one_silent"
    elif len(result_groups) == 1 and len(result_groups[0]["channels"]) > 1:
        overall = result_groups[0]["relation"]
    else:
        overall = "independent"
    return {"class": overall, "groups": result_groups, "silent": silent, "pairs": pairs,
            "excerpts": len(voiced)}


@dataclass(frozen=True, slots=True)
class EffectiveAudio:
    """The channels of an asset that carry distinct sound, and how to read them."""

    channels: tuple[int, ...]
    # ffmpeg ``pan`` filter that yields one channel per lane, or None for all channels.
    pan: str | None
    note: str = ""

    @property
    def count(self) -> int:
        return len(self.channels)


def effective_audio(asset: Asset) -> EffectiveAudio:
    """Distinct channels of ``asset`` from its scan analysis (all of them by default)."""
    info = asset.manual.get("media_info", {})
    count = int(info.get("audio_channels") or 0)
    analysis = info.get("channel_analysis")
    if not isinstance(analysis, dict) or count < 2 or analysis.get("class") == "unknown":
        return EffectiveAudio(tuple(range(count)), None)
    groups = [group for group in analysis.get("groups", [])
              if isinstance(group, dict) and group.get("channels")]
    if not groups or len(groups) == count:
        return EffectiveAudio(tuple(range(count)), None)
    primaries = tuple(int(group.get("primary", group["channels"][0])) for group in groups)
    mapping = "|".join(f"c{index}=c{channel}" for index, channel in enumerate(primaries))
    layout = "mono" if len(primaries) == 1 else f"{len(primaries)}c"
    return EffectiveAudio(primaries, f"pan={layout}|{mapping}",
                          str(analysis.get("class", "")))


def describe(analysis: object) -> str:
    """One line for the material screen and ``uvf tracks``."""
    if not isinstance(analysis, dict):
        return ""
    parts = [str(analysis.get("class", ""))]
    for pair in analysis.get("pairs", []):
        parts.append(f"ch{pair['channels'][0] + 1}/ch{pair['channels'][1] + 1}: "
                     f"r={pair['correlation']:.4f}, Δ{pair['level_difference_db']:+.1f} dB")
    if analysis.get("silent"):
        parts.append("silent: " + ", ".join(f"ch{c + 1}" for c in analysis["silent"]))
    return "; ".join(part for part in parts if part)


def _primary(members: list[int], voiced: list[np.ndarray]) -> int:
    """The loudest channel that does not clip (a linked safety copy is quieter)."""
    if len(members) == 1:
        return members[0]
    def level(channel: int) -> float:
        return float(np.median([_db(item[:, channel]) for item in voiced]))
    def clips(channel: int) -> bool:
        return any(float(np.max(np.abs(item[:, channel]))) >= CLIP_LEVEL for item in voiced)
    ordered = sorted(members, key=level, reverse=True)
    return next((channel for channel in ordered if not clips(channel)), ordered[0])


def _db(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    return 20 * math.log10(rms) if rms > 0 else -120.0


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - float(np.mean(a))
    b = b.astype(np.float64) - float(np.mean(b))
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b)) / denominator if denominator > 0 else 0.0


def _decode(path: Path, start: float, duration: float, channels: int,
            stream_index: int | None) -> np.ndarray | None:
    stream = f"0:{stream_index}" if stream_index is not None else "0:a:0"
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
             "-i", str(path), "-map", stream, "-vn", "-ar", str(SAMPLE_RATE), "-f", "f32le",
             "-"],
            capture_output=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    data = np.frombuffer(result.stdout, dtype="<f4")
    usable = data.size - data.size % channels
    return data[:usable].reshape(-1, channels) if usable else None
