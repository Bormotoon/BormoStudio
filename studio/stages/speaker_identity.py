"""Match speaker labels between files so one person keeps one name.

Each file (a recorder lane, a diarized source) labels its speakers on its own.
A voice signature per label — pyannote's embedding when diarization provides
one, otherwise MFCC statistics of the label's loudest speech — is compared
across files. Labels are merged best-first above a cosine threshold, never
within one file, never between files that sound at the same time, and never
between two different names the user gave.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000


@dataclass(slots=True)
class VoiceLabel:
    group: str  # the file (or simultaneous set) the label comes from
    label: str
    signature: np.ndarray
    first_s: float = 0.0  # first timeline appearance, for stable naming
    fixed_name: str | None = None  # a name the user gave; never renamed


@dataclass(slots=True)
class IdentityMatch:
    names: dict[tuple[str, str], str]
    similarities: list[dict[str, object]] = field(default_factory=list)


def _mel_filters(bands: int, fft_size: int, rate: int) -> np.ndarray:
    def mel(hz: np.ndarray) -> np.ndarray:
        return np.asarray(2595 * np.log10(1 + hz / 700))

    def hz(value: np.ndarray) -> np.ndarray:
        return np.asarray(700 * (10 ** (value / 2595) - 1))

    edges = hz(np.linspace(mel(np.array(60.0)), mel(np.array(rate / 2 * 0.95)), bands + 2))
    bins = np.fft.rfftfreq(fft_size, 1 / rate)
    filters = np.zeros((bands, len(bins)))
    for index in range(bands):
        low, center, high = edges[index:index + 3]
        rising = (bins - low) / (center - low)
        falling = (high - bins) / (high - center)
        filters[index] = np.maximum(0, np.minimum(rising, falling))
    return filters


def voice_signature(samples: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """MFCC mean and spread over the loudest voiced frames, or None if too little speech."""
    samples = np.asarray(samples, dtype=np.float64)
    frame, hop, fft_size, bands, coefficients = (
        int(rate * 0.025), int(rate * 0.01), 512, 40, 20
    )
    if len(samples) < frame * 20:
        return None
    count = 1 + (len(samples) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = samples[index] * np.hamming(frame)
    energy = np.sqrt(np.mean(np.square(frames), axis=1))
    loud = energy > max(1e-4, float(np.percentile(energy, 60)))
    # Bleed from other microphones is quieter: keep frames near this lane's peak.
    loud &= 20 * np.log10(np.maximum(energy, 1e-12)) > 20 * np.log10(energy.max()) - 30
    if loud.sum() < 20:
        return None
    spectrum = np.abs(np.fft.rfft(frames[loud], fft_size)) ** 2
    logmel = np.log(spectrum @ _mel_filters(bands, fft_size, rate).T + 1e-10)
    basis = np.cos(np.pi / bands * (np.arange(bands)[None, :] + 0.5)
                   * np.arange(1, coefficients)[:, None])
    cepstra = logmel @ basis.T
    # Per-coefficient scale keeps low coefficients from dominating the cosine.
    weights = 1 / np.arange(1, coefficients)
    signature: np.ndarray = np.concatenate(
        (cepstra.mean(axis=0) * weights, cepstra.std(axis=0) * weights)
    )
    return signature


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / norm) if norm > 0 else 0.0


def match_labels(
    labels: Iterable[VoiceLabel],
    *,
    threshold: float,
    simultaneous: set[frozenset[str]] | None = None,
) -> IdentityMatch:
    """Merge labels of different files whose voices agree above ``threshold``.

    ``simultaneous`` lists pairs of groups that sound at the same time; their
    labels are different people by construction and are never merged. A pair
    merges only when each is the other's most similar label in its file.
    """
    items = list(labels)
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("speaker match threshold must be a cosine between -1 and 1")
    blocked = simultaneous or set()
    cluster = list(range(len(items)))

    def members(root: int) -> list[int]:
        return [i for i, value in enumerate(cluster) if value == root]

    def compatible(left: int, right: int) -> bool:
        first, second = members(left), members(right)
        groups = {items[i].group for i in first}
        if any(items[j].group in groups for j in second):
            return False
        if any(frozenset((items[i].group, items[j].group)) in blocked
               for i in first for j in second):
            return False
        names = {items[i].fixed_name for i in first + second if items[i].fixed_name}
        return len(names) <= 1

    similarity_of = {
        (i, j): cosine(items[i].signature, items[j].signature)
        for i in range(len(items)) for j in range(len(items))
        if items[i].group != items[j].group
    }

    def best_in_group(i: int, group: str) -> int:
        return max((j for j in range(len(items)) if items[j].group == group),
                   key=lambda j: similarity_of[(i, j)])

    pairs = sorted(
        ((similarity_of[(i, j)], i, j) for i, j in similarity_of if i < j), reverse=True
    )
    report: list[dict[str, object]] = []
    for similarity, i, j in pairs:
        left, right = cluster[i], cluster[j]
        # Mutual best within the two files: absolute similarity of different
        # voices is often high, their ranking is far more reliable.
        mutual = (best_in_group(i, items[j].group) == j
                  and best_in_group(j, items[i].group) == i)
        merged = (similarity >= threshold and mutual and left != right
                  and compatible(left, right))
        report.append({
            "first": f"{items[i].group}:{items[i].label}",
            "second": f"{items[j].group}:{items[j].label}",
            "similarity": round(similarity, 4),
            "merged": merged or left == right,
        })
        if merged:
            cluster = [left if value == right else value for value in cluster]
    names: dict[tuple[str, str], str] = {}
    used: set[str] = set()
    roots = sorted(set(cluster), key=lambda root: min(
        (items[i].first_s, items[i].group) for i in members(root)))
    for root in roots:
        group = sorted(members(root), key=lambda i: (items[i].first_s, items[i].group))
        fixed = next((items[i].fixed_name for i in group if items[i].fixed_name), None)
        name = fixed or items[group[0]].label
        if name in used and not fixed:
            name = f"{name}@{items[group[0]].group}"
        used.add(name)
        for i in group:
            names[(items[i].group, items[i].label)] = name
    return IdentityMatch(names, report)
