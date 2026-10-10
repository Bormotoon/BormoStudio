"""Shared speech-gap and ffmpeg silence interval analysis."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from studio.core.transcript import Word

_SILENCE_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[0-9.]+)")


@dataclass(frozen=True, slots=True)
class Span:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def word_gaps(words: list[Word], *, minimum_s: float) -> list[Span]:
    ordered = sorted(words, key=lambda word: word.start)
    gaps: list[Span] = []
    covered_end = ordered[0].end if ordered else 0.0
    for word in ordered[1:]:
        if word.start - covered_end >= minimum_s:
            gaps.append(Span(covered_end, word.start))
        covered_end = max(covered_end, word.end)
    return gaps


def unknown_sound_spans(
    words: list[Word], silence: list[Span], *, duration_s: float,
    minimum_s: float = 0.3, coverage: Span | None = None,
) -> list[Span]:
    """Audible intervals without words; never infer a sound outside source coverage."""
    bounds = coverage or Span(0.0, duration_s)
    start, end = max(0.0, bounds.start_s), min(duration_s, bounds.end_s)
    occupied = sorted(
        [Span(word.start, word.end) for word in words] + silence,
        key=lambda span: span.start_s,
    )
    result: list[Span] = []
    cursor = start
    for span in occupied:
        if span.end_s <= cursor:
            continue
        right = min(end, span.start_s)
        if right - cursor >= minimum_s:
            result.append(Span(cursor, right))
        cursor = max(cursor, span.end_s)
        if cursor >= end:
            break
    if end - cursor >= minimum_s:
        result.append(Span(cursor, end))
    return result


def parse_silencedetect(stderr: str, duration_s: float | None = None) -> list[Span]:
    pending: float | None = None
    spans: list[Span] = []
    for line in stderr.splitlines():
        start = _SILENCE_START.search(line)
        if start:
            pending = float(start.group(1))
        end = _SILENCE_END.search(line)
        if end and pending is not None:
            value = float(end.group(1))
            if value > pending:
                spans.append(Span(pending, value))
            pending = None
    if pending is not None and duration_s is not None and duration_s > pending:
        spans.append(Span(max(0.0, pending), duration_s))
    return spans


def detect_silence(
    source: Path, *, threshold_db: float = -40.0, minimum_s: float = 0.1,
    duration_s: float | None = None, start_s: float | None = None,
) -> list[Span]:
    """Silent spans of ``source`` in file time.

    With ``start_s`` only ``[start_s, start_s + duration_s]`` is read; otherwise
    ``duration_s`` just closes a silence still open at the end of the file.
    """
    window = [] if start_s is None else ["-ss", f"{start_s:.3f}"] + (
        ["-t", f"{duration_s:.3f}"] if duration_s is not None else [])
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", *window, "-i", str(source),
         "-map", "0:a:0", "-vn", "-af",
         f"silencedetect=noise={threshold_db}dB:d={minimum_s}:mono=0",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"silence analysis failed: {result.stderr.strip()}")
    spans = parse_silencedetect(result.stderr, duration_s)
    if start_s is None:
        return spans
    return [Span(span.start_s + start_s, span.end_s + start_s) for span in spans]


def intersect(left: list[Span], right: list[Span]) -> list[Span]:
    result: list[Span] = []
    for first in left:
        for second in right:
            start = max(first.start_s, second.start_s)
            end = min(first.end_s, second.end_s)
            if end > start:
                result.append(Span(start, end))
    return result
