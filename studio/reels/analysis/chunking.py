"""Deterministic prompt chunks built from transcript segments."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from studio.reels.analysis.contracts import AnalysisChunk, AnalysisChunkUnit

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_WHITESPACE = re.compile(r"\s+")


def transcript_units_from_segments(
    segments: Sequence[Mapping[str, Any]],
) -> list[AnalysisChunkUnit]:
    units: list[AnalysisChunkUnit] = []
    for index, segment in enumerate(segments):
        start, end = _float(segment.get("start")), _float(segment.get("end"))
        text = _WHITESPACE.sub(" ", str(segment.get("text", ""))).strip()
        if start is None or end is None or end <= start or not text:
            continue
        speaker = str(segment.get("speaker") or "").strip()
        sentences = [item.strip() for item in _SENTENCE_SPLIT.split(text) if item.strip()] or [text]
        timings = _word_spans(segment, sentences)
        if timings is None:
            timings = _interpolate(sentences, start, end)
        units.extend(
            AnalysisChunkUnit(index, round(left, 3), round(right, 3), sentence, speaker)
            for sentence, (left, right) in zip(sentences, timings, strict=True)
        )
    return sorted(units, key=lambda unit: (unit.start, unit.end))


def estimate_tokens(text: str) -> int:
    cheap = sum(1 for character in text if character.isascii() and character.isalpha())
    whitespace = sum(1 for character in text if character.isspace())
    return math.ceil(cheap / 4 + (len(text) - cheap - whitespace) / 2.5 + whitespace / 8)


def adaptive_overlap_seconds(chunk_seconds: int, *, low: int = 20, high: int = 45) -> int:
    return max(low, min(high, int(chunk_seconds) // 8))


def build_analysis_chunks(
    segments: Sequence[Mapping[str, Any]],
    *,
    chunk_seconds: int,
    max_chars: int,
    overlap_seconds: int = 30,
) -> list[AnalysisChunk]:
    units = transcript_units_from_segments(segments)
    chunks: list[AnalysisChunk] = []
    index = 0
    while index < len(units):
        current = [units[index]]
        next_index = index + 1
        while next_index < len(units):
            candidate = units[next_index]
            proposed = _render_units([*current, candidate])
            if candidate.end - current[0].start > chunk_seconds or len(proposed) > max_chars:
                break
            current.append(candidate)
            next_index += 1
        chunks.append(
            AnalysisChunk(
                f"chunk_{len(chunks) + 1:03d}",
                current[0].start,
                current[-1].end,
                _render_units(current),
                tuple(sorted({unit.speaker for unit in current if unit.speaker})),
                len(current),
                index > 0,
                next_index < len(units),
                tuple(sorted({unit.source_segment_index for unit in current})),
            )
        )
        if next_index <= index:
            index += 1
        elif overlap_seconds and next_index < len(units):
            threshold = max(current[0].start, current[-1].end - overlap_seconds)
            index = next(
                (item for item in range(index + 1, len(units)) if units[item].start >= threshold),
                next_index,
            )
        else:
            index = next_index
    return chunks


def _render_units(units: Sequence[AnalysisChunkUnit]) -> str:
    return "\n".join(
        _render_unit(unit)
        for unit in units
    )


def _render_unit(unit: AnalysisChunkUnit) -> str:
    speaker = f"({unit.speaker}) " if unit.speaker else ""
    return f"[{int(unit.start)}-{int(unit.end)}] {speaker}{unit.text}"


def _word_spans(
    segment: Mapping[str, Any], sentences: list[str]
) -> list[tuple[float, float]] | None:
    raw_words = segment.get("words")
    if not isinstance(raw_words, list):
        return None
    words: list[tuple[float, float]] = []
    for word in raw_words:
        if not isinstance(word, Mapping):
            return None
        start, end = _float(word.get("start")), _float(word.get("end"))
        if start is None or end is None or end < start:
            return None
        words.append((start, end))
    cursor = 0
    spans: list[tuple[float, float]] = []
    for sentence in sentences:
        count = len(sentence.split())
        if count == 0 or cursor + count > len(words):
            return None
        spans.append((words[cursor][0], words[cursor + count - 1][1]))
        cursor += count
    return spans if cursor == len(words) else None


def _interpolate(sentences: list[str], start: float, end: float) -> list[tuple[float, float]]:
    total = sum(max(1, len(sentence)) for sentence in sentences)
    cursor = start
    result: list[tuple[float, float]] = []
    for index, sentence in enumerate(sentences):
        next_cursor = (
            end if index == len(sentences) - 1 else cursor + (end - start) * len(sentence) / total
        )
        result.append((cursor, max(cursor + 0.01, next_cursor)))
        cursor = next_cursor
    return result


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
