"""Query word and sentence timings for text and reels analysis."""

from __future__ import annotations

import bisect
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studio.core.transcript import Transcript

_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?…][\"»”’)\]]*$")


def normalize_for_compare(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text).lower().replace("ё", "е")))


def normalized_tokens(text: str) -> list[str]:
    normalized = normalize_for_compare(text)
    return normalized.split() if normalized else []


@dataclass(frozen=True, slots=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class TimedSentence:
    start: float
    end: float
    text: str


class TranscriptIndex:
    """Fast timing-aware lookup over a transcript."""

    WORD_ANCHOR_PENALTY_S = 1.5
    INWARD_PENALTY_S = 0.5

    def __init__(
        self, words: Sequence[TimedWord] = (), sentences: Sequence[TimedSentence] = ()
    ) -> None:
        self.words = sorted(words, key=lambda word: (word.start, word.end))
        self.sentences = sorted(sentences, key=lambda sentence: (sentence.start, sentence.end))
        self._word_starts = [word.start for word in self.words]
        sentence_start_words = self._find_sentence_start_words()
        self._sentence_starts = sorted(
            {sentence.start for sentence in self.sentences}
            | {self.words[index].start for index in sentence_start_words}
        )
        self._sentence_ends = sorted(
            {sentence.end for sentence in self.sentences}
            | {self.words[index - 1].end for index in sentence_start_words if index > 0}
        )

    @classmethod
    def from_transcript(cls, transcript: Transcript | Mapping[str, Any]) -> TranscriptIndex:
        data = transcript.to_dict() if isinstance(transcript, Transcript) else transcript
        words: list[TimedWord] = []
        sentences: list[TimedSentence] = []
        raw_segments = data.get("segments", [])
        if isinstance(raw_segments, list):
            for segment in raw_segments:
                if not isinstance(segment, Mapping):
                    continue
                for word in segment.get("words", []):
                    if not isinstance(word, Mapping):
                        continue
                    start, end = _float(word.get("start")), _float(word.get("end"))
                    text = str(word.get("word", word.get("text", ""))).strip()
                    if text and start is not None and end is not None and end >= start:
                        words.append(TimedWord(start, end, text))
        raw_sentences = data.get("sentences", [])
        if isinstance(raw_sentences, list):
            for sentence in raw_sentences:
                item = _sentence(sentence)
                if item is not None:
                    sentences.append(item)
        if not sentences and isinstance(raw_segments, list):
            for segment in raw_segments:
                item = _sentence(segment)
                if item is not None:
                    sentences.append(item)
        return cls(words, sentences)

    def __bool__(self) -> bool:
        return bool(self.words or self.sentences)

    def words_between(self, start: float, end: float) -> list[TimedWord]:
        if end <= start:
            return []
        index = max(0, bisect.bisect_left(self._word_starts, start) - 8)
        return [
            word for word in self.words[index:]
            if word.start < end and word.end > start
        ]

    def text_between(self, start: float, end: float, *, max_chars: int = 0) -> str:
        words = self.words_between(start, end)
        text = " ".join(word.text for word in words) if words else " ".join(
            sentence.text for sentence in self.sentences
            if sentence.end > start and sentence.start < end
        )
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
            boundary = text.rfind(" ")
            if boundary > max_chars * 0.6:
                text = text[:boundary]
            text = text.rstrip() + "…"
        return text.strip()

    def snap_start(self, value: float, *, max_shift: float, limit: float | None = None) -> float:
        if max_shift <= 0:
            return value
        latest = value + max_shift if limit is None else min(value + max_shift, limit)
        candidates = [
            (abs(value - anchor) + (self.INWARD_PENALTY_S if anchor > value else 0), anchor)
            for anchor in _values_within(self._sentence_starts, value - max_shift, latest)
        ]
        word_start = _at_or_before(self._word_starts, value)
        if word_start is not None and value - word_start <= max_shift:
            candidates.append((value - word_start + self.WORD_ANCHOR_PENALTY_S, word_start))
        return min(candidates)[1] if candidates else value

    def snap_end(self, value: float, *, max_shift: float, limit: float | None = None) -> float:
        if max_shift <= 0:
            return value
        earliest = value - max_shift if limit is None else max(value - max_shift, limit)
        candidates = [
            (abs(value - anchor) + (self.INWARD_PENALTY_S if anchor < value else 0), anchor)
            for anchor in _values_within(self._sentence_ends, earliest, value + max_shift)
        ]
        word_end = self._word_end_at_or_after(value)
        if word_end is not None and word_end - value <= max_shift:
            candidates.append((word_end - value + self.WORD_ANCHOR_PENALTY_S, word_end))
        return min(candidates)[1] if candidates else value

    def timed_tokens(self, start: float, end: float) -> list[tuple[str, float, float]]:
        if self.words:
            return [
                (token, word.start, word.end)
                for word in self.words_between(start, end)
                for token in normalized_tokens(word.text)[:1]
            ]
        tokens: list[tuple[str, float, float]] = []
        for sentence in self.sentences:
            if sentence.end <= start or sentence.start >= end:
                continue
            parts = normalized_tokens(sentence.text)
            for index, token in enumerate(parts):
                duration = (sentence.end - sentence.start) / len(parts)
                word_start = sentence.start + index * duration
                tokens.append((token, word_start, word_start + duration))
        return tokens

    def speech_rate(self, start: float, end: float) -> float | None:
        words = self.words_between(start, end)
        return round(len(words) / (end - start), 4) if words and end > start else None

    def _find_sentence_start_words(self) -> list[int]:
        if not self.words:
            return []
        starts = {0}
        starts.update(
            index + 1 for index, word in enumerate(self.words[:-1])
            if _SENTENCE_END_RE.search(word.text)
        )
        for sentence in self.sentences:
            index = bisect.bisect_left(self._word_starts, sentence.start - 1e-6)
            if index < len(self.words):
                starts.add(index)
        return sorted(starts)

    def _word_end_at_or_after(self, target: float) -> float | None:
        index = max(0, bisect.bisect_left(self._word_starts, target) - 8)
        return next((word.end for word in self.words[index:] if word.end >= target), None)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sentence(value: object) -> TimedSentence | None:
    if not isinstance(value, Mapping):
        return None
    start, end = _float(value.get("start")), _float(value.get("end"))
    text = str(value.get("text", "")).strip()
    if not text or start is None or end is None or end <= start:
        return None
    return TimedSentence(start, end, text)


def _values_within(values: Sequence[float], low: float, high: float) -> list[float]:
    if high < low:
        return []
    return list(values[bisect.bisect_left(values, low):bisect.bisect_right(values, high)])


def _at_or_before(values: Sequence[float], target: float) -> float | None:
    index = bisect.bisect_right(values, target)
    return values[index - 1] if index else None