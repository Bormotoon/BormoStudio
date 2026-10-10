"""Speech clips for Whisper and a check for words torn from their phrase.

faster-whisper's VAD glues speech fragments into ≤ 30 s chunks regardless of
the silence between them, then maps word times back by the fragment holding
each word's middle. A noise burst taken for speech, two minutes before a
phrase, can thus end up in one chunk with that phrase, and the phrase's first
word inherits the burst's time ("По" at 14.8 s, "новой," at 141.7 s).

``speech_clips`` builds the clips Studio passes to Whisper instead: real,
continuous ranges of the file that merge fragments only across short pauses,
so every word time is a true file time. ``repair_detached_words`` is the
safety net for any backend and for old transcripts: a 1–2 word series cut off
from the rest of its phrase by a long pause is moved next to the phrase when
speech without words is there, or marked suspect when its own place is silent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from studio.core.transcript import Segment, Transcript, Word

MAX_CLIP_S = 30.0
MERGE_GAP_S = 1.5
MIN_FRAGMENT_S = 0.25
MAX_WORD_GAP_S = 2.0
MAX_SERIES_WORDS = 2
SUSPECT_PROBABILITY = 0.1
# Energy above this (dBFS RMS) counts as sound where the series could belong.
SPEECH_DB = -45.0
_SENTENCE_END = (".", "!", "?", "…", "！", "？")
SEGMENTATION_VERSION = "clips-1"

Energy = Callable[[float, float], float]


def speech_clips(
    fragments: Sequence[tuple[float, float]],
    *,
    merge_gap_s: float = MERGE_GAP_S,
    max_clip_s: float = MAX_CLIP_S,
    min_fragment_s: float = MIN_FRAGMENT_S,
) -> list[tuple[float, float]]:
    """Continuous file ranges (≤ ``max_clip_s``) covering the VAD speech fragments.

    Neighbouring fragments share a clip only across a pause ≤ ``merge_gap_s``;
    the pause stays in the audio, so times inside a clip are file times.
    A fragment shorter than ``min_fragment_s`` that merges with nothing is dropped.
    """
    clips: list[tuple[float, float, int]] = []  # start, end, fragments
    for start, end in sorted(fragments):
        if end <= start:
            continue
        if clips:
            first, last, count = clips[-1]
            if start - last <= merge_gap_s and end - first <= max_clip_s:
                clips[-1] = (first, max(last, end), count + 1)
                continue
        # A fragment longer than one clip is split: Whisper reads 30 s at most.
        while end - start > max_clip_s:
            clips.append((start, start + max_clip_s, 1))
            start += max_clip_s
        clips.append((start, end, 1))
    return [(start, end) for start, end, count in clips
            if count > 1 or end - start >= min_fragment_s]


def repair_detached_words(
    transcript: Transcript,
    energy: Energy | None = None,
    *,
    max_word_gap_s: float = MAX_WORD_GAP_S,
) -> Transcript:
    """Split segments at long word gaps and fix 1–2 word series torn from their phrase.

    ``energy(start, end)`` is the RMS level (dBFS) of the source; without it
    nothing is moved and only silence-free evidence is lacking, so detached
    series are left in place.
    """
    pieces: list[list[Word]] = []
    segment_of: list[Segment] = []
    for segment in transcript.segments:
        if not segment.words:
            pieces.append([])
            segment_of.append(segment)
            continue
        current: list[Word] = []
        for word in segment.words:
            if current and word.start - current[-1].end > max_word_gap_s:
                pieces.append(current)
                segment_of.append(segment)
                current = []
            current.append(word)
        if current:
            pieces.append(current)
            segment_of.append(segment)
    suspects: list[dict[str, Any]] = list(transcript.metadata.get("suspect_words", []))
    moved: list[dict[str, Any]] = []
    for index in range(len(pieces) - 1):
        series, continuation = pieces[index], pieces[index + 1]
        if not series or not continuation or len(series) > MAX_SERIES_WORDS:
            continue
        if continuation[0].start - series[-1].end <= max_word_gap_s:
            continue
        same_segment = segment_of[index] is segment_of[index + 1]
        unfinished = not series[-1].text.rstrip().endswith(_SENTENCE_END)
        lowercase = continuation[0].text.lstrip()[:1].islower()
        first_of_segment = index == 0 or segment_of[index - 1] is not segment_of[index]
        if not same_segment or not (unfinished and lowercase or first_of_segment):
            continue
        length = series[-1].end - series[0].start
        target_end = continuation[0].start - 0.02
        if energy is not None and energy(target_end - length - 0.1, target_end) >= SPEECH_DB:
            shift = target_end - series[-1].end
            pieces[index] = [replace(word, start=word.start + shift, end=word.end + shift)
                             for word in series]
            moved.append({"text": " ".join(word.text for word in series),
                          "from": round(series[0].start, 3),
                          "to": round(series[0].start + shift, 3)})
        elif energy is not None and energy(series[0].start, series[-1].end) < SPEECH_DB:
            pieces[index] = [replace(word, probability=min(word.probability,
                                                           SUSPECT_PROBABILITY))
                             for word in series]
            suspects.append({"text": " ".join(word.text for word in series),
                             "start": round(series[0].start, 3),
                             "reason": "silent where Whisper placed it"})
    segments = _segments(pieces, segment_of)
    metadata = dict(transcript.metadata)
    metadata["segmentation"] = SEGMENTATION_VERSION
    if suspects:
        metadata["suspect_words"] = suspects
    if moved:
        metadata["moved_words"] = [*metadata.get("moved_words", []), *moved]
    return replace(transcript, segments=segments, metadata=metadata)


def _segments(pieces: list[list[Word]], segment_of: list[Segment]) -> list[Segment]:
    """Segments rebuilt from word pieces; a moved series rejoins its continuation."""
    groups: list[list[Word]] = []
    origins: list[Segment] = []
    for words, origin in zip(pieces, segment_of, strict=True):
        if (groups and words and groups[-1] and origins[-1] is origin
                and 0 <= words[0].start - groups[-1][-1].end <= MAX_WORD_GAP_S):
            groups[-1] = [*groups[-1], *words]
            continue
        groups.append(list(words))
        origins.append(origin)
    result = [_rebuilt(origin, tuple(words)) for words, origin in zip(groups, origins,
                                                                       strict=True)]
    result.sort(key=lambda segment: segment.start)
    return result


def _rebuilt(origin: Segment, words: tuple[Word, ...]) -> Segment:
    if words == origin.words:
        return origin
    return Segment(words[0].start, words[-1].end, words, origin.confidence,
                   origin.avg_logprob, None, origin.raw_words)


def suspect_count(transcript: Transcript) -> int:
    return len(transcript.metadata.get("suspect_words", []))
