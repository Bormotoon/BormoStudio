"""The interval a reel is cut from, fitted to its speech (Forge clip_intervals).

With word timings the edges follow the speech: a short lead-in before the
first word of the opening sentence, a short tail after the last word of the
closing one, and neither reaches into a neighbouring word. The cut, its
subtitles and the QA duration check all use this one interval.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studio.core.transcript_index import TranscriptIndex


@dataclass(frozen=True, slots=True)
class ClipEdges:
    """``lead_in_s``/``tail_s``: silence kept before the first and after the
    last word; ``guard_s``: distance kept from a neighbouring word (Whisper
    timings are accurate to about 0.1 s); ``max_extend_s``: how far an edge may
    grow to reach its sentence's start or end; ``max_trim_s``: the longest
    fragment of a neighbouring sentence an edge drops instead."""

    lead_in_s: float = 0.6
    tail_s: float = 0.3
    guard_s: float = 0.1
    max_extend_s: float = 4.0
    max_trim_s: float = 1.5

    @classmethod
    def from_settings(cls, reels: Mapping[str, Any]) -> ClipEdges:
        defaults = cls()
        return cls(**{
            name: max(0.0, float(reels.get(f"edge_{name}", getattr(defaults, name))))
            for name in ("lead_in_s", "tail_s", "guard_s", "max_extend_s", "max_trim_s")
        })


def speech_edges(
    index: TranscriptIndex, start: float, end: float, edges: ClipEdges
) -> tuple[float, float, float, float] | None:
    """``(speech_start, speech_end, lead, tail)`` of a clip, None without words.

    The span opens on a sentence's first word and closes on one's last word:
    an edge inside a sentence grows to the sentence edge (up to
    ``max_extend_s``) or drops the dangling fragment (up to ``max_trim_s``),
    whichever moves it less. ``lead``/``tail`` are the silence that may be
    added without touching the previous/next word.
    """
    words = index.words
    if not words or end <= start:
        return None
    inside = index.words_between(start, end)
    if not inside:
        return None
    first = words.index(inside[0])
    last = words.index(inside[-1])
    starts = index.sentence_start_words
    # Dropping a fragment must leave the clip most of its speech.
    max_trim = min(edges.max_trim_s, (words[last].end - words[first].start) / 2.0)

    if first not in starts:
        options: list[tuple[float, int]] = []
        opening = starts[bisect.bisect_right(starts, first) - 1]
        grow = words[first].start - words[opening].start
        if grow <= edges.max_extend_s:
            options.append((grow, opening))
        following = bisect.bisect_right(starts, first)
        if following < len(starts) and starts[following] <= last:
            trim = words[starts[following]].start - words[first].start
            if trim <= max_trim:
                options.append((trim + 1e-3, starts[following]))
        if options:
            first = min(options)[1]

    # A word closes a sentence when the next one opens a sentence.
    if last + 1 < len(words) and last + 1 not in starts:
        options = []
        following = bisect.bisect_right(starts, last)
        closing = (starts[following] if following < len(starts) else len(words)) - 1
        grow = words[closing].end - words[last].end
        if grow <= edges.max_extend_s:
            options.append((grow, closing))
        previous = starts[following - 1] - 1
        if previous >= first:
            trim = words[last].end - words[previous].end
            if trim <= max_trim:
                options.append((trim + 1e-3, previous))
        if options:
            last = min(options)[1]

    speech_start, speech_end = words[first].start, words[last].end
    lead = edges.lead_in_s
    if first > 0:
        previous_end = max(w.end for w in words[max(0, first - 8):first])
        lead = min(lead, max(0.0, speech_start - previous_end - edges.guard_s))
    tail = edges.tail_s
    if last + 1 < len(words):
        tail = min(tail, max(0.0, words[last + 1].start - speech_end - edges.guard_s))
    return speech_start, speech_end, lead, tail


def clip_intervals(
    bounds: Sequence[tuple[float, float]], index: TranscriptIndex, edges: ClipEdges,
    *, duration: float,
) -> list[tuple[float, float]]:
    """The interval each clip is cut from.

    Lead-in and tail never reach into a neighbouring clip: towards a
    neighbour they are capped at half the gap, and dropped on a side that
    already overlaps one, so two reels never share the same seconds.
    """
    spans = [speech_edges(index, start, end, edges) or (start, end, 0.0, 0.0)
             for start, end in bounds]
    fitted = [(start, end) for start, end, _, _ in spans]
    result: list[tuple[float, float]] = []
    for position, (start, end, before, after) in enumerate(spans):
        if before > 0 or after > 0:
            for other, (o_start, o_end) in enumerate(fitted):
                if other == position or o_end <= o_start:
                    continue
                if o_end <= start:
                    before = min(before, (start - o_end) / 2.0)
                elif o_start >= end:
                    after = min(after, (o_start - end) / 2.0)
                else:
                    if o_start < start:
                        before = 0.0
                    if o_end > end:
                        after = 0.0
        result.append((max(0.0, start - before), min(duration, end + after)))
    return result
