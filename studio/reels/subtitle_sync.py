"""Check and fix reel caption timings with a second Whisper pass (Forge subtitle_sync).

Transcript word timings come from one pass over the whole recording. Before
captions are made, each reel is listened to again: its own sound is
recognized with word timings and matched to the transcript words. When they
drift apart noticeably the captions take the second pass's timings, unmatched
words keep their place relative to the matched ones around them. When too
little matches (music, crosstalk), the second pass is not trusted.
"""

from __future__ import annotations

import bisect
import difflib
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
# Two differently spelled words still count as the same word above this.
_FUZZY_WORD_RATIO = 0.6
# Whisper lets the word after a pause absorb the pause: the end is reliable,
# the start is not, and for captions the start is what shows a cue early.
_WORD_BASE_S = 0.2
_WORD_PER_CHAR_S = 0.11


@dataclass(frozen=True, slots=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class ClipSyncReport:
    reference_words: int = 0
    heard_words: int = 0
    matched_words: int = 0
    match_ratio: float = 0.0
    median_shift_s: float = 0.0
    mean_abs_shift_s: float = 0.0
    p95_abs_shift_s: float = 0.0
    max_abs_shift_s: float = 0.0
    applied: bool = False
    verdict: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {key: round(value, 3) if isinstance(value, float) else value
                for key, value in asdict(self).items()}


def plausible_start(start: float, end: float, text: str) -> float:
    """``start`` moved up to the latest point the word could plausibly begin."""
    letters = sum(1 for char in text if char.isalnum())
    return max(start, end - (_WORD_BASE_S + _WORD_PER_CHAR_S * max(1, letters)))


def normalize_word(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum()).replace("ё", "е")


def match_words(reference: Sequence[TimedWord], heard: Sequence[TimedWord]) -> dict[int, int]:
    """Reference word index to heard word index, in order: exact normalized
    matches, and one-to-one pairs of alike words inside equal-length replacements."""
    ref = [normalize_word(word.text) for word in reference]
    got = [normalize_word(word.text) for word in heard]
    pairs: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, ref, got, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.update({i1 + offset: j1 + offset for offset in range(i2 - i1)})
        elif tag == "replace" and i2 - i1 == j2 - j1:
            for offset in range(i2 - i1):
                a, b = ref[i1 + offset], got[j1 + offset]
                if a and b and difflib.SequenceMatcher(None, a, b).ratio() >= _FUZZY_WORD_RATIO:
                    pairs[i1 + offset] = j1 + offset
    return pairs


def retime(
    reference: Sequence[TimedWord], heard: Sequence[TimedWord], pairs: dict[int, int]
) -> list[TimedWord]:
    """Matched words take the heard span; an unmatched word moves by the shift
    interpolated between the matched words on either side."""
    if not pairs:
        return list(reference)
    anchors = sorted(pairs)
    shift_at = {i: heard[pairs[i]].start - reference[i].start for i in anchors}

    def shift(i: int) -> float:
        if i <= anchors[0]:
            return shift_at[anchors[0]]
        if i >= anchors[-1]:
            return shift_at[anchors[-1]]
        position = bisect.bisect_left(anchors, i)
        left, right = anchors[position - 1], anchors[position]
        t0, t1 = reference[left].start, reference[right].start
        weight = (reference[i].start - t0) / (t1 - t0) if t1 > t0 else 0.5
        return shift_at[left] + (shift_at[right] - shift_at[left]) * weight

    out: list[TimedWord] = []
    previous_start = 0.0
    for i, word in enumerate(reference):
        if i in pairs:
            start, end = heard[pairs[i]].start, heard[pairs[i]].end
        else:
            delta = shift(i)
            start, end = word.start + delta, word.end + delta
        start = max(start, previous_start)
        end = max(end, start + 0.01)
        out.append(TimedWord(round(start, 3), round(end, 3), word.text))
        previous_start = start
    return out


def _percentile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))] if values else 0.0


def judge(
    reference: Sequence[TimedWord], heard: Sequence[TimedWord], *,
    min_match_ratio: float, apply_threshold_s: float,
) -> tuple[list[TimedWord], ClipSyncReport]:
    """The words to caption with and what the second pass found."""
    report = ClipSyncReport(reference_words=len(reference), heard_words=len(heard))
    if not reference:
        report.verdict = "no_words"
        return list(reference), report
    pairs = match_words(reference, heard)
    report.matched_words = len(pairs)
    report.match_ratio = len(pairs) / len(reference)
    # Drift is measured on word ends: both passes agree on them when in sync.
    shifts = [heard[j].end - reference[i].end for i, j in pairs.items()]
    if shifts:
        magnitudes = [abs(shift) for shift in shifts]
        report.median_shift_s = _percentile(shifts, 0.5)
        report.mean_abs_shift_s = sum(magnitudes) / len(magnitudes)
        report.p95_abs_shift_s = _percentile(magnitudes, 0.95)
        report.max_abs_shift_s = max(magnitudes)
    if report.match_ratio < min_match_ratio:
        report.verdict = "unreliable"
        return list(reference), report
    if report.p95_abs_shift_s < apply_threshold_s:
        report.verdict = "in_sync"
        return list(reference), report
    report.verdict = "retimed"
    report.applied = True
    return retime(reference, heard, pairs), report


def heard_words(model: Any, media: Path, language: str | None) -> list[TimedWord]:
    """What Whisper hears in ``media`` (its first audio stream), in its own time."""
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True, check=False, timeout=600,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-300:] or "ffmpeg failed")
    segments, _info = model.transcribe(
        np.frombuffer(result.stdout, dtype=np.float32), language=language or None,
        word_timestamps=True, vad_filter=False, condition_on_previous_text=False, beam_size=5,
    )
    words = []
    for segment in segments:
        for word in segment.words or ():
            text = str(word.word).strip()
            if text and word.end > word.start:
                words.append(TimedWord(plausible_start(float(word.start), float(word.end), text),
                                       float(word.end), text))
    return words


def load_model(name: str, device: str = "auto", compute_type: str = "auto") -> Any:
    """A faster-whisper model by catalog name or path, CPU when the GPU fails."""
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    from studio.modules.models import whisper_model_source

    if device == "auto":
        try:
            import ctranslate2  # type: ignore[import-not-found]

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except ImportError:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    try:
        return WhisperModel(whisper_model_source(name), device=device, compute_type=compute_type)
    except (RuntimeError, ValueError):
        if device == "cpu":
            raise
        return WhisperModel(whisper_model_source(name), device="cpu", compute_type="int8")
