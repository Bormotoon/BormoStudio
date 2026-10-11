"""Text and acoustic alignment of one clip, each checked against the other."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from studio.core.transcript import Transcript
from studio.stages.sync_acoustic import acoustic_coarse_align
from studio.stages.sync_engine import matching_transcript
from studio.stages.sync_match_settings import MatchSettings
from studio.stages.sync_matcher import align, evaluate_alignment
from studio.stages.sync_models import AlignmentMap

# An acoustic map whose line misses the accepted text anchors by more than this
# (median) sits on a false peak: text words cannot be a second away from their
# place, while a weak correlation peak can be anywhere (sony_run: mvi-0850 was
# placed 9000 s away from its words).
CONTRADICTION_S = 1.0


def align_sources(
    target: Transcript, source: Transcript, target_path: Path, source_path: Path,
    conf: dict[str, Any], *, acoustic_first: bool,
    source_range: tuple[float, float] | None = None, acoustic_search: bool = True,
    skipped_search: str = "locate found no acoustic match in the whole source",
) -> AlignmentMap:
    """``target = offset + k * source``; ``source_range`` bounds the acoustic search.

    ``acoustic_search=False`` skips the waveform scan (``skipped_search`` says
    why): ``locate`` already searched the whole source for this clip and found
    nothing, or text placement searches acoustically only around neighbours
    later; a scan of a 3.75 h recorder costs ~20 min per long clip.

    When both methods give a map they are compared: an acoustic map that agrees
    with the words keeps its millisecond line and takes the text anchors (they
    cover what the acoustic grid cannot hear); one that contradicts the words
    is dropped for the text map.
    """
    settings = MatchSettings()
    reasons = []

    def acoustic() -> AlignmentMap | None:
        if not acoustic_search:
            reasons.append(skipped_search)
            return None
        try:
            fit = acoustic_coarse_align(
                target_path, source_path, target.duration, source.duration,
                grid_s=float(conf.get("acoustic_grid_s", 30.0)),
                window_s=float(conf.get("acoustic_window_s", 8.0)),
                min_sharpness=float(conf.get("acoustic_min_sharpness", 50.0)),
                max_k_deviation=settings.alignment_max_k_deviation,
                min_points=settings.alignment_min_acoustic_points,
                rec_range=source_range,
            )
        except RuntimeError as exc:
            reasons.append(str(exc))
            return None
        if fit is None:
            reasons.append("waveform scan found no unambiguous fit")
            return None
        candidate = AlignmentMap([], fit.offset, fit.k, 0.0, "acoustic",
                                 fit.inliers, fit.span_s)
        verdict = evaluate_alignment(candidate, target.duration, settings)
        if verdict.accepted:
            return candidate
        reasons.append(verdict.reason_text)
        return None

    def text() -> AlignmentMap | None:
        if not target.words or not source.words:
            reasons.append("no words to align" if not target.words
                           else "the source has no words")
            return None
        try:
            candidate = align(matching_transcript(target), matching_transcript(source),
                              settings)
        except ValueError as exc:
            reasons.append(str(exc))
            return None
        verdict = evaluate_alignment(candidate, target.duration, settings)
        if verdict.accepted:
            return candidate
        reasons.append(verdict.reason_text)
        return None

    if acoustic_first or not target.words:
        sound = acoustic()
        if sound is not None:
            words = text() if target.words else None
            return sound if words is None else combine(sound, words)
    words = text()
    if words is not None:
        return words
    if not acoustic_first and target.words:
        sound = acoustic()
        if sound is not None:
            return sound
    raise ValueError("no accepted source alignment: " + "; ".join(reasons))


def combine(sound: AlignmentMap, words: AlignmentMap) -> AlignmentMap:
    """The acoustic line with the text anchors, or the text map when they disagree."""
    anchors = words.anchors or []
    if not anchors:
        return sound
    k = sound.k or 1.0
    misses = np.array([abs(anchor.cam_time - (sound.offset + k * anchor.rec_time))
                       for anchor in anchors])
    miss = float(np.median(misses))
    if miss > CONTRADICTION_S:
        return replace(words, notes=[
            *words.notes,
            f"acoustic match ignored: its line misses the words by {miss:.1f} s "
            f"(acoustic start at {-sound.offset / k:.1f} s of the source, "
            f"text at {-words.offset / (words.k or 1.0):.1f} s)"])
    # The acoustic line is exact; text anchors near it fill what it cannot hear.
    near = [anchor for anchor, value in zip(anchors, misses, strict=True)
            if value <= CONTRADICTION_S]
    span = (max(a.rec_time for a in near) - min(a.rec_time for a in near)) if near else 0.0
    return replace(sound, anchors=near, evidence_span_s=max(sound.evidence_span_s, span),
                   notes=[*sound.notes, f"acoustic line agrees with {len(near)} text anchors "
                          f"(median miss {miss * 1000:.0f} ms)"])
