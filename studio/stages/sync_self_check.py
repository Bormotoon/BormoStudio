"""Post-render self-check: verify a rendered voice monolith against the
camera clip's own transcript, with an optional repair pass.

``--verify``/``tools/verify_sync.py`` already measures realized LAG between
the rendered voice and the camera audio via GCC-PHAT cross-correlation — but
it is blind to CONTENT defects (a dropped word, a duplicated phrase, a piece
built from the wrong recorder span): it only measures *where* two waveforms
correlate, so a piece with wrong-but-present speech can still show a small
lag if enough of the surrounding audio still lines up.

This module closes that gap by re-transcribing the rendered voice monolith
with Whisper and comparing its words against the camera clip's OWN
transcript (already computed during alignment — the natural "original" to
check against, not the recorder). The two transcripts should describe the
same speech at (very nearly) the same local timestamps if the render is
correct; a span where they diverge — either a timing drift beyond normal
cross-run Whisper jitter, or words that simply don't match — is flagged.

Two modes build on the same detection: "warn" (the default) just reports
flagged spans for the user to check in the NLE; "repair" additionally
re-aligns each flagged span's own small stretch of recorder audio (transcript
re-match first, falling back to a local GCC-PHAT re-check the same way the
acoustic fallback / Boundary Flex do) and re-renders only the pieces inside
that span — the rest of the clip is untouched. Both a `shifted` (timing-only)
and a `content` (wrong/missing words) span get the SAME treatment: an ffmpeg
render is deterministic, so re-rendering the same recorder span verbatim
would reproduce a content defect byte-for-byte — the only thing that can fix
either kind of defect is re-deriving where in the recorder this piece of
speech actually comes from, i.e. a fresh local alignment.
"""

from __future__ import annotations

import bisect
import difflib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from studio.stages.sync_check_settings import SelfCheckSettings
from studio.stages.sync_matcher import normalize_token
from studio.stages.sync_models import AlignmentMap, Word


class TimedWord(Protocol):
    """What the diagnosis reads: Studio and WhisperSync words both qualify."""

    @property
    def text(self) -> str: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...

logger = logging.getLogger(__name__)

__all__ = [
    "SelfCheckOutcome",
    "SelfCheckSpan",
    "check_rendered_clip",
    "diagnose",
    "diagnose_words",
    "realign_span",
]

# A run of at least this many consecutive matched words with a median timing
# delta above this many seconds is flagged as "shifted". Calibrated on a real
# 7-minute clip whose measured acoustic lag was <25 ms everywhere (i.e. any
# span flagged there is a false positive): comparing a clean-recorder
# transcription against a noisy-camera transcription of the same speech
# produces word-timing disagreements well past the naive ±100 ms estimate —
# echo smears word onsets by 300-500 ms — so the deltas use min-over-edges
# and wrong-occurrence rejection (see _shifted_spans), and these values are
# the softest ones that produced ZERO false spans on that material.
DEFAULT_MIN_RUN_WORDS = 5
DEFAULT_SHIFT_THRESHOLD_S = 0.35

# A content run (words present on one side with no match on the other) at
# least this long is flagged as "content" — short gaps are routine Whisper
# disagreement (the camera's echoey audio makes it mishear or drop words a
# clean recorder track keeps: 3-word one-sided runs were common on real
# material with perfect acoustic sync), not evidence of wrong render audio.
DEFAULT_MIN_CONTENT_WORDS = 5


@dataclass
class SelfCheckSpan:
    """One flagged span of a rendered clip's local timeline.

    ``kind`` is ``"shifted"`` (both sides have matching words, but their
    timing disagrees by more than normal jitter) or ``"content"`` (a run of
    words present on one side has no counterpart on the other at all).
    """

    start: float
    end: float
    kind: str
    detail: str


@dataclass
class SelfCheckOutcome:
    """What the check actually established — not just what it flagged.

    An empty span list used to mean "no problems", which made the check's two
    genuinely different failure modes indistinguishable from success:

    * a render containing NO speech at all against a camera clip with six
      words returned ``[]`` and was reported as clean. Total content loss —
      the single worst defect this check exists to catch — read as a pass.
    * a camera clip with no usable transcript also returned ``[]``, so "we
      could not check this" and "we checked this and it is fine" were the same
      answer.

    ``status`` separates them: ``passed``, ``failed`` (defects found, listed in
    ``spans``), or ``inconclusive`` (no trustworthy reference to compare
    against — the user is told, and nothing is claimed).
    """

    status: str  # "passed" | "failed" | "inconclusive"
    spans: list[SelfCheckSpan] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "passed"


@dataclass
class _Tok:
    norm: str
    start: float
    end: float
    text: str


def _tokens(words: Sequence[TimedWord]) -> list[_Tok]:
    out: list[_Tok] = []
    for w in sorted(words, key=lambda x: x.start):
        norm = normalize_token(w.text)
        if norm:
            out.append(_Tok(norm=norm, start=w.start, end=w.end, text=w.text.strip()))
    return out


def _median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2]


def _has_same_time_counterpart(
    norm: str, t: float, cam_times_by_norm: dict[str, list[float]], tol_s: float
) -> bool:
    """Whether the camera transcript contains the SAME normalized token
    within ``tol_s`` of time ``t``. Used to reject wrong-occurrence matches:
    difflib's LCS happily pairs a common word (or a whole repeated phrase —
    a retake!) with a far-away occurrence of the same text, producing a huge
    but fake timing delta. If the correct-time counterpart exists, the match
    simply went to the wrong twin — the audio at ``t`` is fine."""
    times = cam_times_by_norm.get(norm)
    if not times:
        return False
    i = bisect.bisect_left(times, t)
    return any(0 <= j < len(times) and abs(times[j] - t) <= tol_s for j in (i - 1, i))


def _shifted_spans(
    render_toks: list[_Tok],
    cam_toks: list[_Tok],
    equal_pairs: list[tuple[int, int]],
    min_run_words: int,
    shift_threshold_s: float,
) -> list[SelfCheckSpan]:
    """Flag stretches of consecutive matched word-pairs whose LOCAL timing
    delta exceeds ``shift_threshold_s``. ``equal_pairs`` are (render_idx,
    cam_idx) index pairs for tokens difflib judged equal, in matched order.

    A single greedy run merged across the whole matched sequence (grouping
    only by index adjacency) lets a genuinely shifted tail get diluted by a
    long well-synced prefix sharing the same run — its combined median could
    sit under the threshold even though the tail alone is clearly off. A
    sliding window instead judges every ``min_run_words``-wide neighbourhood
    on its own median, so a local anomaly is caught regardless of what comes
    before or after it; overlapping/adjacent flagged windows are then merged
    into one reported span.

    A pair whose delta exceeds the threshold is first checked for a
    same-time counterpart (``_has_same_time_counterpart``): validated on a
    real 7-minute clip, every large "shift" difflib produced was a common
    word or repeated phrase matched to the wrong occurrence (deltas up to
    5.9 s on a clip whose measured acoustic lag was <25 ms everywhere) — a
    genuine render defect means the content is NOT present at the right
    time, so the existence of a right-time twin clears the pair.
    """
    n = len(equal_pairs)
    if n < min_run_words:
        return []
    cam_times_by_norm: dict[str, list[float]] = {}
    for t in cam_toks:
        cam_times_by_norm.setdefault(t.norm, []).append(t.start)
    for times in cam_times_by_norm.values():
        times.sort()

    deltas: list[float] = []
    for ri, ci in equal_pairs:
        # min over both word edges: Whisper's start times are its noisiest
        # output (breath/onset ambiguity, echo smearing the attack on the
        # camera track can move a start by 300-500 ms between two honest
        # transcriptions of the same speech), while a REAL placement shift
        # moves both edges together — so a word is "in place" if EITHER edge
        # agrees.
        delta = min(
            abs(render_toks[ri].start - cam_toks[ci].start),
            abs(render_toks[ri].end - cam_toks[ci].end),
        )
        if delta > shift_threshold_s and _has_same_time_counterpart(
            render_toks[ri].norm, render_toks[ri].start, cam_times_by_norm, shift_threshold_s
        ):
            delta = 0.0  # wrong-occurrence match; the right-time twin exists
        deltas.append(delta)

    flagged: list[tuple[int, int]] = []  # (start_idx, end_idx) into equal_pairs, inclusive
    for i in range(n - min_run_words + 1):
        window = deltas[i : i + min_run_words]
        if _median(window) > shift_threshold_s:
            flagged.append((i, i + min_run_words - 1))

    # Merge adjacent/overlapping windows (and windows separated only by a
    # small index gap in equal_pairs, i.e. a couple of unmatched words) into
    # single spans.
    spans: list[SelfCheckSpan] = []
    k = 0
    while k < len(flagged):
        lo, hi = flagged[k]
        k += 1
        while k < len(flagged) and flagged[k][0] <= hi + 1:
            hi = max(hi, flagged[k][1])
            k += 1
        run = deltas[lo : hi + 1]
        median = _median(run)
        ri0, _ci0 = equal_pairs[lo]
        ri1, _ci1 = equal_pairs[hi]
        spans.append(
            SelfCheckSpan(
                start=render_toks[ri0].start,
                end=render_toks[ri1].end,
                kind="shifted",
                detail=(
                    f"{hi - lo + 1} word(s) timed {median * 1000:.0f} ms off "
                    "from the camera transcript"
                ),
            )
        )
    return spans


def _content_spans(
    render_toks: list[_Tok],
    cam_toks: list[_Tok],
    opcodes: Sequence[tuple[str, int, int, int, int]],
    min_content_words: int,
) -> list[SelfCheckSpan]:
    """Flag long replace/delete/insert runs from the difflib opcodes as
    likely content mismatches (missing/duplicated/wrong speech), as opposed
    to the routine 1-2 word disagreements between two independent Whisper
    passes over similar audio."""
    spans: list[SelfCheckSpan] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        render_span = i2 - i1
        cam_span = j2 - j1
        if max(render_span, cam_span) < min_content_words:
            continue
        if render_span > 0:
            start = render_toks[i1].start
            end = render_toks[i2 - 1].end
        else:
            # A pure insert on the camera side: speech the camera heard that
            # the render does not contain at all. It has no render-side
            # timestamps, so the span must come from the CAMERA tokens that are
            # missing — using the neighbouring rendered word's boundary instead
            # collapsed the whole defect onto that word. A render keeping only
            # [0, 0.2] of a clip whose camera has six more words out to 20 s
            # reported a span of [0.2, 0.2]: a zero-length flag at the start of
            # the clip for a twenty-second hole at the end, which no repair
            # could act on. The missing words' own times say where the audio
            # should have been.
            start = cam_toks[j1].start
            end = cam_toks[j2 - 1].end
            if i1 > 0:
                # Never claim the defect begins before the last word that DID
                # render correctly.
                start = max(start, render_toks[i1 - 1].end)
            end = max(end, start)
        spans.append(
            SelfCheckSpan(
                start=start,
                end=end,
                kind="content",
                detail=(
                    f"{render_span} rendered word(s) vs {cam_span} camera word(s) "
                    f"don't match ({tag})"
                ),
            )
        )
    return spans


def diagnose(
    render_words: Sequence[TimedWord],
    cam_words: Sequence[TimedWord],
    min_run_words: int = DEFAULT_MIN_RUN_WORDS,
    shift_threshold_s: float = DEFAULT_SHIFT_THRESHOLD_S,
    min_content_words: int = DEFAULT_MIN_CONTENT_WORDS,
    clip_duration: float | None = None,
) -> SelfCheckOutcome:
    """Compare a rendered clip's transcript against the camera clip's own and
    say what was established: ``passed``, ``failed`` (with the offending spans)
    or ``inconclusive``.

    Pure function over word lists — kept separate from ``check_rendered_clip``
    (which does the transcription I/O) so it's directly unit-testable against
    synthetic word lists.

    The three outcomes exist because "no spans" was previously used for all
    three situations. In particular an EMPTY render against a camera clip full
    of speech produced no spans and was reported as clean — the most severe
    possible defect, total content loss, reading as a pass.
    """
    render_toks = _tokens(render_words)
    cam_toks = _tokens(cam_words)

    if not cam_toks:
        # No reference: nothing can be concluded either way. Saying so is the
        # honest answer; saying "passed" would be a claim the data cannot back.
        return SelfCheckOutcome(
            "inconclusive",
            [],
            "the camera clip has no usable transcript to check against",
        )

    if not render_toks:
        end = clip_duration if clip_duration is not None else cam_toks[-1].end
        return SelfCheckOutcome(
            "failed",
            [
                SelfCheckSpan(
                    start=0.0,
                    end=end,
                    kind="content",
                    detail=(
                        f"the rendered audio contains no speech at all, but the camera "
                        f"clip has {len(cam_toks)} word(s) — the voice for this clip is missing"
                    ),
                )
            ],
            "empty render against a non-empty reference",
        )

    matcher = difflib.SequenceMatcher(
        None, [t.norm for t in render_toks], [t.norm for t in cam_toks], autojunk=False
    )
    opcodes = matcher.get_opcodes()

    equal_pairs: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            continue
        equal_pairs.extend(zip(range(i1, i2), range(j1, j2), strict=True))

    spans = _shifted_spans(
        render_toks, cam_toks, equal_pairs, min_run_words, shift_threshold_s
    ) + _content_spans(render_toks, cam_toks, opcodes, min_content_words)
    spans.sort(key=lambda s: s.start)
    if spans:
        return SelfCheckOutcome("failed", spans, f"{len(spans)} flagged span(s)")
    return SelfCheckOutcome("passed", [], f"{len(equal_pairs)} matched word(s)")


def diagnose_words(
    render_words: list[Word],
    cam_words: list[Word],
    min_run_words: int = DEFAULT_MIN_RUN_WORDS,
    shift_threshold_s: float = DEFAULT_SHIFT_THRESHOLD_S,
    min_content_words: int = DEFAULT_MIN_CONTENT_WORDS,
) -> list[SelfCheckSpan]:
    """``diagnose(...).spans`` — the flagged spans only.

    Prefer ``diagnose``: an empty list here still cannot distinguish "clean"
    from "could not be checked".
    """
    return diagnose(
        render_words,
        cam_words,
        min_run_words=min_run_words,
        shift_threshold_s=shift_threshold_s,
        min_content_words=min_content_words,
    ).spans


def check_rendered_clip(
    rendered_words: Sequence[TimedWord],
    camera_words: Sequence[TimedWord],
    config: object,
    clip_duration: float | None = None,
) -> SelfCheckOutcome:
    """Convenience wrapper reading thresholds off ``config`` (duck-typed to
    avoid importing ``SelfCheckSettings`` here and creating a cycle; the
    pipeline passes the real config)."""
    return diagnose(
        rendered_words,
        camera_words,
        min_run_words=getattr(config, "self_check_min_run_words", DEFAULT_MIN_RUN_WORDS),
        shift_threshold_s=getattr(
            config, "self_check_shift_threshold_s", DEFAULT_SHIFT_THRESHOLD_S
        ),
        min_content_words=getattr(
            config, "self_check_min_content_words", DEFAULT_MIN_CONTENT_WORDS
        ),
        clip_duration=clip_duration,
    )


# Extra recorder-time context added on each side of a flagged span before
# re-aligning it: a repair needs a few real words of surrounding context (not
# just the flagged span itself) to fit a meaningful local line/offset, and a
# few seconds of margin so a re-derived boundary doesn't land exactly on the
# old (possibly wrong) one.
REPAIR_CONTEXT_S = 6.0
# Below this many word anchors, a local transcript re-match isn't trustworthy
# (RANSAC through 2-3 points is a guess, not a fit) — fall back to acoustic.
REPAIR_MIN_ANCHORS = 4


def _local_transcript_realign(
    rec_lo: float,
    rec_hi: float,
    cam_lo: float,
    cam_hi: float,
    cam_words: list[Word],
    rec_words: list[Word],
    config: SelfCheckSettings,
) -> AlignmentMap | None:
    """Re-derive ``(offset, k)`` from ONLY the words inside
    ``[rec_lo, rec_hi]`` (recorder) / ``[cam_lo, cam_hi]`` (camera local
    time) — the same normalize+difflib+RANSAC approach ``matcher.align()``
    uses for a whole clip, restricted to this small window so a bad match
    far away in the clip can't influence the repair."""
    from studio.stages.sync_matcher import (
        _anchors_from_words,
        normalize_words,
        ransac_linear_fit,
        reject_gross_outliers,
    )

    cam_window = [w for w in cam_words if cam_lo <= w.start <= cam_hi]
    rec_window = [w for w in rec_words if rec_lo <= w.start <= rec_hi]
    cam_norm = normalize_words(cam_window, config.anchor_min_confidence)
    rec_norm = normalize_words(rec_window, config.anchor_min_confidence)
    if not cam_norm or not rec_norm:
        return None

    anchors = _anchors_from_words(cam_norm, rec_norm)
    if len(anchors) < REPAIR_MIN_ANCHORS:
        return None
    kept = reject_gross_outliers(anchors)
    if len(kept) < 2:
        kept = anchors
    offset, k, inliers = ransac_linear_fit(kept)
    if len(inliers) < REPAIR_MIN_ANCHORS:
        return None
    residuals_ms = [abs((offset + k * a.rec_time) - a.cam_time) * 1000 for a in inliers]
    residual_ms = float(np.median(residuals_ms)) if residuals_ms else 0.0

    # A repair map re-renders real audio, so it faces the same bar as any other
    # map: a local fit over a handful of anchors inside one flagged span is
    # exactly the situation where a line can be fitted through noise. An
    # implausible clock ratio here would re-render the span at a wrong tempo,
    # replacing a defect the user could hear with one they cannot explain.
    # Declining leaves the span reported as a warning, which is the correct
    # outcome when nothing more trustworthy is available.
    max_dev = config.alignment_max_k_deviation
    if not k or abs(k - 1.0) > max_dev:
        logger.info(
            "Repair re-match rejected: implausible clock ratio k=%.4f (allowed 1±%g)",
            k,
            max_dev,
        )
        return None
    if residual_ms > config.alignment_max_residual_ms:
        logger.info(
            "Repair re-match rejected: residual %.0f ms exceeds %.0f ms",
            residual_ms,
            config.alignment_max_residual_ms,
        )
        return None

    rec_times = [a.rec_time for a in inliers]
    return AlignmentMap(
        anchors=kept,
        offset=offset,
        k=k,
        residual_ms=residual_ms,
        provenance="repair",
        inliers=len(inliers),
        evidence_span_s=max(rec_times) - min(rec_times),
    )


def _local_acoustic_realign(
    cam_lo: float,
    cam_hi: float,
    prior_offset: float,
    prior_k: float,
    cam_audio_wav: Path,
    rec_audio_path: Path,
    config: SelfCheckSettings,
) -> AlignmentMap | None:
    """Fallback when the transcript re-match found too few anchors: a single
    GCC-PHAT cross-correlation over the WHOLE flagged span (not per-boundary
    like Boundary Flex), reusing the current clock map ``prior_offset``/
    ``prior_k`` as the search center — a short span rarely has enough
    duration to fit its own trustworthy K, but the offset alone recovers
    "which part of the recorder this really is".

    Only the needed window (± a little margin) of each track is decoded —
    the recorder can be hours long, and decoding all of it to a mono-16k
    array for one local measurement would dominate the repair's cost.
    """
    from studio.stages.sync_acoustic import _REFINE_SR, _window_slice, gcc_phat, load_mono16k_track

    win = max(cam_hi - cam_lo, 2.0)
    cam_mid = (cam_lo + cam_hi) / 2.0
    # The recorder time this camera moment maps to under the CURRENT (possibly
    # stale) alignment — the acoustic probe searches ±max_lag around it.
    rec_mid_guess = (cam_mid - prior_offset) / prior_k if prior_k else cam_mid
    if rec_mid_guess < 0:
        return None

    # Decode a window around each center (window + max-lag margin on each
    # side), then slice the *decoded* arrays relative to their own origins.
    margin = win / 2.0 + config.acoustic_max_lag_s + 0.5
    cam_start = max(0.0, cam_mid - margin)
    rec_start = max(0.0, rec_mid_guess - margin)
    cam_track = load_mono16k_track(cam_audio_wav, start_s=cam_start, duration_s=2 * margin)
    rec_track = load_mono16k_track(rec_audio_path, start_s=rec_start, duration_s=2 * margin)
    cam_sig = _window_slice(cam_track, cam_mid - cam_start, win, _REFINE_SR)
    rec_sig = _window_slice(rec_track, rec_mid_guess - rec_start, win, _REFINE_SR)
    lag_s, sharp = gcc_phat(cam_sig, rec_sig, _REFINE_SR, config.acoustic_max_lag_s, config.gcc_eps)
    if sharp < config.acoustic_fallback_min_sharpness:
        return None
    rec_mid = rec_mid_guess - lag_s
    offset = cam_mid - prior_k * rec_mid
    # `prior_k` comes from the clip's own already-accepted map, so the ratio is
    # inherited rather than re-estimated — only the offset is measured here.
    return AlignmentMap(
        anchors=[],
        offset=offset,
        k=prior_k,
        residual_ms=0.0,
        provenance="repair",
        inliers=1,
        evidence_span_s=win,
    )


def realign_span(
    span: SelfCheckSpan,
    am: AlignmentMap,
    cam_words: list[Word],
    rec_words: list[Word],
    cam_audio_wav: Path | None,
    rec_audio_path: Path,
    rec_duration: float,
    config: SelfCheckSettings,
) -> AlignmentMap | None:
    """Re-align just the flagged span's own stretch of recorder audio.

    Returns a fresh ``AlignmentMap`` valid ONLY near this span (not the whole
    clip) — the caller re-plans pieces for the span's neighbourhood with it —
    or ``None`` if neither a transcript re-match nor an acoustic re-check
    found anything more confident than what's already there, in which case
    the span is left as a warning instead of risking a worse edit.
    ``cam_audio_wav`` is ``None`` when ``config.boundary_flex`` was off for
    the render (see ``RenderJob.cam_audio``) — the acoustic fallback is
    simply skipped then, since it has no camera audio to cross-correlate.
    """
    k = am.k or 1.0
    cam_lo = max(0.0, span.start - REPAIR_CONTEXT_S)
    cam_hi = span.end + REPAIR_CONTEXT_S
    rec_lo = max(0.0, (cam_lo - am.offset) / k)
    rec_hi = min(rec_duration, (cam_hi - am.offset) / k)
    if rec_hi <= rec_lo:
        return None

    result = _local_transcript_realign(rec_lo, rec_hi, cam_lo, cam_hi, cam_words, rec_words, config)
    if result is not None:
        return result
    if cam_audio_wav is None:
        return None
    return _local_acoustic_realign(
        cam_lo, cam_hi, am.offset, k, cam_audio_wav, rec_audio_path, config
    )
