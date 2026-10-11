"""Piece planning extracted from the frozen WhisperSync pipeline."""

from __future__ import annotations

import numpy as np

from studio.stages.sync_geometry import AlignmentMap, PieceSettings

_SENTENCE_PAD_S = 0.08
_MAP_WINDOW_S = 30.0
_PAUSE_FACTOR_MIN = 0.5
_PAUSE_FACTOR_MAX = 2.0
_MIN_PIECE_S = 0.02
HEAD_TRIM_WARN_S = 1.0
MIN_TEMPO_FACTOR = 0.5
MAX_TEMPO_FACTOR = 2.0

def recorder_word_gaps(rec_words: list[tuple[float, float]]) -> list[float]:
    """Midpoints of the silent gaps between consecutive recorder words, sorted.

    ``rec_words`` is a list of ``(start, end)`` word spans (need not be sorted).
    Used by ``clip_pieces`` to snap piece boundaries away from mid-word cuts —
    see ``_snap_to_word_gap``.
    """
    spans = sorted(rec_words)
    return [(a[1] + b[0]) / 2.0 for a, b in zip(spans, spans[1:], strict=False) if b[0] > a[1]]



def _snap_to_word_gap(rec_time: float, gaps: list[float], max_snap_s: float) -> float:
    """Nudge ``rec_time`` to the nearest word-gap midpoint within ``max_snap_s``.

    A piece boundary that falls in the middle of a spoken word (rather than in
    the silence between words) creates an audible mid-word tempo break — a
    stutter like "подготовил" -> "подга-га-товил" when the neighbouring piece's
    atempo factor differs. Snapping the cut point to the nearest inter-word
    silence removes the artifact without touching any piece's tempo factor
    (unlike the old, now-removed, factor-smoothing approach, which fixed the
    stutter by averaging factors but let speech drift off the picture by up to
    ~1.4s). Returns ``rec_time`` unchanged if no gap is close enough.
    """
    if not gaps:
        return rec_time
    import bisect

    i = bisect.bisect_left(gaps, rec_time)
    candidates = [
        g
        for g in (gaps[i - 1] if i > 0 else None, gaps[i] if i < len(gaps) else None)
        if g is not None
    ]
    if not candidates:
        return rec_time
    best = min(candidates, key=lambda g: abs(g - rec_time))
    return best if abs(best - rec_time) <= max_snap_s else rec_time



def _sentence_blocks(
    rec_words: list[tuple[float, float]], min_pause_s: float
) -> list[tuple[float, float]]:
    """Group recorder words into sentences: a pause of at least ``min_pause_s``
    between consecutive words ends a sentence. Returns ``(start, end)`` spans in
    recorder time, sorted. This is the acoustic definition of a sentence — a
    stretch of speech with no safe cut point inside it — which is exactly what
    the renderer needs (punctuation without an actual pause is not cuttable)."""
    blocks: list[tuple[float, float]] = []
    for start, end in sorted(rec_words):
        if blocks and start - blocks[-1][1] < min_pause_s:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], end))
        else:
            blocks.append((start, end))
    return blocks



def _smoothed_map_at(
    am: AlignmentMap, rec_t: float, window_s: float = _MAP_WINDOW_S
) -> tuple[float, float]:
    """The smoothed drift map evaluated at recorder time ``rec_t``: returns
    ``(cam_time, local_rate)``.

    A weighted local linear regression over the anchors within ``±window_s``
    (tricube weights) — individual anchors carry Whisper's word-timing jitter,
    but a 30-second neighbourhood averages it down to a few milliseconds while
    still following genuine non-linear drift. Falls back to the global RANSAC
    line where the neighbourhood is too thin to fit."""
    k = am.k or 1.0
    candidates = [a for a in am.anchors if abs(a.rec_time - rec_t) <= window_s]
    # Acoustic anchors carry their regime: a step between regimes must not be
    # smoothed into a ramp, so only anchors of the regime at ``rec_t`` count.
    regimes = [(abs(a.rec_time - rec_t), _regime(a)) for a in candidates]
    tagged = [item for item in regimes if item[1] is not None]
    if tagged:
        regime = min(tagged)[1]
        candidates = [a for a in candidates if _regime(a) in (None, regime)]
    near = [(a.rec_time, a.cam_time) for a in candidates]
    if len(near) >= 4:
        rec = np.array([p[0] for p in near])
        cam = np.array([p[1] for p in near])
        span = float(rec.max() - rec.min())
        if span >= 5.0:
            w = (1.0 - (np.abs(rec - rec_t) / window_s) ** 3) ** 3
            slope, intercept = np.polyfit(rec, cam, 1, w=np.maximum(w, 1e-6))
            # A locally insane slope (all anchors bunched + jitter) must never
            # leak into a speech factor; keep it within a sane drift range.
            if 0.9 <= slope <= 1.1:
                return float(intercept + slope * rec_t), float(slope)
    return am.offset + k * rec_t, k



def _regime(anchor: object) -> int | None:
    """Regime of an acoustic or jump-map anchor (``acoustic:N``, ``jump:N``)."""
    token = getattr(anchor, "token", "")
    if isinstance(token, str) and token.startswith(("acoustic:", "jump:")):
        return int(token.split(":", 1)[1])
    return None


def _sentence_pieces(
    am: AlignmentMap,
    rec0: float,
    rec1: float,
    rec_words: list[tuple[float, float]],
    config: PieceSettings,
) -> tuple[float, list[tuple[float, float, float]]] | None:
    """Sentence-wise piece plan (strategy 3): cut ONLY between sentences, warp
    speech ONLY at the smoothed drift rate, absorb ALL placement residue in the
    inter-sentence pauses.

    Pieces alternate [pause][sentence][pause][sentence]...[tail], tiling
    ``[rec0, rec1]`` contiguously (no content gap or overlap anywhere — repeats
    are impossible by construction):

    - a SENTENCE piece spans one uncuttable stretch of speech (plus a small
      room-tone pad on each side); its factor is the smoothed local drift rate
      — a fraction of a percent, rendered as a transparent resample. Anchor
      jitter never reaches a speech factor.
    - a PAUSE piece is stationary room tone between sentences; its factor is
      whatever places the NEXT sentence exactly on its smoothed target
      (clamped to [0.5, 2.0] — stretching room tone is inaudible where
      stretching speech is not). Placement error therefore dies in every
      pause instead of accumulating.

    Returns ``None`` when no sentence overlaps the span (caller falls back to
    a single global piece).
    """
    k = am.k or 1.0
    blocks = [
        (max(s, rec0), min(e, rec1))
        for s, e in _sentence_blocks(rec_words, config.phrase_gap_threshold)
        if e > rec0 and s < rec1
    ]
    blocks = [(s, e) for s, e in blocks if e - s > _MIN_PIECE_S]
    if not blocks:
        return None

    # Pad each sentence into the surrounding pause (never past the neighbour).
    padded: list[tuple[float, float]] = []
    for i, (s, e) in enumerate(blocks):
        lo = blocks[i - 1][1] if i > 0 else rec0
        hi = blocks[i + 1][0] if i + 1 < len(blocks) else rec1
        padded.append((max(s - _SENTENCE_PAD_S, lo, rec0), min(e + _SENTENCE_PAD_S, hi, rec1)))

    # Smoothed target position + local rate for every sentence start.
    targets: list[tuple[float, float]] = [_smoothed_map_at(am, s) for s, _e in padded]

    pieces: list[tuple[float, float, float]] = []
    lead = 0.0
    local = 0.0  # running output (camera-local) time after `lead`

    # Head room tone before the first sentence: stretch it to put sentence 0 on
    # target; trim it (cutting silence is free) when even max compression can't
    # fit, pad with lead silence when there isn't enough of it.
    head_in = padded[0][0] - rec0
    head_out = max(targets[0][0], 0.0)
    head_start = rec0
    if head_out <= _MIN_PIECE_S:
        head_in = 0.0
    elif head_in > _MIN_PIECE_S:
        # Whatever will not fit in the available output at maximum compression
        # is necessarily left out — the geometry allows nothing else. See
        # `head_trim_seconds`: the caller reports how much was dropped, because
        # "no transcribed words here" is not the same as "silence here".
        used = min(head_in, head_out * _PAUSE_FACTOR_MAX)
        head_start = padded[0][0] - used
        factor = max(_PAUSE_FACTOR_MIN, min(_PAUSE_FACTOR_MAX, used / head_out))
        out = used / factor
        lead = max(0.0, head_out - out)
        pieces.append((head_start, used, factor))
        local = out
    else:
        lead = head_out

    for j, (s, e) in enumerate(padded):
        if j > 0:
            # Pause piece between sentence j-1 and j: absorb the residue.
            pause_in = s - padded[j - 1][1]
            needed = max(targets[j][0] - lead - local, _MIN_PIECE_S)
            if pause_in > _MIN_PIECE_S:
                factor = max(_PAUSE_FACTOR_MIN, min(_PAUSE_FACTOR_MAX, pause_in / needed))
                out = pause_in / factor
                pieces.append((padded[j - 1][1], pause_in, factor))
                local += out
        # Sentence piece: transparent conform at the smoothed local rate only.
        rate = targets[j][1]
        in_dur = e - s
        factor = max(0.5, min(2.0, 1.0 / rate if rate else 1.0))
        pieces.append((s, in_dur, factor))
        local += in_dur / factor

    # Tail room tone after the last sentence, at the global rate (the assembly
    # pads/trims to the exact clip length anyway).
    tail_in = rec1 - padded[-1][1]
    if tail_in > _MIN_PIECE_S:
        pieces.append((padded[-1][1], tail_in, max(0.5, min(2.0, k and 1.0 / k or 1.0))))

    return lead, pieces



def _drop_implausible_breakpoints(
    bps: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Remove interior breakpoints whose piece would need an out-of-range tempo.

    Removing a breakpoint merges the two pieces it separated, so the surviving
    plan still tiles exactly the same recorder span onto exactly the same
    output span — the invariant a clamp destroys. Repeats until every remaining
    piece is plausible or only the two clip edges are left (which is the Global
    Linear fallback, and is always self-consistent).
    """
    if len(bps) <= 2:
        return bps
    kept = list(bps)
    changed = True
    while changed and len(kept) > 2:
        changed = False
        for i in range(len(kept) - 1):
            (ra, la), (rb, lb) = kept[i], kept[i + 1]
            in_dur, out_dur = rb - ra, lb - la
            factor = float("inf") if out_dur <= 1e-4 else in_dur / out_dur
            if MIN_TEMPO_FACTOR <= factor <= MAX_TEMPO_FACTOR:
                continue
            # Drop whichever endpoint of this piece is interior; prefer the
            # later one so earlier (already-validated) pieces stay put.
            drop = i + 1 if i + 1 < len(kept) - 1 else i
            if drop == 0 or drop == len(kept) - 1:
                # Both endpoints are clip edges: nothing left to drop. The
                # global stretch itself is out of range, which the caller
                # surfaces via validate_pieces.
                return kept
            del kept[drop]
            changed = True
            break
    return kept



def validate_pieces(
    lead: float,
    pieces: list[tuple[float, float, float]],
    clip_duration: float,
    rec_duration: float,
    tol_s: float = 1e-3,
) -> list[str]:
    """Structural problems in a render plan, as human-readable strings.

    The invariants a piece list must satisfy for the rendered audio to land
    where the plan says it does:

    * every source duration and tempo factor is positive and finite;
    * every piece reads inside the recorder;
    * the output tiles ``[lead, lead + sum(in_dur / factor)]`` without exceeding
      the clip — ``sum(in_dur / factor)`` is the ONLY thing that decides where
      the last piece's speech lands, so a plan whose total output length
      disagrees with the clip has already lost sync somewhere in the middle.

    Returns an empty list when the plan is sound. Used as a runtime guard and
    directly in tests, where checking "the boundary moved" is not the same
    question as "the audio ends up in the right place".
    """
    problems: list[str] = []
    if lead < -tol_s:
        problems.append(f"negative lead silence ({lead:.4f}s)")
    total_out = lead
    for i, (rs, rd, factor) in enumerate(pieces):
        if not (rd > 0 and factor > 0):
            problems.append(f"piece {i}: non-positive duration/factor ({rd:.4f}s, {factor:.4f})")
            continue
        if not (MIN_TEMPO_FACTOR - 1e-9 <= factor <= MAX_TEMPO_FACTOR + 1e-9):
            problems.append(f"piece {i}: tempo factor {factor:.4f} outside atempo range")
        if rs < -tol_s or rs + rd > rec_duration + tol_s:
            problems.append(
                f"piece {i}: reads [{rs:.3f}, {rs + rd:.3f}]s outside the "
                f"{rec_duration:.3f}s recorder"
            )
        total_out += rd / factor
    if total_out > clip_duration + tol_s:
        problems.append(
            f"pieces occupy {total_out:.4f}s of a {clip_duration:.4f}s clip "
            f"(overflow {total_out - clip_duration:.4f}s)"
        )
    return problems



def head_trim_seconds(pieces: list[tuple[float, float, float]], rec0: float) -> float:
    """Recorder seconds the plan never reads, before its first piece.

    The sentence planner places the first recognised sentence on its target and
    compresses the room tone before it; whatever cannot fit even at maximum
    compression is left out. That is forced by the geometry — there is nowhere
    else for it to go — but it is NOT necessarily silence. Whisper drops quiet,
    accented or overlapping speech, so a dropped head can contain a real
    opening phrase, and the difference between "trimmed room tone" and "deleted
    speech" is invisible from inside the planner (which has no audio, only word
    timings). Measuring it lets the caller say so rather than the user
    discovering a missing sentence in the edit.
    """
    if not pieces:
        return 0.0
    return max(0.0, pieces[0][0] - rec0)



def clip_pieces(
    am: AlignmentMap,
    clip_duration: float,
    rec_duration: float,
    strategy_id: int,
    config: PieceSettings,
    rec_word_gaps: list[float] | None = None,
    rec_words: list[tuple[float, float]] | None = None,
) -> tuple[float, list[tuple[float, float, float]]]:
    """Contiguous recorder pieces that tile a camera clip, for a continuous warp.

    Returns ``(lead_silence, pieces)`` where each piece is
    ``(rec_start, rec_in_duration, atempo_factor)`` and pieces are in playback
    order with no gaps between them — the recorder span for the clip is simply
    time-stretched (globally or piecewise between sync points) so its speech
    lands under the picture. ``lead_silence`` is the silence (seconds) before the
    first piece, non-zero only when the recorder does not reach the clip start.

    Strategy controls the breakpoint density: 1 = one global stretch (Global
    Linear), 2 = a piece per anchor (Local Time-Stretch, tightest), 3 = a piece
    per phrase (Hybrid — gentle per-phrase stretch, smoother, fewer seams; the
    recommended default).

    ``rec_word_gaps`` (recorder inter-word silence midpoints, from
    ``recorder_word_gaps``) lets interior breakpoints snap away from mid-word
    cuts — see ``_snap_to_word_gap``. Optional so callers/tests that don't have
    a transcript handy can omit it (breakpoints then land exactly on anchors,
    as before).
    """
    k = am.k or 1.0

    def l2r(t_local: float) -> float:
        return (t_local - am.offset) / k

    def r2l(t_rec: float) -> float:
        return am.offset + k * t_rec

    rec0 = min(max(l2r(0.0), 0.0), rec_duration)
    rec1 = min(max(l2r(clip_duration), 0.0), rec_duration)
    if rec1 - rec0 <= 1e-3:
        return 0.0, []

    # Interior breakpoints come from the REAL matched word times — each is
    # (recorder_time, local_clip_time), where local time is the anchor's camera
    # time. Warping between these tracks the actual (non-linear) drift instead of
    # a single global slope.
    pts = sorted(
        (a.rec_time, a.cam_time)
        for a in am.anchors
        if rec0 < a.rec_time < rec1 and 0.0 <= a.cam_time <= clip_duration
    )

    # Strategy 1 (or no usable anchors): one global stretch across the clip.
    if strategy_id == 1 or not pts:
        out_dur = r2l(rec1) - r2l(rec0)
        lead = max(0.0, r2l(rec0))
        if out_dur <= 1e-3:
            return lead, []
        return lead, [(rec0, rec1 - rec0, (rec1 - rec0) / out_dur)]

    # Strategy 3 (Hybrid): sentence-wise rendering — cut ONLY in the real
    # pauses between sentences, conform speech ONLY at the smoothed drift
    # rate, absorb all placement residue in the pause pieces. Falls back to
    # the anchor-thinning path when the recorder's word list isn't available
    # (older callers/tests).
    if strategy_id == 3 and rec_words:
        sentence_plan = _sentence_pieces(am, rec0, rec1, rec_words, config)
        if sentence_plan is not None:
            return sentence_plan

    if strategy_id == 3:
        spacing = max(config.phrase_gap_threshold, 1.0)
        thinned: list[tuple[float, float]] = []
        for rt, ct in pts:
            if not thinned or rt - thinned[-1][0] >= spacing:
                thinned.append((rt, ct))
        pts = thinned

    # Snap each interior breakpoint's recorder time to the nearest inter-word
    # silence (seam-snap-to-silence), so no piece boundary lands mid-word. The
    # camera-time side moves WITH it (scaled by the clip's global rate): moving
    # only the recorder side used to change one neighbour's input length while
    # both output lengths stayed put, kicking the two adjacent tempo factors
    # apart by up to ±30% — an audible tempo see-saw at every snapped seam.
    if rec_word_gaps:
        snapped: list[tuple[float, float]] = []
        for rt, ct in pts:
            nrt = _snap_to_word_gap(rt, rec_word_gaps, config.seam_snap_max_s)
            snapped.append((nrt, ct + (nrt - rt) * k))
        pts = sorted(snapped)

    # Clip edges use the global line; interior uses matched word times. Keep only
    # strictly-increasing (rec, local) breakpoints so every piece is sane.
    raw_bps = [(rec0, r2l(rec0)), *pts, (rec1, r2l(rec1))]
    bps: list[tuple[float, float]] = []
    for rt, lt in raw_bps:
        if not bps or (rt > bps[-1][0] + 1e-3 and lt > bps[-1][1] + 1e-3):
            bps.append((rt, lt))

    # An interior breakpoint implying a tempo outside atempo's usable range is
    # not a piece to be clamped — it is an anchor pair that disagrees with its
    # neighbours, i.e. a bad match. Clamping it kept the WRONG breakpoint and
    # silently broke the timeline geometry instead: a piece asked to play
    # `in_dur` of recorder at a clamped factor no longer occupies `out_dur` of
    # output, so the pieces stopped tiling the clip (a synthetic 10 s clip came
    # out 9.8 s long) and the trailing pad papered over the gap without putting
    # any speech back where it belonged. Dropping the breakpoint MERGES its two
    # neighbours, which keeps `sum(in_dur / factor) == total output` exact by
    # construction. See ``validate_pieces``.
    bps = _drop_implausible_breakpoints(bps)

    pieces: list[tuple[float, float, float]] = []
    for i in range(len(bps) - 1):
        (ra, la), (rb, lb) = bps[i], bps[i + 1]
        in_dur = rb - ra
        out_dur = lb - la
        if in_dur <= 1e-4 or out_dur <= 1e-4:
            continue
        pieces.append((ra, in_dur, in_dur / out_dur))

    lead = max(0.0, bps[0][1])
    return lead, pieces

