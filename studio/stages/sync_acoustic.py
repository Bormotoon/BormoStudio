"""Acoustic cross-correlation for sub-frame lip-sync ("Boundary Flex").

Whisper word timings are only ±50–100 ms accurate. This module measures the true
recorder<->camera lag directly from the audio waveform — independent of the
transcript — using PHAT-weighted cross-correlation (GCC-PHAT, ``gcc_phat``), which
is robust to the different mics and reverb of camera vs recorder.

``refine_piece_boundaries`` (Boundary Flex) uses this to acoustically nudge each
rendered piece's recorder start so speech lands under the picture to sub-frame
accuracy, independent of Whisper's word timings. A measurement is only trusted when
its cross-correlation peak is confidently sharp (silence/wind/music windows are
rejected) and the residual exceeds a deadband, so it only removes real drift rather
than injecting GCC measurement noise.
"""

from __future__ import annotations

import logging
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from studio.stages.sync_acoustic_settings import AcousticSettings
from studio.stages.sync_decode import extract_audio_to_wav

logger = logging.getLogger(__name__)

_REFINE_SR = 16000  # all acoustic analysis happens on mono 16 kHz


@dataclass
class AcousticFit:
    """A waveform-derived clock map plus the evidence that produced it.

    Returning the evidence (not just ``offset, k``) is what lets the caller's
    acceptance gate treat an acoustic match on the same footing as a text one:
    a fit from three unambiguous points spanning ten minutes and a fit from two
    points three seconds apart are both "a line", and only the evidence tells
    them apart.
    """

    offset: float
    k: float
    points: int
    inliers: int
    span_s: float


def read_wav_mono16k(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV into a float array in [-1, 1]. Uses the stdlib ``wave``
    module — no scipy/soundfile dep."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        sampwidth = w.getsampwidth()
        nchannels = w.getnchannels()
        raw = w.readframes(n)
    if sampwidth != 2:
        raise ValueError(f"expected pcm_s16le (2-byte) wav, got sampwidth={sampwidth}")
    # float32, not float64: these arrays are only ever used for correlation
    # and windowing, where 24 bits of mantissa is far more precision than
    # 16-bit source samples carry — and they are LARGE. A one-hour recorder is
    # ~230 MB at float32 against ~461 MB at float64, per decode, before
    # intermediate copies. The FFTs promote to complex64 accordingly.
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    # If the file is stereo, fold to mono.
    if nchannels == 2 and data.size:
        data = data.reshape(-1, 2).mean(axis=1)
    return data / np.float32(32768.0), sr


def load_mono16k_track(
    path: Path, start_s: float | None = None, duration_s: float | None = None
) -> np.ndarray:
    """Decode an audio file (any format/channel layout ffmpeg reads) to a
    mono 16 kHz float array, once. Boundary Flex used to re-run ffmpeg (via
    ``extract_audio_window``) for every single boundary it measured — for a
    clip with hundreds of pieces that's hundreds of short-lived ffmpeg
    processes just to cut small windows. Decoding each full track exactly once
    and slicing the resulting numpy array for every window instead removes
    that spawn overhead entirely (the FFT-based ``gcc_phat`` cost dominates
    once ffmpeg is out of the loop). See PROJECT_ANALYSIS.md §6.2.

    ``start_s``/``duration_s`` decode only that window of the file — for a
    caller needing a few seconds of a multi-hour recorder (the self-check
    repair's local acoustic probe), decoding everything would dominate the
    whole operation's cost. Times in the returned array are then relative to
    ``start_s``, not to the file's own zero.
    """
    import tempfile

    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    import os

    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        extract_audio_to_wav(
            path,
            tmp_path,
            sample_rate=_REFINE_SR,
            mono=True,
            start_s=start_s,
            duration_s=duration_s,
        )
        sig, _ = read_wav_mono16k(tmp_path)
        return sig
    finally:
        tmp_path.unlink(missing_ok=True)


def _window_slice(track: np.ndarray, center_s: float, win_s: float, sr: int) -> np.ndarray:
    """A ``win_s``-second slice of ``track`` centered on ``center_s``, clamped to
    the track's bounds (shorter at the edges rather than raising)."""
    half = int(round(win_s / 2.0 * sr))
    c = int(round(center_s * sr))
    lo = max(0, c - half)
    hi = min(len(track), c + half)
    return track[lo:hi]


def gcc_phat(
    sig_ref: np.ndarray, sig_query: np.ndarray, sr: int, max_lag_s: float, eps: float
) -> tuple[float, float]:
    """PHAT-weighted cross-correlation lag between two signals.

    Returns ``(lag_seconds, sharpness)`` where ``lag`` is the shift to ADD to the
    query's time so it aligns with the reference (positive ⇒ query currently lags;
    its event happens later in the query than in the reference), and ``sharpness``
    = peak / median(|cc|) is a confidence score (≈240–335 for clear speech windows,
    ≈12 for silence/uncorrelated — gate around 50). Sub-sample accurate via
    parabolic interpolation around the integer peak.
    """
    n = min(len(sig_ref), len(sig_query))
    if n < 8:
        return 0.0, 0.0
    a = sig_ref[:n] - sig_ref[:n].mean()
    b = sig_query[:n] - sig_query[:n].mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, nfft)  # noqa: N806 — conventional FFT notation
    B = np.fft.rfft(b, nfft)  # noqa: N806
    R = A * np.conj(B)  # noqa: N806
    R /= np.abs(R) + eps  # PHAT whitening  # noqa: N806
    cc = np.fft.irfft(R, nfft)
    # Reorder so index n-1 is zero lag, spanning lags [-(n-1) .. n-1].
    cc = np.concatenate((cc[-(n - 1) :], cc[:n]))
    abs_cc = np.abs(cc)

    # Restrict the peak search to ±max_lag.
    max_lag = int(min(n - 1, round(max_lag_s * sr)))
    center = n - 1
    lo = center - max_lag
    hi = center + max_lag
    window = abs_cc[lo : hi + 1]
    if window.size == 0:
        return 0.0, 0.0
    rel_peak = int(np.argmax(window))
    peak = lo + rel_peak

    median = float(np.median(abs_cc)) or eps
    sharpness = float(abs_cc[peak] / median)

    # Parabolic sub-sample interpolation around the integer peak.
    delta = 0.0
    if 0 < peak < len(abs_cc) - 1:
        y0, y1, y2 = abs_cc[peak - 1], abs_cc[peak], abs_cc[peak + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom

    lag_samples = (peak - center) + delta
    return lag_samples / sr, sharpness


def _fit_robust_line(
    points: list[tuple[float, float]], max_k_deviation: float
) -> tuple[float, float, int] | None:
    """Least-squares ``(offset, k)`` for ``cam = offset + k * rec`` over
    ``points`` (``(cam_time, rec_time)``), after dropping points more than
    twice the median residual away from a first pass.

    ``np.polyfit`` on raw points is ill-conditioned when the recorder times
    barely vary (every window matching near the same place) and happily
    returns nonsense slopes — a periodic test signal produced ``k ~= 4.25``.
    So the span is checked first, a plain offset-only fit is used when the
    evidence is too concentrated to support a slope, and the resulting ratio
    must stay physically plausible.

    Returns ``(offset, k, n_inliers)`` or ``None``.
    """
    if len(points) < 2:
        return None
    cam = np.array([p[0] for p in points], dtype=np.float64)
    rec = np.array([p[1] for p in points], dtype=np.float64)
    rec_span = float(rec.max() - rec.min())

    def _offset_only() -> tuple[float, float, int]:
        return float(np.median(cam - rec)), 1.0, len(points)

    # Fitting a slope through evidence spanning a few seconds turns GCC noise
    # into a wild clock ratio. Below this span, only the offset is estimated
    # and k is left at 1 — honest about what the data can support.
    if rec_span < _MIN_SLOPE_FIT_SPAN_S:
        return _offset_only()

    coeffs = np.polyfit(rec, cam, 1)
    k, offset = float(coeffs[0]), float(coeffs[1])
    residuals = np.abs((offset + k * rec) - cam)
    med = float(np.median(residuals))
    keep = residuals <= max(2.0 * med, 0.05)
    if keep.sum() >= 3 and keep.sum() < len(points):
        coeffs = np.polyfit(rec[keep], cam[keep], 1)
        k, offset = float(coeffs[0]), float(coeffs[1])
        residuals = np.abs((offset + k * rec) - cam)
        keep = residuals <= max(2.0 * float(np.median(residuals)), 0.05)

    if abs(k - 1.0) > max_k_deviation:
        # An implausible ratio means the points don't describe one clock; fall
        # back to the offset-only reading rather than exporting the nonsense.
        return _offset_only()
    return offset, k, int(keep.sum())


# Below this recorder-time span, grid evidence cannot support a slope estimate
# (GCC jitter dominates), so only the offset is fitted and k stays 1.0.
_MIN_SLOPE_FIT_SPAN_S = 60.0

# A camera window's best recorder match must beat its runner-up (measured at a
# different part of the recorder) by this ratio to count as unambiguous. On
# periodic or repetitive material several places correlate nearly as well, and
# picking the argmax of a near-tie is how a scan "confidently" lands on the
# wrong minute.
_AMBIGUITY_MARGIN = 1.25

# Recorder block length for the coarse scan. Each camera window is correlated
# against a whole block at once, so every alignment position inside the block
# is examined — there is no per-probe lag limit to leave holes between probes.
_SCAN_BLOCK_S = 64.0


def _plan_blocks(rec_len: int, block_n: int, win_n: int) -> list[int]:
    """Start sample of each recorder block, overlapping by one camera window.

    The overlap is what makes coverage total: a camera window that would
    straddle two blocks is wholly contained in at least one of them, so no
    alignment position is examined only partially.
    """
    if rec_len <= block_n:
        return [0]
    step = max(1, block_n - win_n)
    starts = list(range(0, rec_len - block_n + 1, step))
    if starts[-1] != rec_len - block_n:
        starts.append(rec_len - block_n)
    return starts


def _correlate_in_block(
    cam_win: np.ndarray,
    block_fft: np.ndarray,
    nfft: int,
    block_n: int,
    eps: float,
    separation_n: int,
) -> tuple[int, float, float, float]:
    """Best alignment of ``cam_win`` anywhere inside one recorder block.

    Returns ``(offset_samples, peak, rival, noise_floor)``: where in the block
    the camera window starts, how strong the PHAT correlation is there, how
    strong the best RIVAL peak at least ``separation_n`` samples away is, and
    the median |correlation| used as the confidence denominator.

    The rival is not a detail — it is the whole ambiguity test. A single block
    covering the entire recorder would otherwise report its argmax as a
    confident answer even when a second, unrelated place in the recording
    correlates just as well (a repeated phrase, a loop of music, a retake).

    This is a single FFT correlation of a SHORT signal against a LONG one —
    not the equal-length ``gcc_phat`` — which is what removes the old scan's
    blind spots. The old design probed the recorder on a coarse grid and
    searched only ±``max_lag_s`` around each probe, so with a 30 s step and a
    ±1 s search it never even looked at 28 of every 30 seconds.
    """
    win_n = len(cam_win)
    a = cam_win - cam_win.mean()
    fa = np.fft.rfft(a, nfft)
    r = block_fft * np.conj(fa)
    r /= np.abs(r) + eps
    cc = np.abs(np.fft.irfft(r, nfft))
    valid = cc[: max(1, block_n - win_n + 1)]
    peak_idx = int(np.argmax(valid))
    peak = float(valid[peak_idx])

    lo = max(0, peak_idx - separation_n)
    hi = min(len(valid), peak_idx + separation_n + 1)
    rival = 0.0
    if lo > 0:
        rival = max(rival, float(valid[:lo].max()))
    if hi < len(valid):
        rival = max(rival, float(valid[hi:].max()))

    floor = float(np.median(cc[:block_n])) or eps
    return peak_idx, peak, rival, floor


def _scan_camera_window(
    cam_win: np.ndarray,
    block_starts: list[int],
    block_ffts: list[np.ndarray],
    nfft: int,
    block_n: int,
    min_sharpness: float,
    eps: float,
    separation_n: int,
) -> tuple[float, float] | None:
    """Best recorder time (seconds) for one camera window, or None.

    ``None`` covers both "nothing correlated well enough" and "two unrelated
    places correlated about equally well" — the second is the important one:
    on periodic material the argmax of a near-tie is a coin flip dressed up as
    a confident answer, and it is how a scan reports the wrong minute.
    """
    win_n = len(cam_win)
    if win_n < 8:
        return None
    best_sharp = 0.0
    best_pos: int | None = None
    runner_up = 0.0
    for start, block_fft in zip(block_starts, block_ffts, strict=True):
        off, peak, rival, floor = _correlate_in_block(
            cam_win, block_fft, nfft, block_n, eps, separation_n
        )
        sharp = peak / floor
        pos = start + off
        # A rival inside this block always counts: overlapping blocks see the
        # same event twice, but a second peak within ONE block is genuinely a
        # different place in the recording.
        runner_up = max(runner_up, rival / floor)
        if sharp > best_sharp:
            if best_pos is not None and abs(pos - best_pos) > separation_n:
                runner_up = max(runner_up, best_sharp)
            best_sharp, best_pos = sharp, pos
        elif best_pos is not None and abs(pos - best_pos) > separation_n:
            runner_up = max(runner_up, sharp)

    if best_pos is None or best_sharp < min_sharpness:
        return None
    if runner_up > 0 and best_sharp < runner_up * _AMBIGUITY_MARGIN:
        logger.debug(
            "Acoustic scan: ambiguous match (%.0f vs runner-up %.0f) — declining",
            best_sharp,
            runner_up,
        )
        return None
    # The window's CENTRE is the time being located.
    return (best_pos + win_n / 2.0) / _REFINE_SR, best_sharp


def acoustic_coarse_align(
    cam_audio_wav: Path,
    rec_audio_path: Path,
    clip_duration: float,
    rec_duration: float,
    grid_s: float = 30.0,
    window_s: float = 8.0,
    max_lag_s: float = 1.0,
    min_sharpness: float = 50.0,
    gcc_eps: float = 1e-8,
    max_k_deviation: float = 0.05,
    min_points: int = 3,
    rec_range: tuple[float, float] | None = None,
) -> AcousticFit | None:  # noqa: PLR0913
    """Acoustic fallback offset/K estimate when there's no usable transcript
    match (too little speech, music, a foreign language Whisper garbles, or
    near-silence) — the alignment paths in ``matcher.py`` all fail without at
    least a couple of matched words. This works directly on the waveforms,
    exactly like Boundary Flex, but coarsely: cross-correlate a window of the
    camera's own audio against the recorder across the WHOLE recorder span
    (the clip could start anywhere in it), then fit an ``offset, K`` line
    through the confident, UNAMBIGUOUS points.

    Returns an :class:`AcousticFit` (``t_cam = offset + k * t_rec`` plus the
    evidence behind it) or ``None``.

    Two properties matter more than speed here, and the previous version had
    neither.

    **Coverage.** Every camera window is correlated against whole overlapping
    blocks of the recorder, so every possible alignment position is examined.
    The old scan probed the recorder on a ``grid_s`` grid and searched only
    ±``max_lag_s`` around each probe: at the defaults that inspected 2 s out of
    every 30, and a clip whose true offset fell in one of the 28-second holes
    was reported as "no acoustic match" — reproduced with shifts of 0 s and
    30 s found and 10 s not.

    **Unambiguity.** A window whose best match barely beats a rival elsewhere
    in the recording is discarded rather than resolved by argmax: on periodic
    material the argmax is a coin flip presented as a confident answer.

    ``max_lag_s`` is accepted for signature compatibility but no longer bounds
    the search — the block correlation has no per-probe lag limit, which is
    precisely what closed the blind spots. ``grid_s`` now only controls how
    many CAMERA windows are sampled (how much evidence is gathered), not how
    much of the recorder is looked at.
    """
    cam_track = load_mono16k_track(cam_audio_wav)
    # ``rec_range`` (from ``locate``) bounds the search: decoding hours of
    # recorder to place a five-minute clip that is known to sit in one place
    # costs far more than the correlation itself.
    rec_origin = max(0.0, rec_range[0]) if rec_range else 0.0
    rec_track = load_mono16k_track(
        rec_audio_path, start_s=rec_origin or None,
        duration_s=(rec_range[1] - rec_origin) if rec_range else None,
    )
    if rec_track.size < 8 or cam_track.size < 8:
        return None

    half = window_s / 2.0
    win_n = max(8, int(round(window_s * _REFINE_SR)))
    block_n = min(len(rec_track), max(int(round(_SCAN_BLOCK_S * _REFINE_SR)), 4 * win_n))
    nfft = 1 << int(np.ceil(np.log2(2 * block_n)))
    block_starts = _plan_blocks(len(rec_track), block_n, win_n)
    # The recorder's block FFTs depend only on the recorder, so they are
    # computed ONCE and reused for every camera window — the old nested loop
    # re-transformed recorder audio for every (camera window, probe) pair.
    block_ffts = [
        np.fft.rfft(rec_track[s0 : s0 + block_n] - rec_track[s0 : s0 + block_n].mean(), nfft)
        for s0 in block_starts
    ]
    separation_n = max(win_n, int(round(window_s * _REFINE_SR)))

    points: list[tuple[float, float]] = []  # (cam_time, rec_time) of confident matches
    t_cam = half
    while t_cam <= max(half, clip_duration - half):
        cam_win = _window_slice(cam_track, t_cam, window_s, _REFINE_SR)
        found = _scan_camera_window(
            cam_win,
            block_starts,
            block_ffts,
            nfft,
            block_n,
            min_sharpness,
            gcc_eps,
            separation_n,
        )
        if found is not None:
            rec_time, _sharp = found
            rec_time += rec_origin
            if 0.0 <= rec_time <= rec_duration + window_s:
                points.append((t_cam, rec_time))
        t_cam += grid_s

    if len(points) < max(2, min_points):
        logger.info(
            "Acoustic coarse align: only %d unambiguous point(s) (need %d), giving up",
            len(points),
            max(2, min_points),
        )
        return None

    fit = _fit_robust_line(points, max_k_deviation)
    if fit is None:
        return None
    offset, k, inliers = fit
    rec_times = [p[1] for p in points]
    span = max(rec_times) - min(rec_times)
    logger.info(
        "Acoustic coarse align: offset=%.3fs k=%.6f from %d point(s) spanning %.1fs",
        offset,
        k,
        len(points),
        span,
    )
    return AcousticFit(offset=offset, k=k, points=len(points), inliers=inliers, span_s=span)


# (rec_start, rec_in_duration, atempo_factor) — the piece tuple produced by clip_pieces.
Piece = tuple[float, float, float]


def _measure_boundary(
    cam_track: np.ndarray,
    rec_track: np.ndarray,
    cam_mid: float,
    rec_mid: float,
    win: float,
    max_lag_s: float,
    eps: float,
    min_sharpness: float,
    deadband_s: float,
    max_shift_s: float,
) -> float:
    """Measure one boundary's acoustic correction (seconds to add to rec_start),
    by slicing the two pre-decoded tracks in memory — no ffmpeg call, no scratch
    files. Returns 0.0 when the peak is not confident or the residual is within
    the deadband.
    """
    cam_sig = _window_slice(cam_track, cam_mid, win, _REFINE_SR)
    rec_sig = _window_slice(rec_track, rec_mid, win, _REFINE_SR)
    lag_s, sharp = gcc_phat(cam_sig, rec_sig, _REFINE_SR, max_lag_s, eps)
    # gcc_phat's lag is the shift to add to the query's (recorder's) time to
    # align it with the reference (camera); -lag_s is therefore the seconds
    # to add to the recorder read time so the cut lands under the picture.
    if sharp >= min_sharpness and abs(lag_s) > deadband_s:
        return max(-max_shift_s, min(max_shift_s, -lag_s))
    return 0.0


_FLEX_MIN_PIECE_S = 0.05
# Tempo bounds a re-timed predecessor must stay inside (atempo's own range).
_FLEX_MIN_FACTOR = 0.5
_FLEX_MAX_FACTOR = 2.0


def refine_piece_boundaries(
    pieces: list[Piece],
    lead: float,
    cam_audio_wav: Path,
    rec_audio_path: Path,
    clip_duration: float,
    rec_duration: float,
    config: AcousticSettings,
    tmp_dir: Path | None = None,
    workers: int = 1,
    rec_track: np.ndarray | None = None,
    rec_origin_s: float = 0.0,
) -> tuple[float, list[Piece]]:
    """Acoustically nudge each piece's onset so its speech lands under the
    picture, independent of Whisper's word timings ("Boundary Flex").
    Returns ``(lead, pieces)`` — the lead can change when piece 0's onset moves.

    Pieces are contiguous in OUTPUT (camera) time, beginning at ``lead``: piece i's
    local start is ``lead + sum(out_dur[<i])`` where ``out_dur = rec_dur / factor``.
    For each piece we cross-correlate a short camera window at that local time against
    the recorder window at the piece's current ``rec_start``; if the peak is sharp and
    the measured residual exceeds the deadband, the BOUNDARY between this piece and
    its predecessor moves by it (clamped).

    Moving the boundary — not just this piece's start — is what keeps the plan free
    of content gaps and overlaps: the previous piece's duration absorbs the shift
    (its factor is recomputed so its OUTPUT length grows/shrinks by exactly the
    output this piece loses/gains), the shifted piece keeps its own tempo factor, and
    every later piece's output position is untouched. The old behaviour slid the
    whole piece window without touching the neighbour, so a −80 ms nudge made the
    last 80 ms of piece N and the first 80 ms of piece N+1 the SAME recorder
    content played twice — the mid-word micro-repeat («подга-га-товил») users
    heard with Boundary Flex enabled. With sentence-wise plans the boundary sits
    in room tone and the neighbour is a pause piece, so the absorbed shift is
    inaudible by construction.

    Both tracks are decoded to mono 16 kHz numpy arrays exactly once (regardless
    of piece count), and every boundary window is a slice of those arrays — no
    per-boundary ffmpeg subprocess. The independent measurements are spread
    across a thread pool (``gcc_phat``'s FFT calls release the GIL, so threads
    parallelize this fine and avoid the fork-safety concerns of a process pool);
    the geometry is deterministic, so the result is identical regardless of
    worker count. ``tmp_dir`` is accepted for backward compatibility but is
    unused now that no scratch files are written.

    ``rec_track`` lets the caller supply an ALREADY-DECODED recorder array.
    Every render job used to decode the whole recorder again: at 16 kHz mono
    float64 that is ~461 MB per hour of recorder for the final array alone,
    before intermediate copies — repeated once per clip, so a shoot of many
    clips against one long recorder paid that I/O and allocation over and over.
    One decode per recorder, reused across its jobs, removes the repetition
    entirely.
    """
    if not pieces:
        return lead, pieces

    win = config.flex_window_s

    # First pass (cheap): the window geometry for every in-bounds boundary.
    jobs: list[tuple[int, float, float]] = []  # (idx, cam_mid, rec_mid)
    local = lead
    for idx, (rec_start, rec_dur, factor) in enumerate(pieces):
        out_dur = rec_dur / factor if factor else rec_dur
        probe = min(out_dur, win)
        cam_mid = local + probe / 2.0
        rec_mid = rec_start + min(rec_dur, win) / 2.0
        if (
            win / 2.0 <= cam_mid <= clip_duration - win / 2.0
            and win / 2.0 <= rec_mid <= rec_duration - win / 2.0
        ):
            jobs.append((idx, cam_mid, rec_mid))
        local += out_dur

    if not jobs:
        return lead, pieces

    # Decode both full tracks to mono 16k once; every boundary below just
    # slices these arrays. The recorder's decode is reused across jobs when the
    # caller hands one in (see `rec_track`).
    cam_track = load_mono16k_track(cam_audio_wav)
    if rec_track is None:
        # Only the stretch the pieces read (plus a window) is decoded, not the
        # whole multi-hour recorder.
        rec_origin_s = max(0.0, min(p[0] for p in pieces) - win)
        end = min(rec_duration, max(p[0] + p[1] for p in pieces) + win)
        rec_track = load_mono16k_track(rec_audio_path, start_s=rec_origin_s or None,
                                       duration_s=end - rec_origin_s)
    jobs = [(idx, cam_mid, rec_mid - rec_origin_s) for idx, cam_mid, rec_mid in jobs]

    args = (
        win,
        config.acoustic_max_lag_s,
        config.gcc_eps,
        config.flex_min_sharpness,
        config.flex_deadband_s,
        config.flex_max_shift_s,
    )
    corrections: dict[int, float] = {}
    if workers <= 1 or len(jobs) <= 1:
        for idx, cam_mid, rec_mid in jobs:
            corrections[idx] = _measure_boundary(cam_track, rec_track, cam_mid, rec_mid, *args)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_measure_boundary, cam_track, rec_track, cam_mid, rec_mid, *args): idx
                for idx, cam_mid, rec_mid in jobs
            }
            for fut, idx in futs.items():
                corrections[idx] = fut.result()

    # Apply the corrections by RE-TARGETING each piece's read boundary.
    #
    # The previous scheme moved the boundary and let the predecessor absorb the
    # change in BOTH its source and its output length: piece i's onset went to
    # s+δ while its output start went to L+δ/f. Those cancel exactly — an event
    # at recorder time R was output at L + δ/f + (R − s − δ)/f, which is
    # L + (R − s)/f, its original position. Boundary Flex logged "nudged N
    # onsets" and moved no audio at all; a −80 ms correction on the middle of
    # three pieces left a recorder event at 107 s on camera 7 s exactly where
    # it started.
    #
    # What actually moves speech is to change WHERE each piece reads while
    # leaving WHERE and HOW LONG it plays alone. Every piece keeps its output
    # window, so the plan's total length and every later piece's position are
    # invariant by construction (see pipeline.validate_pieces); each piece's
    # source window becomes [t_i, t_{i+1}) with t_i = s_i + δ_i, so the recorder
    # content stays exactly contiguous — no skipped audio and no double-played
    # sliver, which is what the boundary scheme existed to avoid. The tempo
    # factor absorbs the difference. An event then shifts by −δ_i/f at the
    # piece's start, easing linearly to the next boundary's own correction —
    # which is precisely what a piecewise-linear time warp should do.
    n = len(pieces)
    starts = [p[0] for p in pieces]
    src_end = pieces[-1][0] + pieces[-1][1]
    out_durs = [rec_dur / factor if factor else rec_dur for (_s, rec_dur, factor) in pieces]

    deltas = [corrections.get(i, 0.0) for i in range(n)]

    def _build(ds: list[float]) -> tuple[list[float], list[float]] | None:
        """(read boundaries, durations) for a delta set, or None if unusable."""
        targets = [
            min(max(starts[i] + ds[i], 0.0), rec_duration) if i else max(starts[0] + ds[0], 0.0)
            for i in range(n)
        ]
        end = min(max(src_end + ds[-1], 0.0), rec_duration)
        bounds = [*targets, end]
        for i in range(n):
            if bounds[i + 1] - bounds[i] < _FLEX_MIN_PIECE_S:
                return None
        return targets, [bounds[i + 1] - bounds[i] for i in range(n)]

    built = _build(deltas)
    # A single unusable correction must not throw away all the others: retry
    # with the largest offenders zeroed rather than abandoning the pass.
    while built is None and any(d != 0.0 for d in deltas):
        worst = max(range(n), key=lambda i: abs(deltas[i]))
        deltas[worst] = 0.0
        built = _build(deltas)
    if built is None:
        return lead, pieces
    targets, durations = built

    refined: list[tuple[float, float, float]] = []
    for i in range(n):
        out_dur = out_durs[i]
        factor = durations[i] / out_dur if out_dur > 1e-9 else pieces[i][2]
        if not (_FLEX_MIN_FACTOR <= factor <= _FLEX_MAX_FACTOR):
            # This correction would need a tempo outside atempo's range; keep
            # the original piece rather than clamping the factor, which would
            # silently break the source/output length relationship.
            refined.append(pieces[i])
            continue
        refined.append((targets[i], durations[i], factor))
    n_shifted = sum(1 for i in range(n) if refined[i][0] != pieces[i][0])

    logger.info("Boundary Flex: shifted %d/%d piece read positions", n_shifted, len(pieces))
    # The lead is untouched: piece 0's OUTPUT start is what defines it, and no
    # correction moves an output start any more.
    return lead, refined
