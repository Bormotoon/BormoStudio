"""The recorder's clock: find the samples it lost and put them back.

A phone recorder (sony_run: WaveEditor on a Huawei P30 Pro with USB-C
lavaliers) drops buffers when the phone is busy. Against the continuous DJI
recording it lost 14.5 s in 3.6 h, in steps of 15-870 ms, most of them between
4800 and 7500 s; between drops it runs at the cameras' rate (within 8 ppm) and
it never duplicates. Every camera that overlaps a drop sees the same step, and
everything placed on the recorder by one line (the timeline transcript, cuts,
subtitles, speakers) drifts by the sum of the drops before it.

So the recorder is repaired once, for everyone:

1. **Tracking.** Each placed clip is measured against the recorder window by
   window (3 s, every second); the read position follows the last measured
   offset, so drops of any total size stay in view. A jump is accepted only
   when the next window confirms it (a single false peak is ignored).
2. **The loss map.** Jumps of every clip become losses at recorder times;
   losses several cameras see are one; a loss a camera heard across without a
   step is that camera's glitch, not the recorder's. Losses between two
   continuous files of one camera (chapters) come from where the next file
   starts against where the previous one ended.
3. **Location.** Short windows around a jump put it within ~50 ms.
4. **Repair.** The lost time is inserted at each loss, filled from the camera
   that heard it (level-matched, short fades) or with silence. The repaired
   recorder runs with real time; transcripts are moved by the same map and
   every camera is fitted to the repaired file by one line.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from studio.core.transcript import Segment, Transcript, Word
from studio.stages.sync_acoustic import gcc_phat
from studio.stages.sync_models import AlignmentMap, Anchor
from studio.stages.sync_verify import DEFAULT_GCC_EPS

SR = 16000
TRACK_WINDOW_S = 3.0
TRACK_STEP_S = 1.0
TRACK_MAX_LAG_S = 1.0
# After this long without a confident window the tracker searches wider.
REACQUIRE_S = 8.0
REACQUIRE_LAG_S = 3.0
TRACK_MIN_SHARPNESS = 50.0
# Smaller changes of the offset are measurement noise, larger ones are a jump.
JUMP_S = 0.015
BIG_JUMP_S = 0.3
LOST_MIN_MS = 15.0
# Opposite jumps of one clip this close cancel out.
CANCEL_S = 60.0
# Jumps of different clips this close are one loss.
MERGE_S = 2.0
# A clip heard a loss when it has windows this close on both sides of it.
CONTRADICT_S = 10.0
FINE_WINDOW_S = 0.25
FINE_STEP_S = 0.025
# Short windows match within this much of an offset, and above this score.
FINE_SLACK_S = 0.01
FINE_MATCH = 0.3
# Two files of one camera are continuous (chapters) within this.
CONTINUOUS_S = 2.0
FADE_S = 0.005
CHUNK = 1 << 16

Point = tuple[float, float, float]  # (clip time, recorder file time, sharpness)


@dataclass(frozen=True, slots=True)
class Line:
    """Recorder file time of clip time ``t`` for ``cam = offset + k * rec``."""

    offset: float
    k: float = 1.0

    def __call__(self, t: float) -> float:
        return (t - self.offset) / (self.k or 1.0)


@dataclass(slots=True)
class Loss:
    """``lost_ms`` of real time missing from the recorder file at ``rec_s``."""

    rec_s: float
    lost_ms: float
    located_by: str
    seen_by: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)
    # Where the lost sound is in a camera: (camera id, clip time).
    fill: tuple[str, float] | None = None
    # Recorder times between which a loss not located finely happened.
    span: tuple[float, float] | None = None

    @property
    def applied(self) -> bool:
        return self.lost_ms > 0 and len(self.seen_by) > len(self.contradicted_by)

    def to_dict(self) -> dict[str, Any]:
        return {"recorder_s": round(self.rec_s, 3), "lost_ms": round(self.lost_ms, 1),
                "located_by": self.located_by, "seen_by": self.seen_by,
                "contradicted_by": self.contradicted_by, "applied": self.applied,
                "fill": list(self.fill) if self.fill else None}


@dataclass(slots=True)
class RecorderClock:
    recorder_id: str
    losses: list[Loss]

    @property
    def applied(self) -> list[Loss]:
        return sorted((loss for loss in self.losses if loss.applied), key=lambda x: x.rec_s)

    @property
    def total_ms(self) -> float:
        return sum(loss.lost_ms for loss in self.applied)

    def to_repaired(self, rec_s: float) -> float:
        """Repaired-file time of recorder file time ``rec_s``."""
        return rec_s + sum(loss.lost_ms for loss in self.applied if loss.rec_s <= rec_s) / 1000

    def to_file(self, repaired_s: float) -> float:
        """Recorder file time of repaired time (the loss position inside an insert)."""
        shift = 0.0
        for loss in self.applied:
            start = loss.rec_s + shift
            if repaired_s < start:
                break
            if repaired_s < start + loss.lost_ms / 1000:
                return loss.rec_s
            shift += loss.lost_ms / 1000
        return repaired_s - shift


# --------------------------------------------------------------------------- tracking


def track_clip(
    camera: np.ndarray, recorder: np.ndarray, recorder_origin_s: float,
    line: Line, duration_s: float,
) -> list[Point]:
    """Confident ``(clip s, recorder s, sharpness)`` along the clip, following jumps.

    A jump is taken only when the next window (two windows for a jump over
    ``BIG_JUMP_S``) lands on the same new offset; a lone peak elsewhere is a
    false one. After ``REACQUIRE_S`` without a confident window the search
    widens to ``REACQUIRE_LAG_S``, under the same confirmation.
    """
    half = int(TRACK_WINDOW_S * SR / 2)
    shift = 0.0
    pending: list[Point] = []
    pending_shift = 0.0
    points: list[Point] = []
    quiet_since = 0.0
    t = TRACK_WINDOW_S / 2
    while t <= duration_s - TRACK_WINDOW_S / 2 + 1e-9:
        wide = t - quiet_since >= REACQUIRE_S
        max_lag = REACQUIRE_LAG_S if wide else TRACK_MAX_LAG_S
        reach = int(max_lag * SR) if wide else 0
        c = int(t * SR)
        r = int((line(t) + shift - recorder_origin_s) * SR)
        if c - half >= 0 and c + half <= len(camera) and r - half - reach >= 0 and (
                r + half + reach <= len(recorder)):
            query = camera[c - half:c + half]
            reference = recorder[r - half - reach:r + half + reach]
            if reach:
                query = np.pad(query, (reach, reach))
            lag, sharpness = gcc_phat(query, reference, SR, max_lag, DEFAULT_GCC_EPS)
            if sharpness >= TRACK_MIN_SHARPNESS:
                measured = shift - float(lag)
                point = (round(t, 4), line(t) + measured, float(sharpness))
                if abs(measured - shift) <= JUMP_S:
                    shift, pending = measured, []
                    points.append(point)
                    quiet_since = t
                elif pending and abs(measured - pending_shift) <= JUMP_S:
                    pending.append(point)
                    needed = 3 if abs(pending_shift - shift) > BIG_JUMP_S else 2
                    if len(pending) >= needed:
                        shift = measured
                        points.extend(pending)
                        pending = []
                        quiet_since = t
                else:
                    pending, pending_shift = [point], measured
        t += TRACK_STEP_S
    return points


def _jumps(points: Sequence[Point], line: Line) -> list[tuple[int, float]]:
    """``(index after the jump, change of the read offset in seconds)``."""
    result = []
    for index in range(1, len(points)):
        before = points[index - 1][1] - line(points[index - 1][0])
        after = points[index][1] - line(points[index][0])
        if abs(after - before) * 1000 >= LOST_MIN_MS:
            result.append((index, after - before))
    return result


def _locate(
    camera: np.ndarray, recorder: np.ndarray, recorder_origin_s: float,
    line: Line, t0: float, t1: float, before: float, after: float,
) -> float | None:
    """Clip time where the recorder lost ``after - before`` seconds, or None.

    The camera's sound up to the loss matches the recorder read with the old
    offset and stops matching exactly there; the sound after the lost stretch
    matches with the new offset, starting ``lost`` seconds later (what was
    lost exists only in the camera). Short windows find both edges.
    """
    lost = before - after
    half = int(FINE_WINDOW_S * SR / 2)
    slack = int(FINE_SLACK_S * SR)
    rows: list[tuple[float, float, float]] = []
    t = t0 - TRACK_STEP_S
    while t <= t1 + TRACK_STEP_S + lost + 1e-9:
        c = int(t * SR)
        if c - half < 0 or c + half > len(camera):
            t += FINE_STEP_S
            continue
        query = camera[c - half:c + half]
        scores = []
        for offset in (before, after):
            r = int((line(t) + offset - recorder_origin_s) * SR)
            if r - half - slack < 0 or r + half + slack > len(recorder):
                scores.append(float("nan"))
                continue
            scores.append(_match(query, recorder[r - half - slack:r + half + slack], slack))
        rows.append((t, scores[0], scores[1]))
        t += FINE_STEP_S
    # Windows that match one offset only say which side of the loss they are
    # on; silence (neither) and ambiguous ones (both) say nothing.
    sides = [(t, old >= FINE_MATCH) for t, old, new in rows
             if (old >= FINE_MATCH) != (new >= FINE_MATCH)]
    if len(sides) < 4:
        return None
    flags = [is_old for _t, is_old in sides]
    scores = [flags[:cut].count(True) + flags[cut:].count(False) for cut in range(1, len(sides))]
    cut = 1 + int(np.argmax(scores))
    if scores[cut - 1] < 0.8 * len(sides) or not flags[cut - 1] or flags[cut]:
        return None
    # A window still matches while most of it is on its side, so the last
    # old window and the first new one sit at the loss within a window's width.
    last_old = sides[cut - 1][0]
    first_new = sides[cut][0] - lost
    if abs(first_new - last_old) > max(FINE_WINDOW_S, lost):
        return None
    return (last_old + first_new) / 2


def _match(query: np.ndarray, reference: np.ndarray, slack: int) -> float:
    """Best normalized correlation of ``query`` inside ``reference``, every sample of ``slack``."""
    q = (query - query.mean()).astype(np.float64)
    norm_q = float(np.linalg.norm(q))
    if norm_q == 0 or reference.size < q.size:
        return 0.0
    ref = reference.astype(np.float64)
    dots = np.correlate(ref, q, mode="valid")  # one per shift
    # Each shifted reference segment, centred: sum(x²) - n·mean².
    sums = np.concatenate([[0.0], np.cumsum(ref)])
    squares = np.concatenate([[0.0], np.cumsum(ref * ref)])
    n = q.size
    total = sums[n:] - sums[:-n]
    energy = squares[n:] - squares[:-n] - total * total / n
    norms = np.sqrt(np.maximum(energy, 0.0)) * norm_q
    scores = np.where(norms > 0, dots / np.where(norms > 0, norms, 1.0), 0.0)
    return float(scores.max()) if scores.size else 0.0


# --------------------------------------------------------------------------- the loss map


@dataclass(slots=True)
class ClipTrack:
    camera_id: str
    points: list[Point]
    line: Line
    duration_s: float
    # Clip time where each jump was located, filled by ``clip_losses``.
    losses: list[Loss] = field(default_factory=list)


def clip_losses(
    track: ClipTrack, camera: np.ndarray, recorder: np.ndarray, recorder_origin_s: float,
) -> list[Loss]:
    """The losses one clip saw, located as well as its sound allows."""
    result = []
    jumps = _jumps(track.points, track.line)
    # The recorder only loses samples: a jump undone soon after by an equal
    # opposite one is a measurement detour, not two events.
    cancelled: set[int] = set()
    for (i, a), (j, b) in zip(jumps, jumps[1:], strict=False):
        if (track.points[j][0] - track.points[i][0] <= CANCEL_S
                and abs(a + b) <= max(LOST_MIN_MS / 1000, 0.2 * abs(a))):
            cancelled.update((i, j))
    for index, change in jumps:
        if index in cancelled:
            continue
        (t0, r0, _s0), (t1, _r1, _s1) = track.points[index - 1], track.points[index]
        before = r0 - track.line(t0)
        located = _locate(camera, recorder, recorder_origin_s, track.line, t0, t1, before,
                          before + change)
        at = located if located is not None else (t0 + t1) / 2
        result.append(Loss(track.line(at) + before, -change * 1000.0,
                           "fine" if located is not None else "between windows",
                           [track.camera_id], fill=(track.camera_id, at),
                           span=(track.line(t0) + before, track.line(t1) + before)))
    track.losses = result
    return result


def build_clock(
    recorder_id: str, tracks: Sequence[ClipTrack], continuity: Sequence[tuple[str, str]],
    recorder_words: Sequence[tuple[float, float]] = (),
) -> RecorderClock:
    """Merge the losses of every clip; add those between continuous files."""
    losses: list[Loss] = []
    for loss in sorted((item for track in tracks for item in track.losses),
                       key=lambda item: item.rec_s):
        same = next((item for item in losses
                     if abs(item.rec_s - loss.rec_s) <= MERGE_S
                     and abs(item.lost_ms - loss.lost_ms) <= max(20.0, 0.25 * abs(item.lost_ms))),
                    None)
        if same is None:
            losses.append(loss)
            continue
        same.seen_by = sorted({*same.seen_by, *loss.seen_by})
        if same.located_by != "fine" and loss.located_by == "fine":
            same.rec_s, same.located_by, same.fill = loss.rec_s, "fine", loss.fill
    for loss in losses:
        if loss.located_by == "between windows" and loss.span is not None:
            # No sound to say where exactly: the longest pause, where an insert is harmless.
            position = _quiet_point(loss.span[0], loss.span[1], recorder_words)
            if loss.fill is not None:
                loss.fill = (loss.fill[0], loss.fill[1] + position - loss.rec_s)
            loss.rec_s = position
    by_id = {track.camera_id: track for track in tracks}
    for first_id, second_id in continuity:
        first, second = by_id.get(first_id), by_id.get(second_id)
        if first is None or second is None or not (
                _stable(first, first.points[-3:]) and _stable(second, second.points[:3])):
            continue
        last_t, last_r, _ = first.points[-1]
        end = last_r + (first.duration_s - last_t)  # where the first file ends
        start_t, start_r, _ = second.points[0]
        begin = start_r - start_t  # where the second file starts
        lost_ms = (end - begin) * 1000.0
        if lost_ms < LOST_MIN_MS:
            continue
        position = _quiet_point(last_r, start_r, recorder_words)
        losses.append(Loss(position, lost_ms, "between files", [first_id, second_id],
                           fill=(first_id, last_t + (position - last_r))
                           if position <= end else None))
    for loss in losses:
        for track in tracks:
            if track.camera_id in loss.seen_by or loss.located_by == "between files":
                continue
            times = np.array([point[1] for point in track.points])
            if not times.size:
                continue
            before = np.sum((times < loss.rec_s) & (times >= loss.rec_s - CONTRADICT_S))
            after = np.sum((times > loss.rec_s) & (times <= loss.rec_s + CONTRADICT_S))
            if before >= 2 and after >= 2 and not any(
                    abs(own.rec_s - loss.rec_s) <= MERGE_S for own in track.losses):
                loss.contradicted_by.append(track.camera_id)
    return RecorderClock(recorder_id, sorted(losses, key=lambda item: item.rec_s))


def _stable(track: ClipTrack, points: Sequence[Point]) -> bool:
    """Three windows that agree on the read offset (a file end worth trusting)."""
    if len(points) < 3:
        return False
    offsets = [point[1] - track.line(point[0]) for point in points]
    return max(offsets) - min(offsets) <= JUMP_S


def _quiet_point(low: float, high: float, words: Sequence[tuple[float, float]]) -> float:
    """The middle of the longest pause between recorder words in ``[low, high]``."""
    inside = sorted((start, end) for start, end in words if end > low and start < high)
    edges = [low, *[x for span in inside for x in span], high]
    gaps = [(edges[i + 1] - edges[i], (edges[i] + edges[i + 1]) / 2)
            for i in range(0, len(edges) - 1, 2)]
    best = max(gaps, default=(0.0, (low + high) / 2))
    return best[1]


# --------------------------------------------------------------------------- repair


def refit(alignment: AlignmentMap, clock: RecorderClock, points: Sequence[Point],
          clip_duration_s: float) -> AlignmentMap:
    """``alignment`` against the repaired recorder: one line through the tracked points."""
    if len(points) >= 8:
        cam = np.array([point[0] for point in points])
        rec = np.array([clock.to_repaired(point[1]) for point in points])
        keep = np.ones(cam.size, dtype=bool)
        for _round in range(3):
            k, offset = (float(value) for value in np.polyfit(rec[keep], cam[keep], 1))
            residual = np.abs(offset + k * rec - cam)
            limit = max(0.005, 3.0 * float(np.median(residual[keep])))
            keep = residual <= limit
        span = float(rec[keep].max() - rec[keep].min()) if keep.any() else 0.0
        acoustic = span >= 0.5 * clip_duration_s
        return replace(
            alignment, anchors=[], offset=float(offset), k=float(k),
            residual_ms=float(np.median(residual[keep])) * 1000.0,
            provenance="acoustic" if acoustic else alignment.provenance,
            inliers=max(alignment.inliers, int(keep.sum())), evidence_span_s=span,
            notes=[*alignment.notes, f"fitted to the repaired recorder by {int(keep.sum())} "
                   "tracked windows"])
    words = [anchor for anchor in alignment.anchors
             if not str(anchor.token).startswith(("acoustic:", "jump:"))]
    if len(words) >= 2:
        from studio.stages.sync_matcher import ransac_linear_fit

        moved = [Anchor(anchor.cam_time, clock.to_repaired(anchor.rec_time), anchor.token,
                        anchor.confidence) for anchor in words]
        line_offset, line_k, _inliers = ransac_linear_fit(moved)
        return replace(alignment, anchors=moved, offset=line_offset, k=line_k,
                       notes=[*alignment.notes, "text anchors moved to the repaired recorder"])
    rate = alignment.k or 1.0
    middle = clip_duration_s / 2
    at = clock.to_repaired((middle - alignment.offset) / rate)
    return replace(alignment, offset=middle - rate * at,
                   notes=[*alignment.notes, "moved to the repaired recorder at its middle"])


def repair_transcript(transcript: Transcript, clock: RecorderClock, audio: Path) -> Transcript:
    """Word times of the recorder moved to the repaired file."""
    def move(value: float) -> float:
        return clock.to_repaired(value)

    def words(items: Sequence[Word]) -> tuple[Word, ...]:
        return tuple(replace(word, start=move(word.start), end=move(word.end))
                     for word in items)

    segments: list[Segment] = [
        replace(segment, start=move(segment.start), end=move(segment.end),
                words=words(segment.words),
                raw_words=words(segment.raw_words) if segment.raw_words is not None else None)
        for segment in transcript.segments
    ]
    metadata = dict(transcript.metadata)
    if metadata.get("coverage"):
        metadata["coverage"] = [[move(float(a)), move(float(b))] for a, b in metadata["coverage"]]
    metadata["repaired_recorder_ms"] = round(clock.total_ms, 1)
    return replace(transcript, source_audio=audio, segments=segments,
                   duration=transcript.duration + clock.total_ms / 1000.0, metadata=metadata)


def write_repaired(
    source: Path, output: Path, clock: RecorderClock, fills: dict[int, np.ndarray], *,
    sample_rate: int, channels: int, codec: str, stream: int | None = None,
    audio_filter: str | None = None,
) -> Path:
    """``source`` with every applied loss inserted (``fills`` by loss index, else silence)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    decoder = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
         "-map", f"0:{stream}" if stream is not None else "0:a:0", "-vn",
         *(["-af", audio_filter] if audio_filter else []),
         "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sample_rate),
         "-ac", str(channels), "-i", "-", "-c:a", codec, "-rf64", "auto", str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert decoder.stdout is not None and encoder.stdin is not None
    inserts = [(round(loss.rec_s * sample_rate), index, loss)
               for index, loss in enumerate(clock.applied)]
    position = 0
    tail = np.zeros((0, channels), np.float32)
    frame = 4 * channels
    try:
        while True:
            raw = decoder.stdout.read(CHUNK * frame)
            data = np.frombuffer(raw[: len(raw) - len(raw) % frame], dtype="<f4").reshape(
                -1, channels)
            end = position + len(data)
            cursor = 0
            while inserts and (inserts[0][0] < end or not raw and inserts[0][0] <= end):
                at, index, loss = inserts.pop(0)
                cut = max(0, at - position)
                encoder.stdin.write(data[cursor:cut].tobytes())
                tail = np.concatenate([tail, data[cursor:cut]])[-int(0.3 * sample_rate):]
                cursor = cut
                encoder.stdin.write(_fill(fills.get(index), loss, tail, sample_rate,
                                          channels).tobytes())
            encoder.stdin.write(data[cursor:].tobytes())
            tail = np.concatenate([tail, data[cursor:]])[-int(0.3 * sample_rate):]
            position = end
            if not raw:
                break
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
        encoder.wait()
    if decoder.returncode or encoder.returncode:
        raise RuntimeError(f"recorder repair failed for {source.name}: {error.strip()}")
    return output


def _fill(sound: np.ndarray | None, loss: Loss, before: np.ndarray, sample_rate: int,
          channels: int) -> np.ndarray:
    count = round(loss.lost_ms / 1000.0 * sample_rate)
    if sound is None or not sound.size:
        return np.zeros((count, channels), np.float32)
    mono = np.zeros(count, np.float32)
    mono[:min(count, sound.size)] = sound[:count]
    level = float(np.sqrt(np.mean(before ** 2))) if before.size else 0.0
    own = float(np.sqrt(np.mean(mono ** 2)))
    gain = float(np.clip(level / own, 0.05, 20.0)) if own > 0 and level > 0 else 0.0
    mono *= gain
    fade = min(count // 2, int(FADE_S * sample_rate))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        mono[:fade] *= ramp
        mono[-fade:] *= ramp[::-1]
    return np.repeat(mono[:, None], channels, axis=1)


def decode_fill(camera: Path, start_s: float, duration_s: float, sample_rate: int,
                stream: int | None = None, audio_filter: str | None = None) -> np.ndarray:
    """Mono samples of the camera's sound that fill one loss."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, start_s):.4f}",
         "-t", f"{duration_s:.4f}", "-i", str(camera),
         "-map", f"0:{stream}" if stream is not None else "0:a:0", "-vn",
         *(["-af", audio_filter] if audio_filter else []), "-ac", "1",
         "-ar", str(sample_rate), "-f", "f32le", "-"],
        capture_output=True)
    if result.returncode:
        return np.zeros(0, np.float32)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


# --------------------------------------------------------------------------- the sync step


@dataclass(slots=True)
class Repair:
    clock: RecorderClock
    audio: Path  # the repaired recorder in its own format, for every later stage
    transcript: Path  # its transcript, moved to the repaired times
    tracks: dict[str, int]  # tracked windows per clip

    def report(self) -> dict[str, Any]:
        return {"lost_ms": round(self.clock.total_ms, 1), "audio": f"recorder/{self.audio.name}",
                "losses": [loss.to_dict() for loss in self.clock.losses],
                "tracked_windows": self.tracks}


def _track_job(camera_path: Path, duration_s: float, line: Line, recorder_path: Path,
               camera_id: str) -> ClipTrack:
    from studio.stages.sync_grid import decode_range

    origin = max(0.0, line(0.0) - 15.0)
    try:
        camera = decode_range(camera_path, 0.0, duration_s)
        recorder = decode_range(recorder_path, origin, duration_s / (line.k or 1.0) + 30.0)
    except (OSError, RuntimeError, ValueError):
        # A clip without readable sound takes no part in the clock (it is
        # still moved to the repaired recorder by its own map).
        return ClipTrack(camera_id, [], line, duration_s)
    track = ClipTrack(camera_id, track_clip(camera, recorder, origin, line, duration_s), line,
                      duration_s)
    clip_losses(track, camera, recorder, origin)
    return track


def repair_recorders(
    *, source_dir: Path, cameras: Sequence[Any], recorders: list[Any],
    originals: dict[str, Any], transcripts: dict[str, Transcript],
    matching: dict[str, Transcript], fits: dict[tuple[str, str], AlignmentMap],
    sync: dict[str, Any], workers: int, output_dir: Path, scratch_dir: Path,
    published_dir: Path | None = None, progress: Any = None,
) -> dict[str, Repair]:
    """Repair every recorder that lost samples; ``recorders``, ``transcripts``,
    ``matching`` and ``fits`` are updated in place to the repaired files.

    ``published_dir`` is where ``output_dir`` lands once the stage is published:
    transcripts name their audio there, not in the worker's private copy.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from studio.core.channels import effective_audio
    from studio.stages.locate import _natural

    repairs: dict[str, Repair] = {}
    for index, recorder in enumerate(list(recorders)):
        placed = [camera for camera in cameras if (camera.id, recorder.id) in fits]
        if not placed:
            continue
        if progress is not None:
            progress(f"tracking the clock of {Path(originals[recorder.id].path).name} "
                     f"against {len(placed)} clip(s)")
        jobs = [(source_dir / camera.path,
                 float(transcripts[camera.id].duration),
                 Line(fits[camera.id, recorder.id].offset, fits[camera.id, recorder.id].k),
                 source_dir / recorder.path, camera.id) for camera in placed]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(min(workers, len(jobs)),
                                     mp_context=get_context("spawn")) as pool:
                tracks = list(pool.map(_track_job, *zip(*jobs, strict=True)))
        else:
            tracks = [_track_job(*job) for job in jobs]
        continuity = _continuity(placed, originals, _natural)
        words = (matching.get(recorder.id) or transcripts[recorder.id]).words
        clock = build_clock(recorder.id, tracks, continuity,
                            [(word.start, word.end) for word in words])
        if not clock.applied:
            continue
        original = originals[recorder.id]
        info = original.manual.get("media_info", {})
        sample_rate = int(info.get("audio_sample_rate") or 48000)
        bits = int(info.get("audio_bits_per_sample") or 24)
        fills: dict[int, np.ndarray] = {}
        if sync.get("recorder_fill", "camera") == "camera":
            for number, loss in enumerate(clock.applied):
                if loss.fill is None or loss.fill[0] not in originals:
                    continue
                camera = originals[loss.fill[0]]
                fills[number] = decode_fill(
                    source_dir / camera.path, loss.fill[1], loss.lost_ms / 1000.0, sample_rate,
                    camera.manual.get("media_info", {}).get("audio_stream_index"),
                    effective_audio(camera).pan)
        if progress is not None:
            progress(f"writing the repaired {Path(original.path).name}: "
                     f"{len(clock.applied)} loss(es), {clock.total_ms / 1000:.1f} s")
        audio = write_repaired(
            source_dir / original.path, output_dir / f"{recorder.id}.wav", clock, fills,
            sample_rate=sample_rate, channels=int(info.get("audio_channels") or 1),
            codec="pcm_s16le" if bits <= 16 else "pcm_s24le" if bits <= 24 else "pcm_s32le",
            stream=info.get("audio_stream_index"))
        working = audio
        if Path(recorder.path) != Path(original.path):
            # Sync works on the selected stream (decoded, channels picked).
            working_info = recorder.manual.get("media_info", {})
            working = write_repaired(
                source_dir / recorder.path, scratch_dir / f"{recorder.id}-repaired.wav", clock,
                fills, sample_rate=int(working_info.get("audio_sample_rate") or sample_rate),
                channels=int(working_info.get("audio_channels") or 1), codec="pcm_f32le")
        public = (published_dir or output_dir) / audio.name
        repaired = repair_transcript(transcripts[recorder.id], clock, public)
        transcript_path = output_dir / f"{recorder.id}.transcript.json"
        repaired.save(transcript_path)
        transcripts[recorder.id] = repaired
        if recorder.id in matching:
            matching[recorder.id] = repair_transcript(matching[recorder.id], clock, public)
        by_id = {track.camera_id: track for track in tracks}
        for camera in placed:
            fits[camera.id, recorder.id] = refit(
                fits[camera.id, recorder.id], clock, by_id[camera.id].points,
                float(transcripts[camera.id].duration))
        media = {**recorder.manual.get("media_info", {}),
                 "duration_s": float(info.get("duration_s") or repaired.duration)
                 + clock.total_ms / 1000.0}
        recorders[index] = replace(recorder, path=working,
                                   manual={**recorder.manual, "media_info": media})
        repairs[recorder.id] = Repair(clock, audio, transcript_path,
                                      {track.camera_id: len(track.points) for track in tracks})
    return repairs


def _continuity(cameras: Sequence[Any], originals: dict[str, Any],
                natural: Any) -> list[tuple[str, str]]:
    """Consecutive files of one camera whose recording times follow each other."""
    groups: dict[str, list[Any]] = {}
    for camera in cameras:
        source = originals.get(camera.id, camera)
        groups.setdefault(source.group_id or source.device or source.id, []).append(source)
    pairs = []
    for members in groups.values():
        ordered = sorted(members, key=lambda item: natural(str(item.path)))
        for first, second in zip(ordered, ordered[1:], strict=False):
            a = first.manual.get("media_info", {})
            b = second.manual.get("media_info", {})
            start_a = (a.get("recorded_at") or {}).get("epoch")
            start_b = (b.get("recorded_at") or {}).get("epoch")
            if start_a is None or start_b is None:
                continue
            if abs(float(start_b) - float(start_a) - float(a.get("duration_s") or 0.0)) <= (
                    CONTINUOUS_S):
                pairs.append((first.id, second.id))
    return pairs
