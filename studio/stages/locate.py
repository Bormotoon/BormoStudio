"""Find where each camera sits in each recorder before anything is transcribed.

A five-minute clip must not cost a Whisper pass over a 4.5-hour recorder. This
discovery stage locates cameras acoustically: three 8 s windows of the group's
reference camera (the longest one with sound) are correlated against the whole
recorder, streamed through ffmpeg in blocks so memory stays at one block. The
match gives the offset between the group's clock and the recorder's
(``delta``); every other camera of the group is then searched only around
``recorded_at + delta`` (±``sync.locate_search_s``, then ±15 min, then the
whole file). ``prepare`` turns the positions into the recorder ranges that
need a transcript.

``recorded_at`` is often the time the files were copied, not recorded: on
sony_run the deltas of one camera's clips spread over 10 000 s and a narrow
search around such a prediction accepted weak false peaks. So a group delta
is kept only when clips found independently agree on it, a narrow search must
pass a stricter vote than the full pass, and every position must respect the
order of the camera's files; a clip that breaks it is searched again between
its neighbours.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from studio.core.channels import effective_audio
from studio.core.digest import file_digest
from studio.core.parallel import cpu_workers
from studio.core.project import Asset, AssetRole, Project, stable_fingerprint
from studio.core.recorded_at import epoch
from studio.stages.base import (
    Decision,
    GpuUse,
    Requirement,
    StageContext,
    StageOutput,
    select_settings,
)

SR = 16000
WINDOW_S = 8.0
WINDOW_SPACING_S = 120.0
MIN_WINDOWS = 3
MAX_WINDOWS = 6
MIN_CLIP_S = 2.0
BLOCK_S = 64.0
MIN_SHARPNESS = 50.0
CANDIDATE_SHARPNESS = 25.0
PEAKS_PER_BLOCK = 3
KEEP_CANDIDATES = 24
AGREE_S = 0.1
WIDE_SEARCH_S = 900.0
# A narrow search (around a prediction or between neighbouring files) offers
# few competing peaks, so a weak false one wins easily: it needs half of the
# clip's windows and a sharper peak than the full pass. On sony_run the false
# narrow matches peaked at 51-61, the true ones at 64-80 and above.
STRICT_MIN_SHARPNESS = 62.0
# Clips of one group agree on the clock delta within this (two thirds of them
# at least); only then is it a hint. On sony_run the recording-time deltas of
# one camera spread over 14-16 s in 3.5 h (the recorder loses samples), and
# two Canon clips were 240-270 s off; the copy times spread over 10 000 s.
HINT_AGREE_S = 30.0
HINT_MAJORITY = 2 / 3
# Consecutive files of one camera may overlap by this much in recorder time,
# plus this share of the earlier file when its end is only start + duration:
# the sony_run recorder lost 4.2 s inside one 965 s DJI file.
ORDER_TOLERANCE_S = 2.0
ORDER_TOLERANCE_RATE = 0.01
DITHER = 3e-5  # about -90 dBFS
BATCH_WINDOWS = 40


@dataclass(frozen=True, slots=True)
class Located:
    position_s: float  # recorder time of the camera's start
    sharpness: float
    method: str
    # Recorder time of the camera's end when a fitted map knows it: a recorder
    # that loses samples holds a clip in less time than the clip lasts.
    end_s: float | None = None


class LocateStage:
    id = "locate"
    title = "Locate cameras"
    after: tuple[str, ...] = ("scan",)
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = (
        "sync.mode", "sync.placement", "sync.locate_search_s", "transcribe.recorder_scope")
    # compute.cpu_workers changes speed, not the result: it stays out of the fingerprint.
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return [Requirement("binary", "ffmpeg", "locate cameras in recorders")]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        return Decision.run()

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        sources = [
            (asset.id, asset.role.value, str(asset.path), asset.group_id,
             asset.manual.get("media_info", {}).get("recorded_at"),
             file_digest(project.source_dir / asset.path, project.work_dir)
             if (project.source_dir / asset.path).is_file() else None)
            for asset in project.assets if asset.role in {AssetRole.CAMERA, AssetRole.RECORDER}
        ]
        return stable_fingerprint("locate-v3", sources,
                                  select_settings(settings, self.settings_keys))

    def run(self, context: StageContext) -> StageOutput:
        output = context.work_dir / "stages" / "locate" / "positions.json"
        previous = _load(context.project.work_dir / "stages" / "locate" / "positions.json")
        data = locate_project(context.project, context.settings, previous, context.progress)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        return StageOutput((output,))


def locate_project(
    project: Project, settings: dict[str, Any], previous: dict[str, Any] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    sync = settings.get("sync", {}) if isinstance(settings.get("sync"), dict) else {}
    transcribe = settings.get("transcribe", {})
    scope = transcribe.get("recorder_scope", "auto") if isinstance(transcribe, dict) else "auto"
    cameras = [asset for asset in project.assets if asset.role is AssetRole.CAMERA]
    recorders = [asset for asset in project.assets if asset.role is AssetRole.RECORDER]
    data: dict[str, Any] = {"schema_version": 2, "positions": {}, "missing": {}, "deltas": {}}
    placement = sync.get("placement", "text")
    if (scope == "full" or sync.get("mode") == "camera" or placement == "text"
            or not cameras or not recorders):
        data["skipped"] = "nothing to locate" if not cameras or not recorders else (
            "transcribe.recorder_scope=full" if scope == "full"
            else "sync.mode=camera" if sync.get("mode") == "camera"
            else "sync.placement=text: cameras are placed by their speech")
        return data
    search_s = float(sync.get("locate_search_s", 120.0))
    previous_positions = (previous or {}).get("positions", {})
    group_hints = (previous or {}).get("deltas", {})
    workers = cpu_workers(settings)
    windows = {
        camera.id: camera_windows(project.source_dir / camera.path, _duration(camera))
        for camera in cameras if _has_audio(camera)
    }
    for camera in cameras:
        if not windows.get(camera.id):
            reason = "camera has no sound" if not _has_audio(camera) else "clip is too short"
            for recorder in recorders:
                data["missing"].setdefault(camera.id, {})[recorder.id] = reason
    for recorder in recorders:
        source = project.source_dir / recorder.path
        info = recorder.manual.get("media_info", {})
        duration = float(info.get("duration_s") or 0.0)
        rec_epoch = epoch(info)
        pan = effective_audio(recorder).pan
        stream = info.get("audio_stream_index")
        found: dict[str, Located] = {}
        # A clock offset learned on an earlier run narrows the search per camera:
        # the clip's own one first, else one its group agreed on.
        for camera in cameras:
            own = previous_positions.get(camera.id, {}).get(recorder.id, {}).get("delta_s")
            delta = own if own is not None else group_hints.get(
                f"{_group(camera)}|{recorder.id}")
            cam_epoch = epoch(camera.manual.get("media_info", {}))
            if camera.id not in windows or not windows[camera.id] or None in (
                    delta, cam_epoch, rec_epoch):
                continue
            predicted = cam_epoch + delta - rec_epoch
            for name, radius in (("predicted", search_s), ("wide", WIDE_SEARCH_S)):
                span = (max(0.0, predicted - radius),
                        min(duration, predicted + _duration(camera) + radius))
                if span[1] <= span[0]:
                    continue
                result = search_many(source, {camera.id: windows[camera.id]}, span,
                                     pan=pan, stream=stream, workers=workers,
                                     strict=True).get(camera.id)
                if result is not None:
                    found[camera.id] = Located(result[0], result[1], name)
                    break
        # Everything else in one pass over the recorder, all cameras at once.
        rest = {key: value for key, value in windows.items() if value and key not in found}
        if rest:
            count = sum(len(value) for value in rest.values())

            def report(fraction: float, name: str = Path(recorder.path).name,
                       cameras_count: int = len(rest), count: int = count) -> None:
                if progress is not None:
                    progress(fraction, f"{name}: {cameras_count} camera(s), {count} windows")

            for camera_id, (position, sharpness) in search_many(
                    source, rest, None, pan=pan, stream=stream, progress=report,
                    duration_s=duration, workers=workers).items():
                found[camera_id] = Located(position, sharpness, "full")
        rejected = _out_of_order(cameras, found)
        for camera_id in rejected:
            del found[camera_id]
        _search_between_files(
            cameras, windows, found, source, duration, search_s, pan=pan, stream=stream,
            workers=workers,
        )
        for camera in cameras:
            if camera.id not in windows or not windows[camera.id]:
                continue
            located = found.get(camera.id)
            if located is None:
                data["missing"].setdefault(camera.id, {})[recorder.id] = (
                    rejected[camera.id] if camera.id in rejected
                    else "no acoustic match agreed by two windows")
                continue
            entry = {"position_s": round(located.position_s, 3),
                     "sharpness": round(located.sharpness, 1), "method": located.method}
            cam_epoch = epoch(camera.manual.get("media_info", {}))
            if cam_epoch is not None and rec_epoch is not None:
                entry["delta_s"] = round(rec_epoch + located.position_s - cam_epoch, 3)
            data["positions"].setdefault(camera.id, {})[recorder.id] = entry
        data["deltas"].update(_group_deltas(cameras, data["positions"], recorder.id))
    for camera_id in list(data["missing"]):
        if camera_id in data["positions"]:  # found in another recorder
            del data["missing"][camera_id]
    return data


def _group_deltas(
    cameras: list[Asset], positions: dict[str, Any], recorder_id: str,
) -> dict[str, float]:
    """The clock delta of each group whose clips agree on it.

    ``recorded_at`` may be the copy time of the file rather than its recording
    time; then every clip has its own delta and a group value is a false hint.
    Only clips found by the unconstrained pass vote.
    """
    by_group: dict[str, list[float]] = {}
    for camera in cameras:
        entry = positions.get(camera.id, {}).get(recorder_id, {})
        if entry.get("method") == "full" and entry.get("delta_s") is not None:
            by_group.setdefault(_group(camera), []).append(float(entry["delta_s"]))
    result = {}
    for group, values in by_group.items():
        middle = float(np.median(values))
        agree = [value for value in values if abs(value - middle) <= HINT_AGREE_S]
        if len(agree) >= 2 and len(agree) >= HINT_MAJORITY * len(values):
            result[f"{group}|{recorder_id}"] = round(float(np.median(agree)), 3)
    return result


def _out_of_order(cameras: list[Asset], found: dict[str, Located]) -> dict[str, str]:
    """Positions that contradict the order of their camera's files, with the reason.

    Files of one camera are recorded one after another, so their positions in
    a recorder must follow their natural name order without overlapping. The
    largest set of clips that does is kept (ties go to sharper matches); the
    others are false peaks and are searched again between their neighbours.
    """
    rejected: dict[str, str] = {}
    for members in _groups(cameras).values():
        chain = [camera for camera in members if camera.id in found]
        if len(chain) < 2:
            continue
        # Longest chain in name order whose starts follow each other's ends.
        best: list[tuple[float, int]] = []  # (score, previous index)
        for index, camera in enumerate(chain):
            weight = 1.0 + min(found[camera.id].sharpness, 999.0) / 1000.0
            score, link = weight, -1
            for before in range(index):
                if found[camera.id].position_s >= _order_end(found[chain[before].id],
                                                             _duration(chain[before])):
                    if best[before][0] + weight > score:
                        score, link = best[before][0] + weight, before
            best.append((score, link))
        keep: set[str] = set()
        index = max(range(len(chain)), key=lambda item: best[item][0])
        while index >= 0:
            keep.add(chain[index].id)
            index = best[index][1]
        for camera in chain:
            if camera.id not in keep:
                rejected[camera.id] = (
                    f"match at {found[camera.id].position_s:.1f} s contradicts the order "
                    "of the camera's files")
    return rejected


def _order_end(located: Located, duration_s: float) -> float:
    """The earliest start the next file of the same camera may have."""
    if located.end_s is not None:
        return located.end_s - ORDER_TOLERANCE_S
    return located.position_s + duration_s - max(ORDER_TOLERANCE_S,
                                                 ORDER_TOLERANCE_RATE * duration_s)


def _groups(cameras: list[Asset]) -> dict[str, list[Asset]]:
    by_group: dict[str, list[Asset]] = {}
    for camera in cameras:
        by_group.setdefault(_group(camera), []).append(camera)
    return {group: sorted(members, key=lambda item: _natural(str(item.path)))
            for group, members in by_group.items()}


def _search_between_files(
    cameras: list[Asset], windows: dict[str, list[tuple[float, np.ndarray]]],
    found: dict[str, Located], source: Path, duration: float, search_s: float, *,
    pan: str | None, stream: int | None, workers: int,
) -> None:
    """A second, narrow search for clips the full pass missed or misplaced.

    Files of one camera are often consecutive: split at a size limit (DJI and
    Canon at 4 GB) or recorded one after another. That is only a hint — pauses
    happen — so the clip is searched after the end of the previous file of the
    same camera and before the start of the next one, and placed only when its
    windows pass the strict vote. In a narrow range far fewer false peaks
    compete with a weak real match.
    """
    changed = True
    while changed:  # a clip found this way can lead to the next one
        changed = False
        for ordered in _groups(cameras).values():
            for index, camera in enumerate(ordered):
                if camera.id in found or not windows.get(camera.id):
                    continue
                before = next((item for item in reversed(ordered[:index])
                               if item.id in found), None)
                after = next((item for item in ordered[index + 1:] if item.id in found), None)
                if before is None and after is None:
                    continue
                if before is not None:
                    expected = found[before.id].position_s + _duration(before)
                    low = expected - search_s
                    high = (found[after.id].position_s + search_s if after is not None
                            else expected + _duration(camera) + search_s)
                    method = "between_files" if after is not None else "after_previous_file"
                else:
                    assert after is not None
                    low = found[after.id].position_s - _duration(camera) - search_s
                    high = found[after.id].position_s + search_s
                    method = "before_next_file"
                span = (max(0.0, low), min(duration, high))
                if span[1] <= span[0]:
                    continue
                result = search_many(source, {camera.id: windows[camera.id]}, span, pan=pan,
                                     stream=stream, workers=workers,
                                     strict=True).get(camera.id)
                if result is not None:
                    found[camera.id] = Located(result[0], result[1], method)
                    changed = True


def _natural(value: str) -> list[tuple[int, int, str]]:
    import re

    return [(0, int(part), "") if part.isdigit() else (1, 0, part.lower())
            for part in re.split(r"(\d+)", value)]


def locate_camera(
    camera: Path, recorder: Path, recorder_asset: Asset | None, recorder_duration_s: float,
    camera_duration_s: float, predicted_s: float | None, search_s: float, workers: int = 1,
) -> Located | None:
    """Recorder time of ``camera``'s start: around ``predicted_s`` first, then wider."""
    windows = camera_windows(camera, camera_duration_s)
    if not windows:
        return None
    attempts: list[tuple[str, tuple[float, float] | None]] = []
    if predicted_s is not None:
        for name, radius in (("predicted", search_s), ("wide", WIDE_SEARCH_S)):
            low = max(0.0, predicted_s - radius)
            high = min(recorder_duration_s, predicted_s + camera_duration_s + radius)
            if high > low:
                attempts.append((name, (low, high)))
    attempts.append(("full", None))
    pan = effective_audio(recorder_asset).pan if recorder_asset is not None else None
    stream = (recorder_asset.manual.get("media_info", {}).get("audio_stream_index")
              if recorder_asset is not None else None)
    for method, span in attempts:
        found = search_many(recorder, {"camera": windows}, span, pan=pan, stream=stream,
                            duration_s=recorder_duration_s, workers=workers).get("camera")
        if found is not None:
            return Located(found[0], found[1], method)
    return None


def camera_windows(camera: Path, duration_s: float) -> list[tuple[float, np.ndarray]]:
    """``(camera time of the window centre, samples)`` spread evenly over the clip.

    Longer clips get more windows (one per ``WINDOW_SPACING_S``, up to
    ``MAX_WINDOWS``): a distant camera microphone matches the recorder only in
    some places, and the vote needs at least two of them.
    """
    if duration_s < MIN_CLIP_S:
        return []
    count = int(min(MAX_WINDOWS, max(MIN_WINDOWS, duration_s // WINDOW_SPACING_S)))
    length = min(WINDOW_S, duration_s)
    result = []
    for index in range(count):
        centre = duration_s * (index + 0.5) / count
        start = max(0.0, min(duration_s - length, centre - length / 2))
        samples = decode(camera, start, length)
        if samples.size >= SR:
            result.append((start + samples.size / SR / 2, samples))
    return result


def search_many(
    recorder: Path, cameras: dict[str, list[tuple[float, np.ndarray]]],
    span: tuple[float, float] | None, *, pan: str | None = None, stream: int | None = None,
    progress: Callable[[float], None] | None = None, duration_s: float | None = None,
    workers: int = 1, strict: bool = False,
) -> dict[str, tuple[float, float]]:
    """Start of each camera in recorder time, decided by windows that agree.

    ``strict`` is for narrow searches: the start needs half of the clip's
    windows and a peak of ``STRICT_MIN_SHARPNESS``.

    Each window keeps its strongest peaks over the whole search, not just one:
    a recorder with an isolated impulse (a clap, a dropped connector) pulls
    every window's single best PHAT peak onto that impulse, and the place
    where the window really belongs is then only its second or third peak.
    Those false peaks put the camera start somewhere different for every
    window, while the real ones agree to a few milliseconds, so the start
    voted for by the most windows wins. A faint noise floor is added because
    stretches of digital silence collapse the median that sharpness divides by.
    """
    windows = [(camera_id, t, samples) for camera_id, items in cameras.items()
               for t, samples in items]
    if not windows:
        return {}
    # Window spectra are block-sized: a bounded batch keeps memory flat however
    # many cameras there are. Every (batch, stretch of the recorder) pair is an
    # independent task; with several workers the stretches overlap by one
    # window so no alignment position falls between two of them.
    batches = [windows[first:first + BATCH_WINDOWS]
               for first in range(0, len(windows), BATCH_WINDOWS)]
    start = span[0] if span else 0.0
    length = (span[1] - span[0]) if span else duration_s
    workers = max(1, int(workers))
    if workers > 1 and length:
        pieces = max(1, min(workers * 2, int(length // (4 * BLOCK_S))))
        overlap = max(len(samples) for _c, _t, samples in windows) / SR
        step = length / pieces
        spans: list[tuple[float, float] | None] = [
            (start + index * step, min(start + length, start + (index + 1) * step + overlap))
            for index in range(pieces)]
    else:
        spans = [span]
    tasks = [(batch_index, piece) for batch_index in range(len(batches)) for piece in spans]
    collected: list[list[tuple[float, float]]] = [[] for _ in windows]
    offsets = [sum(len(batch) for batch in batches[:index]) for index in range(len(batches))]

    def merge(batch_index: int, found: list[list[tuple[float, float]]]) -> None:
        for local, items in enumerate(found):
            target = collected[offsets[batch_index] + local]
            target.extend(items)
            if len(target) > 4 * KEEP_CANDIDATES:
                target[:] = sorted(target, reverse=True)[:KEEP_CANDIDATES]

    if len(tasks) == 1 or workers == 1:
        for number, (batch_index, piece) in enumerate(tasks):
            def report(fraction: float, number: int = number) -> None:
                if progress is not None:
                    progress((number + fraction) / len(tasks))

            merge(batch_index, _collect(recorder, batches[batch_index], piece, pan=pan,
                                        stream=stream, progress=report,
                                        duration_s=duration_s))
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # spawn: the same behaviour on Linux, macOS and Windows, and no fork()
        # of a process that may already run threads.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                 mp_context=context) as pool:
            futures = {
                pool.submit(_collect, recorder, batches[batch_index], piece, pan=pan,
                            stream=stream): batch_index
                for batch_index, piece in tasks
            }
            for done, future in enumerate(as_completed(futures), start=1):
                merge(futures[future], future.result())
                if progress is not None:
                    progress(done / len(tasks))
    candidates = collected
    result: dict[str, tuple[float, float]] = {}
    for camera_id in cameras:
        votes = [(start, index, sharpness)
                 for index, (owner, _t, _s) in enumerate(windows) if owner == camera_id
                 for sharpness, start in sorted(candidates[index], reverse=True)[:KEEP_CANDIDATES]
                 if sharpness >= CANDIDATE_SHARPNESS]
        count = sum(1 for owner, _t, _s in windows if owner == camera_id)
        decided = (_vote(votes, (count + 1) // 2, STRICT_MIN_SHARPNESS)
                   if strict else _vote(votes))
        if decided is not None:
            result[camera_id] = decided
    return result


def _collect(
    recorder: Path, windows: list[tuple[str, float, np.ndarray]],
    span: tuple[float, float] | None, *, pan: str | None, stream: int | None,
    progress: Callable[[float], None] | None = None, duration_s: float | None = None,
) -> list[list[tuple[float, float]]]:
    """``(sharpness, camera start)`` candidates of each window over one recorder pass."""
    win_n = max(len(samples) for _c, _t, samples in windows)
    block_n = max(int(BLOCK_S * SR), 4 * win_n)
    # Offsets up to block_n - win_n never wrap around, so a block-sized FFT suffices.
    nfft = 1 << int(np.ceil(np.log2(block_n)))
    rng = np.random.default_rng(0)
    spectra = [np.conj(np.fft.rfft(_dithered(samples, rng) - samples.mean(), nfft)).astype(
        np.complex64) for _c, _t, samples in windows]
    candidates: list[list[tuple[float, float]]] = [[] for _ in windows]  # (sharpness, start)
    origin = span[0] if span else 0.0
    length_s = (span[1] - span[0]) if span else duration_s
    for block_start, block in stream_blocks(recorder, span, block_n, win_n, pan=pan,
                                            stream=stream):
        if progress is not None and length_s:
            progress(min(1.0, block_start / SR / length_s))
        if block.size < SR:
            continue
        padded = np.zeros(block_n, np.float32)
        padded[:block.size] = _dithered(block, rng) - block.mean()
        block_fft = np.fft.rfft(padded, nfft).astype(np.complex64)
        for index, ((_c, t, samples), spectrum) in enumerate(zip(windows, spectra,
                                                                strict=True)):
            limit = block.size - samples.size + 1
            if limit < 1:
                continue
            product = block_fft * spectrum
            product /= np.abs(product) + 1e-12
            cc = np.abs(np.fft.irfft(product, nfft))
            # The median of every 16th lag is the same noise floor at a fraction of the cost.
            floor = float(np.median(cc[:block_n:16])) or 1e-12
            for offset in _peaks(cc[:limit], PEAKS_PER_BLOCK, samples.size // 2):
                centre = origin + (block_start + offset + samples.size / 2) / SR
                candidates[index].append((float(cc[offset]) / floor, centre - t))
            if len(candidates[index]) > 4 * KEEP_CANDIDATES:
                candidates[index] = sorted(candidates[index], reverse=True)[:KEEP_CANDIDATES]
    return candidates


def _vote(
    votes: list[tuple[float, int, float]], min_windows: int = 2,
    min_sharpness: float = MIN_SHARPNESS,
) -> tuple[float, float] | None:
    """The start most windows agree on (± ``AGREE_S``), with its best sharpness."""
    votes.sort()
    best: tuple[int, float, float, float] | None = None  # windows, total, start, sharpness
    left = 0
    for right in range(len(votes)):
        while votes[right][0] - votes[left][0] > AGREE_S:
            left += 1
        group = votes[left:right + 1]
        strongest: dict[int, tuple[float, float]] = {}
        for start, index, sharpness in group:
            if sharpness > strongest.get(index, (0.0, 0.0))[0]:
                strongest[index] = (sharpness, start)
        windows = len(strongest)
        total = sum(sharp for sharp, _start in strongest.values())
        top = max(sharp for sharp, _start in strongest.values())
        if windows >= max(2, min_windows) and top >= min_sharpness and (
                best is None or (windows, total) > (best[0], best[1])):
            start = float(np.median([value for _sharp, value in strongest.values()]))
            best = (windows, total, start, top)
    return (best[2], best[3]) if best is not None else None


def _peaks(values: np.ndarray, count: int, separation: int) -> list[int]:
    """Indices of the ``count`` highest values at least ``separation`` apart."""
    # Only the top few hundred values can matter: a partial partition instead
    # of sorting a million correlation values for every window and block.
    pool = min(values.size, count * 64)
    top = np.argpartition(values, values.size - pool)[values.size - pool:]
    order = top[np.argsort(values[top])[::-1]]
    chosen: list[int] = []
    for index in order:
        if all(abs(int(index) - other) >= separation for other in chosen):
            chosen.append(int(index))
            if len(chosen) == count:
                break
    return chosen


def _dithered(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise: np.ndarray = rng.standard_normal(samples.size).astype(np.float32) * DITHER
    result: np.ndarray = samples + noise
    return result


def _group(asset: Asset) -> str:
    return asset.group_id or asset.device or asset.id


def _duration(asset: Asset) -> float:
    return float(asset.manual.get("media_info", {}).get("duration_s") or 0.0)


def stream_blocks(
    path: Path, span: tuple[float, float] | None, block_n: int, overlap_n: int,
    *, pan: str | None = None, stream: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Overlapping mono 16 kHz blocks of ``path`` (sample offset within ``span``)."""
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if span is not None:
        command += ["-ss", f"{span[0]:.3f}", "-t", f"{span[1] - span[0]:.3f}"]
    command += ["-i", str(path), "-map", f"0:{stream}" if stream is not None else "0:a:0",
                "-vn", *(["-af", pan] if pan else ["-ac", "1"]), "-ar", str(SR), "-f", "f32le",
                "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    step = block_n - overlap_n
    buffer = np.zeros(0, np.float32)
    position = 0
    try:
        while True:
            chunk = process.stdout.read((block_n - buffer.size) * 4)
            data = np.frombuffer(chunk[: len(chunk) - len(chunk) % 4], dtype="<f4")
            buffer = np.concatenate([buffer, data])
            if buffer.size >= block_n or (not chunk and buffer.size > overlap_n):
                yield position, buffer[:block_n]
                buffer = buffer[step:] if buffer.size > step else np.zeros(0, np.float32)
                position += step
            if not chunk:
                break
    finally:
        process.stdout.close()
        process.kill()
        process.wait()


def decode(path: Path, start: float, duration: float, *, pan: str | None = None,
           stream: int | None = None) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(path), "-map", f"0:{stream}" if stream is not None else "0:a:0", "-vn",
         *(["-af", pan] if pan else ["-ac", "1"]), "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
    )
    if result.returncode:
        return np.zeros(0, np.float32)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def load_positions(project: Project) -> dict[str, Any]:
    return _load(project.work_dir / "stages" / "locate" / "positions.json") or {}


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _has_audio(asset: Asset) -> bool:
    info = asset.manual.get("media_info", {})
    return bool(info.get("audio_channels") or info.get("audio_stream_index") is not None)
