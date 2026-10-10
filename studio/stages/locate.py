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
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from studio.core.channels import effective_audio
from studio.core.digest import file_digest
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
DITHER = 3e-5  # about -90 dBFS
BATCH_WINDOWS = 40


@dataclass(frozen=True, slots=True)
class Located:
    position_s: float  # recorder time of the camera's start
    sharpness: float
    method: str


class LocateStage:
    id = "locate"
    title = "Locate cameras"
    after: tuple[str, ...] = ("scan",)
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = (
        "sync.mode", "sync.locate_search_s", "transcribe.recorder_scope")
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
        return stable_fingerprint("locate-v1", sources,
                                  select_settings(settings, self.settings_keys))

    def run(self, context: StageContext) -> StageOutput:
        output = context.work_dir / "stages" / "locate" / "positions.json"
        previous = _load(context.project.work_dir / "stages" / "locate" / "positions.json")
        data = locate_project(context.project, context.settings, previous)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        return StageOutput((output,))


def locate_project(
    project: Project, settings: dict[str, Any], previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sync = settings.get("sync", {}) if isinstance(settings.get("sync"), dict) else {}
    transcribe = settings.get("transcribe", {})
    scope = transcribe.get("recorder_scope", "auto") if isinstance(transcribe, dict) else "auto"
    cameras = [asset for asset in project.assets if asset.role is AssetRole.CAMERA]
    recorders = [asset for asset in project.assets if asset.role is AssetRole.RECORDER]
    data: dict[str, Any] = {"schema_version": 2, "positions": {}, "missing": {}, "deltas": {}}
    if scope == "full" or sync.get("mode") == "camera" or not cameras or not recorders:
        data["skipped"] = "nothing to locate" if not cameras or not recorders else (
            "transcribe.recorder_scope=full" if scope == "full" else "sync.mode=camera")
        return data
    search_s = float(sync.get("locate_search_s", 120.0))
    hints = (previous or {}).get("deltas", {})
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
        # A clock offset learned on an earlier run narrows the search per camera.
        for camera in cameras:
            delta = hints.get(f"{_group(camera)}|{recorder.id}")
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
                                     pan=pan, stream=stream).get(camera.id)
                if result is not None:
                    found[camera.id] = Located(result[0], result[1], name)
                    break
        # Everything else in one pass over the recorder, all cameras at once.
        rest = {key: value for key, value in windows.items() if value and key not in found}
        if rest:
            for camera_id, (position, sharpness) in search_many(
                    source, rest, None, pan=pan, stream=stream).items():
                found[camera_id] = Located(position, sharpness, "full")
        for camera in cameras:
            if camera.id not in windows or not windows[camera.id]:
                continue
            located = found.get(camera.id)
            if located is None:
                data["missing"].setdefault(camera.id, {})[recorder.id] = (
                    "no acoustic match agreed by two windows")
                continue
            entry = {"position_s": round(located.position_s, 3),
                     "sharpness": round(located.sharpness, 1), "method": located.method}
            cam_epoch = epoch(camera.manual.get("media_info", {}))
            if cam_epoch is not None and rec_epoch is not None:
                delta = rec_epoch + located.position_s - cam_epoch
                entry["delta_s"] = round(delta, 3)
                data["deltas"][f"{_group(camera)}|{recorder.id}"] = round(delta, 3)
            data["positions"].setdefault(camera.id, {})[recorder.id] = entry
    for camera_id in list(data["missing"]):
        if camera_id in data["positions"]:  # found in another recorder
            del data["missing"][camera_id]
    return data


def locate_camera(
    camera: Path, recorder: Path, recorder_asset: Asset | None, recorder_duration_s: float,
    camera_duration_s: float, predicted_s: float | None, search_s: float,
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
        found = search_many(recorder, {"camera": windows}, span, pan=pan,
                            stream=stream).get("camera")
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
) -> dict[str, tuple[float, float]]:
    """Start of each camera in recorder time, decided by windows that agree.

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
    candidates: list[list[tuple[float, float]]] = []
    # Window spectra are block-sized: a bounded batch keeps memory flat however
    # many cameras there are, at the cost of one more decode of the recorder.
    for first in range(0, len(windows), BATCH_WINDOWS):
        candidates += _collect(recorder, windows[first:first + BATCH_WINDOWS], span,
                               pan=pan, stream=stream)
    result: dict[str, tuple[float, float]] = {}
    for camera_id in cameras:
        votes = [(start, index, sharpness)
                 for index, (owner, _t, _s) in enumerate(windows) if owner == camera_id
                 for sharpness, start in sorted(candidates[index], reverse=True)[:KEEP_CANDIDATES]
                 if sharpness >= CANDIDATE_SHARPNESS]
        decided = _vote(votes)
        if decided is not None:
            result[camera_id] = decided
    return result


def _collect(
    recorder: Path, windows: list[tuple[str, float, np.ndarray]],
    span: tuple[float, float] | None, *, pan: str | None, stream: int | None,
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
    for block_start, block in stream_blocks(recorder, span, block_n, win_n, pan=pan,
                                            stream=stream):
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
            floor = float(np.median(cc[:block_n])) or 1e-12
            for offset in _peaks(cc[:limit], PEAKS_PER_BLOCK, samples.size // 2):
                centre = origin + (block_start + offset + samples.size / 2) / SR
                candidates[index].append((float(cc[offset]) / floor, centre - t))
            if len(candidates[index]) > 4 * KEEP_CANDIDATES:
                candidates[index] = sorted(candidates[index], reverse=True)[:KEEP_CANDIDATES]
    return candidates


def _vote(votes: list[tuple[float, int, float]]) -> tuple[float, float] | None:
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
        if windows >= 2 and top >= MIN_SHARPNESS and (
                best is None or (windows, total) > (best[0], best[1])):
            start = float(np.median([value for _sharp, value in strongest.values()]))
            best = (windows, total, start, top)
    return (best[2], best[3]) if best is not None else None


def _peaks(values: np.ndarray, count: int, separation: int) -> list[int]:
    """Indices of the ``count`` highest values at least ``separation`` apart."""
    order = np.argsort(values)[::-1]
    chosen: list[int] = []
    for index in order[: count * 64]:
        if all(abs(int(index) - other) >= separation for other in chosen):
            chosen.append(int(index))
            if len(chosen) == count:
                break
    return chosen


def _dithered(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return samples + rng.standard_normal(samples.size).astype(np.float32) * DITHER


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


def decode(path: Path, start: float, duration: float) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le",
         "-"],
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
