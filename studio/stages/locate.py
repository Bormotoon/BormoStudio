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
from studio.stages.sync_acoustic import _AMBIGUITY_MARGIN, _correlate_in_block

SR = 16000
WINDOW_S = 8.0
WINDOW_POSITIONS = (0.25, 0.5, 0.75)
BLOCK_S = 64.0
MIN_SHARPNESS = 50.0
AGREE_S = 0.1
WIDE_SEARCH_S = 900.0


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
    data: dict[str, Any] = {"schema_version": 1, "positions": {}, "missing": {}, "deltas": {}}
    if scope == "full" or sync.get("mode") == "camera" or not cameras or not recorders:
        data["skipped"] = "nothing to locate" if not cameras or not recorders else (
            "transcribe.recorder_scope=full" if scope == "full" else "sync.mode=camera")
        return data
    search_s = float(sync.get("locate_search_s", 120.0))
    hints = (previous or {}).get("deltas", {})
    for recorder in recorders:
        source = project.source_dir / recorder.path
        info = recorder.manual.get("media_info", {})
        duration = float(info.get("duration_s") or 0.0)
        rec_epoch = epoch(info)
        groups: dict[str, list[Asset]] = {}
        for camera in cameras:
            groups.setdefault(camera.group_id or camera.device or camera.id, []).append(camera)
        for group, members in groups.items():
            ordered = sorted(members, key=lambda item: -float(
                item.manual.get("media_info", {}).get("duration_s") or 0.0))
            with_sound = [item for item in ordered if _has_audio(item)]
            delta = hints.get(f"{group}|{recorder.id}")
            for camera in with_sound:
                cam_info = camera.manual.get("media_info", {})
                cam_epoch = epoch(cam_info)
                predicted = (cam_epoch + delta - rec_epoch
                             if None not in (cam_epoch, delta, rec_epoch) else None)
                found = locate_camera(
                    project.source_dir / camera.path, source, recorder, duration,
                    float(cam_info.get("duration_s") or 0.0), predicted, search_s,
                )
                if found is None:
                    data["missing"].setdefault(camera.id, {})[recorder.id] = (
                        "no unambiguous acoustic match")
                    continue
                entry = {"position_s": round(found.position_s, 3),
                         "sharpness": round(found.sharpness, 1), "method": found.method}
                if cam_epoch is not None and rec_epoch is not None:
                    delta = rec_epoch + found.position_s - cam_epoch
                    entry["delta_s"] = round(delta, 3)
                    data["deltas"][f"{group}|{recorder.id}"] = round(delta, 3)
                data["positions"].setdefault(camera.id, {})[recorder.id] = entry
            for camera in ordered:
                if camera not in with_sound:
                    data["missing"].setdefault(camera.id, {})[recorder.id] = "camera has no sound"
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
        found = search(recorder, windows, span, pan=pan, stream=stream)
        if found is not None:
            return Located(found[0], found[1], method)
    return None


def camera_windows(camera: Path, duration_s: float) -> list[tuple[float, np.ndarray]]:
    """``(camera time of the window centre, samples)`` for the three probe windows."""
    result = []
    for fraction in WINDOW_POSITIONS:
        start = max(0.0, min(duration_s - WINDOW_S, duration_s * fraction - WINDOW_S / 2))
        samples = decode(camera, start, min(WINDOW_S, duration_s))
        if samples.size >= SR:
            result.append((start + samples.size / SR / 2, samples))
    return result


def search(
    recorder: Path, windows: list[tuple[float, np.ndarray]], span: tuple[float, float] | None,
    *, pan: str | None = None, stream: int | None = None,
) -> tuple[float, float] | None:
    """Camera start in recorder time from windows that agree, or ``None``."""
    win_n = max(len(samples) for _t, samples in windows)
    block_n = max(int(BLOCK_S * SR), 4 * win_n)
    nfft = 1 << int(np.ceil(np.log2(2 * block_n)))
    best = [(0.0, -1, 0.0) for _ in windows]  # (sharpness, position, runner-up)
    origin = span[0] if span else 0.0
    for block_start, block in stream_blocks(recorder, span, block_n, win_n, pan=pan,
                                            stream=stream):
        if block.size < 8:
            continue
        padded = np.zeros(block_n, np.float32)
        padded[:block.size] = block - block.mean()
        block_fft = np.fft.rfft(padded, nfft)
        for index, (_t, samples) in enumerate(windows):
            limit = max(1, block.size - samples.size + 1)
            offset, peak, rival, floor = _correlate_in_block(
                samples, block_fft, nfft, block_n, 1e-8, samples.size)
            if offset >= limit:
                continue
            sharpness, position, runner = best[index]
            current = peak / floor
            here = block_start + offset
            runner = max(runner, rival / floor)
            if current > sharpness:
                if position >= 0 and abs(here - position) > samples.size:
                    runner = max(runner, sharpness)
                best[index] = (current, here, runner)
            else:
                if position >= 0 and abs(here - position) > samples.size:
                    runner = max(runner, current)
                best[index] = (sharpness, position, runner)
    starts: list[tuple[float, float]] = []
    for (t, samples), (sharpness, position, runner) in zip(windows, best, strict=True):
        if position < 0 or sharpness < MIN_SHARPNESS or (
                runner > 0 and sharpness < runner * _AMBIGUITY_MARGIN):
            continue
        centre = origin + (position + samples.size / 2) / SR
        starts.append((centre - t, sharpness))
    if not starts:
        return None
    for candidate, _sharpness in starts:
        agreeing = [(value, sharp) for value, sharp in starts if abs(value - candidate) <= AGREE_S]
        if len(agreeing) >= min(2, len(windows)):
            return (float(np.median([value for value, _s in agreeing])),
                    float(min(sharp for _v, sharp in agreeing)))
    return None


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
