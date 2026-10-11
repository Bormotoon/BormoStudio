"""Second pass of text placement: clips the words did not place, or placed wrongly.

Text placement (``sync.placement=text``) fits every camera by its fast
transcript first. Whatever is left goes through, in this order:

1. **File order.** Files of one camera follow each other in the recorder; a
   text match that breaks the order of its camera's files is a false one
   (repeated phrases) and is dropped (``locate._out_of_order``).
2. **The camera's clock.** The placed clips of a camera tell which moment its
   ``recorded_at`` stamps and how its clock runs against the recorder; a clip
   is searched a few seconds around where that puts it, with its whole sound
   (``sync_camera_clock``). This finds false starts with no words.
3. **Acoustics between neighbours.** A clip whose neighbouring files are
   placed is searched acoustically only between them, with the strict vote of
   ``locate``; a clip with no placed neighbour is searched in the whole
   recorder. A position found this way is fitted acoustically around it.
4. **A stronger transcription** (``sync.rescue=quality``): Whisper in quality
   mode, first on the camera's sound as it is, then band-limited, denoised
   and loudness-normalized, and the words are matched again.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from studio.core.channels import effective_audio
from studio.core.parallel import cpu_workers
from studio.core.project import Asset
from studio.core.transcript import Transcript
from studio.stages.locate import (
    Located,
    _group,
    _has_audio,
    _out_of_order,
    _search_between_files,
    camera_windows,
    search_many,
)
from studio.stages.sync_camera_clock import QUERY_MAX_S, camera_clocks, clock_search
from studio.stages.sync_models import AlignmentMap
from studio.stages.sync_sessions import Fit

# Around an acoustic position the fine fit searches this much more recorder.
FIT_MARGIN_S = 30.0
# A fit around a clock match that starts further from it took another peak.
CLOCK_FIT_AGREE_S = 0.1
# Bad camera sound: speech band, FFT denoise, loudness normalization.
CLEANUP_FILTER = "highpass=f=90,lowpass=f=7500,afftdn=nf=-25,dynaudnorm=f=150:g=15"


def recorder_start(alignment: AlignmentMap) -> float:
    """Recorder time of the clip's first sample (``cam = offset + k * rec``)."""
    return -alignment.offset / (alignment.k or 1.0)


def recorder_end(alignment: AlignmentMap, duration_s: float) -> float:
    """Recorder time of the clip's last sample."""
    return (duration_s - alignment.offset) / (alignment.k or 1.0)


def second_pass(
    *, cameras: list[Asset], recorders: list[Asset], transcripts: dict[str, Transcript],
    originals: dict[str, Asset], source_dir: Path, settings: dict[str, Any],
    output_dir: Path, report: dict[str, Any], warnings: list[str],
    progress: Callable[[str], None] | None = None,
    transcriber: Callable[[Asset, str | None], Transcript] | None = None,
    acoustic: bool = True,
) -> Callable[[Fit, dict[tuple[str, str], AlignmentMap], dict[str, str]], None]:
    """The ``plan_sessions`` hook; results land in ``report`` and ``warnings``.

    ``acoustic=False`` leaves out the acoustic search: ``locate`` already did it.
    """
    sync = settings.get("sync", {}) if isinstance(settings.get("sync"), dict) else {}

    def note(message: str) -> None:
        if progress is not None:
            progress(message)

    def unplaced(fits: dict[tuple[str, str], AlignmentMap]) -> list[Asset]:
        return [camera for camera in cameras
                if not any((camera.id, recorder.id) in fits for recorder in recorders)]

    def run(fit: Fit, fits: dict[tuple[str, str], AlignmentMap],
            rejected: dict[str, str]) -> None:
        order: dict[str, str] = {}
        for recorder in recorders:
            found = _found(cameras, recorder, fits)
            for camera_id, reason in _out_of_order(cameras, found).items():
                del fits[camera_id, recorder.id]
                rejected[f"{camera_id}:{recorder.id}"] = f"text match dropped: {reason}"
                order[camera_id] = reason
        if order:
            report["order_rejected"] = order
        if not unplaced(fits):
            return
        searched: dict[str, Any] = {}
        for recorder in recorders if acoustic else []:
            missing = [camera for camera in unplaced(fits) if _has_audio(camera)]
            if not missing:
                break
            note(f"searching {len(missing)} unplaced clip(s) by their cameras' clocks in "
                 f"{Path(recorder.path).name}")
            clocks = _clock_placement(fit, fits, rejected, cameras, missing, recorder,
                                      source_dir, searched)
            if clocks:
                report.setdefault("camera_clocks", {})[recorder.id] = clocks
            missing = [camera for camera in missing if camera in unplaced(fits)]
            if not missing:
                continue
            note(f"searching {len(missing)} unplaced clip(s) acoustically in "
                 f"{Path(recorder.path).name}")
            for camera, located in _acoustic_search(
                    cameras, missing, recorder, fits, source_dir, sync, settings):
                span = (max(0.0, located.position_s - FIT_MARGIN_S),
                        located.position_s + _duration(camera) + FIT_MARGIN_S)
                result = fit(camera, recorder, source_range=span, acoustic_search=True)
                searched[camera.id] = {
                    "recorder": recorder.id, "position_s": round(located.position_s, 3),
                    "sharpness": round(located.sharpness, 1), "method": located.method,
                    "placed": result is not None,
                }
                if result is not None:
                    fits[camera.id, recorder.id] = result
                    rejected.pop(f"{camera.id}:{recorder.id}", None)
        if searched:
            report["acoustic_search"] = searched
        if sync.get("rescue", "quality") != "quality" or transcriber is None:
            return
        rescued: dict[str, Any] = {}
        for camera in unplaced(fits):
            attempts: list[dict[str, Any]] = []
            for audio_filter in (None, CLEANUP_FILTER):
                note(f"transcribing {Path(camera.path).name} again in quality"
                     + (" with cleaned-up sound" if audio_filter else ""))
                try:
                    transcript = transcriber(originals.get(camera.id, camera), audio_filter)
                except (OSError, RuntimeError, ValueError) as exc:
                    attempts.append({"filter": audio_filter, "error": str(exc)})
                    break
                transcripts[camera.id] = transcript
                saved = output_dir / f"{camera.id}{'-cleaned' if audio_filter else ''}.json"
                saved.parent.mkdir(parents=True, exist_ok=True)
                transcript.save(saved)
                placed = [recorder.id for recorder in recorders
                          if _fit_into(fit, fits, rejected, camera, recorder)]
                attempts.append({"filter": audio_filter, "words": len(transcript.words),
                                 "placed": placed})
                if placed:
                    warnings.append(
                        f"{Path(camera.path).name}: placed after a quality transcription"
                        + (" of cleaned-up sound" if audio_filter else ""))
                    break
            rescued[camera.id] = attempts
        if rescued:
            report["quality_rescue"] = rescued

    return run


def _clock_placement(
    fit: Fit, fits: dict[tuple[str, str], AlignmentMap], rejected: dict[str, str],
    cameras: list[Asset], missing: list[Asset], recorder: Asset, source_dir: Path,
    searched: dict[str, Any],
) -> dict[str, Any]:
    """Place ``missing`` clips where their camera's clock says, when their sound agrees.

    The clock is learned from the clips of the same camera already placed. A
    clip found this way is fitted acoustically around its place; one too short
    for that (no words, too few windows) keeps the found start at the
    recorder's rate, which the recorder-clock tracking refines later.
    """
    starts = {camera.id: recorder_start(fits[camera.id, recorder.id])
              for camera in cameras if (camera.id, recorder.id) in fits}
    clocks = camera_clocks(cameras, starts)
    source, duration, pan, stream = _recorder_audio(recorder, source_dir)
    for camera in missing:
        clock = clocks.get(_group(camera))
        predictions = clock.predict(camera) if clock is not None else []
        if not predictions:
            continue
        match = clock_search(source_dir / camera.path, _duration(camera), source, duration,
                             predictions, pan=pan, stream=stream)
        entry: dict[str, Any] = {
            "recorder": recorder.id, "method": "clock",
            "predicted_s": [round(start, 3) for start, _radius in predictions],
            "placed": match is not None,
        }
        searched[camera.id] = entry
        if match is None:
            continue
        entry.update(position_s=round(match.position_s, 3), sharpness=round(match.sharpness, 1),
                     ratio=round(match.ratio, 2))
        span = (max(0.0, match.position_s - FIT_MARGIN_S),
                match.position_s + _duration(camera) + FIT_MARGIN_S)
        result = fit(camera, recorder, source_range=span, acoustic_search=True)
        if result is None or abs(recorder_start(result) - match.position_s) > CLOCK_FIT_AGREE_S:
            result = AlignmentMap(
                [], -match.position_s, 1.0, 0.0, "acoustic", 1,
                min(_duration(camera), QUERY_MAX_S),
                notes=[f"placed by its camera's clock (predicted at {match.predicted_s:.1f} s) "
                       f"and its whole sound (peak {match.ratio:.1f} times the next one)"])
            entry["map"] = "clock"
        else:
            entry["map"] = "fit"
        fits[camera.id, recorder.id] = result
        rejected.pop(f"{camera.id}:{recorder.id}", None)
    return {group: clock.report() for group, clock in clocks.items()}


def _recorder_audio(recorder: Asset, source_dir: Path) -> tuple[Path, float, str | None,
                                                                int | None]:
    """``(path, duration, pan, stream)`` of the recorder's sound as the searches read it."""
    info = recorder.manual.get("media_info", {})
    # Inside sync the selected stream is already decoded to its own file.
    decoded = Path(recorder.path).is_absolute()
    return (source_dir / recorder.path, float(info.get("duration_s") or 0.0),
            None if decoded else effective_audio(recorder).pan,
            None if decoded else info.get("audio_stream_index"))


def _fit_into(fit: Fit, fits: dict[tuple[str, str], AlignmentMap], rejected: dict[str, str],
              camera: Asset, recorder: Asset) -> bool:
    result = fit(camera, recorder, acoustic_search=False)
    if result is None:
        return False
    fits[camera.id, recorder.id] = result
    rejected.pop(f"{camera.id}:{recorder.id}", None)
    return True


def _found(cameras: list[Asset], recorder: Asset,
           fits: dict[tuple[str, str], AlignmentMap]) -> dict[str, Located]:
    """Placed cameras of ``recorder`` as ``locate`` positions, weighted by evidence.

    The end comes from the map: its rate holds what the recorder lost inside
    the clip (sony_run dji-0927 started 4.6 s before dji-0926's start plus its
    length, and its correct words were dropped as out of order).
    """
    return {
        camera.id: Located(recorder_start(fits[camera.id, recorder.id]),
                           float(fits[camera.id, recorder.id].inliers), "text",
                           recorder_end(fits[camera.id, recorder.id], _duration(camera)))
        for camera in cameras if (camera.id, recorder.id) in fits
    }


def _acoustic_search(
    cameras: list[Asset], missing: list[Asset], recorder: Asset,
    fits: dict[tuple[str, str], AlignmentMap], source_dir: Path, sync: dict[str, Any],
    settings: dict[str, Any],
) -> list[tuple[Asset, Located]]:
    """Recorder positions of ``missing`` clips: between placed neighbours, else anywhere."""
    source, duration, pan, stream = _recorder_audio(recorder, source_dir)
    workers = cpu_workers(settings)
    windows = {camera.id: camera_windows(source_dir / camera.path, _duration(camera))
               for camera in missing}
    found = _found(cameras, recorder, fits)
    before = set(found)
    _search_between_files(cameras, windows, found, source, duration,
                          float(sync.get("locate_search_s", 120.0)), pan=pan, stream=stream,
                          workers=workers)
    lonely = {camera_id: value for camera_id, value in windows.items()
              if value and camera_id not in found}
    if lonely:
        for camera_id, (position, sharpness) in search_many(
                source, lonely, None, pan=pan, stream=stream, duration_s=duration,
                workers=workers).items():
            found[camera_id] = Located(position, sharpness, "full")
        for camera_id in _out_of_order(cameras, found):
            if camera_id not in before:
                found.pop(camera_id, None)
    return [(camera, found[camera.id]) for camera in missing
            if camera.id in found and camera.id not in before]


def quality_transcriber(
    settings: dict[str, Any], source_dir: Path, work_dir: Path,
) -> Callable[[Asset, str | None], Transcript]:
    """Whisper in quality mode for one camera, loaded on first use."""
    from studio.stages.transcribe import TranscribeSettings, WhisperEngine

    transcribe = settings.get("transcribe", {})
    streams = transcribe.get("audio_streams", {}) if isinstance(transcribe, dict) else {}
    engine: list[WhisperEngine] = []

    def run(camera: Asset, audio_filter: str | None) -> Transcript:
        if not engine:
            base = TranscribeSettings.from_mapping(transcribe)
            engine.append(WhisperEngine(replace(base, mode="quality"), digest_cache=work_dir))
        stream = streams.get(camera.id, camera.manual.get("media_info", {}).get(
            "audio_stream_index"))
        pan = effective_audio(camera).pan
        chain = ",".join(part for part in (pan, audio_filter) if part) or None
        return engine[0].transcribe(source_dir / camera.path, stream, chain)

    return run


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _duration(asset: Asset) -> float:
    return float(asset.manual.get("media_info", {}).get("duration_s") or 0.0)
