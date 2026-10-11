"""Second pass of text placement: clips the words did not place, or placed wrongly.

Text placement (``sync.placement=text``) fits every camera by its fast
transcript first. Whatever is left goes through, in this order:

1. **File order.** Files of one camera follow each other in the recorder; a
   text match that breaks the order of its camera's files is a false one
   (repeated phrases) and is dropped (``locate._out_of_order``).
2. **Acoustics between neighbours.** A clip whose neighbouring files are
   placed is searched acoustically only between them, with the strict vote of
   ``locate``; a clip with no placed neighbour is searched in the whole
   recorder. A position found this way is fitted acoustically around it.
3. **A stronger transcription** (``sync.rescue=quality``): Whisper in quality
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
    _has_audio,
    _out_of_order,
    _search_between_files,
    camera_windows,
    search_many,
)
from studio.stages.sync_models import AlignmentMap
from studio.stages.sync_sessions import Fit

# Around an acoustic position the fine fit searches this much more recorder.
FIT_MARGIN_S = 30.0
# Bad camera sound: speech band, FFT denoise, loudness normalization.
CLEANUP_FILTER = "highpass=f=90,lowpass=f=7500,afftdn=nf=-25,dynaudnorm=f=150:g=15"


def recorder_start(alignment: AlignmentMap) -> float:
    """Recorder time of the clip's first sample (``cam = offset + k * rec``)."""
    return -alignment.offset / (alignment.k or 1.0)


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
    """Placed cameras of ``recorder`` as ``locate`` positions, weighted by evidence."""
    return {
        camera.id: Located(recorder_start(fits[camera.id, recorder.id]),
                           float(fits[camera.id, recorder.id].inliers), "text")
        for camera in cameras if (camera.id, recorder.id) in fits
    }


def _acoustic_search(
    cameras: list[Asset], missing: list[Asset], recorder: Asset,
    fits: dict[tuple[str, str], AlignmentMap], source_dir: Path, sync: dict[str, Any],
    settings: dict[str, Any],
) -> list[tuple[Asset, Located]]:
    """Recorder positions of ``missing`` clips: between placed neighbours, else anywhere."""
    source = source_dir / recorder.path
    info = recorder.manual.get("media_info", {})
    duration = float(info.get("duration_s") or 0.0)
    # Inside sync the selected stream is already decoded to its own file.
    decoded = Path(recorder.path).is_absolute()
    pan = None if decoded else effective_audio(recorder).pan
    stream = None if decoded else info.get("audio_stream_index")
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
