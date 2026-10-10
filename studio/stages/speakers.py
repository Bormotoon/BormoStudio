"""Assign transcript words to isolated microphone tracks by relative energy."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from studio.core.channels import effective_audio
from studio.core.project import AssetRole, Project, describe_artifact, stable_fingerprint
from studio.core.timeline import AudioWarpMap, SourcePlacement, TimeDomain
from studio.core.transcript import Transcript, Word
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    source: str = "mics"


class SpeakersStage:
    id = "speakers"
    title = "Speakers"
    after: tuple[str, ...] = ("timeline",)
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ("speakers",)
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        speakers = settings.get("speakers", {})
        if isinstance(speakers, dict) and speakers.get("method") == "pyannote":
            return [Requirement("module", "diarization", "pyannote speaker diarization")]
        return [Requirement("binary", "ffmpeg", "decode microphone channels")]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        speakers = settings.get("speakers", {})
        conf = speakers if isinstance(speakers, dict) else {}
        method = str(conf.get("method", "auto"))
        if method == "off":
            return Decision.skip("speaker detection is off")
        if method == "pyannote":
            from studio.modules.manager import ModuleManager

            if "timeline" not in project.transcripts:
                return Decision.blocked("timeline transcript is missing", "run timeline")
            if not ModuleManager().installed("diarization"):
                return Decision.blocked("diarization module is missing",
                                        "install diarization module")
            return Decision.run({"method": "pyannote"})
        tracks = microphone_tracks(project, conf)
        if len(set(tracks.values())) < 2:
            return (Decision.skip("fewer than two microphone speakers") if method == "auto"
                    else Decision.blocked("two microphone speakers are required",
                                          "configure speakers.tracks"))
        if "timeline" not in project.transcripts:
            return Decision.blocked("timeline transcript is missing", "run timeline")
        missing = {key.split(":", 1)[0] for key in tracks} - {
            placement.asset_id for placement in project.placements
        }
        if missing:
            return Decision.blocked(f"microphone placements are missing: {sorted(missing)}",
                                    "align recorder tracks before speaker attribution")
        return Decision.run({"method": "mics", "tracks": tracks})

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        transcript = project.transcripts.get("timeline")
        content = (
            transcript.read_text(encoding="utf-8")
            if transcript and transcript.is_file()
            else ""
        )
        conf = settings.get("speakers", {})
        conf = conf if isinstance(conf, dict) else {}
        tracks = {} if conf.get("method") == "pyannote" else microphone_tracks(project, conf)
        selected = {key.split(":", 1)[0] for key in tracks}
        if conf.get("method") == "pyannote":
            from studio.stages.timeline import _primary_source_id

            source_id = (Transcript.load(transcript).metadata.get("source_asset_id")
                         or _primary_source_id(project) if transcript and transcript.is_file()
                         else _primary_source_id(project))
            selected = {source_id} if source_id is not None else set()
            if transcript and transcript.is_file() and conf.get("match_files", True):
                try:
                    selected = {item[0] for item in diarization_sources(
                        project, Transcript.load(transcript))}
                except ValueError:
                    pass
        identities = [describe_artifact(project.source_dir, project.source_dir / asset.path,
                                        project.work_dir)
                      for asset in project.assets if asset.id in selected]
        return stable_fingerprint(
            "speakers-v6", content, tracks, identities, project.placements,
            [warp for warp in project.audio_warp_maps if warp.source_asset_id in selected], conf,
        )

    def run(self, context: StageContext) -> StageOutput:
        conf = context.settings.get("speakers", {})
        conf = conf if isinstance(conf, dict) else {}
        transcript = Transcript.load(context.project.transcripts["timeline"])
        if conf.get("method") == "pyannote":
            return run_pyannote(context, transcript, conf)
        step = float(conf.get("step_s", 0.05))
        tracks = microphone_tracks(context.project, conf)
        placements = {item.asset_id: item for item in context.project.placements}
        assets = {asset.id: asset for asset in context.project.assets}
        decoded: dict[str, np.ndarray] = {}
        for track in tracks:
            asset_id = track.rsplit(":", 1)[0]
            if asset_id not in decoded:
                asset = assets[asset_id]
                decoded[asset_id] = decode_channels(
                    context.project.source_dir / asset.path,
                    int(asset.manual.get("media_info", {}).get("audio_channels") or 1),
                )
        identities: dict[str, object] | None = None
        if conf.get("match_files", True):
            tracks, identities = match_microphone_lanes(
                context.project, conf, tracks, decoded, placements,
                threshold=float(conf.get("match_threshold", 0.9)),
            )
        envelopes: dict[str, np.ndarray] = {}
        warped_assets: set[str] = set()
        for track, speaker in tracks.items():
            asset_id, channel_text = track.rsplit(":", 1)
            channel = int(channel_text)
            samples = decoded[asset_id][:, channel]
            warp = recorder_warp(context.project, asset_id)
            warped_assets.update([asset_id] if warp else [])
            envelope = timeline_envelope(samples, placements[asset_id], transcript.duration,
                                         step_s=step, warp=warp)
            # Several files/channels assigned to one speaker share one lane.
            envelopes[speaker] = np.maximum(envelopes.get(speaker, np.zeros_like(envelope)),
                                            envelope)
        turns = assign_words_to_mics(
            transcript.words, envelopes, step_s=step,
            margin_db=float(conf.get("margin_db", 6.0)),
            silence_floor_db=float(conf.get("silence_floor_db", -60.0)),
            max_gap_s=float(conf.get("max_gap_s", 0.3)),
        )
        raw_turns = turns
        turns = smooth_short_turns(
            turns, min_turn_s=float(conf.get("min_turn_s", 0.3)),
            max_gap_s=float(conf.get("max_gap_s", 0.3)),
        )
        output = context.work_dir / "stages" / "speakers" / "diarization.json"
        save_turns(output, turns)
        raw_output = output.with_name("diarization.raw.json")
        save_turns(raw_output, raw_turns)
        report = output.with_name("report.json")
        report.write_text(json.dumps({
            "schema_version": 1, "time_domain": "timeline", "method": "mics",
            "tracks": tracks, "step_s": step,
            "map": {asset_id: "AudioWarpMap" if asset_id in warped_assets else "SourcePlacement"
                    for asset_id in sorted({key.rsplit(":", 1)[0] for key in tracks})},
            "smoothing": {"min_turn_s": float(conf.get("min_turn_s", 0.3)),
                          "max_gap_s": float(conf.get("max_gap_s", 0.3)),
                          "policy": "isolated A-B-A known-speaker islands only"},
            "identities": identities,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return StageOutput((output, report, raw_output), {
            "outputs": {**context.project.outputs, "speakers": [output, report],
                        "speakers_raw": [raw_output]},
        })


def named_microphone_tracks(project: Project, conf: dict[str, object]) -> dict[str, str]:
    """Lanes whose speaker the user named in settings or on the Material screen."""
    from studio.core.recorder_tracks import microphone_map

    configured = conf.get("tracks", {})
    return {**(configured if isinstance(configured, dict) else {}), **microphone_map(project)}


def _overlap_s(first: SourcePlacement, second: SourcePlacement) -> float:
    def span(item: SourcePlacement) -> tuple[float, float]:
        return item.offset_s, item.offset_s + item.duration_s * item.k

    (a, b), (c, d) = span(first), span(second)
    return max(0.0, min(b, d) - max(a, c))


def match_microphone_lanes(
    project: Project,
    conf: dict[str, object],
    tracks: dict[str, str],
    decoded: dict[str, np.ndarray],
    placements: dict[str, SourcePlacement],
    *,
    threshold: float,
    sample_rate: int = 16000,
) -> tuple[dict[str, str], dict[str, object] | None]:
    """Give one person one name across recorder files that do not play together.

    Lanes in one file, or in files that overlap on the timeline, are different
    people. Names the user gave are kept and can be inherited by matching lanes.
    """
    from studio.stages.speaker_identity import VoiceLabel, match_labels, voice_signature

    groups = sorted({track.rsplit(":", 1)[0] for track in tracks})
    simultaneous = {
        frozenset((first, second))
        for index, first in enumerate(groups) for second in groups[index + 1:]
        if first in placements and second in placements
        and _overlap_s(placements[first], placements[second]) > 1.0
    }
    if len(groups) < 2 or len(simultaneous) == len(groups) * (len(groups) - 1) // 2:
        return tracks, None
    named = named_microphone_tracks(project, conf)
    labels: list[VoiceLabel] = []
    for track, speaker in tracks.items():
        asset_id, channel = track.rsplit(":", 1)
        signature = voice_signature(decoded[asset_id][:, int(channel)], sample_rate)
        if signature is None:
            continue
        placement = placements.get(asset_id)
        labels.append(VoiceLabel(
            asset_id, track, signature, placement.offset_s if placement else 0.0,
            fixed_name=speaker if track in named else None,
        ))
    match = match_labels(labels, threshold=threshold, simultaneous=simultaneous)
    renamed = dict(tracks)
    for label in labels:
        name = match.names[(label.group, label.label)]
        renamed[label.label] = name if name not in tracks else tracks[name]
    changes = {track: renamed[track] for track in tracks if renamed[track] != tracks[track]}
    return renamed, {
        "method": "voice-signature", "threshold": threshold,
        "simultaneous": sorted(sorted(pair) for pair in simultaneous),
        "renamed": changes, "pairs": match.similarities,
    }


def _separate_session(
    asset_id: str, named_assets: set[str], placements: dict[str, SourcePlacement],
    conf: dict[str, object],
) -> bool:
    """An unnamed recorder of another session: its lanes join by voice matching.

    Naming the lanes of one session must not hide the next session's lanes;
    cross-file matching then gives them the names of the same people.
    """
    if asset_id in named_assets or not conf.get("match_files", True):
        return False
    placement = placements.get(asset_id)
    named = [placements[item] for item in named_assets if item in placements]
    return placement is not None and bool(named) and all(
        _overlap_s(placement, other) <= 1.0 for other in named)


def microphone_tracks(project: Project, conf: dict[str, object]) -> dict[str, str]:
    configured = conf.get("tracks", {})
    if not isinstance(configured, dict):
        raise ValueError("speakers.tracks must map asset:channel to speaker")
    from studio.core.recorder_tracks import microphone_map

    # Lanes named on the Material screen are project choices and win over
    # settings defaults for the same channel.
    tracks = {**configured, **microphone_map(project)}
    assets = {asset.id: asset for asset in project.assets if asset.role is AssetRole.RECORDER}
    disabled = {
        f"{track.asset_id}:{channel}"
        for track in project.recorder_tracks if not track.enabled
        for channel in track.channels
    }
    named_assets = {key.rsplit(":", 1)[0] for key in tracks if isinstance(key, str)}
    placements = {item.asset_id: item for item in project.placements}
    for asset in assets.values():
        if named_assets and not _separate_session(asset.id, named_assets, placements, conf):
            # Unnamed files beside named ones (a backup recorder) stay out.
            continue
        # Identical or linked channels are one microphone, not two speakers.
        distinct = effective_audio(asset).channels or (0,)
        if len(distinct) > 1 and conf.get("method", "auto") == "auto":
            continue
        for channel in distinct:
            if f"{asset.id}:{channel}" not in disabled:
                tracks[f"{asset.id}:{channel}"] = f"{asset.id}-ch{channel + 1}"
    for key, speaker in tracks.items():
        if not isinstance(key, str) or not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker tracks require nonempty string names")
        try:
            asset_id, channel_text = key.rsplit(":", 1)
            number = int(channel_text)
        except ValueError as exc:
            raise ValueError(f"invalid microphone track: {key}; use asset_id:channel") from exc
        if asset_id not in assets:
            raise ValueError(f"unknown recorder asset: {asset_id}")
        count = int(assets[asset_id].manual.get("media_info", {}).get("audio_channels") or 1)
        if not 0 <= number < count:
            raise ValueError(f"microphone channel out of range: {key}")
    return tracks


Interval = tuple[float, float]


def _subtract(span: Interval, covered: list[Interval]) -> list[Interval]:
    pieces = [span]
    for start, end in covered:
        pieces = [
            part for left, right in pieces
            for part in ((left, min(right, start)), (max(left, end), right))
            if part[1] - part[0] > 1e-6
        ]
    return pieces


def diarization_sources(
    project: Project, transcript: Transcript, *, all_files: bool = True
) -> list[tuple[str, SourcePlacement, list[Interval]]]:
    """Placed sources to diarize and the timeline each one is responsible for.

    The transcript source comes first. Another placed file is added when it
    covers mostly new timeline (a separate session), so overlapping copies of
    one conversation are not diarized twice.
    """
    from studio.stages.timeline import _primary_source_id

    primary = transcript.metadata.get("source_asset_id") or _primary_source_id(project)
    placements = {item.asset_id: item for item in project.placements}
    if primary not in placements:
        raise ValueError("pyannote source placement is missing; run sync")
    roles = {asset.id: asset.role for asset in project.assets}
    others = sorted(
        (item for item in project.placements
         if item.asset_id != primary
         and roles.get(item.asset_id) in {AssetRole.CAMERA, AssetRole.RECORDER}),
        key=lambda item: -item.duration_s * item.k,
    )
    chosen: list[tuple[str, SourcePlacement, list[Interval]]] = []
    covered: list[Interval] = []
    for item in [placements[str(primary)], *(others if all_files else [])]:
        span = (max(0.0, item.offset_s),
                min(transcript.duration, item.offset_s + item.duration_s * item.k))
        if span[1] <= span[0]:
            continue
        new = _subtract(span, covered)
        length = sum(end - start for start, end in new)
        if chosen and (length < 5.0 or length < 0.5 * (span[1] - span[0])):
            continue
        chosen.append((item.asset_id, item, new))
        covered.extend(new)
    return chosen


def _timeline_turns(
    raw: list[dict[str, Any]], placement: SourcePlacement, intervals: list[Interval],
    duration: float, label: Callable[[str], str],
) -> list[SpeakerTurn]:
    turns = []
    for item in raw:
        start = max(float(item["start"]), placement.in_s)
        end = min(float(item["end"]), placement.in_s + placement.duration_s)
        if end <= start:
            continue
        left = max(0.0, placement.offset_s + (start - placement.in_s) * placement.k)
        right = min(duration, placement.offset_s + (end - placement.in_s) * placement.k)
        for low, high in intervals:
            clipped = (max(left, low), min(right, high))
            if clipped[1] > clipped[0]:
                turns.append(SpeakerTurn(*clipped, label(item["speaker"]), "pyannote"))
    return turns


def run_pyannote(context: StageContext, transcript: Transcript, conf: dict) -> StageOutput:
    from studio.stages import speaker_backend
    from studio.stages.speaker_identity import VoiceLabel, match_labels, voice_signature

    project = context.project
    directory = context.work_dir / "stages" / "speakers"
    sources = diarization_sources(project, transcript,
                                  all_files=bool(conf.get("match_files", True)))
    assets = {asset.id: asset for asset in project.assets}
    results = []
    for order, (asset_id, placement, intervals) in enumerate(sources):
        path = project.source_dir / assets[asset_id].path
        diarization = speaker_backend.diarize(
            path, directory if order == 0 else directory / "sources" / asset_id,
            model=str(conf.get("model", "pyannote/speaker-diarization-3.1")),
            device=str(conf.get("device", "cpu")),
        )
        results.append((asset_id, placement, intervals, path, diarization))
    names: dict[tuple[str, str], str] = {
        (asset_id, str(turn["speaker"])): str(turn["speaker"])
        for asset_id, _p, _i, _path, result in results[:1] for turn in result.turns
    }
    identities: dict[str, object] | None = None
    if len(results) > 1:
        labels = sorted({(asset_id, str(turn["speaker"]))
                         for asset_id, _p, _i, _path, result in results
                         for turn in result.turns})
        embedded = all(label in result.embeddings
                       for asset_id, _p, _i, _path, result in results
                       for owner, label in labels if owner == asset_id)
        voices = []
        for order, (asset_id, placement, _intervals, path, result) in enumerate(results):
            samples = None if embedded else decode_channels(path, 1)[:, 0]
            for owner, label in labels:
                if owner != asset_id:
                    continue
                own = [turn for turn in result.turns if turn["speaker"] == label]
                if embedded:
                    signature = np.asarray(result.embeddings[label], dtype=np.float64)
                else:
                    assert samples is not None
                    pieces = [samples[int(float(turn["start"]) * 16000):
                                      int(float(turn["end"]) * 16000)] for turn in own]
                    found = voice_signature(np.concatenate(pieces)) if pieces else None
                    if found is None:
                        continue
                    signature = found
                start = min(float(turn["start"]) for turn in own)
                first = placement.offset_s + (start - placement.in_s) * placement.k
                # Labels of the transcript source name the shared speakers.
                voices.append(VoiceLabel(asset_id, label, signature, order * 1e7 + first))
        threshold = float(conf.get("embedding_threshold" if embedded else "match_threshold",
                                   0.5 if embedded else 0.9))
        match = match_labels(voices, threshold=threshold)
        names.update(match.names)
        identities = {
            "method": "pyannote-embedding" if embedded else "voice-signature",
            "threshold": threshold,
            "sources": [asset_id for asset_id, *_rest in results],
            "names": {f"{group}:{label}": name for (group, label), name in names.items()},
            "pairs": match.similarities,
        }
    def namer(owner: str) -> Callable[[str], str]:
        return lambda label: names.get((owner, label), f"{label}@{owner}")

    turns = [
        turn
        for asset_id, placement, intervals, _path, result in results
        for turn in _timeline_turns(
            result.turns, placement, intervals, transcript.duration, namer(asset_id)
        )
    ]
    turns.sort(key=lambda turn: (turn.start, turn.end))
    # Preserve intersecting turns as explicit overlap for camera selection.
    boundaries = sorted({value for turn in turns for value in (turn.start, turn.end)})
    resolved = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        active = {turn.speaker for turn in turns if turn.start <= start and turn.end >= end}
        if active:
            resolved.append(SpeakerTurn(start, end,
                                       next(iter(active)) if len(active) == 1 else "overlap",
                                       "pyannote"))
    output = directory / "diarization.json"
    raw_output = directory / "diarization.raw.json"
    save_turns(raw_output, turns)
    save_turns(output, smooth_short_turns(merge_adjacent_turns(resolved, 0),
                                        min_turn_s=float(conf.get("min_turn_s", 0.3)),
                                        max_gap_s=0))
    report = directory / "report.json"
    report.write_text(json.dumps({"schema_version": 1, "time_domain": "timeline",
                                 "method": "pyannote", "source_asset_id": sources[0][0],
                                 "model": conf.get("model", "pyannote/speaker-diarization-3.1"),
                                 "device": conf.get("device", "cpu"),
                                 "labels": "anonymous; matched across sources"
                                 if identities else "anonymous, local to one primary source",
                                 "identities": identities},
                                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logs = tuple(sorted(directory.rglob("logs/*.log")))
    return StageOutput((output, raw_output, report, *logs), {"outputs": {
        **project.outputs, "speakers": [output, report], "speakers_raw": [raw_output],
    }})


def decode_channels(path: Path, channels: int, sample_rate: int = 16000) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-vn", "-ar", str(sample_rate), "-ac", str(channels), "-f", "f32le", "-"],
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"microphone decode failed: {result.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, channels)


def timeline_envelope(
    samples: np.ndarray, placement: SourcePlacement, duration_s: float,
    *, step_s: float = 0.05, sample_rate: int = 16000,
    warp: tuple[AudioWarpMap, SourcePlacement] | None = None,
) -> np.ndarray:
    # Integrate sample energy over each mapped bin rather than interpolating a
    # single sample; affine drift and negative offsets preserve RMS evidence.
    count = math.ceil(duration_s / step_s)
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(samples), dtype=np.float64)))
    boundaries = np.arange(count + 1) * step_s
    source_times = placement.in_s + (boundaries - placement.offset_s) / placement.k
    if warp is not None:
        # Piecewise (complex) sync: timeline → target camera file → recorder file.
        warped = warped_source_times(boundaries, *warp)
        source_times = np.where(np.isnan(warped), source_times, warped)
    source_times = np.clip(source_times, placement.in_s,
                           placement.in_s + placement.duration_s)
    positions = np.clip(np.round(source_times * sample_rate).astype(np.int64), 0, len(samples))
    lengths = np.diff(positions)
    energies = np.diff(cumulative[positions]) / np.maximum(lengths, 1)
    return cast(np.ndarray, np.sqrt(np.maximum(energies, 0)))


def warped_source_times(
    timeline_s: np.ndarray, warp: AudioWarpMap, camera: SourcePlacement,
) -> np.ndarray:
    """Recorder file times for timeline times through a warp map; NaN outside it."""
    rendered = camera.in_s + (timeline_s - camera.offset_s) / camera.k
    result = np.full(timeline_s.shape, np.nan)
    for piece in warp.pieces:
        inside = ((rendered >= piece.rendered_start_s)
                  & (rendered <= piece.rendered_start_s + piece.rendered_duration_s))
        ratio = piece.source_duration_s / piece.rendered_duration_s
        result[inside] = (piece.source_start_s
                          + (rendered[inside] - piece.rendered_start_s) * ratio)
    return result


def recorder_warp(
    project: Project, asset_id: str,
) -> tuple[AudioWarpMap, SourcePlacement] | None:
    """The piecewise map of a recorder onto the camera covering the most timeline."""
    cameras = {item.asset_id: item for item in project.placements}
    candidates = [
        (warp, cameras[warp.target_asset_id]) for warp in project.audio_warp_maps
        if warp.source_asset_id == asset_id and warp.target_asset_id in cameras
        and warp.source_domain is TimeDomain.FILE and len(warp.pieces) > 1
    ]
    return max(candidates, key=lambda item: item[1].duration_s * item[1].k, default=None)


def assign_words_to_mics(
    words: list[Word],
    envelopes: dict[str, np.ndarray],
    *,
    step_s: float = 0.05,
    margin_db: float = 6.0,
    silence_floor_db: float = -60.0,
    max_gap_s: float = 0.3,
) -> list[SpeakerTurn]:
    assignments: list[SpeakerTurn] = []
    for word in words:
        start = max(0, int(word.start / step_s))
        end = max(start + 1, math.ceil(word.end / step_s))
        energies = sorted(
            (
                (speaker, float(np.mean(np.square(values[start:end]))))
                for speaker, values in envelopes.items()
                if len(values[start:end])
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not energies or 10 * math.log10(max(energies[0][1], 1e-12)) < silence_floor_db:
            speaker = "unknown"
        elif len(energies) == 1:
            speaker = energies[0][0]
        else:
            first_db = 10 * math.log10(max(energies[0][1], 1e-12))
            second_db = 10 * math.log10(max(energies[1][1], 1e-12))
            speaker = energies[0][0] if first_db - second_db >= margin_db else "overlap"
        assignments.append(SpeakerTurn(word.start, word.end, speaker))
    return merge_adjacent_turns(assignments, max_gap_s)


def smooth_short_turns(
    turns: list[SpeakerTurn], *, min_turn_s: float = 0.3, max_gap_s: float = 0.3,
) -> list[SpeakerTurn]:
    """Relabel isolated short known-speaker islands using original neighbors."""
    if any(not math.isfinite(value) or value < 0 for value in (min_turn_s, max_gap_s)):
        raise ValueError("speaker smoothing thresholds must be finite and non-negative")
    if min_turn_s == 0:
        return list(turns)
    result = list(turns)
    for index in range(1, len(turns) - 1):
        previous, current, following = turns[index - 1:index + 2]
        if (previous.speaker in {"unknown", "overlap"}
                or current.speaker in {"unknown", "overlap"}
                or previous.speaker != following.speaker
                or previous.speaker == current.speaker
                or current.end - current.start >= min_turn_s
                or previous.end - previous.start < min_turn_s
                or following.end - following.start < min_turn_s
                or not 0 <= current.start - previous.end <= max_gap_s
                or not 0 <= following.start - current.end <= max_gap_s
                or not previous.source == current.source == following.source):
            continue
        result[index] = SpeakerTurn(current.start, current.end, previous.speaker, current.source)
    return merge_adjacent_turns(result, max_gap_s)


def merge_adjacent_turns(turns: list[SpeakerTurn], max_gap_s: float = 0.3) -> list[SpeakerTurn]:
    merged: list[SpeakerTurn] = []
    for turn in turns:
        same_speaker = merged and merged[-1].speaker == turn.speaker
        close = merged and turn.start - merged[-1].end <= max_gap_s
        if same_speaker and close:
            previous = merged[-1]
            merged[-1] = SpeakerTurn(previous.start, turn.end, turn.speaker, turn.source)
        else:
            merged.append(turn)
    return merged


def save_turns(path: Path, turns: list[SpeakerTurn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(turn) for turn in turns], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
