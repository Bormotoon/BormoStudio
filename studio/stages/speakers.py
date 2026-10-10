"""Assign transcript words to isolated microphone tracks by relative energy."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from studio.core.project import AssetRole, Project, describe_artifact, stable_fingerprint
from studio.core.timeline import SourcePlacement
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
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, object]) -> list[Requirement]:
        return [Requirement("binary", "ffmpeg", "decode microphone channels")]

    def decide(self, project: Project, settings: dict[str, object]) -> Decision:
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

    def fingerprint(self, project: Project, settings: dict[str, object]) -> str:
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

            selected = ({Transcript.load(transcript).metadata.get("source_asset_id")
                         or _primary_source_id(project)} if transcript and transcript.is_file()
                        else {_primary_source_id(project)})
        identities = [describe_artifact(project.source_dir, project.source_dir / asset.path)
                      for asset in project.assets if asset.id in selected]
        return stable_fingerprint(
            "speakers-v4", content, tracks, identities, project.placements, conf,
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
        envelopes: dict[str, np.ndarray] = {}
        for track, speaker in tracks.items():
            asset_id, channel_text = track.rsplit(":", 1)
            asset = assets[asset_id]
            if asset_id not in decoded:
                decoded[asset_id] = decode_channels(
                    context.project.source_dir / asset.path,
                    int(asset.manual.get("media_info", {}).get("audio_channels") or 1),
                )
            channel = int(channel_text)
            samples = decoded[asset_id][:, channel]
            envelope = timeline_envelope(samples, placements[asset_id], transcript.duration,
                                         step_s=step)
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
            "tracks": tracks, "step_s": step, "map": "SourcePlacement",
            "smoothing": {"min_turn_s": float(conf.get("min_turn_s", 0.3)),
                          "max_gap_s": float(conf.get("max_gap_s", 0.3)),
                          "policy": "isolated A-B-A known-speaker islands only"},
        }, indent=2) + "\n", encoding="utf-8")
        return StageOutput((output, report, raw_output), {
            "outputs": {**context.project.outputs, "speakers": [output, report],
                        "speakers_raw": [raw_output]},
        })


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
    if not tracks:
        for asset in assets.values():
            channels = int(asset.manual.get("media_info", {}).get("audio_channels") or 1)
            if channels > 1 and conf.get("method", "auto") == "auto":
                continue
            for channel in range(channels):
                if f"{asset.id}:{channel}" not in disabled:
                    tracks[f"{asset.id}:{channel}"] = f"{asset.id}-ch{channel + 1}"
    for key, speaker in tracks.items():
        if not isinstance(key, str) or not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker tracks require nonempty string names")
        try:
            asset_id, channel = key.rsplit(":", 1)
            number = int(channel)
        except ValueError as exc:
            raise ValueError(f"invalid microphone track: {key}; use asset_id:channel") from exc
        if asset_id not in assets:
            raise ValueError(f"unknown recorder asset: {asset_id}")
        count = int(assets[asset_id].manual.get("media_info", {}).get("audio_channels") or 1)
        if not 0 <= number < count:
            raise ValueError(f"microphone channel out of range: {key}")
    return tracks


def run_pyannote(context: StageContext, transcript: Transcript, conf: dict) -> StageOutput:
    from studio.stages.speaker_backend import diarize
    from studio.stages.timeline import _primary_source_id

    source_id = transcript.metadata.get("source_asset_id") or _primary_source_id(context.project)
    asset = next(asset for asset in context.project.assets if asset.id == source_id)
    placement = next((item for item in context.project.placements
                      if item.asset_id == source_id), None)
    if placement is None:
        raise ValueError("pyannote source placement is missing; run sync")
    directory = context.work_dir / "stages" / "speakers"
    raw = diarize(context.project.source_dir / asset.path, directory,
                  model=str(conf.get("model", "pyannote/speaker-diarization-3.1")),
                  device=str(conf.get("device", "cpu")))
    turns = []
    for item in raw:
        start = max(float(item["start"]), placement.in_s)
        end = min(float(item["end"]), placement.in_s + placement.duration_s)
        if end <= start:
            continue
        left = max(0.0, placement.offset_s + (start - placement.in_s) * placement.k)
        right = min(transcript.duration,
                    placement.offset_s + (end - placement.in_s) * placement.k)
        if right > left:
            turns.append(SpeakerTurn(left, right, item["speaker"], "pyannote"))
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
                                 "method": "pyannote", "source_asset_id": source_id,
                                 "model": conf.get("model", "pyannote/speaker-diarization-3.1"),
                                 "device": conf.get("device", "cpu"),
                                 "labels": "anonymous, local to one primary source"},
                                indent=2) + "\n", encoding="utf-8")
    logs = tuple(sorted((directory / "logs").glob("*.log")))
    return StageOutput((output, raw_output, report, *logs), {"outputs": {
        **context.project.outputs, "speakers": [output, report], "speakers_raw": [raw_output],
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
) -> np.ndarray:
    # Integrate sample energy over each mapped bin rather than interpolating a
    # single sample; affine drift and negative offsets preserve RMS evidence.
    count = math.ceil(duration_s / step_s)
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(samples), dtype=np.float64)))
    boundaries = np.arange(count + 1) * step_s
    source_times = placement.in_s + (boundaries - placement.offset_s) / placement.k
    source_times = np.clip(source_times, placement.in_s,
                           placement.in_s + placement.duration_s)
    positions = np.clip(np.round(source_times * sample_rate).astype(np.int64), 0, len(samples))
    lengths = np.diff(positions)
    energies = np.diff(cumulative[positions]) / np.maximum(lengths, 1)
    return np.sqrt(np.maximum(energies, 0))


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
