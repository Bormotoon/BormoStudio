"""Optional loudness/silence evidence measured in the transcript time domain."""

import json
import math
import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from studio.core.project import Project, stable_fingerprint
from studio.core.timeline import AudioWarpMap, SourcePlacement, TimeDomain
from studio.core.transcript import Transcript
from studio.core.workspace import published
from studio.modules.models import digest
from studio.reels.analysis.contracts import MomentRecord, replace_record
from studio.reels.analysis.ranking import ranking_value

_MEAN = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_DURATION = re.compile(r"silence_duration:\s*(\d+(?:\.\d+)?)")


def resolve_audio(project: Project, transcript: Transcript) -> Path | None:
    key = {TimeDomain.TIMELINE: "master_wav", TimeDomain.EDITED: "program"}.get(
        transcript.time_domain
    )
    return (
        next((path for path in project.outputs.get(key, []) if path.is_file()), None)
        if key
        else None
    )


def audio_identity(project: Project, transcript: Transcript) -> str:
    path = resolve_audio(project, transcript)
    if path is None:
        voices = synchronized_voices(project)
        if voices:
            report = next(iter(project.outputs.get("program_report", [])), None)
            return stable_fingerprint(
                [(key, digest(value)) for key, value in voices.items()],
                project.placements,
                project.audio_warp_maps,
                report.read_text() if report and report.is_file() else "",
            )
        warps = replayable_warps(project)
        if warps:
            assets = {asset.id: asset for asset in project.assets}
            report = next(iter(project.outputs.get("program_report", [])), None)
            return stable_fingerprint(
                "replay", warps, project.placements,
                [digest(project.source_dir / assets[warp.source_asset_id].path)
                 for warp in warps],
                report.read_text() if report and report.is_file() else "",
            )
        placed = resolve_placed_audio(project, transcript)
        if placed is not None:
            return stable_fingerprint(str(placed[0]), digest(placed[0]), placed[1], placed[2])
    return stable_fingerprint(str(path), digest(path)) if path else "unavailable"


def replayable_warps(project: Project) -> list[AudioWarpMap]:
    """File-domain warps onto placed cameras whose recorder file is present."""
    placed = {item.asset_id for item in project.placements}
    assets = {asset.id: asset for asset in project.assets}
    return [
        warp for warp in project.audio_warp_maps
        if warp.source_domain is TimeDomain.FILE and warp.target_asset_id in placed
        and warp.source_asset_id in assets
        and (project.source_dir / assets[warp.source_asset_id].path).is_file()
    ]


def replay_voices(project: Project, directory: Path) -> dict[str, Path]:
    """Re-render camera voices from raw recorder files through their warp maps.

    Used when synchronization kept no voice WAVs: analysis then hears the same
    pieces, stretch factors and lead silence the sync render would produce.
    """
    from studio.stages.sync_render import render_audio_plan

    assets = {asset.id: asset for asset in project.assets}
    placements = {item.asset_id: item for item in project.placements}
    voices: dict[str, Path] = {}
    for warp in replayable_warps(project):
        target = str(warp.target_asset_id)
        if target in voices:
            continue
        pieces = [(piece.source_start_s, piece.source_duration_s,
                   piece.source_duration_s / piece.rendered_duration_s)
                  for piece in warp.pieces]
        output = directory / f"replay-{target}.wav"
        render_audio_plan(
            project.source_dir / assets[warp.source_asset_id].path, output, pieces,
            lead_silence_s=warp.pieces[0].rendered_start_s,
            duration_s=placements[target].duration_s, map_id=warp.id,
            source_asset_id=warp.source_asset_id, target_asset_id=target,
            strategy=warp.strategy, sample_rate=16000, channels=1, codec="pcm_s16le",
        )
        voices[target] = output
    return voices


def synchronized_voices(project: Project) -> dict[str, Path]:
    return {
        key.split(":", 1)[1]: paths[0]
        for key, paths in project.outputs.items()
        if key.startswith("sync:") and paths and paths[0].is_file()
    }


AudioInput = tuple[Path | None, SourcePlacement | None, str]


@contextmanager
def analysis_audio(project: Project, transcript: Transcript) -> Iterator[AudioInput]:
    source = resolve_audio(project, transcript)
    if source is not None:
        yield source, None, "0:a:0"
        return
    voices = synchronized_voices(project)
    replay = not voices and bool(replayable_warps(project))
    if (voices or replay) and transcript.time_domain in {TimeDomain.TIMELINE, TimeDomain.EDITED}:
        from studio.stages.sync_outputs import render_voice_master

        with tempfile.TemporaryDirectory(prefix="uvf-analysis-audio-") as temporary:
            if replay:
                voices = replay_voices(project, Path(temporary))
            master = render_voice_master(project.placements, voices, Path(temporary) / "master.wav")
            if transcript.time_domain is TimeDomain.TIMELINE:
                yield master, None, "0:a:0"
                return
            report_path = next(iter(project.outputs.get("program_report", [])), None)
            report = json.loads(report_path.read_text()) if report_path else {}
            timing = stable_fingerprint(
                transcript.duration, [(word.start, word.end) for word in transcript.words]
            )
            if report.get("edited_timing_fingerprint") != timing:
                raise ValueError("missing or stale edited audio mapping")
            mapping = report.get("timeline_to_rendered")
            if not isinstance(mapping, list) or not mapping:
                raise ValueError("missing edited audio mapping")
            filters, labels = [], []
            cursor = 0.0
            for i, piece in enumerate(mapping):
                a, b, c, d = [
                    float(piece[key])
                    for key in ("timeline_start", "timeline_end", "edited_start", "edited_end")
                ]
                if (
                    not all(math.isfinite(v) for v in (a, b, c, d))
                    or a < 0
                    or b <= a
                    or c < cursor - 1e-6
                    or d <= c
                    or abs(b - a - d + c) > 1e-6
                    or d > transcript.duration + 1e-6
                ):
                    raise ValueError("invalid edited audio mapping")
                filters.append(
                    f"[0:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,"
                    f"adelay={round(c * 48000)}S:all=1[p{i}]"
                )
                labels.append(f"[p{i}]")
                cursor = d
            filters.append(
                "".join(labels)
                + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0,"
                + f"apad,atrim=duration={transcript.duration}[out]"
            )
            edited = Path(temporary) / "edited.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(master),
                    "-filter_complex",
                    ";".join(filters),
                    "-map",
                    "[out]",
                    "-ar",
                    "48000",
                    str(edited),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            yield edited, None, "0:a:0"
            return
    placed = resolve_placed_audio(project, transcript)
    yield placed if placed else (None, None, "0:a:0")


def resolve_placed_audio(
    project: Project, transcript: Transcript
) -> tuple[Path, SourcePlacement, str] | None:
    if transcript.time_domain is not TimeDomain.TIMELINE:
        return None
    if len(transcript.metadata.get("source_asset_ids", [])) > 1:
        return None
    source_id = transcript.metadata.get("source_asset_id")
    asset = next((asset for asset in project.assets if asset.id == source_id), None)
    placement = next((item for item in project.placements if item.asset_id == source_id), None)
    if asset is None or placement is None:
        return None
    path = project.source_dir / asset.path
    if not path.is_file():
        return None
    stream = asset.manual.get("media_info", {}).get("audio_stream_index")
    stream = (
        f"0:{stream}"
        if isinstance(stream, int) and not isinstance(stream, bool) and stream >= 0
        else "0:a:0"
    )
    return path, placement, stream


def parse_features(stderr: str, span: float) -> tuple[float, float] | None:
    mean = _MEAN.search(stderr)
    if mean is None or span <= 0:
        return None
    starts = [float(value) for value in _START.findall(stderr)]
    durations = [float(value) for value in _DURATION.findall(stderr)]
    total = sum(durations)
    if len(starts) > len(durations):
        total += max(0, span - starts[-1])
    return float(mean.group(1)), round(min(1, max(0, total / span)), 4)


def _cached_features(path: Path) -> tuple[float, float] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return None
        energy, silence = payload["energy_db"], payload["silence_ratio"]
        if (
            isinstance(energy, bool)
            or isinstance(silence, bool)
            or not isinstance(energy, (int, float))
            or not isinstance(silence, (int, float))
            or not math.isfinite(energy)
            or energy > 0
            or not math.isfinite(silence)
            or not 0 <= silence <= 1
        ):
            return None
        return float(energy), float(silence)
    except (OSError, ValueError, KeyError):
        return None


def annotate_audio(
    records: list[MomentRecord],
    source: Path | None,
    settings: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    placement: SourcePlacement | None = None,
    stream: str = "0:a:0",
) -> tuple[list[MomentRecord], dict[str, Any]]:
    report: dict[str, Any] = {
        "status": "unavailable" if source is None else "ok",
        "source": str(source) if source else None,
        "mapping": "SourcePlacement" if placement else "domain_aligned",
        "measured": 0,
        "cache_hits": 0,
        "cache_errors": [],
        "failures": [],
        "budget_skipped": [],
    }
    if source is None:
        return records, report
    source_digest = None
    if cache_dir is not None:
        try:
            source_digest = digest(source)
        except OSError as exc:
            report["cache_errors"].append(str(exc))
    chosen = {
        record.candidate_id
        for record in sorted(records, key=ranking_value, reverse=True)[
            : int(settings["audio_max_candidates"])
        ]
    }
    result = []
    for record in records:
        if record.candidate_id not in chosen:
            report["budget_skipped"].append(record.candidate_id)
            result.append(record)
            continue
        start, span = record.start, record.end - record.start
        filters = ""
        if placement is not None:
            if (
                record.start < placement.offset_s
                or record.end > placement.offset_s + placement.duration_s * placement.k
            ):
                report["failures"].append(
                    {"candidate_id": record.candidate_id, "error": "candidate outside placement"}
                )
                result.append(record)
                continue
            start = placement.in_s + (record.start - placement.offset_s) / placement.k
            span /= placement.k
            rate = 1 / placement.k
            tempos = []
            while rate < 0.5:
                tempos.append("atempo=0.5")
                rate /= 0.5
            while rate > 2:
                tempos.append("atempo=2")
                rate /= 2
            tempos.append(f"atempo={rate}")
            filters = ",".join(tempos) + ","
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-ss",
            str(start),
            "-t",
            str(span),
            "-i",
            str(source),
            "-map",
            stream,
            "-vn",
            "-af",
            filters
            + f"volumedetect,silencedetect=noise={settings['audio_noise_db']}dB:"
            + f"d={settings['audio_silence_min_s']}",
            "-f",
            "null",
            "-",
        ]
        try:
            cache = None
            features = None
            if cache_dir is not None and source_digest is not None:
                key = stable_fingerprint(
                    "audio-features-v2",
                    source_digest,
                    stream,
                    placement,
                    record.start,
                    record.end,
                    settings["audio_noise_db"],
                    settings["audio_silence_min_s"],
                )
                cache = cache_dir / f"{key}.json"
                features = _cached_features(cache)
            if features is not None:
                report["cache_hits"] += 1
            else:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=float(settings["audio_timeout_s"]),
                )
                features = (
                    parse_features(process.stderr, record.end - record.start)
                    if process.returncode == 0
                    else None
                )
                if features is not None and cache is not None:
                    try:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        with published(cache) as temporary:
                            temporary.write_text(
                                json.dumps(
                                    {
                                        "schema_version": 1,
                                        "energy_db": features[0],
                                        "silence_ratio": features[1],
                                    }
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                    except OSError as exc:
                        report["cache_errors"].append(str(exc))
            if features is None:
                raise ValueError("audio probe returned no usable measurement")
            record = replace_record(
                record, audio_energy_db=features[0], audio_silence_ratio=features[1]
            )
            report["measured"] += 1
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            report["failures"].append({"candidate_id": record.candidate_id, "error": str(exc)})
        result.append(record)
    if report["failures"]:
        report["status"] = "partial" if report["measured"] else "failed"
    return result, report
