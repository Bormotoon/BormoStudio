import json
import math
import struct
import wave
from types import SimpleNamespace

import studio.reels.analysis.audio_features as audio
from studio.core.project import Project
from studio.core.timeline import TimeDomain
from studio.core.transcript import Transcript
from studio.reels.analysis.audio_features import annotate_audio, audio_identity, resolve_audio
from studio.reels.analysis.contracts import coerce_moment_record
from studio.stages.reels import _settings


def test_real_audio_measurement_and_content_identity(tmp_path):
    path = tmp_path / "master.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(
            b"".join(
                struct.pack(
                    "<h", int(10000 * math.sin(i * 2 * math.pi * 440 / 16000)) if i < 16000 else 0
                )
                for i in range(32000)
            )
        )
    project = Project(tmp_path, tmp_path / "work", outputs={"master_wav": [path]})
    transcript = Transcript(path, "en", 2, [], time_domain=TimeDomain.TIMELINE)
    before = audio_identity(project, transcript)
    record = coerce_moment_record(
        {"start": 0, "end": 2, "quote": "Evidence", "candidate_id": "one"}
    )
    assert record is not None
    records, report = annotate_audio([record], path, _settings({}))
    assert report["measured"] == 1
    assert records[0].audio_energy_db < 0
    assert 0.45 < records[0].audio_silence_ratio < 0.55
    data = bytearray(path.read_bytes())
    data[-1] = 1
    path.write_bytes(data)
    assert audio_identity(project, transcript) != before


def test_domain_resolution_never_uses_original_audio_for_edited_time(tmp_path):
    path = tmp_path / "source.wav"
    path.write_bytes(b"source")
    project = Project(tmp_path, tmp_path / "work", outputs={"master_wav": [path]})
    transcript = Transcript(path, "en", 2, [], time_domain=TimeDomain.EDITED)
    assert resolve_audio(project, transcript) is None
    project.outputs["program"] = [path]
    assert resolve_audio(project, transcript) == path


def test_cache_reuse_invalidates_content_parameters_and_corruption(tmp_path, monkeypatch):
    path = tmp_path / "source.wav"
    path.write_bytes(b"first")
    cache = tmp_path / "cache"
    record = coerce_moment_record(
        {"start": 0, "end": 2, "quote": "Evidence", "candidate_id": "one"}
    )
    calls = []

    def probe(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=0, stderr="mean_volume: -12 dB")

    monkeypatch.setattr(audio.subprocess, "run", probe)
    settings = _settings({})
    assert annotate_audio([record], path, settings, cache_dir=cache)[1]["cache_hits"] == 0
    assert annotate_audio([record], path, settings, cache_dir=cache)[1]["cache_hits"] == 1
    assert len(calls) == 1
    for entry in cache.glob("*.json"):
        entry.write_text(json.dumps({"schema_version": 1, "energy_db": -12, "silence_ratio": 2}))
    annotate_audio([record], path, settings, cache_dir=cache)
    assert len(calls) == 2
    path.write_bytes(b"other")
    annotate_audio([record], path, settings, cache_dir=cache)
    assert len(calls) == 3
    settings["audio_noise_db"] = -40
    annotate_audio([record], path, settings, cache_dir=cache)
    assert len(calls) == 4
    assert not list(cache.glob("*.tmp"))


def test_placement_probe_uses_inverse_time_and_tempo(tmp_path, monkeypatch):
    from studio.core.timeline import SourcePlacement

    placement = SourcePlacement("camera", 10, 3, 10, k=2)
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    commands = []

    def probe(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="mean_volume: -12 dB")

    monkeypatch.setattr(audio.subprocess, "run", probe)
    record = coerce_moment_record(
        {"start": 12, "end": 16, "quote": "Evidence", "candidate_id": "one"}
    )
    _, report = annotate_audio([record], source, _settings({}), placement=placement, stream="0:2")
    command = commands[0]
    assert command[command.index("-ss") + 1] == "4.0"
    assert command[command.index("-t") + 1] == "2.0"
    assert command[command.index("-map") + 1] == "0:2"
    assert command[command.index("-af") + 1].startswith("atempo=0.5,")
    assert report["mapping"] == "SourcePlacement"


def test_synchronized_voice_and_edited_cut_audio_without_program(tmp_path):
    from studio.core.project import stable_fingerprint
    from studio.core.timeline import SourcePlacement

    voice = tmp_path / "voice.wav"
    with wave.open(str(voice), "wb") as output:
        output.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        output.writeframes(struct.pack("<h", 1000) * 96000)
    project = Project(
        tmp_path,
        tmp_path / "work",
        placements=[SourcePlacement("camera", 0, 0, 2)],
        outputs={"sync:camera": [voice]},
    )
    transcript = Transcript(voice, "en", 2, [], time_domain=TimeDomain.TIMELINE)
    with audio.analysis_audio(project, transcript) as (path, placement, stream):
        assert path.is_file() and placement is None
        assert stream == "0:a:0"
    assert not path.exists()
    transcript.time_domain = TimeDomain.EDITED
    transcript.duration = 1
    report = tmp_path / "render.json"
    report.write_text(
        json.dumps(
            {
                "edited_timing_fingerprint": stable_fingerprint(1, []),
                "timeline_to_rendered": [
                    {"timeline_start": 1, "timeline_end": 2, "edited_start": 0, "edited_end": 1}
                ],
            }
        )
    )
    project.outputs["program_report"] = [report]
    with audio.analysis_audio(project, transcript) as (path, _, _):
        with wave.open(str(path)) as output:
            assert output.getnframes() / output.getframerate() == 1


def test_raw_recorder_is_replayed_through_its_warp_without_voice_files(tmp_path):
    from pathlib import Path

    from studio.core.project import Asset, AssetRole
    from studio.core.timeline import AudioWarpMap, AudioWarpPiece, SourcePlacement

    recorder = tmp_path / "rec.wav"
    with wave.open(str(recorder), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        # Two seconds of silence, then two seconds of tone.
        output.writeframes(b"\x00\x00" * 32000 + struct.pack("<h", 8000) * 32000)
    project = Project(tmp_path, tmp_path / "work")
    project.assets = [Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER)]
    project.placements = [SourcePlacement("camera", 0, 0, 3)]
    # The camera hears recorder seconds 1-4 starting at its own second 0.
    project.audio_warp_maps = [AudioWarpMap("w", "rec", "camera", (
        AudioWarpPiece(1, 1, 0, 1, "copy"), AudioWarpPiece(2, 2, 1, 2, "copy"),
    ), TimeDomain.FILE, 3)]
    transcript = Transcript(recorder, "en", 3, [], time_domain=TimeDomain.TIMELINE)
    assert audio_identity(project, transcript) != "unavailable"
    with audio.analysis_audio(project, transcript) as (path, placement, _stream):
        assert placement is None
        import subprocess

        rate = 16000
        frames = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate),
             "-f", "s16le", "-"], check=True, capture_output=True,
        ).stdout
        values = struct.unpack(f"<{len(frames) // 2}h", frames)
        first = max(abs(v) for v in values[: rate // 2])
        tone = max(abs(v) for v in values[int(rate * 1.5): int(rate * 2.5)])
        assert first < 100 < tone
