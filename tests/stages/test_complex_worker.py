import json
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.timeline import AudioWarpMap, AudioWarpPiece, TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.runner import run_stage_process
from studio.stages.sync import _run_complex


@pytest.mark.parametrize("recorder_count", [1, 2])
def test_complex_worker_publishes_real_voice_and_warp(tmp_path: Path, recorder_count: int) -> None:
    work = tmp_path / "_studio"
    work.mkdir()
    source = tmp_path / "recorder.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x10\x00" * 48000 * 4)
    words = tuple(Word(f"token{i}", 0.2 + i * 0.1, 0.25 + i * 0.1) for i in range(25))
    transcripts = {}
    for name, duration in [("rec", 4.0), ("cam", 3.0)]:
        path = work / f"{name}.json"
        Transcript(source, "en", duration, [Segment(0.0, 3.0, words)]).save(path)
        transcripts[name] = path
    project_path = work / "project.json"
    assets = [
        Asset("rec", Path("recorder.wav"), AssetKind.AUDIO, AssetRole.RECORDER),
        Asset("cam", Path("camera.mov"), AssetKind.VIDEO, AssetRole.CAMERA),
    ]
    if recorder_count == 2:
        assets.append(Asset("rec2", Path("recorder.wav"), AssetKind.AUDIO, AssetRole.RECORDER))
        transcripts["rec2"] = transcripts["rec"]
    Project(tmp_path, work, assets=assets, transcripts=transcripts).save(project_path)
    settings = work / "settings.yaml"
    settings.write_text("sync:\n  mode: complex\n  recorder_mode: all\n"
                        "  master_wav: true\n  voice_segment_minutes: 0.02\n")
    result = run_stage_process("sync", project_path, settings, "complex-fp")
    project = Project.load(project_path)
    voice = project.outputs["sync:cam"][0]
    assert voice == work / "stages" / "sync" / "voice" / "cam.wav"
    assert voice.is_file()
    assert project.audio_warp_maps[0].strategy == 3
    assert [placement.asset_id for placement in project.placements] == (
        ["rec", "cam"] if recorder_count == 1 else ["rec", "rec2", "cam"]
    )
    assert len(project.outputs["sync-tracks:cam"]) == recorder_count
    assert len(project.audio_warp_maps) == recorder_count
    assert project.outputs["master_wav"][0].is_file()
    assert len(project.outputs["voice-segments:cam"]) == 4
    assert result.manifest.reusable(work, "complex-fp")
    assert not list(work.glob(".settings-*"))
    from studio.stages.export import build_sequence

    sequence = build_sequence(project, use_edit=False, retimed=True)
    assert len([clip for clip in sequence.clips if not clip.has_video]) == recorder_count


def test_repair_updates_selected_lane_map_report_and_segment_input(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    source = tmp_path / "recorder.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x10\x00" * 48000 * 4)
    words = tuple(Word(f"token{i}", 0.2 + i * 0.1, 0.25 + i * 0.1) for i in range(25))
    transcripts = {}
    for name, duration in [("rec", 4.0), ("cam", 3.0)]:
        path = work / f"{name}.json"
        Transcript(source, "en", duration, [Segment(0, 3, words)]).save(path)
        transcripts[name] = path
    project = Project(tmp_path, work, assets=[
        Asset("rec", Path("recorder.wav"), AssetKind.AUDIO, AssetRole.RECORDER),
        Asset("cam", Path("camera.mov"), AssetKind.VIDEO, AssetRole.CAMERA),
    ], transcripts=transcripts)
    repair_path = work / "stages" / "sync" / "repair" / "cam-repaired.wav"
    repair_path.parent.mkdir(parents=True)
    repair_path.write_bytes(source.read_bytes())
    repair_report = repair_path.with_name("report.json")
    repair_report.write_text("{}")
    check_report = work / "check.json"
    check_report.write_text("{}")
    repaired_map = AudioWarpMap("warp-rec-cam", "rec", "cam", (
        AudioWarpPiece(0.1, 3, 0, 3, "copy"),
    ), TimeDomain.FILE, 1, {"repair_verified": 1.0})
    with patch("studio.stages.sync_check_step.check_voices", return_value=(check_report,)), patch(
        "studio.stages.sync_repair.repair_voices",
        return_value=({"cam": repair_path}, {"cam": repaired_map},
                      (repair_path, repair_report)),
    ) as repair, patch("studio.stages.sync_outputs.segment_voice", return_value=()) as segment:
        result = _run_complex(StageContext(project, {"sync": {
            "mode": "complex", "recorder_mode": "all", "self_check": "repair",
            "voice_segment_minutes": 1,
        }}, work))
    assert repair.call_args.args[1]["cam"] == tmp_path / "camera.mov"
    assert repair.call_args.args[2]["cam"][0] == "rec"
    segment.assert_called_once()
    assert segment.call_args.args[0] == repair_path
    changes = result.project_changes
    assert changes["outputs"]["sync:cam"] == [repair_path]
    assert changes["outputs"]["sync-tracks:cam"] == [repair_path]
    assert changes["audio_warp_maps"] == [repaired_map]
    published = json.loads((work / "stages" / "sync" / "output.json").read_text())
    assert published["voices"]["cam"] == "repair/cam-repaired.wav"


def test_disjoint_sessions_render_only_matching_recorders(tmp_path: Path) -> None:
    from studio.stages.sync_models import AlignmentMap

    work = tmp_path / "work"
    work.mkdir()
    assets = []
    paths = {}
    for index in range(2):
        source = tmp_path / f"rec{index}.wav"
        with wave.open(str(source), "wb") as audio:
            audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            audio.writeframes((b"\x10\x00" if index == 0 else b"\x20\x00") * 48000 * 4)
        for prefix, kind, role in (("rec", AssetKind.AUDIO, AssetRole.RECORDER),
                                   ("cam", AssetKind.VIDEO, AssetRole.CAMERA)):
            name = f"{prefix}{index}"
            assets.append(Asset(name, source.name, kind, role))
            path = work / f"{name}.json"
            Transcript(tmp_path / f"{name}.wav", "en", 3, []).save(path)
            paths[name] = path
    project = Project(tmp_path, work, assets=assets, transcripts=paths)

    def fitted(target, source, *_args, **_kwargs):
        if target.source_audio.stem[-1] != source.source_audio.stem[-1]:
            raise ValueError("unrelated session")
        return AlignmentMap([], 0, 1, 0, "acoustic", 5, 3)

    with patch("studio.stages.sync_sessions.align_sources", side_effect=fitted):
        result = _run_complex(StageContext(project, {"sync": {
            "mode": "simple", "recorder_mode": "all", "session_offsets": {"cam1": 100},
        }}, work))
    changes = result.project_changes
    pairs = {(warp.source_asset_id, warp.target_asset_id) for warp in changes["audio_warp_maps"]}
    assert pairs == {
        ("rec0", "cam0"), ("rec1", "cam1"),
    }
    assert {p.asset_id: p.offset_s for p in changes["placements"]} == {
        "rec0": 0, "cam0": 0, "rec1": 100, "cam1": 100,
    }
    for camera in ("cam0", "cam1"):
        assert len(changes["outputs"][f"sync-tracks:{camera}"]) == 1
        assert changes["outputs"][f"sync:{camera}"][0].is_file()
