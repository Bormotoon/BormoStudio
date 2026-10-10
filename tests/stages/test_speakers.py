import json
import wave
from pathlib import Path

import numpy as np
import pytest

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.timeline import SourcePlacement, TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.runner import run_stage_process
from studio.stages.speakers import (
    SpeakersStage,
    assign_words_to_mics,
    microphone_tracks,
    timeline_envelope,
)


def test_mic_assignment_handles_bleed_and_overlap() -> None:
    words = [Word("one", 0, 0.1), Word("both", 0.1, 0.2), Word("two", 0.2, 0.3)]
    envelopes = {
        "S1": np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1]),
        "S2": np.array([0.1, 0.1, 1.0, 1.0, 1.0, 1.0]),
    }
    turns = assign_words_to_mics(words, envelopes)
    assert [turn.speaker for turn in turns] == ["S1", "overlap", "S2"]


def test_silent_tracks_do_not_report_overlap() -> None:
    turns = assign_words_to_mics([Word("noise", 0, 0.1)],
                                {"S1": np.zeros(3), "S2": np.zeros(3)})
    assert turns[0].speaker == "unknown"


def test_energy_bins_respect_placement_inpoint_and_clock() -> None:
    envelope = timeline_envelope(
        np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=float),
        SourcePlacement("rec", 1, 1, 2, 2), 6, step_s=1, sample_rate=2,
    )
    assert envelope.tolist() == [0, 1, 1, 1, 1, 0]


def test_worker_stage_decodes_stereo_mics_with_bleed_and_offset(tmp_path: Path) -> None:
    rate = 16000
    time = np.arange(rate * 2) / rate
    tone = np.sin(2 * np.pi * 440 * time) * 0.5
    channels = np.column_stack((tone * np.where(time < 1, 1, 0.1),
                                tone * np.where(time < 1, 0.1, 1)))
    source = tmp_path / "rec.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes((channels * 32767).astype("<i2").tobytes())
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER)]
    project.assets[0].manual["media_info"] = {"audio_channels": 2}
    project.placements = [SourcePlacement("rec", 3, 0, 2)]
    transcript = Transcript(source, "en", 5, [Segment(3, 5, (
        Word("first", 3.2, 3.6), Word("second", 4.2, 4.6),
    ))], time_domain=TimeDomain.TIMELINE)
    path = project.work_dir / "timeline.json"
    transcript.save(path)
    project.transcripts["timeline"] = path
    settings = {"speakers": {"method": "mics", "tracks": {"rec:0": "Alice", "rec:1": "Bob"}}}
    stage = SpeakersStage()
    assert stage.decide(project, settings).kind.value == "run"
    output = stage.run(StageContext(project, settings, tmp_path / "output"))
    turns = json.loads(output.artifacts[0].read_text())
    assert [turn["speaker"] for turn in turns] == ["Alice", "Bob"]
    assert [turn["start"] for turn in turns] == [3.2, 4.2]
    assert stage.decide(project, {}).kind.value == "skip"  # stereo is not two lavaliers
    project_path = project.work_dir / "project.json"
    project.save(project_path)
    settings_path = project.work_dir / "settings.yaml"
    settings_path.write_text("{}", encoding="utf-8")
    fingerprint = stage.fingerprint(project, settings)
    result = run_stage_process(
        "speakers", project_path, settings_path, fingerprint, effective_settings=settings,
    )
    assert result.manifest.reusable(project.work_dir, fingerprint)
    published = Project.load(project_path)
    assert json.loads(published.outputs["speakers"][0].read_text()) == turns
    assert stage.fingerprint(published, settings) == fingerprint
    with pytest.raises(ValueError, match="out of range"):
        microphone_tracks(project, {"tracks": {"rec:2": "Bad"}})


def test_piecewise_warp_maps_mic_energy_where_a_linear_fit_would_drift() -> None:
    from studio.core.timeline import AudioWarpMap, AudioWarpPiece
    from studio.stages.speakers import recorder_warp, warped_source_times

    # Recorder seconds 0-4 play at camera 0-4, its seconds 4-8 are squeezed
    # into camera 4-6 (a drift burst), and 8-10 play at camera 6-8.
    warp = AudioWarpMap("w", "rec", "cam", (
        AudioWarpPiece(0, 4, 0, 4, "copy"), AudioWarpPiece(4, 4, 4, 2, "atempo"),
        AudioWarpPiece(8, 2, 6, 2, "copy"),
    ), TimeDomain.FILE, 3)
    camera = SourcePlacement("cam", 0, 0, 8)
    times = warped_source_times(np.array([1.0, 5.0, 9.0]), warp, camera)
    assert times[:2].tolist() == [1.0, 6.0] and np.isnan(times[2])
    signal = np.zeros(10)
    signal[7] = 1.0  # speech only at recorder second 7
    linear = timeline_envelope(signal, SourcePlacement("rec", 0, 0, 10), 8, step_s=1,
                               sample_rate=1)
    warped = timeline_envelope(signal, SourcePlacement("rec", 0, 0, 10), 8, step_s=1,
                               sample_rate=1, warp=(warp, camera))
    assert linear.argmax() == 7 and warped.argmax() == 5
    project = Project(Path("/m"), Path("/w"))
    project.placements = [camera]
    project.audio_warp_maps = [warp]
    assert recorder_warp(project, "rec") == (warp, camera)
    assert recorder_warp(project, "other") is None
