import json
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.timeline import SourcePlacement, TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.speaker_backend import Diarization
from studio.stages.speaker_identity import VoiceLabel, match_labels, voice_signature
from studio.stages.speakers import SpeakersStage, diarization_sources, run_pyannote

RATE = 16000


def _voice(f0: float, formants: list[float], seconds: float, seed: int) -> np.ndarray:
    """A buzzy vowel-like voice with its own pitch and formants, in syllables."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds)) / RATE
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t + seed))) / RATE
    source = sum(np.sin(k * phase) / k for k in range(1, 30))
    spectrum = np.fft.rfft(source)
    frequencies = np.fft.rfftfreq(len(source), 1 / RATE)
    envelope = sum(np.exp(-(((frequencies - f) / 120) ** 2)) for f in formants) + 0.02
    signal = np.fft.irfft(spectrum * envelope, len(source)) * (np.sin(2 * np.pi * 1.5 * t) > -0.3)
    return signal / np.abs(signal).max() * 0.3 + rng.normal(0, 0.003, len(signal))


LOW = ([110], [700, 1200, 2500])
HIGH = ([210], [400, 2000, 2900])


def _label(group: str, label: str, voice: tuple, seed: int, **extra: object) -> VoiceLabel:
    signature = voice_signature(_voice(voice[0][0], voice[1], 4, seed))
    assert signature is not None
    return VoiceLabel(group, label, signature, **extra)  # type: ignore[arg-type]


def test_labels_match_across_files_but_never_within_one() -> None:
    labels = [
        _label("a", "a1", LOW, 1, first_s=0), _label("a", "a2", HIGH, 2, first_s=1),
        _label("b", "b1", HIGH, 3, first_s=10), _label("b", "b2", LOW, 4, first_s=11),
    ]
    match = match_labels(labels, threshold=0.5)
    assert match.names == {("a", "a1"): "a1", ("a", "a2"): "a2",
                           ("b", "b1"): "a2", ("b", "b2"): "a1"}
    # Two lanes of one file are different people even when they sound alike.
    same = match_labels([_label("a", "x", LOW, 1), _label("a", "y", LOW, 2)], threshold=0)
    assert set(same.names.values()) == {"x", "y"}


def test_simultaneous_files_and_conflicting_names_stay_apart() -> None:
    labels = [_label("a", "a1", LOW, 1), _label("b", "b1", LOW, 2)]
    assert match_labels(labels, threshold=0.5).names[("b", "b1")] == "a1"
    apart = match_labels(labels, threshold=0.5, simultaneous={frozenset(("a", "b"))})
    assert apart.names[("b", "b1")] == "b1"
    named = [_label("a", "a1", LOW, 1, fixed_name="Ivan"),
             _label("b", "b1", LOW, 2, fixed_name="Petr"),
             _label("c", "c1", LOW, 3, first_s=-1)]
    names = match_labels(named, threshold=0.5).names
    assert names[("a", "a1")] == "Ivan" and names[("b", "b1")] == "Petr"
    assert names[("c", "c1")] in {"Ivan", "Petr"}  # an unnamed lane inherits a name
    with pytest.raises(ValueError):
        match_labels(labels, threshold=2)


def _wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def test_two_sessions_of_lavaliers_keep_one_name_per_person(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "_studio")
    silence = np.zeros(RATE * 3)
    low, high = _voice(110, LOW[1], 3, 1), _voice(210, HIGH[1], 3, 2)
    # Session 1 at 0-6 s: lane A1 is the low voice, A2 the high one.
    # Session 2 at 20-26 s on other recorders, plugged the other way round.
    files = {
        "a1": np.concatenate((low, silence)), "a2": np.concatenate((silence, high)),
        "b1": np.concatenate((_voice(208, HIGH[1], 3, 3), silence)),
        "b2": np.concatenate((silence, _voice(112, LOW[1], 3, 4))),
    }
    for name, samples in files.items():
        _wav(tmp_path / f"{name}.wav", samples)
        asset = Asset(name, Path(f"{name}.wav"), AssetKind.AUDIO, AssetRole.RECORDER)
        asset.manual["media_info"] = {"audio_channels": 1}
        project.assets.append(asset)
        project.placements.append(
            SourcePlacement(name, 0 if name.startswith("a") else 20, 0, 6))
    words = (Word("low", 0.5, 2.5), Word("high", 3.5, 5.5),
             Word("high", 20.5, 22.5), Word("low", 23.5, 25.5))
    transcript = Transcript(tmp_path / "a1.wav", "en", 26, [Segment(0, 26, words)],
                            time_domain=TimeDomain.TIMELINE)
    project.transcripts["timeline"] = project.work_dir / "timeline.json"
    transcript.save(project.transcripts["timeline"])
    settings = {"speakers": {"method": "mics"}}
    output = SpeakersStage().run(StageContext(project, settings, tmp_path / "out"))
    turns = json.loads(output.artifacts[0].read_text())
    assert [turn["speaker"] for turn in turns] == ["a1-ch1", "a2-ch1", "a2-ch1", "a1-ch1"]
    report = json.loads(output.artifacts[1].read_text())
    assert report["identities"]["renamed"] == {"b1:0": "a2-ch1", "b2:0": "a1-ch1"}
    unmatched = SpeakersStage().run(StageContext(
        project, {"speakers": {"method": "mics", "match_files": False}}, tmp_path / "plain"))
    plain = json.loads(unmatched.artifacts[0].read_text())
    assert [turn["speaker"] for turn in plain] == ["a1-ch1", "a2-ch1", "b1-ch1", "b2-ch1"]


def test_pyannote_diarizes_each_session_and_unifies_labels(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "work")
    project.assets = [Asset("rec1", Path("rec1.wav"), AssetKind.AUDIO, AssetRole.RECORDER),
                      Asset("rec2", Path("rec2.wav"), AssetKind.AUDIO, AssetRole.RECORDER),
                      Asset("cam", Path("cam.mp4"), AssetKind.VIDEO, AssetRole.CAMERA)]
    project.placements = [SourcePlacement("rec1", 0, 0, 30), SourcePlacement("rec2", 40, 0, 30),
                          SourcePlacement("cam", 0, 0, 30)]  # a second copy of session 1
    transcript = Transcript(tmp_path / "rec1.wav", "en", 70, [],
                            metadata={"source_asset_id": "rec1"})
    assert [item[0] for item in diarization_sources(project, transcript)] == ["rec1", "rec2"]
    results = {
        "rec1.wav": Diarization(
            [{"start": 1, "end": 5, "speaker": "SPEAKER_00"},
             {"start": 6, "end": 9, "speaker": "SPEAKER_01"}],
            {"SPEAKER_00": [1.0, 0.0, 0.1], "SPEAKER_01": [0.0, 1.0, 0.1]}),
        "rec2.wav": Diarization(
            [{"start": 1, "end": 4, "speaker": "SPEAKER_00"},
             {"start": 5, "end": 8, "speaker": "SPEAKER_01"},
             {"start": 9, "end": 12, "speaker": "SPEAKER_02"}],
            {"SPEAKER_00": [0.1, 0.9, 0.0], "SPEAKER_01": [0.9, 0.1, 0.0],
             "SPEAKER_02": [0.0, 0.0, 1.0]}),
    }
    with patch("studio.stages.speaker_backend.diarize",
               side_effect=lambda path, *_a, **_k: results[path.name]):
        output = run_pyannote(StageContext(project, {}, project.work_dir), transcript, {})
    turns = json.loads(output.project_changes["outputs"]["speakers"][0].read_text())
    assert [(t["start"], t["speaker"]) for t in turns] == [
        (1, "SPEAKER_00"), (6, "SPEAKER_01"),
        (41, "SPEAKER_01"), (45, "SPEAKER_00"), (49, "SPEAKER_02"),
    ]
    report = json.loads(output.project_changes["outputs"]["speakers"][1].read_text())
    assert report["identities"]["method"] == "pyannote-embedding"
    assert report["identities"]["sources"] == ["rec1", "rec2"]


def test_named_first_session_keeps_unnamed_second_session_lanes(tmp_path: Path) -> None:
    from studio.core.project import RecorderTrack
    from studio.stages.speakers import microphone_tracks

    project = Project(tmp_path, tmp_path / "_studio")
    for name, offset in (("a1", 0), ("a2", 0), ("b1", 20), ("b2", 20), ("backup", 0)):
        asset = Asset(name, Path(f"{name}.wav"), AssetKind.AUDIO, AssetRole.RECORDER)
        asset.manual["media_info"] = {"audio_channels": 1}
        project.assets.append(asset)
        project.placements.append(SourcePlacement(name, offset, 0, 6))
    project.recorder_tracks = [RecorderTrack("a1", (0,), speaker="Ivan"),
                               RecorderTrack("a2", (0,), speaker="Maria")]
    tracks = microphone_tracks(project, {"method": "mics"})
    # The backup recorder plays alongside the named session and stays out.
    assert tracks == {"a1:0": "Ivan", "a2:0": "Maria", "b1:0": "b1-ch1", "b2:0": "b2-ch1"}
    assert microphone_tracks(project, {"method": "mics", "match_files": False}) == {
        "a1:0": "Ivan", "a2:0": "Maria"}
