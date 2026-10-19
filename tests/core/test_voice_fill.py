import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from studio.core import voice_fill
from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.core.transcript import Segment, Transcript, Word
from studio.core.voice_fill import (
    fill_voice,
    level_gain_db,
    merge,
    playback_voice,
    subtract,
    unheard_spans,
    zero_runs,
)
from studio.stages.base import StageContext
from studio.stages.roughcut import _camera_fill_silence
from studio.stages.timeline import TimelineStage

RATE = 8000


def _write(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    data = np.asarray(samples, dtype=np.float64)
    data = data[:, None] if data.ndim == 1 else data
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _read(path: Path) -> np.ndarray:
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-"],
                            capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype="<f4")


def _speech(seconds: float, level: float, seed: int = 0) -> np.ndarray:
    """Noise bursts, always audible, so every level window counts."""
    rng = np.random.default_rng(seed)
    count = round(seconds * RATE)
    envelope = 0.6 + 0.4 * np.sin(np.arange(count) * 2 * np.pi / RATE * 3)
    return level * envelope * rng.uniform(-1, 1, count)


def test_zero_runs_span_chunk_boundaries_and_skip_short_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_fill, "CHUNK_FRAMES", 1000)
    sound = _speech(10, 0.3)
    sound[2 * RATE:5 * RATE] = 0  # crosses many chunks
    sound[7 * RATE:7 * RATE + RATE // 2] = 0  # half a second: kept as it is
    sound[-2 * RATE:] = 0  # runs to the end of the file
    path = _write(tmp_path / "voice.wav", np.stack([sound, sound], axis=1))
    assert zero_runs(path, min_s=1.0) == [(2 * RATE, 5 * RATE), (8 * RATE, 10 * RATE)]


def test_zero_runs_join_across_a_stray_sample(tmp_path: Path) -> None:
    sound = _speech(6, 0.3)
    sound[RATE:4 * RATE] = 0
    sound[2 * RATE] = 0.5  # one sample between two dead stretches
    path = _write(tmp_path / "voice.wav", sound)
    assert zero_runs(path) == [(RATE, 4 * RATE)]


def test_zero_runs_need_every_channel_silent(tmp_path: Path) -> None:
    left = _speech(4, 0.3)
    right = left.copy()
    left[RATE:3 * RATE] = 0
    path = _write(tmp_path / "voice.wav", np.stack([left, right], axis=1))
    assert zero_runs(path) == []


def test_level_gain_matches_the_camera_to_the_voice(tmp_path: Path) -> None:
    sound = _speech(30, 0.5)
    voice = sound.copy()
    voice[10 * RATE:20 * RATE] = 0
    voice_path = _write(tmp_path / "voice.wav", voice)
    camera = _write(tmp_path / "camera.wav", sound * 0.1)
    gain = level_gain_db(voice_path, camera, [(10.0, 20.0)])
    assert gain == pytest.approx(20.0, abs=0.5)


def test_level_gain_needs_enough_shared_windows(tmp_path: Path) -> None:
    voice = np.zeros(30 * RATE)
    voice[:5 * RATE] = _speech(5, 0.5)
    voice_path = _write(tmp_path / "voice.wav", voice)
    camera = _write(tmp_path / "camera.wav", _speech(30, 0.05))
    assert level_gain_db(voice_path, camera, [(5.0, 30.0)]) is None


def test_fill_voice_takes_the_camera_only_inside_the_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_fill, "CHUNK_FRAMES", 777)
    sound = _speech(6, 0.4)
    voice = sound.copy()
    voice[2 * RATE:4 * RATE] = 0
    voice_path = _write(tmp_path / "voice.wav", np.stack([voice, voice], axis=1))
    camera = _write(tmp_path / "camera.wav", sound * 0.1)
    output = fill_voice(voice_path, camera, tmp_path / "out" / "voice.wav",
                        [(2 * RATE, 4 * RATE)], gain_db=20.0)
    filled = _read(output).reshape(-1, 2)
    original = _read(voice_path).reshape(-1, 2)
    assert len(filled) == len(original)
    np.testing.assert_array_equal(filled[:2 * RATE], original[:2 * RATE])
    np.testing.assert_array_equal(filled[4 * RATE:], original[4 * RATE:])
    middle = slice(2 * RATE + RATE // 10, 4 * RATE - RATE // 10)
    np.testing.assert_allclose(filled[middle, 0], sound[middle], atol=0.01)
    np.testing.assert_array_equal(filled[:, 0], filled[:, 1])
    # The camera fades in rather than starting at full level.
    assert abs(filled[2 * RATE, 0]) < 0.02


def test_spans_merge_and_subtract() -> None:
    assert merge([(5, 6), (0, 2), (1, 3)]) == [(0, 3), (5, 6)]
    assert subtract([(0, 10)], [(2, 3), (8, 12)]) == [(0, 2), (3, 8)]
    assert subtract([(0, 1)], []) == [(0, 1)]


def _project(tmp_path: Path) -> Project:
    """A recorder dead from 10 s to 30 s; a camera covering 5-35 s of the timeline."""
    work = tmp_path / "_studio"
    talk = _speech(40, 0.4, seed=1)
    recorder = talk.copy()
    recorder[10 * RATE:30 * RATE] = 0
    _write(tmp_path / "rec.wav", recorder)
    _write(tmp_path / "cam.wav", talk[5 * RATE:35 * RATE] * 0.2)
    voice = recorder[5 * RATE:35 * RATE]
    voice_path = _write(work / "stages" / "sync" / "voice" / "cam.wav", voice)
    rec_words = Transcript(tmp_path / "rec.wav", "ru", 40, [
        Segment(2, 3, (Word("до", 2, 3),)), Segment(32, 33, (Word("после", 32, 33),))])
    cam_words = Transcript(tmp_path / "cam.wav", "ru", 30, [
        Segment(4, 6, (Word("камера", 4, 5), Word("слышит", 5, 6))),
        Segment(26.5, 28, (Word("рекордер", 26.5, 27), Word("снова", 31, 32))),
    ])
    rec_path = work / "stages" / "transcribe" / "rec.json"
    cam_path = work / "stages" / "transcribe" / "cam.json"
    rec_words.save(rec_path)
    cam_words.save(cam_path)
    return Project(
        tmp_path, work,
        assets=[Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER),
                Asset("cam", Path("cam.wav"), AssetKind.VIDEO, AssetRole.CAMERA,
                      manual={"media_info": {"audio_codec": "pcm_s16le"}})],
        placements=[SourcePlacement("rec", 0, 0, 40), SourcePlacement("cam", 5, 0, 30)],
        transcripts={"rec": rec_path, "cam": cam_path},
        outputs={"sync:cam": [voice_path]},
    )


def test_unheard_spans_are_camera_time_without_live_recorder(tmp_path: Path) -> None:
    project = _project(tmp_path)
    assert unheard_spans(project) == [(10.0, 30.0)]


def test_timeline_takes_camera_words_and_sound_where_the_recorder_is_dead(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    result = TimelineStage().run(StageContext(project, {}, project.work_dir))
    timeline = Transcript.load(result.artifacts[0])
    # Camera words land at camera time + 5 s; only "слышит" (10-11 s) falls where
    # the recorder is dead, the others are where it hears (or would duplicate it).
    assert [(word.text, word.start) for word in timeline.words] == [
        ("до", 2), ("слышит", 10), ("после", 32)]
    filled = timeline.metadata["camera_fill"]
    assert [(item["start"], item["end"], item["asset_id"]) for item in filled] == [
        (10.0, 30.0, "cam")]
    assert filled[0]["gain_db"] == pytest.approx(14.0, abs=0.5)
    voice = result.project_changes["outputs"]["voice:cam"][0]
    assert voice.is_file()
    assert result.warnings and "no recorder heard 20.0 s" in result.warnings[0]
    project.outputs = result.project_changes["outputs"]
    assert playback_voice(project, "cam") == voice
    samples = _read(voice)
    camera = _read(tmp_path / "cam.wav")
    middle = slice(10 * RATE, 20 * RATE)  # camera time 10-20 s = timeline 15-25 s
    assert np.sqrt(np.mean(samples[middle] ** 2)) == pytest.approx(
        np.sqrt(np.mean(camera[middle] ** 2)) * 10 ** (14 / 20), rel=0.15)


def test_timeline_leaves_sound_and_words_alone_when_filling_is_off(tmp_path: Path) -> None:
    project = _project(tmp_path)
    settings = {"sync": {"recorder_fill": "silence"}}
    result = TimelineStage().run(StageContext(project, settings, project.work_dir))
    timeline = Transcript.load(result.artifacts[0])
    assert [word.text for word in timeline.words] == ["до", "после"]
    assert "camera_fill" not in timeline.metadata
    assert not any(key.startswith("voice:") for key in result.project_changes["outputs"])


def test_stale_playback_voices_are_dropped(tmp_path: Path) -> None:
    project = _project(tmp_path)
    project.outputs["voice:old"] = [tmp_path / "gone.wav"]
    result = TimelineStage().run(StageContext(project, {}, project.work_dir))
    assert "voice:old" not in result.project_changes["outputs"]


def test_roughcut_reads_silence_from_the_camera_where_it_fills(tmp_path: Path) -> None:
    project = _project(tmp_path)
    quiet = np.concatenate([_speech(10, 0.2), np.zeros(10 * RATE), _speech(10, 0.2)])
    _write(tmp_path / "cam.wav", quiet)
    transcript = Transcript(tmp_path / "rec.wav", "ru", 40, [], metadata={
        "camera_fill": [{"start": 10.0, "end": 30.0, "asset_id": "cam", "gain_db": 14.0}]})
    from studio.core.audio_spans import Span

    silence = _camera_fill_silence(project, transcript, [Span(0, 1), Span(9, 31)], -40.0)
    spans = [(round(span.start_s, 1), round(span.end_s, 1)) for span in silence]
    # The recorder's zeros inside 10-30 s are replaced by what the camera heard.
    assert spans == [(0, 1), (9, 10), (15.0, 25.0), (30, 31)]


def test_playback_voice_falls_back_to_the_sync_voice(tmp_path: Path) -> None:
    project = _project(tmp_path)
    assert playback_voice(project, "cam") == project.outputs["sync:cam"][0]
    project.outputs["voice:cam"] = [tmp_path / "missing.wav"]
    assert playback_voice(project, "cam") == project.outputs["sync:cam"][0]
    assert playback_voice(project, "other") is None


def test_dead_air_is_recorder_time_with_neither_sound_nor_camera(tmp_path: Path) -> None:
    project = _project(tmp_path)
    project.placements = [SourcePlacement("rec", 0, 0, 40), SourcePlacement("cam", 15, 0, 20)]
    sound = voice_fill.recorder_sound(project)
    assert sound is not None
    assert sound.dead_air() == [(10.0, 15.0)]
    assert sound.unheard() == [(15.0, 30.0)]
    result = TimelineStage().run(StageContext(project, {}, project.work_dir))
    assert Transcript.load(result.artifacts[0]).metadata["dead_air"] == [[10.0, 15.0]]


def test_the_rough_cut_removes_dead_air_even_inside_a_noisy_gap() -> None:
    from studio.core.audio_spans import Span
    from studio.stages.roughcut import build_edit_list

    transcript = Transcript(Path("rec.wav"), "ru", 200, [
        Segment(27, 28, (Word("включил?", 27, 28),)),
        Segment(171, 172, (Word("Ну,", 171, 172),)),
    ])
    # Sounds at 30-33 s keep the 27-171 s gap from being one quiet pause.
    silence = [Span(28, 30), Span(33, 171)]
    options = dict(mode="cut", minimum_pause_s=1.0, keep_pause_s=0.4, edge_pad_s=0.2,
                   silence=silence)
    plain = build_edit_list(transcript, **options)
    assert not [cut for cut in plain.cuts if cut.reason == "pause"]
    edit = build_edit_list(transcript, dead_air=[Span(34, 170.6)], **options)
    cuts = [(round(cut.start, 1), round(cut.end, 1)) for cut in edit.cuts if cut.reason == "pause"]
    assert cuts == [(34.2, 170.4)]
    assert [cut.note for cut in edit.cuts if cut.reason == "pause"] == [
        "nothing was recorded 136.20s"]
