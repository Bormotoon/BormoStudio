import wave
from pathlib import Path

import pytest

from studio.core.av_calibration import calibration_view, picture_placement, picture_shifts
from studio.core.project import Asset, AssetRole, Project
from studio.core.settings import SettingsError, fingerprint_view, load_settings
from studio.core.timeline import EditMap, KeepRange, SourcePlacement
from studio.stages.export import build_sequence
from studio.stages.program import program_pieces


def project(tmp_path: Path) -> Project:
    result = Project(tmp_path, tmp_path / "_studio")
    result.assets = [
        Asset("a", Path("CamA/C0001.MP4"), "video", AssetRole.CAMERA, group_id="cam-a"),
        Asset("b", Path("CamB/C0002.MP4"), "video", AssetRole.CAMERA, group_id="cam-b"),
        Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER),
    ]
    result.placements = [SourcePlacement("a", 0.0, 0.0, 10.0),
                         SourcePlacement("b", 2.0, 1.0, 8.0, k=1.0005),
                         SourcePlacement("rec", 0.0, 0.0, 12.0)]
    return result


def sync(default: float = 0.0, **by_camera: float) -> dict:
    return {"sync": {"camera_av_offset_ms": default, "camera_av_offset_ms_by_camera": by_camera}}


def test_nothing_calibrated_changes_nothing(tmp_path: Path) -> None:
    assert calibration_view({}) == {} and calibration_view(sync()) == {}
    assert picture_shifts(project(tmp_path), sync()) == {}


def test_per_camera_value_wins_by_id_group_folder_or_file(tmp_path: Path) -> None:
    source = project(tmp_path)
    assert picture_shifts(source, sync(40)) == {"a": 0.04, "b": 0.04}
    for name in ("b", "cam-b", "CamB", "C0002.MP4", "C0002"):
        assert picture_shifts(source, sync(40, **{name: -25})) == {"a": 0.04, "b": -0.025}
    # A camera calibrated to zero is not moved even when the default is set.
    assert picture_shifts(source, sync(40, a=0)) == {"b": 0.04}


def test_picture_moves_earlier_and_loses_frames_before_the_origin() -> None:
    later = picture_placement(SourcePlacement("b", 2.0, 1.0, 8.0, k=1.0005), -0.05)
    assert (later.offset_s, later.in_s, later.duration_s) == pytest.approx((2.05, 1.0, 8.0))
    earlier = picture_placement(SourcePlacement("a", 0.02, 0.0, 10.0, k=2.0), 0.1)
    # 0.08 s of timeline before zero are 0.04 s of a file running at k=2.
    assert (earlier.offset_s, earlier.in_s, earlier.duration_s) == pytest.approx(
        (0.0, 0.04, 9.96))
    with pytest.raises(ValueError, match="no picture"):
        picture_placement(SourcePlacement("a", 0.0, 0.0, 0.05), 0.1)


def test_export_moves_only_the_camera_picture(tmp_path: Path) -> None:
    source = project(tmp_path)
    for name in ("voice-a.wav", "rec.wav"):
        with wave.open(str(tmp_path / name), "wb") as audio:
            audio.setparams((1, 2, 48000, 0, "NONE", "PCM"))
            audio.writeframes(b"\x00\x00" * 48000 * 12)
    source.outputs["sync:a"] = [tmp_path / "voice-a.wav"]
    plain = {clip.asset_id: clip for clip in build_sequence(source).clips}
    moved = {clip.asset_id: clip for clip in build_sequence(
        source, picture_shift=picture_shifts(source, sync(0, a=-40, b=30))).clips}
    assert moved["a"].timeline_start_s == pytest.approx(0.04)
    assert moved["b"].timeline_start_s == pytest.approx(1.97)
    assert moved["b"].source_in_s == plain["b"].source_in_s
    for clip_id in ("voice-a", "rec"):
        assert moved[clip_id] == plain[clip_id]


def test_program_reads_picture_and_sound_by_their_own_placements(tmp_path: Path) -> None:
    source = project(tmp_path)
    edit = EditMap("cut", (KeepRange(1.0, 1.5, "a"),))
    plain, = program_pieces(source, edit)
    moved, = program_pieces(source, edit, {"a": 0.04})
    assert plain.video_in_s == plain.audio_in_s == pytest.approx(1.0)
    # The picture is 40 ms earlier, so the frame shown at timeline 1 s is file 1.04 s.
    assert moved.video_in_s == pytest.approx(1.04)
    assert moved.audio_in_s == pytest.approx(1.0)


def test_program_covers_only_where_picture_and_sound_both_exist(tmp_path: Path) -> None:
    source = project(tmp_path)
    with pytest.raises(ValueError, match="no camera covers"):
        program_pieces(source, EditMap("cut", (KeepRange(9.98, 10.0, "a"),)), {"a": 0.04})


def test_calibration_never_resynchronizes_and_ducking_counts_only_when_on() -> None:
    view = fingerprint_view("sync", {"mode": "auto", "camera_av_offset_ms": 40,
                                     "camera_av_offset_ms_by_camera": {"a": 5},
                                     "pause_duck_db": -30.0})
    assert view == {"mode": "auto"}
    on = fingerprint_view("sync", {"pause_duck_enabled": True, "pause_duck_db": -30.0})
    assert on == {"pause_duck_enabled": True, "pause_duck_db": -30.0}


def test_settings_reject_absurd_calibration_and_positive_ducking(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    for text in ("sync:\n  camera_av_offset_ms: 5000\n",
                 "sync:\n  camera_av_offset_ms_by_camera: {' ': 10}\n",
                 "sync:\n  pause_duck_db: 3\n"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(SettingsError):
            load_settings([path])
    path.write_text("sync:\n  camera_av_offset_ms_by_camera: {CamA: -33.3}\n"
                    "  pause_duck_enabled: true\n", encoding="utf-8")
    loaded = load_settings([path])
    assert loaded.sync.camera_av_offset_ms_by_camera == {"CamA": -33.3}
    assert loaded.sync.pause_duck_enabled is True
