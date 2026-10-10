import wave
from pathlib import Path

import numpy as np
import pytest

from studio.core.channels import analyze_channels, classify, describe, effective_audio
from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.recorder_tracks import default_tracks, refresh_tracks, set_track

RATE = 8000


def _voice(seconds: float = 5.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    envelope = (np.sin(np.linspace(0, 40, int(seconds * RATE))) > 0) * 0.1 + 0.01
    return (rng.standard_normal(int(seconds * RATE)) * envelope).astype(np.float32)


def _excerpts(right: callable) -> list[np.ndarray]:
    result = []
    for seed in range(4):
        left = _voice(seed=seed)
        result.append(np.stack([left, right(left, seed)], axis=1))
    return result


def test_identical_channels_are_dual_mono() -> None:
    analysis = classify(_excerpts(lambda left, _seed: left.copy()), 2)
    assert analysis["class"] == "dual_mono"
    assert analysis["groups"] == [{"channels": [0, 1], "relation": "dual_mono", "primary": 0}]


def test_safety_copy_is_linked_and_the_loud_channel_is_primary() -> None:
    analysis = classify(_excerpts(lambda left, _seed: left * 10 ** (-12 / 20)), 2)
    assert analysis["class"] == "linked"
    assert analysis["groups"][0]["primary"] == 0
    assert analysis["pairs"][0]["level_difference_db"] == pytest.approx(12, abs=0.1)
    assert "ch1/ch2" in describe(analysis)


def test_independent_microphones_stay_two_lanes() -> None:
    analysis = classify(_excerpts(lambda _left, seed: _voice(seed=seed + 10)), 2)
    assert analysis["class"] == "independent"
    assert len(analysis["groups"]) == 2


def test_silent_right_channel_leaves_one_active_lane() -> None:
    analysis = classify(_excerpts(lambda left, _seed: np.zeros_like(left)), 2)
    assert analysis["class"] == "one_silent"
    assert analysis["silent"] == [1]


def _write(path: Path, data: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())


def _recorder(tmp_path: Path, data: np.ndarray) -> Asset:
    path = tmp_path / "rec.wav"
    _write(path, data)
    analysis = analyze_channels(path, channels=data.shape[1], duration_s=len(data) / RATE)
    return Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER, manual={
        "media_info": {"audio_channels": data.shape[1], "channel_analysis": analysis}})


def test_dual_mono_file_is_one_lane_and_reads_as_mono(tmp_path: Path) -> None:
    left = _voice(40.0)
    asset = _recorder(tmp_path, np.stack([left, left], axis=1))
    tracks = default_tracks(asset)
    assert [track.channels for track in tracks] == [(0, 1)]
    assert "identical" in tracks[0].name
    assert effective_audio(asset).pan == "pan=mono|c0=c0"


def test_manual_split_survives_a_rescan(tmp_path: Path) -> None:
    left = _voice(40.0)
    asset = _recorder(tmp_path, np.stack([left, left], axis=1))
    project = Project(tmp_path, tmp_path / "_studio", [asset],
                      recorder_tracks=refresh_tracks([asset], []))
    set_track(project, "rec", (0,))
    assert sorted(track.channels for track in project.recorder_tracks) == [(0,), (1,)]
    rescanned = refresh_tracks([asset], project.recorder_tracks)
    assert sorted(track.channels for track in rescanned) == [(0,), (1,)]


def test_old_per_channel_proposal_is_replaced(tmp_path: Path) -> None:
    left = _voice(40.0)
    asset = _recorder(tmp_path, np.stack([left, left], axis=1))
    old = [*default_tracks(Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER,
                                 manual={"media_info": {"audio_channels": 2}}))]
    assert [track.channels for track in refresh_tracks([asset], old)] == [(0, 1)]
