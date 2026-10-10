import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import EditMap, KeepRange, SourcePlacement
from studio.stages.program import render_project_program
from studio.stages.program_audio import (
    SAMPLE_RATE,
    PartAudio,
    decode_part,
    handle_samples,
    join_gains,
    mix_parts,
)


def _read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        data = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return data.reshape(-1, 2).astype(np.float32) / 32768


def _constant(levels: dict[int, float]):
    def decode(part, pre, post, scratch):
        value = levels[id(part)]
        return np.full((pre + part.samples + post, 2), value * 32768, dtype=np.float32)

    return decode


def test_gains_are_complementary() -> None:
    fade_out, fade_in = join_gains(480, continuous=False)
    assert np.allclose(fade_out**2 + fade_in**2, 1, atol=1e-6)
    fade_out, fade_in = join_gains(480, continuous=True)
    assert np.allclose(fade_out + fade_in, 1, atol=1e-6)


def test_mix_keeps_exact_length_and_blends_edit_join(tmp_path: Path) -> None:
    first = PartAudio(None, 0.0, 1.0, 4800)
    second = PartAudio(None, 0.0, 1.0, 9600)
    output = tmp_path / "audio.wav"
    report = mix_parts(
        [first, second], output, 10.0, _constant({id(first): 0.5, id(second): -0.5})
    )
    audio = _read(output)
    assert len(audio) == 4800 + 9600 == report["audio_samples"]
    assert report["audio_crossfade_ms"] == pytest.approx(10.0)
    assert report["audio_joins"] == {"edit": 1, "continuous": 0}
    join = audio[4800 - 240:4800 + 240, 0]
    assert join[0] == pytest.approx(0.5, abs=0.01) and join[-1] == pytest.approx(-0.5, abs=0.01)
    assert np.max(np.abs(np.diff(audio[:, 0]))) < 0.01


def test_continuous_join_keeps_level_and_short_pieces_shrink_handles(tmp_path: Path) -> None:
    parts = [PartAudio(None, 0, 1, 2000), PartAudio(None, 0, 1, 100, True)]
    assert handle_samples(parts, 10.0) == 50
    report = mix_parts(parts, tmp_path / "a.wav", 10.0, _constant({id(p): 0.25 for p in parts}))
    assert report["audio_joins"] == {"edit": 0, "continuous": 1}
    assert np.allclose(_read(tmp_path / "a.wav")[:, 0], 0.25, atol=1e-3)
    assert handle_samples(parts[:1], 10.0) == 0


def test_decode_pads_missing_preroll_with_silence(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=1:sample_rate=48000", "-ac", "2", str(source),
    ], check=True)
    data = decode_part(PartAudio(source, 0.0, 1.0, 4800), 240, 240, tmp_path)
    assert data.shape == (5280, 2)
    assert not np.any(data[:240])
    assert np.any(data[240:])


def test_program_render_crossfades_without_changing_duration(tmp_path: Path) -> None:
    camera = tmp_path / "camera.mp4"
    voice = tmp_path / "voice.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=red:s=160x120:r=25:d=3", "-c:v", "libx264", str(camera),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=3", str(voice),
    ], check=True)
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mp4"), "video", AssetRole.CAMERA)]
    project.assets[0].manual["media_info"] = {"width": 160, "height": 120, "fps": 25}
    project.placements = [SourcePlacement("cam", 10, 0, 3)]
    project.outputs["sync:cam"] = [voice]
    edit = EditMap("cut", (KeepRange(10.4, 11.2), KeepRange(12, 12.6)))
    output = tmp_path / "program.mp4"
    report = render_project_program(project, edit, output, encoder="cpu", crossfade_ms=10)
    assert report["audio_crossfade_ms"] == pytest.approx(10.0)
    assert report["audio_samples"] == round(1.4 * SAMPLE_RATE)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_format", "-of", "json", str(output)
    ], check=True, capture_output=True, text=True)
    assert float(json.loads(probe.stdout)["format"]["duration"]) == pytest.approx(1.4, abs=0.08)
