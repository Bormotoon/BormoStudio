import tracemalloc
import wave
from pathlib import Path

import numpy as np
import pytest

from studio.core.audio_spans import Span
from studio.core.project import Asset, AssetKind, AssetRole
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.locate import locate_camera
from studio.stages.prepare import transcribe_ranges
from studio.stages.roughcut import build_edit_list
from studio.stages.transcribe import TranscribeSettings, WhisperEngine, tiles_for

SR = 16000


def _speech(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(int(seconds * SR)).astype(np.float32) * 0.003
    t = 0.3
    while t < seconds - 1:
        length = rng.uniform(0.2, 0.6)
        start, end = int(t * SR), int((t + length) * SR)
        signal[start:end] += rng.standard_normal(end - start).astype(np.float32) * 0.2
        t += length + rng.uniform(0.1, 0.5)
    return signal


def _wav(path: Path, data: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture(scope="module")
def material(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    folder = tmp_path_factory.mktemp("locate")
    recorder = _speech(1800, 1)
    camera = recorder[int(900 * SR):int(1020 * SR)] + np.random.default_rng(2).standard_normal(
        int(120 * SR)).astype(np.float32) * 0.003
    return _wav(folder / "rec.wav", recorder), _wav(folder / "cam.wav", camera)


def test_clip_in_a_long_recorder_is_found_and_needs_few_tiles(material) -> None:
    recorder, camera = material
    tracemalloc.start()
    found = locate_camera(camera, recorder, None, 1800.0, 120.0, None, 120.0)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Streamed in blocks: decoding the 30-minute recorder whole would take 115 MB.
    assert peak < 64 * 1024 * 1024
    assert found is not None and found.method == "full"
    assert found.position_s == pytest.approx(900.0, abs=0.01)
    ranges = [[found.position_s - 60, found.position_s + 120 + 60]]
    assert len(tiles_for(ranges, 1800.0)) <= 3


def test_predicted_position_is_searched_first(material) -> None:
    recorder, camera = material
    found = locate_camera(camera, recorder, None, 1800.0, 120.0, 905.0, 120.0)
    assert found is not None and found.method == "predicted"
    assert found.position_s == pytest.approx(900.0, abs=0.01)


def _asset(asset_id: str, role: AssetRole, duration: float) -> Asset:
    return Asset(asset_id, Path(f"{asset_id}.wav"), AssetKind.AUDIO, role,
                 manual={"media_info": {"duration_s": duration}})


def test_ranges_follow_located_cameras_and_fall_back_to_the_whole_file() -> None:
    recorder = _asset("rec", AssetRole.RECORDER, 16000.0)
    cameras = [_asset("cam1", AssetRole.CAMERA, 292.0), _asset("cam2", AssetRole.CAMERA, 100.0)]
    positions = {"cam1": {"rec": {"position_s": 47.8}}, "cam2": {"rec": {"position_s": 200.0}}}
    ranges, reasons = transcribe_ranges([recorder], cameras, positions, {})
    assert ranges == {"rec": [[0.0, 399.8]]}
    assert "around cameras" in reasons["rec"]
    ranges, reasons = transcribe_ranges([recorder], cameras, {"cam1": positions["cam1"]}, {})
    assert ranges == {} and "cam2 not located" in reasons["rec"]
    ranges, reasons = transcribe_ranges(
        [recorder], cameras, positions, {"recorder_scope": "full"})
    assert ranges == {} and "full" in reasons["rec"]
    short = _asset("rec", AssetRole.RECORDER, 450.0)
    ranges, reasons = transcribe_ranges([short], cameras, positions, {})
    assert ranges == {} and "cover" in reasons["rec"]


class _Backend:
    device = "cpu"
    compute_type = "float32"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: Path, settings: TranscribeSettings) -> Transcript:
        self.calls += 1
        return Transcript(audio, "ru", 310.0, [
            Segment(word.start, word.end, (word,))
            for word in (Word("в", 5.0, 5.2), Word("ядре", 150.0, 150.4),
                         Word("край", 305.0, 305.3))
        ])


def test_second_camera_reuses_recognized_tiles(material, tmp_path: Path) -> None:
    recorder, _camera = material
    backend = _Backend()
    engine = WhisperEngine(TranscribeSettings(), backend, tmp_path / "cache")
    first = engine.transcribe_ranges(recorder, [[880.0, 1080.0]], 1800.0)
    assert backend.calls == 2  # tiles 2 and 3 (600-1200 s)
    assert first.metadata["coverage"] == [[600.0, 1200.0]]
    assert first.duration == 1800.0
    texts = [word.text for word in first.words]
    # Each tile keeps only words whose middle lies in its core.
    # Tile 2 reads 590-910 s: "в" (595 s) belongs to tile 1, "край" (895 s) is its own.
    assert texts == ["ядре", "край", "ядре", "край"]
    assert [round(word.start) for word in first.words] == [740, 895, 1040, 1195]
    engine.transcribe_ranges(recorder, [[880.0, 1080.0], [1600.0, 1700.0]], 1800.0)
    assert backend.calls == 3


def test_untranscribed_recorder_parts_are_not_unknown_sound() -> None:
    words = [Word("раз", 100.0, 100.5), Word("два", 101.0, 101.5)]
    transcript = Transcript(Path("a.wav"), "ru", 1000.0, [Segment(100, 101.5, tuple(words))])
    edit = build_edit_list(
        transcript, mode="cut", minimum_pause_s=1, keep_pause_s=0.4, edge_pad_s=0.2,
        silence=[Span(60, 99)], coverage=[Span(40, 340)], head_tail_sound="suggest",
    )
    sounds = [marker.at for marker in edit.markers if marker.kind == "unknown-sound"]
    assert sounds and all(40 <= at < 340 for at in sounds)
    head = next(cut for cut in edit.cuts if cut.reason == "head")
    assert head.start == 40.0


def test_digital_silence_and_an_impulse_do_not_capture_the_search(tmp_path: Path) -> None:
    # The sony_run recorder: a clap-like impulse near the start, then five
    # minutes of exact zeros; every window's single best peak sat on the impulse.
    recorder = _speech(900, 7)
    recorder[int(19.4 * SR):int(19.402 * SR)] = 0.95
    recorder[int(52 * SR):int(352 * SR)] = 0.0
    camera = recorder[int(500 * SR):int(800 * SR)] * 0.2 + np.random.default_rng(8).standard_normal(
        int(300 * SR)).astype(np.float32) * 0.01
    rec_path, cam_path = _wav(tmp_path / "rec.wav", recorder), _wav(tmp_path / "cam.wav", camera)
    found = locate_camera(cam_path, rec_path, None, 900.0, 300.0, None, 120.0)
    assert found is not None and found.position_s == pytest.approx(500.0, abs=0.01)
