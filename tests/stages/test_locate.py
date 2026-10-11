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
ACOUSTIC = {"sync": {"placement": "acoustic"}}


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


def test_parallel_search_matches_the_serial_one(material) -> None:
    recorder, camera = material
    serial = locate_camera(camera, recorder, None, 1800.0, 120.0, None, 120.0, workers=1)
    parallel = locate_camera(camera, recorder, None, 1800.0, 120.0, None, 120.0, workers=3)
    assert serial is not None and parallel is not None
    assert parallel.position_s == pytest.approx(serial.position_s, abs=1e-6)
    assert parallel.position_s == pytest.approx(900.0, abs=0.01)


def test_cpu_workers_default_leaves_a_core_free(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from studio.core.parallel import cpu_workers

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert cpu_workers({}) == 7
    assert cpu_workers({"compute": {"cpu_workers": 2}}) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert cpu_workers({"compute": {"cpu_workers": 0}}) == 1


@pytest.mark.parametrize("confirmed", [True, False])
def test_next_file_is_searched_after_the_previous_but_never_placed_blindly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed: bool,
) -> None:
    from studio.core.project import Project
    from studio.stages import locate as module

    def camera(name: str, duration: float) -> Asset:
        return Asset(name.lower(), Path(f"{name}.mov"), AssetKind.VIDEO, AssetRole.CAMERA,
                     group_id="dji", manual={"media_info": {"duration_s": duration,
                                                            "audio_channels": 2}})

    recorder = _asset("rec", AssetRole.RECORDER, 5000.0)
    project = Project(tmp_path, tmp_path / "_studio",
                      [camera("DJI_0925", 939.0), camera("DJI_0926", 964.0), recorder])
    monkeypatch.setattr(module, "camera_windows", lambda path, duration: [(1.0, np.zeros(8))])
    spans = []

    def fake_search(source, cameras, span, **kwargs):  # type: ignore[no-untyped-def]
        spans.append((sorted(cameras), span))
        if span is None:
            return {"dji_0925": (3911.0, 120.0)}
        return {"dji_0926": (4850.5, 60.0)} if confirmed else {}

    monkeypatch.setattr(module, "search_many", fake_search)
    data = module.locate_project(project, ACOUSTIC)
    assert spans[-1][0] == ["dji_0926"]
    # From 2 min before the previous file's end, clipped to the recorder's end.
    assert spans[-1][1] == pytest.approx((3911.0 + 939.0 - 120.0, 5000.0))
    if confirmed:
        entry = data["positions"]["dji_0926"]["rec"]
        assert entry["method"] == "after_previous_file" and entry["position_s"] == 4850.5
    else:
        assert "dji_0926" not in data["positions"] and "dji_0926" in data["missing"]


def _dji(name: str, duration: float, recorded_at: float | None = None) -> Asset:
    info: dict[str, object] = {"duration_s": duration, "audio_channels": 2}
    if recorded_at is not None:
        info["recorded_at"] = {"epoch": recorded_at}
    return Asset(name.lower(), Path(f"{name}.mov"), AssetKind.VIDEO, AssetRole.CAMERA,
                 group_id="dji", manual={"media_info": info})


def test_a_match_against_the_file_order_is_searched_again_between_neighbours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # sony_run: mvi-0856 sat on a sharp false peak at 24 s although its
    # neighbours were at 6797 s and 7596 s.
    from studio.core.project import Project
    from studio.stages import locate as module

    recorder = _asset("rec", AssetRole.RECORDER, 9000.0)
    project = Project(tmp_path, tmp_path / "_studio", [
        _dji("DJI_0854", 900.0), _dji("DJI_0855", 700.0), _dji("DJI_0856", 60.0),
        _dji("DJI_0857", 600.0), recorder])
    monkeypatch.setattr(module, "camera_windows", lambda path, duration: [(1.0, np.zeros(8))])
    calls = []

    def fake_search(source, cameras, span, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((sorted(cameras), span, kwargs.get("strict", False)))
        if span is None:
            return {"dji_0854": (5800.0, 70.0), "dji_0855": (6797.0, 66.0),
                    "dji_0856": (24.6, 225.0),
                    "dji_0857": (7596.0, 98.0)}
        return {"dji_0856": (7510.0, 80.0)}

    monkeypatch.setattr(module, "search_many", fake_search)
    data = module.locate_project(project, ACOUSTIC)
    names, span, strict = calls[-1]
    assert names == ["dji_0856"] and strict
    assert span == pytest.approx((6797.0 + 700.0 - 120.0, 7596.0 + 120.0))
    entry = data["positions"]["dji_0856"]["rec"]
    assert entry["method"] == "between_files" and entry["position_s"] == 7510.0


def test_an_out_of_order_match_without_a_better_place_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.core.project import Project
    from studio.stages import locate as module

    recorder = _asset("rec", AssetRole.RECORDER, 9000.0)
    project = Project(tmp_path, tmp_path / "_studio", [
        _dji("DJI_0842", 30.0), _dji("DJI_0845", 500.0), _dji("DJI_0846", 900.0), recorder])
    monkeypatch.setattr(module, "camera_windows", lambda path, duration: [(1.0, np.zeros(8))])

    def fake_search(source, cameras, span, **kwargs):  # type: ignore[no-untyped-def]
        if span is None:
            return {"dji_0842": (8367.0, 56.0), "dji_0845": (415.0, 76.0),
                    "dji_0846": (974.0, 132.0)}
        return {}

    monkeypatch.setattr(module, "search_many", fake_search)
    data = module.locate_project(project, ACOUSTIC)
    assert "dji_0842" not in data["positions"]
    assert "order" in data["missing"]["dji_0842"]["rec"]
    assert set(data["positions"]) == {"dji_0845", "dji_0846"}


def test_a_group_delta_is_a_hint_only_when_its_clips_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # recorded_at is the copy time on sony_run: every clip has its own delta.
    from studio.core.project import Project
    from studio.stages import locate as module

    recorder = Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER,
                     manual={"media_info": {"duration_s": 9000.0,
                                            "recorded_at": {"epoch": 1000.0}}})
    cameras = [_dji("DJI_0001", 600.0, 90000.0), _dji("DJI_0002", 600.0, 90040.0),
               _dji("DJI_0003", 600.0, 90080.0)]
    project = Project(tmp_path, tmp_path / "_studio", [*cameras, recorder])
    monkeypatch.setattr(module, "camera_windows", lambda path, duration: [(1.0, np.zeros(8))])
    copied = {"dji_0001": (100.0, 150.0), "dji_0002": (900.0, 150.0),
              "dji_0003": (1700.0, 150.0)}
    monkeypatch.setattr(module, "search_many",
                        lambda source, cams, span, **kwargs: {
                            key: value for key, value in copied.items() if key in cams})
    data = module.locate_project(project, ACOUSTIC)
    assert data["deltas"] == {}
    deltas = {key: value["rec"]["delta_s"] for key, value in data["positions"].items()}
    assert len(set(deltas.values())) == 3  # one per clip
    agreeing = {"dji_0001": (100.0, 150.0), "dji_0002": (140.0, 150.0),
                "dji_0003": (180.0, 150.0)}
    copied.clear()
    copied.update(agreeing)
    project = Project(tmp_path, tmp_path / "_studio", [
        _dji("DJI_0001", 30.0, 90000.0), _dji("DJI_0002", 30.0, 90040.0),
        _dji("DJI_0003", 30.0, 90080.0), recorder])
    data = module.locate_project(project, ACOUSTIC)
    assert list(data["deltas"]) == ["dji|rec"]
    # One clip far off (sony_run mvi-0845: 240 s) does not cancel the others' hint.
    copied["dji_0003"] = (430.0, 150.0)
    data = module.locate_project(project, ACOUSTIC)
    assert data["deltas"]["dji|rec"] == pytest.approx(1000.0 + 100.0 - 90000.0)


def test_a_narrow_search_needs_a_stricter_vote() -> None:
    from studio.stages.locate import STRICT_MIN_SHARPNESS, _vote

    weak = [(12162.6, 0, 60.8), (12162.62, 1, 55.0)]
    assert _vote(list(weak)) is not None
    assert _vote(list(weak), 2, STRICT_MIN_SHARPNESS) is None
    assert _vote([(10.0, 0, 90.0), (10.01, 1, 80.0)], 3) is None
