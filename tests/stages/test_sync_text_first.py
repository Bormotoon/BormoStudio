"""Text-first placement: text against acoustics, the recorder jump map, the second pass."""

from pathlib import Path

import pytest

from studio.core.project import Asset, AssetKind, AssetRole
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.sync_alignment import combine
from studio.stages.sync_drift import DriftCheck, anchor_regime
from studio.stages.sync_grid import Regime, RegimeModel, Step
from studio.stages.sync_jumps import apply_jumps, recorder_jumps, unseen_jumps
from studio.stages.sync_models import AlignmentMap, Anchor


def _words_map(offset: float, k: float = 1.0, *, count: int = 40, span: float = 400.0,
               jump: tuple[float, float] | None = None) -> AlignmentMap:
    """Text anchors of a clip at ``cam = offset + k * rec``; ``jump`` = (rec_s, ms)."""
    start = -offset / k
    anchors = []
    for index in range(count):
        rec = start + 5.0 + index * span / count
        late = jump[1] / 1000.0 if jump is not None and rec > jump[0] else 0.0
        # The word sounds ``late`` later in the recorder file than the line says.
        anchors.append(Anchor(offset + k * rec, rec + late, f"w{index}", 0.9))
    return AlignmentMap(anchors, offset, k, 40.0, "text", count, span)


def test_acoustic_line_that_agrees_with_the_words_takes_their_anchors() -> None:
    words = _words_map(-3074.0)
    sound = AlignmentMap([], -3074.002, 1.0000001, 0.0, "acoustic", 12, 300.0)
    merged = combine(sound, words)
    assert merged.provenance == "acoustic" and merged.offset == sound.offset
    assert len(merged.anchors) == len(words.anchors)
    assert "agrees" in merged.notes[-1]


def test_acoustic_peak_far_from_the_words_is_ignored() -> None:
    # sony_run mvi-0850: acoustics at 12162 s, the words (and the truth) at 3074 s.
    words = _words_map(-3074.0)
    sound = AlignmentMap([], -12162.6, 1.0, 0.0, "acoustic", 4, 200.0)
    merged = combine(sound, words)
    assert merged.provenance == "text" and merged.offset == words.offset
    assert "ignored" in merged.notes[-1]


def _check(offset: float, steps: list[tuple[float, float]], confident: list[float],
           duration: float = 600.0) -> DriftCheck:
    """An acoustic check of a clip with steps at clip times (t_s, delta_ms)."""
    regimes, at = [], 0.0
    for t_s, _delta in steps:
        regimes.append(Regime(at, t_s, 0.0))
        at = t_s
    regimes.append(Regime(at, duration + 1, 0.0))
    model = RegimeModel("regimes" if steps else "constant", regimes,
                        [Step(t, t - 1, t + 1, delta) for t, delta in steps])
    alignment = AlignmentMap([], offset, 1.0, 2.0, "acoustic", len(confident), duration)
    return DriftCheck(alignment, {}, False, model, confident)


def test_a_jump_two_cameras_see_is_one_recorder_jump() -> None:
    # dji-0921 and mvi-0845 both moved by about -516 ms at recorder 843 s.
    checks = {
        ("dji-0921", "rec"): _check(-170.6, [(672.4, -517.0)],
                                    [float(t) for t in range(5, 900, 3)]),
        ("mvi-0845", "rec"): _check(-415.0, [(428.0, -515.0)],
                                    [float(t) for t in range(5, 480, 3)]),
    }
    fits = {key: value.alignment for key, value in checks.items()}
    jumps = recorder_jumps(checks, fits)
    assert len(jumps) == 1
    jump = jumps[0]
    assert jump.rec_s == pytest.approx(843.0, abs=0.1)
    assert jump.delta_ms == pytest.approx(-516.0)
    assert jump.seen_by == ["dji-0921", "mvi-0845"] and jump.accepted


def test_a_camera_that_heard_both_sides_without_a_step_contradicts_it() -> None:
    checks = {
        ("cam-a", "rec"): _check(-100.0, [(300.0, -230.0)], [float(t) for t in range(5, 590, 3)]),
        ("cam-b", "rec"): _check(-100.0, [], [float(t) for t in range(5, 590, 3)]),
    }
    fits = {key: value.alignment for key, value in checks.items()}
    jump = recorder_jumps(checks, fits)[0]
    assert jump.contradicted_by == ["cam-b"] and not jump.accepted


def test_words_confirm_a_large_jump() -> None:
    checks = {("cam-a", "rec"): _check(-100.0, [(300.0, -500.0)],
                                       [float(t) for t in range(5, 590, 3)])}
    words = _words_map(-50.0, count=80, span=500.0, jump=(400.0, -500.0))
    fits = {("cam-a", "rec"): checks["cam-a", "rec"].alignment, ("cam-b", "rec"): words}
    jump = recorder_jumps(checks, fits)[0]
    assert jump.text_confirmed_by == ["cam-b"]


def test_a_clip_without_its_own_acoustics_follows_the_jump_map() -> None:
    checks = {("cam-a", "rec"): _check(-100.0, [(300.0, -500.0)],
                                       [float(t) for t in range(5, 590, 3)])}
    words = _words_map(-50.0, count=80, span=500.0, jump=(400.0, -500.0))
    fits = {("cam-a", "rec"): checks["cam-a", "rec"].alignment, ("cam-b", "rec"): words}
    jumps = recorder_jumps(checks, fits)
    bent = apply_jumps(words, jumps, 550.0)
    assert bent is not None and bent.provenance == "text+jumps"
    before = [a for a in bent.anchors if a.rec_time < 399.0]
    after = [a for a in bent.anchors if a.rec_time > 401.0]
    assert {anchor_regime(a) for a in before} == {0} and {anchor_regime(a) for a in after} == {1}
    # Before the jump the clip reads the recorder on its line, after it 0.5 s earlier.
    assert before[-1].rec_time == pytest.approx(before[-1].cam_time + 50.0, abs=1e-6)
    assert after[0].rec_time == pytest.approx(after[0].cam_time + 50.0 - 0.5, abs=1e-6)
    assert bent.residual_ms == pytest.approx(0.0, abs=1e-3)
    assert unseen_jumps(checks["cam-a", "rec"], jumps, 600.0) == []


def test_a_jump_outside_the_clip_changes_nothing() -> None:
    checks = {("cam-a", "rec"): _check(-100.0, [(300.0, -500.0)],
                                       [float(t) for t in range(5, 590, 3)])}
    words = _words_map(-2000.0)
    jumps = recorder_jumps(checks, {("cam-a", "rec"): checks["cam-a", "rec"].alignment})
    assert apply_jumps(words, jumps, 400.0) is None


def _camera(name: str, duration: float) -> Asset:
    return Asset(name.lower(), Path(f"{name}.mov"), AssetKind.VIDEO, AssetRole.CAMERA,
                 group_id="mvi", manual={"media_info": {"duration_s": duration,
                                                        "audio_channels": 1}})


def test_second_pass_drops_a_text_match_against_the_file_order_and_transcribes_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.stages import sync_rescue

    cameras = [_camera(f"MVI_08{number}", 300.0) for number in (47, 48, 49, 50)]
    recorder = Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER,
                     manual={"media_info": {"duration_s": 13000.0, "audio_channels": 1}})
    transcripts = {camera.id: Transcript(Path(camera.path), "ru", 300.0, [])
                   for camera in cameras}
    fits = {("mvi_0847", "rec"): AlignmentMap([], -2220.0, 1.0, 30.0, "text", 40, 250.0),
            ("mvi_0848", "rec"): AlignmentMap([], -12000.0, 1.0, 30.0, "text", 9, 100.0),
            ("mvi_0850", "rec"): AlignmentMap([], -3074.0, 1.0, 30.0, "text", 30, 250.0)}
    rejected: dict[str, str] = {}
    calls = []

    def fit(target, source, source_range=None, acoustic_search=None):  # type: ignore[no-untyped-def]
        calls.append((target.id, acoustic_search))
        if target.id == "mvi_0849" and transcripts[target.id].words:
            return AlignmentMap([], -2700.0, 1.0, 50.0, "text", 12, 200.0)
        return None

    def transcriber(camera: Asset, audio_filter: str | None) -> Transcript:
        words = (Word("слово", 1.0, 1.5),) if audio_filter else ()
        return Transcript(Path(camera.path), "ru", 300.0,
                          [Segment(1.0, 1.5, words)] if words else [])

    report: dict[str, object] = {}
    warnings: list[str] = []
    hook = sync_rescue.second_pass(
        cameras=cameras, recorders=[recorder], transcripts=transcripts, originals={},
        source_dir=tmp_path, settings={}, output_dir=tmp_path / "rescue", report=report,
        warnings=warnings, transcriber=transcriber, acoustic=False,
    )
    hook(fit, fits, rejected)
    assert ("mvi_0848", "rec") not in fits and "order" in rejected["mvi_0848:rec"]
    assert ("mvi_0849", "rec") in fits  # placed by the cleaned-up quality words
    assert [item["filter"] is not None for item in report["quality_rescue"]["mvi_0849"]] == [
        False, True]
    assert any("cleaned-up" in message for message in warnings)
    assert (tmp_path / "rescue" / "mvi_0849-cleaned.json").is_file()
    assert all(search is False for _id, search in calls)
