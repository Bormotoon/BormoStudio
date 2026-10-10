import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from studio.core.timeline import AudioWarpMap, AudioWarpPiece, TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.sync_repair import repair_voices


@pytest.mark.parametrize("lag_status,content_ok,accepted", [
    ("passed", True, True), ("inconclusive", True, False),
    ("failed", True, False), ("passed", False, False),
])
def test_repair_requires_both_checks_and_publishes_matching_map(
    tmp_path: Path, lag_status: str, content_ok: bool, accepted: bool,
) -> None:
    source = tmp_path / "recorder.wav"
    source.write_bytes(b"original recorder")
    camera = tmp_path / "camera.mov"
    reference = Transcript(camera, "en", 3, [Segment(0, 3, tuple(
        Word(f"word{i}", i * 0.3, i * 0.3 + 0.2) for i in range(6)
    ))])
    source_transcript = Transcript(source, "en", 3, reference.segments)
    report = tmp_path / "check.json"
    report.write_text(json.dumps({"clips": [{"asset_id": "cam", "status": "failed"}]}))
    warp = AudioWarpMap("warp-rec-cam", "rec", "cam", (
        AudioWarpPiece(0, 3, 0, 3, "copy"),
    ), TimeDomain.FILE, 1)

    def render(*args, **kwargs):
        args[3].write_bytes(b"candidate PCM")
        return SimpleNamespace(warp=warp, refine=None)

    with ExitStack() as stack:
        factory = stack.enter_context(patch("studio.stages.sync_repair.WhisperEngine"))
        engine = factory.return_value
        engine.transcribe.return_value = Transcript(
            camera, "en", 3, reference.segments if content_ok else [],
        )
        align = stack.enter_context(patch("studio.stages.sync_repair.align_sources"))
        stack.enter_context(patch("studio.stages.sync_repair.render_aligned_clip",
                                  side_effect=render))
        measure = stack.enter_context(patch("studio.stages.sync_repair.measure"))
        measure.return_value.verdict.return_value = (lag_status, "test evidence")
        measure.return_value.summary.return_value = {"coverage": 1.0}
        voices, maps, artifacts = repair_voices(
            {"cam": reference}, {"cam": camera},
            {"cam": ("rec", source, source_transcript, 1)},
            tmp_path / "repair", report, {},
        )
        assert align.call_args.args[2] == camera
        engine.backend.unload.assert_called_once()
    assert source.read_bytes() == b"original recorder"
    assert not list((tmp_path / "repair").glob("*-candidate.wav"))
    result = json.loads(artifacts[-1].read_text())["clips"][0]
    assert result["accepted"] is accepted
    if accepted:
        assert voices["cam"].read_bytes() == b"candidate PCM"
        assert maps["cam"].id == warp.id
        assert maps["cam"].pieces == warp.pieces
        assert maps["cam"].evidence["repair_verified"] == 1.0
        assert Transcript.load(artifacts[-2]).source_audio == voices["cam"]
    else:
        assert voices == maps == {}


def test_repair_skips_passed_and_inconclusive_clips(tmp_path: Path) -> None:
    report = tmp_path / "check.json"
    report.write_text(json.dumps({"clips": [
        {"asset_id": "cam1", "status": "passed"},
        {"asset_id": "cam2", "status": "inconclusive"},
    ]}))
    with patch("studio.stages.sync_repair.WhisperEngine") as factory:
        voices, maps, artifacts = repair_voices({}, {}, {}, tmp_path / "repair", report, {})
        factory.return_value.transcribe.assert_not_called()
    assert voices == maps == {}
    assert json.loads(artifacts[-1].read_text())["clips"] == []


def test_repair_backend_error_discards_candidate(tmp_path: Path) -> None:
    report = tmp_path / "check.json"
    report.write_text(json.dumps({"clips": [{"asset_id": "cam", "status": "failed"}]}))
    reference = Transcript(tmp_path / "camera.mov", "en", 3, [])
    with patch("studio.stages.sync_repair.WhisperEngine") as factory, patch(
        "studio.stages.sync_repair.align_sources", side_effect=RuntimeError("no reliable fit"),
    ):
        voices, maps, artifacts = repair_voices(
            {"cam": reference}, {"cam": reference.source_audio},
            {"cam": ("rec", tmp_path / "rec.wav", reference, 1)},
            tmp_path / "repair", report, {},
        )
        factory.return_value.backend.unload.assert_called_once()
    assert voices == maps == {}
    assert "no reliable fit" in json.loads(artifacts[-1].read_text())["clips"][0]["reason"]
