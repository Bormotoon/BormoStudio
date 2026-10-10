import subprocess
import wave
from pathlib import Path

import pytest

from studio.core.preview import choose_source, preview_ranges, render_preview
from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement, TimeDomain
from studio.core.transcript import Segment, Transcript, Word


def test_preview_ranges_skip_the_cut_only_when_applied() -> None:
    assert preview_ranges(5, 7, context_s=2, applied=False, duration=20) == [(3, 9)]
    assert preview_ranges(5, 7, context_s=2, applied=True, duration=20) == [(3, 5), (7, 9)]
    assert preview_ranges(0, 7, context_s=2, applied=True, duration=8) == [(7, 8)]


def _tone(path: Path, seconds: float) -> None:
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        f"sine=frequency=440:duration={seconds}", str(path),
    ], check=True)


def test_preview_without_master_uses_transcript_source_placement(tmp_path: Path) -> None:
    source = tmp_path / "rec.wav"
    _tone(source, 10)
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER)]
    project.placements = [SourcePlacement("rec", 5.0, 0.0, 10.0)]
    timeline = tmp_path / "timeline.json"
    Transcript(source, "ru", 15, [Segment(6, 7, (Word("да", 6, 7),))],
               time_domain=TimeDomain.TIMELINE,
               metadata={"source_asset_id": "rec"}).save(timeline)
    project.transcripts["timeline"] = timeline
    assert choose_source(project, 6, 9).kind == "transcript source"
    with pytest.raises(ValueError):
        choose_source(project, 1, 3)
    output, kind = render_preview(project, [(6.0, 7.0), (9.0, 10.0)], tmp_path / "p.wav")
    assert kind == "transcript source"
    with wave.open(str(output)) as result:
        assert result.getnframes() / result.getframerate() == pytest.approx(1.99, abs=0.03)
