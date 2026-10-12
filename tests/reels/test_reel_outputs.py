import json
import subprocess
from pathlib import Path

import pytest

from studio.core.project import Project
from studio.core.timeline import TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.reel_finish import check_clip
from studio.stages.reel_render import ReelRenderStage


def _project(tmp_path: Path) -> Project:
    source = tmp_path / "program.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
         "color=c=blue:s=160x90:r=25:d=12", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
        check=True)
    words = [Word(f"w{i}{'.' if i % 4 == 3 else ''}", 0.5 + i * 0.5, 0.9 + i * 0.5)
             for i in range(20)]
    transcript_path = tmp_path / "edited.json"
    Transcript(source, "en", 12, [Segment(0.5, 10.4, tuple(words))],
               time_domain=TimeDomain.EDITED).save(transcript_path)
    moments = tmp_path / "moments.json"
    moments.write_text(json.dumps([
        {"start": 0.5, "end": 4.4, "candidate_id": "good", "score": 8, "title": "Good",
         "quote": "w0 w1 w2", "why": "a payoff"},
        {"start": 6.5, "end": 8.4, "candidate_id": "weak", "score": 4, "title": "Weak"},
    ]))
    return Project(tmp_path, tmp_path / "work", transcripts={"edited": transcript_path},
                   outputs={"program": [source], "reels": [moments]})


@pytest.mark.parametrize("render_rejected", [False, True])
def test_render_filters_exports_and_preview(tmp_path: Path, render_rejected: bool) -> None:
    project = _project(tmp_path)
    settings = {"reels": {"render": True, "covers": False, "loudness_lufs": 0,
                          "min_score": 7, "min_duration_s": 0, "encoder": "libx264",
                          "render_rejected": render_rejected, "export_webm": True,
                          "export_gif": True, "export_audio": True}}
    output = ReelRenderStage().run(StageContext(project, settings, project.work_dir))
    folder = project.work_dir / "stages" / "reel_render"
    report = json.loads((folder / "render.json").read_text())
    rejected = json.loads((folder / "rejected.json").read_text())
    assert [row["candidate_id"] for row in rejected] == ["weak"]
    assert rejected[0]["reasons"] == ["score 4.0 < 7.0"]
    assert rejected[0]["encoded"] is render_rejected
    assert (folder / "rejected" / "reel-002.mp4").is_file() is render_rejected
    good = report["reels"][0]
    assert good["outcome"] == "ok" and good["qa"] == []
    assert set(good["exports"]) == {"webm", "gif", "audio"}
    for suffix in (".webm", ".gif", ".m4a", ".md", ".txt"):
        assert (folder / f"reel-001{suffix}").is_file()
    assert "Good" in (folder / "reel-001.md").read_text()
    assert report["preview"] == "reels-preview.mp4" and (folder / "reels-preview.mp4").is_file()
    assert report["encoder"]["codec"] == "libx264"
    assert not output.warnings


def test_qa_names_a_truncated_reel(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
         "color=c=black:s=64x64:r=25:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True)
    problems = check_clip(clip, expected_s=10, blackdetect=True)
    assert "qa: no audio stream" in problems
    assert any("duration 2.0s, expected 10.0s" in problem for problem in problems)
    assert any("is black" in problem for problem in problems)
