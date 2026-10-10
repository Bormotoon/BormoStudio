import json
import subprocess
from pathlib import Path

import pytest

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.timeline import SourcePlacement, TimeDomain
from studio.core.transcript import Transcript
from studio.stages.timeline_preview import TimelinePreview, render_timeline_preview


def _project(tmp_path: Path) -> Project:
    camera = tmp_path / "camera.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=1280x720:r=25:d=6", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=6", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(camera)], check=True)
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mp4"), AssetKind.VIDEO, AssetRole.CAMERA)]
    project.assets[0].manual["media_info"] = {"width": 1280, "height": 720, "fps": 25}
    project.placements = [SourcePlacement("cam", 0, 0, 6)]
    timeline = project.work_dir / "timeline.json"
    Transcript(camera, "en", 6, [], time_domain=TimeDomain.TIMELINE).save(timeline)
    project.transcripts["timeline"] = timeline
    edit = project.work_dir / "stages" / "roughcut" / "edit.json"
    edit.parent.mkdir(parents=True)
    edit.write_text(json.dumps({"keep": [{"start": 0, "end": 2}, {"start": 4, "end": 6}]}))
    return project


def test_whole_edit_preview_is_a_cached_low_resolution_proxy(tmp_path: Path) -> None:
    project = _project(tmp_path)
    preview = render_timeline_preview(project, {})
    assert not preview.reused
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                            "-of", "json", str(preview.video)],
                           check=True, capture_output=True, text=True)
    data = json.loads(probe.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (640, 360)
    assert float(data["format"]["duration"]) == pytest.approx(4, abs=0.1)
    assert preview.to_timeline(3.0) == pytest.approx(5.0)
    assert preview.to_edited(3.0) == pytest.approx(2.0)  # inside the cut: next kept moment
    again = render_timeline_preview(project, {})
    assert again.reused and again.video == preview.video
    edit = project.work_dir / "stages" / "roughcut" / "edit.json"
    edit.write_text(json.dumps({"keep": [{"start": 0, "end": 6}]}))
    changed = render_timeline_preview(project, {})
    assert not changed.reused and not preview.video.exists()  # stale proxy removed


def test_time_mapping_ends_and_gaps() -> None:
    preview = TimelinePreview(Path("p.mp4"), (
        {"timeline_start": 1, "timeline_end": 2, "edited_start": 0, "edited_end": 1},
        {"timeline_start": 5, "timeline_end": 7, "edited_start": 1, "edited_end": 3},
    ), False)
    assert preview.to_edited(0) == 0 and preview.to_edited(9) == 3
    assert preview.to_timeline(2.5) == 6.5 and preview.to_timeline(10) == 7
    with pytest.raises(ValueError):
        render_timeline_preview(Project(Path("/m"), Path("/w")), {})
