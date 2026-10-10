import json
import subprocess
from pathlib import Path

import pytest

from studio.core.framing import load_reel_overrides, reel_settings, save_reel_override
from studio.core.project import Project
from studio.core.settings import SettingsError, load_settings
from studio.core.timeline import TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.reel_layout import reel_title, title_ass, title_band_height
from studio.stages.reel_render import ReelRenderStage


def _program(path: Path, size: str = "320x180") -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=s={size}:r=25:d=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )


def _dimensions(path: Path) -> tuple[int, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    return stream["width"], stream["height"]


def _has_audio(path: Path) -> bool:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return bool(json.loads(probe.stdout)["streams"])


def _project(tmp_path: Path) -> Project:
    source = tmp_path / "program.mp4"
    _program(source)
    transcript_path = tmp_path / "edited.json"
    Transcript(
        source, "en", 3, [Segment(0.5, 2.5, (Word("hello", 0.5, 2.5),))],
        time_domain=TimeDomain.EDITED,
    ).save(transcript_path)
    moments = tmp_path / "moments.json"
    moments.write_text(json.dumps([
        {"start": 0.5, "end": 2.0, "candidate_id": "a", "hook": "Why it works", "title": "T"},
        {"start": 1.0, "end": 2.5, "candidate_id": "b", "hook": "", "title": "Second"},
    ]))
    return Project(
        tmp_path, tmp_path / "work", transcripts={"edited": transcript_path},
        outputs={"program": [source], "reels": [moments]},
    )


def test_title_and_audiogram_layouts_render_portrait_reels(tmp_path: Path) -> None:
    project = _project(tmp_path)
    base = {"render": True, "width": 180, "height": 320, "framing": "crop", "covers": False,
            "loudness_lufs": 0}
    stage = ReelRenderStage()
    output = stage.run(StageContext(project, {"reels": {**base, "layout": "title"}},
                                    project.work_dir))
    report = json.loads(next(p for p in output.artifacts if p.name == "render.json").read_text())
    first, second = report["reels"]
    assert first["layout"] == "title" and first["title"] == "Why it works"
    assert second["title"] == "Second"  # empty hook falls back to the title
    video = project.work_dir / "stages" / "reel_render" / "reel-001.mp4"
    assert _dimensions(video) == (180, 320)
    assert (video.parent / first["frame"]).is_file() and first["source_size"] == [320, 180]
    assert not (video.parent / "reel-001-framed.mp4").exists()

    output = stage.run(StageContext(project, {"reels": {**base, "layout": "audiogram"}},
                                    project.work_dir))
    report = json.loads(next(p for p in output.artifacts if p.name == "render.json").read_text())
    assert report["reels"][0]["framing"] == "audiogram" and report["reels"][0]["frame"] is None
    assert _dimensions(video) == (180, 320) and _has_audio(video)


def test_reel_override_changes_one_reel_and_the_fingerprint(tmp_path: Path) -> None:
    project = _project(tmp_path)
    settings = {"reels": {"render": True, "width": 180, "height": 320, "framing": "fit",
                          "covers": False, "loudness_lufs": 0}}
    stage = ReelRenderStage()
    before = stage.fingerprint(project, settings)
    save_reel_override(project.work_dir, "b", {"framing": "crop", "crop_x": 0.0})
    assert load_reel_overrides(project.work_dir) == {"b": {"framing": "crop", "crop_x": 0.0}}
    assert stage.fingerprint(project, settings) != before
    output = stage.run(StageContext(project, settings, project.work_dir))
    report = json.loads(next(p for p in output.artifacts if p.name == "render.json").read_text())
    first, second = report["reels"]
    assert (first["framing"], first["override"]) == ("fit", False)
    assert (second["framing"], second["crop_x"], second["override"]) == ("crop", 0.0, True)
    save_reel_override(project.work_dir, "b", None)
    assert load_reel_overrides(project.work_dir) == {}


@pytest.mark.parametrize("values", [
    {"framing": "zoom"}, {"crop_x": 2}, {"tracking": "yes"}, {"tracking": True, "framing": "fit"},
    {"width": 100},
])
def test_invalid_reel_overrides_are_rejected(tmp_path: Path, values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        save_reel_override(tmp_path, "a", values)
    assert not (tmp_path / "reels" / "overrides.json").exists()


def test_override_tracking_needs_crop_with_project_framing() -> None:
    with pytest.raises(ValueError):
        reel_settings({"framing": "fit"}, {"tracking": True})
    assert reel_settings({"framing": "fit"}, {"framing": "crop", "title": "x"}) == {
        "framing": "crop"}


def test_title_script_sits_in_the_band_and_respects_duration() -> None:
    settings = {"width": 1080, "height": 1920, "title_band": 0.25, "title_seconds": 3.0}
    assert title_band_height(settings) == 480
    script = title_ass("Hook {bold} text", 10.0, settings)
    assert "\\pos(540,240)" in script and "0:00:03.00" in script and "{bold}" not in script
    assert reel_title({"hook": "", "title": "Name"}, {"title_field": "hook"}, None) == "Name"
    assert reel_title({"hook": "H"}, {}, "  Manual ") == "Manual"


@pytest.mark.parametrize("override", [
    "reels.layout=split", "reels.title_band=0.9", "reels.title_color=white",
    "reels.title_font=A,B", "reels.title_seconds=-1",
])
def test_invalid_layout_settings(override: str) -> None:
    with pytest.raises(SettingsError):
        load_settings([], [override])


def test_caption_preset_styles_reels_until_captions_are_edited(tmp_path: Path) -> None:
    from studio.core.subtitles import Cue, SubtitleStyle, save_edits

    project = _project(tmp_path)
    Transcript(
        project.outputs["program"][0], "en", 3,
        [Segment(0.5, 2.5, tuple(Word(w, 0.5 + i * 0.3, 0.75 + i * 0.3)
                                 for i, w in enumerate("one two three four five six".split())))],
        time_domain=TimeDomain.EDITED,
    ).save(project.transcripts["edited"])
    settings = {"reels": {"render": True, "width": 180, "height": 320, "framing": "crop",
                          "covers": False, "loudness_lufs": 0, "burn_subtitles": True,
                          "caption_preset": "hormozi"}}
    output = ReelRenderStage().run(StageContext(project, settings, project.work_dir))
    folder = project.work_dir / "stages" / "reel_render"
    script = (folder / "reel-001.ass").read_text()
    assert "Montserrat Black" in script and "\\rHighlight" in script
    assert "ONE TWO THREE FOUR" in (folder / "reel-001.srt").read_text()
    assert (folder / "reel-001-captioned.mp4").is_file()
    report = json.loads(next(p for p in output.artifacts if p.name == "render.json").read_text())
    assert report["reels"][0]["caption_preset"] == "hormozi"
    local = Transcript.load(folder / "reel-001.json")
    save_edits(project.work_dir, "reel-001", local, [Cue(0.1, 0.9, "Edited")], SubtitleStyle())
    ReelRenderStage().run(StageContext(project, settings, project.work_dir))
    assert "Montserrat" not in (folder / "reel-001.ass").read_text()
    assert "Edited" in (folder / "reel-001.srt").read_text()
