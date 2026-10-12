import json
import subprocess
from pathlib import Path

from studio.core.project import Project
from studio.core.subtitles import Cue, SubtitleStyle, save_edits
from studio.core.timeline import TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.reel_render import ReelRenderStage, clip_transcript


def test_clip_transcript_clips_and_rebases_words(tmp_path: Path) -> None:
    source = Transcript(
        tmp_path / "source.mp4",
        "en",
        10,
        [
            Segment(
                2,
                8,
                (Word("first", 2, 4), Word("last", 6, 8)),
            )
        ],
        time_domain=TimeDomain.EDITED,
    )
    result = clip_transcript(source, 3, 7, tmp_path / "reel.mp4")
    assert [(word.start, word.end) for word in result.words] == [(0, 1), (3, 4)]
    assert result.duration == 4
    assert result.time_domain is TimeDomain.FILE


def test_edited_reel_real_render_and_stable_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "program.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x90:r=25:d=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    transcript_path = tmp_path / "edited.json"
    Transcript(
        source, "en", 3, [Segment(1, 2, (Word("hello", 1, 2),))], time_domain=TimeDomain.EDITED
    ).save(transcript_path)
    moments = tmp_path / "moments.json"
    moments.write_text('[{"start": 1, "end": 2, "candidate_id": "one"}]')
    project = Project(
        tmp_path,
        tmp_path / "work",
        transcripts={"edited": transcript_path},
        outputs={"program": [source], "reels": [moments]},
    )
    stage = ReelRenderStage()
    settings = {
        "reels": {
            "render": True,
            "burn_subtitles": True,
            "framing": "crop",
            "width": 90,
            "height": 160,
            "min_score": 0,
            "min_duration_s": 0,
        }
    }
    before = stage.fingerprint(project, settings)
    output = stage.run(StageContext(project, settings, project.work_dir))
    project.outputs = output.project_changes["outputs"]
    assert stage.fingerprint(project, settings) == before
    video = next(path for path in output.artifacts if path.suffix == ".mp4")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # The edges follow the speech: 0.6 s before the word, 0.3 s after it.
    assert abs(float(json.loads(probe.stdout)["format"]["duration"]) - 1.9) < 0.1
    assert "00:00:00,600 --> 00:00:01,600" in video.with_suffix(".srt").read_text()
    assert video.with_suffix(".md").is_file() and video.with_suffix(".txt").is_file()
    assert any(path.name.endswith("-captioned.mp4") for path in output.artifacts)
    local = Transcript.load(video.with_suffix(".json"))
    save_edits(
        project.work_dir,
        "reel-001",
        local,
        [Cue(0.1, 0.9, "Manual correction")],
        SubtitleStyle(size=60),
    )
    assert stage.fingerprint(project, settings) != before
    stage.run(StageContext(project, settings, project.work_dir))
    assert "Manual correction" in video.with_suffix(".srt").read_text()
    assert "DejaVu Sans,60" in video.with_suffix(".ass").read_text()
    assert "PlayResX: 1080\nPlayResY: 1920" in video.with_suffix(".ass").read_text()
