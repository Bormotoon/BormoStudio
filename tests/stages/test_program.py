import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import EditMap, KeepRange, SourcePlacement, TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.program import (
    align_transcript_to_frames,
    map_transcript_to_edited,
    program_pieces,
    render_project_program,
)
from studio.stages.program_encoder import EncoderChoice


def test_program_transcript_uses_edited_time_and_drops_cut_words() -> None:
    transcript = Transcript(
        Path("audio.wav"),
        "en",
        10,
        [
            Segment(
                0,
                9,
                (
                    Word("keep", 1, 2),
                    Word("cut", 4, 5),
                    Word("again", 7, 8),
                ),
            )
        ],
        time_domain=TimeDomain.TIMELINE,
    )
    edit = EditMap("edit", (KeepRange(0, 3), KeepRange(6, 10)))
    result = map_transcript_to_edited(transcript, edit)
    assert result.time_domain is TimeDomain.EDITED
    assert [(word.text, word.start, word.end) for word in result.words] == [
        ("keep", 1, 2),
        ("again", 4, 5),
    ]
    assert result.duration == 7


def test_render_uses_placement_and_synced_audio_on_real_media(tmp_path: Path) -> None:
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
    project.assets[0].manual["media_info"] = {"width":160, "height":120, "fps":25}
    project.placements = [SourcePlacement("cam", 10, 0, 3)]
    project.outputs["sync:cam"] = [voice]
    edit = EditMap("cut", (KeepRange(10.4, 11.2), KeepRange(12, 12.6)))
    output = tmp_path / "program.mp4"
    report = render_project_program(project, edit, output, encoder="cpu")
    assert report["final_codec"] == "libx264"
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)
    ], check=True, capture_output=True, text=True)
    data = json.loads(probe.stdout)
    assert {stream["codec_type"] for stream in data["streams"]} == {"audio", "video"}
    assert float(data["format"]["duration"]) == pytest.approx(1.4, abs=0.08)
    assert next(stream for stream in data["streams"] if stream["codec_type"] == "video")[
        "width"
    ] == 160


def test_runtime_gpu_failure_rerenders_all_parts_on_cpu(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "work")
    project.assets = [Asset("cam", Path("camera.mp4"), "video", AssetRole.CAMERA)]
    project.assets[0].manual["media_info"] = {"width": 160, "height": 120, "fps": 25}
    project.placements = [SourcePlacement("cam", 0, 0, 3)]
    edit = EditMap("cut", (KeepRange(0, 1), KeepRange(2, 3)))
    commands = []

    def run(command):
        commands.append(command.copy())
        if len(commands) == 2:
            raise RuntimeError("GPU disappeared")

    with patch("studio.stages.program.select_encoder", side_effect=[
        EncoderChoice("auto", "h264_nvenc", "probe passed"),
        EncoderChoice("cpu", "libx264", "CPU requested"),
    ]), patch("studio.stages.program._run_ffmpeg", side_effect=run):
        report = render_project_program(project, edit, tmp_path / "out.mp4")
    assert report["requested_encoder"] == "auto"
    assert report["initial_codec"] == "h264_nvenc"
    assert report["final_codec"] == "libx264"
    assert report["runtime_fallback_reason"] == "GPU disappeared"
    assert [cmd[cmd.index("-c:v") + 1] for cmd in commands] == [
        "h264_nvenc", "h264_nvenc", "libx264", "libx264", "copy",
    ]


@pytest.mark.parametrize("target_fps", ["25", "30000/1001"])
def test_real_mixed_fps_and_vfr_master_has_uniform_frame_timing(
    tmp_path: Path, target_fps: str,
) -> None:
    from fractions import Fraction

    project = Project(tmp_path, tmp_path / "work")
    for index, fps in enumerate(["24", "30000/1001", "60"]):
        source = tmp_path / f"camera-{index}.mp4"
        command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                   f"testsrc2=s=160x120:r={fps}:d=2"]
        if index == 2:
            command.extend(["-vf", "select='if(lt(t,1),1,not(mod(n,3)))'",
                            "-fps_mode", "vfr"])
        command.extend(["-c:v", "libx264", str(source)])
        subprocess.run(command, check=True)
        asset = Asset(f"cam{index}", Path(source.name), "video", AssetRole.CAMERA)
        asset.manual["media_info"] = {"width": 160, "height": 120, "fps": fps}
        project.assets.append(asset)
        project.placements.append(SourcePlacement(asset.id, index * 2, 0, 2))
    # Confirm this fixture really contains unequal presentation-time steps.
    vfr = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json",
        str(tmp_path / "camera-2.mp4"),
    ], check=True, capture_output=True, text=True)
    pts = [float(frame["best_effort_timestamp_time"])
           for frame in json.loads(vfr.stdout)["frames"]]
    assert len({round(b - a, 4) for a, b in zip(pts, pts[1:], strict=False)}) > 1
    # Repeated fractional-frame cuts exercise cumulative rounding.
    edit = EditMap("mixed", tuple(
        KeepRange(index * 2 + cut * 0.13, index * 2 + cut * 0.13 + 0.11)
        for index in range(3) for cut in range(4)
    ))
    output = tmp_path / "master.mp4"
    report = render_project_program(project, edit, output, encoder="libx264", fps=target_fps)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(output),
    ], check=True, capture_output=True, text=True)
    times = [float(frame["best_effort_timestamp_time"])
             for frame in json.loads(probe.stdout)["frames"]]
    rate = float(Fraction(target_fps))
    assert len(times) == report["frame_count"] == round(1.32 * rate)
    assert all(b - a == pytest.approx(1 / rate, abs=0.002)
               for a, b in zip(times, times[1:], strict=False))
    assert times[-1] + 1 / rate == pytest.approx(report["duration_s"], abs=0.002)


def test_edited_transcript_tracks_frame_padding_and_dropped_subframe_pieces() -> None:
    transcript = Transcript(Path("voice.wav"), "en", 0.12, [Segment(0, 0.12, (
        Word("first", 0, 0.03), Word("dropped", 0.03, 0.04),
        Word("last", 0.04, 0.12),
    ))], time_domain=TimeDomain.EDITED)
    result = align_transcript_to_frames(transcript, {
        "fps": "25", "piece_durations_s": [0.03, 0.01, 0.08], "frame_counts": [1, 0, 2],
    })
    assert [word.text for word in result.words] == ["first", "last"]
    assert result.words[-1].start == pytest.approx(0.04)
    assert result.duration == pytest.approx(0.12)


def test_range_only_the_recorder_covers_is_black_with_recorder_sound(tmp_path: Path) -> None:
    camera, recorder = tmp_path / "camera.mp4", tmp_path / "rec.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=white:s=160x120:r=25:d=3", "-c:v", "libx264", str(camera)],
                   check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=6", str(recorder)], check=True)
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mp4"), "video", AssetRole.CAMERA),
                      Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER)]
    project.assets[0].manual["media_info"] = {"width": 160, "height": 120, "fps": 25}
    # The recorder started 2 s before the camera.
    project.placements = [SourcePlacement("cam", 2, 0, 3), SourcePlacement("rec", 0, 0, 6)]
    edit = EditMap("cut", (KeepRange(1.0, 3.0),))
    pieces = program_pieces(project, edit)
    assert [(piece.video, piece.audio, piece.audio_in_s) for piece in pieces] == [
        (None, recorder, 1.0), (camera, None, 0.0)]
    output = tmp_path / "program.mp4"
    render_project_program(project, edit, output, encoder="cpu")
    frames = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(output), "-vf",
         "select='eq(n\\,10)+eq(n\\,40)',scale=1:1", "-vsync", "0", "-f", "rawvideo",
         "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout
    assert frames[0] < 30 and frames[1] > 220  # black, then the white camera
    level = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(output), "-t", "0.9", "-af",
         "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr
    assert "mean_volume: -inf" not in level and "mean_volume" in level


def test_seeking_into_long_sources_keeps_the_exact_frame(tmp_path: Path) -> None:
    # Frame n has brightness n; keyframes every 2 s, so a seek lands between them.
    camera = tmp_path / "camera.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "nullsrc=s=64x64:r=20:d=12,geq=lum='N':cb=128:cr=128", "-g", "40",
                    "-c:v", "libx264", "-qp", "0", str(camera)], check=True)
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mp4"), "video", AssetRole.CAMERA)]
    project.assets[0].manual["media_info"] = {"width": 64, "height": 64, "fps": 20}
    project.placements = [SourcePlacement("cam", 0, 0, 12)]
    output = tmp_path / "program.mp4"
    render_project_program(project, EditMap("cut", (KeepRange(7.3, 8.0),)), output,
                           encoder="libx264")
    frames = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(output), "-vf", "scale=2:2",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], check=True, capture_output=True).stdout
    luma = frames[::6]  # 4 Y + 1 U + 1 V bytes per 2x2 frame
    assert len(luma) == 14 and abs(luma[0] - 146) <= 1 and abs(luma[-1] - 159) <= 1
