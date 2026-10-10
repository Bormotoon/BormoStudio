import json
import wave
from fractions import Fraction
from pathlib import Path

import pytest

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.stages.export import Sequence, SequenceClip, SequenceMarker, build_sequence
from studio.stages.export_otio import write_otio


def _clip_ranges(document: dict) -> list[tuple[str, str, float, float]]:
    found = []
    for track in document["tracks"]["children"]:
        cursor = 0.0
        for child in track["children"]:
            length = child["source_range"]["duration"]["value"]
            if child["OTIO_SCHEMA"].startswith("Clip"):
                found.append((track["kind"], child["media_reference"]["name"], cursor, length))
            cursor += length
    return found


def test_otio_places_clips_on_frames_with_gaps_and_markers(tmp_path: Path) -> None:
    source = tmp_path / "camera.mov"
    sequence = Sequence("episode", Fraction(25, 1), 1920, 1080, (
        SequenceClip("a", source, 2.0, 5.0, 1.0, 0, True),
        SequenceClip("a", source, 9.0, 2.0, 8.0, 0, True, audio_enabled=False),
    ), (SequenceMarker(3.0, "pause 1.2 s", "pause"),))
    document = json.loads(write_otio(sequence, tmp_path / "e.otio").read_text())
    assert _clip_ranges(document) == [
        ("Video", "camera.mov", 25, 125), ("Video", "camera.mov", 200, 50),
        ("Audio", "camera.mov", 25, 125), ("Audio", "camera.mov", 200, 50),
    ]
    audio = document["tracks"]["children"][1]["children"]
    assert [item["enabled"] for item in audio if item["OTIO_SCHEMA"] == "Clip.1"] == [True, False]
    assert document["tracks"]["markers"][0]["marked_range"]["start_time"]["value"] == 75
    assert document["tracks"]["children"][0]["children"][1]["source_range"][
        "start_time"]["value"] == 50


def test_overlapping_clips_on_one_lane_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "camera.mov"
    sequence = Sequence("episode", Fraction(25, 1), 1920, 1080, (
        SequenceClip("a", source, 0.0, 5.0, 0.0, 0, False),
        SequenceClip("b", source, 0.0, 5.0, 4.0, 0, False),
    ))
    with pytest.raises(ValueError, match="overlap"):
        write_otio(sequence, tmp_path / "e.otio")


def test_opentimelineio_reads_the_export(tmp_path: Path) -> None:
    otio = pytest.importorskip("opentimelineio")
    voice = tmp_path / "voice.wav"
    with wave.open(str(voice), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "PCM"))
        audio.writeframes(b"\x00\x00" * 48000 * 5)
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mov"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("cam", 1.0, 0.0, 5.0)]
    project.assets[0].manual["media_info"] = {"audio_codec": "aac"}
    project.outputs["sync:cam"] = [voice]
    path = write_otio(build_sequence(project), tmp_path / "e.otio")
    timeline = otio.adapters.read_from_file(str(path))
    clips = list(timeline.find_clips())
    assert {clip.media_reference.target_url.rsplit("/", 1)[1] for clip in clips} == {
        "camera.mov", "voice.wav"}
    assert timeline.video_tracks() and timeline.audio_tracks()
