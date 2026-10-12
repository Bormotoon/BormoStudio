import json
import wave
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import (
    AudioWarpMap,
    AudioWarpPiece,
    EditMap,
    KeepRange,
    SourcePlacement,
    TimeDomain,
)
from studio.stages.export import (
    Sequence,
    SequenceClip,
    SequenceMarker,
    _parse_time,
    build_sequence,
    multicam_camera_plan,
)
from studio.stages.multicam import write_multicam


def _wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "PCM"))
        audio.writeframes(b"\x00\x00" * round(48000 * seconds))
    return path


def _warp(source: str, target: str) -> AudioWarpMap:
    return AudioWarpMap("w", source, target, (AudioWarpPiece(0, 1, 0, 1, "copy"),),
                        TimeDomain.FILE, 1)


def _sources(clip: ET.Element) -> list[tuple[str, str]]:
    return [(source.get("angleID"), source.get("srcEnable")) for source in clip]


def test_angles_follow_cameras_and_the_recorder_is_heard(tmp_path: Path) -> None:
    sequence = Sequence("Demo", Fraction(25), 1920, 1080, (
        SequenceClip("a1", tmp_path / "a1.mov", 0, 2, 0, 0, True, audio_enabled=False,
                     angle="A"),
        SequenceClip("b", tmp_path / "b.mov", 0, 2, 1, 1, True, angle="B"),
        SequenceClip("a2", tmp_path / "a2.mov", 0, 1, 4, 2, True, angle="A"),
        SequenceClip("rec", tmp_path / "rec.wav", 0, 6, 0, -3000, True, has_video=False,
                     angle="Recorder"),
    ))
    root = ET.parse(write_multicam(sequence, tmp_path / "out.fcpxml",
                                   media_base=tmp_path)).getroot()
    angles = root.findall("./resources/media/multicam/mc-angle")
    assert [angle.get("name") for angle in angles] == ["A", "B", "Recorder"]
    # An angle is a storyline: the hole between the two files is a gap.
    assert [(child.tag, _parse_time(child.get("offset")), _parse_time(child.get("duration")))
            for child in angles[0]] == [("asset-clip", 0, 2), ("gap", 2, 2),
                                        ("asset-clip", 4, 1)]
    assert [child.tag for child in angles[1]] == ["gap", "asset-clip"]
    assert all("lane" not in child.attrib for angle in angles for child in angle)
    # Angles keep their own sound; the project chooses what is heard.
    assert angles[0].find(".//adjust-volume") is None
    clips = root.findall(".//project/sequence/spine/mc-clip")
    # A camera without material holds the angle; no switch per file.
    assert [(_parse_time(clip.get("offset")), _parse_time(clip.get("duration")))
            for clip in clips] == [(0, 2), (2, 2), (4, 2)]
    assert [_sources(clip) for clip in clips] == [
        [("angle-1", "video"), ("angle-3", "audio")],
        [("angle-2", "video"), ("angle-3", "audio")],
        [("angle-1", "video"), ("angle-3", "audio")],
    ]
    assert root.find(".//project/sequence/spine/gap") is None
    # The browser lists the multicam clip and every file, not only the project.
    event = root.find("./library/event")
    assert event.get("name") == "Demo"
    browser = event.find("mc-clip")
    assert browser.get("name") == "Demo Multicam"
    assert _parse_time(browser.get("duration")) == 6
    assert _sources(browser) == [("angle-1", "video"), ("angle-3", "audio")]
    assert sorted(clip.get("name") for clip in event.findall("asset-clip")) == [
        "a1", "a2", "b", "rec"]
    refs = {asset.get("id") for asset in root.findall("./resources/asset")}
    assert all(clip.get("ref") in refs for clip in root.iter("asset-clip"))


def test_without_a_recorder_each_angle_brings_its_own_sound(tmp_path: Path) -> None:
    sequence = Sequence("Demo", Fraction(25), 1920, 1080, (
        SequenceClip("a", tmp_path / "a.mov", 0, 4, 0, 0, True, angle="A"),
        SequenceClip("b", tmp_path / "b.mov", 0, 4, 0, 1, True, angle="B"),
    ))
    path = write_multicam(sequence, tmp_path / "selected.fcpxml", media_base=tmp_path,
                          camera_plan=EditMap("plan", (KeepRange(0, 2, "a"),
                                                       KeepRange(2, 4, "b"))))
    clips = ET.parse(path).getroot().findall(".//project/sequence/spine/mc-clip")
    assert [_sources(clip) for clip in clips] == [[("angle-1", "all")], [("angle-2", "all")]]


def test_the_edit_lives_in_the_project_over_whole_angles(tmp_path: Path) -> None:
    sequence = Sequence("Cut", Fraction(25), 1920, 1080, (
        SequenceClip("a", tmp_path / "a.mov", 0, 10, 0, 0, True, angle="A"),
        SequenceClip("b", tmp_path / "b.mov", 0, 10, 0, 1, True, angle="B"),
        SequenceClip("rec", tmp_path / "rec.wav", 0, 10, 0, -3000, True, has_video=False,
                     angle="Recorder"),
    ))
    edit = EditMap("cut", (KeepRange(1, 3), KeepRange(6, 8)))
    root = ET.parse(write_multicam(
        sequence, tmp_path / "cut.fcpxml", media_base=tmp_path, edit=edit,
        markers=(SequenceMarker(2, "quiet pause", "pause"),
                 SequenceMarker(2.6, "um", "filler")),
    )).getroot()
    angle = root.find("./resources/media/multicam/mc-angle")
    assert _parse_time(angle.find("asset-clip").get("duration")) == 10
    clips = root.findall(".//project/sequence/spine/mc-clip")
    assert [(_parse_time(clip.get("offset")), _parse_time(clip.get("start")),
             _parse_time(clip.get("duration"))) for clip in clips] == [(0, 1, 2), (2, 6, 2)]
    assert _parse_time(root.find(".//project/sequence").get("duration")) == 4
    # Markers use the multicam clock of the project clip that holds them.
    assert [(marker.get("value"), _parse_time(marker.get("start")))
            for marker in clips[1].findall("marker")] == [("quiet pause", 6), ("um", 6.6)]


def test_speaker_plan_stays_in_original_time(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "work")
    project.assets = [Asset("a", Path("a.mov"), "video", AssetRole.CAMERA),
                      Asset("b", Path("b.mov"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("a", 0, 0, 10), SourcePlacement("b", 0, 0, 10)]
    speakers = tmp_path / "speakers.json"
    speakers.write_text(json.dumps([{"start": 0, "end": 5, "speaker": "Alice"},
                                    {"start": 5, "end": 10, "speaker": "Bob"}]))
    edit = tmp_path / "edit.json"
    edit.write_text(json.dumps({"mode": "cut", "keep": [{"start": 0, "end": 2},
                                                         {"start": 6, "end": 8}]}))
    project.outputs = {"roughcut": [edit], "speakers": [speakers]}
    sequence = Sequence("Demo", Fraction(25), 1920, 1080, ())
    plan = multicam_camera_plan(project, {"program": {"speaker_cameras": {
        "Alice": "a", "Bob": "b"}, "min_shot_s": 0}}, True, sequence)
    assert plan.keep == (KeepRange(0, 2, "a"), KeepRange(6, 8, "b"))


def test_recorder_replaces_the_voices_rendered_from_it(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "work")
    camera = Asset("dji-1", Path("DJI_1.mov"), "video", AssetRole.CAMERA, group_id="dji")
    camera.manual["media_info"] = {"fps": "25", "audio_codec": "aac", "duration_s": 4}
    recorder = Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER)
    project.assets = [recorder, camera]
    project.placements = [SourcePlacement("rec", 0, 0, 10.0, 1.25),
                          SourcePlacement("dji-1", 2, 1, 3)]
    project.audio_warp_maps = [_warp("rec", "dji-1")]
    repaired = _wav(tmp_path / "repaired.wav", 10)
    _wav(tmp_path / "rec.wav", 9)
    project.outputs = {"sync:dji-1": [_wav(tmp_path / "voice.wav", 4)],
                       "recorder:rec": [repaired]}
    sequence = build_sequence(project, retimed=True)
    assert [(clip.asset_id, clip.angle) for clip in sequence.clips] == [
        ("dji-1", "DJI"), ("rec", "Recorder")]
    camera_clip, recorder_clip = sequence.clips
    assert not camera_clip.audio_enabled
    assert recorder_clip.path == repaired
    assert recorder_clip.duration_s == 12.5 and recorder_clip.rate == 1.25
    assert recorder_clip.source_duration_s == 10
    # Not the file stem: Final Cut must not take the original for it by name.
    assert recorder_clip.name == "Recorder (repaired)"
    root = ET.parse(write_multicam(sequence, tmp_path / "rec.fcpxml",
                                   media_base=tmp_path)).getroot()
    points = root.findall("./library/event/sync-clip//asset-clip[@lane='-1']/timeMap/timept")
    assert [_parse_time(point.get("value")) for point in points] == [0, 10]


def test_one_camera_and_a_recorder_make_a_synchronized_clip(tmp_path: Path) -> None:
    sequence = Sequence("Solo", Fraction(25), 1920, 1080, (
        SequenceClip("c1", tmp_path / "c1.mov", 0, 2, 1, 0, True, audio_enabled=False,
                     angle="CAM"),
        SequenceClip("c2", tmp_path / "c2.mov", 0, 3, 4, 1, True, audio_enabled=False,
                     angle="CAM"),
        SequenceClip("rec", tmp_path / "rec.wav", 0, 8, 0, -3000, True, has_video=False,
                     angle="Recorder", label="Recorder (repaired)"),
    ))
    edit = EditMap("cut", (KeepRange(0, 3), KeepRange(5, 8)))
    root = ET.parse(write_multicam(
        sequence, tmp_path / "solo.fcpxml", media_base=tmp_path, edit=edit,
        markers=(SequenceMarker(4, "retake", "retake"),),
    )).getroot()
    assert root.find("./resources/media") is None
    browser = root.find("./library/event/sync-clip")
    assert browser.get("name") == "Solo Synchronized"
    assert _parse_time(browser.get("duration")) == 8
    # The camera files form the storyline, holes as gaps.
    assert [(child.tag, _parse_time(child.get("offset")), _parse_time(child.get("duration")))
            for child in browser if child.tag != "sync-source"] == [
        ("gap", 0, 1), ("asset-clip", 1, 2), ("gap", 3, 1), ("asset-clip", 4, 3)]
    # The recorder hangs below the first element, in that element's clock.
    recorder = browser[0].find("asset-clip")
    assert (recorder.get("name"), recorder.get("lane"), recorder.get("offset")) == (
        "Recorder (repaired)", "-1", "0s")
    assert browser.find(".//adjust-volume") is None
    assert browser.find("sync-source/audio-role-source").get("active") == "0"
    clips = root.findall(".//project/sequence/spine/sync-clip")
    assert [(_parse_time(clip.get("offset")), _parse_time(clip.get("start")),
             _parse_time(clip.get("duration"))) for clip in clips] == [(0, 0, 3), (3, 5, 3)]
    assert clips[1][-1].tag == "sync-source"
    assert _parse_time(clips[1].find("marker").get("start")) == 6


def test_recorder_does_not_replace_voices_from_an_unplaced_recorder(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "work")
    project.assets = [Asset("cam", Path("cam.mov"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("cam", 0, 0, 2)]
    project.audio_warp_maps = [_warp("rec", "cam")]
    project.outputs = {"sync:cam": [_wav(tmp_path / "voice.wav", 2)]}
    sequence = build_sequence(project)
    assert [(clip.asset_id, clip.angle) for clip in sequence.clips] == [
        ("cam", "CAM"), ("voice-cam", "Voice")]


def test_affine_angles_trim_in_source_clock(tmp_path: Path) -> None:
    sequence = Sequence("Fast", Fraction(25), 1920, 1080, (
        SequenceClip("a", tmp_path / "a.mov", 2, 1, 0, 0, True, rate=0.5, angle="A"),
    ))
    path = write_multicam(sequence, tmp_path / "fast.fcpxml", media_base=tmp_path)
    root = ET.parse(path).getroot()
    points = root.findall("./resources/media/multicam/mc-angle/asset-clip/timeMap/timept")
    assert [_parse_time(point.get("time")) for point in points] == [1, 2]
    assert [_parse_time(point.get("value")) for point in points] == [2, 4]
    assert _parse_time(root.find("./resources/asset").get("duration")) == 4
