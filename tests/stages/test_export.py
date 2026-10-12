import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.stages.export import (
    Sequence,
    SequenceClip,
    SequenceMarker,
    _parse_time,
    build_sequence,
    fcpxml_intervals,
    write_fcpxml,
    write_xmeml,
    xmeml_intervals,
)


def test_fcpxml_round_trip_preserves_sequence_intervals(tmp_path: Path) -> None:
    source = tmp_path / "camera.mov"
    source.write_bytes(b"placeholder")
    sequence = Sequence(
        "episode",
        Fraction(25, 1),
        1920,
        1080,
        (
            SequenceClip("a", source, 2.0, 5.0, 1.0, 0, True),
            SequenceClip("b", source, 0.0, 3.0, 7.0, 1, True),
        ),
    )
    output = write_fcpxml(sequence, tmp_path / "episode.fcpxml", media_base=tmp_path)
    assert fcpxml_intervals(output) == [("camera", 1.0, 5.0), ("camera", 7.0, 3.0)]
    assert "camera.mov" in output.read_text(encoding="utf-8")


def test_xmeml_round_trip_preserves_sequence_intervals(tmp_path: Path) -> None:
    source = tmp_path / "camera clip.mov"
    source.write_bytes(b"placeholder")
    sequence = Sequence(
        "episode",
        Fraction(25, 1),
        1920,
        1080,
        (SequenceClip("a", source, 2.0, 5.0, 1.0, 0, True),),
    )
    output = write_xmeml(sequence, tmp_path / "episode.xml")
    assert xmeml_intervals(output) == [("camera clip", 1.0, 5.0)]
    text = output.read_text(encoding="utf-8")
    assert "<!DOCTYPE xmeml>" in text
    assert "file://localhost/" in text


def test_synced_voice_is_connected_and_original_audio_is_muted(tmp_path: Path) -> None:
    voice = tmp_path / "voice.wav"
    import wave

    with wave.open(str(voice), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "PCM"))
        audio.writeframes(b"\x00\x00" * 48000 * 5)
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mov"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("cam", 1.0, 0.0, 5.0)]
    project.assets[0].manual["media_info"] = {"audio_codec": "aac"}
    project.outputs["sync:cam"] = [voice]
    sequence = build_sequence(project)
    assert len(sequence.clips) == 2
    output = write_fcpxml(sequence, tmp_path / "out.fcpxml", media_base=tmp_path)
    root = ET.parse(output).getroot()
    assert root.find(".//spine/gap") is not None
    assert root.find(".//adjust-volume").get("amount") == "-96dB"
    assert any(asset.get("hasVideo") == "0" for asset in root.findall(".//asset"))
    xml = write_xmeml(sequence, tmp_path / "out.xml")
    assert ET.parse(xml).find(".//audio/track/clipitem/sourcetrack/mediatype").text == "audio"
    root = ET.parse(xml)
    camera_audio = [item for item in root.findall(".//audio/track/clipitem")
                    if item.findtext("name") == "camera"]
    assert len(camera_audio) == 2
    assert all(item.findtext("enabled") == "FALSE" for item in camera_audio)
    assert [item.findtext("sourcetrack/trackindex") for item in camera_audio] == ["1", "2"]
    assert all(item.find("link/linkclipref") is not None for item in camera_audio)


def test_edit_is_applied_identically_to_video_and_synced_voice(tmp_path: Path) -> None:
    voice = tmp_path / "voice.wav"
    import wave

    with wave.open(str(voice), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "PCM"))
        audio.writeframes(b"\x00\x00" * 48000 * 5)
    edit = tmp_path / "edit.json"
    edit.write_text(
        '{"mode":"cut","keep":[{"start":1,"end":3},{"start":5,"end":6}]}'
    )
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mov"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("cam", 1.0, 0.0, 5.0)]
    project.outputs = {"sync:cam": [voice], "roughcut": [edit]}
    sequence = build_sequence(project)
    videos = [clip for clip in sequence.clips if clip.has_video]
    voices = [clip for clip in sequence.clips if not clip.has_video]
    assert [(clip.source_in_s, clip.timeline_start_s, clip.duration_s) for clip in videos] == [
        (0.0, 0.0, 2.0), (4.0, 2.0, 1.0),
    ]
    assert [(clip.source_in_s, clip.timeline_start_s, clip.duration_s) for clip in voices] == [
        (0.0, 0.0, 2.0), (4.0, 2.0, 1.0),
    ]
    output = write_fcpxml(sequence, tmp_path / "out.fcpxml", media_base=tmp_path)
    assert len(ET.parse(output).findall(".//resources/asset")) == 2


def test_cut_and_filler_markers_use_edited_time_in_both_formats(tmp_path: Path) -> None:
    edit = tmp_path / "edit.json"
    edit.write_text(
        '{"mode":"cut","keep":[{"start":0,"end":2},{"start":4,"end":6}],'
        '"cuts":[{"start":2,"end":4,"reason":"pause","note":"quiet pause"}],'
        '"markers":[{"at":5,"kind":"filler","text":"um"}]}'
    )
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mov"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("cam", 0, 0, 6)]
    project.outputs["roughcut"] = [edit]
    sequence = build_sequence(project)
    assert sequence.markers == (
        SequenceMarker(2, "quiet pause", "pause"), SequenceMarker(3, "um", "filler"),
    )
    fcpxml = write_fcpxml(sequence, tmp_path / "out.fcpxml", media_base=tmp_path)
    assert [marker.get("start") for marker in ET.parse(fcpxml).findall(".//marker")] == [
        "50/25s", "75/25s",
    ]
    xml = write_xmeml(sequence, tmp_path / "out.xml")
    assert [marker.findtext("in") for marker in ET.parse(xml).findall(".//marker")] == [
        "50", "75",
    ]
    uncut = build_sequence(project, use_edit=False)
    assert not uncut.markers
    assert len(uncut.clips) == 1
    assert uncut.clips[0].duration_s == 6


def test_fcpxml_names_formats_and_links_media_from_the_document(tmp_path: Path) -> None:
    source = tmp_path / "MVI_1.mov"
    # A 25 fps file on a 29.97 timeline: rounding to timeline frames must not
    # make the clip, or the asset, longer than the file.
    sequence = Sequence("episode", Fraction(30000, 1001), 1920, 1080, (
        SequenceClip("mvi", source, 0, 486.08, 0, 0, True, source_fps=Fraction(25),
                     source_duration_s=486.08),
    ))
    output = write_fcpxml(sequence, tmp_path / "_studio" / "export" / "e.fcpxml",
                          media_base=tmp_path)
    root = ET.parse(output).getroot()
    formats = {item.get("id"): item for item in root.findall("./resources/format")}
    assert formats["r1"].get("name") == "FFVideoFormat1080p2997"
    asset = root.find("./resources/asset")
    assert formats[asset.get("format")].get("name") == "FFVideoFormat1080p25"
    assert asset.get("duration") == "12152/25s"
    assert asset.find("media-rep").get("src") == "../../MVI_1.mov"
    clip = root.find(".//asset-clip")
    assert _parse_time(clip.get("start")) + _parse_time(clip.get("duration")) <= 486.08


def test_marker_kinds_keep_only_cuts_by_request(tmp_path: Path) -> None:
    edit = tmp_path / "edit.json"
    edit.write_text(
        '{"mode":"cut","keep":[{"start":0,"end":2},{"start":4,"end":6}],'
        '"cuts":[{"start":2,"end":4,"reason":"pause","note":"quiet pause"}],'
        '"markers":[{"at":5,"kind":"unknown-sound","text":"untranscribed audio 0.6s"}]}'
    )
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("camera.mov"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("cam", 0, 0, 6)]
    project.outputs["roughcut"] = [edit]
    assert [marker.kind for marker in build_sequence(project, marker_kinds="cuts").markers] \
        == ["pause"]
    assert not build_sequence(project, marker_kinds="none").markers
    # The 2 s pause reaches the threshold; a longer threshold leaves it unmarked.
    assert build_sequence(project, marker_kinds="cuts", marker_min_s=2.0).markers
    assert not build_sequence(project, marker_kinds="cuts", marker_min_s=2.5).markers


def test_stage_links_media_from_where_the_runner_publishes(tmp_path: Path) -> None:
    from studio.stages.base import StageContext
    from studio.stages.export import ExportStage

    project = Project(tmp_path, tmp_path / "_studio")
    asset = Asset("cam", Path("camera.mov"), "video", AssetRole.CAMERA)
    asset.manual["media_info"] = {"fps": "25", "duration_s": 6}
    project.assets = [asset]
    project.placements = [SourcePlacement("cam", 0, 0, 6)]
    result = ExportStage().run(StageContext(project, {}, tmp_path / "runner" / "output"))
    fcpxml = next(path for path in result.artifacts if path.suffix == ".fcpxml")
    assert fcpxml == tmp_path / "runner" / "output" / "export" / "Studio.fcpxml"
    root = ET.parse(fcpxml).getroot()
    assert root.find(".//media-rep").get("src") == "../../camera.mov"
    assert root.find("./resources/media/multicam") is not None
    assert root.find("./library/event").get("name") == tmp_path.name
    assert root.find(".//project").get("name") == tmp_path.name
    named = ExportStage().run(StageContext(project, {"export": {"name": "Episode 7"}},
                                           tmp_path / "again"))
    root = ET.parse(next(path for path in named.artifacts if path.suffix == ".fcpxml")).getroot()
    assert root.find("./library/event").get("name") == "Episode 7"


def test_unplaced_files_are_listed_in_the_event(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "_studio")
    placed = Asset("cam", Path("camera.mov"), "video", AssetRole.CAMERA)
    placed.manual["media_info"] = {"fps": "25", "duration_s": 6}
    lost = Asset("lost", Path("lost.mov"), "video", AssetRole.CAMERA)
    lost.manual["media_info"] = {"fps": "25", "duration_s": 3, "audio_codec": "aac"}
    false_start = Asset("short", Path("short.mov"), "video", AssetRole.IGNORE)
    false_start.manual["media_info"] = {"fps": "25", "duration_s": 1}
    project.assets = [placed, lost, false_start]
    project.placements = [SourcePlacement("cam", 0, 0, 6)]
    sequence = build_sequence(project)
    assert [clip.asset_id for clip in sequence.unplaced] == ["lost", "short"]
    root = ET.parse(write_fcpxml(sequence, tmp_path / "e.fcpxml", media_base=tmp_path)).getroot()
    event = root.find("./library/event")
    listed = {clip.get("name"): clip for clip in event.findall("asset-clip")}
    assert set(listed) == {"camera", "lost", "short"}
    assert listed["lost"].find("keyword").get("value") == "Not synchronized"
    assert listed["camera"].find("keyword") is None
    assert [clip.get("name") for clip in root.findall(".//spine//asset-clip")] == ["camera"]
