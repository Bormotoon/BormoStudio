import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.stages.base import StageContext
from studio.stages.export import ExportStage, _parse_time


@pytest.mark.parametrize("rate", [0.5, 1.25, 1.0])
def test_flat_exports_retime_all_media_after_roughcut(tmp_path: Path, rate: float) -> None:
    project = Project(tmp_path, tmp_path / "work")
    asset = Asset("cam", Path("camera.mov"), "video", AssetRole.CAMERA)
    asset.manual["media_info"] = {"fps": "25", "audio_codec": "aac", "audio_channels": 2}
    project.assets = [asset]
    project.placements = [SourcePlacement("cam", 2, 1, 8, rate)]
    voice, room = tmp_path / "voice.wav", tmp_path / "room.wav"
    import wave

    for path in (voice, room):
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 48000, 0, "NONE", "PCM"))
            audio.writeframes(b"\x00\x00" * 48000 * 10)
    start, end = 2 + rate, 2 + 3 * rate
    edit = tmp_path / "edit.json"
    edit.write_text(json.dumps({"mode": "cut", "keep": [{"start": start, "end": end}]}))
    project.outputs = {"sync:cam": [voice], "ambience:cam": [room], "roughcut": [edit]}
    result = ExportStage().run(StageContext(project, {"export": {
        "targets": ["fcpxml", "xmeml"],
    }}, project.work_dir))
    fcpxml = ET.parse(result.artifacts[0]).getroot()
    clips = fcpxml.findall(".//spine/gap/asset-clip")
    assert len(clips) == 3
    for clip in clips:
        assert _parse_time(clip.get("duration")) == pytest.approx(round(50 * rate) / 25)
        points = clip.findall("timeMap/timept")
        if rate == 1:
            assert not points
            assert _parse_time(clip.get("start")) == 2
        else:
            assert [_parse_time(point.get("value")) for point in points] == [2, 4]
            assert _parse_time(points[-1].get("time")) - _parse_time(
                points[0].get("time")) == pytest.approx(_parse_time(clip.get("duration")))
    xmeml = ET.parse(result.artifacts[1]).getroot()
    items = xmeml.findall(".//clipitem")
    assert len(items) == 5  # video, two linked original channels, voice, ambience
    for item in items:
        assert int(item.findtext("start")) == 0
        assert int(item.findtext("end")) == round(50 * rate)
        assert int(item.findtext("in")) == 50
        assert int(item.findtext("out")) == 100
        assert int(item.findtext("duration")) == round(50 * rate)
        parameters = {p.findtext("parameterid"): p.findtext("value")
                      for p in item.findall("filter/effect/parameter")}
        if rate == 1:
            assert not parameters
        else:
            assert float(parameters["speed"]) == pytest.approx(100 / rate)
            assert parameters["variablespeed"] == "FALSE"
    linked_audio = [item for item in items if item.get("id").startswith("a-")]
    assert all(item.findtext("enabled") == "FALSE" for item in linked_audio)
    assert all(item.findtext("filter/effect/mediatype") == "audio"
               for item in linked_audio if rate != 1)


def test_mixed_source_fps_use_distinct_source_and_sequence_clocks(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "work")
    for name, fps, width in [("a", "25", 1920), ("b", "30000/1001", 1280)]:
        asset = Asset(name, Path(f"{name}.mov"), "video", AssetRole.CAMERA)
        asset.manual["media_info"] = {"fps": fps, "width": width, "height": 720,
                                      "audio_codec": "aac", "audio_channels": 1}
        project.assets.append(asset)
        project.placements.append(SourcePlacement(name, 0, 2, 4, 1.25))
    result = ExportStage().run(StageContext(project, {"export": {
        "targets": ["fcpxml", "xmeml", "multicam"],
    }}, project.work_dir))
    fcpxml_path = next(path for path in result.artifacts if path.name == "Studio.fcpxml")
    root = ET.parse(fcpxml_path).getroot()
    formats = {element.get("id"): element for element in root.findall("resources/format")}
    asset_b = next(asset for asset in root.findall("resources/asset") if asset.get("name") == "b")
    fmt = formats[asset_b.get("format")]
    assert fmt.get("frameDuration") == "1001/30000s"
    assert fmt.get("width") == "1280"
    xmeml_path = next(path for path in result.artifacts if path.suffix == ".xml")
    xmeml = ET.parse(xmeml_path).getroot()
    b = next(item for item in xmeml.findall(".//video/track/clipitem")
             if item.findtext("name") == "b")
    assert b.findtext("start") == "0"
    assert b.findtext("end") == "125"  # five seconds in the 25 fps sequence
    assert b.findtext("in") == "60"
    assert b.findtext("out") == "180"  # source seconds 2..6 at 30000/1001
    assert b.findtext("rate/timebase") == "30"
    assert b.findtext("rate/ntsc") == "TRUE"
    assert b.findtext("file/media/video/samplecharacteristics/width") == "1280"
    linked = next(item for item in xmeml.findall(".//audio/track/clipitem")
                  if item.findtext("name") == "b")
    assert linked.findtext("in") == "60"
    assert linked.findtext("rate/ntsc") == "TRUE"
