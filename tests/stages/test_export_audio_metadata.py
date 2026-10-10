import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.stages.export import build_sequence, write_fcpxml, write_xmeml


def test_rendered_wav_format_and_roles_survive_export(tmp_path: Path) -> None:
    voice, room = tmp_path / "voice.wav", tmp_path / "room.wav"
    for path, channels, rate, codec in [(voice, 1, 44100, "pcm_s24le"),
                                       (room, 2, 48000, "pcm_s16le")]:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1", "-ac", str(channels), "-ar", str(rate),
                        "-c:a", codec, str(path)], check=True)
    project = Project(tmp_path, tmp_path / "work")
    camera = Asset("cam", Path("cam.mov"), "video", AssetRole.CAMERA)
    camera.manual["media_info"] = {"fps": "25", "audio_channels": 4, "audio_codec": "aac"}
    project.assets = [camera]
    project.placements = [SourcePlacement("cam", 0, 0, 1)]
    project.outputs = {"sync:cam": [voice], "ambience:cam": [room]}
    sequence = build_sequence(project, retimed=True)
    voice_clip = next(clip for clip in sequence.clips if clip.asset_id == "voice-cam")
    assert (voice_clip.audio_channels, voice_clip.audio_sample_rate, voice_clip.audio_depth) == (
        1, 44100, 24,
    )
    xml = write_fcpxml(sequence, tmp_path / "out.fcpxml", media_base=tmp_path)
    root = ET.parse(xml).getroot()
    asset = next(asset for asset in root.findall("resources/asset") if asset.get("name") == "voice")
    assert (asset.get("audioChannels"), asset.get("audioRate")) == ("1", "44100")
    roles = {clip.get("name"): clip.get("audioRole")
             for clip in root.findall(".//spine/gap/asset-clip")}
    assert roles["voice"] == "dialogue"
    assert roles["room"] == "effects"
    xmeml = write_xmeml(sequence, tmp_path / "out.xml")
    root = ET.parse(xmeml).getroot()
    item = next(item for item in root.findall(".//clipitem") if item.findtext("name") == "voice")
    assert item.findtext("file/media/audio/channelcount") == "1"
    assert item.findtext("file/media/audio/samplecharacteristics/depth") == "24"
    assert item.findtext("file/media/audio/samplecharacteristics/samplerate") == "44100"


def test_camera_scan_preserves_pcm_depth_and_leaves_aac_unknown(tmp_path: Path) -> None:
    from studio.stages.scan import scan

    for name, codec in [("pcm", "pcm_s24le"), ("compressed", "aac")]:
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=s=160x120:r=25:d=1", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=1", "-c:v", "libx264", "-c:a", codec,
            "-ar", "44100", str(tmp_path / f"{name}.mov"),
        ], check=True)
    assets, _ = scan(tmp_path)
    by_name = {asset.path.stem: asset for asset in assets}
    assert by_name["pcm"].manual["media_info"]["audio_bits_per_sample"] == 24
    assert by_name["compressed"].manual["media_info"]["audio_bits_per_sample"] is None
    project = Project(tmp_path, tmp_path / "work", assets=assets)
    project.placements = [SourcePlacement(asset.id, 0, 0, 1) for asset in assets]
    path = write_xmeml(build_sequence(project, retimed=True), tmp_path / "camera.xml")
    root = ET.parse(path).getroot()
    items = {item.findtext("name"): item for item in root.findall(".//video/track/clipitem")}
    assert items["pcm"].findtext("file/media/audio/samplecharacteristics/depth") == "24"
    assert items["compressed"].find("file/media/audio/samplecharacteristics/depth") is None
    assert items["pcm"].findtext("file/media/audio/samplecharacteristics/samplerate") == "44100"


def test_depth_comes_from_default_audio_stream(tmp_path: Path) -> None:
    import json
    from unittest.mock import patch

    from studio.core.media import probe

    data = {"format": {"duration": "1"}, "streams": [
        {"codec_type": "audio", "bits_per_sample": 16, "sample_rate": "48000"},
        {"codec_type": "audio", "bits_per_sample": 32, "bits_per_raw_sample": "24",
         "sample_rate": "44100", "disposition": {"default": 1}},
    ]}
    with patch("studio.core.media.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = json.dumps(data)
        info = probe(tmp_path / "multi.mov")
    assert info.audio_stream_index == 1
    assert info.audio_sample_rate == 44100
    assert info.audio_bits_per_sample == 24


def test_probe_uses_absolute_stream_index(tmp_path: Path) -> None:
    import json
    from unittest.mock import patch

    from studio.core.media import probe

    data = {"format": {"duration": "1"}, "streams": [
        {"index": 0, "codec_type": "video"},
        {"index": 1, "codec_type": "audio", "channels": 1},
        {"index": 3, "codec_type": "audio", "channels": 2,
         "disposition": {"default": 1}},
    ]}
    with patch("studio.core.media.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = json.dumps(data)
        info = probe(tmp_path / "multi.mov")
    assert info.audio_stream_index == 3
    assert info.audio_channels == 2
