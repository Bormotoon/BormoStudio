import subprocess
from pathlib import Path

import numpy as np

from studio.core.media import probe
from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.stages.sync_media import selected_media


def test_default_nonfirst_stream_is_used_and_private_decode_is_removed(tmp_path: Path) -> None:
    source = tmp_path / "multi.mkv"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=s=160x120:r=25:d=1", "-f", "lavfi", "-i",
        "anullsrc=r=48000:cl=mono", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=1", "-map", "0:v", "-map", "1:a", "-map", str(2) + ":a",
        "-t", "1", "-c:v", "libx264", "-c:a", "pcm_s16le",
        "-disposition:a:0", "0", "-disposition:a:1", "default", str(source),
    ], check=True)
    info = probe(source)
    assert info.audio_stream_index == 2
    asset = Asset("cam", source, AssetKind.VIDEO, AssetRole.CAMERA,
                  manual={"media_info": info.to_dict()})
    project = Project(tmp_path, tmp_path / "work", assets=[asset])
    with selected_media(project, tmp_path) as selected:
        decoded = selected.assets[0].path
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(decoded), "-f", "f32le", "-",
        ], capture_output=True, check=True)
        assert np.max(np.abs(np.frombuffer(result.stdout, dtype="<f4"))) > 0.1
    assert not decoded.exists()
    assert project.assets[0].path == source
    assert source.is_file()
    with selected_media(project, tmp_path, {"cam": 1}) as selected:
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(selected.assets[0].path),
            "-f", "f32le", "-",
        ], capture_output=True, check=True)
        assert np.max(np.abs(np.frombuffer(result.stdout, dtype="<f4"))) == 0
    from studio.core.timeline import SourcePlacement
    from studio.stages.base import StageContext
    from studio.stages.export import ExportStage

    project.work_dir.mkdir()
    project.placements = [SourcePlacement("cam", 0, 0, 1)]
    exported = ExportStage().run(StageContext(project, {}, project.work_dir))
    audio_path = next(path for path in exported.artifacts if path.suffix == ".wav")
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(audio_path), "-f", "f32le", "-",
    ], capture_output=True, check=True)
    assert np.max(np.abs(np.frombuffer(result.stdout, dtype="<f4"))) > 0.1
    import xml.etree.ElementTree as ET

    xml_path = next(path for path in exported.artifacts if path.suffix == ".xml")
    root = ET.parse(xml_path).getroot()
    assert any(audio_path.as_uri() == (node.text or "").replace("file://localhost/", "file:///")
               for node in root.findall(".//pathurl"))
