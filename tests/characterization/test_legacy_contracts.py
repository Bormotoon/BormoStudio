import hashlib
import json
import subprocess
import sys
import wave
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest
from podcast_reels_forge.utils.fingerprint import StageState, fingerprint
from whispersync.engine.export import check_fcpxml, fcpxml_intervals, generate_fcpxml
from whispersync.engine.media import MediaInfo
from whispersync.engine.proc import run_logged
from whispersync.engine.transcript_export import dump_srt, transcript_to_dict
from whispersync.engine.workspace import OutputLockedError, output_lock, published
from whispersync.models import MediaClip, Segment, SyncPlan, Transcript, Word

GOLDEN = Path(__file__).parent / "golden"


def test_legacy_transcript_json_and_srt_contract(tmp_path: Path) -> None:
    transcript = Transcript(Path("recorder.wav"), "en", 5.0, [
        Segment(0.1254, 1.5, [
            Word("Hello", 0.1254, 0.75, 0.9876), Word("world.", 0.8, 1.5, 0.9),
        ]),
        Segment(3.0, 4.0, [Word("Again", 3.0, 4.0, 0.8)]),
    ])
    actual = transcript_to_dict(
        transcript, audio_path=Path("recorder.wav"), model="tiny",
        device="cpu", compute_type="float32", mode="fast",
    )
    assert actual == json.loads((GOLDEN / "transcript.json").read_text())
    srt = tmp_path / "transcript.srt"
    dump_srt(srt, transcript)
    assert srt.read_text() == (
        "1\n00:00:00,125 --> 00:00:01,500\nHello world.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nAgain\n"
    )
    from studio.core.transcript import Transcript as StudioTranscript
    from studio.core.transcript import to_srt

    converted = StudioTranscript.from_dict(actual)
    assert converted.to_dict()["sentences"] == actual["sentences"]
    assert to_srt(converted) == srt.read_text()


def test_legacy_export_overlap_staging_and_replacement_audio(tmp_path: Path) -> None:
    final = tmp_path / "project"
    (final / "audio").mkdir(parents=True)
    voice = final / "audio" / "voice.wav"
    with wave.open(str(voice), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000 * 8)
    infos = []
    clips = []
    for name, start, duration in [("a", 0.0, 10.0), ("b", 3.0, 4.0), ("c", 12.0, 5.0)]:
        path = tmp_path / f"{name}.mov"
        infos.append(MediaInfo(
            path, duration, Fraction(25), 1920, 1080, "h264", "aac", 2, 48000,
            container_duration=duration,
        ))
        clips.append(MediaClip(
            path, "video", start, 0.0, duration, 1, source_audio_enabled=False,
        ))
    clips.append(MediaClip(voice, "audio", 1.0, 0.0, 8.0, -1, role="Dialogue"))
    output = tmp_path / "staged.fcpxml"
    generate_fcpxml(SyncPlan(3, clips, 17.0), infos, output, media_base_dir=final)
    root = ET.parse(output).getroot()
    actual = {
        "intervals": {name: list(interval) for name, interval in fcpxml_intervals(output).items()},
        "spine": [element.get("name") for element in root.findall(".//spine/asset-clip")],
        "promoted_lane": root.find(".//asset-clip[@name='b']").get("lane"),
        "source_audio": root.find(".//asset-clip[@name='a']").get("srcEnable"),
        "voice_src": root.find(".//asset[@name='voice']/media-rep").get("src"),
        "frame_duration": root.find(".//format").get("frameDuration"),
    }
    assert actual == json.loads((GOLDEN / "overlap.json").read_text())
    assert check_fcpxml(output, check_media=False) == []


def test_studio_export_intervals_match_legacy_overlap_fixture(tmp_path: Path) -> None:
    from studio.stages.export import Sequence, SequenceClip, write_fcpxml

    clips = tuple(SequenceClip(name, tmp_path / f"{name}.mov", 0, duration, start, 0, False)
                  for name, start, duration in [("a", 0, 10), ("b", 3, 4), ("c", 12, 5)])
    output = write_fcpxml(Sequence("Fixture", Fraction(25), 1920, 1080, clips),
                          tmp_path / "studio.fcpxml", media_base=tmp_path)
    actual = fcpxml_intervals(output)
    expected = json.loads((GOLDEN / "overlap.json").read_text())["intervals"]
    for name in ("a", "b", "c"):
        assert list(actual[name]) == expected[name]


def test_legacy_fingerprint_bytes_and_adoption_policy(tmp_path: Path) -> None:
    serialized = '[{"a": 1, "b": 2}, "prompt"]'
    expected = hashlib.sha256(serialized.encode()).hexdigest()[:32]
    assert fingerprint({"b": 2, "a": 1}, "prompt") == expected
    state = StageState(tmp_path)
    # Characterize adoption, but do not inherit it: Studio must verify artifacts.
    assert state.decide("sync", expected) == "adopted"
    assert state.decide("sync", expected) == "same"
    assert state.decide("sync", "different") == "changed"
    assert state.get("sync") == expected


def test_legacy_lock_and_cancelled_publication(tmp_path: Path) -> None:
    result = tmp_path / "result"
    result.write_bytes(b"last complete result")
    with output_lock(tmp_path):
        with pytest.raises(OutputLockedError), output_lock(tmp_path):
            pass
        with pytest.raises(KeyboardInterrupt), published(result) as staging:
            staging.write_bytes(b"partial")
            raise KeyboardInterrupt()
    assert result.read_bytes() == b"last complete result"
    assert not (tmp_path / ".whispersync-run.lock").exists()
    assert not staging.exists()


def test_legacy_process_error_and_bounded_log_contract(tmp_path: Path) -> None:
    result = run_logged(
        [sys.executable, "-c", "import sys; print('x' * 10000); print('failure', file=sys.stderr);"
         "sys.exit(7)"],
        timeout=5, log_dir=tmp_path, tail_bytes=32, keep_logs=True,
    )
    assert result.returncode == 7
    assert result.stdout == "x" * 31 + "\n"
    assert result.stderr == "failure\n"
    assert result.stdout_path.stat().st_size > 10000
    assert result.stderr_path.read_text() == "failure\n"
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.1, log_dir=tmp_path / "timeout",
        )
    assert list((tmp_path / "timeout").iterdir()) == []
