from pathlib import Path

import pytest

from tools.make_fixtures import generate


def test_generate_requires_force_for_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    output.mkdir()
    with pytest.raises(FileExistsError, match="--force"):
        generate(output)


def test_generate_creates_expected_media_tree(tmp_path: Path) -> None:
    output = generate(tmp_path / "fixture")
    assert (output / "fixture.json").is_file()
    assert (output / "camera-a" / "DJI_0001.MP4").stat().st_size > 0
    assert (output / "camera-b" / "GX010024.MP4").stat().st_size > 0
    assert (output / "camera-b" / "GX020024.MP4").stat().st_size > 0
    assert (output / "recorder" / "ZOOM0001_Tr1.WAV").stat().st_size > 0


def test_defect_scenarios_cover_the_pos17v2_findings(tmp_path: Path) -> None:
    import wave

    from studio.core.channels import analyze_channels
    from tools.make_fixtures import DEFECT_SCENARIOS, generate_defects

    output = generate_defects(tmp_path / "defects")
    for name in DEFECT_SCENARIOS:
        assert (output / name / "camera" / "DJI_0909.MOV").stat().st_size > 0
        assert (output / name / "recorder" / "2026-10-05_103823465.wav").is_file()
    dual = output / "dual-mono" / "recorder" / "2026-10-05_103823465.wav"
    with wave.open(str(dual), "rb") as handle:
        duration = handle.getnframes() / handle.getframerate()
    analysis = analyze_channels(dual, channels=2, duration_s=duration)
    assert analysis is not None and analysis["class"] == "dual_mono"
