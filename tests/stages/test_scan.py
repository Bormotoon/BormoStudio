from pathlib import Path

from studio.core.project import Asset, AssetKind, AssetRole
from studio.stages.scan import apply_manual_overrides, ignore_short, scan
from tools.make_fixtures import generate


def test_scan_classifies_fixture_and_links_gopro_chapters(tmp_path: Path) -> None:
    source = generate(tmp_path / "fixture")
    assets, warnings = scan(source)
    assert warnings == []
    assert [asset.role for asset in assets].count(AssetRole.CAMERA) == 3
    assert [asset.role for asset in assets].count(AssetRole.RECORDER) == 1
    chapters = [asset for asset in assets if asset.path.parent.name == "camera-b"]
    assert chapters[0].chapter_of is None
    assert chapters[1].chapter_of == chapters[0].id
    assert all(asset.manual["media_info"]["duration_s"] > 0 for asset in assets)


def test_scan_keeps_explicit_manual_overrides() -> None:
    previous = [
        Asset(
            "camera-a",
            Path("camera-a.mp4"),
            AssetKind.VIDEO,
            AssetRole.CAMERA,
            manual={
                "media_info": {"duration_s": 1},
                "overrides": {"role": "ignore", "group_id": "wide", "device": "Manual"},
            },
        )
    ]
    scanned = [
        Asset(
            "camera-a",
            Path("camera-a.mp4"),
            AssetKind.VIDEO,
            AssetRole.CAMERA,
            device="Detected",
            group_id="detected",
            manual={"media_info": {"duration_s": 2}},
        )
    ]

    result = apply_manual_overrides(previous, scanned)

    assert result[0].role is AssetRole.IGNORE
    assert result[0].group_id == "wide"
    assert result[0].device == "Manual"
    assert result[0].manual == {
        "media_info": {"duration_s": 2},
        "overrides": {"role": "ignore", "group_id": "wide", "device": "Manual"},
    }


def _clip(name: str, duration: float, role: AssetRole = AssetRole.CAMERA) -> Asset:
    return Asset(name, Path(f"{name}.mov"), AssetKind.VIDEO, role,
                 manual={"media_info": {"duration_s": duration}})


def test_false_starts_shorter_than_the_threshold_are_ignored() -> None:
    # sony_run MVI_0851: a camera switched on and off by mistake (1.08 s).
    assets = [_clip("mvi-0851", 1.08), _clip("mvi-0849", 5.52), _clip("broken", 0.0)]
    warnings = ignore_short(assets, 3.0)
    assert [asset.role for asset in assets] == [AssetRole.IGNORE, AssetRole.CAMERA,
                                                AssetRole.CAMERA]
    assert len(warnings) == 1 and "mvi-0851.mov: 1.1 s" in warnings[0]
    assert ignore_short([_clip("mvi-0851", 1.08)], 0.0) == []


def test_a_manual_role_brings_a_short_clip_back() -> None:
    previous = [_clip("mvi-0851", 1.08)]
    previous[0].manual["overrides"] = {"role": "camera"}
    scanned = [_clip("mvi-0851", 1.08)]
    ignore_short(scanned, 3.0)
    assert apply_manual_overrides(previous, scanned)[0].role is AssetRole.CAMERA
