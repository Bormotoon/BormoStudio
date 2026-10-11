"""Media discovery, classification, stable IDs, grouping, and chapters."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from studio.core.media import MediaInfo, probe
from studio.core.project import Asset, AssetKind, AssetRole, Project, stable_fingerprint
from studio.core.recorded_at import recorded_at
from studio.stages.base import (
    Decision,
    GpuUse,
    Requirement,
    StageContext,
    StageOutput,
    select_settings,
)

MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".wav", ".flac", ".mp3", ".m4a"}
IGNORED_SUFFIXES = {".lrf", ".lrv", ".thm"}
GOPRO_CHAPTER = re.compile(r"^G[HX](\d{2})(\d{4})$", re.IGNORECASE)


class ScanStage:
    id = "scan"
    title = "Material scan"
    after: tuple[str, ...] = ()
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ("scan.grouping", "scan.channel_analysis",
                                      "scan.min_duration_s")
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return [Requirement("binary", "ffprobe", "read media metadata")]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        return Decision.run() if project.source_dir.is_dir() else Decision.blocked(
            "source directory does not exist", "choose an existing directory"
        )

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        files = _media_paths(project.source_dir)
        identities = [
            (
                str(path.relative_to(project.source_dir)),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in files
        ]
        overrides = {
            asset.id: asset.manual.get("overrides", {})
            for asset in project.assets
            if asset.manual.get("overrides")
        }
        return stable_fingerprint("scan-v7", identities, overrides,
                                  select_settings(settings, self.settings_keys))

    def run(self, context: StageContext) -> StageOutput:
        assets, warnings = scan(context.project.source_dir)
        conf = context.settings.get("scan", {})
        conf = conf if isinstance(conf, dict) else {}
        warnings += ignore_short(assets, float(conf.get("min_duration_s", 3.0)))
        assets = apply_manual_overrides(context.project.assets, assets)
        if conf.get("channel_analysis", "auto") != "off":
            analyze_assets(context.project.source_dir, assets)
        report = context.work_dir / "stages" / "scan" / "report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        import json

        data = {"assets": [_asset_json(asset) for asset in assets], "warnings": warnings}
        report.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )
        from studio.core.recorder_tracks import refresh_tracks

        tracks = refresh_tracks(assets, context.project.recorder_tracks)
        return StageOutput((report,), {"assets": assets, "recorder_tracks": tracks},
                           tuple(warnings))


def scan(source_dir: Path) -> tuple[list[Asset], list[str]]:
    paths = _media_paths(source_dir)
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as pool:
        infos = list(pool.map(probe, paths))
    records = list(zip(paths, infos, strict=True))
    ids = _stable_ids([path for path, _ in records], source_dir)
    assets: list[Asset] = []
    warnings: list[str] = []
    chapter_roots: dict[str, str] = {}
    for path, info in records:
        relative = path.relative_to(source_dir)
        kind = (
            AssetKind.VIDEO
            if info.video_codec
            else AssetKind.AUDIO
            if info.audio_codec
            else AssetKind.OTHER
        )
        role = (
            AssetRole.CAMERA
            if kind is AssetKind.VIDEO
            else AssetRole.RECORDER
            if kind is AssetKind.AUDIO
            else AssetRole.IGNORE
        )
        group = relative.parent.name if relative.parent != Path(".") else _device_group(path, info)
        chapter_key = _gopro_key(path)
        chapter_of = chapter_roots.setdefault(chapter_key, ids[path]) if chapter_key else None
        if chapter_of == ids[path]:
            chapter_of = None
        asset = Asset(
            ids[path],
            relative,
            kind,
            role,
            _device_name(info),
            group,
            chapter_of,
            {"media_info": {
                **info.to_dict(),
                "recorded_at": recorded_at(path, info.tags, info.duration_s,
                                           info.audio_sample_rate),
            }},
        )
        assets.append(asset)
        if kind is AssetKind.VIDEO and not info.audio_codec:
            warnings.append(f"{relative}: video has no audio; acoustic sync is unavailable")
    return assets, warnings


def ignore_short(assets: list[Asset], min_duration_s: float) -> list[str]:
    """Ignore files shorter than ``min_duration_s``: false starts, not material.

    A second-long clip (a camera switched on and off by mistake) has nothing to
    place on a recorder and nothing to edit. A manual role override still wins.
    """
    warnings = []
    for asset in assets:
        duration = float(asset.manual.get("media_info", {}).get("duration_s") or 0.0)
        if asset.role is AssetRole.IGNORE or not 0 < duration < min_duration_s:
            continue
        asset.role = AssetRole.IGNORE
        warnings.append(f"{asset.path}: {duration:.1f} s, shorter than scan.min_duration_s "
                        f"({min_duration_s:g} s); ignored as a false start")
    return warnings


def analyze_assets(source_dir: Path, assets: list[Asset]) -> None:
    """Add ``media_info.channel_analysis`` to audio files with two or more channels."""
    from studio.core.channels import analyze_channels

    candidates = [
        asset for asset in assets
        if asset.kind is AssetKind.AUDIO
        and int(asset.manual.get("media_info", {}).get("audio_channels") or 0) >= 2
    ]

    def analyze(asset: Asset) -> None:
        info = asset.manual["media_info"]
        analysis = analyze_channels(
            source_dir / asset.path, channels=int(info["audio_channels"]),
            duration_s=float(info.get("duration_s") or 0.0),
            stream_index=info.get("audio_stream_index"),
        )
        if analysis is not None:
            info["channel_analysis"] = analysis

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(candidates)))) as pool:
        list(pool.map(analyze, candidates))


def apply_manual_overrides(previous: list[Asset], scanned: list[Asset]) -> list[Asset]:
    """Keep explicit user choices while refreshing probe-derived asset metadata."""
    overrides = {
        asset.id: asset.manual.get("overrides")
        for asset in previous
        if isinstance(asset.manual.get("overrides"), dict)
    }
    for asset in scanned:
        values = overrides.get(asset.id)
        if not values:
            continue
        role = values.get("role")
        group_id = values.get("group_id")
        device = values.get("device")
        if isinstance(role, str):
            try:
                asset.role = AssetRole(role)
            except ValueError:
                pass
        if isinstance(group_id, str) or group_id is None:
            asset.group_id = group_id
        if isinstance(device, str) or device is None:
            asset.device = device
        asset.manual["overrides"] = values
    return scanned


def _media_paths(source_dir: Path) -> list[Path]:
    return sorted(
        (
            path for path in source_dir.rglob("*")
            if path.is_file()
            and not any(
                part.startswith(".") or part == "_studio"
                for part in path.relative_to(source_dir).parts
            )
            and path.suffix.lower() in MEDIA_SUFFIXES
            and path.suffix.lower() not in IGNORED_SUFFIXES
        ),
        key=lambda path: _natural_key(str(path.relative_to(source_dir))),
    )


def _stable_ids(paths: list[Path], source_dir: Path) -> dict[Path, str]:
    used: set[str] = set()
    result: dict[Path, str] = {}
    for path in paths:
        relative_stem = str(path.relative_to(source_dir).with_suffix("")).lower()
        base = re.sub(r"[^a-z0-9]+", "-", relative_stem).strip("-") or "source"
        candidate = base
        number = 2
        while candidate in used:
            candidate = f"{base}-{number}"
            number += 1
        used.add(candidate)
        result[path] = candidate
    return result


def _device_group(path: Path, info: MediaInfo) -> str:
    return _device_name(info) or path.stem.split("_")[0].lower()


def _device_name(info: MediaInfo) -> str | None:
    make = info.tags.get("com.apple.quicktime.make") or info.tags.get("make")
    model = info.tags.get("com.apple.quicktime.model") or info.tags.get("model")
    return " ".join(part for part in (make, model) if part) or None


def _gopro_key(path: Path) -> str | None:
    match = GOPRO_CHAPTER.match(path.stem)
    return f"gopro-{match.group(2)}" if match else None


def _natural_key(value: str) -> list[tuple[int, int, str]]:
    return [
        (0, int(part), "") if part.isdigit() else (1, 0, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    ]


def _asset_json(asset: Asset) -> dict[str, object]:
    return {
        "id": asset.id, "path": str(asset.path), "kind": asset.kind.value,
        "role": asset.role.value, "device": asset.device, "group_id": asset.group_id,
        "chapter_of": asset.chapter_of, "manual": asset.manual,
    }
