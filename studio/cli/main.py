"""Headless Studio commands."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from studio import __version__
from studio.core.digest import configure_from_settings, set_rehash
from studio.core.project import AssetRole, Project
from studio.core.publication import recover_publications
from studio.core.settings import SettingsError, load_settings
from studio.core.workspace import project_lock
from studio.llm.session import BatchLlamaSession, ManagedLlamaSession, effective_llm_settings
from studio.modules.cancellation import InstallationCancelled
from studio.modules.manager import ModuleManager, module_names
from studio.stages.base import stage_uses_llm
from studio.stages.events import EventType, StageEvent
from studio.stages.fetch import fetch_to_directory
from studio.stages.planner import build_plan, fingerprint_parts
from studio.stages.runner import StageProcessError, run_stage_process
from studio.stages.worker import stage_registry


def _project(source: Path) -> tuple[Project, Path, Path]:
    source = source.resolve()
    work = source / "_studio"
    project_path = work / "project.json"
    settings_path = work / "settings.yaml"
    if project_path.is_file():
        project = Project.load(project_path)
    else:
        project = Project(source, work, producer_version=__version__)
        project.save(project_path)
    if not settings_path.exists():
        settings_path.write_text("sync:\n  mode: auto\n", encoding="utf-8")
    return project, project_path, settings_path


def _settings(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    settings = effective_llm_settings(load_settings([path], overrides).to_dict())
    configure_from_settings(settings)
    return settings


def _run_discovery(
    source: Path,
    *,
    include_prepare: bool,
    overrides: list[str] | None = None,
    cancel_file: Path | None = None,
) -> Project:
    project, project_path, settings_path = _project(source)
    settings = _settings(settings_path, overrides)
    selected = [stage_registry()["scan"]]
    if include_prepare:
        selected.append(stage_registry()["prepare"])
    with project_lock(project.work_dir):
        recover_publications(project.work_dir)
        for stage in selected:
            if cancel_file is not None and cancel_file.exists():
                raise InstallationCancelled("Discovery cancelled.")
            project = Project.load(project_path)
            fingerprint = stage.fingerprint(project, settings)
            try:
                result = run_stage_process(
                    stage.id,
                    project_path,
                    settings_path,
                    fingerprint,
                    effective_settings=settings,
                    fingerprint_parts=fingerprint_parts(stage, project, settings),
                    cancelled=lambda: cancel_file is not None and cancel_file.exists(),
                )
            except StageProcessError:
                if cancel_file is not None and cancel_file.exists():
                    raise InstallationCancelled("Discovery cancelled.") from None
                raise
            if not result.manifest.reusable(project.work_dir, fingerprint):
                raise RuntimeError(f"stage {stage.id} did not publish a reusable result")
    return Project.load(project_path)


def _run_pipeline(
    source: Path,
    overrides: list[str] | None = None,
    *,
    only: set[str] | None = None,
    skip: set[str] | None = None,
    stream: bool = False,
    cancel_file: Path | None = None,
    batch_session: BatchLlamaSession | None = None,
) -> Project:
    project, project_path, settings_path = _project(source)
    settings = _settings(settings_path, overrides)
    registry = stage_registry()
    unknown = ((only or set()) | (skip or set())) - registry.keys()
    if unknown:
        raise SettingsError(f"unknown stages: {sorted(unknown)}")

    def forward(event: StageEvent) -> None:
        print(event.to_json(), flush=True)

    def report(event: StageEvent) -> None:
        if stream:
            forward(event)
        elif event.type is EventType.WARNING:
            print(f"{event.stage}: {event.payload.get('message', '')}", file=sys.stderr)

    with project_lock(project.work_dir), ExitStack() as cleanup:
        if batch_session is None:
            session = ManagedLlamaSession(
                settings.get("llm", {}),
                project.work_dir,
                lambda: cancel_file is not None and cancel_file.exists(),
            )
            cleanup.callback(session.close)
        else:
            session = batch_session.select(settings, project.work_dir)
        recover_publications(project.work_dir)
        failures: dict[str, str] = {}
        stale_reasons: dict[str, str] = {}
        cancelled_run = False
        while True:
            if cancel_file is not None and cancel_file.exists():
                cancelled_run = True
                break
            project = Project.load(project_path)
            plan = build_plan(
                registry.values(),
                project,
                settings,
                only=only,
                skip=(skip or set()) | failures.keys(),
            )
            candidate = next(
                (
                    item
                    for item in plan.stages
                    if item.decision.kind.value == "run" and not item.will_reuse
                ),
                None,
            )
            if candidate is None:
                break
            if candidate.stale_reason:
                stale_reasons[candidate.stage_id] = candidate.stale_reason
            required = [item for item in candidate.missing if not item.optional]
            if required:
                failures[candidate.stage_id] = "missing requirements: " + "; ".join(
                    f"{item.value} ({item.why}): {item.hint}" for item in required
                )
                print(failures[candidate.stage_id], file=sys.stderr)
                if stream:
                    forward(StageEvent(EventType.WARNING, candidate.stage_id, {
                        "message": failures[candidate.stage_id],
                        "missing": [item.to_dict() for item in candidate.missing],
                    }))
                continue
            try:
                if stage_uses_llm(registry[candidate.stage_id], settings):
                    session.ensure()
                else:
                    session.close()
                run_stage_process(
                    candidate.stage_id,
                    project_path,
                    settings_path,
                    candidate.fingerprint,
                    effective_settings=settings,
                    fingerprint_parts=fingerprint_parts(
                        registry[candidate.stage_id], project, settings
                    ),
                    on_event=report,
                    cancelled=lambda: cancel_file is not None and cancel_file.exists(),
                )
            except StageProcessError as exc:
                if cancel_file is not None and cancel_file.exists():
                    cancelled_run = True
                    break
                failures[candidate.stage_id] = str(exc)
                print(str(exc), file=sys.stderr)
        report = {
            "schema_version": 1,
            "status": "cancelled" if cancelled_run else "failed" if failures else "ok",
            "failures": failures,
            "stages": [
                {
                    "id": item.stage_id,
                    "decision": item.decision.kind.value,
                    "reason": item.decision.reason,
                    "reuse": item.will_reuse,
                    "missing": [entry.to_dict() for entry in item.missing],
                    **({"stale": item.stale_reason} if item.stale_reason else {}),
                }
                for item in (plan.stages if not cancelled_run else [])
            ],
            "stale_reasons": stale_reasons,
        }
        (project.work_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        if cancelled_run:
            raise InstallationCancelled("Processing cancelled; published results are retained.")
        if failures:
            raise StageProcessError(f"pipeline failed: {', '.join(failures)}; see report.json")
    return Project.load(project_path)


def _doctor() -> int:
    checks = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "python": shutil.which("python3.12"),
    }
    for name, path in checks.items():
        print(f"{name}: {path or 'MISSING'}")
    if checks["ffmpeg"]:
        result = subprocess.run(
            [checks["ffmpeg"], "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
        )
        for name, codec in (("nvenc", "h264_nvenc"), ("videotoolbox", "h264_videotoolbox")):
            print(f"{name}: {'available' if codec in result.stdout else 'unavailable'}")
    return 0 if all(checks.values()) else 1


def _update_material(
    source: Path,
    asset_ids: str | list[str],
    *,
    role: str | None,
    group: str | None,
    device: str | None,
) -> Project:
    if role is None and group is None and device is None:
        raise ValueError("material requires --role, --group, or --device")
    project, project_path, _ = _project(source)
    with project_lock(project.work_dir):
        recover_publications(project.work_dir)
        ids = [asset_ids] if isinstance(asset_ids, str) else asset_ids
        by_id = {item.id: item for item in project.assets}
        unknown = [asset_id for asset_id in ids if asset_id not in by_id]
        if unknown:
            raise ValueError(f"unknown asset: {', '.join(unknown)}; run scan first")
        for asset_id in ids:
            asset = by_id[asset_id]
            overrides = asset.manual.setdefault("overrides", {})
            if role is not None:
                asset.role = AssetRole(role)
                overrides["role"] = role
            if group is not None:
                asset.group_id = group or None
                overrides["group_id"] = asset.group_id
            if device is not None:
                asset.device = device or None
                overrides["device"] = asset.device
        project.plan_revision += 1
        project.save(project_path)
    return project


def _update_tracks(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    from studio.core.recorder_tracks import set_track

    project, project_path, _ = _project(source)
    if args.asset is None:
        return _tracks_document(project)
    if not args.channels:
        raise ValueError("tracks --asset requires --channels, for example --channels 0,1")
    try:
        channels = tuple(int(item) for item in args.channels.split(","))
    except ValueError as exc:
        raise ValueError("--channels must be comma-separated channel numbers") from exc
    with project_lock(project.work_dir):
        recover_publications(project.work_dir)
        project = Project.load(project_path)
        set_track(
            project, args.asset, channels, speaker=args.speaker, name=args.name,
            enabled=True if args.enable else False if args.disable else None,
        )
        project.plan_revision += 1
        project.save(project_path)
    return _tracks_document(project)


def _update_cameras(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    project, project_path, settings_path = _project(source)
    if getattr(args, "suggest", False):
        return _suggest_cameras(project, project_path, settings_path, apply=args.apply)
    if args.speaker is None:
        return _tracks_document(project)
    speaker = args.speaker.strip()
    if not speaker:
        raise ValueError("cameras --speaker must not be empty")
    with project_lock(project.work_dir):
        recover_publications(project.work_dir)
        project = Project.load(project_path)
        if args.clear:
            project.speaker_cameras.pop(speaker, None)
        else:
            camera = (args.camera or "").strip()
            known = {
                value for asset in project.assets if asset.role is AssetRole.CAMERA
                for value in (asset.id, asset.group_id) if value
            }
            if camera not in known:
                raise ValueError(f"unknown camera: {camera or '(empty)'}; use a camera group or ID")
            project.speaker_cameras[speaker] = camera
        project.plan_revision += 1
        project.save(project_path)
    return _tracks_document(project)


def _tune(args: argparse.Namespace) -> dict[str, Any]:
    from studio.core.autotune import tune_project
    from studio.core.settings_store import save_settings

    project, _project_path, settings_path = _project(args.source)
    if args.sample_s <= 0:
        raise ValueError("tune --sample-s must be positive")
    both = not args.whisper and not args.llama
    # Recorders first: their speech is the cleanest sample.
    sources = [project.source_dir / asset.path
               for role in (AssetRole.RECORDER, AssetRole.CAMERA)
               for asset in project.assets if asset.role is role]
    report = tune_project(sources, _settings(settings_path), project.work_dir,
                          whisper=args.whisper or both, llama=args.llama or both,
                          sample_s=args.sample_s)
    path = project.work_dir / "tuning.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.apply and report["recommendations"]:
        save_settings(project.work_dir, report["recommendations"])
        report["applied"] = True
    return report


def _update_reel(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    from studio.core.framing import load_reel_overrides, save_reel_override

    project, _project_path, _settings_path = _project(source)
    if args.reel is not None:
        current = load_reel_overrides(project.work_dir).get(args.reel, {})
        if args.clear:
            values: dict[str, object] = {}
        else:
            changes = {"framing": args.framing, "crop_x": args.crop_x,
                       "tracking": args.tracking, "title": args.title}
            values = {**current, **{k: v for k, v in changes.items() if v is not None}}
            if values.get("title") == "":
                values.pop("title")
        save_reel_override(project.work_dir, args.reel, values or None)
    return {"reels": load_reel_overrides(project.work_dir)}


def _suggest_cameras(
    project: Project, project_path: Path, settings_path: Path, *, apply: bool,
) -> dict[str, Any]:
    from studio.stages.camera_binding import collect_evidence, suggest_bindings

    speakers_paths = project.outputs.get("speakers", [])
    if not speakers_paths or not speakers_paths[0].is_file():
        raise ValueError("camera suggestions need speaker turns; run the speakers stage first")
    turns = json.loads(speakers_paths[0].read_text(encoding="utf-8"))
    names = sorted({str(turn["speaker"]) for turn in turns} - {"unknown", "overlap"})
    device = load_settings([settings_path]).reels.tracking_device
    suggestion = suggest_bindings(collect_evidence(project, turns, device=device), names)
    applied: dict[str, str] = {}
    if apply:
        with project_lock(project.work_dir):
            recover_publications(project.work_dir)
            project = Project.load(project_path)
            for speaker, camera in suggestion["speaker_cameras"].items():
                # Choices the user made stay; suggestions only fill gaps.
                if speaker not in project.speaker_cameras:
                    project.speaker_cameras[speaker] = camera
                    applied[speaker] = camera
            if applied:
                project.plan_revision += 1
                project.save(project_path)
    return {**_tracks_document(project), "suggestion": suggestion, "applied": applied}


def _tracks_document(project: Project) -> dict[str, Any]:
    from studio.core.channels import describe

    analyses = {asset.id: asset.manual.get("media_info", {}).get("channel_analysis")
                for asset in project.assets}
    return {
        "tracks": [
            {"id": track.id, "asset_id": track.asset_id, "channels": list(track.channels),
             "speaker": track.speaker, "name": track.name, "enabled": track.enabled,
             **({"channel_analysis": describe(analyses[track.asset_id])}
                if analyses.get(track.asset_id) else {})}
            for track in project.recorder_tracks
        ],
        "speaker_cameras": dict(project.speaker_cameras),
    }


def _run_batch(
    sources: list[Path],
    overrides: list[str],
    *,
    only: set[str] | None,
    skip: set[str] | None,
    stream: bool,
    cancel_file: Path | None = None,
    min_free_gb: float = 0.0,
) -> tuple[int, list[dict[str, str]]]:
    results: list[dict[str, str]] = []
    owner = BatchLlamaSession(lambda: cancel_file is not None and cancel_file.exists())
    try:
        for index, source in enumerate(sources):
            if cancel_file is not None and cancel_file.exists():
                results.extend(
                    {"source": str(path), "status": "not_started"} for path in sources[index:]
                )
                return 130, results
            free_gb = shutil.disk_usage(source).free / 1e9 if source.exists() else None
            if min_free_gb and free_gb is not None and free_gb < min_free_gb:
                # Starting another project risks a half-written one on a full disk.
                error = f"{free_gb:.1f} GB free, batch.min_free_disk_gb is {min_free_gb:g}"
                results.append({"source": str(source), "status": "failed", "error": error})
                results.extend(
                    {"source": str(path), "status": "not_started"} for path in sources[index + 1 :]
                )
                return 1, results
            try:
                options: dict[str, Any] = {"batch_session": owner}
                if cancel_file is not None:
                    options["cancel_file"] = cancel_file
                _run_pipeline(source, overrides, only=only, skip=skip, stream=stream, **options)
            except InstallationCancelled as exc:
                results.append({"source": str(source), "status": "cancelled", "error": str(exc)})
                results.extend(
                    {"source": str(path), "status": "not_started"} for path in sources[index + 1 :]
                )
                return 130, results
            except (OSError, RuntimeError, StageProcessError) as exc:
                owner.close()
                results.append({"source": str(source), "status": "failed", "error": str(exc)})
            else:
                results.append({"source": str(source), "status": "ok"})
        return (1 if any(result["status"] == "failed" for result in results) else 0), results
    finally:
        owner.close()


def _fetch_project(
    url: str, destination: Path, cancelled: Callable[[], bool] = lambda: False,
    conf: dict[str, Any] | None = None,
) -> tuple[Project, dict[str, Any]]:
    from studio.stages.fetch import resumable_fetch, yt_dlp_options

    destination = destination.resolve()
    if (destination.exists() and any(destination.iterdir())
            and not resumable_fetch(destination, url)):
        raise ValueError(
            "fetch destination must be a new or empty directory, "
            "or an interrupted download of the same URL"
        )
    destination.mkdir(parents=True, exist_ok=True)
    media, info = fetch_to_directory(url, destination, cancelled,
                                     options=yt_dlp_options(conf or {}))
    project = _run_discovery(destination, include_prepare=False)
    report_path = project.work_dir / "stages" / "fetch" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "url": url, "media": str(media), "info": str(info)}
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    project.outputs["fetch"] = [media, info, report_path]
    project.save(project.work_dir / "project.json")
    return project, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studio")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("--json", action="store_true")
        command.add_argument("--set", action="append", default=[])
        if name == "run":
            command.add_argument("--only", nargs="+", metavar="STAGE")
            command.add_argument("--skip", nargs="+", metavar="STAGE")
            command.add_argument("--events", action="store_true")
            command.add_argument(
                "--rehash", action="store_true",
                help="recompute source and artifact digests instead of using the cache",
            )
        command.add_argument("--cancel-file", type=Path)
    commands.add_parser("doctor")
    compute = commands.add_parser("compute")
    compute.add_argument("--probe-whisper", action="store_true", help=argparse.SUPPRESS)
    compute.add_argument("--bench-whisper", metavar="JSON", help=argparse.SUPPRESS)
    tune = commands.add_parser("tune", help="measure Whisper/llama settings on this machine")
    tune.add_argument("source", type=Path)
    tune.add_argument("--whisper", action="store_true", help="tune Whisper only")
    tune.add_argument("--llama", action="store_true", help="tune the owned llama-server only")
    tune.add_argument("--sample-s", type=float, default=60.0)
    tune.add_argument("--apply", action="store_true", help="save the recommendations")
    verify = commands.add_parser("verify")
    verify.add_argument("camera", type=Path)
    verify.add_argument("voice", type=Path)
    verify.add_argument("--grid-s", type=float, default=5.0)
    verify.add_argument("--window-s", type=float, default=4.0)
    verify.add_argument("--threshold-ms", type=float, default=None,
                        help="default: sync.max_drift_ms of the project")
    verify.add_argument("--project", type=Path,
                        help="material folder whose settings apply (default: the camera's)")
    verify.add_argument("--min-bad-windows", type=int, default=3)
    verify.add_argument("--max-bad-fraction", type=float, default=0.1)
    review = commands.add_parser("review")
    review.add_argument("source", type=Path)
    review.add_argument("--cut")
    action = review.add_mutually_exclusive_group()
    action.add_argument("--accept", action="store_true")
    action.add_argument("--reject", action="store_true")
    modules = commands.add_parser("modules")
    modules.add_argument("action", choices=("list", "install"), nargs="?", default="list")
    modules.add_argument("name", nargs="?")
    modules.add_argument("--events", action="store_true")
    modules.add_argument("--cancel-file", type=Path)
    models = commands.add_parser("models")
    models.add_argument("action", choices=("list", "install"), nargs="?", default="list")
    models.add_argument("name", nargs="?")
    models.add_argument("--events", action="store_true")
    models.add_argument("--cancel-file", type=Path)
    material = commands.add_parser("material")
    material.add_argument("source", type=Path)
    material.add_argument("--asset", required=True, action="append")
    material.add_argument("--role", choices=[role.value for role in AssetRole])
    material.add_argument("--group")
    material.add_argument("--device")
    material.add_argument("--json", action="store_true")
    tracks = commands.add_parser("tracks", help="list or edit recorder track lanes")
    tracks.add_argument("source", type=Path)
    tracks.add_argument("--asset")
    tracks.add_argument("--channels", help="comma-separated channels forming one lane")
    tracks.add_argument("--speaker")
    tracks.add_argument("--name")
    enabled = tracks.add_mutually_exclusive_group()
    enabled.add_argument("--enable", action="store_true")
    enabled.add_argument("--disable", action="store_true")
    tracks.add_argument("--json", action="store_true")
    cameras = commands.add_parser("cameras", help="list or edit speaker camera choices")
    cameras.add_argument("source", type=Path)
    cameras.add_argument("--speaker")
    target = cameras.add_mutually_exclusive_group()
    target.add_argument("--camera")
    target.add_argument("--clear", action="store_true")
    target.add_argument("--suggest", action="store_true",
                        help="suggest bindings from faces and active speakers (vision module)")
    cameras.add_argument("--apply", action="store_true",
                         help="with --suggest: save suggestions for speakers without a choice")
    cameras.add_argument("--json", action="store_true")
    preview = commands.add_parser("preview", help="render a low-resolution proxy of the edit")
    preview.add_argument("source", type=Path)
    preview.add_argument("--height", type=int, default=360)
    preview.add_argument("--set", action="append", default=[])
    reel = commands.add_parser("reel", help="list or edit per-reel framing and titles")
    reel.add_argument("source", type=Path)
    reel.add_argument("--reel", help="candidate_id or reel-NNN stem")
    reel.add_argument("--framing", choices=("source", "crop", "fit"))
    reel.add_argument("--crop-x", type=float)
    reel.add_argument("--tracking", action=argparse.BooleanOptionalAction, default=None)
    reel.add_argument("--title")
    reel.add_argument("--clear", action="store_true", help="return the reel to project settings")
    preset = commands.add_parser("preset", help="list or apply packaged settings presets")
    preset.add_argument("source", type=Path, nargs="?")
    preset.add_argument("--apply", metavar="NAME")
    batch = commands.add_parser("batch")
    batch.add_argument("sources", nargs="+", type=Path)
    batch.add_argument("--json", action="store_true")
    batch.add_argument("--set", action="append", default=[])
    batch.add_argument("--only", nargs="+", metavar="STAGE")
    batch.add_argument("--skip", nargs="+", metavar="STAGE")
    batch.add_argument("--events", action="store_true")
    batch.add_argument("--cancel-file", type=Path)
    batch.add_argument("--report", type=Path)
    batch.add_argument("--notify-when", choices=("always", "failure", "never"),
                        help="default: batch.notify_when")
    batch.add_argument("--notify-webhook", metavar="URL",
                        help="POST the JSON run summary here (Slack/Discord/ntfy compatible)")
    batch.add_argument("--notify-desktop", action="store_true", default=None,
                        help="show a desktop notification when the run ends")
    channel = commands.add_parser("channel")
    channel.add_argument("url")
    channel.add_argument("destination", type=Path)
    channel.add_argument("--limit", type=int, default=10)
    channel.add_argument("--set", action="append", default=[])
    channel.add_argument("--only", nargs="+")
    channel.add_argument("--skip", nargs="+")
    channel.add_argument("--events", action="store_true")
    channel.add_argument("--cancel-file", type=Path)
    channel.add_argument("--report", type=Path)
    channel.add_argument("--notify-when", choices=("always", "failure", "never"),
                        help="default: batch.notify_when")
    channel.add_argument("--notify-webhook", metavar="URL",
                        help="POST the JSON run summary here (Slack/Discord/ntfy compatible)")
    channel.add_argument("--notify-desktop", action="store_true", default=None,
                        help="show a desktop notification when the run ends")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("url")
    fetch.add_argument("destination", type=Path)
    fetch.add_argument("--json", action="store_true")
    fetch.add_argument("--cancel-file", type=Path)
    fetch.add_argument("--set", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except InstallationCancelled as exc:
        print(json.dumps({"type": "installation_cancelled", "message": str(exc)}), flush=True)
        return 130
    except SettingsError as exc:
        print(f"settings error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    except (StageProcessError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"run error: {exc}", file=sys.stderr)
        return 1


def _main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compute":
        from studio.core.capabilities import inspect_compute, whisper_probe

        if args.bench_whisper:
            from studio.core.autotune import whisper_benchmark

            print(json.dumps(whisper_benchmark(json.loads(args.bench_whisper))))
            return 0
        print(json.dumps(whisper_probe() if args.probe_whisper else inspect_compute(), indent=2))
        return 0
    if args.command == "tune":
        print(json.dumps(_tune(args), ensure_ascii=False, indent=2))
        return 0
    install_options = {}
    if getattr(args, "cancel_file", None):
        install_options["cancelled"] = args.cancel_file.exists

    def installation_progress(event: dict) -> None:
        if args.events:
            print(json.dumps({"type": "installation_progress", **event}), flush=True)

    if args.command == "models":
        from studio.modules.models import ModelManager

        model_manager = ModelManager()
        if args.action == "install":
            if not args.name:
                raise ValueError("models install requires a model name")
            print(
                json.dumps(
                    {
                        "name": args.name,
                        "path": str(
                            model_manager.install(
                                args.name, progress=installation_progress, **install_options
                            )
                        ),
                    }
                )
            )
        else:
            if args.name:
                raise ValueError("models list does not accept a model name")
            print(
                json.dumps(
                    {
                        "models": [model_manager.describe(name) for name in model_manager.names()]
                    }
                )
            )
        return 0
    if args.command == "modules":
        if args.action == "list" and args.name:
            raise ValueError("modules list does not accept a module name")
        if args.action == "install" and not args.name:
            raise ValueError("modules install requires a module name")
        manager = ModuleManager()
        if args.action == "install":
            interpreter = manager.install(
                args.name, progress=installation_progress, **install_options
            )
            print(json.dumps({"name": args.name, "interpreter": str(interpreter)}))
            return 0
        print(
            json.dumps(
                {
                    "modules": [
                        {
                            "name": name,
                            "installed": manager.installed(name),
                            "interpreter": str(manager.interpreter(name)),
                        }
                        for name in module_names()
                    ],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "material":
        project = _update_material(
            args.source, args.asset, role=args.role, group=args.group, device=args.device
        )
        documents = [
            {
                "id": asset.id,
                "role": asset.role.value,
                "group_id": asset.group_id,
                "device": asset.device,
            }
            for asset_id in args.asset
            for asset in project.assets
            if asset.id == asset_id
        ]
        material: object = documents[0] if len(documents) == 1 else {"assets": documents}
        print(json.dumps(material, ensure_ascii=False, indent=2) if args.json else material)
        return 0
    if args.command == "preview":
        from studio.stages.timeline_preview import render_timeline_preview

        if not 64 <= args.height <= 2160:
            raise ValueError("preview --height must be between 64 and 2160")
        project, _project_path, settings_path = _project(args.source)
        result = render_timeline_preview(project, _settings(settings_path, args.set),
                                         height=args.height)
        print(json.dumps({"video": str(result.video), "spans": list(result.spans),
                          "reused": result.reused}, indent=2))
        return 0
    if args.command == "reel":
        print(json.dumps(_update_reel(args.source, args), ensure_ascii=False, indent=2))
        return 0
    if args.command in {"tracks", "cameras"}:
        handler = _update_tracks if args.command == "tracks" else _update_cameras
        print(json.dumps(handler(args.source, args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "preset":
        from studio.core.presets import apply_preset, list_presets

        if args.apply is None:
            print("\n".join(list_presets()))
            return 0
        if args.source is None:
            raise ValueError("preset --apply requires a project folder")
        _, _, settings_path = _project(args.source)
        applied = apply_preset(settings_path.parent, args.apply)
        print(json.dumps(applied, ensure_ascii=False, indent=2))
        return 0
    if args.command in {"batch", "channel"}:
        from studio.cli.channel import acquire_channel, write_report
        from studio.cli.notify import NotifyOptions, send

        queue_settings = load_settings(
            [args.destination / "settings.yaml"] if args.command == "channel" else [], args.set
        )
        defaults = queue_settings.batch
        notify = NotifyOptions(
            args.notify_when or defaults.notify_when,
            args.notify_webhook if args.notify_webhook is not None else defaults.notify_webhook,
            args.notify_desktop if args.notify_desktop is not None else defaults.notify_desktop,
        )
        fetch_conf = queue_settings.to_dict()["fetch"]
        acquisition: list[dict[str, Any]] = []
        sources = args.sources if args.command == "batch" else []
        status = 0
        results: list[dict[str, str]] = []
        if args.command == "channel":
            try:
                sources, acquisition = acquire_channel(
                    args.url,
                    args.destination,
                    args.limit,
                    lambda url, folder, cancelled: _fetch_project(
                        url, folder, cancelled, fetch_conf
                    ),
                    lambda: args.cancel_file is not None and args.cancel_file.exists(),
                    fetch_conf,
                )
            except InstallationCancelled:
                status = 130
                report_path = args.destination / "channel-acquisition.json"
                if report_path.is_file():
                    acquisition = json.loads(report_path.read_text(encoding="utf-8"))["videos"]
        if status != 130:
            status, results = _run_batch(
                sources,
                args.set,
                only=set(args.only or []),
                skip=set(args.skip or []),
                stream=args.events,
                cancel_file=args.cancel_file,
                min_free_gb=defaults.min_free_disk_gb,
            )
        data = {
            "schema_version": 1,
            "status": "cancelled" if status == 130 else "failed" if status else "ok",
            "projects": results,
            "acquisition": acquisition,
        }
        if status != 130 and any(item["status"] == "failed" for item in acquisition):
            status = 1
            data["status"] = "failed"
        notifications = send(notify, data, str(args.report) if args.report else None)
        if notifications:
            data["notifications"] = notifications
        if args.report:
            write_report(args.report, data)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if args.events:
            print(
                json.dumps(
                    {
                        "type": "queue_finished",
                        "status": data["status"],
                        "report": str(args.report) if args.report else None,
                    }
                ),
                flush=True,
            )
        return status
    if args.command == "fetch":
        _, report = _fetch_project(
            args.url, args.destination,
            lambda: args.cancel_file is not None and args.cancel_file.exists(),
            load_settings([], args.set).to_dict()["fetch"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else report)
        return 0
    if args.command == "doctor":
        return _doctor()
    if args.command == "verify":
        from studio.stages.sync_verify import measure, validate_parameters

        threshold = args.threshold_ms
        if threshold is None:
            settings_path = (args.project or args.camera.resolve().parent) / "_studio" / (
                "settings.yaml")
            threshold = float(load_settings([settings_path]).sync.max_drift_ms)
        validate_parameters(args.grid_s, args.window_s, median_threshold_ms=threshold)
        if args.min_bad_windows < 1 or not 0 <= args.max_bad_fraction <= 1:
            raise ValueError("--min-bad-windows >= 1 and --max-bad-fraction in [0, 1]")
        verification = measure(args.camera, args.voice, grid_s=args.grid_s, window_s=args.window_s)
        verdict, reason = verification.verdict(
            threshold, min_bad_windows=args.min_bad_windows,
            max_bad_fraction=args.max_bad_fraction,
        )
        print(json.dumps({"status": verdict, "reason": reason,
                          **verification.summary(threshold, args.min_bad_windows)}, indent=2))
        return 0 if verdict == "passed" else 1
    if args.command == "review":
        from studio.stages.roughcut import review_cut

        project, _, _ = _project(args.source)
        with project_lock(project.work_dir):
            recover_publications(project.work_dir)
            if args.cut:
                if not (args.accept or args.reject):
                    raise ValueError("review --cut requires --accept or --reject")
                review_cut(project, args.cut, accepted=args.accept)
            elif args.accept or args.reject:
                raise ValueError("review --accept/--reject requires --cut")
            paths = project.outputs.get("roughcut", [])
            if not paths:
                raise ValueError("no roughcut decisions to review")
            print(paths[0].read_text(encoding="utf-8"), end="")
        return 0
    if getattr(args, "rehash", False):
        set_rehash(True)
    project = (
        _run_pipeline(
            args.source,
            args.set,
            only=set(args.only or []),
            skip=set(args.skip or []),
            stream=args.events,
            cancel_file=args.cancel_file,
        )
        if args.command == "run"
        else _run_discovery(
            args.source,
            include_prepare=args.command == "plan",
            overrides=args.set,
            cancel_file=args.cancel_file,
        )
    )
    if args.command == "scan":
        data = project.to_dict()
    else:
        settings = _settings(project.work_dir / "settings.yaml", args.set)
        plan = build_plan(stage_registry().values(), project, settings)
        data = {
            "revision": plan.revision,
            "stages": [
                {
                    "id": item.stage_id,
                    "decision": item.decision.kind.value,
                    "reason": item.decision.reason,
                    "reuse": item.will_reuse,
                    "missing": [entry.to_dict() for entry in item.missing],
                    **({"stale": item.stale_reason} if item.stale_reason else {}),
                }
                for item in plan.stages
            ],
        }
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
