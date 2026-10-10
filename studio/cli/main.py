"""Headless Studio commands."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from studio import __version__
from studio.core.project import AssetRole, Project
from studio.core.publication import recover_publications
from studio.core.settings import SettingsError, load_settings
from studio.core.workspace import project_lock
from studio.llm.session import BatchLlamaSession, ManagedLlamaSession, effective_llm_settings
from studio.modules.cancellation import InstallationCancelled
from studio.modules.manager import ModuleManager, module_names
from studio.stages.base import stage_uses_llm
from studio.stages.events import StageEvent
from studio.stages.fetch import fetch_to_directory
from studio.stages.planner import build_plan
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
    return effective_llm_settings(load_settings([path], overrides).to_dict())


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
                    on_event=forward if stream else None,
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
                }
                for item in (plan.stages if not cancelled_run else [])
            ],
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
        print(f"nvenc: {'available' if 'h264_nvenc' in result.stdout else 'unavailable'}")
    return 0 if all(checks.values()) else 1


def _update_material(
    source: Path,
    asset_id: str,
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
        asset = next((item for item in project.assets if item.id == asset_id), None)
        if asset is None:
            raise ValueError(f"unknown asset: {asset_id}; run scan first")
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


def _run_batch(
    sources: list[Path],
    overrides: list[str],
    *,
    only: set[str] | None,
    skip: set[str] | None,
    stream: bool,
    cancel_file: Path | None = None,
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
            try:
                options = {"batch_session": owner}
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


def _fetch_project(url: str, destination: Path) -> tuple[Project, dict[str, str]]:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("fetch destination must be a new or empty directory")
    destination.mkdir(parents=True, exist_ok=True)
    media, info = fetch_to_directory(url, destination)
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
        command.add_argument("--cancel-file", type=Path)
    commands.add_parser("doctor")
    compute = commands.add_parser("compute")
    compute.add_argument("--probe-whisper", action="store_true", help=argparse.SUPPRESS)
    verify = commands.add_parser("verify")
    verify.add_argument("camera", type=Path)
    verify.add_argument("voice", type=Path)
    verify.add_argument("--grid-s", type=float, default=5.0)
    verify.add_argument("--window-s", type=float, default=4.0)
    verify.add_argument("--threshold-ms", type=float, default=20.0)
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
    material.add_argument("--asset", required=True)
    material.add_argument("--role", choices=[role.value for role in AssetRole])
    material.add_argument("--group")
    material.add_argument("--device")
    material.add_argument("--json", action="store_true")
    batch = commands.add_parser("batch")
    batch.add_argument("sources", nargs="+", type=Path)
    batch.add_argument("--json", action="store_true")
    batch.add_argument("--set", action="append", default=[])
    batch.add_argument("--only", nargs="+", metavar="STAGE")
    batch.add_argument("--skip", nargs="+", metavar="STAGE")
    batch.add_argument("--events", action="store_true")
    batch.add_argument("--cancel-file", type=Path)
    batch.add_argument("--report", type=Path)
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
    fetch = commands.add_parser("fetch")
    fetch.add_argument("url")
    fetch.add_argument("destination", type=Path)
    fetch.add_argument("--json", action="store_true")
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

        print(json.dumps(whisper_probe() if args.probe_whisper else inspect_compute(), indent=2))
        return 0
    install_options = {}
    if getattr(args, "cancel_file", None):
        install_options["cancelled"] = args.cancel_file.exists

    def installation_progress(event: dict) -> None:
        if args.events:
            print(json.dumps({"type": "installation_progress", **event}), flush=True)

    if args.command == "models":
        from studio.modules.models import ModelManager, recipes

        manager = ModelManager()
        if args.action == "install":
            if not args.name:
                raise ValueError("models install requires a model name")
            print(
                json.dumps(
                    {
                        "name": args.name,
                        "path": str(
                            manager.install(
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
                        "models": [
                            {
                                "name": name,
                                "installed": manager.installed(name),
                                "path": str(manager.path(name)),
                            }
                            for name in recipes()
                        ]
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
        asset = next(item for item in project.assets if item.id == args.asset)
        data = {
            "id": asset.id,
            "role": asset.role.value,
            "group_id": asset.group_id,
            "device": asset.device,
        }
        print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else data)
        return 0
    if args.command in {"batch", "channel"}:
        from studio.cli.channel import acquire_channel, write_report

        acquisition = []
        sources = args.sources if args.command == "batch" else []
        if args.command == "channel":
            sources, acquisition = acquire_channel(
                args.url,
                args.destination,
                args.limit,
                _fetch_project,
                lambda: args.cancel_file is not None and args.cancel_file.exists(),
            )
        status, results = _run_batch(
            sources,
            args.set,
            only=set(args.only or []),
            skip=set(args.skip or []),
            stream=args.events,
            cancel_file=args.cancel_file,
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
        _, report = _fetch_project(args.url, args.destination)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else report)
        return 0
    if args.command == "doctor":
        return _doctor()
    if args.command == "verify":
        from studio.stages.sync_verify import measure, validate_parameters

        validate_parameters(args.grid_s, args.window_s, median_threshold_ms=args.threshold_ms)
        report = measure(args.camera, args.voice, grid_s=args.grid_s, window_s=args.window_s)
        status, reason = report.verdict(args.threshold_ms)
        print(json.dumps({"status": status, "reason": reason, **report.summary()}, indent=2))
        return 0 if status == "passed" else 1
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
                }
                for item in plan.stages
            ],
        }
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
