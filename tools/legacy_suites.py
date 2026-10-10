"""Run the frozen WhisperSync and Forge test suites as comparison sets.

Each suite runs unmodified inside ``legacy/<name>`` with its own interpreter
(the locked environment of that application) and writes JUnit XML. The tool
turns the XML into a per-test outcome report and compares it with a recorded
report: tests that stopped passing, newly passing, disappeared or appeared.
Together with the expected totals of the frozen snapshots this makes any drift
of the legacy oracle visible instead of being hidden in a summary count.

Usage::

    python -m tools.legacy_suites run --suite whispersync --python /path/to/py \\
        --output reports/legacy
    python -m tools.legacy_suites compare old/whispersync.json new/whispersync.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Suite:
    name: str
    directory: Path
    expected_passed: int


SUITES = {
    "whispersync": Suite("whispersync", ROOT / "legacy" / "whispersync", 403),
    "forge": Suite("forge", ROOT / "legacy" / "forge", 620),
}


def parse_junit(path: Path) -> dict[str, str]:
    """Map ``module::Class::test`` IDs to passed/failed/error/skipped."""
    outcomes: dict[str, str] = {}
    for case in ElementTree.parse(path).getroot().iter("testcase"):
        test_id = "::".join(
            part for part in (case.get("classname", ""), case.get("name", "")) if part
        )
        if case.find("error") is not None:
            outcome = "error"
        elif case.find("failure") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        outcomes[test_id] = outcome
    return outcomes


def summarize(suite: Suite, outcomes: dict[str, str], returncode: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for outcome in outcomes.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    passed = counts.get("passed", 0)
    return {
        "schema_version": 1,
        "suite": suite.name,
        "returncode": returncode,
        "counts": counts,
        "expected_passed": suite.expected_passed,
        "matches_snapshot": returncode == 0 and passed == suite.expected_passed,
        "tests": dict(sorted(outcomes.items())),
    }


def compare(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, list[str]]:
    before, after = previous["tests"], current["tests"]
    return {
        "regressed": sorted(
            test for test, outcome in after.items()
            if outcome in {"failed", "error"} and before.get(test) == "passed"
        ),
        "fixed": sorted(
            test for test, outcome in after.items()
            if outcome == "passed" and before.get(test) in {"failed", "error"}
        ),
        "missing": sorted(set(before) - set(after)),
        "added": sorted(set(after) - set(before)),
    }


def run_suite(suite: Suite, python: str, output: Path, extra: list[str]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    junit = (output / f"{suite.name}.xml").resolve()
    environment = {**os.environ, "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM",
                                                                    "offscreen")}
    # The frozen code must import from its own tree, never from Studio.
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}",
         *extra],
        cwd=suite.directory, env=environment,
    )
    if not junit.is_file():
        raise RuntimeError(f"{suite.name}: pytest wrote no JUnit report (exit {result.returncode})")
    report = summarize(suite, parse_junit(junit), result.returncode)
    (output / f"{suite.name}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--suite", choices=sorted(SUITES), action="append", required=True)
    run.add_argument("--python", default=sys.executable,
                     help="interpreter of the suite's locked environment")
    run.add_argument("--output", type=Path, default=Path("legacy-suites"))
    run.add_argument("--baseline", type=Path,
                     help="directory with earlier <suite>.json reports to compare")
    run.add_argument("pytest_args", nargs=argparse.REMAINDER)
    diff = commands.add_parser("compare")
    diff.add_argument("previous", type=Path)
    diff.add_argument("current", type=Path)
    args = parser.parse_args(argv)
    if args.command == "compare":
        changes = compare(json.loads(args.previous.read_text(encoding="utf-8")),
                          json.loads(args.current.read_text(encoding="utf-8")))
        print(json.dumps(changes, indent=2))
        return 1 if changes["regressed"] or changes["missing"] else 0
    status = 0
    for name in args.suite:
        report = run_suite(SUITES[name], args.python, args.output,
                           [item for item in args.pytest_args if item != "--"])
        line = (f"{name}: {report['counts']} expected {report['expected_passed']} passed "
                f"-> {'matches snapshot' if report['matches_snapshot'] else 'DIFFERS'}")
        print(line)
        if not report["matches_snapshot"]:
            status = 1
        if args.baseline and (args.baseline / f"{name}.json").is_file():
            changes = compare(
                json.loads((args.baseline / f"{name}.json").read_text(encoding="utf-8")), report
            )
            print(json.dumps(changes, indent=2))
            if changes["regressed"] or changes["missing"]:
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
