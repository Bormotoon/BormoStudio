import json
import sys
from pathlib import Path

from tools.legacy_suites import Suite, compare, main, parse_junit, run_suite, summarize

JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.test_a" name="test_ok"/>
  <testcase classname="tests.test_a" name="test_bad"><failure message="x"/></testcase>
  <testcase classname="tests.test_b" name="test_err"><error message="y"/></testcase>
  <testcase classname="tests.test_b" name="test_skip"><skipped message="z"/></testcase>
</testsuite></testsuites>
"""


def test_junit_outcomes_and_snapshot_comparison(tmp_path: Path) -> None:
    path = tmp_path / "r.xml"
    path.write_text(JUNIT, encoding="utf-8")
    outcomes = parse_junit(path)
    assert outcomes == {
        "tests.test_a::test_ok": "passed", "tests.test_a::test_bad": "failed",
        "tests.test_b::test_err": "error", "tests.test_b::test_skip": "skipped",
    }
    report = summarize(Suite("x", tmp_path, 1), outcomes, 1)
    assert report["counts"] == {"passed": 1, "failed": 1, "error": 1, "skipped": 1}
    assert report["matches_snapshot"] is False
    previous = {"tests": {"tests.test_a::test_ok": "passed", "tests.test_a::test_bad": "passed",
                          "tests.test_b::test_err": "failed", "tests.gone::test": "passed"}}
    current = {"tests": {**outcomes, "tests.test_b::test_err": "passed"}}
    assert compare(previous, current) == {
        "regressed": ["tests.test_a::test_bad"],
        "fixed": ["tests.test_b::test_err"],
        "missing": ["tests.gone::test"],
        "added": ["tests.test_b::test_skip"],
    }


def test_run_suite_executes_pytest_in_the_frozen_tree(tmp_path: Path) -> None:
    tree = tmp_path / "legacy"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests" / "test_one.py").write_text(
        "def test_one():\n    assert True\n", encoding="utf-8"
    )
    report = run_suite(Suite("demo", tree, 1), sys.executable, tmp_path / "out", [])
    assert report["matches_snapshot"] and report["counts"] == {"passed": 1}
    saved = tmp_path / "out" / "demo.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["suite"] == "demo"
    assert main(["compare", str(saved), str(saved)]) == 0
