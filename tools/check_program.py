"""Re-transcribe a published master and report word integrity in edited time."""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import asdict
from pathlib import Path

from studio.core.project import Project
from studio.core.transcript import Transcript
from studio.stages.sync_matcher import normalize_token
from studio.stages.sync_self_check import diagnose
from studio.stages.transcribe import TranscribeSettings, WhisperEngine


def compare_words(expected: Transcript, actual: Transcript) -> dict:
    reference = [word for word in expected.words if normalize_token(word.text)]
    rendered = [word for word in actual.words if normalize_token(word.text)]
    matcher = difflib.SequenceMatcher(
        None, [normalize_token(word.text) for word in reference],
        [normalize_token(word.text) for word in rendered], autojunk=False,
    )
    differences = [
        {"kind": kind, "expected": [asdict(word) for word in reference[a:b]],
         "recognized": [asdict(word) for word in rendered[c:d]]}
        for kind, a, b, c, d in matcher.get_opcodes() if kind != "equal"
    ]
    outcome = diagnose(rendered, reference, clip_duration=expected.duration)
    return {
        "schema_version": 1,
        "status": ("inconclusive" if not reference else
                   "failed" if differences or outcome.status == "failed" else "passed"),
        "expected_word_count": len(reference), "recognized_word_count": len(rendered),
        "differences": differences, "timing": asdict(outcome),
        "interpretation": (
            "ASR disagreements require listening; exact matches alone do not certify seams"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Project.load(args.project)
    master = project.outputs.get("program", [])
    expected_path = project.transcripts.get("edited")
    if not master or not master[0].is_file() or expected_path is None:
        parser.error("render program first: master and edited transcript are required")
    expected = Transcript.load(expected_path)
    engine = WhisperEngine(TranscribeSettings(
        model=args.model, device=args.device, language=expected.language,
        mode="quality", use_cache=False,
    ))
    try:
        actual = engine.transcribe(master[0])
    finally:
        engine.backend.unload()  # type: ignore[attr-defined]
    report = compare_words(expected, actual)
    report.update(master=str(master[0]), model=args.model)
    output = args.output or project.work_dir / "acceptance" / "program-word-check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
