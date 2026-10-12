import json
from pathlib import Path

import studio.stages.reels as reels
from studio.core.project import Project
from studio.core.timeline import TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.reels import ReelsStage


class FakeProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, grammar: str | None = None, refresh: bool = False) -> str:
        self.prompts.append(prompt)
        if "# Transcript digest" in prompt:
            return '{"summary": "A discussion of valid evidence", "topics": ["evidence"]}'
        if "Candidates" in prompt:
            return '{"decisions": []}'
        return json.dumps(
            {
                "moments": [
                    {
                        "start": 0,
                        "end": 40,
                        "title": "Why it matters",
                        "quote": "This is a valid quote",
                        "why": "A clear conflict with useful advice",
                        "score": 8,
                    }
                ]
            }
        )


def test_reels_stage_publishes_ranked_moments(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    project = Project(tmp_path, tmp_path / "_studio")
    path = project.work_dir / "timeline.json"
    path.parent.mkdir(parents=True)
    Transcript(
        tmp_path / "audio.wav",
        "en",
        40,
        [
            Segment(
                0,
                40,
                tuple(
                    Word(word, i, i + 1) for i, word in enumerate("This is a valid quote".split())
                ),
            )
        ],
        time_domain=TimeDomain.TIMELINE,
    ).save(path)
    project.transcripts["timeline"] = path
    metadata = tmp_path / "source.info.json"
    metadata.write_text(
        json.dumps(
            {"title": "Evidence episode", "chapters": [{"start_time": 0, "title": "Opening"}]}
        )
    )
    project.outputs["fetch"] = [metadata]
    turns = tmp_path / "diarization.json"
    speaker_report = tmp_path / "report.json"
    turns.write_text(
        '[{"start": 0, "end": 2, "speaker": "A"},{"start": 2, "end": 5, "speaker": "B"}]'
    )
    speaker_report.write_text('{"time_domain": "timeline"}')
    project.outputs["speakers"] = [turns, speaker_report]
    provider = FakeProvider()
    monkeypatch.setattr(reels, "LlamaProvider", lambda *args: provider)

    output = ReelsStage().run(
        StageContext(project, {"reels": {"enabled": True, "min_duration_s": 0}}, project.work_dir)
    )

    moments = json.loads((project.work_dir / "stages" / "reels" / "moments.json").read_text())
    assert moments[0]["title"] == "Why it matters"
    assert len(output.artifacts) == 4
    report = json.loads(output.artifacts[2].read_text())
    assert report["context_status"] == "ok"
    assert report["speaker_context"] == {"status": "ok", "change_starts": [2.0]}
    assert [review["stage"] for review in report["reviews"]] == ["cleanup", "judge"]
    assert [limit["review_count"] for limit in report["review_limits"]] == [1, 1]
    assert '"evidence"' in provider.prompts[2]
    assert "This is a valid quote" in report["context_digest"]
    assert "Evidence episode" in provider.prompts[0]
    assert "chapter [0.0s source]: Opening" in report["metadata_context"]
    # Like Forge, the overview reaches the prompts that ask for it: scout and judge.
    assert "A discussion of valid evidence" in provider.prompts[1]
    assert "A discussion of valid evidence" in provider.prompts[3]
    assert "A discussion of valid evidence" not in provider.prompts[2]
    for prompt in provider.prompts[1:]:
        assert not {"{requirements}", "{target_candidates}", "{chunk_json}",
                    "{transcript}", "{candidates_json}", "{episode_context}"} & set(
            part for part in prompt.split() if part.startswith("{"))


def test_reels_rejects_invented_quote(tmp_path: Path, monkeypatch) -> None:
    project = Project(tmp_path, tmp_path / "work")
    path = tmp_path / "timeline.json"
    Transcript(
        tmp_path / "audio.wav",
        "en",
        40,
        [Segment(0, 40, (Word("unrelated", 0, 1),))],
        time_domain=TimeDomain.TIMELINE,
    ).save(path)
    project.transcripts["timeline"] = path
    monkeypatch.setattr(reels, "LlamaProvider", lambda *args: FakeProvider())
    output = ReelsStage().run(StageContext(project, {"reels": {"enabled": True}}, project.work_dir))
    assert json.loads(output.artifacts[0].read_text()) == []
    assert len(json.loads(output.artifacts[2].read_text())["quote_rejected"]) == 1


def test_context_failure_preserves_quote_verification(tmp_path: Path, monkeypatch) -> None:
    original = FakeProvider.complete

    def complete(self, prompt, **kwargs):
        if "# Transcript digest" in prompt:
            return '{"summary": 42}'
        return original(self, prompt, **kwargs)

    monkeypatch.setattr(FakeProvider, "complete", complete)
    test_reels_rejects_invented_quote(tmp_path, monkeypatch)
    report = json.loads((tmp_path / "work/stages/reels/report.json").read_text())
    assert report["context_status"] == "failed"
    assert report["episode_context"] is None
    assert report["failures"][0].startswith("episode_context:")


def test_metadata_changes_invalidate_reels_only_when_used(tmp_path: Path) -> None:
    project = Project(tmp_path, tmp_path / "work")
    metadata = tmp_path / "source.info.json"
    project.outputs["fetch"] = [metadata]
    stage = ReelsStage()
    metadata.write_text('{"title": "first"}')
    before = stage.fingerprint(project, {})
    unused = {"reels": {"episode_context": False, "replay": False}}
    disabled = stage.fingerprint(project, unused)
    metadata.write_text('{"title": "second"}')
    assert stage.fingerprint(project, {}) != before
    assert stage.fingerprint(project, unused) == disabled


def test_judge_budget_limits_actual_request(tmp_path: Path, monkeypatch) -> None:
    project = Project(tmp_path, tmp_path / "work")
    path = tmp_path / "timeline.json"
    quotes = ["First valid evidence", "Second different evidence", "Third distinct evidence"]
    Transcript(
        tmp_path / "audio.wav",
        "en",
        180,
        [
            Segment(
                i * 60,
                i * 60 + 40,
                tuple(
                    Word(word, i * 60 + j, i * 60 + j + 1) for j, word in enumerate(quote.split())
                ),
            )
            for i, quote in enumerate(quotes)
        ],
        time_domain=TimeDomain.TIMELINE,
    ).save(path)
    project.transcripts["timeline"] = path
    prompts = []

    class Provider:
        def complete(self, prompt):
            prompts.append(prompt)
            if len(prompts) > 1:
                return '{"decisions": []}'
            return json.dumps(
                {
                    "moments": [
                        {
                            "start": i * 60,
                            "end": i * 60 + 40,
                            "quote": quote,
                            "title": str(i),
                            "score": 9 - i,
                        }
                        for i, quote in enumerate(quotes)
                    ]
                }
            )

    monkeypatch.setattr(reels, "LlamaProvider", lambda *args: Provider())
    output = ReelsStage().run(
        StageContext(
            project,
            {"reels": {"episode_context": False, "cleanup": False, "judge_max_candidates": 1,
                       "min_duration_s": 0}},
            project.work_dir,
        )
    )
    report = json.loads(output.artifacts[2].read_text())
    judge_limit = report["review_limits"][1]
    assert judge_limit["input_count"] == 3
    assert judge_limit["review_count"] == 1
    assert len(judge_limit["budget_excluded_ids"]) == 2
    assert len(prompts) == 2
    assert quotes[0] in prompts[1]
    assert report["reviews"][0]["candidate_ids"][0] in prompts[1]
    for candidate_id in judge_limit["budget_excluded_ids"]:
        assert candidate_id not in prompts[1]


def test_malformed_metadata_falls_back_to_transcript(tmp_path: Path, monkeypatch) -> None:
    project = Project(tmp_path, tmp_path / "work")
    transcript = tmp_path / "timeline.json"
    Transcript(
        tmp_path / "audio.wav",
        "en",
        40,
        [Segment(0, 40, (Word("unrelated", 0, 1),))],
        time_domain=TimeDomain.TIMELINE,
    ).save(transcript)
    project.transcripts["timeline"] = transcript
    metadata = tmp_path / "source.info.json"
    metadata.write_text("not JSON")
    project.outputs["fetch"] = [metadata]
    monkeypatch.setattr(reels, "LlamaProvider", lambda *args: FakeProvider())
    output = ReelsStage().run(StageContext(project, {}, project.work_dir))
    report = json.loads(output.artifacts[2].read_text())
    assert report["context_status"] == "ok"
    assert report["metadata_context"] == ""
    assert report["failures"][0].startswith("episode_metadata:")
