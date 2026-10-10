import pytest

from studio.stages.events import EventType, StageEvent


def test_event_round_trip() -> None:
    event = StageEvent(EventType.PROGRESS, "scan", {"value": 0.5, "message": "half"})
    assert StageEvent.from_json(event.to_json()) == event


@pytest.mark.parametrize(
    "line",
    ["not-json", '{"t":"unknown","stage":"scan"}', '{"t":"progress","stage":"scan","value":2}'],
)
def test_malformed_worker_events_are_rejected(line: str) -> None:
    with pytest.raises(ValueError):
        StageEvent.from_json(line)


def test_worker_streams_throttled_progress_events(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    from studio.core.project import Project
    from studio.stages import worker
    from studio.stages.base import StageOutput
    from studio.stages.events import EventType, StageEvent

    class Slow:
        id = "slow"

        def run(self, context):  # type: ignore[no-untyped-def]
            for step in range(50):
                context.progress(step / 50, f"step {step}")
            context.progress(1.0, "done")
            return StageOutput()

    monkeypatch.setattr(worker, "stage_registry", lambda: {"slow": Slow()})
    project = tmp_path / "project.json"
    Project(tmp_path, tmp_path / "_studio").save(project)
    worker.execute("slow", project, tmp_path / "none.yaml", tmp_path / "out",
                   tmp_path / "result.json")
    events = [StageEvent.from_json(line) for line in capsys.readouterr().out.splitlines()]
    progress = [event for event in events if event.type is EventType.PROGRESS]
    assert progress and len(progress) <= 2  # first update, then the final one
    assert progress[-1].payload == {"value": 1.0, "message": "done"}
    assert events[0].type is EventType.START and events[-1].type is EventType.DONE
    assert Path(tmp_path / "result.json").is_file()
