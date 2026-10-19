from pathlib import Path

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.timeline import SourcePlacement, TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.timeline import TimelineStage


def test_disjoint_transcripts_merge_without_overlap_duplicates(tmp_path: Path) -> None:
    from studio.core.project import Asset, AssetKind, AssetRole

    work = tmp_path / "work"
    work.mkdir()
    assets = [Asset(name, Path(f"{name}.wav"), AssetKind.AUDIO, AssetRole.RECORDER)
              for name in ("rec0", "duplicate", "rec1")]
    paths = {}
    for asset in assets:
        path = work / f"{asset.id}.json"
        Transcript(tmp_path / asset.path, "en", 10, [Segment(
            1, 2, (Word(asset.id, 1, 2),),
        )]).save(path)
        paths[asset.id] = path
    project = Project(tmp_path, work, assets=assets, transcripts=paths, placements=[
        SourcePlacement("rec0", 0, 0, 10), SourcePlacement("duplicate", 0, 0, 10),
        SourcePlacement("rec1", 100, 0, 10),
    ])
    result = TimelineStage().run(StageContext(project, {}, work))
    transcript = Transcript.load(result.artifacts[0])
    assert [(word.text, word.start) for word in transcript.words] == [("rec0", 1), ("rec1", 101)]


def test_timeline_stage_maps_words_without_retranscription(tmp_path: Path) -> None:
    work = tmp_path / "_studio"
    source_path = work / "stages" / "transcribe" / "rec.json"
    source = Transcript(
        Path("rec.wav"),
        "en",
        10,
        [Segment(2, 3, (Word("mapped", 2.25, 2.75),))],
    )
    source.save(source_path)
    project = Project(
        tmp_path,
        work,
        assets=[Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER)],
        placements=[SourcePlacement("rec", -1.0, 0.0, 10.0, 1.01, "text")],
        transcripts={"rec": source_path},
    )
    result = TimelineStage().run(StageContext(project, {}, work))
    timeline = Transcript.load(result.artifacts[0])
    assert timeline.time_domain is TimeDomain.TIMELINE
    assert timeline.words[0].start == 1.272
    assert timeline.words[0].end == 1.777
    assert timeline.map_id == "placement:rec"


def test_term_fixes_correct_the_timeline_transcript(tmp_path: Path) -> None:
    work = tmp_path / "_studio"
    source_path = work / "stages" / "transcribe" / "rec.json"
    Transcript(Path("rec.wav"), "ru", 10, [Segment(1, 4, (
        Word("копируешь", 1, 1.5), Word("со", 1.6, 1.8), Word("стекловерфлею,", 1.9, 2.8),
        Word("то", 3.0, 3.2),
    ))]).save(source_path)
    project = Project(
        tmp_path, work,
        assets=[Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER)],
        transcripts={"rec": source_path},
    )
    settings = {"text": {"term_fixes": {"стекловерфлею": "Stack Overflow"}}}
    result = TimelineStage().run(StageContext(project, settings, work))
    timeline = Transcript.load(result.artifacts[0])
    assert timeline.segments[0].text == "копируешь со Stack Overflow, то"
    assert [word.text for word in timeline.words] == [
        "копируешь", "со", "Stack", "Overflow,", "то"]
    assert timeline.words[2].start >= 1.9 and timeline.words[3].end <= 2.8
    assert timeline.metadata["term_fixes_applied"] == 1
    assert "text.term_fixes" in TimelineStage.settings_keys
