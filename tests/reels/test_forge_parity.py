"""Golden parity of reels selection with the frozen Forge snapshot.

Both pipelines read the same transcript and get the same answers from a
scripted LLM that decides from the structured part of each prompt (the chunk
window, the candidate ids). Whatever differs in the selected moments is a
difference in the Python logic: quotas, gates, cleanup/judge bookkeeping,
snapping, typing, filters and MMR selection.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import zlib
from pathlib import Path
from typing import Any

import pytest

import studio.stages.reels as reels
from studio.core.project import Project
from studio.core.timeline import TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.reels import ReelsStage

TOPICS = (
    "garden soil compost seeds watering harvest tomatoes roots",
    "rocket engine fuel orbit launch payload booster thrust",
    "violin concert rehearsal orchestra melody tempo bow strings",
    "mortgage interest bank savings budget loan inflation rates",
    "mountain hiking trail summit weather tent backpack ridge",
    "kitchen recipe flour oven dough butter knife sauce",
)
FILLER = "really then because maybe also still quite often later every".split()


def _transcript(seed: int, minutes: int) -> Transcript:
    """Sentences of 8-16 words, 0.4 s per word, pauses between sentences."""
    rng = random.Random(seed)
    segments = []
    at = 1.0
    while at < minutes * 60:
        topic = TOPICS[int(at // 300) % len(TOPICS)].split()
        words = []
        for _ in range(rng.randint(8, 16)):
            text = rng.choice(topic) if rng.random() < 0.6 else rng.choice(FILLER)
            words.append(Word(text, round(at, 3), round(at + 0.35, 3)))
            at += 0.4
        words[-1] = Word(words[-1].text + ".", words[-1].start, words[-1].end)
        segments.append(Segment(words[0].start, words[-1].end, tuple(words)))
        at += rng.choice((0.3, 0.6, 1.2))
    return Transcript(Path("audio.wav"), "en", round(at, 3), segments,
                      time_domain=TimeDomain.TIMELINE)


def _crc(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def _answers(transcript: Transcript) -> Any:
    sentences = [(segment.start, segment.end, " ".join(w.text for w in segment.words))
                 for segment in transcript.segments]

    def answer(prompt: str) -> str:
        chunk = re.search(r'\{"chunk_id"[^{}]*\}', prompt)
        if chunk:
            window = json.loads(chunk.group(0))
            inside = [s for s in sentences if window["start"] <= s[0] and s[1] <= window["end"]]
            moments = []
            for number, (start, _end, text) in enumerate(inside):
                if (_crc(text) + number) % 4:
                    continue
                length = (12, 25, 45, 58, 95)[_crc(text) % 5]
                quote = " ".join(text.rstrip(".").split()[:7])
                moments.append({
                    "start": round(start - 1.5, 2), "end": round(start + length, 2),
                    "quote": quote, "title": f"Moment {quote[:24]}",
                    "why": "a concrete payoff with a clear question", "score": 6 + _crc(text) % 4,
                    "reason_codes": ["payoff"],
                })
            return json.dumps({"moments": moments})
        found = re.search(r'\[\{"candidate_id".*\}\]', prompt, re.S)
        if not found:
            return json.dumps({"summary": "An episode about many things", "topics": ["life"]})
        items = json.loads(found.group(0))
        if "duration" in items[0]:  # the judge payload carries durations
            return json.dumps({"reviews": [
                {"candidate_id": item["candidate_id"], "keep": _crc(item["candidate_id"]) % 6 != 0,
                 "score": 5 + _crc(item["candidate_id"]) % 5,
                 "title": f"Judged {item['candidate_id']}", "hook": "Why would anyone do that?"}
                for item in items]})
        return json.dumps({"decisions": [
            {"candidate_id": item["candidate_id"], "keep": _crc(item["candidate_id"]) % 4 != 0}
            for item in items]})

    return answer


# (Forge processing overrides, the same as Studio reels settings)
CASES = {
    "defaults": ({}, {}),
    "no_filters": (
        {"quality_filters": {"min_score": 0, "min_duration": 0, "max_duration": 0}},
        {"min_score": 0, "min_duration_s": 0, "max_duration_s": 0},
    ),
    "fixed_counts": (
        {"clips_per_hour": 0,
         "quality_filters": {"min_score": 0, "min_duration": 0, "max_duration": 0},
         "analysis": {"diversity": {"enabled": False}, "selection": {"max_overlap_ratio": 0}}},
        {"clips_per_hour": 0, "min_score": 0, "min_duration_s": 0, "max_duration_s": 0,
         "diversity": False, "max_overlap_ratio": 0},
    ),
    "tight_review": (
        {"analysis": {"cleanup_cap": 6, "judge_context": {"max_candidates": 4, "head_seconds": 5},
                      "quote_verification": {"min_ratio": 0.7, "min_final_ratio": 0.9},
                      "boundary_snap": {"max_shift_s": 1.0}},
         "quality_filters": {"min_score": 6, "min_duration": 0, "max_duration": 0}},
        {"review_batch_size": 6, "judge_batch_size": 4, "judge_head_s": 5,
         "scout_quote_ratio": 0.7, "min_quote_ratio": 0.9, "snap_max_shift_s": 1.0,
         "min_score": 6, "min_duration_s": 0, "max_duration_s": 0},
    ),
}


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        merged[key] = (_merge(merged[key], value)
                       if isinstance(value, dict) and isinstance(merged.get(key), dict) else value)
    return merged


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize(("seed", "minutes"), [(1, 35), (7, 75)])
def test_selection_matches_forge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 forge_snapshot: None, seed: int, minutes: int,
                                 case: str) -> None:
    forge_overrides, studio_overrides = CASES[case]
    from podcast_reels_forge.config import LlamaCppRoleMapping as roles_type
    from podcast_reels_forge.stages import analyze_stage
    transcript = _transcript(seed, minutes)
    answer = _answers(transcript)
    path = tmp_path / "timeline.json"
    transcript.save(path)

    class ForgeProvider:
        async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
            return answer(prompt)

    monkeypatch.setattr(analyze_stage, "_make_stage_provider",
                        lambda *_args, **_kwargs: ForgeProvider())
    processing = _merge({
        "clips_per_hour": 10,
        "clips": {"stories": {"count": 2, "max_duration": 15},
                  "reels": {"count": 3, "max_duration": 60},
                  "long_reels": {"count": 1, "max_duration": 180},
                  "highlights": {"count": 1, "moments_count": 5}},
        "quality_filters": {"min_score": 7, "min_duration": 15, "max_duration": 180},
        "analysis": {"audio_features": {"enabled": False},
                     "episode_context": {"enabled": False}, "llm_cache": False},
    }, forge_overrides)
    forge = asyncio.run(analyze_stage.run_staged_analysis(
        transcript_path=path, outdir=tmp_path / "forge", provider_name="llama_cpp",
        url="http://127.0.0.1:1/completion", api_key=None,
        roles=roles_type(scout="m", cleanup_refine="m", judge_metadata="m", proofread="m"),
        llama_cpp_conf={"url": "http://127.0.0.1:1/completion", "chunk_seconds": 600,
                        "max_chars_chunk": 6000},
        prompts_conf={"language": "en", "variant": "default"},
        processing_conf=processing,
        quiet=True,
    ))

    class StudioProvider:
        def complete(self, prompt: str, **_kwargs: Any) -> str:
            return answer(prompt)

    monkeypatch.setattr(reels, "LlamaProvider", lambda *_args: StudioProvider())
    project = Project(tmp_path, tmp_path / "_studio")
    project.transcripts["timeline"] = path
    ReelsStage().run(StageContext(
        project, {"reels": {"enabled": True, "episode_context": False, "replay": False,
                            **studio_overrides}},
        project.work_dir))
    studio = json.loads((project.work_dir / "stages" / "reels" / "moments.json").read_text())

    def key(moment: dict[str, Any]) -> tuple[Any, ...]:
        return (moment["candidate_id"], moment["start"], moment["end"], moment["quote"],
                moment["clip_type"], moment["score"], moment["title"], moment.get("hook"),
                tuple(moment.get("hashtags") or ()), moment.get("caption"))

    assert forge, "the scripted answers must select something"
    assert [key(m) for m in studio] == [key(m.to_dict()) for m in forge]
    rejected = json.loads((tmp_path / "forge" / "rejected_candidates.json").read_text())
    ours = json.loads(
        (project.work_dir / "stages" / "reels" / "rejected_candidates.json").read_text())
    assert [(row["candidate_id"], row["rejection_reason"]) for row in ours] == [
        (row["candidate_id"], row["rejection_reason"]) for row in rejected]


@pytest.mark.parametrize("seed", [3, 11])
def test_clip_edges_match_forge(forge_snapshot: None, seed: int) -> None:
    from podcast_reels_forge.analysis.transcript_index import TranscriptIndex as ForgeIndex
    from podcast_reels_forge.utils.clip_intervals import ClipEdges as ForgeEdges
    from podcast_reels_forge.utils.clip_intervals import padded_intervals

    from studio.core.transcript_index import TranscriptIndex
    from studio.reels.clip_edges import ClipEdges, clip_intervals

    transcript = _transcript(seed, 20)
    rng = random.Random(seed)
    bounds = sorted((start, start + rng.uniform(8, 70))
                    for start in (rng.uniform(5, 1100) for _ in range(12)))
    settings = {"edge_lead_in_s": 0.8, "edge_tail_s": 0.4, "edge_guard_s": 0.15,
                "edge_max_extend_s": 3.0, "edge_max_trim_s": 1.2}
    ours = clip_intervals(bounds, TranscriptIndex.from_transcript(transcript),
                          ClipEdges.from_settings(settings), duration=transcript.duration)
    theirs = padded_intervals(
        bounds, 0.0, index=ForgeIndex.from_transcript(transcript.to_dict()),
        edges=ForgeEdges(0.8, 0.4, 0.15, 3.0, 1.2))
    assert ours == pytest.approx(theirs)
