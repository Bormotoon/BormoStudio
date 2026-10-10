import hashlib
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from studio.core import digest
from studio.core.project import ArtifactStatus, Project, StageManifest, describe_artifact
from studio.stages.planner import build_plan


@pytest.fixture(autouse=True)
def _fresh_cache():
    digest.clear_memory()
    digest.set_rehash(False)
    yield
    digest.clear_memory()
    digest.set_rehash(False)


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    reads: list[int] = []
    original = digest._full_hash

    def counting(path: Path) -> str:
        reads.append(path.stat().st_size)
        return original(path)

    monkeypatch.setattr(digest, "_full_hash", counting)
    return reads


def test_digest_is_sha256_and_cached_on_disk(tmp_path: Path, counted: list[int]) -> None:
    source = tmp_path / "rec.wav"
    source.write_bytes(os.urandom(300_000))
    work = tmp_path / "_studio"
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest.file_digest(source, work) == expected
    digest.clear_memory()
    assert digest.file_digest(source, work) == expected
    assert counted == [300_000]
    assert digest.cache_path(work).is_file()


def test_changed_content_with_restored_mtime_is_rehashed(
    tmp_path: Path, counted: list[int]
) -> None:
    source = tmp_path / "rec.wav"
    source.write_bytes(b"a" * 200_000)
    before = source.stat()
    first = digest.file_digest(source, tmp_path)
    with source.open("r+b") as handle:  # same size, a byte in the unsampled middle
        handle.seek(150_000)
        handle.write(b"b")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    digest.clear_memory()
    assert digest.file_digest(source, tmp_path) != first
    assert len(counted) == 2


def test_rehash_ignores_disk_cache_but_hashes_once_per_process(
    tmp_path: Path, counted: list[int]
) -> None:
    source = tmp_path / "rec.wav"
    source.write_bytes(b"x" * 1000)
    digest.file_digest(source, tmp_path)
    digest.clear_memory()
    digest.set_rehash(True)
    digest.file_digest(source, tmp_path)
    digest.file_digest(source, tmp_path)
    assert len(counted) == 2


def test_coarse_timestamps_follow_the_racy_rule() -> None:
    second = 1_000_000_000
    coarse = SimpleNamespace(st_mtime_ns=100 * second, st_ctime_ns=100 * second)
    fine = SimpleNamespace(st_mtime_ns=100 * second + 7, st_ctime_ns=100 * second + 9)
    assert not digest._trusted({"recorded_ns": 101 * second}, coarse)
    assert digest._trusted({"recorded_ns": 102 * second}, coarse)
    assert digest._trusted({"recorded_ns": 100 * second + 10}, fine)


def test_corrupted_artifact_with_same_size_and_mtime_is_not_reused(tmp_path: Path) -> None:
    artifact = tmp_path / "voice.wav"
    artifact.write_bytes(b"a" * 4096)
    manifest = StageManifest(
        "sync", "fp", {}, [describe_artifact(tmp_path, artifact, tmp_path)],
        ArtifactStatus.OK, "test",
    )
    assert manifest.reusable(tmp_path, "fp")
    before = artifact.stat()
    artifact.write_bytes(b"b" * 4096)
    os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not manifest.reusable(tmp_path, "fp")


def test_concurrent_writers_keep_a_valid_cache(tmp_path: Path) -> None:
    files = []
    for index in range(16):
        path = tmp_path / f"f{index}.bin"
        path.write_bytes(os.urandom(5000))
        files.append(path)
    threads = [threading.Thread(target=digest.file_digest, args=(path, tmp_path))
               for path in files]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    entries = digest._load(digest.cache_path(tmp_path))
    assert entries  # readable JSON; lost updates are allowed
    for key, entry in entries.items():
        assert entry["sha256"] == hashlib.sha256(Path(key).read_bytes()).hexdigest()


class _SourceStage:
    title = "Source"
    after: tuple[str, ...] = ()

    def __init__(self, stage_id: str, source: Path, work: Path) -> None:
        self.id = stage_id
        self.source = source
        self.work = work

    def requirements(self, settings):
        return []

    def decide(self, project, settings):
        from studio.stages.base import Decision

        return Decision.run()

    def fingerprint(self, project, settings):
        return digest.file_digest(self.source, self.work)

    def run(self, context):
        raise AssertionError


def test_repeated_plan_on_unchanged_project_reads_no_content(
    tmp_path: Path, counted: list[int]
) -> None:
    work = tmp_path / "_studio"
    source = tmp_path / "rec.wav"
    source.write_bytes(os.urandom(100_000))
    artifact = work / "stages" / "x.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(os.urandom(50_000))
    stages = [_SourceStage(name, source, work) for name in ("scan", "prepare")]
    project = Project(tmp_path, work)
    for stage in stages:
        StageManifest(
            stage.id, digest.file_digest(source, work),
            {}, [describe_artifact(work, artifact, work)], ArtifactStatus.OK, "t",
        ).save(work / "manifests" / f"{stage.id}.json")
    build_plan(stages, project, {})
    digest.clear_memory()
    counted.clear()
    plan = build_plan(stages, project, {})
    assert all(item.will_reuse for item in plan.stages)
    assert counted == []
