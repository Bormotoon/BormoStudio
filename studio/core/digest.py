"""Cached SHA-256 digests of large source files and stage artifacts.

A digest is reused while the file's identity — ``(dev, ino, size, mtime_ns,
ctime_ns)`` — and a sample hash of its first, middle and last 64 KiB are
unchanged. Rewriting a file and restoring its mtime still changes ``ctime``,
so content identity stays exact without rereading tens of gigabytes on every
planning pass. The cache lives in ``<work_dir>/cache/digests.json`` and in
process memory; a lost concurrent update only costs one recomputation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SAMPLE_BYTES = 64 * 1024
CACHE_SCHEMA = "studio/digests/1"
# Timestamps of coarse filesystems (FAT, HFS+, some network shares) cannot tell
# two writes within the same tick apart: such entries are trusted only once the
# file is this much older than the moment it was hashed (git's racy-clean rule).
RACY_WINDOW_NS = 2_000_000_000

_lock = threading.Lock()
_memory: dict[str, dict[str, Any]] = {}
REHASH_ENV = "UVF_REHASH"
_rehash = os.environ.get(REHASH_ENV) == "1"


def set_rehash(enabled: bool) -> None:
    """Ignore the on-disk cache (``uvf run --rehash`` / ``scan.verify_digests=always``).

    Each file is still hashed at most once per process; stage workers inherit
    the choice through the environment.
    """
    global _rehash
    _rehash = bool(enabled)
    if _rehash:
        os.environ[REHASH_ENV] = "1"
    else:
        os.environ.pop(REHASH_ENV, None)


def configure_from_settings(settings: dict[str, Any]) -> None:
    scan = settings.get("scan", {}) if isinstance(settings, dict) else {}
    if isinstance(scan, dict) and scan.get("verify_digests") == "always":
        set_rehash(True)


def cache_path(work_dir: Path) -> Path:
    return work_dir / "cache" / "digests.json"


def file_digest(path: Path, cache_dir: Path | None = None) -> str:
    """SHA-256 of ``path``, reused while its stat identity and sample match.

    ``cache_dir`` is the project work directory (``_studio``); without it only
    the in-process cache is used.
    """
    resolved = Path(os.path.abspath(path))
    stat = resolved.stat()
    key = str(resolved)
    identity = _identity(stat)
    with _lock:
        cached = _memory.get(key)
    if cached is None and cache_dir is not None and not _rehash:
        cached = _load(cache_path(cache_dir)).get(key)
    if cached is not None and cached.get("stat") == identity and _trusted(cached, stat):
        sample = _sample(resolved, stat.st_size)
        if cached.get("sample") == sample:
            with _lock:
                _memory[key] = cached
            return str(cached["sha256"])
    sample = _sample(resolved, stat.st_size)
    digest = _full_hash(resolved)
    entry = _entry(identity, sample, digest)
    # The file may have changed while it was read: keep the result for this call
    # but cache it only when the identity is still the one that was hashed.
    if _identity(resolved.stat()) == identity:
        _remember(key, entry, cache_dir)
    return digest


def remember_digest(path: Path, digest: str, cache_dir: Path | None) -> None:
    """Record a digest computed elsewhere (a freshly published artifact)."""
    resolved = Path(os.path.abspath(path))
    stat = resolved.stat()
    _remember(str(resolved), _entry(_identity(stat), _sample(resolved, stat.st_size), digest),
              cache_dir)


def clear_memory() -> None:
    with _lock:
        _memory.clear()


def _identity(stat: os.stat_result) -> list[int]:
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _entry(identity: list[int], sample: str, digest: str) -> dict[str, Any]:
    return {"stat": identity, "sample": sample, "sha256": digest, "recorded_ns": time.time_ns()}


def _coarse(stat: os.stat_result) -> bool:
    return stat.st_mtime_ns % 1_000_000_000 == 0 and stat.st_ctime_ns % 1_000_000_000 == 0


def _trusted(entry: dict[str, Any], stat: os.stat_result) -> bool:
    if not _coarse(stat):
        return True
    newest = max(stat.st_mtime_ns, stat.st_ctime_ns)
    return int(entry.get("recorded_ns", 0)) - newest >= RACY_WINDOW_NS


def _sample(path: Path, size: int) -> str:
    sample = hashlib.blake2b(digest_size=16)
    sample.update(str(size).encode())
    with path.open("rb") as handle:
        offsets = sorted({0, max(0, size // 2 - SAMPLE_BYTES // 2), max(0, size - SAMPLE_BYTES)})
        for offset in offsets:
            handle.seek(offset)
            sample.update(handle.read(SAMPLE_BYTES))
    return sample.hexdigest()


def _full_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remember(key: str, entry: dict[str, Any], cache_dir: Path | None) -> None:
    with _lock:
        _memory[key] = entry
    if cache_dir is None:
        return
    target = cache_path(cache_dir)
    with contextlib.suppress(OSError):
        entries = _load(target)
        entries[key] = entry
        _atomic_write(target, {"schema": CACHE_SCHEMA, "entries": entries})


def _load(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != CACHE_SCHEMA:
        return {}
    entries = data.get("entries")
    return {
        str(key): value for key, value in entries.items() if isinstance(value, dict)
    } if isinstance(entries, dict) else {}


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".digests.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
