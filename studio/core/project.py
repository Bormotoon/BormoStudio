"""Versioned project and artifact-manifest persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from studio.core.digest import file_digest
from studio.core.enums import StrEnum
from studio.core.timeline import AudioWarpMap, AudioWarpPiece, SourcePlacement, TimeDomain

PROJECT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


class AssetKind(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    OTHER = "other"


class AssetRole(StrEnum):
    CAMERA = "camera"
    RECORDER = "recorder"
    IGNORE = "ignore"
    UNKNOWN = "unknown"


class ArtifactStatus(StrEnum):
    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"


@dataclass(slots=True)
class Asset:
    id: str
    path: Path
    kind: AssetKind
    role: AssetRole
    device: str | None = None
    group_id: str | None = None
    chapter_of: str | None = None
    manual: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecorderTrack:
    """One microphone lane of a recorder file: one or more of its channels."""

    asset_id: str
    channels: tuple[int, ...]
    speaker: str | None = None
    name: str = ""
    enabled: bool = True
    # Set by an explicit user edit (uvf tracks / the material screen): a rescan
    # never replaces such a lane with its own proposal.
    manual: bool = False

    def __post_init__(self) -> None:
        self.channels = tuple(self.channels)
        if not self.channels or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.channels
        ) or len(set(self.channels)) != len(self.channels):
            raise ValueError("recorder track channels must be unique non-negative integers")
        if self.speaker is not None and not self.speaker.strip():
            self.speaker = None

    @property
    def id(self) -> str:
        return f"{self.asset_id}:{'+'.join(str(item) for item in self.channels)}"


@dataclass(slots=True)
class Project:
    source_dir: Path
    work_dir: Path
    assets: list[Asset] = field(default_factory=list)
    placements: list[SourcePlacement] = field(default_factory=list)
    audio_warp_maps: list[AudioWarpMap] = field(default_factory=list)
    transcripts: dict[str, Path] = field(default_factory=dict)
    outputs: dict[str, list[Path]] = field(default_factory=dict)
    recorder_tracks: list[RecorderTrack] = field(default_factory=list)
    speaker_cameras: dict[str, str] = field(default_factory=dict)
    plan_revision: int = 1
    version: int = 1
    schema_version: int = PROJECT_SCHEMA_VERSION
    producer_version: str = "studio-dev"

    def save(self, path: Path) -> None:
        _atomic_json(path, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "schema_version": self.schema_version,
            "producer_version": self.producer_version,
            "source_dir": str(self.source_dir),
            "work_dir": str(self.work_dir),
            "plan_revision": self.plan_revision,
            "assets": [
                {
                    **asdict(asset),
                    "path": str(asset.path),
                    "kind": asset.kind.value,
                    "role": asset.role.value,
                }
                for asset in self.assets
            ],
            "placements": [asdict(placement) for placement in self.placements],
            "audio_warp_maps": [
                {
                    "id": warp.id,
                    "source_asset_id": warp.source_asset_id,
                    "target_asset_id": warp.target_asset_id,
                    "pieces": [asdict(piece) for piece in warp.pieces],
                    "source_domain": warp.source_domain.value,
                    "strategy": warp.strategy,
                    "evidence": warp.evidence,
                }
                for warp in self.audio_warp_maps
            ],
            "transcripts": {key: str(value) for key, value in self.transcripts.items()},
            "outputs": {key: [str(path) for path in paths] for key, paths in self.outputs.items()},
            "recorder_tracks": [
                {**asdict(track), "channels": list(track.channels)}
                for track in self.recorder_tracks
            ],
            "speaker_cameras": dict(self.speaker_cameras),
        }

    @classmethod
    def load(cls, path: Path) -> Project:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != PROJECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema: {data.get('schema_version')}")
        return cls(
            source_dir=Path(data["source_dir"]),
            work_dir=Path(data["work_dir"]),
            assets=[
                Asset(
                    id=item["id"],
                    path=Path(item["path"]),
                    kind=AssetKind(item["kind"]),
                    role=AssetRole(item["role"]),
                    device=item.get("device"),
                    group_id=item.get("group_id"),
                    chapter_of=item.get("chapter_of"),
                    manual=item.get("manual", {}),
                )
                for item in data.get("assets", [])
            ],
            placements=[SourcePlacement(**item) for item in data.get("placements", [])],
            audio_warp_maps=[
                AudioWarpMap(
                    id=item["id"],
                    source_asset_id=item["source_asset_id"],
                    target_asset_id=item.get("target_asset_id"),
                    pieces=tuple(AudioWarpPiece(**piece) for piece in item["pieces"]),
                    source_domain=TimeDomain(item["source_domain"]),
                    strategy=int(item["strategy"]),
                    evidence=item.get("evidence"),
                )
                for item in data.get("audio_warp_maps", [])
            ],
            transcripts={
                key: Path(value) for key, value in data.get("transcripts", {}).items()
            },
            outputs={
                key: [Path(value) for value in values]
                for key, values in data.get("outputs", {}).items()
            },
            recorder_tracks=[
                RecorderTrack(
                    asset_id=item["asset_id"],
                    channels=tuple(item["channels"]),
                    speaker=item.get("speaker"),
                    name=str(item.get("name", "")),
                    enabled=bool(item.get("enabled", True)),
                    manual=bool(item.get("manual", False)),
                )
                for item in data.get("recorder_tracks", [])
            ],
            speaker_cameras={
                str(key): str(value) for key, value in data.get("speaker_cameras", {}).items()
            },
            plan_revision=int(data.get("plan_revision", 1)),
            version=int(data.get("version", 1)),
            schema_version=int(data["schema_version"]),
            producer_version=str(data.get("producer_version", "unknown")),
        )


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    sha256: str
    size: int


@dataclass(slots=True)
class StageManifest:
    stage: str
    fingerprint: str
    inputs: dict[str, Any]
    artifacts: list[Artifact]
    status: ArtifactStatus
    producer_version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def save(self, path: Path) -> None:
        data = asdict(self)
        data["status"] = self.status.value
        _atomic_json(path, data)

    @classmethod
    def load(cls, path: Path) -> StageManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {data.get('schema_version')}")
        return cls(
            stage=data["stage"], fingerprint=data["fingerprint"], inputs=data["inputs"],
            artifacts=[Artifact(**item) for item in data["artifacts"]],
            status=ArtifactStatus(data["status"]), producer_version=data["producer_version"],
            created_at=data["created_at"], schema_version=data["schema_version"],
        )

    def reusable(self, root: Path, fingerprint: str) -> bool:
        if self.status is not ArtifactStatus.OK or self.fingerprint != fingerprint:
            return False
        return all(_artifact_matches(root, artifact) for artifact in self.artifacts)


def describe_artifact(root: Path, path: Path, cache_dir: Path | None = None) -> Artifact:
    """Describe ``path``; ``cache_dir`` (the work directory) reuses cached digests."""
    resolved = path if path.is_absolute() else root / path
    relative = resolved.relative_to(root)
    return Artifact(str(relative), file_digest(resolved, cache_dir), resolved.stat().st_size)


def stable_fingerprint(*parts: Any) -> str:
    material = json.dumps(
        parts, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _artifact_matches(root: Path, artifact: Artifact) -> bool:
    path = root / artifact.path
    try:
        return (path.stat().st_size == artifact.size
                and file_digest(path, root) == artifact.sha256)
    except OSError:
        return False


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
