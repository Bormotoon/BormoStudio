"""Recorder track lanes, speaker names and speaker-camera choices.

Scan proposes one track per recorder channel. Users may merge channels into
one lane (a stereo lavalier pair), disable a lane, and name its speaker; those
choices survive rescans while the file keeps enough channels.
"""

from __future__ import annotations

from studio.core.project import Asset, AssetRole, Project, RecorderTrack


def channel_count(asset: Asset) -> int:
    value = asset.manual.get("media_info", {}).get("audio_channels") or 0
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def default_tracks(asset: Asset) -> list[RecorderTrack]:
    return [RecorderTrack(asset.id, (channel,)) for channel in range(channel_count(asset))]


def refresh_tracks(assets: list[Asset], previous: list[RecorderTrack]) -> list[RecorderTrack]:
    """Keep valid earlier lanes; propose per-channel lanes for new recorders."""
    by_id = {asset.id: asset for asset in assets}
    kept = [
        track for track in previous
        if track.asset_id in by_id and max(track.channels) < channel_count(by_id[track.asset_id])
    ]
    covered = {track.asset_id for track in kept}
    result = list(kept)
    for asset in assets:
        if asset.role is AssetRole.RECORDER and asset.id not in covered:
            result.extend(default_tracks(asset))
    order = {asset.id: index for index, asset in enumerate(assets)}
    return sorted(result, key=lambda track: (order[track.asset_id], track.channels))


def set_track(
    project: Project,
    asset_id: str,
    channels: tuple[int, ...],
    *,
    speaker: str | None = None,
    name: str | None = None,
    enabled: bool | None = None,
) -> RecorderTrack:
    """Create or update the lane using exactly ``channels`` of one asset.

    Channels taken into the new lane are removed from other lanes of the same
    file, and those lanes drop out if nothing remains.
    """
    asset = next((item for item in project.assets if item.id == asset_id), None)
    if asset is None:
        raise ValueError(f"unknown asset: {asset_id}; run scan first")
    count = channel_count(asset)
    if not channels or max(channels) >= count or min(channels) < 0:
        raise ValueError(f"{asset_id} has {count} audio channels; got {list(channels)}")
    channels = tuple(sorted(set(channels)))
    existing = next(
        (track for track in project.recorder_tracks
         if track.asset_id == asset_id and track.channels == channels),
        None,
    )
    track = existing or RecorderTrack(asset_id, channels)
    if speaker is not None:
        track.speaker = speaker.strip() or None
    if name is not None:
        track.name = name.strip()
    if enabled is not None:
        track.enabled = enabled
    updated: list[RecorderTrack] = []
    for other in project.recorder_tracks:
        if other is existing:
            updated.append(track)
            continue
        if other.asset_id == asset_id and set(other.channels) & set(channels):
            remaining = tuple(item for item in other.channels if item not in channels)
            if remaining:
                updated.append(RecorderTrack(
                    asset_id, remaining, other.speaker, other.name, other.enabled
                ))
            continue
        updated.append(other)
    if existing is None:
        updated.append(track)
    order = {item.id: index for index, item in enumerate(project.assets)}
    project.recorder_tracks = sorted(
        updated, key=lambda item: (order.get(item.asset_id, len(order)), item.channels)
    )
    return track


def microphone_map(project: Project) -> dict[str, str]:
    """``asset:channel`` → speaker for enabled lanes that name a speaker."""
    return {
        f"{track.asset_id}:{channel}": track.speaker
        for track in project.recorder_tracks
        if track.enabled and track.speaker
        for channel in track.channels
    }


def effective_speaker_cameras(project: Project, configured: object) -> dict[str, str]:
    """Settings give defaults; choices made for this project win."""
    base = dict(configured) if isinstance(configured, dict) else {}
    return {**base, **project.speaker_cameras}


def project_speakers(project: Project) -> list[str]:
    names = {track.speaker for track in project.recorder_tracks if track.speaker}
    names.update(project.speaker_cameras)
    return sorted(names)
