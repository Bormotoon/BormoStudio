"""Neutral edit sequence and Final Cut FCPXML export."""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import quote

from studio.core.project import AssetRole, Project, describe_artifact, stable_fingerprint
from studio.core.timeline import EditMap, KeepRange, timeline_to_edited
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput


@dataclass(frozen=True, slots=True)
class SequenceClip:
    asset_id: str
    path: Path
    source_in_s: float
    duration_s: float
    timeline_start_s: float
    lane: int
    has_audio: bool
    has_video: bool = True
    audio_enabled: bool = True
    audio_channels: int = 2
    rate: float = 1.0
    source_fps: Fraction | None = None
    source_width: int | None = None
    source_height: int | None = None
    audio_sample_rate: int = 48000
    audio_depth: int | None = 16
    audio_role: str = "dialogue"


@dataclass(frozen=True, slots=True)
class SequenceMarker:
    at_s: float
    text: str
    kind: str


@dataclass(frozen=True, slots=True)
class Sequence:
    name: str
    fps: Fraction
    width: int
    height: int
    clips: tuple[SequenceClip, ...]
    markers: tuple[SequenceMarker, ...] = ()


class ExportStage:
    id = "export"
    title = "NLE export"
    after: tuple[str, ...] = ("timeline", "roughcut", "speakers")
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ("export", "roughcut", "program", "transcribe.audio_streams")
    optional_after = ("roughcut", "speakers")
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return []

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        return Decision.run() if any(
            asset.role is AssetRole.CAMERA for asset in project.assets
        ) else Decision.skip("no camera assets")

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        inputs = [
            describe_artifact(project.work_dir, path, project.work_dir)
            for key, paths in project.outputs.items()
             if key.startswith(("sync:", "sync-tracks:", "ambience:"))
             or key in {"roughcut", "speakers"}
            for path in paths if path.is_file()
        ]
        return stable_fingerprint(
            "export-v14", project.assets, project.placements, project.audio_warp_maps,
            inputs, settings.get("export", {}), settings.get("roughcut", {}),
            settings.get("program", {}), project.speaker_cameras,
            settings.get("transcribe", {}).get("audio_streams", {}),
            [describe_artifact(project.source_dir, project.source_dir / asset.path,
                               project.work_dir)
             for asset in project.assets if asset.role is AssetRole.CAMERA
             and (project.source_dir / asset.path).is_file()],
        )

    def run(self, context: StageContext) -> StageOutput:
        roughcut = context.settings.get("roughcut", {})
        use_edit = roughcut.get("enabled", True) if isinstance(roughcut, dict) else True
        camera_audio = export_camera_audio(context)
        sequence = build_sequence(context.project, use_edit=bool(use_edit), retimed=True,
                                  camera_audio=camera_audio)
        sequence = replace(sequence, clips=tuple(
            replace(clip, path=context.project.work_dir / clip.path.relative_to(context.work_dir))
            if clip.path in camera_audio.values() else clip for clip in sequence.clips
        ))
        fcpxml = context.work_dir / "export" / f"{sequence.name}.fcpxml"
        xmeml = context.work_dir / "export" / f"{sequence.name}.xml"
        conf = context.settings.get("export", {})
        conf = conf if isinstance(conf, dict) else {}
        artifacts = list(camera_audio.values())
        targets = conf.get("targets", ["fcpxml", "xmeml"])
        if "multicam" in targets:
            from studio.stages.multicam import write_multicam

            multicam_sequence = sequence
            artifacts.append(write_multicam(
                multicam_sequence, fcpxml.with_name(f"{sequence.name}-multicam.fcpxml"),
                media_base=context.project.source_dir,
                version=str(conf.get("fcpxml_version", "1.9")),
                camera_plan=multicam_camera_plan(context.project, context.settings,
                                                bool(use_edit), multicam_sequence),
            ))
        if "fcpxml" in targets:
            write_fcpxml(
                sequence, fcpxml, media_base=context.project.source_dir,
                version=str(conf.get("fcpxml_version", "1.9")),
            )
            artifacts.append(fcpxml)
        if "xmeml" in targets:
            write_xmeml(sequence, xmeml)
            artifacts.append(xmeml)
        if "otio" in targets:
            from studio.stages.export_otio import write_otio

            artifacts.append(write_otio(sequence, fcpxml.with_suffix(".otio")))
        outputs = {**context.project.outputs, "export": artifacts}
        return StageOutput(tuple(artifacts), {"outputs": outputs})


def multicam_camera_plan(project: Project, settings: dict, use_edit: bool,
                        sequence: Sequence) -> EditMap | None:
    from studio.core.recorder_tracks import effective_speaker_cameras
    from studio.stages.camera_selection import CameraPolicy, speaker_camera_edit
    from studio.stages.program import load_edit_map

    conf = settings.get("program", {})
    mapping = effective_speaker_cameras(project, conf.get("speaker_cameras", {}))
    paths = project.outputs.get("roughcut", [])
    edit = None
    if use_edit and paths and paths[0].is_file():
        if json.loads(paths[0].read_text(encoding="utf-8")).get("mode") == "cut":
            edit = load_edit_map(paths[0])
    if not mapping and not (edit and any(item.camera_id for item in edit.keep)):
        return None
    if edit is None:
        end = max(clip.timeline_start_s + clip.duration_s for clip in sequence.clips)
        edit = EditMap("uncut", (KeepRange(0, end),))
    turns = []
    if mapping:
        speakers = project.outputs.get("speakers", [])
        if not speakers or not speakers[0].is_file():
            raise ValueError("multicam speaker selection requires speakers output")
        turns = json.loads(speakers[0].read_text(encoding="utf-8"))
    plan = speaker_camera_edit(project, edit, turns, mapping,
                               policy=CameraPolicy.from_settings(conf))
    # Keep all source angles; only project selection is mapped to edited time.
    ranges = []
    for item in plan.keep:
        start = timeline_to_edited(item.start_s, edit)
        if start is not None:
            ranges.append(KeepRange(start, start + item.end_s - item.start_s, item.camera_id))
    return EditMap("multicam-selection", tuple(ranges))


def build_sequence(project: Project, name: str = "Studio", *, use_edit: bool = True,
                    retimed: bool = False,
                    camera_audio: dict[str, Path] | None = None) -> Sequence:
    assets = {asset.id: asset for asset in project.assets}
    clips: list[SequenceClip] = []
    camera_index = 0
    for placement in project.placements:
        asset = assets.get(placement.asset_id)
        if asset is None or asset.role is not AssetRole.CAMERA:
            continue
        media = asset.manual.get("media_info", {})
        voices = project.outputs.get(f"sync:{asset.id}", [])
        voice = next((path for path in voices if path.is_file()), None)
        rate = placement.k if retimed else 1.0
        duration = placement.duration_s * rate
        clips.append(
            SequenceClip(
                asset.id,
                project.source_dir / asset.path,
                placement.in_s,
                duration,
                placement.offset_s,
                camera_index,
                bool(media.get("audio_codec")),
                audio_enabled=voice is None and asset.id not in (camera_audio or {}),
                audio_channels=int(media.get("audio_channels") or 2),
                rate=rate,
                source_fps=Fraction(str(media.get("fps") or "25")),
                source_width=int(media.get("width") or 1920),
                source_height=int(media.get("height") or 1080),
                audio_sample_rate=int(media.get("audio_sample_rate") or 48000),
                audio_depth=(int(media["audio_bits_per_sample"])
                             if media.get("audio_bits_per_sample") else None),
            )
        )
        if voice is not None:
            clips.append(SequenceClip(
                f"voice-{asset.id}", voice, placement.in_s, duration,
                placement.offset_s, -camera_index - 1, True, has_video=False,
                rate=rate,
            ))
            clips[-1] = audio_metadata(clips[-1])
            lanes = project.outputs.get(f"sync-tracks:{asset.id}", [])
            for index, lane in enumerate(lanes):
                if lane == voice or not lane.is_file():
                    continue
                clips.append(audio_metadata(SequenceClip(
                    f"voice-{asset.id}-track-{index + 1}", lane, placement.in_s, duration,
                    placement.offset_s, -100 - camera_index * 100 - index,
                    True, has_video=False, rate=rate,
                )))
        elif asset.id in (camera_audio or {}):
            assert camera_audio is not None
            clips.append(audio_metadata(SequenceClip(
                f"camera-audio-{asset.id}", camera_audio[asset.id], placement.in_s, duration,
                placement.offset_s, -camera_index - 1, True, has_video=False, rate=rate,
            )))
        room = project.outputs.get(f"ambience:{asset.id}", [])
        if room and room[0].is_file():
            clips.append(SequenceClip(
                f"ambience-{asset.id}", room[0], placement.in_s, duration,
                placement.offset_s, -1000 - camera_index, True, has_video=False,
                audio_channels=int(media.get("audio_channels") or 2),
                rate=rate,
                audio_role="effects",
            ))
            clips[-1] = audio_metadata(clips[-1])
        camera_index += 1
    if not clips:
        raise ValueError("cannot build a sequence without placed camera clips")
    first_asset = assets[clips[0].asset_id]
    media = first_asset.manual.get("media_info", {})
    fps = Fraction(str(media.get("fps") or "25"))
    edit_paths = project.outputs.get("roughcut", [])
    markers: list[SequenceMarker] = []
    if use_edit and edit_paths and edit_paths[0].is_file():
        data = json.loads(edit_paths[0].read_text(encoding="utf-8"))
        edit = None
        if data.get("mode") == "cut":
            edit = EditMap("roughcut", tuple(
                KeepRange(float(item["start"]), float(item["end"]), item.get("camera_id"))
                for item in data.get("keep", [])
            ))
            edited_clips = []
            for clip in clips:
                for keep in edit.keep:
                    start = max(clip.timeline_start_s, keep.start_s)
                    end = min(clip.timeline_start_s + clip.duration_s, keep.end_s)
                    if end <= start:
                        continue
                    edited_start = timeline_to_edited(start, edit)
                    assert edited_start is not None
                    edited_clips.append(replace(
                        clip, source_in_s=clip.source_in_s
                        + (start - clip.timeline_start_s) / clip.rate,
                        timeline_start_s=edited_start, duration_s=end - start,
                    ))
            clips = edited_clips
            if not clips:
                raise ValueError("edit removes all placed camera clips")
        for marker in data.get("markers", []):
            original = float(marker["at"])
            at = timeline_to_edited(original, edit) if edit is not None else original
            if at is not None:
                markers.append(SequenceMarker(at, marker["text"], marker["kind"]))
        for cut in data.get("cuts", []):
            if not cut.get("accepted", True):
                continue
            start, end = float(cut["start"]), float(cut["end"])
            # A removed span has no inverse time. Its marker belongs to the
            # splice, after all retained material preceding the removed span.
            at = (sum(max(0.0, min(start, keep.end_s) - keep.start_s)
                      for keep in edit.keep) if edit is not None else start)
            markers.append(SequenceMarker(at, cut["note"], cut["reason"]))
            if end < start:
                raise ValueError("cut end precedes start")
    return Sequence(
        name,
        fps,
        int(media.get("width") or 1920),
        int(media.get("height") or 1080),
        tuple(clips),
        tuple(sorted(markers, key=lambda marker: marker.at_s)),
    )


def export_camera_audio(context: StageContext) -> dict[str, Path]:
    """Detach selected streams so both NLE formats reference unambiguous PCM."""
    result: dict[str, Path] = {}
    overrides = context.settings.get("transcribe", {}).get("audio_streams", {})
    for asset in context.project.assets:
        if asset.role is not AssetRole.CAMERA:
            continue
        if any(path.is_file() for path in context.project.outputs.get(f"sync:{asset.id}", [])):
            continue
        media = asset.manual.get("media_info", {})
        stream = overrides.get(asset.id, media.get("audio_stream_index"))
        if stream is None:
            continue
        output = context.work_dir / "export" / "camera-audio" / f"{asset.id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        bits = media.get("audio_bits_per_sample") or 24
        codec = "pcm_s16le" if bits <= 16 else "pcm_s24le" if bits <= 24 else "pcm_s32le"
        decoded = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i",
            str(context.project.source_dir / asset.path), "-map", f"0:{stream}",
            "-vn", "-c:a", codec, "-rf64", "auto", str(output),
        ], capture_output=True, text=True)
        if decoded.returncode:
            raise RuntimeError(f"camera audio export failed for {asset.id}: {decoded.stderr}")
        result[asset.id] = output
    return result


def audio_metadata(clip: SequenceClip) -> SequenceClip:
    """Read rendered PCM metadata instead of inheriting camera channel counts."""
    import wave

    try:
        with wave.open(str(clip.path), "rb") as audio:
            return replace(clip, audio_channels=audio.getnchannels(),
                           audio_sample_rate=audio.getframerate(),
                           audio_depth=audio.getsampwidth() * 8)
    except (wave.Error, EOFError):
        # Python 3.10 cannot read WAVE_FORMAT_EXTENSIBLE / RF64 headers.
        from studio.stages.sync_media import probe

        info = probe(clip.path)
        return replace(clip, audio_channels=info.audio_channels or clip.audio_channels,
                       audio_sample_rate=info.audio_sample_rate or clip.audio_sample_rate,
                       audio_depth=info.audio_bits_per_sample or clip.audio_depth)


def write_fcpxml(
    sequence: Sequence, output: Path, *, media_base: Path, version: str = "1.9",
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("fcpxml", version=version)
    resources = ET.SubElement(root, "resources")
    frame = _rational(1 / float(sequence.fps), 1_000_000)
    ET.SubElement(
        resources,
        "format",
        id="r1",
        name="StudioFormat",
        frameDuration=frame,
        width=str(sequence.width),
        height=str(sequence.height),
    )
    resource_ids: dict[str, str] = {}
    formats: dict[tuple[Fraction, int, int], str] = {}
    for index, clip in enumerate(sequence.clips, 2):
        if clip.asset_id in resource_ids:
            continue
        resource_id = f"r{index}"
        resource_ids[clip.asset_id] = resource_id
        asset_element = ET.SubElement(
            resources,
            "asset",
            id=resource_id,
            name=clip.path.stem,
            start="0s",
            duration=_frame_time(max(
                item.source_in_s + item.duration_s / item.rate for item in sequence.clips
                if item.asset_id == clip.asset_id
            ), sequence.fps),
            hasVideo="1" if clip.has_video else "0",
            hasAudio="1" if clip.has_audio else "0",
        )
        asset_element.append(
            ET.Element(
                "media-rep", kind="original-media", src=_media_src(clip.path, media_base)
            )
        )
        if clip.has_video:
            source_fps = clip.source_fps or sequence.fps
            key = (source_fps, clip.source_width or sequence.width,
                   clip.source_height or sequence.height)
            if key not in formats:
                format_id = f"source-format-{len(formats) + 1}"
                formats[key] = format_id
                ET.SubElement(resources, "format", id=format_id,
                              frameDuration=f"{source_fps.denominator}/{source_fps.numerator}s",
                              width=str(key[1]), height=str(key[2]))
            asset_element.set("format", formats[key])
        if clip.has_audio:
            asset_element.set("audioSources", "1")
            asset_element.set("audioChannels", str(clip.audio_channels))
            asset_element.set("audioRate", str(clip.audio_sample_rate))
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="Studio")
    project = ET.SubElement(event, "project", name=sequence.name)
    duration = max(
        clip.timeline_start_s + clip.duration_s for clip in sequence.clips
    )
    sequence_el = ET.SubElement(
        project,
        "sequence",
        format="r1",
        duration=_frame_time(duration, sequence.fps),
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence_el, "spine")
    bed = ET.SubElement(spine, "gap", name="Timeline", offset="0s", start="0s",
                        duration=_frame_time(duration, sequence.fps))
    for clip in sequence.clips:
        element = ET.SubElement(
            bed,
            "asset-clip",
            ref=resource_ids[clip.asset_id],
            name=clip.path.stem,
            offset=_frame_time(clip.timeline_start_s, sequence.fps),
            start=_frame_time(clip.source_in_s, sequence.fps),
            duration=_frame_time(clip.duration_s, sequence.fps),
            lane=str(clip.lane + 1) if clip.has_video else str(clip.lane),
        )
        if clip.has_audio and not clip.audio_enabled:
            ET.SubElement(element, "adjust-volume", amount="-96dB")
        if clip.has_audio:
            element.set("audioRole", clip.audio_role)
        if abs(clip.rate - 1) > 1e-9:
            # FCPXML start is in the adjusted clip clock; timeMap values
            # remain in the original media clock, including nonzero in-points.
            adjusted_start = clip.source_in_s * clip.rate
            element.set("start", _frame_time(adjusted_start, sequence.fps))
            time_map = ET.Element("timeMap", frameSampling="floor", preservesPitch="1")
            ET.SubElement(time_map, "timept", time=_frame_time(adjusted_start, sequence.fps),
                          value=_rational(clip.source_in_s, 1_000_000), interp="linear")
            ET.SubElement(time_map, "timept",
                          time=_rational(
                               _parse_time(element.attrib["start"])
                               + _parse_time(element.attrib["duration"]), 1_000_000),
                          value=_rational(clip.source_in_s + clip.duration_s / clip.rate,
                                          1_000_000), interp="linear")
            element.insert(0, time_map)
    for marker in sequence.markers:
        ET.SubElement(
            bed, "marker", start=_frame_time(marker.at_s, sequence.fps),
            duration=_frame_time(1 / float(sequence.fps), sequence.fps),
            value=marker.text, note=marker.kind,
        )
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def fcpxml_intervals(path: Path) -> list[tuple[str, float, float]]:
    root = ET.parse(path).getroot()
    return [
        (
            element.get("name", ""),
            _parse_time(element.get("offset", "0s")),
            _parse_time(element.get("duration", "0s")),
        )
        for element in root.findall(".//spine//asset-clip")
    ]


def write_xmeml(sequence: Sequence, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("xmeml", version="4")
    sequence_el = ET.SubElement(root, "sequence", id="sequence-1")
    ET.SubElement(sequence_el, "name").text = sequence.name
    _xmeml_rate(sequence_el, sequence.fps)
    duration = max(
        _frames(clip.timeline_start_s + clip.duration_s, sequence.fps)
        for clip in sequence.clips
    )
    ET.SubElement(sequence_el, "duration").text = str(duration)
    for marker in sequence.markers:
        element = ET.SubElement(sequence_el, "marker")
        ET.SubElement(element, "name").text = marker.text
        ET.SubElement(element, "comment").text = marker.kind
        ET.SubElement(element, "in").text = str(_frames(marker.at_s, sequence.fps))
        ET.SubElement(element, "out").text = "-1"
    media = ET.SubElement(sequence_el, "media")
    video = ET.SubElement(media, "video")
    audio = ET.SubElement(media, "audio")
    characteristics = ET.SubElement(ET.SubElement(video, "format"), "samplecharacteristics")
    _xmeml_rate(characteristics, sequence.fps)
    ET.SubElement(characteristics, "width").text = str(sequence.width)
    ET.SubElement(characteristics, "height").text = str(sequence.height)
    ET.SubElement(characteristics, "pixelaspectratio").text = "square"
    ET.SubElement(characteristics, "fielddominance").text = "none"
    channel_tracks = {}
    link_groups = []
    for lane in sorted({clip.lane for clip in sequence.clips}):
        lane_clips = [clip for clip in sequence.clips if clip.lane == lane]
        track = ET.SubElement(video if lane_clips[0].has_video else audio, "track")
        for index, clip in enumerate(
            (item for item in sequence.clips if item.lane == lane), 1
        ):
            source_fps = clip.source_fps or sequence.fps
            item = ET.SubElement(track, "clipitem", id=f"v-{lane}-{index}")
            ET.SubElement(item, "name").text = clip.path.stem
            ET.SubElement(item, "enabled").text = "TRUE"
            ET.SubElement(item, "start").text = str(
                _frames(clip.timeline_start_s, sequence.fps)
            )
            ET.SubElement(item, "end").text = str(
                _frames(clip.timeline_start_s + clip.duration_s, sequence.fps)
            )
            ET.SubElement(item, "in").text = str(
                _frames(clip.source_in_s, source_fps)
            )
            ET.SubElement(item, "out").text = str(
                _frames(clip.source_in_s + clip.duration_s / clip.rate, source_fps)
            )
            ET.SubElement(item, "duration").text = str(
                _frames(clip.timeline_start_s + clip.duration_s, sequence.fps)
                - _frames(clip.timeline_start_s, sequence.fps)
            )
            _xmeml_rate(item, source_fps)
            _xmeml_speed(item, clip.rate, "video" if clip.has_video else "audio")
            file_el = ET.SubElement(item, "file", id=f"file-{clip.asset_id}")
            ET.SubElement(file_el, "name").text = clip.path.name
            ET.SubElement(file_el, "pathurl").text = _xmeml_url(clip.path)
            _xmeml_rate(file_el, source_fps)
            ET.SubElement(file_el, "duration").text = str(max(
                _frames(other.source_in_s + other.duration_s / other.rate, source_fps)
                for other in sequence.clips if other.asset_id == clip.asset_id
            ))
            file_media = ET.SubElement(file_el, "media")
            if clip.has_video:
                sample = ET.SubElement(ET.SubElement(file_media, "video"), "samplecharacteristics")
                _xmeml_rate(sample, source_fps)
                ET.SubElement(sample, "width").text = str(clip.source_width or sequence.width)
                ET.SubElement(sample, "height").text = str(clip.source_height or sequence.height)
            if clip.has_audio:
                file_audio = ET.SubElement(file_media, "audio")
                ET.SubElement(file_audio, "channelcount").text = str(clip.audio_channels)
                sample = ET.SubElement(file_audio, "samplecharacteristics")
                if clip.audio_depth is not None:
                    ET.SubElement(sample, "depth").text = str(clip.audio_depth)
                ET.SubElement(sample, "samplerate").text = str(clip.audio_sample_rate)
            if not clip.has_video:
                source_track = ET.SubElement(item, "sourcetrack")
                ET.SubElement(source_track, "mediatype").text = "audio"
                ET.SubElement(source_track, "trackindex").text = "1"
            elif clip.has_audio:
                # Preserve camera sound as separate, linked channel clipitems;
                # original channels remain available even when voice replaces them.
                group = [item]
                for channel in range(1, clip.audio_channels + 1):
                    key = (lane, channel)
                    if key not in channel_tracks:
                        channel_tracks[key] = ET.SubElement(audio, "track")
                    audio_track = channel_tracks[key]
                    audio_item = ET.SubElement(
                        audio_track, "clipitem", id=f"a-{lane}-{index}-{channel}",
                    )
                    for tag in ("name", "start", "end", "in", "out"):
                        ET.SubElement(audio_item, tag).text = item.findtext(tag)
                    ET.SubElement(audio_item, "enabled").text = (
                        "TRUE" if clip.audio_enabled else "FALSE"
                    )
                    ET.SubElement(audio_item, "duration").text = item.findtext("duration")
                    _xmeml_rate(audio_item, source_fps)
                    _xmeml_speed(audio_item, clip.rate, "audio")
                    ET.SubElement(audio_item, "file", id=f"file-{clip.asset_id}")
                    source_track = ET.SubElement(audio_item, "sourcetrack")
                    ET.SubElement(source_track, "mediatype").text = "audio"
                    ET.SubElement(source_track, "trackindex").text = str(channel)
                    group.append(audio_item)
                link_groups.append(group)
    # Resolve track indices after all external-audio and camera-channel tracks
    # exist. Each member links to the full group, not just its video partner.
    locations = {}
    for media_type, container in (("video", video), ("audio", audio)):
        for track_index, track in enumerate(container.findall("track"), 1):
            for clip_index, item in enumerate(track.findall("clipitem"), 1):
                locations[item.get("id")] = (media_type, track_index, clip_index)
    for group_index, group in enumerate(link_groups, 1):
        for owner in group:
            for linked in group:
                link = ET.SubElement(owner, "link")
                ET.SubElement(link, "linkclipref").text = linked.get("id")
                media_type, track_index, clip_index = locations[linked.get("id")]
                ET.SubElement(link, "mediatype").text = media_type
                ET.SubElement(link, "trackindex").text = str(track_index)
                ET.SubElement(link, "clipindex").text = str(clip_index)
                ET.SubElement(link, "groupindex").text = str(group_index)
    ET.indent(root)
    tree = ET.ElementTree(root)
    with output.open("wb") as handle:
        handle.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n')
        tree.write(handle, encoding="utf-8", xml_declaration=False)
    return output


def _xmeml_speed(item: ET.Element, rate: float, media_type: str) -> None:
    """XMEML constant speed is source frames per timeline frame, in percent."""
    if abs(rate - 1) <= 1e-9:
        return
    effect = ET.SubElement(ET.SubElement(item, "filter"), "effect")
    for name, value in (("name", "Time Remap"), ("effectid", "timeremap"),
                        ("effectcategory", "motion"), ("effecttype", "motion"),
                        ("mediatype", media_type)):
        ET.SubElement(effect, name).text = value
    for parameter_id, name, value in (
        ("variablespeed", "variablespeed", "FALSE"),
        ("speed", "speed", format(100 / rate, ".12g")),
        ("reverse", "reverse", "FALSE"),
        ("frameblending", "frameblending", "FALSE"),
    ):
        parameter = ET.SubElement(effect, "parameter")
        ET.SubElement(parameter, "parameterid").text = parameter_id
        ET.SubElement(parameter, "name").text = name
        ET.SubElement(parameter, "value").text = value


def xmeml_intervals(path: Path) -> list[tuple[str, float, float]]:
    root = ET.parse(path).getroot()
    sequence = root.find("sequence")
    if sequence is None:
        raise ValueError("xmeml has no sequence")
    fps = _xmeml_fps(sequence)
    intervals: list[tuple[str, float, float]] = []
    for item in root.findall(".//video/track/clipitem"):
        start = int(item.findtext("start", "0")) / float(fps)
        end = int(item.findtext("end", "0")) / float(fps)
        intervals.append((item.findtext("name", ""), start, end - start))
    return intervals


def _media_src(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return quote(str(resolved.relative_to(base.resolve())))
    except ValueError:
        return resolved.as_uri()


def _frame_time(seconds: float, fps: Fraction) -> str:
    frames = _frames(seconds, fps)
    return f"{frames * fps.denominator}/{fps.numerator}s"


def _rational(seconds: float, denominator: int) -> str:
    value = Fraction(seconds).limit_denominator(denominator)
    return f"{value.numerator}/{value.denominator}s"


def _parse_time(value: str) -> float:
    raw = value.removesuffix("s")
    return float(Fraction(raw))


def _frames(seconds: float, fps: Fraction) -> int:
    return round(seconds * float(fps))


def _xmeml_rate(parent: ET.Element, fps: Fraction) -> None:
    rate = ET.SubElement(parent, "rate")
    ntsc = fps.denominator != 1
    timebase = round(float(fps)) if ntsc else fps.numerator
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"


def _xmeml_fps(sequence: ET.Element) -> Fraction:
    timebase = int(sequence.findtext("rate/timebase", "25"))
    if sequence.findtext("rate/ntsc") == "TRUE":
        return Fraction(timebase * 1000, 1001)
    return Fraction(timebase, 1)


def _xmeml_url(path: Path) -> str:
    uri = path.resolve().as_uri()
    return "file://localhost/" + uri.removeprefix("file:///")
