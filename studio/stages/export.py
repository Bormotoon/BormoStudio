"""Neutral edit sequence and Final Cut FCPXML export."""

from __future__ import annotations

import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import quote

from studio.core.project import AssetKind, AssetRole, Project, describe_artifact, stable_fingerprint
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
    # Multicam angle this clip belongs to: a camera group or a recorder.
    angle: str | None = None
    # Length of the media file in its own clock, when known.
    source_duration_s: float | None = None
    # Name shown in the NLE; the file name when unset.
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or self.path.stem


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
    # Files with no place on the timeline; Final Cut still lists them in the event.
    unplaced: tuple[SequenceClip, ...] = ()


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
             if key.startswith(("sync:", "sync-tracks:", "ambience:", "recorder:"))
             or key in {"roughcut", "speakers"}
            for path in paths if path.is_file()
        ]
        return stable_fingerprint(
            "export-v18", project.assets, project.placements, project.audio_warp_maps,
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
        conf = context.settings.get("export", {})
        conf = conf if isinstance(conf, dict) else {}
        markers = str(conf.get("markers", "cuts"))
        marker_min_s = float(conf.get("marker_min_s", 0.0))
        # Final Cut event, project and multicam carry this name; files stay Studio.*.
        name = str(conf.get("name") or "").strip() or context.project.source_dir.name
        camera_audio = export_camera_audio(context)

        def located(sequence: Sequence) -> Sequence:
            return replace(sequence, clips=tuple(
                replace(clip, path=context.project.work_dir
                        / clip.path.relative_to(context.work_dir))
                if clip.path in camera_audio.values() else clip for clip in sequence.clips
            ))

        sequence = located(build_sequence(context.project, name, use_edit=bool(use_edit),
                                          retimed=True, camera_audio=camera_audio,
                                          marker_kinds=markers, marker_min_s=marker_min_s))
        fcpxml = context.work_dir / "export" / "Studio.fcpxml"
        xmeml = context.work_dir / "export" / "Studio.xml"
        artifacts = list(camera_audio.values())
        targets = conf.get("targets", ["multicam", "xmeml"])
        # The runner publishes work_dir into the project's work folder.
        published = context.project.work_dir / fcpxml.parent.relative_to(context.work_dir)
        if "multicam" in targets:
            from studio.stages.multicam import write_multicam

            # Angles hold the whole synchronized material; the edit lives in
            # the project as multicam clips, so every cut can be reopened.
            uncut = located(build_sequence(context.project, name, use_edit=False,
                                           retimed=True, camera_audio=camera_audio))
            edit = roughcut_edit(context.project) if use_edit else None
            artifacts.append(write_multicam(
                uncut, fcpxml if "fcpxml" not in targets
                else fcpxml.with_name("Studio-multicam.fcpxml"),
                media_base=context.project.source_dir,
                version=str(conf.get("fcpxml_version", "1.9")),
                published_dir=published, edit=edit, markers=sequence.markers,
                camera_plan=multicam_camera_plan(context.project, context.settings,
                                                bool(use_edit), uncut),
            ))
        if "fcpxml" in targets:
            write_fcpxml(
                sequence, fcpxml, media_base=context.project.source_dir,
                version=str(conf.get("fcpxml_version", "1.9")), published_dir=published,
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
    # The selection stays in original time, the time of the multicam angles.
    plan = speaker_camera_edit(project, edit, turns, mapping,
                               policy=CameraPolicy.from_settings(conf))
    return EditMap("multicam-selection", plan.keep)


def roughcut_edit(project: Project) -> EditMap | None:
    """The accepted rough cut in original time, when the edit removes material."""
    paths = project.outputs.get("roughcut", [])
    if not paths or not paths[0].is_file():
        return None
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    if data.get("mode") != "cut":
        return None
    return EditMap("roughcut", tuple(
        KeepRange(float(item["start"]), float(item["end"]), item.get("camera_id"))
        for item in data.get("keep", [])
    ))


def build_sequence(project: Project, name: str = "Studio", *, use_edit: bool = True,
                    retimed: bool = False,
                    camera_audio: dict[str, Path] | None = None,
                    marker_kinds: str = "all", marker_min_s: float = 0.0) -> Sequence:
    """``marker_kinds``: ``all``, ``cuts`` (accepted cuts only) or ``none``;
    cuts shorter than ``marker_min_s`` get no marker."""
    assets = {asset.id: asset for asset in project.assets}
    clips: list[SequenceClip] = []
    recorders = [placement for placement in project.placements
                 if placement.asset_id in assets
                 and assets[placement.asset_id].role is AssetRole.RECORDER]
    # A placed recorder is exported whole; voices rendered from it repeat it.
    voice_sources = {warp.target_asset_id: warp.source_asset_id
                     for warp in project.audio_warp_maps}
    placed = {placement.asset_id for placement in recorders}
    camera_index = 0
    for placement in project.placements:
        asset = assets.get(placement.asset_id)
        if asset is None or asset.role is not AssetRole.CAMERA:
            continue
        media = asset.manual.get("media_info", {})
        voices = project.outputs.get(f"sync:{asset.id}", [])
        voice = next((path for path in voices if path.is_file()), None)
        replaced_by_recorder = voice is not None and voice_sources.get(asset.id) in placed
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
                rate=rate,
                angle=(asset.group_id or asset.id).upper(),
                **_media_fields(media),
            )
        )
        if voice is not None and not replaced_by_recorder:
            clips.append(SequenceClip(
                f"voice-{asset.id}", voice, placement.in_s, duration,
                placement.offset_s, -camera_index - 1, True, has_video=False,
                rate=rate, angle=None if placed else "Voice",
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
                angle=f"Room {(asset.group_id or asset.id).upper()}",
            ))
            clips[-1] = audio_metadata(clips[-1])
        camera_index += 1
    if not clips:
        raise ValueError("cannot build a sequence without placed camera clips")
    for index, placement in enumerate(recorders):
        asset = assets[placement.asset_id]
        repaired = project.outputs.get(f"recorder:{asset.id}", [])
        # The repaired recorder is the clock its placement was measured on.
        path = next((path for path in repaired if path.is_file()),
                    project.source_dir / asset.path)
        rate = placement.k if retimed else 1.0
        angle = "Recorder" if len(recorders) == 1 else f"Recorder {index + 1}"
        clips.append(audio_metadata(SequenceClip(
            asset.id, path, placement.in_s, placement.duration_s * rate,
            placement.offset_s, -3000 - index, True, has_video=False, rate=rate,
            angle=angle,
            # Unlike the file stem, this never resembles the original recorder,
            # which Final Cut would offer, too short, when relinking by name.
            label=f"{angle} (repaired)" if path != project.source_dir / asset.path else angle,
        )))
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
        for marker in data.get("markers", []) if marker_kinds == "all" else ():
            original = float(marker["at"])
            at = timeline_to_edited(original, edit) if edit is not None else original
            if at is not None:
                markers.append(SequenceMarker(at, marker["text"], marker["kind"]))
        for cut in data.get("cuts", []) if marker_kinds != "none" else ():
            if not cut.get("accepted", True):
                continue
            start, end = float(cut["start"]), float(cut["end"])
            if end < start:
                raise ValueError("cut end precedes start")
            if end - start < marker_min_s:
                continue
            # A removed span has no inverse time. Its marker belongs to the
            # splice, after all retained material preceding the removed span.
            at = (sum(max(0.0, min(start, keep.end_s) - keep.start_s)
                      for keep in edit.keep) if edit is not None else start)
            markers.append(SequenceMarker(at, cut["note"], cut["reason"]))
    placed_ids = {placement.asset_id for placement in project.placements}
    unplaced = tuple(
        SequenceClip(asset.id, project.source_dir / asset.path, 0.0,
                     float(info["duration_s"]), 0.0, 0,
                     bool(info.get("audio_codec")), **_media_fields(info))
        for asset in project.assets
        if asset.kind == AssetKind.VIDEO and asset.id not in placed_ids
        and (info := asset.manual.get("media_info", {})).get("duration_s")
    )
    return Sequence(
        name,
        fps,
        int(media.get("width") or 1920),
        int(media.get("height") or 1080),
        tuple(clips),
        tuple(sorted(markers, key=lambda marker: marker.at_s)),
        unplaced,
    )


def _media_fields(media: dict[str, Any]) -> dict[str, Any]:
    """SequenceClip fields describing a probed video file."""
    return {
        "audio_channels": int(media.get("audio_channels") or 2),
        "source_fps": Fraction(str(media.get("fps") or "25")),
        "source_width": int(media.get("width") or 1920),
        "source_height": int(media.get("height") or 1080),
        "audio_sample_rate": int(media.get("audio_sample_rate") or 48000),
        "audio_depth": (int(media["audio_bits_per_sample"])
                        if media.get("audio_bits_per_sample") else None),
        "source_duration_s": float(media["duration_s"]) if media.get("duration_s") else None,
    }


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
                           audio_depth=audio.getsampwidth() * 8,
                           source_duration_s=audio.getnframes() / audio.getframerate())
    except (wave.Error, EOFError):
        # Python 3.10 cannot read WAVE_FORMAT_EXTENSIBLE / RF64 headers.
        from studio.stages.sync_media import probe

        info = probe(clip.path)
        return replace(clip, audio_channels=info.audio_channels or clip.audio_channels,
                       audio_sample_rate=info.audio_sample_rate or clip.audio_sample_rate,
                       audio_depth=info.audio_bits_per_sample or clip.audio_depth,
                       source_duration_s=info.duration or clip.source_duration_s)


def write_fcpxml(
    sequence: Sequence, output: Path, *, media_base: Path, version: str = "1.9",
    published_dir: Path | None = None,
) -> Path:
    """``published_dir``: where the document ends up, when not beside ``output``;
    relative media links start there."""
    output.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("fcpxml", version=version)
    resources = ET.SubElement(root, "resources")
    formats: dict[tuple[Fraction, int, int], str] = {}

    def format_id(fps: Fraction, width: int, height: int) -> str:
        key = (fps, width, height)
        if key not in formats:
            formats[key] = "r1" if not formats else f"source-format-{len(formats)}"
            element = ET.SubElement(resources, "format", id=formats[key],
                                    frameDuration=f"{fps.denominator}/{fps.numerator}s",
                                    width=str(width), height=str(height))
            # Final Cut resolves a format by its name and rejects unknown ones.
            name = _video_format_name(fps, width, height)
            if name:
                element.set("name", name)
        return formats[key]

    format_id(sequence.fps, sequence.width, sequence.height)
    resource_ids: dict[str, str] = {}
    lengths: dict[str, Fraction] = {}
    for index, clip in enumerate((*sequence.clips, *sequence.unplaced), 2):
        if clip.asset_id in resource_ids:
            continue
        resource_id = f"r{index}"
        resource_ids[clip.asset_id] = resource_id
        lengths[clip.asset_id] = _asset_length(clip, sequence)
        asset_element = ET.SubElement(
            resources,
            "asset",
            id=resource_id,
            name=clip.name,
            start="0s",
            duration=_fraction_time(lengths[clip.asset_id]),
            hasVideo="1" if clip.has_video else "0",
            hasAudio="1" if clip.has_audio else "0",
        )
        asset_element.append(
            ET.Element(
                "media-rep", kind="original-media",
                src=_media_src(clip.path, media_base, published_dir or output.parent),
            )
        )
        if clip.has_video:
            asset_element.set("format", format_id(
                clip.source_fps or sequence.fps, clip.source_width or sequence.width,
                clip.source_height or sequence.height,
            ))
        if clip.has_audio:
            asset_element.set("audioSources", "1")
            asset_element.set("audioChannels", str(clip.audio_channels))
            asset_element.set("audioRate", str(clip.audio_sample_rate))
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=sequence.name)
    # Event clips make every file visible in the browser, not only the project.
    roles = {clip.asset_id: clip.audio_role for clip in (*sequence.clips, *sequence.unplaced)}
    unplaced = {clip.asset_id for clip in sequence.unplaced}
    for asset_element in resources.findall("asset"):
        clip_id = next(key for key, value in resource_ids.items()
                       if value == asset_element.get("id"))
        length = lengths[clip_id]
        if asset_element.get("hasVideo") != "1":
            length = Fraction(math.floor(length * sequence.fps)) / sequence.fps
        element = ET.SubElement(event, "asset-clip", ref=asset_element.attrib["id"],
                                name=asset_element.attrib["name"], offset="0s", start="0s",
                                duration=_fraction_time(length))
        if asset_element.get("hasAudio") == "1":
            element.set("audioRole", roles[clip_id])
        if clip_id in unplaced:
            # A keyword collection gathers what sync could not place.
            ET.SubElement(element, "keyword", start="0s", duration=_fraction_time(length),
                          value="Not synchronized")
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
        audioLayout="stereo",
        audioRate="48k",
    )
    spine = ET.SubElement(sequence_el, "spine")
    bed = ET.SubElement(spine, "gap", name="Timeline", offset="0s", start="0s",
                        duration=_frame_time(duration, sequence.fps))
    for clip in sequence.clips:
        clip = _within_media(clip, lengths[clip.asset_id], sequence.fps)
        element = ET.SubElement(
            bed,
            "asset-clip",
            ref=resource_ids[clip.asset_id],
            name=clip.name,
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


def _video_format_name(fps: Fraction, width: int, height: int) -> str | None:
    rates = {Fraction(24000, 1001): "2398", Fraction(24): "24", Fraction(25): "25",
             Fraction(30000, 1001): "2997", Fraction(30): "30", Fraction(50): "50",
             Fraction(60000, 1001): "5994", Fraction(60): "60"}
    sizes = {(1920, 1080): "1080p", (1280, 720): "720p", (3840, 2160): "3840x2160p",
             (4096, 2160): "4096x2160p", (2048, 1080): "2048x1080p"}
    if fps not in rates or (width, height) not in sizes:
        return None
    return f"FFVideoFormat{sizes[width, height]}{rates[fps]}"


def _asset_length(clip: SequenceClip, sequence: Sequence) -> Fraction:
    """Media length on its own grid; Final Cut refuses clips that run past it."""
    if clip.source_duration_s:
        if clip.has_video:
            fps = clip.source_fps or sequence.fps
            return math.floor(clip.source_duration_s * fps + 1e-6) / fps
        rate = clip.audio_sample_rate
        return Fraction(math.floor(clip.source_duration_s * rate + 1e-6), rate)
    used = max(item.source_in_s + item.duration_s / item.rate for item in sequence.clips
               if item.asset_id == clip.asset_id)
    return Fraction(_frames(used, sequence.fps)) / sequence.fps


def _within_media(clip: SequenceClip, length: Fraction, fps: Fraction) -> SequenceClip:
    """Shorten a clip by its frame rounding where that would pass the media end."""
    start = Fraction(_frames(clip.source_in_s, fps)) / fps
    available = float(min(length - start, length - Fraction(clip.source_in_s))) * clip.rate
    if _frames(clip.duration_s, fps) / float(fps) <= available + 1e-9:
        return clip
    return replace(clip, duration_s=max(0.0, math.floor(available * float(fps) + 1e-6)
                                        / float(fps)))


def _fraction_time(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}s"


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
            ET.SubElement(item, "name").text = clip.name
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


def _media_src(path: Path, base: Path, document: Path) -> str:
    """Media under ``base`` is linked relative to the document, which Final Cut
    resolves from the document's own folder; anything else gets a file URL."""
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()):
        return resolved.as_uri()
    return quote(Path(os.path.relpath(resolved, document.resolve())).as_posix())


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
