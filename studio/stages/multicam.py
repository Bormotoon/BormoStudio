"""FCPXML multicam: one angle per camera group and recorder, the edit in the project."""

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from studio.core.timeline import EditMap, KeepRange
from studio.stages.export import (
    Sequence,
    SequenceMarker,
    _frame_time,
    _frames,
    _parse_time,
    write_fcpxml,
)


def _child(parent: ET.Element, path: str) -> ET.Element:
    found = parent.find(path)
    if found is None:
        raise ValueError(f"generated FCPXML lacks {path}")
    return found


def write_multicam(sequence: Sequence, output: Path, *, media_base: Path,
                   version: str = "1.9", published_dir: Path | None = None,
                   edit: EditMap | None = None,
                   camera_plan: EditMap | None = None,
                   markers: tuple[SequenceMarker, ...] = ()) -> Path:
    """Write ``sequence`` (uncut, original time) as a multicam clip and a project.

    One camera with a recorder becomes a synchronized clip instead: there is
    nothing to switch between. The project holds one such clip per kept range
    of ``edit`` and per camera choice of ``camera_plan``, both in original
    time. ``markers`` are in project (edited) time.
    """
    clips = [clip for clip in sequence.clips if clip.angle]
    if not any(clip.has_video for clip in clips):
        raise ValueError("multicam requires video angles")
    fps = sequence.fps
    write_fcpxml(replace(sequence, clips=tuple(clips), markers=()), output,
                 media_base=media_base, version=version, published_dir=published_dir)
    tree = ET.parse(output)
    root = tree.getroot()
    resources = _child(root, "resources")
    sequence_el = _child(root, ".//project/sequence")
    spine = _child(sequence_el, "spine")
    bed = _child(spine, "gap")

    # Video angles first, in order of appearance, then audio-only angles.
    videos_names = list(dict.fromkeys(clip.angle for clip in clips
                                      if clip.angle and clip.has_video))
    names = videos_names + [name for name in dict.fromkeys(clip.angle for clip in clips
                                                           if clip.angle)
                            if name not in videos_names]
    elements = list(bed.findall("asset-clip"))

    def members(name: str) -> list[ET.Element]:
        return sorted((element for clip, element in zip(clips, elements, strict=True)
                       if clip.angle == name),
                      key=lambda element: _parse_time(element.get("offset", "0s")))

    end = max(clip.timeline_start_s + clip.duration_s for clip in clips)
    has_recorder = any(name.startswith("Recorder") for name in names)
    if len(videos_names) == 1 and has_recorder:
        template = _synchronized(f"{sequence.name} Synchronized", videos_names[0],
                                 [name for name in names if name != videos_names[0]],
                                 members, fps)

        def piece(video: str) -> ET.Element:
            return deepcopy(template)
    else:
        angle_ids = {name: f"angle-{index}" for index, name in enumerate(names, 1)}
        recorder = next((angle_ids[name] for name in names if name.startswith("Recorder")),
                        None)
        media = ET.SubElement(resources, "media", id="r-multicam",
                              name=f"{sequence.name} Multicam")
        multicam = ET.SubElement(media, "multicam", format="r1", tcStart="0s",
                                 tcFormat="NDF")
        length = Fraction(0)
        for name in names:
            angle = ET.SubElement(multicam, "mc-angle", name=name, angleID=angle_ids[name])
            length = max(length, _lay_out(angle, members(name)))
        template = ET.Element("mc-clip", ref="r-multicam", name=f"{sequence.name} Multicam",
                              offset="0s", start="0s",
                              duration=_time(math.floor(length * fps), fps))
        _select(template, angle_ids[names[0]], recorder)

        def piece(video: str) -> ET.Element:
            element = ET.Element("mc-clip", ref="r-multicam")
            _select(element, angle_ids[video], recorder)
            return element
    # The clip itself in the event browser, ready to cut with.
    event = _child(root, "./library/event")
    event.insert(0, deepcopy(template))
    keep = edit.keep if edit is not None else (KeepRange(0.0, end),)
    videos = [clip for clip in clips if clip.has_video]
    choices = _choices(keep, videos, camera_plan)
    spine.remove(bed)
    edited = 0.0
    position = 0
    for item, pieces in zip(keep, choices, strict=True):
        for start, stop, video in pieces:
            # Frame positions come from edited time so clips stay contiguous.
            offset = _frames(edited + start - item.start_s, fps)
            duration = _frames(edited + stop - item.start_s, fps) - offset
            if duration <= 0:
                continue
            element = piece(video)
            element.attrib.update(name=sequence.name, offset=_time(offset, fps),
                                  start=_frame_time(start, fps),
                                  duration=_time(duration, fps))
            spine.append(element)
            position = offset + duration
        edited += item.end_s - item.start_s
    sequence_el.set("duration", _time(position, fps))
    for marker in markers:
        at = _frames(marker.at_s, fps)
        for element in spine:
            offset = _frames(_parse_time(element.attrib["offset"]), fps)
            if offset <= at < offset + _frames(_parse_time(element.attrib["duration"]), fps):
                start = _frames(_parse_time(element.attrib["start"]), fps)
                note = ET.Element("marker", start=_time(start + at - offset, fps),
                                  duration=_time(1, fps), value=marker.text, note=marker.kind)
                # Markers precede a synchronized clip's audio source settings.
                tail = next((index for index, child in enumerate(element)
                             if child.tag == "sync-source"), len(element))
                element.insert(tail, note)
                break
    ET.indent(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _synchronized(name: str, video: str, audio: list[str],
                  members: Callable[[str], list[ET.Element]], fps: Fraction) -> ET.Element:
    """One camera's storyline with the recorder (and any room tone) connected below.

    The camera's own sound stays in the clip but is switched off, as Final
    Cut's Synchronize Clips does.
    """
    clip = ET.Element("sync-clip", name=name, offset="0s", start="0s", format="r1",
                      tcFormat="NDF")
    length = _lay_out(clip, members(video))
    anchor = clip[0]
    anchor_offset = Fraction(anchor.attrib["offset"].removesuffix("s"))
    anchor_start = Fraction(anchor.attrib["start"].removesuffix("s"))
    for lane, angle in enumerate(audio, 1):
        for element in members(angle):
            for adjustment in element.findall("adjust-volume"):
                element.remove(adjustment)
            offset = Fraction(element.attrib["offset"].removesuffix("s"))
            # Connected clips use the anchor's own clock.
            element.set("offset", _fraction(anchor_start + offset - anchor_offset))
            element.set("lane", str(-lane))
            anchor.append(element)
            length = max(length, offset + Fraction(element.attrib["duration"].removesuffix("s")))
    clip.set("duration", _time(math.floor(length * fps), fps))
    source = ET.SubElement(clip, "sync-source", sourceID="storyline")
    ET.SubElement(source, "audio-role-source", role="dialogue", active="0")
    return clip


def _select(element: ET.Element, video: str, recorder: str | None) -> None:
    """Picture from ``video``; sound from the recorder when there is one."""
    if recorder is None:
        ET.SubElement(element, "mc-source", angleID=video, srcEnable="all")
    else:
        ET.SubElement(element, "mc-source", angleID=video, srcEnable="video")
        ET.SubElement(element, "mc-source", angleID=recorder, srcEnable="audio")


def _lay_out(angle: ET.Element, members: list[ET.Element]) -> Fraction:
    """An angle is a storyline: holes become gaps and overlaps are trimmed."""
    cursor = Fraction(0)
    for element in members:
        element.attrib.pop("lane", None)
        # Each angle carries its own sound; the project picks what is heard.
        for adjustment in element.findall("adjust-volume"):
            element.remove(adjustment)
        offset = Fraction(element.attrib["offset"].removesuffix("s"))
        duration = Fraction(element.attrib["duration"].removesuffix("s"))
        if offset < cursor:
            trim = cursor - offset
            if trim >= duration:
                continue
            start = Fraction(element.attrib["start"].removesuffix("s"))
            element.set("start", _fraction(start + trim))
            element.set("duration", _fraction(duration - trim))
            element.set("offset", _fraction(cursor))
            offset, duration = cursor, duration - trim
        elif offset > cursor:
            ET.SubElement(angle, "gap", name="Gap", offset=_fraction(cursor), start="0s",
                          duration=_fraction(offset - cursor))
        angle.append(element)
        cursor = offset + duration
    return cursor


def _choices(keep: tuple[KeepRange, ...], videos: list, camera_plan: EditMap | None,
             ) -> list[list[tuple[float, float, str]]]:
    """Video angle per piece of each kept range, in original time.

    A piece follows the camera plan where it names an available camera and
    otherwise stays on the current angle while that angle has material, so a
    new file of the same camera never forces a switch.
    """
    edges = {value for clip in videos
             for value in (clip.timeline_start_s, clip.timeline_start_s + clip.duration_s)}
    if camera_plan is not None:
        edges |= {value for item in camera_plan.keep for value in (item.start_s, item.end_s)}
    order = list(dict.fromkeys(clip.angle for clip in videos))
    current = order[0]
    result: list[list[tuple[float, float, str]]] = []
    for item in keep:
        pieces: list[tuple[float, float, str]] = []
        bounds = sorted({item.start_s, item.end_s}
                        | {value for value in edges if item.start_s < value < item.end_s})
        for start, stop in zip(bounds, bounds[1:], strict=False):
            available = [clip for clip in videos if clip.timeline_start_s <= start + 1e-9
                         and clip.timeline_start_s + clip.duration_s >= stop - 1e-9]
            angles = [angle for angle in order if any(clip.angle == angle for clip in available)]
            desired = item.camera_id or (next(
                (plan.camera_id for plan in camera_plan.keep
                 if plan.start_s <= start + 1e-9 and plan.end_s >= stop - 1e-9), None)
                if camera_plan is not None else None)
            wanted = next((clip.angle for clip in available if desired is not None
                           and (clip.asset_id == desired or clip.angle == desired.upper())),
                          None)
            if wanted is not None:
                current = wanted
            elif angles and current not in angles:
                current = angles[0]
            if pieces and pieces[-1][2] == current and pieces[-1][1] == start:
                pieces[-1] = (pieces[-1][0], stop, current)
            else:
                pieces.append((start, stop, current))
        result.append(pieces)
    return result


def _time(frames: int, fps: Fraction) -> str:
    return f"{frames * fps.denominator}/{fps.numerator}s"


def _fraction(value: Fraction) -> str:
    return "0s" if value == 0 else f"{value.numerator}/{value.denominator}s"
