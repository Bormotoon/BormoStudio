"""FCPXML multicam resources from an aligned sequence with affine time maps."""

import xml.etree.ElementTree as ET
from pathlib import Path

from studio.core.timeline import EditMap
from studio.stages.export import Sequence, _frame_time, _parse_time, write_fcpxml


def _child(parent: ET.Element, path: str) -> ET.Element:
    found = parent.find(path)
    if found is None:
        raise ValueError(f"generated FCPXML lacks {path}")
    return found


def _attribute(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None:
        raise ValueError(f"generated FCPXML element {element.tag} lacks {name}")
    return value


def write_multicam(sequence: Sequence, output: Path, *, media_base: Path,
                  version: str = "1.9", camera_plan: EditMap | None = None) -> Path:
    videos = [clip for clip in sequence.clips if clip.has_video]
    if not videos:
        raise ValueError("multicam requires video angles")
    write_fcpxml(sequence, output, media_base=media_base, version=version)
    tree = ET.parse(output)
    root = tree.getroot()
    resources = _child(root, "resources")
    spine = _child(_child(root, ".//project/sequence"), "spine")
    bed = _child(spine, "gap")
    media = ET.SubElement(resources, "media", id="mc-resource", name=sequence.name + " Multicam")
    multicam = ET.SubElement(media, "multicam", format="r1", tcStart="0s", tcFormat="NDF")
    angles: dict[tuple[bool, int], ET.Element] = {}
    elements: dict[int, ET.Element] = {}
    audio_angles: dict[int, ET.Element] = {}
    for clip, element in zip(sequence.clips, list(bed.findall("asset-clip")), strict=True):
        elements[id(clip)] = element
        if clip.asset_id.startswith("ambience-"):
            continue
        key = (clip.has_video, clip.lane)
        if key not in angles:
            angle_id = f"angle-{len(angles) + 1}"
            angles[key] = ET.SubElement(multicam, "mc-angle", name=clip.path.stem,
                                       angleID=angle_id)
        element.attrib.pop("lane", None)
        angles[key].append(element)
    for camera in videos:
        room = next((clip for clip in sequence.clips
                     if clip.asset_id == f"ambience-{camera.asset_id}"
                     and abs(clip.timeline_start_s - camera.timeline_start_s) < 1e-9
                     and abs(clip.duration_s - camera.duration_s) < 1e-9), None)
        if room is None:
            continue
        voice = next((clip for clip in sequence.clips
                      if clip.asset_id == f"voice-{camera.asset_id}"
                      and abs(clip.timeline_start_s - camera.timeline_start_s) < 1e-9
                      and abs(clip.duration_s - camera.duration_s) < 1e-9), None)
        key = (False, voice.lane if voice else room.lane)
        if key not in angles:
            angles[key] = ET.SubElement(multicam, "mc-angle", name=f"Audio {camera.path.stem}",
                                       angleID=f"angle-{len(angles) + 1}")
        angle = angles[key]
        primary = elements[id(voice or camera)]
        if voice:
            angle.remove(primary)
        else:
            from copy import deepcopy

            primary = deepcopy(primary)
            primary.set("srcEnable", "audio")
            for adjustment in primary.findall("adjust-volume"):
                primary.remove(adjustment)
        composite = ET.SubElement(angle, "clip", name=f"Voice and ambience {camera.path.stem}",
                                  offset=_frame_time(camera.timeline_start_s, sequence.fps),
                                  start="0s", duration=_frame_time(camera.duration_s, sequence.fps))
        primary.set("offset", "0s")
        composite.append(primary)
        background = elements[id(room)]
        background.set("offset", "0s")
        background.set("lane", "-1")
        composite.append(background)
        audio_angles[id(camera)] = angle
    for angle in angles.values():
        angle[:] = sorted(angle, key=lambda element: _parse_time(element.get("offset", "0s")))
    # Partition coverage so the project never selects an unavailable angle.
    boundaries = sorted({0.0} | {value for clip in sequence.clips
                                 for value in (clip.timeline_start_s,
                                               clip.timeline_start_s + clip.duration_s)} | {
        value for item in (camera_plan.keep if camera_plan else ())
        for value in (item.start_s, item.end_s)
    })
    spine.remove(bed)
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end <= start:
            continue
        available = [clip for clip in videos if clip.timeline_start_s <= start + 1e-9
                     and clip.timeline_start_s + clip.duration_s >= end - 1e-9]
        if not available:
            ET.SubElement(spine, "gap", offset=_frame_time(start, sequence.fps), start="0s",
                          duration=_frame_time(end - start, sequence.fps))
            continue
        desired = (next((item.camera_id for item in camera_plan.keep
                         if item.start_s <= start and item.end_s >= end), None)
                   if camera_plan else None)
        selected = next((clip for clip in available if clip.asset_id == desired), available[0])
        element = ET.SubElement(spine, "mc-clip", ref="mc-resource", name=sequence.name,
                                offset=_frame_time(start, sequence.fps),
                                start=_frame_time(start, sequence.fps),
                                duration=_frame_time(end - start, sequence.fps))
        voices = [clip for clip in sequence.clips if not clip.has_video
                  and clip.asset_id == f"voice-{selected.asset_id}"
                  and clip.timeline_start_s <= start + 1e-9
                  and clip.timeline_start_s + clip.duration_s >= end - 1e-9]
        video_angle = _attribute(angles[(True, selected.lane)], "angleID")
        audio_angle = audio_angles.get(id(selected))
        ET.SubElement(element, "mc-source", angleID=video_angle,
                      srcEnable="video" if voices or audio_angle is not None else "all")
        if audio_angle is not None:
            ET.SubElement(element, "mc-source", angleID=_attribute(audio_angle, "angleID"),
                          srcEnable="audio")
        elif voices:
            ET.SubElement(element, "mc-source",
                          angleID=_attribute(angles[(False, voices[0].lane)], "angleID"),
                          srcEnable="audio")
    # Markers use multicam source time within the corresponding project clip.
    for marker in sequence.markers:
        for element in spine.findall("mc-clip"):
            start = _parse_time(_attribute(element, "start"))
            end = start + _parse_time(_attribute(element, "duration"))
            if start <= marker.at_s < end:
                ET.SubElement(element, "marker", start=_frame_time(marker.at_s, sequence.fps),
                              duration=_frame_time(1 / float(sequence.fps), sequence.fps),
                              value=marker.text, note=marker.kind)
                break
    ET.indent(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output
