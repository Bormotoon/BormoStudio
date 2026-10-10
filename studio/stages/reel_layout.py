"""Reel layouts beyond plain video: a hook-title band and an audiogram.

Both draw the title with libass from the packaged fonts. The title layout
places the framed picture under a band of ``title_band`` of the height; the
audiogram needs only audio and draws its waveform in the middle of the frame.
"""

from __future__ import annotations

import shutil
import tempfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from studio.core.proc import run_logged


def title_band_height(settings: dict[str, Any]) -> int:
    height = int(settings.get("height", 1920))
    return max(2, round(height * float(settings.get("title_band", 0.22)) / 2) * 2)


def reel_title(moment: dict[str, Any], settings: dict[str, Any], override: str | None) -> str:
    """The text a title layout shows: the reel override, then the chosen field."""
    if override is not None and override.strip():
        return override.strip()
    first = str(settings.get("title_field", "hook"))
    second = "title" if first == "hook" else "hook"
    return str(moment.get(first) or moment.get(second) or "").strip()


def _ass_color(value: str) -> str:
    return "&H00" + value[5:7] + value[3:5] + value[1:3]


def _clock(seconds: float) -> str:
    ticks = round(seconds * 100)
    hours, ticks = divmod(ticks, 360000)
    minutes, ticks = divmod(ticks, 6000)
    secs, ticks = divmod(ticks, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{ticks:02}"


def title_ass(text: str, duration: float, settings: dict[str, Any]) -> str:
    """One centered title event inside the top band, sized for the band."""
    width, height = int(settings.get("width", 1080)), int(settings.get("height", 1920))
    band = title_band_height(settings)
    seconds = float(settings.get("title_seconds", 0.0))
    end = duration if seconds <= 0 else min(duration, seconds)
    # Two lines of title fit in the band; long titles wrap within the margins.
    size = max(12, round(band * 0.3))
    margin = max(8, round(width * 0.06))
    color = _ass_color(str(settings.get("title_color", "#FFFFFF")))
    font = str(settings.get("title_font", "Montserrat Black"))
    clean = " ".join(text.replace("\\", "").replace("{", "").replace("}", "").split())
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Title,{font},{size},{color},{color},&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,0,0,5,{margin},{margin},0,1\n"
        "\n[Events]\nFormat: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,{_clock(0)},{_clock(end)},Title,,0,0,0,,"
        f"{{\\an5\\pos({width // 2},{band // 2})}}{clean}\n"
    )


def _escape(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:")


def _encode(command: list[str], output: Path, timeout: float) -> None:
    result = run_logged(
        [*command, "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(output)],
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)


def packaged_fonts(folder: Path) -> str:
    """Copy the packaged fonts to a filter-safe folder; return it escaped for libass."""
    fonts = folder / "fonts"
    with as_file(files("studio.resources") / "fonts") as packaged:
        shutil.copytree(packaged, fonts, ignore=shutil.ignore_patterns("licenses"))
    return _escape(fonts)


def _title_filter(folder: Path, text: str, duration: float, settings: dict[str, Any]) -> str:
    """A libass filter for the title, with fonts and script copied to a safe folder."""
    script = folder / "title.ass"
    script.write_text(title_ass(text, duration, settings), encoding="utf-8")
    return f"subtitles=filename='{_escape(script)}':fontsdir='{packaged_fonts(folder)}'"


def compose_title(
    video: Path, output: Path, text: str, duration: float, settings: dict[str, Any]
) -> None:
    """Put a picture framed to width x (height - band) under the title band."""
    width, height = int(settings.get("width", 1080)), int(settings.get("height", 1920))
    band = title_band_height(settings)
    background = str(settings.get("background_color", "#111114")).replace("#", "0x")
    with tempfile.TemporaryDirectory(prefix="uvf-title-") as temporary:
        filters = f"pad={width}:{height}:0:{band}:color={background},setsar=1"
        if text:
            filters += "," + _title_filter(Path(temporary), text, duration, settings)
        _encode(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-vf", filters,
             "-map", "0:v:0", "-map", "0:a:0?", "-c:a", "copy"],
            output,
            timeout=max(600, duration * 20),
        )


def render_audiogram(
    audio: Path, output: Path, text: str, duration: float, settings: dict[str, Any]
) -> None:
    """A waveform reel from audio alone: background, title band and live waveform."""
    width, height = int(settings.get("width", 1080)), int(settings.get("height", 1920))
    fps = str(settings.get("render_fps", "25"))
    wave_height = max(2, round(height * 0.25 / 2) * 2)
    background = str(settings.get("background_color", "#111114")).replace("#", "0x")
    wave = str(settings.get("waveform_color", "#FFFFFF")).replace("#", "0x")
    with tempfile.TemporaryDirectory(prefix="uvf-audiogram-") as temporary:
        graph = (
            f"[1:a]aformat=channel_layouts=mono,showwaves=s={width}x{wave_height}"
            f":mode=cline:scale=sqrt:draw=full:rate={fps}:colors={wave}[wave];"
            f"[0:v][wave]overlay=0:(H-h)/2:shortest=1,setsar=1"
        )
        if text:
            graph += "," + _title_filter(Path(temporary), text, duration, settings)
        graph += "[v]"
        _encode(
            ["ffmpeg", "-nostdin", "-v", "error", "-y",
             "-f", "lavfi", "-i", f"color=c={background}:s={width}x{height}:r={fps}"
             f":d={duration:.3f}",
             "-i", str(audio), "-filter_complex", graph, "-map", "[v]", "-map", "1:a:0",
             "-t", f"{duration:.3f}", "-c:a", "aac", "-b:a", "192k"],
            output,
            timeout=max(600, duration * 20),
        )
