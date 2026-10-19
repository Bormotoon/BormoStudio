"""Forge caption looks for reels: word-timed cues, line layout and highlights.

The looks come from ``subtitle-presets.js`` shared with the web subtitle editor,
so a preset means the same thing in both. A cue is built from transcript words:
it ends at a pause, at the duration or word limit, and is wrapped into at most
``max_lines`` lines, shorter on top. Highlight modes dress the active word:
``word`` colours it, ``fill`` colours everything spoken so far, ``reveal``
types the cue out, ``pop`` scales the word in, ``karaoke`` sweeps it with \\kf.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any

PLAY_RES = (1080, 1920)
REFERENCE_FONT_SIZE = 96

# Family names of the packaged fonts (fc-scan) and their average glyph width
# per pixel of font size, for a font-aware characters-per-line estimate.
FONTS = {
    "assets/fonts/bignoodletoooblique.ttf": ("BigNoodleTooOblique", 0.42),
    "assets/fonts/MontserratBlack.ttf": ("Montserrat Black", 0.72),
    "assets/fonts/OswaldBold.ttf": ("Oswald Bold", 0.5),
    "assets/fonts/PressStart2P-Regular.ttf": ("Press Start 2P", 1.0),
    "assets/fonts/RubikMonoOne-Regular.ttf": ("Rubik Mono One", 0.95),
    "assets/fonts/RussoOne-Regular.ttf": ("Russo One", 0.62),
    "assets/fonts/UnboundedBlack.ttf": ("Unbounded Black", 0.8),
}
HIGHLIGHTS = {"none", "word", "fill", "reveal", "pop", "karaoke"}
CASES = {"none", "upper", "lower", "title"}
PUNCTUATION = {"keep", "periods", "all"}
CENSOR_STYLES = {"middle", "first", "whole"}


@lru_cache(maxsize=1)
def load_presets() -> dict[str, Any]:
    """``{"lookDefaults", "presets", ...}`` from the editor's generated presets file."""
    text = (files("studio.resources") / "subtitle_editor" / "assets" /
            "subtitle-presets.js").read_text(encoding="utf-8")
    value, _end = json.JSONDecoder().raw_decode(text[text.index("{"):])
    if not isinstance(value, dict) or "presets" not in value:
        raise ValueError("subtitle-presets.js has no presets")
    return value


def preset_names() -> list[str]:
    return sorted(load_presets()["presets"])


@dataclass(frozen=True, slots=True)
class CaptionOptions:
    preset: str = "forge"
    highlight: str = "none"
    text_case: str = "none"
    punctuation: str = "keep"
    max_words: int = 0
    max_chars: int = 0
    max_lines: int = 2
    width_ratio: float = 0.74
    font_size: int = REFERENCE_FONT_SIZE
    highlight_color: str = ""
    pause_split_s: float = 0.5
    max_duration_s: float = 7.0
    min_duration_s: float = 0.0
    min_gap_s: float = 0.15
    fade_in_s: float = 0.12
    fade_out_s: float = 0.08
    fade_min_gap_s: float = 0.3
    blur: float = 0.0
    censor: tuple[str, ...] = ()
    censor_style: str = "middle"
    # A change of speaker starts a new cue; colours go to speakers in order.
    split_on_speaker: bool = True
    speaker_colors: tuple[str, ...] = ()
    # "style" keeps the look's row; top/center/bottom move it. The offset is a
    # share of the frame height pushed away from the anchored edge.
    vertical_align: str = "style"
    vertical_offset: float = 0.0
    look: dict[str, Any] = field(default_factory=dict)
    # Looks are designed for 9:16; a landscape reel scales them by its height.
    play_res: tuple[int, int] = PLAY_RES

    @property
    def scale(self) -> float:
        return self.font_size / REFERENCE_FONT_SIZE * self.play_res[1] / PLAY_RES[1]

    @classmethod
    def from_settings(cls, reels: dict[str, Any]) -> CaptionOptions:
        """Preset render defaults, overridden by explicitly set reels.caption_* keys."""
        data = load_presets()
        name = str(reels.get("caption_preset") or "forge")
        if name not in data["presets"]:
            raise ValueError(f"unknown caption preset: {name}")
        preset = data["presets"][name]
        render = preset.get("render", {})

        def chosen(key: str, render_key: str, default: object, unset: object = "") -> Any:
            value = reels.get(key, unset)
            return render.get(render_key, default) if value == unset else value

        return cls(
            preset=name,
            highlight=str(chosen("caption_highlight", "highlight", "none")),
            text_case=str(chosen("caption_case", "text_case", "none")),
            punctuation=str(chosen("caption_punctuation", "strip_punctuation", "keep")),
            max_words=int(chosen("caption_max_words", "max_words_per_cue", 0, -1)),
            max_chars=int(reels.get("caption_max_chars", 0)),
            max_lines=int(chosen("caption_max_lines", "max_lines", 2, 0)),
            width_ratio=float(reels.get("caption_width_ratio", 0.74)),
            font_size=int(reels.get("caption_font_size", REFERENCE_FONT_SIZE)),
            highlight_color=str(reels.get("caption_highlight_color", "")),
            pause_split_s=float(reels.get("caption_pause_split_s", 0.5)),
            max_duration_s=float(reels.get("caption_max_duration_s", 7.0)),
            min_duration_s=float(render.get("min_duration_s", 0.0)),
            min_gap_s=float(reels.get("caption_min_gap_s", 0.15)),
            fade_in_s=float(render.get("fade_in_duration", 0.12)),
            fade_out_s=float(render.get("fade_out_duration", 0.08)),
            fade_min_gap_s=float(reels.get("caption_fade_min_gap_s", 0.3)),
            blur=float(render.get("blur", 0.0)),
            censor=tuple(str(item) for item in reels.get("caption_censor", [])),
            censor_style=str(reels.get("caption_censor_style", "middle")),
            split_on_speaker=bool(reels.get("caption_split_on_speaker", True)),
            speaker_colors=tuple(str(item) for item in reels.get("caption_speaker_colors", [])),
            vertical_align=str(reels.get("caption_vertical_align", "style")),
            vertical_offset=float(reels.get("caption_vertical_offset", 0.0)),
            look={**data["lookDefaults"], **preset.get("look", {})},
        )


@dataclass(frozen=True, slots=True)
class TimedWord:
    text: str
    start: float
    end: float
    speaker: str = ""


@dataclass(frozen=True, slots=True)
class CaptionCue:
    start: float
    end: float
    words: tuple[TimedWord, ...]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


def _censored(word: str, patterns: tuple[str, ...], style: str) -> str:
    core = re.sub(r"[^\w]", "", word.lower())
    for pattern in patterns:
        pattern = pattern.lower().strip()
        hit = core.startswith(pattern[:-1]) if pattern.endswith("*") else core == pattern
        if pattern and hit:
            letters = [index for index, char in enumerate(word) if char.isalnum()]
            if style == "whole":
                hidden = set(letters)
            elif style == "first":
                hidden = set(letters[1:])
            else:
                hidden = set(letters[1:-1]) if len(letters) > 2 else set(letters[1:])
            return "".join("*" if index in hidden else char for index, char in enumerate(word))
    return word


def display_word(text: str, options: CaptionOptions) -> str:
    """A word as shown: censored, with the chosen case and punctuation."""
    word = _censored(text.strip(), options.censor, options.censor_style)
    if options.punctuation == "all":
        word = re.sub(r"(?<!\w)[^\w*]+|[^\w*]+(?!\w)", "", word)
    elif options.punctuation == "periods":
        word = re.sub(r"[.,;:]+$", "", word)
    if options.text_case == "upper":
        word = word.upper()
    elif options.text_case == "lower":
        word = word.lower()
    elif options.text_case == "title":
        word = "-".join(part[:1].upper() + part[1:].lower() for part in word.split("-"))
    return word


def chars_per_line(options: CaptionOptions) -> int:
    if options.max_chars > 0:
        return options.max_chars
    look = options.look
    size = look.get("fontSizePx", REFERENCE_FONT_SIZE) * options.scale
    width = FONTS.get(str(look.get("fontPath")), ("", 0.55))[1]
    usable = options.play_res[0] * options.width_ratio - 2 * float(look.get("outline", 0))
    return max(4, int(usable / (size * width)))


def build_cues(words: list[TimedWord], options: CaptionOptions) -> list[CaptionCue]:
    """Group words into readable cues and keep the timing rules between them."""
    limit = chars_per_line(options) * max(1, options.max_lines)
    shown = [TimedWord(display_word(w.text, options), w.start, w.end, w.speaker) for w in words]
    shown = [word for word in shown if word.text]
    # Whisper sometimes emits closing punctuation as its own word ("«Почему" "?»")
    # and splits hyphenated words ("кто" "-то", "карго" "-культ"); both belong to
    # the word before, never at the start of the next line or cue.
    joined: list[TimedWord] = []
    for word in shown:
        # Even across a speaker boundary: diarization cuts between words, not
        # inside "«Почему?»".
        hyphenated = word.text[:1] == "-" and word.text[1:2].isalpha()
        # A dash stands between words with spaces around it ("да — нет").
        dash = word.text.strip() in {"—", "–", "-"}
        if joined and not dash and (hyphenated or not any(ch.isalnum() for ch in word.text)):
            last = joined[-1]
            joined[-1] = TimedWord(last.text + word.text, last.start, max(last.end, word.end),
                                   last.speaker)
        else:
            joined.append(word)
    shown = joined
    groups: list[list[TimedWord]] = []
    for word in shown:
        current = groups[-1] if groups else None
        if (
            current is None
            or word.start - current[-1].end >= options.pause_split_s
            or (options.split_on_speaker and word.speaker != current[-1].speaker)
            or word.end - current[0].start > options.max_duration_s
            or 0 < options.max_words <= len(current)
            or len(" ".join(w.text for w in [*current, word])) > limit
        ):
            groups.append([word])
        else:
            current.append(word)
    cues: list[CaptionCue] = []
    for index, group in enumerate(groups):
        start, end = group[0].start, group[-1].end
        following = groups[index + 1][0].start if index + 1 < len(groups) else None
        if options.min_duration_s > 0:
            end = max(end, start + options.min_duration_s)
        if following is not None:
            end = min(end, following - options.min_gap_s)
        cues.append(CaptionCue(start, max(end, start + 0.05), tuple(group)))
    return cues


def wrap(texts: list[str], per_line: int, max_lines: int) -> set[int]:
    """Word indices that start a new line; lines bottom-heavy (shorter on top)."""
    if max_lines < 2 or len(texts) < 2 or len(" ".join(texts)) <= per_line:
        return set()
    best: tuple[float, int] | None = None
    for split in range(1, len(texts)):
        top, bottom = len(" ".join(texts[:split])), len(" ".join(texts[split:]))
        over = max(0, top - per_line) + max(0, bottom - per_line)
        # Prefer fitting, then a top line no longer than the bottom one.
        cost = over * 100 + abs(bottom - top) + (50 if top > bottom else 0)
        if best is None or cost < best[0]:
            best = (cost, split)
    assert best is not None
    return {best[1]}


def _ass_color(hex_color: str, opacity: float = 1.0) -> str:
    alpha = round(255 * (1 - max(0.0, min(1.0, opacity))))
    value = hex_color.lstrip("#")
    return f"&H{alpha:02X}{value[4:6]}{value[2:4]}{value[0:2]}".upper()


def _override(hex_color: str) -> str:
    value = hex_color.lstrip("#").upper()
    return f"\\1c&H{value[4:6]}{value[2:4]}{value[0:2]}&"


def style_lines(options: CaptionOptions) -> list[str]:
    look = options.look
    scale = options.scale
    margin = options.play_res[1] / PLAY_RES[1]
    family = FONTS.get(str(look.get("fontPath")), ("BigNoodleTooOblique", 0))[0]

    def style(name: str, primary: str, outline: float, outline_color: str, back: str,
              shadow: float, border: int, size_percent: float) -> str:
        return (
            f"Style: {name},{family},{round(look['fontSizePx'] * scale)},{primary},"
            f"{_ass_color(look['secondaryColor'], look.get('secondaryOp', 1.0))},"
            f"{outline_color},{back},{-1 if look.get('bold') else 0},"
            f"{-1 if look.get('italic') else 0},0,0,{size_percent:g},{size_percent:g},"
            f"{look.get('spacingPx', 0)},0,{border},{outline * scale:g},{shadow * scale:g},"
            f"{look.get('alignment', 2)},{round(look.get('marginL', 140) * margin)},"
            f"{round(look.get('marginR', 140) * margin)},"
            f"{round(look.get('marginV', 470) * margin)},1"
        )

    lines = [style(
        "Default", _ass_color(look["primaryColor"], look.get("primaryOp", 1.0)),
        look.get("outline", 0), _ass_color(look["outlineColor"], look.get("outlineOp", 1.0)),
        _ass_color(look["backColor"], look.get("backOp", 0.5)), look.get("shadow", 0),
        int(look.get("borderStyle", 1)), look.get("scaleX", 100),
    )]
    if look.get("hlEnabled"):
        lines.append(style(
            "Highlight", _ass_color(look["hlColor"], look.get("hlOp", 1.0)),
            look.get("hlOutline", 0),
            _ass_color(look["hlOutlineColor"], look.get("hlOutlineOp", 1.0)),
            _ass_color(look["hlBackColor"], look.get("hlBackOp", 0.5)), look.get("hlShadow", 0),
            int(look.get("hlBorderStyle", 1)), look.get("hlScale", 100),
        ))
    return lines


def _clock(seconds: float) -> str:
    ticks = round(max(0.0, seconds) * 100)
    hours, ticks = divmod(ticks, 360000)
    minutes, ticks = divmod(ticks, 6000)
    secs, ticks = divmod(ticks, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{ticks:02}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "").replace("}", "")


def _fades(cues: list[CaptionCue], index: int, options: CaptionOptions) -> tuple[int, int]:
    cue = cues[index]
    gap_before = cue.start - cues[index - 1].end if index else float("inf")
    gap_after = cues[index + 1].start - cue.end if index + 1 < len(cues) else float("inf")
    fade_in = options.fade_in_s if gap_before >= options.fade_min_gap_s else 0.0
    fade_out = options.fade_out_s if gap_after >= options.fade_min_gap_s else 0.0
    total, duration = fade_in + fade_out, cue.end - cue.start
    if total > duration > 0:
        fade_in, fade_out = fade_in * duration / total, fade_out * duration / total
    return round(fade_in * 1000), round(fade_out * 1000)


def _tag(body: str) -> str:
    return "{" + body + "}" if body else ""


def _join(parts: list[str], breaks: set[int]) -> str:
    return "".join(("\\N" if index in breaks else " ") + part if index else part
                   for index, part in enumerate(parts))


def _position(options: CaptionOptions) -> tuple[str, int]:
    """``\\an``/``\\pos`` tags and the event MarginV for vertical_align/offset.

    MarginV 0 inherits the style, which is what an unmoved cue keeps.
    """
    look = options.look
    width, height = options.play_res
    scale = height / PLAY_RES[1]
    alignment = int(look.get("alignment", 2))
    style_row = "bottom" if alignment in (1, 2, 3) else "center" if alignment in (4, 5, 6) \
        else "top"
    align, offset = options.vertical_align, options.vertical_offset
    margin_v = round(look.get("marginV", 470) * scale)
    tag = ""
    if align != "style":
        column = (alignment - 1) % 3
        tag = f"\\an{ {'bottom': 1, 'center': 4, 'top': 7}[align] + column}"
        if align == "center" and offset:
            # libass ignores MarginV for middle rows, so an offset needs \\pos.
            left = round(look.get("marginL", 140) * scale)
            right = round(look.get("marginR", 140) * scale)
            x = (left, left + (width - left - right) / 2, width - right)[column]
            tag += f"\\pos({round(x)},{round(height / 2 - offset * height)})"
    if align in ("style", "center", style_row):
        return tag, max(1, margin_v + round(offset * height)) if offset else 0
    # Moving a centred look to an edge: start from that edge's safe zone.
    base = round((470 if align == "bottom" else 250) * scale)
    return tag, max(1, base + round(offset * height))


def speaker_colours(cues: list[CaptionCue], options: CaptionOptions) -> dict[str, str]:
    """Speaker label to colour, in order of first appearance."""
    colours: dict[str, str] = {}
    if not options.speaker_colors:
        return colours
    for cue in cues:
        label = cue.words[0].speaker if cue.words else ""
        if label and label not in colours:
            colours[label] = options.speaker_colors[len(colours) % len(options.speaker_colors)]
    return colours


def cue_events(cues: list[CaptionCue], options: CaptionOptions) -> list[str]:
    look = options.look
    anchor, margin_v = _position(options)
    colours = speaker_colours(cues, options)
    per_line = chars_per_line(options)
    blur = f"\\blur{options.blur:g}" if options.blur > 0 else ""
    has_highlight_style = bool(look.get("hlEnabled"))
    if has_highlight_style:
        active = "\\rHighlight" + blur + (_override(options.highlight_color)
                                         if options.highlight_color else "")
        inactive, after = blur, "\\rDefault" + blur
    else:
        spoken = options.highlight_color or look["primaryColor"]
        unspoken = look["secondaryColor"] if options.highlight != "reveal" else spoken
        active, inactive = blur + _override(spoken), blur + _override(unspoken)
        after = inactive
    events: list[str] = []

    def event(start: float, end: float, text: str) -> None:
        events.append(
            f"Dialogue: 0,{_clock(start)},{_clock(end)},Default,,0,0,{margin_v},,{text}")

    base_inactive = inactive
    for index, cue in enumerate(cues):
        # A speaker's colour replaces the unspoken words' colour.
        colour = colours.get(cue.words[0].speaker, "") if cue.words else ""
        inactive = base_inactive + (_override(colour) if colour else "")
        texts = [_escape(word.text) for word in cue.words]
        breaks = wrap(texts, per_line, options.max_lines)
        fade_in, fade_out = _fades(cues, index, options)
        fade = f"\\fad({fade_in},{fade_out})" if fade_in or fade_out else ""
        mode = options.highlight
        if mode == "none" or len(cue.words) == 0:
            event(cue.start, cue.end, _tag(anchor + fade + blur
                                           + (_override(colour) if colour else ""))
                  + _join(texts, breaks))
            continue
        if mode == "karaoke":
            lead = round((cue.words[0].start - cue.start) * 100)
            parts = [(f"{{\\k{lead}}}" if position == 0 and lead > 0 else "")
                     + f"{{\\kf{round((word.end - word.start) * 100)}}}" + text
                     for position, (word, text) in enumerate(zip(cue.words, texts, strict=True))]
            event(cue.start, cue.end, _tag(anchor + fade + blur
                                           + (_override(colour) if colour else ""))
                  + _join(parts, breaks))
            continue
        # One event per spoken word with the whole cue laid out the same way,
        # so the text does not jump between words.
        count = len(cue.words)
        previous = cue.start
        for position in range(count):
            start = cue.start if position == 0 else max(previous, cue.words[position].start)
            end = cue.end if position == count - 1 else max(start,
                                                            cue.words[position + 1].start)
            if end - start < 0.01:
                continue
            previous = end
            parts = []
            for other, text in enumerate(texts):
                if mode == "reveal" and other > position:
                    parts.append(("{\\alpha&HFF&}" if other == position + 1 else "") + text)
                elif other == position or (mode == "fill" and other < position):
                    pop = ""
                    if mode == "pop" and other == position:
                        size = look.get("hlScale", 100) if has_highlight_style else 100
                        pop = (f"\\fscx{size * 1.25:g}\\fscy{size * 1.25:g}"
                               f"\\t(0,{min(120, round((end - start) * 1000))},"
                               f"\\fscx{size:g}\\fscy{size:g})")
                    parts.append(_tag(active + pop) + text + _tag(after))
                else:
                    parts.append(text)
            edge = (f"\\fad({fade_in if position == 0 else 0},"
                    f"{fade_out if position == count - 1 else 0})"
                    if (position == 0 and fade_in) or (position == count - 1 and fade_out)
                    else "")
            event(start, end, _tag(anchor + edge + inactive) + _join(parts, breaks))
    return events


def to_styled_ass(cues: list[CaptionCue], options: CaptionOptions) -> str:
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {options.play_res[0]}\nPlayResY: {options.play_res[1]}\nWrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(style_lines(options))
        + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )
    return header + "\n".join(cue_events(cues, options)) + "\n"
