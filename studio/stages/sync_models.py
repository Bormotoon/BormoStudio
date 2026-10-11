"""Internal matching models preserved from WhisperSync."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float
    norm: str = ""



@dataclass
class Segment:
    start: float
    end: float
    words: list[Word] = field(default_factory=list)



@dataclass
class Transcript:
    source_path: Path
    language: str
    duration: float
    segments: list[Segment]

    @property
    def words(self) -> list[Word]:
        return [w for seg in self.segments for w in seg.words]



@dataclass
class Anchor:
    cam_time: float
    rec_time: float
    token: str
    confidence: float



@dataclass
class AlignmentMap:
    """``t_camera = offset + k * t_recorder`` — one clip's clock map.

    ``provenance`` records HOW the map was derived, because "how many anchors
    does it have" is not the same question as "is it any good": the acoustic
    fallback produces a perfectly usable map with an EMPTY anchor list, and
    code that measured confidence by ``len(anchors)`` therefore threw away
    every successful acoustic match while happily accepting a two-anchor text
    fit (through two points any line fits with ~zero residual, including a
    completely wrong one). ``evidence`` carries the numbers a gate needs —
    see ``matcher.evaluate_alignment``.
    """

    anchors: list[Anchor]
    offset: float
    k: float
    residual_ms: float
    # "text" (transcript word anchors), "acoustic" (GCC-PHAT waveform scan),
    # or "repair" (a local re-alignment of one flagged span).
    provenance: str = "text"
    # Anchors that survived the robust line fit (<= len(anchors)). For an
    # acoustic map this is the number of confident grid points instead.
    inliers: int = 0
    # Recorder-time extent the supporting evidence actually spans, in seconds.
    # A map fitted from evidence covering 3 s of a 10-minute clip is an
    # extrapolation, however small its residual.
    evidence_span_s: float = 0.0
    # Why this map was chosen over another one (text against acoustics).
    notes: list[str] = field(default_factory=list)

