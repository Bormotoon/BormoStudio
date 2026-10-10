"""Whole-timeline overview of rough-cut decisions."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

_REASON_COLORS = {
    "pause": "#5b8def",
    "head": "#8a8f98",
    "tail": "#8a8f98",
    "retake": "#e0a030",
    "llm": "#b062d8",
    "manual": "#d05050",
}


@dataclass(frozen=True, slots=True)
class TimelineCut:
    id: str
    start: float
    end: float
    reason: str
    note: str
    accepted: bool


class CutTimeline(QWidget):
    """Draws kept material, accepted and suggested cuts and markers.

    Accepted cuts are solid, suggested (unaccepted) ones are outlined, so the
    resulting program can be read at a glance. Clicking a cut selects it.
    """

    cut_selected = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.duration = 0.0
        self.cuts: list[TimelineCut] = []
        self.markers: list[tuple[float, str]] = []
        self.selected: str | None = None
        self.playhead: float | None = None
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setAccessibleName("Rough cut timeline")

    def set_edit(self, duration: float, cuts: list[TimelineCut],
                 markers: list[tuple[float, str]]) -> None:
        self.duration = max(0.0, duration)
        self.cuts = sorted(cuts, key=lambda cut: cut.start)
        self.markers = markers
        if self.selected not in {cut.id for cut in cuts}:
            self.selected = None
        self.update()

    def select(self, cut_id: str | None) -> None:
        self.selected = cut_id
        self.update()

    def set_playhead(self, at: float | None) -> None:
        self.playhead = at
        self.update()

    def kept_duration(self) -> float:
        removed = 0.0
        cursor = 0.0
        for cut in self.cuts:
            if not cut.accepted:
                continue
            start, end = max(cursor, cut.start), max(cursor, cut.end)
            removed += end - start
            cursor = max(cursor, cut.end)
        return max(0.0, self.duration - removed)

    def cut_at(self, x: float) -> TimelineCut | None:
        if self.duration <= 0 or self.width() <= 0:
            return None
        at = x / self.width() * self.duration
        slack = 3 / self.width() * self.duration
        hits = [cut for cut in self.cuts if cut.start - slack <= at <= cut.end + slack]
        return min(hits, key=lambda cut: cut.end - cut.start) if hits else None

    def _x(self, at: float) -> float:
        return at / self.duration * self.width() if self.duration > 0 else 0.0

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        height = self.height()
        band = QRectF(0, 14, self.width(), height - 28)
        painter.fillRect(band, QColor("#2f7d4f"))
        for cut in self.cuts:
            rect = QRectF(self._x(cut.start), band.top(),
                          max(2.0, self._x(cut.end) - self._x(cut.start)), band.height())
            color = QColor(_REASON_COLORS.get(cut.reason, "#d05050"))
            if cut.accepted:
                painter.fillRect(rect, color)
            else:
                painter.fillRect(rect, QBrush(color, Qt.BrushStyle.BDiagPattern))
                painter.setPen(QPen(color, 1))
                painter.drawRect(rect)
            if cut.id == self.selected:
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.drawRect(rect.adjusted(-1, -3, 1, 3))
        painter.setPen(QPen(QColor("#f2c94c"), 1))
        for at, _text in self.markers:
            x = self._x(at)
            painter.drawLine(int(x), 2, int(x), 12)
        if self.playhead is not None:
            painter.setPen(QPen(QColor("#ffffff"), 1))
            x = int(self._x(self.playhead))
            painter.drawLine(x, 0, x, height)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        cut = self.cut_at(event.position().x())
        if cut is not None:
            self.select(cut.id)
            self.cut_selected.emit(cut.id)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        cut = self.cut_at(event.position().x())
        if cut is None:
            QToolTip.hideText()
            return
        state = "accepted" if cut.accepted else "suggested"
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{cut.reason} ({state}) {cut.start:.2f}–{cut.end:.2f}s\n{cut.note}",
            self,
        )
