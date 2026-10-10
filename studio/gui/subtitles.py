"""Native Qt subtitle editing with project persistence and media preview."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from studio.core.framing import load_reel_overrides, save_framing, save_reel_override
from studio.core.project import Project
from studio.core.settings import load_settings
from studio.core.subtitles import PRESETS, Cue, SubtitleStyle, load_edits, save_edits, validate
from studio.core.transcript import Transcript


class CaptionOverlay(QLabel):
    caption_style = SubtitleStyle()
    portrait = False

    def paintEvent(self, event: object) -> None:
        if not self.text():
            return
        reference_width, reference_height = (1080, 1920) if self.portrait else (1920, 1080)
        scale = min(self.width() / reference_width, self.height() / reference_height)
        frame_width, frame_height = reference_width * scale, reference_height * scale
        frame_left, frame_top = (self.width() - frame_width) / 2, (self.height() - frame_height) / 2
        font = self.font()
        font.setFamily(self.caption_style.font)
        font.setPixelSize(max(1, round(self.caption_style.size * scale)))
        metrics = QFontMetricsF(font)
        lines = []
        for paragraph in self.text().splitlines():
            current = ""
            for word in paragraph.split():
                proposed = f"{current} {word}".strip()
                if current and metrics.horizontalAdvance(proposed) > frame_width - 80 * scale:
                    lines.append(current)
                    current = word
                else:
                    current = proposed
            lines.append(current)
        line_height = metrics.height()
        top = (
            frame_top + self.caption_style.margin * scale
            if self.caption_style.alignment == 8
            else frame_top + frame_height - self.caption_style.margin * scale
            - len(lines) * line_height
        )
        path = QPainterPath()
        for number, line in enumerate(lines):
            x = frame_left + (frame_width - metrics.horizontalAdvance(line)) / 2
            path.addText(x, top + metrics.ascent() + number * line_height, font, line)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.caption_style.outline:
            painter.setPen(
                QPen(
                    QColor("black"),
                    self.caption_style.outline * scale * 2,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawPath(path)
        painter.fillPath(path, QColor(self.caption_style.color))
        painter.end()


def crop_rect(
    source: tuple[int, int], output: tuple[int, int], mode: str, crop_x: float
) -> tuple[float, float, float, float]:
    """The part of the source frame a reel keeps, as x, y, width, height."""
    width, height = source
    if mode != "crop":
        return 0.0, 0.0, float(width), float(height)
    aspect = output[0] / output[1]
    if width / height > aspect:
        kept = height * aspect
        return (width - kept) * crop_x, 0.0, kept, float(height)
    kept = width / aspect
    return 0.0, (height - kept) / 2, float(width), kept


class FramePreview(QWidget):
    """An unframed still of the reel with the chosen crop drawn over it.

    Redrawn on every control change, so the crop is visible before rendering.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()
        self.mode, self.crop_x, self.output = "source", 0.5, (1080, 1920)
        self.tracking = False
        self.setMinimumSize(160, 120)
        self.setAccessibleName("Reel frame preview")

    def set_image(self, path: Path | None) -> None:
        self.image = QImage(str(path)) if path is not None and path.is_file() else QImage()
        self.update()

    def set_framing(
        self, mode: str, crop_x: float, output: tuple[int, int], tracking: bool
    ) -> None:
        self.mode, self.crop_x, self.output, self.tracking = mode, crop_x, output, tracking
        self.update()

    def kept_rect(self) -> tuple[float, float, float, float] | None:
        if self.image.isNull():
            return None
        return crop_rect((self.image.width(), self.image.height()), self.output,
                         self.mode, self.crop_x)

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        if self.image.isNull():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Render reels once to preview framing")
            painter.end()
            return
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        left = (self.width() - self.image.width() * scale) / 2
        top = (self.height() - self.image.height() * scale) / 2
        target = QRectF(left, top, self.image.width() * scale, self.image.height() * scale)
        painter.drawImage(target, self.image)
        rect = self.kept_rect()
        if rect is not None and self.mode == "crop":
            x, y, w, h = (value * scale for value in rect)
            kept = QRectF(left + x, top + y, w, h)
            shade = QColor(0, 0, 0, 150)
            painter.fillRect(QRectF(target.left(), target.top(), kept.left() - target.left(),
                                    target.height()), shade)
            painter.fillRect(QRectF(kept.right(), target.top(), target.right() - kept.right(),
                                    target.height()), shade)
            painter.fillRect(QRectF(kept.left(), target.top(), kept.width(),
                                    kept.top() - target.top()), shade)
            painter.fillRect(QRectF(kept.left(), kept.bottom(), kept.width(),
                                    target.bottom() - kept.bottom()), shade)
            painter.setPen(QPen(QColor("#FFD400"), 2))
            painter.drawRect(kept)
            if self.tracking:
                painter.drawText(kept.adjusted(6, 6, -6, -6), Qt.AlignmentFlag.AlignTop,
                                 "Tracking moves this window")
        painter.end()


class SubtitleEditor(QWidget):
    render_requested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.project: Project | None = None
        self.transcript: Transcript | None = None
        self.identity = ""
        self.dirty = False
        self.loading = False
        self.documents: list[tuple[str, Path, Path]] = []
        self.selector = QComboBox()
        self.selector.setAccessibleName("Subtitle video")
        self.selector.currentIndexChanged.connect(self._select)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(180)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self._position)
        self.player.durationChanged.connect(lambda value: self.seek.setMaximum(value))
        self.player.metaDataChanged.connect(self._media_dimensions)
        self.player.errorOccurred.connect(lambda _error, text: self.status.setText(text))
        self.caption = CaptionOverlay("")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.TextFormat.PlainText)
        self.caption.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.preview = QWidget()
        stack = QStackedLayout(self.preview)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        stack.addWidget(self.video)
        stack.addWidget(self.caption)
        stack.setCurrentWidget(self.caption)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setAccessibleName("Video position")
        self.seek.sliderMoved.connect(self.player.setPosition)
        play = QPushButton("Play / pause")
        play.clicked.connect(self._play)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Start (s)", "End (s)", "Subtitle text"])
        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self.table.setAccessibleName("Subtitle cues")
        self.table.cellChanged.connect(self._changed)
        self.table.cellClicked.connect(self._jump)
        self.preset = QComboBox()
        self.preset.addItems(PRESETS)
        self.preset.activated.connect(lambda: self._set_style(PRESETS[self.preset.currentText()]))
        self.font_family = QLineEdit("DejaVu Sans")
        self.color = QLineEdit("#FFFFFF")
        self.font_size, self.outline, self.margin = QSpinBox(), QSpinBox(), QSpinBox()
        self.font_size.setRange(12, 160)
        self.outline.setRange(0, 10)
        self.margin.setRange(0, 500)
        self.alignment = QComboBox()
        self.alignment.addItems(["Bottom", "Top"])
        self.framing = QComboBox()
        self.framing.addItems(["source", "crop", "fit"])
        self.tracking = QCheckBox("Follow face / active speaker")
        self.frame_width, self.frame_height = QSpinBox(), QSpinBox()
        for field in (self.frame_width, self.frame_height):
            field.setRange(64, 4096)
            field.setSingleStep(2)
        self.frame_width.setValue(1080)
        self.frame_height.setValue(1920)
        self.crop_x = QSlider(Qt.Orientation.Horizontal)
        self.crop_x.setRange(0, 100)
        self.crop_x.setValue(50)
        self.crop_x.setToolTip("0: left edge; 50: center; 100: right edge")
        self.scope = QComboBox()
        self.scope.addItems(["All reels (project)", "This reel only"])
        self.scope.setToolTip("Save framing for the whole project or only this reel")
        self.reel_title = QLineEdit()
        self.reel_title.setPlaceholderText("Hook from reels selection")
        self.reel_title.setToolTip("Text of the title band in title and audiogram layouts")
        self.frame_preview = FramePreview()
        self.reel_entries: dict[str, dict[str, object]] = {}
        self.overrides: dict[str, dict[str, object]] = {}
        self.project_framing: tuple[str, float, bool] = ("source", 0.5, False)
        form = QFormLayout()
        for name, control in (
            ("Preset", self.preset),
            ("Font", self.font_family),
            ("Size (1080p)", self.font_size),
            ("Color (#RRGGBB)", self.color),
            ("Outline", self.outline),
            ("Vertical margin", self.margin),
            ("Position", self.alignment),
            ("Framing for", self.scope),
            ("Reels framing", self.framing),
            ("Automatic tracking", self.tracking),
            ("Output width", self.frame_width),
            ("Output height", self.frame_height),
            ("Crop position", self.crop_x),
            ("Reel title", self.reel_title),
        ):
            form.addRow(name, control)
            control.setAccessibleName(name)
        for text_field in (self.font_family, self.color):
            text_field.textChanged.connect(self._changed)
        for number_field in (self.font_size, self.outline, self.margin):
            number_field.valueChanged.connect(self._changed)
        self.alignment.currentIndexChanged.connect(self._changed)
        self.framing.currentIndexChanged.connect(self._changed)
        self.frame_width.valueChanged.connect(self._changed)
        self.frame_height.valueChanged.connect(self._changed)
        self.crop_x.valueChanged.connect(self._changed)
        self.tracking.toggled.connect(self._changed)
        self.scope.currentIndexChanged.connect(self._scope_changed)
        self.reel_title.textChanged.connect(self._changed)
        save = QPushButton("Save subtitles")
        save.clicked.connect(self.save)
        render = QPushButton("Save and render captions")
        render.clicked.connect(self._render)
        add = QPushButton("Add cue")
        add.clicked.connect(self._add_cue)
        remove = QPushButton("Delete selected cue")
        remove.clicked.connect(self._remove_cue)
        self.status = QLabel("Select a project with rendered reels or a review program.")
        self.status.setWordWrap(True)
        actions = QHBoxLayout()
        actions.addWidget(play)
        actions.addWidget(save)
        actions.addWidget(render)
        actions.addWidget(add)
        actions.addWidget(remove)
        layout = QVBoxLayout(self)
        layout.addWidget(self.selector)
        layout.addWidget(self.preview, 2)
        layout.addWidget(self.seek)
        layout.addLayout(actions)
        lower = QHBoxLayout()
        lower.addWidget(self.table, 3)
        side = QVBoxLayout()
        side.addLayout(form)
        side.addWidget(self.frame_preview, 1)
        lower.addLayout(side, 1)
        layout.addLayout(lower, 2)
        layout.addWidget(self.status)
        self._set_style(SubtitleStyle())
        self.dirty = False

    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved subtitles",
            "Discard unsaved subtitle edits?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def load_project(self, project: Project) -> None:
        if not self.confirm_discard():
            return
        self.player.stop()
        self.project = project
        self.transcript = None
        self.identity = ""
        self.table.setRowCount(0)
        self.caption.clear()
        self.loading = True
        try:
            framing = load_settings([project.work_dir / "settings.yaml"]).reels
            self.framing.setCurrentText(framing.framing)
            self.frame_width.setValue(framing.width)
            self.frame_height.setValue(framing.height)
            self.crop_x.setValue(round(framing.crop_x * 100))
            self.tracking.setChecked(framing.tracking)
            self.project_framing = (framing.framing, framing.crop_x, framing.tracking)
            self.overrides = load_reel_overrides(project.work_dir)
            self.reel_entries = {}
            for report in project.outputs.get("reel_render", []):
                if report.name == "render.json" and report.is_file():
                    for entry in json.loads(report.read_text(encoding="utf-8"))["reels"]:
                        self.reel_entries[Path(str(entry["video"])).stem] = {
                            **entry, "folder": str(report.parent)}
        except (ValueError, OSError, KeyError) as exc:
            self.loading = False
            self.status.setText(str(exc))
            return
        self.selector.clear()
        self.documents = []
        for path in project.outputs.get("reel_render", []):
            if path.suffix == ".json" and path.name != "render.json":
                video = path.with_suffix(".mp4")
                if video.is_file():
                    self.documents.append((path.stem, path, video))
        if project.outputs.get("program") and project.transcripts.get("edited"):
            self.documents.append(
                ("program", project.transcripts["edited"], project.outputs["program"][0])
            )
        self.selector.addItems([name for name, _path, _video in self.documents])
        self.loading = False
        self.dirty = False
        self._select(self.selector.currentIndex())

    def _select(self, index: int) -> None:
        if self.loading or index < 0 or self.project is None:
            return
        if not self.confirm_discard():
            self.selector.blockSignals(True)
            self.selector.setCurrentIndex(
                next((i for i, doc in enumerate(self.documents) if doc[0] == self.identity), 0)
            )
            self.selector.blockSignals(False)
            return
        identity, path, video = self.documents[index]
        try:
            transcript = Transcript.load(path)
            cues, style = load_edits(self.project.work_dir, identity, transcript)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            self.transcript = None
            self.player.stop()
            self.table.setRowCount(0)
            self.status.setText(str(exc))
            return
        self.identity, self.transcript = identity, transcript
        for control in (
            self.framing,
            self.frame_width,
            self.frame_height,
            self.crop_x,
            self.tracking,
            self.scope,
            self.reel_title,
        ):
            control.setEnabled(identity != "program")
        self.loading = True
        self._show_reel_framing(identity)
        self.table.setRowCount(len(cues))
        for row, cue in enumerate(cues):
            for column, value in enumerate((f"{cue.start:.3f}", f"{cue.end:.3f}", cue.text)):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self._set_style(style)
        self.loading = False
        self.dirty = False
        self.player.setSource(QUrl.fromLocalFile(str(video.resolve())))
        self.status.setText("Click a cue to seek. Edit its text or times, then save.")

    def _reel_key(self, identity: str) -> str:
        return str(self.reel_entries.get(identity, {}).get("reel_key") or identity)

    def _show_reel_framing(self, identity: str) -> None:
        """Fill framing controls with this reel's override, else the project values."""
        entry = self.reel_entries.get(identity, {})
        frame = entry.get("frame")
        self.frame_preview.set_image(
            Path(str(entry["folder"])) / str(frame) if frame and "folder" in entry else None
        )
        override = self.overrides.get(self._reel_key(identity), {})
        mode, crop_x, tracking = self.project_framing
        self.scope.setCurrentIndex(1 if override else 0)
        self.framing.setCurrentText(str(override.get("framing", mode)))
        self.crop_x.setValue(round(float(override.get("crop_x", crop_x)) * 100))  # type: ignore[arg-type]
        self.tracking.setChecked(bool(override.get("tracking", tracking)))
        self.reel_title.setText(str(override.get("title", "")))
        self._update_frame_preview()

    def _scope_changed(self, index: int) -> None:
        if self.loading:
            return
        if index == 0:
            # Back to the project framing; saving removes this reel's override.
            mode, crop_x, tracking = self.project_framing
            self.loading = True
            self.framing.setCurrentText(mode)
            self.crop_x.setValue(round(crop_x * 100))
            self.tracking.setChecked(tracking)
            self.loading = False
        self._changed()

    def _update_frame_preview(self) -> None:
        self.frame_preview.set_framing(
            self.framing.currentText(), self.crop_x.value() / 100,
            (self.frame_width.value(), self.frame_height.value()), self.tracking.isChecked(),
        )

    def _save_framing(self) -> None:
        assert self.project is not None
        mode, crop_x, tracking = (
            self.framing.currentText(), self.crop_x.value() / 100, self.tracking.isChecked()
        )
        key = self._reel_key(self.identity)
        override: dict[str, object] = {}
        if self.scope.currentIndex() == 1:
            override = {"framing": mode, "crop_x": crop_x, "tracking": tracking}
            mode, crop_x, tracking = self.project_framing
        if self.reel_title.text().strip():
            override["title"] = self.reel_title.text().strip()
        save_framing(self.project.work_dir, mode, self.frame_width.value(),
                     self.frame_height.value(), crop_x, tracking=tracking)
        self.project_framing = (mode, crop_x, tracking)
        if override or key in self.overrides:
            save_reel_override(self.project.work_dir, key, override or None)
        self.overrides = load_reel_overrides(self.project.work_dir)

    def _style(self) -> SubtitleStyle:
        return SubtitleStyle(
            self.font_family.text(),
            self.font_size.value(),
            self.color.text(),
            self.outline.value(),
            self.margin.value(),
            8 if self.alignment.currentIndex() else 2,
        )

    def _media_dimensions(self) -> None:
        from PyQt6.QtMultimedia import QMediaMetaData

        resolution = self.player.metaData().value(QMediaMetaData.Key.Resolution)
        if resolution is not None and hasattr(resolution, "width"):
            self.caption.portrait = resolution.height() > resolution.width()
            self.caption.update()

    def _set_style(self, style: SubtitleStyle) -> None:
        self.font_family.setText(style.font)
        self.font_size.setValue(style.size)
        self.color.setText(style.color)
        self.outline.setValue(style.outline)
        self.margin.setValue(style.margin)
        self.alignment.setCurrentIndex(1 if style.alignment == 8 else 0)

    def _cell(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text() if item is not None else ""

    def _cues(self) -> list[Cue]:
        # An empty time cell raises ValueError like any other invalid time.
        return [
            Cue(float(self._cell(row, 0)), float(self._cell(row, 1)), self._cell(row, 2))
            for row in range(self.table.rowCount())
        ]

    def _changed(self, *_args: object) -> None:
        self._update_frame_preview()
        if not self.loading:
            self.dirty = True
            self.status.setText("Unsaved subtitle changes.")
            self._position(self.player.position())

    def _position(self, milliseconds: int) -> None:
        if not self.seek.isSliderDown():
            self.seek.setValue(milliseconds)
        try:
            cue = next(
                (cue for cue in self._cues() if cue.start <= milliseconds / 1000 < cue.end), None
            )
            self.caption.setText(cue.text if cue else "")
            style = self._style()
            self.caption.caption_style = style
            self.caption.update()
        except (ValueError, AttributeError):
            self.caption.setText("")

    def _jump(self, row: int, _column: int) -> None:
        try:
            self.player.setPosition(round(float(self._cell(row, 0)) * 1000))
        except ValueError:
            pass

    def _play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _add_cue(self) -> None:
        if self.transcript is None:
            return
        start = self.player.position() / 1000
        row = self.table.rowCount()
        try:
            row = next((i for i, cue in enumerate(self._cues()) if cue.start > start), row)
        except ValueError:
            self.status.setText("Fix invalid cue times before adding a cue.")
            return
        self.table.insertRow(row)
        for column, value in enumerate(
            (f"{start:.3f}", f"{min(start + 2, self.transcript.duration):.3f}", "New subtitle")
        ):
            self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.setCurrentCell(row, 2)
        self.table.editItem(self.table.item(row, 2))

    def _remove_cue(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self._changed()

    def save(self) -> bool:
        if self.project is None or self.transcript is None:
            return False
        try:
            cues, style = self._cues(), self._style()
            validate(cues, style, self.transcript.duration)
            if self.identity != "program":
                self._save_framing()
            save_edits(self.project.work_dir, self.identity, self.transcript, cues, style)
        except (ValueError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return False
        self.dirty = False
        self.status.setText("Saved subtitle edits and SRT/ASS. Render captions to update video.")
        return True

    def _render(self) -> None:
        if self.save():
            self.player.stop()
            self.render_requested.emit(
                "program_subtitles" if self.identity == "program" else "reel_render"
            )
