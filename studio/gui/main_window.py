"""Phase-one Studio desktop shell over the shared CLI and project model."""

from __future__ import annotations

import json
import sys
from importlib.resources import files
from pathlib import Path

from PyQt6.QtCore import QProcess, QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QTextCursor
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from studio.core.launch import python_command
from studio.core.project import Project
from studio.gui.bridge import RunnerBridge
from studio.gui.cut_timeline import CutTimeline, TimelineCut
from studio.gui.material import MaterialPage
from studio.gui.modules import ModulesPage
from studio.gui.settings_page import BasicOptions, SettingsPage
from studio.gui.setup_wizard import SetupWizard
from studio.gui.source_timeline import SourceTimeline
from studio.gui.subtitles import SubtitleEditor
from studio.stages.timeline_preview import TimelinePreview

# Navigation rows the basic mode hides: Material, Plan, Edit, Subtitles, Settings.
BASIC_HIDDEN = (1, 2, 3, 6, 8)


class MainWindow(QMainWindow):
    def __init__(self, preferences: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle("BormoStudio")
        self.resize(1100, 720)
        # Per-user GUI choices such as the mode; project settings live in the project.
        self.preferences = preferences or QSettings("BormoStudio", "Studio")
        self.bridge = RunnerBridge(self)
        self.source: Path | None = None
        self.pending_view = "material"
        self.navigation = QListWidget()
        self.navigation.addItems(
            ["Start", "Material", "Plan", "Edit", "Work", "Result", "Subtitles"]
        )
        self.subtitles = SubtitleEditor()
        self.modules_page = ModulesPage()
        self.navigation.addItem("Modules and models")
        self.navigation.addItem("Settings")
        self.settings_page = SettingsPage()
        self.modules_page.model_installed.connect(self.settings_page.use_model)
        self.pages = QStackedWidget()
        self.material = MaterialPage()
        self.material.command_requested.connect(self._material_command)
        self.review = QTreeWidget()
        self.review.setHeaderLabels(["ID", "Reason", "Start", "End", "Decision"])
        self.review_status = QLabel("No edit decisions available.")
        self.review_player = QMediaPlayer(self)
        self.review_audio = QAudioOutput(self)
        self.review_player.setAudioOutput(self.review_audio)
        self.review_timer = QTimer(self)
        self.review_timer.setSingleShot(True)
        self.review_timer.timeout.connect(self.review_player.pause)
        self.review_seek: float | None = None
        self.review_ranges: list[tuple[float, float]] = []
        self.review_player.mediaStatusChanged.connect(self._review_ready)
        self.review_player.positionChanged.connect(self._review_position)
        self.cut_timeline = CutTimeline()
        self.source_timeline = SourceTimeline()
        self.cut_timeline.cut_selected.connect(self._select_cut_from_timeline)
        # Whole-edit proxy: rendered by `bstudio preview` in the background.
        self.timeline_preview: TimelinePreview | None = None
        self.timeline_video = QVideoWidget()
        self.timeline_video.setMinimumHeight(180)
        self.timeline_video.setAccessibleName("Whole timeline preview")
        self.timeline_video.hide()
        self.timeline_player = QMediaPlayer(self)
        self.timeline_audio = QAudioOutput(self)
        self.timeline_player.setAudioOutput(self.timeline_audio)
        self.timeline_player.setVideoOutput(self.timeline_video)
        self.timeline_player.positionChanged.connect(self._timeline_position)
        self.preview_process = QProcess(self)
        self.preview_process.finished.connect(self._timeline_preview_finished)
        self.preview_process.errorOccurred.connect(
            lambda _error: self._timeline_preview_failed(self.preview_process.errorString()))
        self.cut_timeline.position_requested.connect(self._seek_timeline_preview)
        self.review.currentItemChanged.connect(self._select_cut_from_list)
        self.plan_label = QLabel("Scan a folder to build the plan.")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setAccessibleName("Processing details")
        self.progress_label = QLabel("Nothing is running.")
        self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.stage_titles: dict[str, str] | None = None
        self.result = QLabel("No results yet.")
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._build_pages()
        self.advanced = QCheckBox("Advanced mode")
        self.advanced.setToolTip("Every processing setting, the material and plan pages, "
                                 "cut review and subtitles")
        side = QVBoxLayout()
        side.addWidget(self.navigation)
        side.addWidget(self.advanced)
        layout = QHBoxLayout()
        layout.addLayout(side, 1)
        layout.addWidget(self.pages, 4)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.navigation.currentRowChanged.connect(self._navigate)
        self.navigation.setCurrentRow(0)
        self.bridge.output.connect(self._append_output)
        self.bridge.finished.connect(self._finished)
        self.bridge.failed.connect(self._append_log)
        self.bridge.stage_event.connect(self._plan_event)
        self.subtitles.render_requested.connect(self._render_subtitles)
        self.advanced.setChecked(bool(self.preferences.value("advanced", False, type=bool)))
        self.advanced.toggled.connect(self._mode_toggled)
        self._apply_mode()

    def _render_subtitles(self, stage: str) -> None:
        if self.source is None:
            return
        try:
            arguments = (
                ["--set", "program.burn_subtitles=true"]
                if stage == "program_subtitles"
                else ["--set", "reels.render=true", "--set", "reels.burn_subtitles=true"]
            )
            self.bridge.start(
                "run",
                self.source,
                [
                    "--only",
                    stage,
                    *arguments,
                ],
            )
        except RuntimeError as exc:
            self.subtitles.status.setText(str(exc))
            return
        self.pending_view = "subtitles"
        self.navigation.setCurrentRow(4)

    def _build_pages(self) -> None:
        start = QWidget()
        start_layout = QVBoxLayout(start)
        title = QLabel("Turn raw recordings into an editable project")
        title.setObjectName("hero")
        start_layout.addWidget(title)
        start_layout.addWidget(self._basic_start())
        self.advanced_start = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_start)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        choose = QPushButton("Choose material folder")
        choose.clicked.connect(self._choose)
        advanced_layout.addWidget(choose)
        setup = QPushButton("Set up processing")
        setup.clicked.connect(self._setup)
        advanced_layout.addWidget(setup)
        start_layout.addWidget(self.advanced_start)
        start_layout.addStretch()
        self.pages.addWidget(start)
        self.pages.addWidget(self.material)
        plan = self._page(self.plan_label, "Build plan", self._plan)
        self.pages.addWidget(plan)
        run = QPushButton("Run processing plan")
        run.clicked.connect(self._run)
        self.install_target: tuple[str, str] | None = None
        self.install_button = QPushButton("Install missing requirement")
        self.install_button.setEnabled(False)
        self.install_button.clicked.connect(self._open_install)
        plan_layout = plan.layout()
        if plan_layout is not None:
            plan_layout.addWidget(run)
            plan_layout.addWidget(self.install_button)
        self.pages.addWidget(self._review_page())
        work = QWidget()
        work_layout = QVBoxLayout(work)
        work_layout.addWidget(self.progress_label)
        work_layout.addWidget(self.progress_bar)
        work_layout.addWidget(self.log, 1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.bridge.cancel)
        work_layout.addWidget(cancel)
        self.pages.addWidget(work)
        result = self._page(self.result, "Open export folder", self._open_result)
        result_layout = result.layout()
        if isinstance(result_layout, QVBoxLayout):
            result_layout.addStretch()
        self.pages.addWidget(result)
        self.pages.addWidget(self.subtitles)
        self.pages.addWidget(self.modules_page)
        self.pages.addWidget(self.settings_page)

    def _basic_start(self) -> QWidget:
        """Basic mode: choose a folder, glance at four choices, process."""
        self.basic_start = QWidget()
        layout = QVBoxLayout(self.basic_start)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Choose the folder with the recordings. Studio finds the cameras and the "
            "recorder, synchronizes them, transcribes the speech, removes pauses and "
            "retakes and writes Final Cut Pro and Premiere projects into _studio/export."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        choose = QPushButton("Choose material folder")
        choose.clicked.connect(self._choose_basic)
        layout.addWidget(choose)
        self.folder_label = QLabel("No folder chosen yet.")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)
        self.basic_options = BasicOptions()
        self.basic_options.setEnabled(False)
        layout.addWidget(self.basic_options)
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self._process)
        layout.addWidget(self.process_button)
        self.basic_status = QLabel("")
        self.basic_status.setWordWrap(True)
        layout.addWidget(self.basic_status)
        return self.basic_start

    def _mode_toggled(self, advanced: bool) -> None:
        # The basic mode hides the settings page: unsaved edits there would block runs.
        if not advanced and not self.settings_page.confirm_discard():
            self.advanced.blockSignals(True)
            self.advanced.setChecked(True)
            self.advanced.blockSignals(False)
            return
        if advanced and self.source is not None and self.basic_options.dirty:
            try:
                self.basic_options.save(self.source / "_studio")
            except (ValueError, OSError, RuntimeError) as exc:
                self.basic_status.setText(str(exc))
        self.preferences.setValue("advanced", advanced)
        self._apply_mode()

    def _apply_mode(self) -> None:
        advanced = self.advanced.isChecked()
        for row in BASIC_HIDDEN:
            item = self.navigation.item(row)
            if item is not None:
                item.setHidden(not advanced)
        self.basic_start.setVisible(not advanced)
        self.advanced_start.setVisible(advanced)
        if not advanced:
            self._load_basic()
            if self.navigation.currentRow() in BASIC_HIDDEN:
                self.navigation.setCurrentRow(0)

    def _load_basic(self) -> None:
        if self.source is None:
            return
        self.folder_label.setText(f"Material: {self.source}")
        try:
            self.basic_options.load(self.source / "_studio")
        except (ValueError, OSError) as exc:
            self.basic_options.setEnabled(False)
            self.process_button.setEnabled(False)
            self.basic_status.setText(f"{exc}\nFix the project settings in the advanced mode.")
            return
        self.basic_options.setEnabled(True)
        self.process_button.setEnabled(True)
        self.basic_status.setText("")

    def _choose_basic(self) -> None:
        if not self.settings_page.confirm_discard() or not self.subtitles.confirm_discard():
            return
        selected = QFileDialog.getExistingDirectory(self, "Choose material folder")
        if not selected:
            return
        self.source = Path(selected)
        self.subtitles.dirty = False
        self._load_basic()

    def _process(self) -> None:
        """Save the basic choices and run the whole pipeline on the chosen folder."""
        if self.source is None:
            return
        if self.bridge.process.state() != QProcess.ProcessState.NotRunning:
            self.basic_status.setText("Processing is already running.")
            return
        try:
            self.basic_options.save(self.source / "_studio")
            self.bridge.start("run", self.source, ["--events"])
        except (ValueError, OSError, RuntimeError) as exc:
            self.basic_status.setText(str(exc))
            return
        self._reset_progress()
        self.pending_view = "result"
        self.navigation.setCurrentRow(4)

    def _reset_progress(self) -> None:
        self.log.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…")

    def _show_progress(self, value: dict) -> None:
        stage = str(value.get("stage", ""))
        if self.stage_titles is None:
            from studio.stages.worker import stage_registry

            self.stage_titles = {key: item.title for key, item in stage_registry().items()}
        title = self.stage_titles.get(stage) or stage
        kind = value.get("t")
        if kind == "start":
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"{title}…")
        elif kind == "progress":
            fraction = value.get("value", 0)
            if isinstance(fraction, (int, float)):
                self.progress_bar.setValue(round(float(fraction) * 100))
            message = str(value.get("message") or "")
            self.progress_label.setText(f"{title}: {message}" if message else f"{title}…")
        elif kind == "done" and value.get("status") == "failed":
            self.progress_label.setText(f"{title} failed; the details are below.")

    def _result_text(self, project: Project) -> str:
        """What the run produced, in words, with the full output list when advanced."""
        lines = [f"Material: {project.source_dir}"]
        try:
            report = json.loads((project.work_dir / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        status = report.get("status")
        if status == "needs_review":
            lines.append("Done, but check the sync of these clips:")
            lines += [f"  {clip}: {reason}" for clip, reason in report.get("review", {}).items()]
        elif status == "failed":
            lines.append("Some steps failed: " + ", ".join(report.get("failures", {})))
        elif status == "blocked":
            lines.append("Some steps could not run:")
            lines += [f"  {stage}: {reason}" for stage, reason in report.get("blocked", {}).items()]
        elif status == "ok":
            lines.append("Done.")
        names = {".fcpxml": "Final Cut Pro", ".xml": "Premiere Pro / DaVinci Resolve",
                 ".otio": "DaVinci Resolve (OTIO)"}
        for path in project.outputs.get("export", []):
            if path.suffix in names:
                lines.append(f"{names[path.suffix]}: {path}")
        for key in ("program", "program_subtitles"):
            lines += [f"Review video: {path}" for path in project.outputs.get(key, [])
                      if path.suffix == ".mp4"]
        reels = [path for path in project.outputs.get("reel_render", []) if path.suffix == ".mp4"]
        if reels:
            lines.append(f"Reels: {len(reels)} in {reels[0].parent}")
        if self.advanced.isChecked():
            lines.append("\nAll outputs:")
            lines += [f"  {key}: {', '.join(str(path) for path in paths)}"
                      for key, paths in project.outputs.items()]
        return "\n".join(lines)

    def _review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.source_timeline)
        layout.addWidget(self.timeline_video, 2)
        layout.addWidget(self.cut_timeline)
        whole = QHBoxLayout()
        self.preview_button = QPushButton("Preview whole edit")
        self.preview_button.setToolTip("Renders a quick low-resolution copy of the edit with "
                                       "accepted cuts and camera switches")
        self.preview_button.clicked.connect(self._start_timeline_preview)
        play = QPushButton("Play / pause edit")
        play.clicked.connect(self._toggle_timeline_preview)
        whole.addWidget(self.preview_button)
        whole.addWidget(play)
        layout.addLayout(whole)
        layout.addWidget(self.review)
        layout.addWidget(self.review_status)
        actions = QHBoxLayout()
        accept = QPushButton("Accept cut")
        accept.clicked.connect(lambda: self._review_cut(True))
        reject = QPushButton("Reject cut")
        reject.clicked.connect(lambda: self._review_cut(False))
        actions.addWidget(accept)
        actions.addWidget(reject)
        listen = QPushButton("Listen with cut")
        listen.clicked.connect(lambda: self._listen_cut(applied=True))
        original = QPushButton("Listen to original")
        original.clicked.connect(lambda: self._listen_cut(applied=False))
        stop = QPushButton("Stop preview")
        stop.clicked.connect(self.review_player.stop)
        actions.addWidget(listen)
        actions.addWidget(original)
        actions.addWidget(stop)
        layout.addLayout(actions)
        return page

    def _start_timeline_preview(self) -> None:
        if self.source is None or self.preview_process.state() != QProcess.ProcessState.NotRunning:
            return
        self.timeline_player.stop()
        self.preview_button.setEnabled(False)
        self.review_status.setText("Rendering a preview of the whole edit...")
        executable, *arguments = python_command("studio.cli.main", ["preview", str(self.source)])
        self.preview_process.start(executable, arguments)

    def _timeline_preview_failed(self, message: str) -> None:
        self.preview_button.setEnabled(True)
        self.review_status.setText(f"Preview unavailable: {message.strip()[-500:]}")

    def _timeline_preview_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self.preview_button.setEnabled(True)
        if code:
            self._timeline_preview_failed(
                self.preview_process.readAllStandardError().data().decode(errors="replace"))
            return
        try:
            data = json.loads(self.preview_process.readAllStandardOutput().data())
            self.timeline_preview = TimelinePreview(
                Path(data["video"]), tuple(data["spans"]), bool(data["reused"]))
        except (ValueError, KeyError, TypeError) as exc:
            self._timeline_preview_failed(str(exc))
            return
        self.timeline_video.show()
        self.timeline_player.setSource(QUrl.fromLocalFile(str(self.timeline_preview.video)))
        self.review_status.setText(
            "Whole edit ready: play it, or click the cut overview to jump.")

    def _toggle_timeline_preview(self) -> None:
        if self.timeline_preview is None:
            self._start_timeline_preview()
        elif self.timeline_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.timeline_player.pause()
        else:
            self.review_player.stop()
            self.timeline_player.play()

    def _timeline_position(self, position: int) -> None:
        if self.timeline_preview is not None:
            self.cut_timeline.set_playhead(self.timeline_preview.to_timeline(position / 1000))

    def _seek_timeline_preview(self, timeline_s: float) -> None:
        if self.timeline_preview is not None:
            self.timeline_player.setPosition(
                round(self.timeline_preview.to_edited(timeline_s) * 1000))

    def _select_cut_from_timeline(self, cut_id: str) -> None:
        for index in range(self.review.topLevelItemCount()):
            item = self.review.topLevelItem(index)
            if item is not None and item.data(0, Qt.ItemDataRole.UserRole) == cut_id:
                self.review.setCurrentItem(item)
                return

    def _select_cut_from_list(self, item: QTreeWidgetItem | None, _previous: object) -> None:
        self.cut_timeline.select(
            str(item.data(0, Qt.ItemDataRole.UserRole)) if item is not None else None
        )

    def _listen_cut(self, applied: bool = True) -> None:
        from studio.core.preview import preview_ranges, render_preview

        item = self.review.currentItem()
        if item is None or self.source is None:
            return
        project = Project.load(self.source / "_studio" / "project.json")
        cut_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        ranges = preview_ranges(
            float(item.text(2)), float(item.text(3)), context_s=2.0, applied=applied,
            duration=self.cut_timeline.duration or float(item.text(3)) + 2.0,
        )
        target = project.work_dir / "cache" / "preview" / (
            f"{cut_id}-{'cut' if applied else 'original'}.wav"
        )
        try:
            path, kind = render_preview(project, ranges, target)
        except (OSError, ValueError, RuntimeError) as exc:
            self.review_status.setText(f"Preview unavailable: {exc}")
            return
        self.review_status.setText(
            f"Playing {'with cut' if applied else 'original'} from {kind}"
        )
        self.review_timer.stop()
        self.review_ranges = ranges
        self.review_seek = 0.0
        self.review_player.setSource(QUrl())
        self.review_player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self._review_ready(self.review_player.mediaStatus())

    def _review_ready(self, status: QMediaPlayer.MediaStatus) -> None:
        if self.review_seek is not None and status in {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        }:
            self.review_seek = None
            self.review_player.setPosition(0)
            self.review_player.play()
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.cut_timeline.set_playhead(None)

    def _review_position(self, position: int) -> None:
        elapsed = position / 1000
        for start, end in self.review_ranges:
            if elapsed <= end - start:
                self.cut_timeline.set_playhead(start + elapsed)
                return
            elapsed -= end - start

    def _page(self, content: QWidget, action: str, callback: object) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(content)
        button = QPushButton(action)
        button.clicked.connect(callback)  # type: ignore[arg-type]
        layout.addWidget(button)
        return page

    def _choose(self) -> None:
        if not self.settings_page.confirm_discard():
            return
        if not self.subtitles.confirm_discard():
            return
        selected = QFileDialog.getExistingDirectory(self, "Choose material folder")
        if not selected:
            return
        self.source = Path(selected)
        self.subtitles.dirty = False
        self.pending_view = "material"
        self.navigation.setCurrentRow(4)
        self.bridge.start("scan", self.source)

    def _setup(self) -> None:
        if self.source is None:
            self._choose()
            return
        wizard = SetupWizard(self.source / "_studio", self)
        wizard.exec()

    def _plan(self) -> None:
        if self.source:
            self.pending_view = "plan"
            self.navigation.setCurrentRow(4)
            self.bridge.start("plan", self.source)

    def _run(self) -> None:
        if self.settings_page.dirty:
            self.plan_label.setText("Save processing settings before running.")
            self.navigation.setCurrentRow(8)
            return
        if self.source is None:
            self.plan_label.setText("Choose a material folder first.")
            return
        try:
            self.bridge.start("run", self.source, ["--events"])
        except RuntimeError as exc:
            self.plan_label.setText(str(exc))
            return
        self._reset_progress()
        self.pending_view = "result"
        self.navigation.setCurrentRow(4)

    def _open_install(self) -> None:
        if self.install_target is None:
            return
        self.modules_page.select_target(*self.install_target)
        self.navigation.setCurrentRow(7)

    def _note_missing(self, missing: object) -> list[str]:
        lines: list[str] = []
        for item in missing if isinstance(missing, list) else []:
            if not isinstance(item, dict):
                continue
            optional = " (optional)" if item.get("optional") else ""
            lines.append(f"  missing{optional}: {item.get('value')} — {item.get('hint')}")
            install = item.get("install")
            if isinstance(install, list) and len(install) == 2 and (
                self.install_target is None or not item.get("optional")
            ):
                self.install_target = (str(install[0]), str(install[1]))
        self.install_button.setEnabled(self.install_target is not None)
        return lines

    def _plan_event(self, value: dict) -> None:
        if value.get("t") in {"start", "progress", "done"} and "stage" in value:
            self._show_progress(value)
        if (value.get("t") == "warning" and "missing" not in value
                and not self.advanced.isChecked()):
            # The advanced log already shows the raw event.
            self._append_log(f"{value.get('stage', '')}: {value.get('message', '')}\n")
        if value.get("t") == "warning" and "missing" in value:
            self._append_log("\n".join(self._note_missing(value["missing"])) + "\n")
            return
        stages = value.get("stages")
        if isinstance(stages, list):
            self.install_target = None
            details = {
                stage["id"]: self._note_missing(stage.get("missing")) for stage in stages
            }
            self.plan_label.setText(
                "\n".join(
                    f"{stage['id']}: {'reuse' if stage.get('reuse') else stage['decision']} — "
                    f"{stage.get('reason', '')}"
                    + (f"\n  {stage['stale']}" if stage.get("stale")
                       and stage["stale"] not in stage.get("reason", "") else "")
                    + "".join("\n" + line for line in details[stage["id"]])
                    for stage in stages
                )
            )

    def _navigate(self, index: int) -> None:
        if index != 3:
            self.review_player.stop()
            self.review_timer.stop()
        self.pages.setCurrentIndex(index)
        if index != 6:
            self.subtitles.player.pause()
        elif self.source is not None:
            path = self.source / "_studio" / "project.json"
            if path.is_file() and not self.subtitles.dirty:
                self.subtitles.load_project(Project.load(path))
        if index == 8 and self.source and not self.settings_page.dirty:
            self.settings_page.load_project(self.source / "_studio")
        if index != 3 or self.source is None:
            return
        path = self.source / "_studio" / "project.json"
        if path.is_file():
            self._show_review(Project.load(path))

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        from PyQt6.QtCore import QProcess

        if self.bridge.process.state() != QProcess.ProcessState.NotRunning:
            self._append_log("Processing is active. Use Cancel and wait before closing.\n")
            self.navigation.setCurrentRow(4)
            event.ignore()
            return
        if self.settings_page.probe.state() != QProcess.ProcessState.NotRunning:
            self.settings_page.status.setText(
                "Wait for compute inspection to finish before closing."
            )
            self.navigation.setCurrentRow(8)
            event.ignore()
            return
        if self.modules_page.process.state() != QProcess.ProcessState.NotRunning:
            self.modules_page.status.setText("Wait for installation to finish before closing.")
            self.navigation.setCurrentRow(7)
            event.ignore()
            return
        if self.settings_page.confirm_discard() and self.subtitles.confirm_discard():
            self.subtitles.player.stop()
            event.accept()
        else:
            event.ignore()

    def _finished(self, code: int) -> None:
        if code:
            self.progress_label.setText(
                "Cancelled." if code == 130 else "Stopped with an error; the details are below.")
        if not self.source or code:
            return
        self.progress_bar.setValue(100)
        self.progress_label.setText("Finished.")
        path = self.source / "_studio" / "project.json"
        if path.is_file():
            project = Project.load(path)
            self._show_material(project)
            self._show_review(project)
            self.result.setText(self._result_text(project))
            self.navigation.setCurrentRow(
                {"subtitles": 6, "review": 3, "plan": 2, "result": 5}.get(self.pending_view, 1)
            )

    def _show_material(self, project: Project) -> None:
        self.material.load(project)

    def _material_command(self, command: str, arguments: list[str]) -> None:
        if self.source is None:
            return
        self.pending_view = "material"
        self.navigation.setCurrentRow(4)
        try:
            self.bridge.start(command, self.source, arguments)
        except RuntimeError as exc:
            self.material.status.setText(str(exc))

    def _show_review(self, project: Project) -> None:
        # Decisions may have changed: the next preview request re-renders (or reuses) it.
        self.timeline_preview = None
        self.timeline_player.stop()
        self.timeline_player.setSource(QUrl())
        self.timeline_video.hide()
        self.source_timeline.load(project)
        self.review.clear()
        paths = project.outputs.get("roughcut", [])
        if not paths or not paths[0].is_file():
            self.cut_timeline.set_edit(0, [], [])
            self.review_status.setText("No edit decisions available.")
            return
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        timeline_cuts = [
            TimelineCut(
                str(cut.get("id", "")), float(cut.get("start", 0.0)), float(cut.get("end", 0.0)),
                str(cut.get("reason", "")), str(cut.get("note", "")),
                bool(cut.get("accepted", True)),
            )
            for cut in data.get("cuts", [])
        ]
        duration = max(
            [float(item.get("end", 0.0)) for item in data.get("keep", [])]
            + [cut.end for cut in timeline_cuts] + [0.0]
        )
        self.cut_timeline.set_edit(duration, timeline_cuts, [
            (float(marker.get("at", 0.0)), str(marker.get("text", "")))
            for marker in data.get("markers", [])
        ])
        for cut in data.get("cuts", []):
            accepted = bool(cut.get("accepted", True))
            item = QTreeWidgetItem(
                self.review,
                [
                    str(cut.get("id", "")),
                    str(cut.get("reason", "")),
                    f"{float(cut.get('start', 0.0)):.2f}",
                    f"{float(cut.get('end', 0.0)):.2f}",
                    "Accepted" if accepted else "Not applied",
                ],
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(cut.get("id", "")))
            confidence = float(cut.get("confidence", 1.0))
            note = str(cut.get("note", ""))
            for column in range(item.columnCount()):
                item.setToolTip(column, f"{note} (confidence {confidence:.2f})")
            if confidence < 0.8:
                item.setText(1, f"{cut.get('reason', '')} · proposal")
        kept = self.cut_timeline.kept_duration()
        self.review_status.setText(
            f"{self.review.topLevelItemCount()} edit decisions; "
            f"program {kept / 60:.1f} of {duration / 60:.1f} min"
        )

    def _review_cut(self, accepted: bool) -> None:
        item = self.review.currentItem()
        if item is None or self.source is None:
            return
        cut_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        self.pending_view = "review"
        self.navigation.setCurrentRow(4)
        self.bridge.start(
            "review",
            self.source,
            ["--cut", cut_id, "--accept" if accepted else "--reject"],
        )

    def _append_output(self, text: str) -> None:
        """Worker stdout: JSON events, shown raw only in the advanced mode."""
        if self.advanced.isChecked():
            self._append_log(text)

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def _open_result(self) -> None:
        if self.source:
            folder = self.source / "_studio" / "export"
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def main() -> int:
    app = QApplication(sys.argv)
    stylesheet = files("studio.gui").joinpath("theme.qss").read_text(encoding="utf-8")
    app.setStyleSheet(stylesheet)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
