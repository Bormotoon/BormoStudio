"""Phase-one Studio desktop shell over the shared CLI and project model."""

from __future__ import annotations

import json
import sys
from importlib.resources import files
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from studio.core.project import Project
from studio.gui.bridge import RunnerBridge
from studio.gui.cut_timeline import CutTimeline, TimelineCut
from studio.gui.modules import ModulesPage
from studio.gui.settings_page import SettingsPage
from studio.gui.setup_wizard import SetupWizard
from studio.gui.subtitles import SubtitleEditor


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ultimate Video Forge")
        self.resize(1100, 720)
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
        self.pages = QStackedWidget()
        self.material = QTreeWidget()
        self.material.setHeaderLabels(["Asset", "Role", "Group", "Device"])
        self.material.currentItemChanged.connect(self._select_material)
        self.material_role = QComboBox()
        self.material_role.addItems(["camera", "recorder", "ignore", "unknown"])
        self.material_group = QLineEdit()
        self.material_device = QLineEdit()
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
        self.cut_timeline.cut_selected.connect(self._select_cut_from_timeline)
        self.review.currentItemChanged.connect(self._select_cut_from_list)
        self.plan_label = QLabel("Scan a folder to build the plan.")
        self.log = QLabel("")
        self.log.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.log.setWordWrap(True)
        self.result = QLabel("No results yet.")
        self._build_pages()
        layout = QHBoxLayout()
        layout.addWidget(self.navigation, 1)
        layout.addWidget(self.pages, 4)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.navigation.currentRowChanged.connect(self._navigate)
        self.navigation.setCurrentRow(0)
        self.bridge.output.connect(self._append_log)
        self.bridge.finished.connect(self._finished)
        self.bridge.failed.connect(self._append_log)
        self.bridge.stage_event.connect(self._plan_event)
        self.subtitles.render_requested.connect(self._render_subtitles)

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
        choose = QPushButton("Choose material folder")
        choose.clicked.connect(self._choose)
        start_layout.addWidget(title)
        start_layout.addWidget(choose)
        setup = QPushButton("Set up processing")
        setup.clicked.connect(self._setup)
        start_layout.addWidget(setup)
        start_layout.addStretch()
        self.pages.addWidget(start)
        self.pages.addWidget(self._material_page())
        self.pages.addWidget(self._page(self.plan_label, "Build plan", self._plan))
        run = QPushButton("Run processing plan")
        run.clicked.connect(self._run)
        self.pages.widget(2).layout().addWidget(run)
        self.pages.addWidget(self._review_page())
        self.pages.addWidget(self._page(self.log, "Cancel", self.bridge.cancel))
        self.pages.addWidget(self._page(self.result, "Open export folder", self._open_result))
        self.pages.addWidget(self.subtitles)
        self.pages.addWidget(self.modules_page)
        self.pages.addWidget(self.settings_page)

    def _material_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.material)
        fields = QFormLayout()
        fields.addRow("Role", self.material_role)
        fields.addRow("Group", self.material_group)
        fields.addRow("Device", self.material_device)
        layout.addLayout(fields)
        save = QPushButton("Save material assignment")
        save.clicked.connect(self._save_material)
        layout.addWidget(save)
        return page

    def _review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.cut_timeline)
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
        self.pending_view = "result"
        self.navigation.setCurrentRow(4)

    def _plan_event(self, value: dict) -> None:
        stages = value.get("stages")
        if isinstance(stages, list):
            self.plan_label.setText(
                "\n".join(
                    f"{stage['id']}: {'reuse' if stage.get('reuse') else stage['decision']} — "
                    f"{stage.get('reason', '')}"
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

    def closeEvent(self, event: object) -> None:
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
        if not self.source or code:
            return
        path = self.source / "_studio" / "project.json"
        if path.is_file():
            project = Project.load(path)
            self._show_material(project)
            self._show_review(project)
            self.result.setText(str(project.outputs))
            self.navigation.setCurrentRow(
                {"subtitles": 6, "review": 3, "plan": 2, "result": 5}.get(self.pending_view, 1)
            )

    def _show_material(self, project: Project) -> None:
        self.material.clear()
        for asset in project.assets:
            item = QTreeWidgetItem(
                self.material,
                [str(asset.path), asset.role.value, asset.group_id or "", asset.device or ""],
            )
            item.setData(0, Qt.ItemDataRole.UserRole, asset.id)

    def _select_material(
        self,
        item: QTreeWidgetItem | None,
        previous: QTreeWidgetItem | None,
    ) -> None:
        del previous
        if not item or not self.source:
            return
        path = self.source / "_studio" / "project.json"
        if not path.is_file():
            return
        project = Project.load(path)
        asset_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        asset = next((entry for entry in project.assets if entry.id == asset_id), None)
        if asset is None:
            return
        self.material_role.setCurrentText(asset.role.value)
        self.material_group.setText(asset.group_id or "")
        self.material_device.setText(asset.device or "")

    def _save_material(self) -> None:
        item = self.material.currentItem()
        if item is None or self.source is None:
            return
        asset_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        self.pending_view = "material"
        self.navigation.setCurrentRow(4)
        self.bridge.start(
            "material",
            self.source,
            [
                "--asset",
                asset_id,
                "--role",
                self.material_role.currentText(),
                "--group",
                self.material_group.text(),
                "--device",
                self.material_device.text(),
            ],
        )

    def _show_review(self, project: Project) -> None:
        self.review.clear()
        paths = project.outputs.get("roughcut", [])
        if not paths or not paths[0].is_file():
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

    def _append_log(self, text: str) -> None:
        self.log.setText(self.log.text() + text)

    def _open_result(self) -> None:
        if self.source:
            folder = self.source / "_studio" / "export"
            QFileDialog.getOpenFileName(self, "Ultimate Video Forge export", str(folder))


def main() -> int:
    app = QApplication(sys.argv)
    stylesheet = files("studio.gui").joinpath("theme.qss").read_text(encoding="utf-8")
    app.setStyleSheet(stylesheet)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
