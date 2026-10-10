"""Material screen: groups with drag and drop, recorder lanes and speaker cameras."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from studio.core.project import AssetRole, Project
from studio.core.recorder_tracks import channel_count, project_speakers
from studio.stages.camera_selection import WIDE_SHOT

ASSET_ROLE = Qt.ItemDataRole.UserRole
GROUP_ROLE = Qt.ItemDataRole.UserRole + 1
NO_GROUP = "(no group)"


class MaterialTree(QTreeWidget):
    """Assets under their group; dropping assets on a group regroups them.

    The tree never moves items itself: the drop is turned into a request and
    the view is rebuilt from the saved project, so it cannot drift from disk.
    """

    regroup_requested = pyqtSignal(list, str)

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderLabels(["Asset", "Role", "Device", "Channels"])
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setAccessibleName("Material groups")

    def selected_assets(self) -> list[str]:
        return [
            str(item.data(0, ASSET_ROLE)) for item in self.selectedItems()
            if item.data(0, ASSET_ROLE)
        ]

    def drop_group(self, target: QTreeWidgetItem | None) -> str | None:
        """Group named by a drop target: a group node or any asset inside it."""
        if target is None:
            return None
        group = target.data(0, GROUP_ROLE)
        parent = target.parent()
        if group is None and parent is not None:
            group = parent.data(0, GROUP_ROLE)
        return None if group is None else str(group)

    def dropEvent(self, event: QDropEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        group = self.drop_group(self.itemAt(event.position().toPoint()))
        assets = self.selected_assets()
        event.setDropAction(Qt.DropAction.IgnoreAction)
        event.accept()
        if group is not None and assets:
            self.regroup_requested.emit(assets, group)


class MaterialPage(QWidget):
    """Edits are emitted as CLI commands so GUI and CLI share one code path."""

    command_requested = pyqtSignal(str, list)

    def __init__(self) -> None:
        super().__init__()
        self.project: Project | None = None
        self.tree = MaterialTree()
        self.tree.currentItemChanged.connect(self._select_asset)
        self.tree.regroup_requested.connect(self._regroup)
        self.role = QComboBox()
        self.role.addItems([role.value for role in AssetRole])
        self.group = QLineEdit()
        self.group.setPlaceholderText("Type a new group or drag assets onto one")
        self.device = QLineEdit()
        self.tracks = QTreeWidget()
        self.tracks.setHeaderLabels(["Lane", "Channels", "Speaker", "Enabled"])
        self.tracks.setAccessibleName("Recorder tracks")
        self.tracks.currentItemChanged.connect(self._select_track)
        self.track_channels = QLineEdit()
        self.track_channels.setPlaceholderText("0 or 0,1 for a stereo pair")
        self.track_speaker = QLineEdit()
        self.track_enabled = QCheckBox("Use this lane")
        self.track_enabled.setChecked(True)
        self.camera_speaker = QComboBox()
        self.camera_speaker.setEditable(True)
        self.camera_choice = QComboBox()
        self.cameras = QTreeWidget()
        self.cameras.setHeaderLabels(["Speaker", "Camera"])
        self.cameras.setAccessibleName("Speaker cameras")
        self.status = QLabel("Scan a folder to see its material.")
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree, 3)
        asset_box = QGroupBox("Asset")
        fields = QFormLayout(asset_box)
        fields.addRow("Role", self.role)
        fields.addRow("Group", self.group)
        fields.addRow("Device", self.device)
        save = QPushButton("Save asset")
        save.clicked.connect(self._save_asset)
        fields.addRow(save)
        track_box = QGroupBox("Recorder tracks")
        track_layout = QVBoxLayout(track_box)
        track_layout.addWidget(self.tracks)
        track_fields = QFormLayout()
        track_fields.addRow("Channels", self.track_channels)
        track_fields.addRow("Speaker", self.track_speaker)
        track_fields.addRow(self.track_enabled)
        save_track = QPushButton("Save lane")
        save_track.clicked.connect(self._save_track)
        track_fields.addRow(save_track)
        track_layout.addLayout(track_fields)
        camera_box = QGroupBox("Speaker cameras")
        camera_layout = QVBoxLayout(camera_box)
        camera_layout.addWidget(self.cameras)
        camera_fields = QFormLayout()
        camera_fields.addRow("Speaker", self.camera_speaker)
        camera_fields.addRow("Camera", self.camera_choice)
        buttons = QHBoxLayout()
        assign = QPushButton("Assign camera")
        assign.clicked.connect(self._assign_camera)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear_camera)
        buttons.addWidget(assign)
        buttons.addWidget(clear)
        camera_fields.addRow(buttons)
        camera_layout.addLayout(camera_fields)
        lower = QHBoxLayout()
        lower.addWidget(asset_box)
        lower.addWidget(track_box)
        lower.addWidget(camera_box)
        layout.addLayout(lower, 2)
        layout.addWidget(self.status)

    def load(self, project: Project) -> None:
        self.project = project
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for asset in project.assets:
            name = asset.group_id or NO_GROUP
            parent = groups.get(name)
            if parent is None:
                parent = QTreeWidgetItem(self.tree, [name])
                parent.setData(0, GROUP_ROLE, asset.group_id or "")
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
                groups[name] = parent
            child = QTreeWidgetItem(parent, [
                str(asset.path), asset.role.value, asset.device or "",
                str(channel_count(asset) or ""),
            ])
            child.setData(0, ASSET_ROLE, asset.id)
            child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsDropEnabled)
        self.tree.expandAll()
        self.tracks.clear()
        for track in project.recorder_tracks:
            item = QTreeWidgetItem(self.tracks, [
                track.name or track.asset_id, ",".join(str(c) for c in track.channels),
                track.speaker or "", "yes" if track.enabled else "no",
            ])
            item.setData(0, ASSET_ROLE, track.asset_id)
        self.cameras.clear()
        for speaker, camera in sorted(project.speaker_cameras.items()):
            label = "Wide shot" if speaker == WIDE_SHOT else speaker
            QTreeWidgetItem(self.cameras, [label, camera])
        self.camera_speaker.clear()
        self.camera_speaker.addItems(
            [name for name in project_speakers(project) if name != WIDE_SHOT]
        )
        # The reserved entry assigns the wide shot used for overlaps and inserts.
        self.camera_speaker.addItem("Wide shot (overlaps, rapid exchanges)", WIDE_SHOT)
        self.camera_choice.clear()
        choices: list[str] = []
        for asset in project.assets:
            if asset.role is AssetRole.CAMERA:
                for value in (asset.group_id, asset.id):
                    if value and value not in choices:
                        choices.append(value)
        self.camera_choice.addItems(choices)
        self.status.setText(
            f"{len(project.assets)} assets in {len(groups)} groups, "
            f"{len(project.recorder_tracks)} recorder lanes"
        )

    def _asset(self, asset_id: str):  # type: ignore[no-untyped-def]
        if self.project is None:
            return None
        return next((item for item in self.project.assets if item.id == asset_id), None)

    def _select_asset(self, item: QTreeWidgetItem | None, _previous: object) -> None:
        asset = self._asset(str(item.data(0, ASSET_ROLE))) if item is not None else None
        if asset is None:
            return
        self.role.setCurrentText(asset.role.value)
        self.group.setText(asset.group_id or "")
        self.device.setText(asset.device or "")

    def _select_track(self, item: QTreeWidgetItem | None, _previous: object) -> None:
        if item is None:
            return
        self.track_channels.setText(item.text(1))
        self.track_speaker.setText(item.text(2))
        self.track_enabled.setChecked(item.text(3) == "yes")

    def _regroup(self, assets: list[str], group: str) -> None:
        arguments: list[str] = []
        for asset_id in assets:
            arguments.extend(["--asset", asset_id])
        self.command_requested.emit("material", [*arguments, "--group", group])

    def _save_asset(self) -> None:
        assets = self.tree.selected_assets()
        if not assets:
            self.status.setText("Select an asset first.")
            return
        arguments: list[str] = []
        for asset_id in assets:
            arguments.extend(["--asset", asset_id])
        self.command_requested.emit("material", [
            *arguments, "--role", self.role.currentText(), "--group", self.group.text(),
            "--device", self.device.text(),
        ])

    def _save_track(self) -> None:
        item = self.tracks.currentItem()
        if item is None:
            self.status.setText("Select a recorder lane first.")
            return
        channels = self.track_channels.text().replace(" ", "")
        self.command_requested.emit("tracks", [
            "--asset", str(item.data(0, ASSET_ROLE)), "--channels", channels,
            "--speaker", self.track_speaker.text(),
            "--enable" if self.track_enabled.isChecked() else "--disable",
        ])

    def _speaker_key(self) -> str:
        text = self.camera_speaker.currentText().strip()
        index = self.camera_speaker.findText(text)
        data = self.camera_speaker.itemData(index) if index >= 0 else None
        return str(data) if data else text

    def _assign_camera(self) -> None:
        speaker = self._speaker_key()
        camera = self.camera_choice.currentText()
        if not speaker or not camera:
            self.status.setText("Choose a speaker and a camera.")
            return
        self.command_requested.emit("cameras", ["--speaker", speaker, "--camera", camera])

    def _clear_camera(self) -> None:
        speaker = self._speaker_key()
        if speaker:
            self.command_requested.emit("cameras", ["--speaker", speaker, "--clear"])
