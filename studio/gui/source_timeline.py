"""Read-only synchronized source overview with native keyboard navigation."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QProgressBar, QTreeWidget, QTreeWidgetItem

from studio.core.project import Project
from studio.core.voice_fill import playback_voice


class SourceTimeline(QTreeWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setHeaderLabels(["Source", "Start (s)", "End (s)", "Clock rate", "Coverage"])
        self.setAccessibleName("Synchronized source timeline")
        self.setMinimumHeight(120)

    def load(self, project: Project) -> None:
        self.clear()
        assets = {asset.id: asset for asset in project.assets}
        duration = max((p.offset_s + p.duration_s * p.k for p in project.placements), default=0)
        groups: dict[str, QTreeWidgetItem] = {}
        for placement in project.placements:
            asset = assets.get(placement.asset_id)
            if asset is None:
                continue
            group = f"{asset.role.value}: {asset.group_id or asset.id}"
            parent = groups.get(group)
            if parent is None:
                parent = QTreeWidgetItem(self, [group])
                groups[group] = parent
            end = placement.offset_s + placement.duration_s * placement.k
            item = QTreeWidgetItem(parent, [str(asset.path), f"{placement.offset_s:.3f}",
                                          f"{end:.3f}", f"{placement.k:.6f}"])
            voice = playback_voice(project, asset.id)
            detail = (f"{placement.provenance}; source in: {placement.in_s:.3f}s; "
                      f"voice: {voice.name if voice else 'original audio'}")
            item.setToolTip(0, detail)
            item.setData(0, Qt.ItemDataRole.UserRole, asset.id)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(round(1000 * placement.duration_s * placement.k / duration)
                         if duration > 0 else 0)
            bar.setFormat(f"{placement.offset_s:.2f} - {end:.2f}s")
            bar.setAccessibleName(f"{asset.id} coverage: {placement.offset_s:.2f} to {end:.2f}s")
            self.setItemWidget(item, 4, bar)
        self.expandAll()
        if self.topLevelItemCount():
            self.setCurrentItem(self.topLevelItem(0))
