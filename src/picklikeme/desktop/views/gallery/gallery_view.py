"""Gallery view infrastructure built on Qt Model/View."""

from __future__ import annotations

from PySide6.QtWidgets import QListView


class GalleryView(QListView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setViewMode(QListView.IconMode)
        self.setUniformItemSizes(True)
        self.setResizeMode(QListView.Adjust)
