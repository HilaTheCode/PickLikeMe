"""Gallery view infrastructure built on Qt Model/View."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QListView


class GalleryView(QListView):
    keyPressSignal = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setViewMode(QListView.IconMode)
        self.setUniformItemSizes(True)
        self.setResizeMode(QListView.Adjust)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_K, Qt.Key.Key_R, Qt.Key.Key_N, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.keyPressSignal.emit(key)
            event.accept()
        else:
            super().keyPressEvent(event)
