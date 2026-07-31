"""Gallery view infrastructure built on Qt Model/View."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QListView

from .thumbnail_delegate import ThumbnailCardDelegate


class GalleryView(QListView):
    keyPressSignal = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setViewMode(QListView.IconMode)
        self.setUniformItemSizes(True)
        self.setResizeMode(QListView.Adjust)
        self.setSpacing(4)
        self.setMouseTracking(True)  # enable hover state in delegate
        self._delegate = ThumbnailCardDelegate(self)
        self.setItemDelegate(self._delegate)
        self.setGridSize(QSize(self._delegate.CARD_WIDTH, self._delegate.CARD_HEIGHT))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # K/R/N are deliberately NOT handled here: MainWindow's Review menu
        # QActions already own those shortcuts window-wide (and are shared
        # with the toolbar) - intercepting them a second time here would
        # register an ambiguous duplicate shortcut. Return/Enter has no
        # QAction shortcut of its own, so it's safe and useful to open the
        # Loupe for the current selection directly from the gallery.
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.keyPressSignal.emit(key)
            event.accept()
        else:
            super().keyPressEvent(event)
