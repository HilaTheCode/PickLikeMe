"""Gallery view infrastructure built on Qt Model/View."""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import QListView

from ... import theme
from .thumbnail_delegate import ThumbnailCardDelegate


class GalleryView(QListView):
    keyPressSignal = Signal(int)
    # Emitted when a card's own Keep/Reject/Neutral button is clicked -
    # (image_path, status). Deliberately separate from the view's own
    # selection: clicking a card's button acts on that card only and must
    # not disturb whatever multi-selection the user already has.
    decisionRequested = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setViewMode(QListView.IconMode)
        self.setUniformItemSizes(True)
        self.setResizeMode(QListView.Adjust)
        self.setSpacing(10)
        self.setMouseTracking(True)  # enable hover state in delegate
        self._delegate = ThumbnailCardDelegate(self)
        self.setItemDelegate(self._delegate)
        self.setGridSize(QSize(self._delegate.CARD_WIDTH, self._delegate.CARD_HEIGHT))
        self._empty_message = "Open a folder to begin reviewing images"

    def set_empty_message(self, message: str) -> None:
        """Text shown centered in the viewport when the model has zero
        rows - a folder with no images, or a filter that matches none,
        should say so instead of leaving a blank void."""
        self._empty_message = message
        self.viewport().update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        super().paintEvent(event)
        model = self.model()
        if model is not None and model.rowCount() == 0 and self._empty_message:
            painter = QPainter(self.viewport())
            painter.setPen(QColor(theme.current_palette().text_muted))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self._empty_message)
            painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override signature
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                card_rect = self.visualRect(index)
                for status, btn_rect in self._delegate.button_rects(card_rect).items():
                    if btn_rect.contains(event.pos()):
                        item = self.model().item_at(index.row())
                        if item is not None:
                            self.decisionRequested.emit(item.path, status)
                        event.accept()
                        return
        super().mousePressEvent(event)

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
        elif key == Qt.Key.Key_Space:
            # Keyboard-only multi-select: toggle the current row into/out
            # of the selection without touching the mouse, so a whole burst
            # can be selected with arrow keys + Space before K/R/N applies
            # the decision to all of them at once (MainWindow.apply_review_status).
            index = self.currentIndex()
            if index.isValid():
                self.selectionModel().select(index, QItemSelectionModel.SelectionFlag.Toggle)
            event.accept()
        else:
            super().keyPressEvent(event)
