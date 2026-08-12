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

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        super().resizeEvent(event)
        self._center_grid()

    def _center_grid(self) -> None:
        # QListView's icon-mode flow always starts a row at the viewport's
        # left edge, so a viewport wider than an exact multiple of the grid
        # cell leaves the remainder as dead space on the right only. Adding
        # equal left/right viewport margins re-centers the grid as a whole
        # without touching how it wraps or scrolls - the row/column layout
        # itself is untouched, only where it starts.
        #
        # Deliberately hung off resizeEvent (this widget's own geometry
        # change), not viewportEvent: setViewportMargins() below resizes the
        # *viewport* child widget, which would re-trigger a viewportEvent
        # hook and, since a wider/narrower viewport can flip whether the
        # content needs a vertical scrollbar, that scrollbar appearing or
        # disappearing changes the viewport width again - an unbounded
        # resize<->margin feedback loop that stack-overflowed in practice.
        # resizeEvent isn't part of that loop: setViewportMargins never
        # resizes this widget itself, only its viewport.
        item_width = self.gridSize().width()
        if item_width <= 0:
            return
        margins = self.viewportMargins()
        left, right = margins.left(), margins.right()
        available = self.viewport().width() + left + right
        columns = max(1, available // item_width)
        extra = max(0, available - columns * item_width)
        # Shrinking the viewport to *exactly* columns * item_width (extra
        # margin split with none left over) puts the post-margin viewport
        # right at an exact multiple of the cell width - and QListView's own
        # grid layout drops a column right at that boundary (confirmed by
        # measuring visualRect: an exact-fit viewport rendered one fewer
        # column than the arithmetic said should fit). Keeping 1px of
        # leftover slack unclaimed keeps the post-margin viewport strictly
        # wider than columns * item_width, so Qt reliably keeps all of them.
        usable_extra = max(0, extra - 1)
        new_left = usable_extra // 2
        new_right = usable_extra - new_left
        if (new_left, new_right) != (left, right):
            self.setViewportMargins(new_left, 0, new_right, 0)

    def set_empty_message(self, message: str) -> None:
        """Text shown centered in the viewport when the model has zero
        rows - a folder with no images, or a filter that matches none,
        should say so instead of leaving a blank void."""
        self._empty_message = message
        self.viewport().update()

    def set_color_source(self, strategy_id: str | None, *, score_range: tuple[float, float] | None = None) -> None:
        """Which of the two Color modes the cards are painted in (see
        `ThumbnailCardDelegate._resolve_status`). `None` means "User
        Decision" - Keep/Reject/Undecided from the photographer's own
        decisions alone. A strategy id colors each card by THAT strategy's
        own score, spread across `score_range` (the min/max currently
        visible)."""
        self._delegate.set_color_source(strategy_id)
        self._delegate.set_score_range(score_range)
        self.viewport().update()

    def set_show_burst_badges(self, enabled: bool) -> None:
        """Show the "+N other in burst" badge on a card whose burst has
        other members - only meaningful once Collapse Bursts is on (see
        MainWindow._on_toggle_collapse_bursts); off by default so the
        gallery's normal, one-card-per-image view is never decorated with
        burst chrome it did not ask for."""
        self._delegate.set_show_burst_badge(enabled)
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
