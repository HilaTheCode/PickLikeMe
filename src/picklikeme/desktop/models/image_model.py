"""Qt Model/View infrastructure for the desktop gallery."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt

from .image_item import ImageItem


class ImageModel(QAbstractListModel):
    def __init__(self, items: list[ImageItem] | None = None) -> None:
        super().__init__()
        self._items = items or []
        self._thumbnail_provider: Callable[[str], Any] | None = None

    def set_thumbnail_provider(self, provider: Callable[[str], Any] | None) -> None:
        """A callable returning a QPixmap (or None) for a given image path,
        used lazily to populate Qt.DecorationRole - only ever called for
        rows Qt actually asks to paint, not the whole gallery up front."""
        self._thumbnail_provider = provider

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        del parent
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        item = self._items[index.row()]
        if role == Qt.DisplayRole:
            suffix = f" [{item.review_status}]"
            if item.ai_suggestion and item.ai_suggestion != item.review_status:
                suffix += f" (AI:{item.ai_suggestion})"
            return item.display_name + suffix
        if role == Qt.DecorationRole and self._thumbnail_provider is not None:
            return self._thumbnail_provider(item.path)
        if role == Qt.UserRole:
            return item
        return None

    def set_items(self, items: list[ImageItem]) -> None:
        self.beginResetModel()
        self._items = items
        self.endResetModel()

    def items(self) -> list[ImageItem]:
        return list(self._items)

    def item_at(self, row: int) -> ImageItem | None:
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def notify_thumbnail_ready(self, path: str) -> None:
        """A background thumbnail decode for `path` finished - repaint just
        that row instead of the caller doing a full set_items()/model reset
        (which would re-request decoration data, and therefore re-trigger
        loading, for every visible cell). Safe to call for a path that is
        no longer in the model (folder changed, filtered out mid-load): a
        stale in-flight decode simply finds nothing and does nothing."""
        for row, item in enumerate(self._items):
            if item.path == path:
                index = self.index(row, 0)
                self.dataChanged.emit(index, index, [Qt.DecorationRole])
                return
