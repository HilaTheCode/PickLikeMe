"""Proxy model scaffolding for gallery filtering and sorting."""

from __future__ import annotations

from PySide6.QtCore import QSortFilterProxyModel


class GalleryProxyModel(QSortFilterProxyModel):
    def __init__(self) -> None:
        super().__init__()
        self.setDynamicSortFilter(True)
