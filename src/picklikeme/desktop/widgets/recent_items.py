"""A reusable "recent items" QMenu: most-recent-first, deduplicated,
length-limited, persisted via QSettings.

Built generic on purpose - it knows nothing about folders. It stores and
displays strings; what a string means (a folder path today, perhaps a
project file tomorrow) and what happens when one is chosen is entirely up
to the owner's `on_select` callback. See MainWindow's Recent Folders menu
for the current use, and its `_open_recent_folder` for how folder-specific
concerns (does it still exist on disk, actually opening it) are kept
outside this class rather than baked into it - a future Recent Projects
menu can reuse this file unchanged.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QFontMetrics
from PySide6.QtWidgets import QMenu

DEFAULT_RECENT_ITEMS_LIMIT = 5

# Menu items elide past this width, with the full value always available as
# a tooltip - long folder paths would otherwise force the whole menu wide.
_MAX_LABEL_WIDTH_PX = 480


class RecentItemsMenu:
    """Owns one QMenu's contents and one QSettings list."""

    def __init__(
        self,
        menu: QMenu,
        settings: QSettings,
        *,
        settings_key: str,
        on_select: Callable[[str], None],
        empty_label: str = "Nothing recent yet",
        clear_label: str = "Clear",
        limit: int = DEFAULT_RECENT_ITEMS_LIMIT,
    ) -> None:
        self._menu = menu
        self._settings = settings
        self._settings_key = settings_key
        self._on_select = on_select
        self._empty_label = empty_label
        self._clear_label = clear_label
        self._limit = limit
        self._items: list[str] = []
        self._actions: list[QAction] = []
        self._menu.setToolTipsVisible(True)
        self.reload()

    def items(self) -> list[str]:
        return list(self._items)

    def reload(self) -> None:
        """Re-read from QSettings, discarding any in-memory state."""
        raw = self._settings.value(self._settings_key, [])
        if isinstance(raw, str):
            raw = [raw]
        items = [item for item in raw if isinstance(item, str) and item]
        self._items = list(dict.fromkeys(items))[: self._limit]
        self._rebuild()

    def remember(self, item: str) -> None:
        """Move `item` to the front (inserting it if new), trim to the
        configured limit, and persist."""
        self._items = [entry for entry in self._items if entry != item]
        self._items.insert(0, item)
        self._items = self._items[: self._limit]
        self._persist()

    def remove(self, item: str) -> None:
        if item not in self._items:
            return
        self._items = [entry for entry in self._items if entry != item]
        self._persist()

    def clear(self) -> None:
        self._items = []
        self._persist()

    def _persist(self) -> None:
        self._settings.setValue(self._settings_key, self._items)
        self._rebuild()

    def _rebuild(self) -> None:
        for action in self._actions:
            self._menu.removeAction(action)
            action.deleteLater()
        self._actions = []

        if not self._items:
            self._menu.setEnabled(False)
            placeholder = QAction(self._empty_label, self._menu)
            placeholder.setEnabled(False)
            self._menu.addAction(placeholder)
            self._actions.append(placeholder)
            return

        self._menu.setEnabled(True)
        metrics = QFontMetrics(self._menu.font())
        for item in self._items:
            label = metrics.elidedText(item, Qt.ElideMiddle, _MAX_LABEL_WIDTH_PX)
            action = QAction(label, self._menu)
            action.setToolTip(item)
            action.setStatusTip(item)
            action.triggered.connect(lambda checked=False, selected=item: self._on_select(selected))
            self._menu.addAction(action)
            self._actions.append(action)

        separator = self._menu.addSeparator()
        self._actions.append(separator)
        clear_action = QAction(self._clear_label, self._menu)
        clear_action.triggered.connect(self.clear)
        self._menu.addAction(clear_action)
        self._actions.append(clear_action)
