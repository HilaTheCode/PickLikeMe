"""Background thumbnail decoding for the gallery.

Qt calls ImageModel.data(Qt.DecorationRole) from the paint path, on the UI
thread. review_thumbnail() decodes a RAW file the first time it's asked
for - real work, hundreds of ms to seconds depending on the format and
file size - and every plain synchronous call there blocked the UI thread
for that long. Combined with a folder full of real RAW files and a gallery
that (before this module existed) reset its whole model on every loading-
progress tick, this was enough for Windows to mark the app "Not
Responding" on large folders.

The fix: never decode on the UI thread. A cache miss returns None
immediately (the delegate already paints a blank card until data arrives)
and kicks off a background QRunnable; when it finishes, a queued Qt signal
(safe to emit from a worker thread - QObject.moveToThread semantics mean
the connected slot still runs on the GUI thread) tells the model to repaint
just that one row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QPixmap


class ThumbnailReadySignal(QObject):
    """Owned by the GUI thread; emitting from a worker thread is safe -
    Qt auto-queues the connected slot onto this object's (the GUI) thread."""

    ready = Signal(str, QPixmap)


class ThumbnailLoadTask(QRunnable):
    """Decodes one thumbnail off the UI thread and reports back via signal."""

    def __init__(self, path: str, thumbnail_path_fn: Callable[[str], Path | None], signal: ThumbnailReadySignal) -> None:
        super().__init__()
        self._path = path
        self._thumbnail_path_fn = thumbnail_path_fn
        self._signal = signal
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            thumbnail_path = self._thumbnail_path_fn(self._path)
        except Exception:  # noqa: BLE001 - a bad frame must not break the gallery
            return
        if thumbnail_path is None:
            return
        pixmap = QPixmap(str(thumbnail_path))
        if pixmap.isNull():
            return
        self._signal.ready.emit(self._path, pixmap)
