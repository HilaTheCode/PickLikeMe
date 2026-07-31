import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtCore import Qt

from picklikeme.desktop.core.caching import CacheManager
from picklikeme.desktop.core.commands import OpenFolderCommand
from picklikeme.desktop.core.events import EventBus
from picklikeme.desktop.core.thumbnail_loader import ThumbnailLoadTask, ThumbnailReadySignal
from picklikeme.desktop.models.image_item import ImageItem
from picklikeme.desktop.models.image_model import ImageModel
from picklikeme.desktop.views.gallery.gallery_view import GalleryView

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]


class DummyService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def open_folder(self, folder: str) -> dict[str, str]:
        self.calls.append(folder)
        return {"input_folder": folder}


def test_event_bus_delivers_payload() -> None:
    bus = EventBus()
    events: list[dict[str, object]] = []

    def observer(event) -> None:
        events.append(event.payload or {})

    bus.subscribe("folder-opened", observer)
    bus.publish("folder-opened", {"folder": "/tmp/shoot"})

    assert events == [{"folder": "/tmp/shoot"}]


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_thumbnail_load_task_emits_failed_not_silence_on_exception() -> None:
    """Regression: run() used to just `return` on a bad frame, emitting
    nothing at all - indistinguishable from "still loading" to the caller,
    which left that image's (path, with_boxes) key stuck in
    MainWindow._thumbnails_loading forever (see _on_thumbnail_failed), with
    no way to ever retry it for the rest of the session."""
    QApplication.instance() or QApplication([])
    signal = ThumbnailReadySignal()
    failed: list[tuple[str, bool]] = []
    ready: list[tuple[str, bool]] = []
    signal.failed.connect(lambda path, with_boxes: failed.append((path, with_boxes)))
    signal.ready.connect(lambda path, with_boxes, pixmap: ready.append((path, with_boxes)))

    def raising_load_fn():
        raise RuntimeError("simulated decode failure")

    ThumbnailLoadTask("bad/frame.NEF", False, raising_load_fn, signal).run()

    assert failed == [("bad/frame.NEF", False)]
    assert ready == []


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_thumbnail_load_task_emits_failed_on_none_result() -> None:
    QApplication.instance() or QApplication([])
    signal = ThumbnailReadySignal()
    failed: list[tuple[str, bool]] = []
    signal.failed.connect(lambda path, with_boxes: failed.append((path, with_boxes)))

    ThumbnailLoadTask("no/thumbnail.NEF", True, lambda: None, signal).run()

    assert failed == [("no/thumbnail.NEF", True)]


def test_cache_manager_tracks_hits_and_misses() -> None:
    cache = CacheManager()
    cache.put_thumbnail("one", "img")

    assert cache.get_thumbnail("one") == "img"
    assert cache.get_thumbnail("two") is None
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_open_folder_command_delegates_to_service() -> None:
    service = DummyService()
    command = OpenFolderCommand(service=service, folder="/tmp/shoot")

    result = command.execute()

    assert result == {"input_folder": "/tmp/shoot"}
    assert service.calls == ["/tmp/shoot"]


def test_image_model_exposes_items() -> None:
    model = ImageModel([ImageItem(path="/tmp/demo.jpg", file_name="demo.jpg")])

    assert model.rowCount() == 1
    assert model.data(model.index(0, 0), role=Qt.DisplayRole) == "demo.jpg [neutral]"


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_gallery_view_can_be_created() -> None:
    app = QApplication.instance() or QApplication([])
    view = GalleryView()

    assert view is not None
    app.quit()
