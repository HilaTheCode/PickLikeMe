"""Desktop wiring for Burst Analysis: the Collapse Bursts gallery toggle and
burst-scoped Loupe navigation.

Backend burst grouping/ranking is tested in isolation in
test_burst_analysis.py; ReviewSession's own wiring on top of it in
test_review_session.py's BurstInfoTests. This file covers only the desktop
layer built on top of both: MainWindow._apply_filter's collapsing,
ThumbnailCardDelegate's badge, and _open_loupe_for_item's burst-scoped
navigation.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _window(tmp_path):
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    return MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    ), service


def _burst_items():
    """Two bursts: "b1" (3 members, images[1] is burst_best) and "b2" (a
    singleton, always its own burst_best)."""
    from picklikeme.desktop.models.image_item import ImageItem

    return [
        ImageItem(path="/x/a.nef", file_name="a.nef", burst_id="b1", burst_size=3, burst_rank=2, burst_best=False),
        ImageItem(path="/x/b.nef", file_name="b.nef", burst_id="b1", burst_size=3, burst_rank=1, burst_best=True),
        ImageItem(path="/x/c.nef", file_name="c.nef", burst_id="b1", burst_size=3, burst_rank=3, burst_best=False),
        ImageItem(path="/x/d.nef", file_name="d.nef", burst_id="b2", burst_size=1, burst_rank=1, burst_best=True),
    ]


def test_the_gallery_is_unchanged_by_default(app, tmp_path) -> None:
    """Collapse Bursts starts off - every image shows individually, exactly
    as before Burst Analysis existed."""
    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._apply_filter()
        assert len(window._gallery_model.items()) == 4
        assert window._gallery_view._delegate._show_burst_badge is False
    finally:
        window.close()
        service.close()


def test_collapse_bursts_shows_only_the_best_of_each_burst(app, tmp_path) -> None:
    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._on_toggle_collapse_bursts(True)

        visible = {item.path for item in window._gallery_model.items()}
        assert visible == {"/x/b.nef", "/x/d.nef"}
        assert window._gallery_view._delegate._show_burst_badge is True

        window._on_toggle_collapse_bursts(False)
        assert len(window._gallery_model.items()) == 4
        assert window._gallery_view._delegate._show_burst_badge is False
    finally:
        window.close()
        service.close()


def test_opening_a_card_normally_scopes_the_loupe_to_the_visible_gallery(app, tmp_path, monkeypatch) -> None:
    from picklikeme.desktop import main_window as main_window_module

    captured = {}

    class _FakeLoupeDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def exec(self):
            return None

    monkeypatch.setattr(main_window_module, "LoupeDialog", _FakeLoupeDialog)

    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._apply_filter()  # collapse is off

        window._open_loupe_for_item(window._gallery_model.item_at(2))  # c.nef

        assert captured["image_paths"] == ["/x/a.nef", "/x/b.nef", "/x/c.nef", "/x/d.nef"]
        assert captured["start_index"] == 2
    finally:
        window.close()
        service.close()


def test_opening_a_collapsed_card_scopes_the_loupe_to_its_burst_in_rank_order(app, tmp_path, monkeypatch) -> None:
    """The reported requirement: selecting a burst opens the Loupe scoped to
    that burst's own members, ordered by burst_rank - not the whole
    collapsed gallery, and not left/right-file order."""
    from picklikeme.desktop import main_window as main_window_module

    captured = {}

    class _FakeLoupeDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def exec(self):
            return None

    monkeypatch.setattr(main_window_module, "LoupeDialog", _FakeLoupeDialog)

    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._on_toggle_collapse_bursts(True)

        # Only b.nef (burst_best of "b1") is a visible row now.
        (visible_b1,) = [i for i in window._gallery_model.items() if i.burst_id == "b1"]
        window._open_loupe_for_item(visible_b1)

        # Ordered by burst_rank (b=1, a=2, c=3), not by burst_size membership
        # order or the original gallery order.
        assert captured["image_paths"] == ["/x/b.nef", "/x/a.nef", "/x/c.nef"]
        assert captured["start_index"] == 0
    finally:
        window.close()
        service.close()


def test_opening_a_collapsed_singleton_burst_shows_only_itself(app, tmp_path, monkeypatch) -> None:
    from picklikeme.desktop import main_window as main_window_module

    captured = {}

    class _FakeLoupeDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def exec(self):
            return None

    monkeypatch.setattr(main_window_module, "LoupeDialog", _FakeLoupeDialog)

    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._on_toggle_collapse_bursts(True)

        (visible_b2,) = [i for i in window._gallery_model.items() if i.burst_id == "b2"]
        window._open_loupe_for_item(visible_b2)

        assert captured["image_paths"] == ["/x/d.nef"]
    finally:
        window.close()
        service.close()


def test_changing_color_source_re_ranks_bursts_by_that_strategy(app, tmp_path, monkeypatch) -> None:
    """The reported requirement: burst_rank/burst_best follow the selected
    ranking strategy - here, the same Color Source selector Part 2 added."""
    from picklikeme.desktop.main_window import DEFAULT_STRATEGY_ID

    window, service = _window(tmp_path)
    try:
        calls = []
        monkeypatch.setattr(
            service, "set_burst_strategy",
            lambda strategy_id: (calls.append(strategy_id), service.load_session())[1],
        )

        window._color_combo.setCurrentIndex(0)  # "Review Status" (None)
        window._on_color_source_changed(0)
        assert calls[-1] == DEFAULT_STRATEGY_ID  # None falls back to the AI model

        classic_index = window._color_combo.findData("classic-vision")
        window._color_combo.setCurrentIndex(classic_index)
        window._on_color_source_changed(classic_index)
        assert calls[-1] == "classic-vision"
    finally:
        window.close()
        service.close()
