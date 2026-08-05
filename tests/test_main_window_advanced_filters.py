"""Product Direction, Phase A/B: Advanced Filters lives in the Review
Window, narrows the same gallery model the simple Filter combo and Collapse
Bursts already narrow, and Loupe navigation (_open_loupe_for_item, which
reads self._gallery_model.items()) inherits that narrowing automatically -
see desktop/filtering.py's own docstring on why no separate Loupe-side
wiring is needed for this.

Mirrors test_burst_ui.py's own MainWindow-construction pattern.
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


def _items():
    from picklikeme.desktop.models.image_item import ImageItem

    return [
        ImageItem(
            path="/shoot/a.nef", file_name="a.nef", review_status="keep",
            algorithm_suggestion="keep", ranking_results={"ai-model": {"score": 0.9, "rank": 1}},
        ),
        ImageItem(
            path="/shoot/b.nef", file_name="b.nef", review_status="reject",
            algorithm_suggestion="keep", ranking_results={"ai-model": {"score": 0.4, "rank": 2}},
        ),
        ImageItem(
            path="/shoot/_Selected/c.nef", file_name="c.nef", review_status="neutral",
            algorithm_suggestion="reject", ranking_results={"ai-model": {"score": 0.1, "rank": 3}},
        ),
    ]


def test_advanced_filters_panel_is_part_of_the_review_window(app, tmp_path) -> None:
    from picklikeme.desktop.widgets.advanced_filters_panel import AdvancedFiltersPanel

    window, service = _window(tmp_path)
    try:
        assert isinstance(window._advanced_filters_panel, AdvancedFiltersPanel)
    finally:
        window.close()
        service.close()


def test_advanced_filters_narrows_the_main_gallery_live(app, tmp_path) -> None:
    window, service = _window(tmp_path)
    try:
        window._all_items = _items()
        window._apply_filter()
        assert len(window._gallery_model.items()) == 3

        # No Apply button - setting the control alone must update the grid.
        window._advanced_filters_panel._user_decision_combo.setCurrentText("Reject")

        assert [i.path for i in window._gallery_model.items()] == ["/shoot/b.nef"]
    finally:
        window.close()
        service.close()


def test_advanced_filters_combines_with_the_simple_filter_combo(app, tmp_path) -> None:
    """Species = Kingfisher AND User Reject AND Score > 0.80 AND Eye
    Confidence < 0.90 - style combination, using the simple Filter combo
    (algorithm_keep) alongside an Advanced Filters range control."""
    window, service = _window(tmp_path)
    try:
        window._all_items = _items()
        window._current_filter = "algorithm_keep"  # a.nef, b.nef
        window._apply_filter()
        assert {i.path for i in window._gallery_model.items()} == {"/shoot/a.nef", "/shoot/b.nef"}

        window._advanced_filters_panel._range_checks["score"].setChecked(True)
        window._advanced_filters_panel._range_mins["score"].setValue(0.5)

        # Only a.nef (score 0.9) clears both the simple filter AND the range.
        assert [i.path for i in window._gallery_model.items()] == ["/shoot/a.nef"]
    finally:
        window.close()
        service.close()


def test_clear_all_filters_restores_the_full_gallery(app, tmp_path) -> None:
    window, service = _window(tmp_path)
    try:
        window._all_items = _items()
        window._apply_filter()
        window._advanced_filters_panel._user_decision_combo.setCurrentText("Reject")
        assert len(window._gallery_model.items()) == 1

        window._advanced_filters_panel.reset()

        assert len(window._gallery_model.items()) == 3
    finally:
        window.close()
        service.close()


def test_refresh_from_state_populates_folder_options_from_the_open_folder(app, tmp_path) -> None:
    window, service = _window(tmp_path)
    try:
        window._all_items = _items()
        window._refresh_advanced_filter_options()

        from pathlib import Path

        combo = window._advanced_filters_panel._folder_combo
        options = {combo.itemText(i) for i in range(combo.count())}
        assert str(Path("/shoot")) in options
        assert str(Path("/shoot/_Selected")) in options
    finally:
        window.close()
        service.close()


def test_refresh_from_state_populates_species_options_from_the_species_cache(app, tmp_path, monkeypatch) -> None:
    from picklikeme.species.classifier import SpeciesPrediction

    window, service = _window(tmp_path)
    try:
        window._all_items = _items()
        # Pre-seed rather than exercising the real on-disk species cache -
        # matches _refresh_species_cache's own "already resolved, never
        # re-queried" incremental contract.
        window._species_by_path = {
            "/shoot/a.nef": SpeciesPrediction(species="Kingfisher", confidence=0.95, classifier_id="bioclip2:x"),
            "/shoot/b.nef": None,
            "/shoot/_Selected/c.nef": None,
        }
        window._refresh_advanced_filter_options()

        combo = window._advanced_filters_panel._species_combo
        options = {combo.itemText(i) for i in range(combo.count())}
        assert "Kingfisher" in options
    finally:
        window.close()
        service.close()


def test_advanced_filters_species_and_score_range_use_seeded_cache(app, tmp_path) -> None:
    from picklikeme.species.classifier import SpeciesPrediction

    window, service = _window(tmp_path)
    try:
        window._all_items = _items()
        window._species_by_path = {
            "/shoot/a.nef": SpeciesPrediction(species="Kingfisher", confidence=0.95, classifier_id="x"),
            "/shoot/b.nef": SpeciesPrediction(species="Osprey", confidence=0.80, classifier_id="x"),
            "/shoot/_Selected/c.nef": None,
        }
        window._refresh_advanced_filter_options()
        window._apply_filter()

        window._advanced_filters_panel._species_combo.setCurrentText("Kingfisher")

        assert [i.path for i in window._gallery_model.items()] == ["/shoot/a.nef"]
    finally:
        window.close()
        service.close()


def test_loupe_navigation_inherits_advanced_filters_narrowing(app, tmp_path, monkeypatch) -> None:
    """Phase B: the same gallery model Advanced Filters narrows is what
    _open_loupe_for_item reads its Prev/Next list from - no separate
    Loupe-side filter wiring required."""
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
        window._all_items = _items()
        window._apply_filter()

        window._advanced_filters_panel._user_decision_combo.setCurrentText("Reject")
        assert len(window._gallery_model.items()) == 1

        window._open_loupe_for_item(window._gallery_model.item_at(0))

        assert captured["image_paths"] == ["/shoot/b.nef"]
        assert captured["start_index"] == 0
    finally:
        window.close()
        service.close()


def test_opening_a_new_folder_resets_the_species_cache(app, tmp_path) -> None:
    from picklikeme.species.classifier import SpeciesPrediction

    window, service = _window(tmp_path)
    try:
        window._species_by_path = {
            "/old/folder/a.nef": SpeciesPrediction(species="Kingfisher", confidence=0.9, classifier_id="x"),
        }

        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()
        window.open_folder(str(empty_folder))

        assert "/old/folder/a.nef" not in window._species_by_path
    finally:
        window.close()
        service.close()
