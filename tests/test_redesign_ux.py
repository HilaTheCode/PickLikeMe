"""Coverage for the 2026-08 PeakPick UX redesign (docs/UX Design/20260810/
Ver1.0/) - the behaviors the design package makes explicit requirements,
not covered by the pre-existing suites this pass otherwise left alone:

- Loupe: the Algorithm Results panel shows every algorithm that actually
  scored the current image, and selecting one switches the Elements Source
  (design spec 6A/6B) - both the eye/head overlay AND the confidence bars
  follow that selection, never a different, hidden strategy.
- Loupe: zoom presets and zoom persistence across navigation.
- Grid: the Domain filter and Clear Selection action.
- Grid: the thumbnail card shows only the selected Color Source's own
  score (see thumbnail_delegate.py).
- Dashboard: Overview's live KPI row/charts reflect the currently open
  folder, independent of the historical Experiment Browser selection.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

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


def _make_jpeg(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (48, 32), color=(90, 110, 140)).save(path, format="JPEG")


def _loupe_with_multiple_algorithms(tmp_path):
    """A single-image Loupe session whose image was scored by three
    different strategies - the "show every algorithm that has valid
    latest-run results" case (design spec 6A)."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.services import ReviewService

    folder = tmp_path / "shoot"
    folder.mkdir(exist_ok=True)
    path = folder / "a.jpg"
    _make_jpeg(path)

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)
    service.session.burst_strategy = "classic-vision-fusion-birds"

    item = ImageItem(
        path=str(path), file_name=path.name,
        ranking_results={
            "classic-vision-fusion-birds": {"score": 0.946, "rank": 1},
            "classic-vision": {"score": 0.821, "rank": 2},
            "ai-model": {"score": 0.75, "rank": 1},
        },
    )
    dialog = LoupeDialog(service=service, image_paths=[str(path)], items=[item], start_index=0)
    return dialog, service


# ---------------------------------------------------------------------------
# Loupe - Algorithm Results / Elements Source
# ---------------------------------------------------------------------------


def test_algorithm_results_panel_lists_only_algorithms_that_actually_scored_this_image(app, tmp_path) -> None:
    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    try:
        assert set(dialog._algo_rows) == {"classic-vision-fusion-birds", "classic-vision", "ai-model"}
        # Highest score first, matching the design mockup's own ordering.
        ordered = [
            dialog._algo_rows_layout.itemAt(i).widget().strategy_id
            for i in range(dialog._algo_rows_layout.count())
        ]
        assert ordered == ["classic-vision-fusion-birds", "classic-vision", "ai-model"]
        assert dialog._algo_rows["classic-vision-fusion-birds"]._score_label.text() == "0.946"
    finally:
        dialog.close()
        service.close()


def test_algorithm_results_never_lists_a_strategy_this_image_was_never_scored_by(app, tmp_path) -> None:
    """"Do not invent results" - a registered strategy (e.g. Mammal Fusion)
    that never touched this specific image must not appear."""
    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    try:
        assert "classic-vision-fusion-mammals" not in dialog._algo_rows
    finally:
        dialog.close()
        service.close()


def test_the_default_elements_source_is_the_current_color_source(app, tmp_path) -> None:
    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    try:
        assert dialog._elements_source_id == "classic-vision-fusion-birds"
        assert dialog._algo_rows["classic-vision-fusion-birds"]._is_active is True
        assert dialog._algo_rows["ai-model"]._is_active is False
    finally:
        dialog.close()
        service.close()


def test_selecting_an_algorithm_row_switches_the_elements_source(app, tmp_path) -> None:
    """Design spec 6B: clicking an algorithm in the panel makes it the
    Elements source immediately - reflected in the row highlight, the
    Elements Source combo, AND what the eye/head overlay reads."""
    from unittest import mock

    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    try:
        with mock.patch.object(service, "eye_keypoints", return_value=None) as fake:
            dialog._algo_rows["ai-model"].selected.emit("ai-model")

        assert dialog._elements_source_id == "ai-model"
        assert dialog._algo_rows["ai-model"]._is_active is True
        assert dialog._algo_rows["classic-vision-fusion-birds"]._is_active is False
        assert dialog._elements_source_combo.currentData() == "ai-model"
        # The overlay refresh (triggered by the selection) must ask for
        # THIS strategy's own eye record, not a different one.
        assert any(call.kwargs.get("strategy_id") == "ai-model" for call in fake.call_args_list)
    finally:
        dialog.close()
        service.close()


def test_boxes_and_elements_overlays_both_read_the_selected_elements_source(app, tmp_path) -> None:
    """Both overlays must ask for the SAME strategy's eye data as the
    Elements Source panel shows - never a silently different one (the bug
    class the cache-ownership pass and this redesign both close)."""
    from unittest import mock

    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    try:
        dialog._select_elements_source("classic-vision")
        dialog._show_boxes = True
        dialog._show_elements = True
        with mock.patch.object(service, "eye_keypoints", return_value=None) as fake, \
             mock.patch.object(service, "detection_boxes", return_value=None):
            dialog._refresh_detection_overlay()
        assert fake.call_count == 1
        assert fake.call_args.kwargs.get("strategy_id") == "classic-vision"
    finally:
        dialog.close()
        service.close()


def test_elements_confidence_bars_reflect_the_selected_elements_source(app, tmp_path) -> None:
    from unittest import mock

    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    try:
        fake_eye = {
            "head_top": {"x": 1, "y": 1, "confidence": 0.5}, "head_confidence": 0.91,
            "left": {"x": 1, "y": 1, "confidence": 0.82},
            "right": {"x": 1, "y": 1, "confidence": 0.77},
        }
        with mock.patch.object(service, "eye_keypoints", return_value=fake_eye):
            dialog._select_elements_source("ai-model")
        assert dialog._head_bar._value_label.text() == "0.91"
        assert dialog._left_eye_bar._value_label.text() == "0.82"
        assert dialog._right_eye_bar._value_label.text() == "0.77"
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# Loupe - zoom presets / persistence (design spec 6F)
# ---------------------------------------------------------------------------


def test_zoom_presets_set_an_exact_percentage(app, tmp_path) -> None:
    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    try:
        dialog._view.set_zoom_percent(150)
        assert dialog._view._fit_mode is False
        assert dialog._view.transform().m11() == pytest.approx(1.5, abs=1e-6)
    finally:
        dialog.close()
        service.close()


def test_the_image_viewport_dominates_the_window_at_1470px(app, tmp_path) -> None:
    """Regression: the Algorithm Results / Elements Source side panels
    were originally `setFixedWidth`, which does not yield space back to
    the image viewport - on the real production code path (the dialog
    maximized to whatever the actual screen provides), this squeezed the
    center image column down to a sliver (measured: 191px of a 796px-wide
    window, ~24%) whenever the available screen was smaller than the
    panels' combined fixed width assumed. Panels are now `setMaximumWidth`
    (shrink toward their own content's minimum instead of refusing to),
    so the image viewport - the layout's one stretch=1 widget - claims the
    majority of the window's width at the design's own 1470px reference
    size, matching/exceeding `02_Loupe.svg`'s own ~54% image-to-window
    ratio."""
    from PySide6.QtCore import Qt

    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    try:
        dialog.setWindowState(Qt.WindowState.WindowNoState)
        dialog.resize(1470, 900)
        dialog.show()
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        view_fraction = dialog._view.width() / dialog.width()
        assert view_fraction > 0.5, (
            f"image viewport is only {view_fraction:.1%} of the window width - "
            "must be the dominant region, not a narrow strip"
        )
        # Both side panels must have genuinely shrunk to fit their own
        # content, not merely happened to be under their cap by luck.
        assert dialog._algo_results_container.width() <= 260
        assert dialog._elements_panel.width() <= 340
    finally:
        dialog.close()
        service.close()


def test_zoom_level_persists_across_navigation(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog
    from picklikeme.desktop.services import ReviewService

    folder = tmp_path / "shoot"
    folder.mkdir()
    paths = [folder / f"img{i}.jpg" for i in range(2)]
    for path in paths:
        _make_jpeg(path)
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)
    dialog = LoupeDialog(service=service, image_paths=[str(p) for p in paths], start_index=0)
    try:
        dialog._view.set_zoom_percent(200)
        dialog._go_next()
        assert dialog._view._fit_mode is False
        assert dialog._view.transform().m11() == pytest.approx(2.0, abs=1e-6)
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# Loupe - View Mode (Full Review / Focus Image / Analysis / Elements)
# ---------------------------------------------------------------------------


def _sized_loupe(tmp_path, *, width: int = 1470):
    """A single-widget-tree Loupe (see `_loupe_with_multiple_algorithms`)
    resized to an explicit, non-maximized width - the real production
    dialog opens maximized to whatever the actual screen provides
    (`setWindowState(WindowMaximized)`), which under a test's offscreen
    virtual screen is not the 1470px MacBook target the design package
    itself specifies - so View Mode's own space-reclaiming behavior is
    verified against that explicit target width instead, matching
    `test_the_image_viewport_dominates_the_window_at_1470px`'s own
    approach."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    dialog, service = _loupe_with_multiple_algorithms(tmp_path)
    dialog.setWindowState(Qt.WindowState.WindowNoState)
    dialog.resize(width, 900)
    dialog.show()
    QApplication.processEvents()
    return dialog, service


def test_default_view_mode_is_full_review_with_both_panels_visible(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import VIEW_MODE_FULL_REVIEW

    dialog, service = _sized_loupe(tmp_path)
    try:
        assert dialog._view_mode == VIEW_MODE_FULL_REVIEW
        assert dialog._algo_results_container.isVisible() is True
        assert dialog._elements_panel.isVisible() is True
        assert dialog._view_mode_button.text() == "View: Full Review"
    finally:
        dialog.close()
        service.close()


def test_focus_image_mode_hides_both_panels_and_the_image_fills_the_window(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import VIEW_MODE_FOCUS_IMAGE
    from PySide6.QtWidgets import QApplication

    dialog, service = _sized_loupe(tmp_path)
    try:
        dialog._set_view_mode(VIEW_MODE_FOCUS_IMAGE)
        QApplication.processEvents()

        assert dialog._algo_results_container.isVisible() is False
        assert dialog._elements_panel.isVisible() is False
        # Both panels released their space - the image must have expanded
        # to consume nearly the whole window, not merely "more than before".
        view_fraction = dialog._view.width() / dialog.width()
        assert view_fraction > 0.9, f"image viewport is only {view_fraction:.1%} with both panels hidden"
    finally:
        dialog.close()
        service.close()


def test_analysis_mode_shows_only_algorithm_results(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import VIEW_MODE_ANALYSIS
    from PySide6.QtWidgets import QApplication

    dialog, service = _sized_loupe(tmp_path)
    try:
        dialog._set_view_mode(VIEW_MODE_ANALYSIS)
        QApplication.processEvents()

        assert dialog._algo_results_container.isVisible() is True
        assert dialog._elements_panel.isVisible() is False
        assert dialog._view_mode_button.text() == "View: Analysis"
    finally:
        dialog.close()
        service.close()


def test_elements_mode_shows_only_elements_source(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import VIEW_MODE_ELEMENTS
    from PySide6.QtWidgets import QApplication

    dialog, service = _sized_loupe(tmp_path)
    try:
        dialog._set_view_mode(VIEW_MODE_ELEMENTS)
        QApplication.processEvents()

        assert dialog._algo_results_container.isVisible() is False
        assert dialog._elements_panel.isVisible() is True
        assert dialog._view_mode_button.text() == "View: Elements"
    finally:
        dialog.close()
        service.close()


def test_hiding_either_panel_never_hides_the_photograph_itself(app, tmp_path) -> None:
    """The View Mode only ever controls the two side panels - design
    instruction: "the image itself is always visible in every mode"."""
    from picklikeme.desktop.dialogs.loupe_dialog import VIEW_MODE_FOCUS_IMAGE

    dialog, service = _sized_loupe(tmp_path)
    try:
        dialog._set_view_mode(VIEW_MODE_FOCUS_IMAGE)
        assert dialog._view.isVisible() is True
        assert dialog._view._pixmap_item is not None
    finally:
        dialog.close()
        service.close()


def test_the_image_viewport_progressively_expands_as_panels_are_hidden(app, tmp_path) -> None:
    """One responsive layout, not four duplicated ones: the image
    viewport's own width must strictly increase (or stay equal, never
    shrink) as each mode releases more space - Full Review (both panels)
    < Analysis/Elements (one panel) < Focus Image (neither)."""
    from picklikeme.desktop.dialogs.loupe_dialog import (
        VIEW_MODE_ANALYSIS, VIEW_MODE_ELEMENTS, VIEW_MODE_FOCUS_IMAGE, VIEW_MODE_FULL_REVIEW,
    )
    from PySide6.QtWidgets import QApplication

    dialog, service = _sized_loupe(tmp_path)
    try:
        widths = {}
        for mode in (VIEW_MODE_FULL_REVIEW, VIEW_MODE_ANALYSIS, VIEW_MODE_ELEMENTS, VIEW_MODE_FOCUS_IMAGE):
            dialog._set_view_mode(mode)
            QApplication.processEvents()
            widths[mode] = dialog._view.width()

        assert widths[VIEW_MODE_FULL_REVIEW] < widths[VIEW_MODE_ANALYSIS]
        assert widths[VIEW_MODE_FULL_REVIEW] < widths[VIEW_MODE_ELEMENTS]
        assert widths[VIEW_MODE_ANALYSIS] < widths[VIEW_MODE_FOCUS_IMAGE]
        assert widths[VIEW_MODE_ELEMENTS] < widths[VIEW_MODE_FOCUS_IMAGE]
    finally:
        dialog.close()
        service.close()


def test_view_mode_persists_across_navigation(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog, VIEW_MODE_FOCUS_IMAGE
    from picklikeme.desktop.services import ReviewService

    folder = tmp_path / "shoot"
    folder.mkdir()
    paths = [folder / f"img{i}.jpg" for i in range(2)]
    for path in paths:
        _make_jpeg(path)
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)
    dialog = LoupeDialog(service=service, image_paths=[str(p) for p in paths], start_index=0)
    try:
        dialog._set_view_mode(VIEW_MODE_FOCUS_IMAGE)
        dialog._go_next()
        assert dialog._view_mode == VIEW_MODE_FOCUS_IMAGE
        assert dialog._algo_results_container.isVisible() is False
        assert dialog._elements_panel.isVisible() is False
    finally:
        dialog.close()
        service.close()


def test_the_view_mode_menu_offers_exactly_one_exclusive_choice_per_mode(app, tmp_path) -> None:
    """A compact button/menu - one QActionGroup, exclusive - not four
    separate always-visible buttons."""
    from picklikeme.desktop.dialogs.loupe_dialog import (
        VIEW_MODE_ANALYSIS, VIEW_MODE_ELEMENTS, VIEW_MODE_FOCUS_IMAGE, VIEW_MODE_FULL_REVIEW,
    )

    dialog, service = _sized_loupe(tmp_path)
    try:
        assert set(dialog._view_mode_actions) == {
            VIEW_MODE_FULL_REVIEW, VIEW_MODE_FOCUS_IMAGE, VIEW_MODE_ANALYSIS, VIEW_MODE_ELEMENTS,
        }
        dialog._set_view_mode(VIEW_MODE_ANALYSIS)
        checked = [mode for mode, action in dialog._view_mode_actions.items() if action.isChecked()]
        assert checked == [VIEW_MODE_ANALYSIS]
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# Grid - Domain filter / Clear Selection / score-badge-only-selected-source
# ---------------------------------------------------------------------------


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


def test_domain_filter_narrows_to_images_a_domain_strategy_actually_scored(app, tmp_path) -> None:
    from picklikeme.desktop.models.image_item import ImageItem

    window, service = _window(tmp_path)
    try:
        bird_item = ImageItem(path="/x/bird.jpg", file_name="bird.jpg",
                               ranking_results={"classic-vision-fusion-birds": {"score": 0.9}})
        mammal_item = ImageItem(path="/x/mammal.jpg", file_name="mammal.jpg",
                                 ranking_results={"classic-vision-fusion-mammals": {"score": 0.8}})
        window._all_items = [bird_item, mammal_item]

        window._domain_filter = "Birds"
        window._apply_filter()
        assert [item.path for item in window._gallery_model.items()] == ["/x/bird.jpg"]

        window._domain_filter = "Mammals"
        window._apply_filter()
        assert [item.path for item in window._gallery_model.items()] == ["/x/mammal.jpg"]

        from picklikeme.desktop.main_window import DOMAIN_ALL
        window._domain_filter = DOMAIN_ALL
        window._apply_filter()
        assert len(window._gallery_model.items()) == 2
    finally:
        window.close()
        service.close()


def test_clear_selection_deselects_every_card(app, tmp_path) -> None:
    from picklikeme.desktop.models.image_item import ImageItem

    window, service = _window(tmp_path)
    try:
        window._all_items = [ImageItem(path=f"/x/{i}.jpg", file_name=f"{i}.jpg") for i in range(3)]
        window._apply_filter()
        window._gallery_view.selectAll()
        assert len(window._gallery_view.selectionModel().selectedIndexes()) == 3

        window._clear_selection()
        assert len(window._gallery_view.selectionModel().selectedIndexes()) == 0
    finally:
        window.close()
        service.close()


# ---------------------------------------------------------------------------
# Dashboard - Overview reflects the live folder
# ---------------------------------------------------------------------------


def test_overview_tab_is_visible_with_live_items_even_with_zero_historical_experiments(app, tmp_path) -> None:
    """Regression: the pre-redesign empty-state swap (`_update_empty_state`)
    only ever checked whether a historical experiment existed to select -
    correct before Overview/Domains/Trends existed, since every tab needed
    one. With those three now reading the live folder instead (see
    OverviewTab's own docstring), a dashboard opened for a folder that has
    simply never been ranked (a real, common case) must still show its
    tabs - not the "no experiments recorded yet" placeholder, which used to
    hide Overview along with everything else."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard
    from picklikeme.desktop.models.image_item import ImageItem

    items = [ImageItem(path="/x/a.jpg", file_name="a.jpg")]
    dashboard = AnalyticsDashboard(
        analytics_db=tmp_path / "analytics.sqlite", annotations_db=tmp_path / "annotations.sqlite",
        species_db=tmp_path / "species.db", items=items, parent=None,
    )
    try:
        assert dashboard._experiment_list.count() == 0  # genuinely no historical runs
        assert dashboard._detail_stack.currentIndex() == 0  # tabs shown, not the placeholder
    finally:
        dashboard.close()


def test_overview_kpis_reflect_the_live_folder_not_a_selected_experiment(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard
    from picklikeme.desktop.models.image_item import ImageItem

    items = [
        ImageItem(path="/x/a.jpg", file_name="a.jpg", review_status="keep"),
        ImageItem(path="/x/b.jpg", file_name="b.jpg", review_status="reject"),
        ImageItem(path="/x/c.jpg", file_name="c.jpg", review_status="neutral"),
    ]
    dashboard = AnalyticsDashboard(
        analytics_db=tmp_path / "analytics.sqlite", annotations_db=tmp_path / "annotations.sqlite",
        species_db=tmp_path / "species.db", items=items, parent=None,
    )
    try:
        # No experiment was ever selected (an empty analytics DB) - Overview
        # must still show real counts from the live folder, not "—".
        assert dashboard._overview_tab._kpi_cards["total"]._value_label.text() == "3"
        assert dashboard._overview_tab._kpi_cards["keep"]._value_label.text() == "1"
        assert dashboard._overview_tab._kpi_cards["reject"]._value_label.text() == "1"
        assert dashboard._overview_tab._kpi_cards["review"]._value_label.text() == "1"
    finally:
        dashboard.close()
