"""Regression coverage for the Mac top-toolbar layout bug: a ranking
strategy with a long display_name (e.g. "Classic Vision Ranking
(EyePose-v0, recommended)") made the Sort/Color combo boxes grow to fit
that single longest item - Qt's QComboBox default sizing
(AdjustToContentsOnFirstShow) sizes the closed box to its widest entry,
not to what's actually selected. On a MacBook-width window that pushed the
Color combo and Collapse Bursts action off the visible toolbar.

Mirrors test_main_window_advanced_filters.py's own MainWindow-construction
pattern.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication, QComboBox, QToolBar
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")

# The primary row (Open/Filter/review-decisions/Sort/Color/Collapse Bursts)
# must fit within this on its own - verified against a real MacBook's actual
# default logical screen width (1470px) on the real Cocoa platform, where
# row 1's sizeHint measured ~1400px. This suite runs under the offscreen QPA
# platform, whose fallback font metrics render text a bit wider (~1470px
# for the same row) - the threshold leaves headroom for that gap while
# still catching a real regression (the pre-fix single-row toolbar's
# sizeHint was ~2700-3150px).
ROW1_MAX_COMFORTABLE_WIDTH = 1600


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


def test_toolbar_combos_do_not_grow_to_fit_their_longest_item(app, tmp_path) -> None:
    """Filter/Sort/Color/AI-cutoff-preset all carry at least one very long
    item label (a Conflict filter, a long strategy display name, etc.) -
    each combo must cap its own width instead of ballooning to fit it."""
    window, service = _window(tmp_path)
    try:
        for combo in (
            window._filter_combo, window._sort_combo,
            window._color_combo, window._cutoff_combo,
        ):
            assert combo.maximumWidth() < 300, (
                f"{combo.objectName() or combo} has no effective width cap "
                f"(maximumWidth={combo.maximumWidth()}) and can grow to fit "
                "its longest item"
            )
            assert combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    finally:
        window.close()
        service.close()


def test_toolbar_selected_value_tooltip_shows_full_text(app, tmp_path) -> None:
    """A combo box narrow enough to elide its selected value must still
    expose the full text somewhere - via a tooltip."""
    window, service = _window(tmp_path)
    try:
        window._sort_combo.setCurrentIndex(0)
        assert window._sort_combo.toolTip() == window._sort_combo.currentText()
    finally:
        window.close()
        service.close()


def test_toolbar_row_1_holds_filter_sort_color_and_collapse_bursts(app, tmp_path) -> None:
    """The two controls the bug report calls out by name - Color and
    Collapse Bursts (burst grouping) - live on the primary toolbar row
    alongside Filter/Sort, and that row must be narrow enough to fit a
    MacBook window without Qt's overflow ">>" chevron swallowing them.

    Uses sizeHint(), not isVisible()/geometry() - those depend on the real
    screen's available width and on this machine's persisted QSettings
    toolbar state (see TOOLBAR_STATE_VERSION), neither of which this suite
    controls. sizeHint is deterministic and platform-independent.
    """
    window, service = _window(tmp_path)
    try:
        toolbar = window.findChild(QToolBar, "main_toolbar")
        assert toolbar is not None

        for combo in (window._filter_combo, window._sort_combo, window._color_combo):
            assert toolbar.isAncestorOf(combo)

        collapse_widget = toolbar.widgetForAction(window._collapse_bursts_action)
        assert collapse_widget is not None
        assert toolbar.isAncestorOf(collapse_widget)

        assert toolbar.sizeHint().width() < ROW1_MAX_COMFORTABLE_WIDTH, (
            f"Primary toolbar row needs {toolbar.sizeHint().width()}px, "
            f"more than a MacBook window comfortably offers "
            f"({ROW1_MAX_COMFORTABLE_WIDTH}px) - Color/Collapse Bursts will "
            "fall into Qt's overflow chevron"
        )
    finally:
        window.close()
        service.close()


def test_toolbar_splits_into_two_rows_with_a_real_break(app, tmp_path) -> None:
    """Row 2 (AI ranking/cutoff, Organize/Import/Species/Crop, Settings)
    must start on a genuinely new row - not just visually appended after
    row 1 - so it gets the full window width to lay out in, independent of
    how much room row 1 used."""
    window, service = _window(tmp_path)
    try:
        toolbar2 = window.findChild(QToolBar, "main_toolbar_2")
        assert toolbar2 is not None
        assert window.toolBarBreak(toolbar2) is True

        settings_widget = toolbar2.widgetForAction(window._settings_action)
        assert settings_widget is not None
    finally:
        window.close()
        service.close()


def test_toolbar_state_version_is_bumped_so_stale_settings_do_not_win(app, tmp_path) -> None:
    """A pre-fix install may already have a saved single-row toolbar layout
    in QSettings (window/state). QMainWindow.restoreState() replays that
    blob verbatim unless its version argument mismatches - so
    save/restoreState must be called with the same explicit version,
    otherwise the newly-built two-row layout would silently lose to
    whatever was on disk before this fix. See _restore_state/_save_state.
    """
    from picklikeme.desktop import main_window as main_window_module

    assert main_window_module.TOOLBAR_STATE_VERSION >= 1

    window, service = _window(tmp_path)
    try:
        import inspect

        assert "TOOLBAR_STATE_VERSION" in inspect.getsource(window._restore_state)
        assert "TOOLBAR_STATE_VERSION" in inspect.getsource(window._save_state)
    finally:
        window.close()
        service.close()
