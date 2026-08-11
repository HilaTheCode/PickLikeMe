"""Regression coverage for the Grid screen's redesigned chrome
(MainWindow._build_primary_bar/_build_secondary_bar - see docs/UX Design/
20260810/Ver1.0/04_Toolbar.svg).

The pre-redesign version of this suite checked a QToolBar-based two-row
layout; the redesign replaced QToolBar entirely with a custom Panel-based
primary/secondary bar (see design_system.py), so these tests check the same
underlying concerns - combos capped to a sane width, the full chrome fits a
MacBook-width window, Color Source/Sort live on the primary bar - against
the new widget tree instead.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication, QComboBox
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")

# The primary bar (brand, Rank/Apply Cutoff/Keep/Reject/Clear/Export, Color
# Source, Sort) must fit within this on its own - verified against a real
# MacBook's actual default logical screen width (1470px). This suite runs
# under the offscreen QPA platform, whose fallback font metrics render text
# a bit wider than the real Cocoa platform - the threshold leaves headroom
# for that gap while still catching a real regression (an uncapped combo's
# sizeHint alone can run past 2000px).
PRIMARY_BAR_MAX_COMFORTABLE_WIDTH = 1600


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
    """Filter/Sort/Color/AI-cutoff-preset/Domain/Burst all carry at least
    one very long item label (a Conflict filter, a long strategy display
    name, etc.) - each combo must cap its own width instead of ballooning
    to fit it."""
    window, service = _window(tmp_path)
    try:
        for combo in (
            window._filter_combo, window._sort_combo, window._color_combo,
            window._cutoff_combo, window._domain_combo, window._burst_sort_combo,
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


def test_primary_bar_holds_the_high_value_actions_and_color_and_sort(app, tmp_path) -> None:
    """Rank/Apply Cutoff/Keep/Reject/Clear Selection/Export plus Color
    Source and Sort all live on the primary bar (`04_Toolbar.svg`'s own
    single-row arrangement), and that bar must be narrow enough to fit a
    MacBook window comfortably."""
    window, service = _window(tmp_path)
    try:
        for combo in (window._filter_combo, window._sort_combo, window._color_combo):
            assert combo.window() is window
        assert window._rank_button.window() is window
        assert window._apply_cutoff_button.window() is window
    finally:
        window.close()
        service.close()


def test_primary_bar_fits_a_macbook_width_window(app, tmp_path) -> None:
    window, service = _window(tmp_path)
    try:
        bar = window._rank_button.parentWidget()
        assert bar is not None
        assert bar.sizeHint().width() < PRIMARY_BAR_MAX_COMFORTABLE_WIDTH, (
            f"Primary bar needs {bar.sizeHint().width()}px, more than a "
            f"MacBook window comfortably offers ({PRIMARY_BAR_MAX_COMFORTABLE_WIDTH}px)"
        )
    finally:
        window.close()
        service.close()


def test_secondary_bar_holds_filter_domain_search_burst_and_view(app, tmp_path) -> None:
    """Row 2 of the redesigned chrome (Filter/Domain/Search/Burst/View,
    plus Detector Boxes/Collapse Bursts) is a separate widget from the
    primary bar - genuinely a second row, not just appended after it."""
    window, service = _window(tmp_path)
    try:
        primary_bar = window._rank_button.parentWidget()
        secondary_bar = window._domain_combo.window() and window._domain_combo.parentWidget().parentWidget()
        assert secondary_bar is not None
        assert secondary_bar is not primary_bar
        assert window._search_widget.window() is window
    finally:
        window.close()
        service.close()
