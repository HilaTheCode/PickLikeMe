"""Regression coverage for two Burst Loupe fixes:

1. Window sizing - the Loupe window was reported wider than the available
   screen on a MacBook. Root cause: the burst info row's "Color Source:
   <strategy display name>" QLabel had no width cap, so a long strategy
   name (ranking.classic's "Classic Vision Ranking (EyePose-v0,
   recommended)" is 46+ chars) could push the bottom bar's own minimum
   size hint wider than the screen - Qt then compresses the whole layout
   below its minimum, clipping button text in the row above it too (see
   loupe_dialog.py's `_set_elided_text`/`_BURST_INFO_LABEL_MAX_WIDTH`).
   LoupeDialog.__init__ also now explicitly sets its geometry to
   `self.screen().availableGeometry()` rather than relying solely on
   `setWindowState(WindowMaximized)` to land on the right bounds.

2. Left/Right arrow keyboard navigation - LoupeDialog.keyPressEvent already
   handled Key_Left/Key_Right (routing to the same _go_prev/_go_next the
   Prev/Next buttons use), but _ZoomView (the dialog's main/central widget,
   a QGraphicsView) accepted keyboard focus by default and silently
   consumed arrow keys for its own viewport scrolling before they ever
   reached the dialog - verified empirically: sending Key_Right to a
   focused _ZoomView left LoupeDialog.index unchanged. Fixed by giving
   _ZoomView Qt.FocusPolicy.NoFocus.

Mirrors test_desktop_workflow.py's own LoupeDialog-construction pattern.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
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

    Image.new("RGB", (32, 24), color=(80, 120, 160)).save(path, format="JPEG")


def _burst_dialog(tmp_path, *, burst_strategy: str | None = "ai-model", n: int = 3):
    """A burst-scoped LoupeDialog with `n` members, using a real
    ReviewService/ReviewSession (burst_strategy is set directly on the
    session, matching how MainWindow's Color Source selector does it via
    ReviewService.set_burst_strategy)."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.services import ReviewService

    folder = tmp_path / "shoot"
    folder.mkdir(exist_ok=True)
    paths = [folder / f"img{i}.jpg" for i in range(n)]
    for path in paths:
        _make_jpeg(path)

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)
    if burst_strategy is not None:
        service.session.burst_strategy = burst_strategy

    items = [
        ImageItem(
            path=str(paths[i]), file_name=paths[i].name, burst_id="burst-0001",
            burst_size=n, burst_rank=i + 1, burst_best=(i == 0),
            ranking_results={burst_strategy: {"score": 0.9 - i * 0.1, "rank": i + 1}} if burst_strategy else {},
        )
        for i in range(n)
    ]
    dialog = LoupeDialog(
        service=service, image_paths=[i.path for i in items], items=items, start_index=0, burst_scoped=True,
    )
    return dialog, service


# ---------------------------------------------------------------------------
# 1. Window sizing / layout
# ---------------------------------------------------------------------------


def test_burst_color_source_label_is_capped_elided_and_keeps_full_text_as_tooltip(app, tmp_path) -> None:
    very_long_strategy_id = "x" * 120  # unregistered -> _strategy_label falls back to the raw id verbatim
    dialog, service = _burst_dialog(tmp_path, burst_strategy=very_long_strategy_id)
    try:
        label = dialog._burst_color_source_label
        full_text = f"Color Source: {very_long_strategy_id}"

        assert label.maximumWidth() <= 260
        assert label.text() != full_text  # elided, not shown in full
        assert len(label.text()) < len(full_text)
        assert label.toolTip() == full_text
    finally:
        dialog.close()
        service.close()


def test_score_label_is_capped_and_elided_when_multiple_strategies_have_scored(app, tmp_path) -> None:
    """The score bar (_scores_text) concatenates EVERY analysis module's
    score onto one unbounded QLabel - realistic and unremarkable-looking
    with one module scored, but measured to push the bottom bar's own
    minimum width past a real 1470px-wide MacBook screen once a second
    module (here: AI Model + Classic Vision (EyePose)) has also scored the
    same image, visibly compressing/clipping the Keep/Reject/Neutral
    buttons sharing that row - this is not burst-specific; it reproduces on
    a plain, non-burst Loupe session too."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.services import ReviewService

    folder = tmp_path / "shoot"
    folder.mkdir()
    path_a = folder / "a.jpg"
    _make_jpeg(path_a)
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    item = ImageItem(
        path=str(path_a), file_name="a.jpg",
        ranking_results={
            "ai-model": {"score": 0.9000, "rank": 1},
            "classic-vision-eyepose-v0": {"score": 0.8500, "rank": 1},
        },
    )
    dialog = LoupeDialog(service=service, image_paths=[str(path_a)], items=[item], start_index=0)
    try:
        label = dialog._score_label
        full_text = dialog._scores_text(dialog._current_item_info())
        assert "AI" in full_text and "Classic" in full_text  # both modules present, as intended

        assert label.maximumWidth() <= 200
        assert label.text() != full_text
        assert label.text().endswith("…")
        assert full_text in label.toolTip()  # full text still reachable on hover
    finally:
        dialog.close()
        service.close()


def test_bottom_bar_fits_a_real_macbook_width_with_two_ranking_modules_scored(app, tmp_path) -> None:
    """Direct regression for the measured deficit: with BOTH the burst info
    row's Color Source label and the score bar unbounded, a burst scored by
    two modules needed 1607px against this MacBook's actual 1470px-wide
    screen (frame fit; the CONTENT didn't - Qt compressed the layout below
    its own minimum, clipping Keep/Reject/Neutral's button text). Asserts
    against a fixed, real budget rather than "whatever this test machine's
    screen happens to be", so it stays meaningful under the offscreen
    platform too."""
    from PySide6.QtWidgets import QWidget

    dialog, service = _burst_dialog(tmp_path, burst_strategy="classic-vision-eyepose-v0", n=4)
    try:
        for item in dialog.items:
            item.ranking_results["ai-model"] = dict(item.ranking_results["classic-vision-eyepose-v0"])
        dialog._update_info_labels()
        dialog.show()
        app.processEvents()

        bottom_bar = next(w for w in dialog.findChildren(QWidget) if w.objectName() == "loupeBottomBar")
        # A real 14"-class MacBook's available width, measured directly
        # (see the fix's own commit notes) - the actual budget this bar's
        # minimum size hint must fit inside without the layout compressing
        # below it.
        macbook_available_width = 1470
        assert bottom_bar.sizeHint().width() <= macbook_available_width
    finally:
        dialog.close()
        service.close()


def test_loupe_window_never_exceeds_the_available_screen_geometry(app, tmp_path) -> None:
    """Even under extreme content-width pressure (an absurdly long Color
    Source name in the burst info row), the window itself must stay within
    screen().availableGeometry() - the bug report's literal complaint was
    the window's own right edge falling off-screen, not just internal
    clipping."""
    very_long_strategy_id = "y" * 200
    dialog, service = _burst_dialog(tmp_path, burst_strategy=very_long_strategy_id)
    try:
        dialog.show()
        app.processEvents()

        screen = dialog.screen()
        assert screen is not None
        available = screen.availableGeometry()
        frame = dialog.frameGeometry()

        assert frame.left() >= available.left()
        assert frame.top() >= available.top()
        assert frame.right() <= available.right()
        assert frame.bottom() <= available.bottom()
    finally:
        dialog.close()
        service.close()


def test_non_burst_loupe_also_fits_the_available_screen_geometry(app, tmp_path) -> None:
    """The window-sizing fix applies unconditionally (not just to burst
    sessions) - a plain, non-burst Loupe must fit too."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog
    from picklikeme.desktop.services import ReviewService

    folder = tmp_path / "shoot"
    folder.mkdir()
    path_a = folder / "a.jpg"
    _make_jpeg(path_a)
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    dialog = LoupeDialog(service=service, image_paths=[str(path_a)], start_index=0)
    try:
        dialog.show()
        app.processEvents()
        screen = dialog.screen()
        available = screen.availableGeometry()
        frame = dialog.frameGeometry()
        assert frame.right() <= available.right()
        assert frame.bottom() <= available.bottom()
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# 2. Left/Right arrow keyboard navigation
# ---------------------------------------------------------------------------


def test_zoom_view_never_takes_keyboard_focus(app, tmp_path) -> None:
    dialog, service = _burst_dialog(tmp_path)
    try:
        assert dialog._view.focusPolicy() == Qt.FocusPolicy.NoFocus
        dialog.show()
        app.processEvents()
        assert app.focusWidget() is not dialog._view
    finally:
        dialog.close()
        service.close()


def _press_key(app, dialog, key) -> None:
    """Sends the key event to whichever widget Qt's OWN focus routing would
    actually deliver it to - app.focusWidget() if something has claimed
    focus, else the dialog itself (matching a real top-level window with no
    focused child; see _ZoomView's own fix). Deliberately NOT
    QTest.keyClick(dialog, ...) unconditionally: that sends the event
    directly to `dialog` regardless of what's actually focused, which
    would make this test pass even without the _ZoomView focus-policy fix
    (verified: with the fix reverted, sending straight to `dialog` still
    worked, while sending to the real focusWidget() - _view - did not, the
    exact bug this test exists to catch)."""
    QTest.keyClick(app.focusWidget() or dialog, key)


def test_right_and_left_arrow_keys_navigate_reusing_go_next_go_prev(app, tmp_path) -> None:
    """Verifies the actual failure mode end-to-end: with the dialog shown
    (as it would be in real use, so whatever Qt naturally focuses is what
    receives the key), arrow keys must move dialog.index via the SAME
    _go_next/_go_prev the Prev/Next buttons call - not a second,
    independent navigation path."""
    dialog, service = _burst_dialog(tmp_path, n=3)
    try:
        dialog.show()
        app.processEvents()

        assert dialog.index == 0
        _press_key(app, dialog, Qt.Key.Key_Right)
        assert dialog.index == 1
        _press_key(app, dialog, Qt.Key.Key_Right)
        assert dialog.index == 2
        _press_key(app, dialog, Qt.Key.Key_Left)
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


def test_arrow_key_navigation_respects_first_and_last_image_boundaries(app, tmp_path) -> None:
    dialog, service = _burst_dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()

        assert dialog.index == 0
        _press_key(app, dialog, Qt.Key.Key_Left)  # already first - must not go negative/wrap
        assert dialog.index == 0

        _press_key(app, dialog, Qt.Key.Key_Right)
        assert dialog.index == 1
        _press_key(app, dialog, Qt.Key.Key_Right)  # already last - must not advance past it
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


def test_arrow_keys_still_navigate_after_clicking_keep_reject_neutral(app, tmp_path) -> None:
    """The highest-frequency real path, not just a freshly-opened dialog:
    Keep/Reject/Neutral is the primary review action, done on every image.
    A clicked QPushButton keeps keyboard focus afterward by default, and -
    verified in isolation, independent of this app's own code - a focused
    QPushButton under this app's Fusion style silently consumes
    Key_Left/Key_Right before QDialog.keyPressEvent ever sees them. Every
    button in the bottom bar is NoFocus for exactly this reason (see
    __init__): clicking Keep/Reject/Neutral must never leave arrow-key
    navigation dead until the user clicks the image or presses Tab away."""
    dialog, service = _burst_dialog(tmp_path, n=3)
    try:
        dialog.show()
        app.processEvents()

        QTest.mouseClick(dialog._status_buttons["neutral"], Qt.MouseButton.LeftButton)
        app.processEvents()
        assert app.focusWidget() is not dialog._status_buttons["neutral"]  # NoFocus - never keeps it
        assert dialog.index == 1  # Neutral auto-advances - see _apply_status

        _press_key(app, dialog, Qt.Key.Key_Right)
        assert dialog.index == 2
        _press_key(app, dialog, Qt.Key.Key_Left)
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


def test_arrow_key_navigation_does_not_reorder_or_change_sort_related_state(app, tmp_path) -> None:
    """Explicit check against the task's own concern: pressing arrows must
    not re-sort the burst or otherwise touch anything beyond `index`."""
    dialog, service = _burst_dialog(tmp_path, n=3)
    try:
        dialog.show()
        app.processEvents()
        paths_before = list(dialog.image_paths)
        items_before = list(dialog.items)

        _press_key(app, dialog, Qt.Key.Key_Right)
        _press_key(app, dialog, Qt.Key.Key_Right)
        _press_key(app, dialog, Qt.Key.Key_Left)

        assert dialog.image_paths == paths_before
        assert dialog.items == items_before
    finally:
        dialog.close()
        service.close()
