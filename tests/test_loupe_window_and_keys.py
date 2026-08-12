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


# ---------------------------------------------------------------------------
# Redesign: no Close button, prominent normalized score, zoom (keyboard,
# pinch, persistence across navigation), Elements mode.
# ---------------------------------------------------------------------------


def test_there_is_no_close_button(app, tmp_path) -> None:
    """Removed - the window's own native controls (and Escape, see
    keyPressEvent) already close it; a dedicated "Close" button was just
    spending bar space on something every window already offers for free."""
    from PySide6.QtWidgets import QPushButton

    dialog, service = _burst_dialog(tmp_path, n=1)
    try:
        button_texts = {b.text() for b in dialog.findChildren(QPushButton)}
        assert "Close" not in button_texts
    finally:
        dialog.close()
        service.close()


def test_primary_score_is_normalized_with_exactly_three_decimals(app, tmp_path) -> None:
    """"IMAGE SCORE: 0.900", never "90.0" (the old *100-scaled burst-row
    reading), "0.90" (two decimals), or "0.9000" (four, matching the
    secondary multi-strategy line's own precision) - exactly three."""
    dialog, service = _burst_dialog(tmp_path, burst_strategy="ai-model", n=1)
    try:
        text = dialog._primary_score_label.text()
        assert text == "IMAGE SCORE: 0.900"
        assert "90.0" not in text
        assert "0.9000" not in text
    finally:
        dialog.close()
        service.close()


def test_primary_score_shows_an_em_dash_when_the_active_strategy_never_scored_this_image(app, tmp_path) -> None:
    dialog, service = _burst_dialog(tmp_path, burst_strategy=None, n=1)
    try:
        assert dialog._primary_score_label.text() == "IMAGE SCORE: —"
    finally:
        dialog.close()
        service.close()


def test_zoom_level_persists_across_navigation(app, tmp_path) -> None:
    """Image A zoomed in -> Right Arrow -> Image B is at the same zoom - the
    task's own explicit example ("Image A -> zoom to 150% -> press Right
    Arrow -> Image B should remain at 150%"). _ZoomView.set_pixmap already
    reapplies _manual_scale/_fit_mode on every image load (see that
    method); this checks the WHOLE path end to end, through real
    navigation - not a specific absolute scale number (zoom_by's factor is
    relative to whatever Fit mode's own scale already was for this
    - deliberately tiny, fast-to-generate - test image, which is not the
    "150%" a real multi-megapixel photo's Fit scale would start near; what
    matters here is that navigating preserves it exactly, at whatever value
    it actually is)."""
    dialog, service = _burst_dialog(tmp_path, n=3)
    try:
        dialog.show()
        app.processEvents()

        assert dialog._view._fit_mode is True  # starts at Fit, not a manual zoom
        fit_scale = dialog._view.transform().m11()
        dialog._view.zoom_by(1.5)
        assert dialog._view._fit_mode is False
        scale_after_zoom = dialog._view._manual_scale
        assert scale_after_zoom != pytest.approx(fit_scale)  # zoom actually changed something

        dialog._go_next()
        assert dialog._view._fit_mode is False
        assert dialog._view._manual_scale == pytest.approx(scale_after_zoom)
        assert dialog._view.transform().m11() == pytest.approx(scale_after_zoom)

        dialog._go_next()
        assert dialog._view._manual_scale == pytest.approx(scale_after_zoom)
    finally:
        dialog.close()
        service.close()


def test_zoom_by_is_clamped_to_sane_bounds(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import MAX_MANUAL_SCALE, MIN_MANUAL_SCALE

    dialog, service = _burst_dialog(tmp_path, n=1)
    try:
        dialog.show()
        app.processEvents()
        for _ in range(60):
            dialog._view.zoom_by(1.5)
        assert dialog._view._manual_scale <= MAX_MANUAL_SCALE
        for _ in range(60):
            dialog._view.zoom_by(1 / 1.5)
        assert dialog._view._manual_scale >= MIN_MANUAL_SCALE
    finally:
        dialog.close()
        service.close()


def test_keyboard_plus_minus_zoom_reuses_the_same_zoom_mechanism(app, tmp_path) -> None:
    """+ / - are keyboard shortcuts for the same _view.zoom_by Ctrl+wheel
    and trackpad pinch already call - not a second zoom implementation.
    Both Key_Plus and Key_Equal are accepted for zoom-in, matching a
    standard US keyboard where "+" is Shift+= ."""
    dialog, service = _burst_dialog(tmp_path, n=1)
    try:
        dialog.show()
        app.processEvents()
        assert dialog._view._fit_mode is True

        _press_key(app, dialog, Qt.Key.Key_Equal)
        assert dialog._view._fit_mode is False
        scale_after_one_zoom_in = dialog._view._manual_scale
        assert scale_after_one_zoom_in > 1.0

        _press_key(app, dialog, Qt.Key.Key_Minus)
        assert dialog._view._manual_scale < scale_after_one_zoom_in
    finally:
        dialog.close()
        service.close()


def test_trackpad_pinch_native_gesture_zooms(app, tmp_path) -> None:
    """macOS trackpad pinch arrives as QNativeGestureEvent/
    ZoomNativeGesture, not QPinchGesture - see _ZoomView.event()'s own
    docstring for why. value() is an incremental scale delta (e.g. +0.02
    per tick), composed multiplicatively via the same zoom_by() Ctrl+wheel
    and the keyboard shortcuts use - so a +0.2 tick means "scale by 1.2",
    relative to whatever the view's current scale already was (Fit mode's
    own scale here, same caveat as the zoom-persistence test above: not a
    specific absolute number for this tiny test image)."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QNativeGestureEvent, QPointingDevice

    dialog, service = _burst_dialog(tmp_path, n=1)
    try:
        dialog.show()
        app.processEvents()
        # A known, exact starting scale (1.0) rather than whatever Fit
        # mode's own scale happens to be for this test image at this
        # viewport size (irrelevant to what's under test here: does the
        # gesture event apply the correct RELATIVE factor via zoom_by).
        dialog._view._fit_mode = False
        dialog._view.resetTransform()
        dialog._view.scale(1.0, 1.0)
        dialog._view._manual_scale = 1.0
        scale_before = dialog._view.transform().m11()
        assert scale_before == pytest.approx(1.0)

        device = QPointingDevice.primaryPointingDevice()
        point = QPointF(50, 50)
        gesture = QNativeGestureEvent(
            Qt.NativeGestureType.ZoomNativeGesture, device, 0, point, point, point, 0.2, QPointF(0, 0),
        )
        # dialog._view.event(gesture) directly, not QApplication.sendEvent -
        # the offscreen QPA platform this suite runs under doesn't fully
        # support routing a synthetic NativeGesture through the real
        # notify()/event-loop pipeline (sendEvent() reliably returns False
        # for one here even though the override itself handles it
        # correctly - verified directly). A real trackpad pinch is
        # delivered by genuine OS/Cocoa machinery no offscreen test can
        # replicate anyway; what this test can and should verify is that
        # _ZoomView.event() itself correctly recognizes and handles the
        # event once Qt hands it one.
        assert dialog._view.event(gesture) is True
        assert dialog._view._fit_mode is False
        assert dialog._view._manual_scale == pytest.approx(scale_before * 1.2, rel=0.05)
    finally:
        dialog.close()
        service.close()


def test_elements_and_boxes_are_independently_toggleable(app, tmp_path) -> None:
    """Manual QA: Elements must not accidentally hide Boxes, or vice versa -
    checking one must never uncheck the other. Both, either, or neither can
    be active at once (see _refresh_detection_overlay's own docstring)."""
    dialog, service = _burst_dialog(tmp_path, n=1)
    try:
        dialog._boxes_btn.setChecked(True)
        assert dialog._show_boxes is True
        assert dialog._show_elements is False

        dialog._elements_btn.setChecked(True)
        assert dialog._show_elements is True
        assert dialog._show_boxes is True  # unchanged - Boxes stays on
        assert dialog._boxes_btn.isChecked() is True

        dialog._boxes_btn.setChecked(False)
        assert dialog._show_boxes is False
        assert dialog._show_elements is True  # unchanged - Elements stays on
        assert dialog._elements_btn.isChecked() is True
    finally:
        dialog.close()
        service.close()


def test_elements_overlay_draws_left_eye_right_eye_and_head_with_confidence(app, tmp_path) -> None:
    """Bounding rectangle + "Name — confidence" label (exactly two decimals)
    for each of the three elements - synthesized from data the eye detector
    already computed (see _ZoomView.set_elements_overlay's own docstring),
    no detection algorithm involved."""
    from unittest import mock

    from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsSimpleTextItem

    dialog, service = _burst_dialog(tmp_path, n=1)
    fake_eye = {
        "source_size": (640, 480),
        "accepted": True,
        "confidence": 0.94,
        "box": (280.0, 180.0, 360.0, 260.0),
        "left": {"x": 300.0, "y": 200.0, "confidence": 0.94},
        "right": {"x": 340.0, "y": 205.0, "confidence": 0.87},
        "head_top": {"x": 320.0, "y": 140.0, "confidence": 0.7},
        "beak": None, "left_shoulder": None, "right_shoulder": None,
        "head_confidence": 0.99,
    }
    try:
        with mock.patch.object(service, "eye_keypoints", return_value=fake_eye):
            dialog._elements_btn.setChecked(True)
            dialog.show()
            app.processEvents()

            boxes = [i for i in dialog._view._overlay_items if isinstance(i, QGraphicsRectItem)]
            labels = [i for i in dialog._view._overlay_items if isinstance(i, QGraphicsSimpleTextItem)]
            # 3 element boxes + 3 label-backing rects = 6 QGraphicsRectItem.
            assert len(boxes) == 6
            assert len(labels) == 3
            label_texts = {label.text() for label in labels}
            assert label_texts == {"Left Eye — 0.94", "Right Eye — 0.87", "Head — 0.99"}
    finally:
        dialog.close()
        service.close()


def test_elements_overlay_skips_an_element_with_no_confidence_to_show(app, tmp_path) -> None:
    """Never fabricates a confidence - a backend (or record) with no
    head_confidence at all must simply not draw a Head element, not draw
    one with a made-up number."""
    from unittest import mock

    dialog, service = _burst_dialog(tmp_path, n=1)
    fake_eye = {
        "source_size": (640, 480),
        "accepted": True,
        "confidence": 0.94,
        "box": (280.0, 180.0, 360.0, 260.0),
        "left": {"x": 300.0, "y": 200.0, "confidence": 0.94},
        "right": None,
        "head_top": {"x": 320.0, "y": 140.0, "confidence": 0.7},
        "beak": None, "left_shoulder": None, "right_shoulder": None,
        "head_confidence": None,
    }
    try:
        with mock.patch.object(service, "eye_keypoints", return_value=fake_eye):
            dialog._elements_btn.setChecked(True)
            dialog.show()
            app.processEvents()

            from PySide6.QtWidgets import QGraphicsSimpleTextItem

            labels = {
                i.text() for i in dialog._view._overlay_items if isinstance(i, QGraphicsSimpleTextItem)
            }
            assert labels == {"Left Eye — 0.94"}
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# 3. A corrupt cached preview must self-heal, not show a permanently blank
#    Loupe with no error at all - see review.thumbnails.review_preview's own
#    atomic-write fix and this file's _load_current isNull() check.
# ---------------------------------------------------------------------------


def test_a_corrupt_cached_preview_is_deleted_and_regenerated(app, tmp_path) -> None:
    """Reproduces the real failure: QPixmap silently returns a null pixmap
    (no exception) for a truncated/corrupt cached preview JPEG - previously
    invisible to _load_current, which never checked isNull(). The cache
    lookup only ever checked existence, so a corrupt file was served
    forever. Both are fixed now: a stale corrupt cache entry is deleted and
    regenerated on the very next load."""
    dialog, service = _burst_dialog(tmp_path, n=1)
    try:
        path = dialog._current_path()
        cached_preview = service.preview_path(path)
        assert cached_preview.is_file()

        # Simulate an interrupted write: truncate the real cached preview to
        # a handful of garbage bytes - non-empty, so a naive "0 bytes means
        # corrupt" check alone would miss it; only a real decode attempt
        # (QPixmap's own) catches this.
        cached_preview.write_bytes(b"\xff\xd8\xff\x00garbage-not-a-real-jpeg")

        dialog._load_current()  # must not raise, and must not stay blank

        assert dialog._current_raw_pixmap is not None
        assert not dialog._current_raw_pixmap.isNull()
        # The cache healed itself: the same path now holds a real, decodable
        # image again rather than the garbage bytes written above.
        assert cached_preview.stat().st_size > len(b"\xff\xd8\xff\x00garbage-not-a-real-jpeg")
    finally:
        dialog.close()
        service.close()


def test_a_missing_preview_source_still_shows_a_warning_not_a_silent_blank(app, tmp_path, monkeypatch) -> None:
    """If regeneration itself fails (e.g. the source file was deleted from
    disk between opening the Loupe and navigating to it), the failure must
    be visible - a warning dialog, not an indistinguishable blank frame."""
    from unittest import mock

    dialog, service = _burst_dialog(tmp_path, n=1)
    try:
        path = dialog._current_path()
        cached_preview = service.preview_path(path)
        cached_preview.write_bytes(b"not a real jpeg at all")
        # Regeneration itself now fails too (source unreadable/removed).
        with mock.patch.object(service, "preview_path", side_effect=[cached_preview, FileNotFoundError("gone")]):
            with mock.patch(
                "picklikeme.desktop.dialogs.loupe_dialog.QMessageBox.warning"
            ) as warn:
                dialog._load_current()
                assert warn.called
        assert dialog._current_raw_pixmap.isNull()
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# A metrics value the Loupe cannot format must never stop the Loupe opening.
#
# THE regression: Crop Sharpness records `relative_subject_size: null` for a
# full-frame-fallback image (no subject was located, so there was nothing to
# measure - see ranking.crop_sharpness.ImageMetrics). The diagnostics line
# formatted every metric with `:.3f`, which raises TypeError on None. That
# ran inside the Loupe's own construction, so the exception escaped through
# the `doubleClicked` slot - Qt prints it to a stderr a windowed app never
# shows, then simply does not open the dialog. The symptom was a card that
# silently did nothing when double-clicked, on 1,582 of 5,986 real images.
# ---------------------------------------------------------------------------


def test_a_none_metric_renders_as_not_measured_instead_of_raising() -> None:
    from picklikeme.desktop.widgets.design_system import format_metric_value

    assert format_metric_value(None) == "not measured"
    assert format_metric_value(0.0) == "0.000", "zero is a measurement, not an absence"
    assert format_metric_value(0.1754747) == "0.175"


def test_a_non_numeric_metric_never_raises_either() -> None:
    """A metrics report is whatever the strategy chose to record, so a value
    is not guaranteed to be a float. `has_subject_detection` is a bool, and
    bool survives `:.3f` (bool is an int) as a nonsense "1.000"."""
    from picklikeme.desktop.widgets.design_system import format_metric_value

    assert format_metric_value(True) == "yes"
    assert format_metric_value(False) == "no"
    assert format_metric_value("n/a") == "n/a"


def test_the_diagnostics_line_survives_a_full_frame_fallback_images_metrics(app) -> None:
    """The exact payload `ranking.crop_sharpness.write_metrics_report` writes
    for an image with no detected subject, through the real method."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    fallback = {
        "metrics": {
            "crop-sharpness": {
                "crop_sharpness": 1.1090635061264038,
                "relative_subject_size": None,
                "has_subject_detection": False,
            }
        }
    }

    text = LoupeDialog._diagnostics_text(fallback)

    assert "not measured" in text
    assert "1.109" in text
    assert "Relative Subject Size" in text


def test_the_diagnostics_line_still_shows_a_real_subject_size(app) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    detected = {
        "metrics": {
            "crop-sharpness": {
                "crop_sharpness": 0.8218353986740112,
                "relative_subject_size": 0.17547475564033874,
                "has_subject_detection": True,
            }
        }
    }

    text = LoupeDialog._diagnostics_text(detected)

    assert "0.822" in text
    assert "0.175" in text
    assert "not measured" not in text
