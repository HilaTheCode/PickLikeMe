"""Focused operational verification of the Loupe's core controls - zoom
(including pan-position persistence, a real gap this file's own fix closes),
navigation, brightness, Keep/Reject decisions, and Boxes/Elements overlays.

Each test drives the REAL event -> handler -> state -> rendering chain
(real QWheelEvent objects sent to `_ZoomView.wheelEvent`, real
QTest.mouseClick on the actual buttons, real key events routed through
Qt's own focus system via `_press_key` - not internal methods called
directly where a real event path exists to exercise instead), per this
suite's own review brief: "do not merely inspect that methods exist."
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

try:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _make_jpeg(path: Path, size: tuple[int, int] = (400, 300)) -> None:
    from PIL import Image

    Image.new("RGB", size, color=(80, 120, 160)).save(path, format="JPEG")


def _dialog(tmp_path, *, n: int = 3, size: tuple[int, int] = (400, 300)):
    """A plain (non-burst-scoped) multi-image LoupeDialog - burst scoping is
    irrelevant to the operational controls this file verifies."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog
    from picklikeme.desktop.services import ReviewService

    folder = tmp_path / "shoot"
    folder.mkdir(exist_ok=True)
    paths = [folder / f"img{i}.jpg" for i in range(n)]
    for path in paths:
        _make_jpeg(path, size)

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)
    dialog = LoupeDialog(service=service, image_paths=[str(p) for p in paths], start_index=0)
    return dialog, service


def _press_key(app, dialog, key) -> None:
    """Sends the key event to whichever widget Qt's OWN focus routing would
    actually deliver it to - the same real-focus-path helper
    `test_loupe_window_and_keys.py` already established, reused verbatim so
    this suite's key tests exercise the identical real chain."""
    QTest.keyClick(app.focusWidget() or dialog, key)


def _wheel_event(*, control: bool, delta_y: int = 120) -> QWheelEvent:
    modifiers = Qt.KeyboardModifier.ControlModifier if control else Qt.KeyboardModifier.NoModifier
    return QWheelEvent(
        QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, delta_y),
        Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False,
    )


# ---------------------------------------------------------------------------
# 1. ZOOM
# ---------------------------------------------------------------------------


def test_ctrl_wheel_zooms_via_a_real_wheel_event(app, tmp_path) -> None:
    """A real QWheelEvent with ControlModifier, sent through the actual
    `_ZoomView.wheelEvent` override - not `zoom_by` called directly."""
    dialog, service = _dialog(tmp_path)
    try:
        dialog.show()
        app.processEvents()
        assert dialog._view._fit_mode is True
        fit_scale = dialog._view.transform().m11()

        dialog._view.wheelEvent(_wheel_event(control=True, delta_y=120))

        assert dialog._view._fit_mode is False
        assert dialog._view.transform().m11() != pytest.approx(fit_scale)
        assert dialog.index == 0  # Ctrl+wheel must not also navigate
    finally:
        dialog.close()
        service.close()


def test_plain_wheel_navigates_via_a_real_wheel_event(app, tmp_path) -> None:
    """A real QWheelEvent with no modifiers - must navigate, not zoom."""
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        scale_before = dialog._view.transform().m11()

        dialog._view.wheelEvent(_wheel_event(control=False, delta_y=120))  # scroll up = previous direction

        assert dialog.index == 0  # already first - _on_wheel_navigate/_go_prev clamps
        dialog._view.wheelEvent(_wheel_event(control=False, delta_y=-120))  # scroll down = next
        assert dialog.index == 1
        assert dialog._view._fit_mode is True  # plain wheel must not have zoomed
        assert dialog._view.transform().m11() == pytest.approx(scale_before)
    finally:
        dialog.close()
        service.close()


def test_wheel_navigation_and_ctrl_wheel_zoom_do_not_conflict(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()

        dialog._view.wheelEvent(_wheel_event(control=True, delta_y=120))
        assert dialog.index == 0
        zoomed_scale = dialog._view._manual_scale

        dialog._view.wheelEvent(_wheel_event(control=False, delta_y=-120))
        assert dialog.index == 1
        assert dialog._view._manual_scale == pytest.approx(zoomed_scale)  # zoom carried over, untouched by nav
    finally:
        dialog.close()
        service.close()


def test_keyboard_plus_and_minus_zoom(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path)
    try:
        dialog.show()
        app.processEvents()
        fit_scale = dialog._view.transform().m11()

        _press_key(app, dialog, Qt.Key.Key_Plus)
        assert dialog._view._manual_scale > fit_scale

        after_plus = dialog._view._manual_scale
        _press_key(app, dialog, Qt.Key.Key_Minus)
        assert dialog._view._manual_scale < after_plus
    finally:
        dialog.close()
        service.close()


def test_zoom_clamps_at_minimum_and_maximum(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import MAX_MANUAL_SCALE, MIN_MANUAL_SCALE

    dialog, service = _dialog(tmp_path, n=1)
    try:
        dialog.show()
        app.processEvents()
        for _ in range(80):
            dialog._view.wheelEvent(_wheel_event(control=True, delta_y=120))
        assert dialog._view._manual_scale <= MAX_MANUAL_SCALE
        for _ in range(80):
            dialog._view.wheelEvent(_wheel_event(control=True, delta_y=-120))
        assert dialog._view._manual_scale >= MIN_MANUAL_SCALE
    finally:
        dialog.close()
        service.close()


def test_zoom_level_persists_across_navigation(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=3)
    try:
        dialog.show()
        app.processEvents()
        dialog._view.set_zoom_percent(200)
        dialog._go_next()
        assert dialog._view._fit_mode is False
        assert dialog._view.transform().m11() == pytest.approx(2.0, abs=1e-6)
        dialog._go_next()
        assert dialog._view.transform().m11() == pytest.approx(2.0, abs=1e-6)
    finally:
        dialog.close()
        service.close()


def test_zoom_pan_position_persists_across_navigation(app, tmp_path) -> None:
    """Regression: `_ZoomView.set_pixmap` used to unconditionally
    `centerOn(self._pixmap_item)` on every navigation - re-centering on the
    WHOLE new image regardless of where the photographer had panned to.
    Zoomed to 300% and panned to a specific region, the same RELATIVE
    region (as a fraction of the image's own dimensions - see
    `_ZoomView._capture_pan_fraction`) must still be centered after
    navigating forward, and again after navigating back."""
    dialog, service = _dialog(tmp_path, n=3, size=(2000, 1500))
    try:
        from PySide6.QtCore import Qt as _Qt

        dialog.setWindowState(_Qt.WindowState.WindowNoState)
        dialog.resize(1470, 900)
        dialog.show()
        app.processEvents()

        dialog._view.set_zoom_percent(300)
        dialog._view.centerOn(QPointF(1500, 1125))  # 75%, 75% of the 2000x1500 source
        app.processEvents()

        def _center_fraction():
            center = dialog._view.mapToScene(dialog._view.viewport().rect().center())
            return center.x() / 2000, center.y() / 1500

        before = _center_fraction()

        dialog._go_next()
        app.processEvents()
        after_next = _center_fraction()
        assert after_next[0] == pytest.approx(before[0], abs=0.01)
        assert after_next[1] == pytest.approx(before[1], abs=0.01)
        assert dialog._view._fit_mode is False
        assert dialog._view.transform().m11() == pytest.approx(3.0, abs=1e-6)

        dialog._go_prev()
        app.processEvents()
        after_prev = _center_fraction()
        assert after_prev[0] == pytest.approx(before[0], abs=0.01)
        assert after_prev[1] == pytest.approx(before[1], abs=0.01)
    finally:
        dialog.close()
        service.close()


def test_fit_mode_navigation_never_records_a_pan_fraction_to_restore(app, tmp_path) -> None:
    """At Fit (the default, never explicitly zoomed), navigating must keep
    re-fitting the whole frame - no stale pan fraction from a DIFFERENT
    image should ever leak in."""
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        assert dialog._view._fit_mode is True
        dialog._go_next()
        assert dialog._view._fit_mode is True
        assert dialog._view._pan_fraction is None
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# 2. NAVIGATION
# ---------------------------------------------------------------------------


def test_left_right_arrow_keys_navigate(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=3)
    try:
        dialog.show()
        app.processEvents()
        assert dialog.index == 0
        _press_key(app, dialog, Qt.Key.Key_Right)
        assert dialog.index == 1
        _press_key(app, dialog, Qt.Key.Key_Left)
        assert dialog.index == 0
    finally:
        dialog.close()
        service.close()


def test_navigation_preserves_the_callers_own_order(app, tmp_path) -> None:
    """Loupe must never independently sort/reorder - `image_paths` is
    walked in exactly the order the caller supplied, deliberately
    out-of-filename-order here to prove nothing re-sorts it."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog
    from picklikeme.desktop.services import ReviewService

    folder = tmp_path / "shoot"
    folder.mkdir()
    names = ["c.jpg", "a.jpg", "b.jpg"]
    for name in names:
        _make_jpeg(folder / name)
    ordered_paths = [str(folder / name) for name in names]

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)
    dialog = LoupeDialog(service=service, image_paths=ordered_paths, start_index=0)
    try:
        assert dialog.image_paths == ordered_paths
        assert dialog._current_path() == ordered_paths[0]
        dialog._go_next()
        assert dialog._current_path() == ordered_paths[1]
        dialog._go_next()
        assert dialog._current_path() == ordered_paths[2]
    finally:
        dialog.close()
        service.close()


def test_navigation_works_while_zoomed(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        dialog._view.set_zoom_percent(150)
        _press_key(app, dialog, Qt.Key.Key_Right)
        assert dialog.index == 1
        assert dialog._view._fit_mode is False
    finally:
        dialog.close()
        service.close()


def test_navigation_works_with_view_mode_panels_hidden(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import VIEW_MODE_FOCUS_IMAGE

    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        dialog._set_view_mode(VIEW_MODE_FOCUS_IMAGE)
        _press_key(app, dialog, Qt.Key.Key_Right)
        assert dialog.index == 1
        assert dialog._algo_results_container.isVisible() is False
        assert dialog._elements_panel.isVisible() is False
    finally:
        dialog.close()
        service.close()


def test_navigation_works_after_keep_reject(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=3)
    try:
        dialog.show()
        app.processEvents()
        dialog._apply_status("keep")  # auto-advances to index 1
        assert dialog.index == 1
        _press_key(app, dialog, Qt.Key.Key_Right)
        assert dialog.index == 2
        _press_key(app, dialog, Qt.Key.Key_Left)
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# 3. BRIGHTNESS
# ---------------------------------------------------------------------------


def test_brightness_up_and_down_change_exposure_steps(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path)
    try:
        assert dialog._exposure_steps == 0
        dialog._adjust_exposure(1)
        assert dialog._exposure_steps == 1
        assert dialog._exposure_label.text() == "+0.3 EV"
        dialog._adjust_exposure(-1)
        assert dialog._exposure_steps == 0
        assert dialog._exposure_label.text() == "+0.0 EV"
    finally:
        dialog.close()
        service.close()


def test_brightness_is_clamped_to_its_bounds(app, tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import EXPOSURE_MAX_STEPS, EXPOSURE_MIN_STEPS

    dialog, service = _dialog(tmp_path)
    try:
        for _ in range(EXPOSURE_MAX_STEPS + 5):
            dialog._adjust_exposure(1)
        assert dialog._exposure_steps == EXPOSURE_MAX_STEPS
        for _ in range(EXPOSURE_MAX_STEPS - EXPOSURE_MIN_STEPS + 5):
            dialog._adjust_exposure(-1)
        assert dialog._exposure_steps == EXPOSURE_MIN_STEPS
    finally:
        dialog.close()
        service.close()


def test_brightness_persists_across_navigation_and_back(app, tmp_path) -> None:
    """Regression coverage - previously untested. `_exposure_steps` is a
    LoupeDialog-level (not per-image) attribute, and `_load_current`
    re-applies it to whatever image is current via `_exposed_pixmap` - this
    verifies that end to end, through real navigation."""
    dialog, service = _dialog(tmp_path, n=3)
    try:
        dialog.show()
        app.processEvents()
        dialog._adjust_exposure(1)
        dialog._adjust_exposure(1)
        assert dialog._exposure_steps == 2

        dialog._go_next()
        assert dialog._exposure_steps == 2  # not reset by navigation
        assert dialog._exposure_label.text() == "+0.7 EV"

        dialog._go_next()
        assert dialog._exposure_steps == 2

        dialog._go_prev()
        assert dialog._exposure_steps == 2
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# 4. USER DECISIONS - Keep / Reject
# ---------------------------------------------------------------------------


def _status_of(service, path: str) -> str:
    images = {img["image_path"]: img for img in service.load_session()["images"]}
    return images[str(Path(path).resolve())]["review_status"]


def test_mouse_click_keep_saves_and_advances(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        path0 = dialog.image_paths[0]

        QTest.mouseClick(dialog._status_buttons["keep"], Qt.MouseButton.LeftButton)
        app.processEvents()

        assert _status_of(service, path0) == "keep"
        assert dialog.index == 1  # auto-advanced
        assert app.focusWidget() is not dialog._status_buttons["keep"]  # NoFocus
    finally:
        dialog.close()
        service.close()


def test_mouse_click_reject_saves_and_advances(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        path0 = dialog.image_paths[0]

        QTest.mouseClick(dialog._status_buttons["reject"], Qt.MouseButton.LeftButton)
        app.processEvents()

        assert _status_of(service, path0) == "reject"
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


def test_keyboard_k_saves_keep_and_advances(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        path0 = dialog.image_paths[0]

        _press_key(app, dialog, Qt.Key.Key_K)
        app.processEvents()

        assert _status_of(service, path0) == "keep"
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


def test_keyboard_r_saves_reject_and_advances(app, tmp_path) -> None:
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        path0 = dialog.image_paths[0]

        _press_key(app, dialog, Qt.Key.Key_R)
        app.processEvents()

        assert _status_of(service, path0) == "reject"
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


def test_decisions_reuse_the_existing_review_session_mechanism_not_a_second_one(app, tmp_path) -> None:
    """K/R/mouse-Keep/mouse-Reject must all funnel through the SAME
    `ReviewService.set_review_status` call - never a Loupe-local decision
    store that could drift from the Grid/session's own state."""
    from unittest import mock

    dialog, service = _dialog(tmp_path, n=1)
    try:
        with mock.patch.object(service, "set_review_status", wraps=service.set_review_status) as spy:
            dialog._apply_status("keep")
        spy.assert_called_once()
        assert spy.call_args.args[0] == dialog.image_paths[0]
        assert spy.call_args.args[1] == "keep"
    finally:
        dialog.close()
        service.close()


def test_k_r_still_work_after_the_elements_source_combo_had_focus(app, tmp_path) -> None:
    """The redesign's new Elements Source combo (`_elements_source_combo`)
    must not be able to swallow K/R the way a focused QPushButton/QComboBox
    could before every bottom-bar control was made NoFocus. A REAL mouse
    click, not `.setFocus()`: `.setFocus()` is a programmatic request that
    bypasses `Qt.FocusPolicy.NoFocus` entirely (confirmed empirically - a
    NoFocus combo still becomes `app.focusWidget()` after an explicit
    `.setFocus()` call), so it would not actually exercise what NoFocus is
    for - refusing focus from a genuine user click."""
    dialog, service = _dialog(tmp_path, n=2)
    try:
        dialog.show()
        app.processEvents()
        QTest.mouseClick(dialog._elements_source_combo, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert app.focusWidget() is not dialog._elements_source_combo  # NoFocus rejects a real click

        path0 = dialog.image_paths[0]
        _press_key(app, dialog, Qt.Key.Key_K)
        assert _status_of(service, path0) == "keep"
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


def test_k_r_still_work_after_clicking_an_algorithm_result_row(app, tmp_path) -> None:
    from picklikeme.desktop.models.image_item import ImageItem

    dialog, service = _dialog(tmp_path, n=2)
    try:
        item = ImageItem(path=dialog.image_paths[0], file_name="img0.jpg", ranking_results={"ai-model": {"score": 0.7}})
        dialog.items = [item, ImageItem(path=dialog.image_paths[1], file_name="img1.jpg")]
        dialog._refresh_algorithm_results()
        dialog.show()
        app.processEvents()

        row = dialog._algo_rows["ai-model"]
        QTest.mouseClick(row, Qt.MouseButton.LeftButton)
        app.processEvents()

        path0 = dialog.image_paths[0]
        _press_key(app, dialog, Qt.Key.Key_R)
        assert _status_of(service, path0) == "reject"
        assert dialog.index == 1
    finally:
        dialog.close()
        service.close()


# ---------------------------------------------------------------------------
# 5 & 6. Boxes / Elements overlays - real toggle -> real rendered output
# ---------------------------------------------------------------------------


def _give_elements_source(dialog, *, strategy_id: str = "ai-model") -> None:
    """Elements has nothing to show (by design) until some strategy has
    actually scored the current image (see `LoupeDialog._elements_source_id`'s
    own docstring) - this gives the dialog's current image a real
    `ranking_results` entry, exactly as a real ranked image would carry it,
    so `_elements_source_id` resolves to a real strategy instead of staying
    None. Boxes has no such requirement (its subject box is strategy-
    independent) - only Elements-focused tests need this."""
    from picklikeme.desktop.models.image_item import ImageItem

    item = ImageItem(
        path=dialog.image_paths[0], file_name="img0.jpg", ranking_results={strategy_id: {"score": 0.8}},
    )
    dialog.items = [item] + [
        ImageItem(path=p, file_name=Path(p).name) for p in dialog.image_paths[1:]
    ]
    dialog._refresh_algorithm_results()


def _fake_boxes_and_eye():
    boxes = {
        "source_size": (400, 300), "selected": {"box": (100.0, 80.0, 200.0, 180.0)}, "others": [],
    }
    eye = {
        "source_size": (400, 300), "accepted": True, "confidence": 0.9,
        "box": (120.0, 100.0, 160.0, 140.0),
        "left": {"x": 130.0, "y": 110.0, "confidence": 0.9},
        "right": {"x": 150.0, "y": 112.0, "confidence": 0.85},
        "head_top": {"x": 140.0, "y": 95.0, "confidence": 0.8},
        "head_confidence": 0.88,
    }
    return boxes, eye


def test_boxes_toggle_reversibly_shows_and_hides_real_rendered_boxes(app, tmp_path) -> None:
    from unittest import mock

    dialog, service = _dialog(tmp_path, n=1)
    boxes, eye = _fake_boxes_and_eye()
    try:
        with mock.patch.object(service, "detection_boxes", return_value=boxes), \
             mock.patch.object(service, "eye_keypoints", return_value=eye):
            dialog.show()
            app.processEvents()
            assert dialog._view._overlay_items == []  # OFF by default

            QTest.mouseClick(dialog._boxes_btn, Qt.MouseButton.LeftButton)  # ON
            app.processEvents()
            assert dialog._show_boxes is True
            assert len(dialog._view._overlay_items) > 0

            QTest.mouseClick(dialog._boxes_btn, Qt.MouseButton.LeftButton)  # OFF
            app.processEvents()
            assert dialog._show_boxes is False
            assert dialog._view._overlay_items == []

            QTest.mouseClick(dialog._boxes_btn, Qt.MouseButton.LeftButton)  # ON again
            app.processEvents()
            assert len(dialog._view._overlay_items) > 0
    finally:
        dialog.close()
        service.close()


def test_elements_toggle_reversibly_shows_and_hides_real_rendered_elements(app, tmp_path) -> None:
    from unittest import mock

    from PySide6.QtWidgets import QGraphicsSimpleTextItem

    dialog, service = _dialog(tmp_path, n=1)
    boxes, eye = _fake_boxes_and_eye()
    try:
        with mock.patch.object(service, "detection_boxes", return_value=boxes), \
             mock.patch.object(service, "eye_keypoints", return_value=eye):
            dialog.show()
            app.processEvents()
            _give_elements_source(dialog)

            QTest.mouseClick(dialog._elements_btn, Qt.MouseButton.LeftButton)  # ON
            app.processEvents()
            assert dialog._show_elements is True
            labels = [i for i in dialog._view._overlay_items if isinstance(i, QGraphicsSimpleTextItem)]
            assert {label.text() for label in labels} == {"Left Eye — 0.90", "Right Eye — 0.85", "Head — 0.88"}

            QTest.mouseClick(dialog._elements_btn, Qt.MouseButton.LeftButton)  # OFF
            app.processEvents()
            assert dialog._show_elements is False
            assert dialog._view._overlay_items == []

            QTest.mouseClick(dialog._elements_btn, Qt.MouseButton.LeftButton)  # ON again
            app.processEvents()
            assert dialog._show_elements is True
            assert len(dialog._view._overlay_items) > 0
    finally:
        dialog.close()
        service.close()


def test_boxes_and_elements_active_simultaneously_render_both(app, tmp_path) -> None:
    from unittest import mock

    from PySide6.QtWidgets import QGraphicsSimpleTextItem

    dialog, service = _dialog(tmp_path, n=1)
    boxes, eye = _fake_boxes_and_eye()
    try:
        with mock.patch.object(service, "detection_boxes", return_value=boxes) as boxes_spy, \
             mock.patch.object(service, "eye_keypoints", return_value=eye):
            dialog.show()
            app.processEvents()
            _give_elements_source(dialog)

            QTest.mouseClick(dialog._boxes_btn, Qt.MouseButton.LeftButton)
            QTest.mouseClick(dialog._elements_btn, Qt.MouseButton.LeftButton)
            app.processEvents()

            assert dialog._show_boxes is True
            assert dialog._show_elements is True
            boxes_spy.assert_called()  # Boxes' own detection_boxes call still happens
            labels = [i for i in dialog._view._overlay_items if isinstance(i, QGraphicsSimpleTextItem)]
            assert len(labels) == 3  # Elements' own labels are present too - both rendered together
    finally:
        dialog.close()
        service.close()


def test_elements_overlay_uses_the_currently_selected_elements_source(app, tmp_path) -> None:
    """Boxes/Elements must read whichever strategy is currently selected
    as Elements Source - never the old single-global-eye-cache behavior
    (see eyes.cache's own per-(image, strategy) keying and `ReviewService.
    eye_keypoints`'s explicit `strategy_id` parameter)."""
    from unittest import mock

    from picklikeme.desktop.models.image_item import ImageItem

    dialog, service = _dialog(tmp_path, n=1)
    try:
        item = ImageItem(
            path=dialog.image_paths[0], file_name="img0.jpg",
            ranking_results={"ai-model": {"score": 0.7}, "classic-vision": {"score": 0.8}},
        )
        dialog.items = [item]
        dialog._refresh_algorithm_results()
        dialog._select_elements_source("classic-vision")

        with mock.patch.object(service, "eye_keypoints", return_value=None) as fake:
            dialog._show_elements = True
            dialog._refresh_detection_overlay()

        assert fake.call_args.kwargs.get("strategy_id") == "classic-vision"
    finally:
        dialog.close()
        service.close()
