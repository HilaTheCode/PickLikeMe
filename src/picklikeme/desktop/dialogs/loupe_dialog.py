"""Loupe (zoom) review dialog - fast per-image Keep/Reject/Neutral triage.

Mirrors the browser Lightbox's core workflow: zoom/pan the current image,
apply a Keep/Reject/Neutral decision with an optional reason, move to the
next image in the filtered set, and optionally save a quick JPEG for
sharing before any RAW development happens.

The image viewer fills nearly the entire dialog; a single dark overlay bar
at the bottom carries every control and status readout, matching the web
Lightbox's information density instead of splitting controls across a
top bar and a control row.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...analyzer.annotations import (
    REVIEW_REASON_BAD_QUALITY,
    REVIEW_REASON_CLEAR_EYES_SEEN,
    REVIEW_REASON_EYES_NOT_SEEN,
    REVIEW_REASON_GOOD_QUALITY,
    REVIEW_REASON_OTHER,
)
from ...analyzer.contactsheets import OTHER_BOX, SELECTED_BOX
from .. import theme
from ..models.image_item import ImageItem
from ..services import ReviewService

REASON_LABELS: dict[str, str] = {
    REVIEW_REASON_EYES_NOT_SEEN: "Eyes not seen",
    REVIEW_REASON_CLEAR_EYES_SEEN: "Clear eyes seen",
    REVIEW_REASON_GOOD_QUALITY: "Overall good quality",
    REVIEW_REASON_BAD_QUALITY: "Overall bad quality",
    REVIEW_REASON_OTHER: "Other",
}

# Display-only exposure preview, mirroring the web Lightbox's brightness
# filter (review/page.py EXPOSURE_STEP/_MIN_STEPS/_MAX_STEPS): 1/3-stop
# clicks, +-3.0 EV, never written back to the file. Persists across
# navigation within one Loupe session - closer to FastRawViewer's exposure
# preview than a per-image setting - and resets when the dialog is closed.
EXPOSURE_STEP = 1 / 3
EXPOSURE_MIN_STEPS = -9
EXPOSURE_MAX_STEPS = 9

STATUS_LABELS = {"keep": "Keep", "reject": "Reject", "neutral": "Neutral"}
# Always the dark palette's semantic colors, regardless of the app theme -
# the Loupe's overlay bar is permanently dark-chrome (see module docstring
# and theme.py), so it needs colors tuned for that background specifically.
STATUS_COLORS = {"keep": theme.DARK.keep_fg, "reject": theme.DARK.reject_fg, "neutral": theme.DARK.neutral_fg}


def _apply_brightness(pixmap: QPixmap, ev: float) -> QPixmap:
    """Approximate a CSS brightness(2^ev) filter on a QPixmap.

    Good enough to judge a dark/bright frame at a glance - the same bar the
    web preview sets for itself - not a color-accurate exposure simulation.
    """
    if pixmap.isNull() or abs(ev) < 1e-6:
        return pixmap
    factor = 2.0**ev
    result = QPixmap(pixmap.size())
    result.fill(Qt.GlobalColor.transparent)
    painter = QPainter(result)
    painter.drawPixmap(0, 0, pixmap)
    if factor >= 1.0:
        overlay = QColor(255, 255, 255)
        overlay.setAlphaF(max(0.0, min(1.0, factor - 1.0)))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        painter.fillRect(result.rect(), overlay)
    else:
        gray = max(0, min(255, int(255 * factor)))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
        painter.fillRect(result.rect(), QColor(gray, gray, gray))
    painter.end()
    return result


class _ZoomView(QGraphicsView):
    """A QGraphicsView with Lightroom-style navigation:

    - Fit-to-window by default
    - Ctrl+wheel zooms; plain wheel requests prev/next navigation
    - Click-drag pans (QGraphicsView's ScrollHandDrag)
    - Double-click toggles between Fit and 100%
    - The current zoom mode (fit vs. a manual scale) carries over to the
      next image loaded, so flipping through a burst at 100% doesn't reset
      to fit on every frame.
    """

    zoomChanged = Signal(float)
    navigateRequested = Signal(int)  # -1 = previous, +1 = next

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(theme.DARK.window_bg))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._fit_mode = True
        self._manual_scale = 1.0
        self._overlay_items: list = []

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Load a new image, applying the current zoom mode (fit or manual)."""
        self._scene.clear()
        self._overlay_items = []  # scene.clear() already deleted these Qt objects
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        if not pixmap.isNull():
            self.resetTransform()
            if self._fit_mode:
                self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            else:
                self.scale(self._manual_scale, self._manual_scale)
                self.centerOn(self._pixmap_item)
        self._emit_zoom()

    def update_pixmap(self, pixmap: QPixmap) -> None:
        """Replace the current pixmap in place, preserving zoom/pan (for exposure changes)."""
        if self._pixmap_item is not None:
            self._pixmap_item.setPixmap(pixmap)

    def clear_detection_overlay(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []

    def set_detection_overlay(self, boxes_data: dict | None) -> None:
        """Draw the detector's boxes (review/thumbnails.detection_boxes_for)
        over the current image - solid green for the box that became the
        crop the model scored, dashed amber for runners-up it passed over.
        Same colors as the gallery's own with_boxes thumbnail overlay
        (contactsheets.SELECTED_BOX/OTHER_BOX) so both views agree.

        Coordinates in boxes_data are full-frame source pixels; scaled here
        against the currently-displayed pixmap's own size rather than
        assumed to match 1:1, since nothing guarantees the preview was
        never resized relative to the source frame.
        """
        self.clear_detection_overlay()
        if boxes_data is None or self._pixmap_item is None:
            return
        source_size = boxes_data.get("source_size")
        if not source_size or source_size[0] <= 0 or source_size[1] <= 0:
            return
        pixmap_size = self._pixmap_item.pixmap().size()
        scale_x = pixmap_size.width() / source_size[0]
        scale_y = pixmap_size.height() / source_size[1]

        def to_scene_rect(box: dict) -> QRectF:
            x1, y1, x2, y2 = box["box"]
            return QRectF(x1 * scale_x, y1 * scale_y, (x2 - x1) * scale_x, (y2 - y1) * scale_y)

        for box in boxes_data.get("others", []):
            item = self._scene.addRect(to_scene_rect(box), QPen(QColor(*OTHER_BOX), 2, Qt.PenStyle.DashLine))
            item.setZValue(10)
            self._overlay_items.append(item)

        selected = boxes_data.get("selected")
        if selected is not None:
            item = self._scene.addRect(to_scene_rect(selected), QPen(QColor(*SELECTED_BOX), 3))
            item.setZValue(11)
            self._overlay_items.append(item)

    def _emit_zoom(self) -> None:
        self.zoomChanged.emit(self.transform().m11())

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override signature
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            self._fit_mode = False
            self._manual_scale = self.transform().m11()
            self._emit_zoom()
        else:
            direction = -1 if event.angleDelta().y() > 0 else 1
            self.navigateRequested.emit(direction)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override signature
        if self._pixmap_item is None:
            return
        if self._fit_mode:
            self._fit_mode = False
            self._manual_scale = 1.0
            self.resetTransform()
            self.centerOn(self._pixmap_item)
        else:
            self._fit_mode = True
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_zoom()
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        # set_pixmap() calls fitInView() using whatever viewport size exists
        # at that moment - but the constructor loads the first image before
        # the dialog has been shown/laid out to its final size, so that
        # first fitInView() computes its scale against a near-empty
        # viewport and bakes in a tiny transform. Re-fitting on every
        # resize (cheap; only runs while actually resizing) is the standard
        # fix for QGraphicsView-based "fit to window" viewers.
        super().resizeEvent(event)
        if self._fit_mode and self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._emit_zoom()


class LoupeDialog(QDialog):
    """Full-screen-style zoom review for a filtered set of images."""

    def __init__(
        self,
        *,
        service: ReviewService,
        image_paths: list[str],
        start_index: int = 0,
        items: list[ImageItem] | None = None,
        show_boxes: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        self.service = service
        self.image_paths = image_paths
        self.items = items
        self.index = max(0, min(start_index, len(image_paths) - 1))
        self._exposure_steps = 0
        self._current_raw_pixmap: QPixmap | None = None
        # Starts synced with the gallery's own Detector Boxes toggle (see
        # MainWindow._open_loupe); toggling it here only affects this Loupe
        # session; matches the web review UI where boxes is one shared
        # client-side flag covering both the grid and the Lightbox.
        self._show_boxes = show_boxes
        self.setWindowTitle("PeakPic - Loupe")
        # QDialog hides the maximize/minimize buttons by default on some
        # platforms (Windows in particular), and the default 1280x860 size
        # read as "too small" on a large monitor - both fixed by explicitly
        # requesting resize/maximize window hints and opening maximized.
        # The user can still resize/restore-down normally from there.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(1280, 860)
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Only queried when the caller doesn't already have ImageItems on
        # hand (see MainWindow._open_loupe) - built once, then patched
        # locally on decisions, never re-fetched per navigation.
        self._info_by_path: dict[str, dict] = {}
        if self.items is None:
            try:
                state = self.service.load_session()
                self._info_by_path = {
                    self._resolve_path(img["image_path"]): img for img in state.get("images", []) if img.get("image_path")
                }
            except Exception:  # noqa: BLE001 - info display must not block review
                self._info_by_path = {}

        self._view = _ZoomView(self)
        self._view.navigateRequested.connect(self._on_wheel_navigate)
        self._view.zoomChanged.connect(self._on_zoom_changed)

        self._reason_combo = QComboBox(self)
        for value, label in REASON_LABELS.items():
            self._reason_combo.addItem(label, value)
        self._reason_combo.currentIndexChanged.connect(self._on_reason_changed)

        self._reason_note = QLineEdit(self)
        self._reason_note.setPlaceholderText("Note (Other)")
        self._reason_note.setVisible(False)
        self._reason_note.setMaximumWidth(160)

        self._counter_label = QLabel(self)
        self._name_label = QLabel(self)
        self._score_label = QLabel(self)
        self._status_label = QLabel(self)
        self._ai_badge_label = QLabel(self)
        self._zoom_label = QLabel("Fit", self)
        self._exposure_label = QLabel("+0.0 EV", self)

        prev_btn = QPushButton("< Prev", self)
        next_btn = QPushButton("Next >", self)
        keep_btn = QPushButton("Keep (K)", self)
        reject_btn = QPushButton("Reject (R)", self)
        neutral_btn = QPushButton("Neutral (N)", self)
        exp_down_btn = QPushButton("−", self)
        exp_up_btn = QPushButton("+", self)
        self._boxes_btn = QPushButton("Boxes", self)
        self._boxes_btn.setCheckable(True)
        self._boxes_btn.setChecked(self._show_boxes)
        self._boxes_btn.setToolTip("Show the AI's detected-subject bounding boxes on this image")
        save_jpeg_btn = QPushButton("Save JPEG", self)
        close_btn = QPushButton("Close", self)
        for button in (exp_down_btn, exp_up_btn):
            button.setMaximumWidth(28)

        prev_btn.clicked.connect(self._go_prev)
        next_btn.clicked.connect(lambda: self._go_next())
        keep_btn.clicked.connect(lambda: self._apply_status("keep"))
        reject_btn.clicked.connect(lambda: self._apply_status("reject"))
        neutral_btn.clicked.connect(lambda: self._apply_status("neutral"))
        exp_down_btn.clicked.connect(lambda: self._adjust_exposure(-1))
        exp_up_btn.clicked.connect(lambda: self._adjust_exposure(1))
        self._boxes_btn.toggled.connect(self._on_boxes_toggled)
        save_jpeg_btn.clicked.connect(self._save_jpeg)
        close_btn.clicked.connect(self.accept)

        bottom_bar = QWidget(self)
        bottom_bar.setObjectName("loupeBottomBar")
        bottom_bar.setStyleSheet(
            f"#loupeBottomBar {{ background-color: {theme.DARK.panel_bg}; }}"
            f"#loupeBottomBar QLabel {{ color: {theme.DARK.text_primary}; }}"
            "#loupeBottomBar QPushButton { padding: 4px 10px; }"
        )
        bar_layout = QHBoxLayout(bottom_bar)
        bar_layout.setContentsMargins(10, 6, 10, 6)
        bar_layout.setSpacing(10)
        bar_layout.addWidget(self._counter_label)
        bar_layout.addWidget(self._name_label)
        bar_layout.addWidget(self._score_label)
        bar_layout.addWidget(self._status_label)
        bar_layout.addWidget(self._ai_badge_label)
        bar_layout.addStretch(1)
        bar_layout.addWidget(self._zoom_label)
        bar_layout.addWidget(exp_down_btn)
        bar_layout.addWidget(self._exposure_label)
        bar_layout.addWidget(exp_up_btn)
        bar_layout.addWidget(self._boxes_btn)
        bar_layout.addWidget(save_jpeg_btn)
        bar_layout.addWidget(QLabel("Reason:"))
        bar_layout.addWidget(self._reason_combo)
        bar_layout.addWidget(self._reason_note)
        bar_layout.addWidget(prev_btn)
        bar_layout.addWidget(keep_btn)
        bar_layout.addWidget(reject_btn)
        bar_layout.addWidget(neutral_btn)
        bar_layout.addWidget(next_btn)
        bar_layout.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, 1)
        layout.addWidget(bottom_bar)

        self._load_current()

    @staticmethod
    def _resolve_path(path: str) -> str:
        try:
            return str(Path(path).resolve())
        except OSError:
            return path

    def _on_reason_changed(self) -> None:
        self._reason_note.setVisible(self._reason_combo.currentData() == REVIEW_REASON_OTHER)

    def _current_path(self) -> str:
        return self.image_paths[self.index]

    def _current_item_info(self) -> dict:
        if self.items is not None:
            item = self.items[self.index]
            return {
                "score": item.score,
                "rank": item.rank,
                "review_status": item.review_status,
                "ai_suggestion": item.ai_suggestion,
            }
        return self._info_by_path.get(self._resolve_path(self._current_path()), {})

    def _load_current(self) -> None:
        path = self._current_path()
        try:
            preview = self.service.preview_path(path)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not crash the loupe
            QMessageBox.warning(self, "PeakPic - Loupe", f"Could not load preview:\n{exc}")
            return
        pixmap = QPixmap(str(preview))
        self._current_raw_pixmap = pixmap
        self._view.set_pixmap(self._exposed_pixmap())
        self._refresh_detection_overlay()
        self._update_info_labels()

    def _refresh_detection_overlay(self) -> None:
        if not self._show_boxes:
            self._view.clear_detection_overlay()
            return
        try:
            boxes = self.service.detection_boxes(self._current_path())
        except Exception:  # noqa: BLE001 - a missing/unreadable detection record must not break the loupe
            boxes = None
        self._view.set_detection_overlay(boxes)

    def _on_boxes_toggled(self, checked: bool) -> None:
        self._show_boxes = checked
        self._refresh_detection_overlay()

    def _exposed_pixmap(self) -> QPixmap:
        if self._current_raw_pixmap is None:
            return QPixmap()
        if self._exposure_steps == 0:
            return self._current_raw_pixmap
        return _apply_brightness(self._current_raw_pixmap, self._exposure_steps * EXPOSURE_STEP)

    def _update_info_labels(self) -> None:
        path = self._current_path()
        info = self._current_item_info()

        self._counter_label.setText(f"{self.index + 1} / {len(self.image_paths)}")
        self._name_label.setText(Path(path).name)

        score = info.get("score")
        rank = info.get("rank")
        parts = []
        if score is not None:
            parts.append(f"Score {score:.4f}")
        if rank is not None:
            parts.append(f"Rank {rank}")
        self._score_label.setText(" · ".join(parts) if parts else "Unranked")

        status = info.get("review_status") or "neutral"
        self._status_label.setText(STATUS_LABELS.get(status, status.capitalize()))
        color = STATUS_COLORS.get(status, STATUS_COLORS["neutral"])
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

        ai_suggestion = info.get("ai_suggestion")
        if ai_suggestion and ai_suggestion != status:
            self._ai_badge_label.setText(f"AI suggests {STATUS_LABELS.get(ai_suggestion, ai_suggestion)}")
            self._ai_badge_label.setStyleSheet(f"color: {theme.DARK.accent};")
        else:
            self._ai_badge_label.setText("")

    def _on_zoom_changed(self, scale: float) -> None:
        self._zoom_label.setText("Fit" if self._view._fit_mode else f"{scale * 100:.0f}%")  # noqa: SLF001

    def _adjust_exposure(self, direction: int) -> None:
        self._exposure_steps = max(EXPOSURE_MIN_STEPS, min(EXPOSURE_MAX_STEPS, self._exposure_steps + direction))
        ev = self._exposure_steps * EXPOSURE_STEP
        self._exposure_label.setText(f"{'+' if ev >= 0 else ''}{ev:.1f} EV")
        self._view.update_pixmap(self._exposed_pixmap())

    def _apply_status(self, status: str) -> None:
        reason = self._reason_combo.currentData() if status != "neutral" else None
        note = self._reason_note.text().strip() or None if reason == REVIEW_REASON_OTHER else None
        try:
            self.service.set_review_status(self._current_path(), status, reason=reason, reason_note=note)
        except Exception as exc:  # noqa: BLE001 - surfaced to the photographer, not fatal
            QMessageBox.warning(self, "PeakPic - Loupe", f"Could not save decision:\n{exc}")
            return
        if self.items is not None:
            self.items[self.index].review_status = status
        else:
            key = self._resolve_path(self._current_path())
            if key in self._info_by_path:
                self._info_by_path[key]["review_status"] = status
        self._go_next(auto=True)

    def _on_wheel_navigate(self, direction: int) -> None:
        if direction > 0:
            self._go_next()
        else:
            self._go_prev()

    def _go_next(self, *, auto: bool = False) -> None:
        if self.index + 1 < len(self.image_paths):
            self.index += 1
            self._load_current()
        elif auto:
            self.accept()

    def _go_prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._load_current()

    def _save_jpeg(self) -> None:
        path = self._current_path()
        suggested = Path(path).with_suffix(".jpg").name
        destination, _ = QFileDialog.getSaveFileName(self, "Save as JPEG", suggested, "JPEG Images (*.jpg)")
        if not destination:
            return
        try:
            self.service.save_jpeg(path, destination)
        except Exception as exc:  # noqa: BLE001 - surfaced to the photographer, not fatal
            QMessageBox.warning(self, "PeakPic - Save as JPEG", f"Could not save JPEG:\n{exc}")
            return
        QMessageBox.information(self, "PeakPic - Save as JPEG", f"Saved to {destination}")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override signature
        key = event.key()
        if key == Qt.Key.Key_K:
            self._apply_status("keep")
        elif key == Qt.Key.Key_R:
            self._apply_status("reject")
        elif key == Qt.Key.Key_N:
            self._apply_status("neutral")
        elif key == Qt.Key.Key_Right:
            self._go_next()
        elif key == Qt.Key.Key_Left:
            self._go_prev()
        elif key == Qt.Key.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)
