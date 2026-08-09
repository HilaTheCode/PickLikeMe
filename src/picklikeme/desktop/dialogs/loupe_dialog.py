"""Loupe (zoom) review dialog - fast per-image Keep/Reject/Neutral triage.

Mirrors the browser Lightbox's core workflow: zoom/pan the current image,
apply a Keep/Reject/Neutral decision with an optional reason, move to the
next image in the filtered set, and optionally save a quick JPEG for
sharing before any RAW development happens.

The image viewer fills nearly the entire dialog; a single dark overlay bar
at the bottom carries every control and status readout, matching the web
Lightbox's information density instead of splitting controls across a
top bar and a control row.

Burst member order (`burst_scoped=True`, opened from a collapsed burst
card): this dialog does NOT choose or change it. `items`/`image_paths` are
navigated in EXACTLY the order the caller (MainWindow._open_loupe_for_item)
hands in, and that order never changes for the life of the dialog - no
in-Loupe re-sorting, no sort-mode combo. This is a deliberate
simplification: the Loupe previously carried its own Capture Time / Burst
Score toggle and re-sorted itself on every change, which is what caused a
string of Capture-Time/Score synchronization bugs (a mode change not
actually reordering navigation, or one mode silently breaking the other).
Burst order is now decided in exactly one place - see main_window.py's own
BURST_SORT_* and the View menu's "Burst Order" submenu - so there is
nothing left inside the Loupe that could disagree with it. To change burst
order: close the Loupe, change Burst Order in the Main Grid, reopen it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
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
from ...analyzer.contactsheets import EYE_BOX_ACCEPTED, EYE_BOX_REJECTED, OTHER_BOX, SELECTED_BOX
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

# Persistent tint on the Keep/Reject/Neutral buttons themselves, so the
# action each button takes is color-coded even before a decision is made -
# not just the eventual status readout. The active status additionally gets
# a bright border (_ACTIVE_BUTTON_BORDER) so "what did I already pick for
# this image" is answerable at a glance, matching the image-viewer border
# below.
_STATUS_BUTTON_STYLES = {
    "keep": f"background-color: {theme.DARK.keep_bg}; color: {theme.DARK.keep_fg}; font-weight: 600;",
    "reject": f"background-color: {theme.DARK.reject_bg}; color: {theme.DARK.reject_fg}; font-weight: 600;",
    "neutral": f"background-color: {theme.DARK.neutral_bg}; color: {theme.DARK.neutral_fg}; font-weight: 600;",
}
_ACTIVE_BUTTON_BORDER = "border: 2px solid #ffffff;"

# A thick border around the whole image viewer color-coded to the current
# review status - the "can't miss it" signal the status label alone
# (STATUS_COLORS, just colored text) doesn't provide at a glance across a
# large maximized window. Neutral gets no border rather than a third color,
# so an undecided image doesn't read as "has a status" at a glance.
_VIEW_BORDER_COLORS = {"keep": theme.DARK.keep_fg, "reject": theme.DARK.reject_fg}
_VIEW_BORDER_WIDTH = 8


# Caps on the bottom bar's own unbounded-text labels - neither grows with a
# fixed bound built into its own content:
#   - Color Source: <strategy display name> - a ranking strategy's display
#     name is not length-limited (see ranking.classic's "Classic Vision
#     Ranking (EyePose-v0, recommended)").
#   - the score bar (_scores_text) - concatenates EVERY analysis module's
#     score onto one line, so it grows with however many strategies have
#     scored the current image (2 today; nothing caps a 3rd/4th/5th module
#     from being added later).
# This bottom bar has no scroll/wrap, so either one left unbounded is
# exactly what was pushing the Loupe wider than a MacBook screen (measured:
# with both AI Model and Classic Vision (EyePose) scores present, the bottom
# bar's minimum width was 1607px against a 1470px-wide screen) - Qt then
# compresses the whole layout below its own minimum size hint, visibly
# clipping button text elsewhere in the same bar. Matches the toolbar
# combo-box fix's pattern: cap the width, elide, keep the full text
# reachable via tooltip.
_BURST_INFO_LABEL_MAX_WIDTH = 140
_SCORE_LABEL_MAX_WIDTH = 140


def _set_elided_text(label: QLabel, text: str, max_width: int, *, tooltip: str | None = None) -> None:
    """`tooltip` defaults to the full (un-elided) `text`; pass an explicit
    value to combine it with other detail (see _score_label's diagnostics
    breakdown in _update_info_labels) instead of losing that tooltip."""
    label.setMaximumWidth(max_width)
    metrics = QFontMetrics(label.font())
    label.setText(metrics.elidedText(text, Qt.TextElideMode.ElideRight, max_width))
    label.setToolTip(text if tooltip is None else tooltip)


def _format_burst_label(burst_id: str) -> str:
    """"burst-0018" (burst_analysis.analyze_bursts's own id format) -> "Burst
    18" - the bare, human-facing number the photographer actually cares
    about. Falls back to the raw id verbatim for anything that doesn't
    follow that pattern, so an unexpected id still displays instead of
    raising."""
    suffix = burst_id.removeprefix("burst-") if burst_id else burst_id
    try:
        return f"Burst {int(suffix)}"
    except ValueError:
        return f"Burst {burst_id}"


def _strategy_label(strategy_id: str | None) -> str:
    """"ai-model" -> "AI Model" - the same registry `_scores_text` already
    reads from, so the Loupe's own "Color Source: ..." readout never
    invents a second naming scheme for a strategy the score bar already
    names. Falls back to the raw id for one the registry does not
    recognise (or None) rather than raising - a label is never worth
    crashing the Loupe over."""
    if not strategy_id:
        return "(none)"
    from ...ranking import score_labels

    return score_labels().get(strategy_id, strategy_id)


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


# Detector-box overlay pen widths - ~5x the Loupe's original (4/3px), the
# same "the overlay is our primary debugging tool" motivation that
# multiplied contactsheets.annotate_thumbnail's own `line` by ~5x, applied
# here to the Loupe's separate vector overlay (a QGraphicsScene, drawn from
# the same detection data but never sharing code with the Gallery's raster
# one - so it needed its own, matching change). Eye keypoint markers
# (_add_eye_overlay's ellipses) are deliberately NOT scaled with these: they
# are points, not boxes, and were already clearly visible.
BOX_PEN_WIDTH_OTHER = 15
BOX_PEN_WIDTH_SELECTED = 20
BOX_PEN_WIDTH_EYE = 20


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
        # QGraphicsView (via QAbstractScrollArea) accepts keyboard focus by
        # default and has its own Left/Right/arrow-key handling (scrolling
        # the viewport) - since this is the dialog's main/central widget, it
        # is also the widget Qt hands initial focus to on show(), which
        # meant Left/Right/K/R/N/Escape were silently swallowed here instead
        # of ever reaching LoupeDialog.keyPressEvent (verified: sending
        # Key_Right to a focused _ZoomView left LoupeDialog.index
        # unchanged). Every keyboard shortcut in this dialog is meant to work
        # regardless of which child widget was last clicked, and the view
        # itself has no legitimate use for its own arrow-key scrolling
        # (fit/zoom/pan are all mouse-driven - see the class docstring), so
        # it simply never takes focus; with nothing else claiming it either,
        # Qt delivers key events straight to the dialog.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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

    def set_detection_overlay(self, boxes_data: dict | None, eye_data: dict | None = None) -> None:
        """Draw the detector's boxes (review/thumbnails.detection_boxes_for)
        and, when available, the eye detector's result
        (review/thumbnails.eye_keypoints_for) over the current image - solid
        green for the box that became the crop the model scored, dashed amber
        for runners-up it passed over, and magenta for the eye Classic Vision
        measured (solid when accepted, dashed when detected but distrusted -
        see `eyes.detector.EyeDetection.accepted`). Same colors as the
        gallery's own with_boxes thumbnail overlay
        (contactsheets.SELECTED_BOX/OTHER_BOX/EYE_BOX_ACCEPTED/EYE_BOX_REJECTED)
        so both views agree.

        Coordinates in both dicts are full-frame source pixels; scaled here
        against the currently-displayed pixmap's own size rather than
        assumed to match 1:1, since nothing guarantees the preview was
        never resized relative to the source frame.
        """
        self.clear_detection_overlay()
        if self._pixmap_item is None:
            return
        source_size = (boxes_data or {}).get("source_size") or (eye_data or {}).get("source_size")
        if not source_size or source_size[0] <= 0 or source_size[1] <= 0:
            return
        pixmap_size = self._pixmap_item.pixmap().size()
        scale_x = pixmap_size.width() / source_size[0]
        scale_y = pixmap_size.height() / source_size[1]

        def to_scene_rect(box) -> QRectF:
            x1, y1, x2, y2 = box
            return QRectF(x1 * scale_x, y1 * scale_y, (x2 - x1) * scale_x, (y2 - y1) * scale_y)

        if boxes_data is not None:
            for box in boxes_data.get("others", []):
                item = self._scene.addRect(
                    to_scene_rect(box["box"]), QPen(QColor(*OTHER_BOX), BOX_PEN_WIDTH_OTHER, Qt.PenStyle.DashLine)
                )
                item.setZValue(10)
                self._overlay_items.append(item)

            selected = boxes_data.get("selected")
            if selected is not None:
                item = self._scene.addRect(
                    to_scene_rect(selected["box"]), QPen(QColor(*SELECTED_BOX), BOX_PEN_WIDTH_SELECTED)
                )
                item.setZValue(11)
                self._overlay_items.append(item)

        if eye_data is not None:
            self._add_eye_overlay(eye_data, to_scene_rect, scale_x, scale_y)

    def _add_eye_overlay(self, eye_data: dict, to_scene_rect, scale_x: float, scale_y: float) -> None:
        """The eye box plus both raw left/right keypoints - the debugging
        signal for "why did an image with no visible eye still get scored":
        a dashed box or widely separated crosshairs means the eye detector
        found something the acceptance gate should have caught (or did, and
        a different metric carried the ranking anyway)."""
        colour = QColor(*(EYE_BOX_ACCEPTED if eye_data.get("accepted") else EYE_BOX_REJECTED))
        pen = (
            QPen(colour, BOX_PEN_WIDTH_EYE)
            if eye_data.get("accepted")
            else QPen(colour, BOX_PEN_WIDTH_EYE, Qt.PenStyle.DashLine)
        )
        item = self._scene.addRect(to_scene_rect(eye_data["box"]), pen)
        item.setZValue(12)
        self._overlay_items.append(item)

        radius = 6
        for keypoint in (eye_data.get("left"), eye_data.get("right")):
            if keypoint is None:
                continue
            kx = keypoint["x"] * scale_x
            ky = keypoint["y"] * scale_y
            marker = self._scene.addEllipse(QRectF(kx - radius, ky - radius, radius * 2, radius * 2), QPen(colour, 3))
            marker.setZValue(12)
            self._overlay_items.append(marker)

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
        burst_scoped: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        self.service = service
        # Navigated in exactly this order for the life of the dialog - see
        # the module docstring's "Burst member order" section. Never
        # re-sorted, re-derived, or mutated in place.
        self.image_paths = image_paths
        self.items = items
        self.index = max(0, min(start_index, len(image_paths) - 1))
        # Set by MainWindow._open_loupe_for_item only when this Loupe session
        # was opened from a single collapsed burst card (Collapse Bursts on) -
        # never inferred from `items` itself, since a normal, non-burst Loupe
        # session's items still carry burst_id/burst_rank (every image is
        # "a burst of one" - see ImageItem's own docstring) and must NOT show
        # burst info UI that would only ever read "Burst Rank #1 of 1".
        self._burst_scoped = burst_scoped and self.items is not None
        # The strategy Burst Analysis is currently keyed to (the gallery's
        # Color Source selector - see ReviewSession.burst_strategy) - and
        # whether it has actually scored ANY member of this burst. When it
        # hasn't (photographer ranked with a different strategy, or hasn't
        # ranked at all), burst_rank/Score carry no real signal - purely
        # informational here (see _burst_ranking_status_label below); which
        # order was ACTUALLY used to navigate is decided upstream by
        # MainWindow before this dialog is ever constructed.
        self._burst_color_source = getattr(self.service.session, "burst_strategy", None) if self._burst_scoped else None
        self._burst_score_available = (
            self.items is not None and self._burst_color_source is not None
            and any(item.score_for(self._burst_color_source) is not None for item in self.items)
        )
        self._exposure_steps = 0
        self._current_raw_pixmap: QPixmap | None = None
        # Starts synced with the gallery's own Detector Boxes toggle (see
        # MainWindow._open_loupe); toggling it here only affects this Loupe
        # session; matches the web review UI where boxes is one shared
        # client-side flag covering both the grid and the Lightbox.
        self._show_boxes = show_boxes
        self.setWindowTitle("PeakPic - Loupe")
        # QDialog hides the maximize/minimize buttons by default on some
        # platforms (Windows in particular), and a fixed default size reads
        # as "too small" on a large monitor and can overflow a small one -
        # both fixed by explicitly requesting resize/maximize window hints
        # and opening maximized. The user can still resize/restore-down
        # normally from there.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        # setWindowState(WindowMaximized) alone should be enough - Qt is
        # supposed to resize the native window to the screen's own
        # availableGeometry() - but that native "maximize" is a window-
        # manager-level request, so its correctness isn't guaranteed to be
        # identical everywhere the app runs, and the actual reported bug was
        # the window not fitting a specific (Mac) screen despite this call
        # already being present. Also explicitly computing and applying the
        # target screen's availableGeometry() directly removes any
        # dependency on that platform-level request actually landing:
        # whichever screen this dialog is about to appear on (its parent's
        # screen if it has one, else wherever Qt would otherwise place it),
        # not a hardcoded resolution - "the MacBook" vs. "a Windows desktop"
        # never has to be special-cased, and a future different Mac display
        # resolution is handled the same way.
        screen = self.screen()
        if screen is not None:
            self.setGeometry(screen.availableGeometry())
        else:  # pragma: no cover - no screen at all is not a real desktop session
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

        # Read-only info - see the module docstring: the Loupe never offers
        # a way to change burst order, only to see what it currently is and
        # whether Score means anything for this particular burst. To change
        # it: close the Loupe, change Burst Order in the Main Grid's View
        # menu, reopen.
        self._burst_id_label = QLabel(self)
        self._burst_rank_label = QLabel(self)
        self._burst_best_label = QLabel(self)
        self._burst_score_label = QLabel(self)
        self._burst_color_source_label = QLabel(self)
        self._burst_ranking_status_label = QLabel(self)
        if self._burst_scoped:
            _set_elided_text(
                self._burst_color_source_label, f"Color Source: {_strategy_label(self._burst_color_source)}",
                _BURST_INFO_LABEL_MAX_WIDTH,
            )
            if self._burst_score_available:
                self._burst_ranking_status_label.setText("Ranking: Available")
                self._burst_ranking_status_label.setStyleSheet(f"color: {theme.DARK.keep_fg};")
            else:
                self._burst_ranking_status_label.setText("Ranking: Not available")
                self._burst_ranking_status_label.setStyleSheet(f"color: {theme.DARK.reject_fg};")
                self._burst_ranking_status_label.setToolTip(
                    "This burst has no score from the current Color Source "
                    f"({_strategy_label(self._burst_color_source)}) - Burst Rank/Score above are not "
                    "meaningful for it. Rank this folder with that strategy, or switch Color Source to "
                    "one that already has, then reopen the Loupe."
                )

        prev_btn = QPushButton("< Prev", self)
        next_btn = QPushButton("Next >", self)
        keep_btn = QPushButton("Keep (K)", self)
        reject_btn = QPushButton("Reject (R)", self)
        neutral_btn = QPushButton("Neutral (N)", self)
        self._status_buttons = {"keep": keep_btn, "reject": reject_btn, "neutral": neutral_btn}
        for name, btn in self._status_buttons.items():
            btn.setStyleSheet(_STATUS_BUTTON_STYLES[name])
        exp_down_btn = QPushButton("−", self)
        exp_up_btn = QPushButton("+", self)
        self._boxes_btn = QPushButton("Boxes", self)
        self._boxes_btn.setCheckable(True)
        self._boxes_btn.setChecked(self._show_boxes)
        self._boxes_btn.setToolTip(
            "Show the AI's detected-subject bounding box and, when Classic Vision has run, "
            "the eye it measured, on this image"
        )
        save_jpeg_btn = QPushButton("Save JPEG", self)
        close_btn = QPushButton("Close", self)
        for button in (exp_down_btn, exp_up_btn):
            button.setMaximumWidth(28)
        # None of these buttons has any legitimate use for arrow keys, and a
        # FOCUSED QPushButton silently swallows Key_Left/Key_Right under this
        # app's Fusion style (verified: a bare QPushButton with focus never
        # even reaches its parent QDialog's keyPressEvent) - Qt hands the
        # very first focusable widget in the dialog initial focus on show(),
        # and after that, clicking Keep/Reject/Neutral (the primary review
        # action, done constantly) leaves THAT button focused. Either way,
        # arrow-key navigation would silently stop working the moment any of
        # these had focus. Same fix and same reasoning as _ZoomView's own
        # NoFocus (see that class) - mouse clicks still work identically,
        # since QPushButton.clicked isn't gated on focus policy.
        for button in (
            prev_btn, next_btn, keep_btn, reject_btn, neutral_btn,
            exp_down_btn, exp_up_btn, self._boxes_btn, save_jpeg_btn, close_btn,
        ):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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
        # Three-column grid (stretch 1 : 0 : 1) rather than a single row of
        # widgets - a plain QHBoxLayout can only pin content to the edges or
        # to one side of a single stretch, it can't center a group. Equal
        # stretch on the outer columns keeps the center column (the core
        # Prev/Keep/Reject/Neutral/Next workflow) visually centered in the
        # bar regardless of how wide the info/tools columns end up.
        bar_layout = QGridLayout(bottom_bar)
        bar_layout.setContentsMargins(10, 6, 10, 6)
        bar_layout.setHorizontalSpacing(10)
        bar_layout.setColumnStretch(0, 1)
        bar_layout.setColumnStretch(1, 0)
        bar_layout.setColumnStretch(2, 1)

        info_group = QWidget(bottom_bar)
        info_layout = QHBoxLayout(info_group)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(10)
        info_layout.addWidget(self._counter_label)
        info_layout.addWidget(self._name_label)
        info_layout.addWidget(self._score_label)
        info_layout.addWidget(self._status_label)
        info_layout.addWidget(self._ai_badge_label)

        actions_group = QWidget(bottom_bar)
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(10)
        actions_layout.addWidget(QLabel("Reason:"))
        actions_layout.addWidget(self._reason_combo)
        actions_layout.addWidget(self._reason_note)
        actions_layout.addWidget(prev_btn)
        actions_layout.addWidget(keep_btn)
        actions_layout.addWidget(reject_btn)
        actions_layout.addWidget(neutral_btn)
        actions_layout.addWidget(next_btn)

        tools_group = QWidget(bottom_bar)
        tools_layout = QHBoxLayout(tools_group)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(10)
        tools_layout.addWidget(self._zoom_label)
        tools_layout.addWidget(exp_down_btn)
        tools_layout.addWidget(self._exposure_label)
        tools_layout.addWidget(exp_up_btn)
        tools_layout.addWidget(self._boxes_btn)
        tools_layout.addWidget(save_jpeg_btn)
        tools_layout.addWidget(close_btn)

        bar_layout.addWidget(info_group, 0, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        bar_layout.addWidget(actions_group, 0, 1, Qt.AlignmentFlag.AlignCenter)
        bar_layout.addWidget(tools_group, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        if self._burst_scoped:
            # A second bar row, only added for a burst-scoped session - read-
            # only "Burst 18 / Burst Rank #2 of 7 / Best Image: Yes / Score
            # 94.8" info (see the module docstring: no sort control here),
            # kept separate from info_group above so a non-burst Loupe
            # session's bar is byte-for-byte unchanged.
            burst_group = QWidget(bottom_bar)
            burst_layout = QHBoxLayout(burst_group)
            burst_layout.setContentsMargins(0, 0, 0, 0)
            burst_layout.setSpacing(10)
            burst_layout.addWidget(self._burst_id_label)
            burst_layout.addWidget(self._burst_rank_label)
            burst_layout.addWidget(self._burst_best_label)
            burst_layout.addWidget(self._burst_score_label)
            burst_layout.addWidget(self._burst_color_source_label)
            burst_layout.addWidget(self._burst_ranking_status_label)
            bar_layout.addWidget(burst_group, 1, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)

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
                "ranking_results": item.ranking_results,
                "filter_reasons": item.filter_reasons,
                "metrics": item.metrics,
                "review_status": item.review_status,
                "ai_suggestion": item.ai_suggestion,
            }
        return self._info_by_path.get(self._resolve_path(self._current_path()), {})

    @staticmethod
    def _scores_text(info: dict) -> str:
        """Every analysis module's score - or, for one that filtered this
        image instead of scoring it, why - on one bar line.

        Iterates whatever the session supplied rather than naming the modules
        that exist today, so a new one appears here for free. Labels come from
        the ranking registry, falling back to the raw strategy id so results
        from a module PeakPic no longer ships are still shown.
        """
        from ...ranking import score_labels
        from ...ranking.filters import REJECT_REASON_LABELS

        scores = info.get("ranking_results") or {}
        filter_reasons = info.get("filter_reasons") or {}
        labels = score_labels()
        strategy_ids = set(scores) | set(filter_reasons)
        parts = []
        for strategy_id in sorted(strategy_ids, key=lambda sid: (sid != "ai-model", sid)):
            entry = scores.get(strategy_id) or {}
            score = entry.get("score")
            label = labels.get(strategy_id, strategy_id)
            if score is not None:
                rank = entry.get("rank")
                parts.append(f"{label} {score:.4f}" + (f" (#{rank})" if rank else ""))
            else:
                reason = filter_reasons.get(strategy_id)
                if reason is None:
                    continue
                parts.append(f"{label}: {REJECT_REASON_LABELS.get(reason, reason)}")
        return "   ".join(parts) if parts else "Unranked"

    @staticmethod
    def _diagnostics_text(info: dict) -> str:
        """The raw measurements behind each strategy's combined score (see
        `ranking.classic.write_metrics_report`) - "why did this image rank
        where it did", not just the single weighted-sum number the score
        line shows. Shown as the score label's tooltip rather than more bar
        real estate, since it is a hover-for-detail diagnostic, not a
        first-glance one.

        Empty (no tooltip) for an image no module wrote a metrics report
        for - most images, most of the time, since only Classic Vision
        writes one today.
        """
        from ...ranking import metric_labels, score_labels

        metrics = info.get("metrics") or {}
        if not metrics:
            return ""
        strategy_labels = score_labels()
        names_by_strategy = metric_labels()
        lines = []
        for strategy_id in sorted(metrics, key=lambda sid: (sid != "ai-model", sid)):
            values = metrics.get(strategy_id) or {}
            if not values:
                continue
            names = names_by_strategy.get(strategy_id, {})
            breakdown = "  ".join(f"{names.get(name, name)}: {value:.3f}" for name, value in values.items())
            lines.append(f"{strategy_labels.get(strategy_id, strategy_id)} - {breakdown}")
        return "\n".join(lines)

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
        try:
            eye = self.service.eye_keypoints(self._current_path())
        except Exception:  # noqa: BLE001 - a missing/unreadable eye record must not break the loupe
            eye = None
        self._view.set_detection_overlay(boxes, eye)

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

        self._counter_label.setText(f"Image {self.index + 1} of {len(self.image_paths)}")
        self._name_label.setText(Path(path).name)
        self._update_burst_info_labels()

        full_scores_text = self._scores_text(info)
        diagnostics_text = self._diagnostics_text(info)
        tooltip = f"{full_scores_text}\n\n{diagnostics_text}" if diagnostics_text else full_scores_text
        _set_elided_text(self._score_label, full_scores_text, _SCORE_LABEL_MAX_WIDTH, tooltip=tooltip)

        status = info.get("review_status") or "neutral"
        self._status_label.setText(STATUS_LABELS.get(status, status.capitalize()))
        color = STATUS_COLORS.get(status, STATUS_COLORS["neutral"])
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self._update_status_indicators(status)

        ai_suggestion = info.get("ai_suggestion")
        if ai_suggestion and ai_suggestion != status:
            self._ai_badge_label.setText(f"AI suggests {STATUS_LABELS.get(ai_suggestion, ai_suggestion)}")
            self._ai_badge_label.setStyleSheet(f"color: {theme.DARK.accent};")
        else:
            self._ai_badge_label.setText("")

    def _update_burst_info_labels(self) -> None:
        """Burst ID / Burst Rank #X of N / Best Image / Score - only
        populated for a burst-scoped session (see `_burst_scoped`); the
        labels stay empty text otherwise so a non-burst Loupe session shows
        nothing where this row would be."""
        if not self._burst_scoped or self.items is None:
            return
        item = self.items[self.index]
        self._burst_id_label.setText(_format_burst_label(item.burst_id) if item.burst_id else "")
        self._burst_rank_label.setText(f"Burst Rank #{item.burst_rank} of {item.burst_size}")
        self._burst_best_label.setText(f"Best Image: {'Yes' if item.burst_best else 'No'}")
        # The score that actually produced burst_rank - whichever ranking
        # strategy Burst Analysis is currently keyed to (ReviewSession.
        # burst_strategy), not necessarily the AI model. *100 to match a
        # "94.8"-style readout instead of the score bar's raw "0.9480".
        strategy_id = getattr(self.service.session, "burst_strategy", None)
        score = item.score_for(strategy_id) if strategy_id else None
        self._burst_score_label.setText(f"Score {score * 100:.1f}" if score is not None else "Score -")


    def _update_status_indicators(self, status: str) -> None:
        """Color-code the current review status two ways at once: a bright
        border around the whole image viewer (visible even at a glance
        across a maximized window) and a highlighted border on whichever of
        the Keep/Reject/Neutral buttons is currently active."""
        border_color = _VIEW_BORDER_COLORS.get(status)
        if border_color:
            self._view.setStyleSheet(f"border: {_VIEW_BORDER_WIDTH}px solid {border_color};")
        else:
            self._view.setStyleSheet("border: none;")
        for name, btn in self._status_buttons.items():
            style = _STATUS_BUTTON_STYLES[name]
            if name == status:
                style += _ACTIVE_BUTTON_BORDER
            btn.setStyleSheet(style)

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
