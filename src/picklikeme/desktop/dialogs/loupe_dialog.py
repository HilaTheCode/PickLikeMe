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

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
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

# The bottom bar's own visual language - two rows (see LoupeDialog's own
# __init__), Row 1 given a slightly lighter background and a bottom rule so
# it reads as "the glance-at info" distinct from Row 2's "the controls",
# without needing a heavier divider that would eat into the already-tight
# vertical space of a maximized-but-not-fullscreen window.
_BOTTOM_BAR_STYLESHEET = (
    f"#loupeBottomBar {{ background-color: {theme.DARK.panel_bg}; border-top: 1px solid {theme.DARK.border}; }}"
    f"#loupeBottomBar QLabel {{ color: {theme.DARK.text_primary}; }}"
    "#loupeBottomBar QPushButton { padding: 5px 12px; }"
    f"#loupeRow1 {{ background-color: {theme.DARK.hover_bg}; border-bottom: 1px solid {theme.DARK.border}; }}"
)
# "IMAGE SCORE: 0.384" - a filled, rounded badge (not just bold text) so it
# reads as the one number that matters most in the bar at a glance, matching
# the redesign's own "prominent image score" priority. accent (blue), not
# keep/reject green/red: this is a raw algorithm score, not a decision - see
# _get_background_color's own docstring on why those two concepts are kept
# visually separate everywhere else in the app.
_PRIMARY_SCORE_LABEL_STYLE = (
    f"background-color: {theme.DARK.accent}; color: {theme.DARK.window_bg}; "
    "font-weight: 700; font-size: 15px; padding: 6px 16px; border-radius: 6px;"
)
# Elements - an accent-colored border ONLY while active (Manual QA: the
# border must not show all the time - only the active visual state should
# be shown when Elements is actually on), via :checked rather than an
# unconditional border every other tool button lacks. font-weight stays
# unconditional (a checkable button's own label reads fine slightly
# bolder regardless of state, and it keeps the button's width stable
# between checked/unchecked rather than reflowing the bottom bar).
_ELEMENTS_BUTTON_STYLE = (
    f"QPushButton {{ font-weight: 600; }}"
    f"QPushButton:checked {{ border: 2px solid {theme.DARK.accent}; }}"
)


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
# Manual QA: Elements' own boxes (_add_element_box) are much smaller than a
# subject/eye detection box (they bound a single eye or head region, not a
# whole bird), so BOX_PEN_WIDTH_EYE's 20px reads as a solid block that
# obscures the exact location it exists to point at. Thin enough to trace
# the actual boundary at a normal zoom level while staying visible.
ELEMENT_BOX_PEN_WIDTH = 3

# Bounds on _ZoomView's manual scale (1.0 = 100%) - shared by Ctrl+wheel,
# trackpad pinch, and the keyboard +/- shortcuts (see _ZoomView.zoom_by).
# Purely a sanity clamp against "pinch/scroll past the point of being
# useful", not a meaningful product decision about a specific max quality.
MIN_MANUAL_SCALE = 0.1
MAX_MANUAL_SCALE = 8.0


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
        """Wipes every overlay item, Boxes' and Elements' alike - the one
        shared clear point, called once by `LoupeDialog.
        _refresh_detection_overlay` before drawing whichever of
        `set_detection_overlay`/`set_elements_overlay` are currently active.
        Neither of those two clears on its own (see their own docstrings),
        specifically so calling both in the same refresh composes rather
        than each erasing the other's work - Boxes and Elements are
        independently toggleable, not mutually exclusive."""
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

        Does NOT clear the overlay itself - see `clear_detection_overlay`'s
        own docstring for why: Boxes and Elements must be independently
        toggleable and drawable together, so the caller
        (`LoupeDialog._refresh_detection_overlay`) clears once, up front,
        then calls whichever of this/`set_elements_overlay` are currently
        active - never either of them clearing out what the other just drew.
        """
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

    def set_elements_overlay(self, eye_data: dict | None) -> None:
        """"Elements" mode (see LoupeDialog's Elements button) - Left Eye /
        Right Eye / Head, each its own labeled bounding rectangle plus a
        "Name — confidence" text label, for visually inspecting what the
        eye detector found and how confident it was. Shares
        `_overlay_items` with `set_detection_overlay` (Boxes), but neither
        clears it internally (see `clear_detection_overlay`'s own
        docstring) - the two are independently toggleable and compose when
        both are active, not mutually exclusive.

        Synthesized entirely from data the eye detector ALREADY computed
        and eye_keypoints_for already exposes - `left`/`right` (the two eye
        channels' own keypoints - see eyes.detector.EyeDetection's own
        docstring on why these are genuinely anatomical left/right, not
        "primary vs secondary") and `head_top` (one of EyePose-v0's six
        landmarks), paired with `head_confidence` (the holistic "is a head
        here" scalar, not head_top's own landmark-position confidence - see
        eye_keypoints_for's own docstring on why these are deliberately
        different numbers). No detection runs here - this only draws three
        already-known points as boxes. Box size is borrowed from the
        primary eye box's own dimensions (`eye_data["box"]`) - the one
        region size the detector already committed to for this image -
        rather than an arbitrary fixed pixel count that would look right on
        one image's resolution and wrong on another's; Head reuses that
        same size scaled up, since a head is larger than a single eye.
        """
        if self._pixmap_item is None or eye_data is None:
            return
        source_size = eye_data.get("source_size")
        if not source_size or source_size[0] <= 0 or source_size[1] <= 0:
            return
        pixmap_size = self._pixmap_item.pixmap().size()
        scale_x = pixmap_size.width() / source_size[0]
        scale_y = pixmap_size.height() / source_size[1]

        box = eye_data.get("box")
        if box:
            half_w = max(abs(box[2] - box[0]) / 2 * scale_x, 10.0)
            half_h = max(abs(box[3] - box[1]) / 2 * scale_y, 10.0)
        else:
            half_w = half_h = 20.0

        # Left Eye/Right Eye use their OWN keypoint's confidence (no
        # separate scalar exists for either side individually). Head is
        # NOT keypoint["confidence"] - deliberately always head_confidence
        # specifically, even when that is None (a backend/record with no
        # head_confidence at all), rather than silently falling back to
        # head_top's own landmark-position confidence, a different, weaker
        # claim (see eye_keypoints_for's own docstring on why these are two
        # different numbers) - a None here means Head is skipped below.
        elements = (
            ("Left Eye", eye_data.get("left"), half_w, half_h, QColor(0, 229, 255), "own"),
            ("Right Eye", eye_data.get("right"), half_w, half_h, QColor(255, 64, 255), "own"),
            ("Head", eye_data.get("head_top"), half_w * 2.5, half_h * 2.5, QColor(255, 213, 79), eye_data.get("head_confidence")),
        )
        for name, keypoint, half_width, half_height, colour, confidence_source in elements:
            if keypoint is None:
                continue
            confidence = keypoint.get("confidence") if confidence_source == "own" else confidence_source
            if confidence is None:
                continue
            self._add_element_box(name, keypoint, half_width, half_height, colour, confidence, scale_x, scale_y)

    def _add_element_box(
        self, name: str, keypoint: dict, half_width: float, half_height: float,
        colour: QColor, confidence: float, scale_x: float, scale_y: float,
    ) -> None:
        cx = keypoint["x"] * scale_x
        cy = keypoint["y"] * scale_y
        rect = QRectF(cx - half_width, cy - half_height, half_width * 2, half_height * 2)
        box_item = self._scene.addRect(rect, QPen(colour, ELEMENT_BOX_PEN_WIDTH))
        box_item.setZValue(13)
        self._overlay_items.append(box_item)

        label = QGraphicsSimpleTextItem(f"{name} — {confidence:.2f}")
        font = label.font()
        font.setBold(True)
        font.setPointSize(14)
        label.setFont(font)
        label.setBrush(QBrush(colour))
        label_pos_y = rect.top() - label.boundingRect().height() - 4
        label.setPos(rect.left(), label_pos_y)
        label.setZValue(15)

        # A dark backing rect behind the label - readable over any image
        # content, the same reason review_thumbnail's own text overlays get
        # one.
        backing_rect = label.boundingRect().translated(rect.left(), label_pos_y).adjusted(-3, -1, 3, 1)
        backing = self._scene.addRect(backing_rect, QPen(Qt.PenStyle.NoPen), QBrush(QColor(0, 0, 0, 170)))
        backing.setZValue(14)
        self._scene.addItem(label)
        self._overlay_items.append(backing)
        self._overlay_items.append(label)

    def _emit_zoom(self) -> None:
        self.zoomChanged.emit(self.transform().m11())

    def zoom_by(self, factor: float) -> None:
        """Scale by `factor` around whatever QGraphicsView's own
        AnchorUnderMouse/transformation anchor currently resolves to (the
        cursor for wheel/pinch, the view center when the mouse is
        elsewhere) - the single mechanism Ctrl+wheel, trackpad pinch
        (event(), below) and the keyboard +/- shortcuts (LoupeDialog.
        keyPressEvent) all route through, so "how zoom actually changes" has
        exactly one implementation. Clamped to MIN/MAX_MANUAL_SCALE so
        repeated pinch/scroll/key input can't zoom out past a speck or in
        past a useless blur. Leaves fit-to-window mode - a manual zoom
        level, once set, is what should persist across navigation (see
        set_pixmap), not silently snap back to Fit.
        """
        if self._pixmap_item is None:
            return
        current = self.transform().m11()
        target = max(MIN_MANUAL_SCALE, min(MAX_MANUAL_SCALE, current * factor))
        if abs(target - current) < 1e-9:
            return
        applied_factor = target / current
        self.scale(applied_factor, applied_factor)
        self._fit_mode = False
        self._manual_scale = target
        self._emit_zoom()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override signature
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
        else:
            direction = -1 if event.angleDelta().y() > 0 else 1
            self.navigateRequested.emit(direction)
        event.accept()

    def event(self, event) -> bool:  # noqa: N802 - Qt override signature
        # Trackpad pinch on macOS arrives as a native gesture event, not a
        # QPinchGesture (that's the cross-platform QGestureEvent API, which
        # needs an explicit grabGesture() registration and - per Qt's own
        # macOS platform notes - is NOT how trackpad pinch is actually
        # delivered there; QNativeGestureEvent/ZoomNativeGesture is the
        # documented, reliable path). `value()` is the incremental scale
        # delta for this event (e.g. 0.02 for a small pinch-out tick, not an
        # absolute zoom level), so it composes into a multiplicative factor
        # the same way one wheel-tick's fixed 1.15 does.
        if event.type() == QEvent.Type.NativeGesture:
            from PySide6.QtGui import QNativeGestureEvent

            if (
                isinstance(event, QNativeGestureEvent)
                and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture
            ):
                self.zoom_by(1.0 + event.value())
                return True
        return super().event(event)

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
        # Elements mode - see _on_elements_toggled/_refresh_detection_overlay.
        # Always starts off, unlike _show_boxes: there is no gallery-wide
        # "Elements" toggle to sync with (Boxes mirrors the gallery's own
        # Detector Boxes setting - see MainWindow._open_loupe - Elements is
        # Loupe-only), and it is the more detailed/heavier of the two
        # overlays, better opted into per session than carried over.
        self._show_elements = False
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
        # NoFocus, same reasoning as every button below: Qt hands whichever
        # widget was added first and accepts focus the dialog's INITIAL
        # focus on show() - before this fix, that was _reason_combo (the
        # first focusable widget added, since _view/the buttons are already
        # NoFocus), which silently swallowed Key_Equal/Key_Minus (the
        # keyboard zoom shortcuts) before LoupeDialog.keyPressEvent ever saw
        # them, verified empirically the same way the button fix was: a
        # focused QComboBox never even reached the dialog's own handler.
        # Still fully usable with the mouse (click to open, click an item) -
        # only its own keyboard type-ahead/arrow-key selection is gone,
        # which was never this dialog's point of "usable without a mouse"
        # (that's about image NAVIGATION - see the module docstring).
        self._reason_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._reason_note = QLineEdit(self)
        self._reason_note.setPlaceholderText("Note (Other)")
        self._reason_note.setVisible(False)
        self._reason_note.setMaximumWidth(160)
        # Deliberately NOT NoFocus, unlike everything else above - this is a
        # genuine free-text field for the "Other" reason, and a photographer
        # must be able to click into it and type normally (including
        # Left/Right to move the cursor, and literal "+"/"-" characters) -
        # the one place in this dialog where those keys correctly mean what
        # a text field always means, not a global shortcut.

        self._counter_label = QLabel(self)
        self._name_label = QLabel(self)
        self._status_label = QLabel(self)
        self._ai_badge_label = QLabel(self)
        # THE prominent score - see _primary_score/_update_info_labels.
        # Deliberately its own, visually distinct label rather than folded
        # into _score_label's multi-strategy line below: "IMAGE SCORE:
        # 0.384", exactly three decimals, normalized (never the old *100
        # "38.4" reading) - see _primary_score_text.
        self._primary_score_label = QLabel(self)
        self._primary_score_label.setObjectName("primaryScoreLabel")
        self._primary_score_label.setStyleSheet(_PRIMARY_SCORE_LABEL_STYLE)
        self._primary_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # The full "every module's own score" line - still useful, but
        # secondary/diagnostic detail now that one strategy's score has its
        # own prominent readout above - see the module's Row 1/Row 2 split.
        self._score_label = QLabel(self)
        self._zoom_label = QLabel("Fit", self)
        self._exposure_label = QLabel("+0.0 EV", self)

        # Read-only info - see the module docstring: the Loupe never offers
        # a way to change burst order, only to see what it currently is and
        # whether Score means anything for this particular burst. To change
        # it: close the Loupe, change Burst Order in the Main Grid's View
        # menu, reopen. Burst Rank/Best (the "key ranking information" a
        # burst session adds) live in Row 1 alongside the score; Color
        # Source/Ranking Status are secondary detail in Row 2.
        self._burst_id_label = QLabel(self)
        self._burst_rank_label = QLabel(self)
        self._burst_best_label = QLabel(self)
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
                    f"({_strategy_label(self._burst_color_source)}) - Burst Rank above is not "
                    "meaningful for it. Rank this folder with that strategy, or switch Color Source to "
                    "one that already has, then reopen the Loupe."
                )

        prev_btn = QPushButton("‹ Prev", self)
        next_btn = QPushButton("Next ›", self)
        keep_btn = QPushButton("Keep (K)", self)
        reject_btn = QPushButton("Reject (R)", self)
        neutral_btn = QPushButton("Neutral (N)", self)
        self._status_buttons = {"keep": keep_btn, "reject": reject_btn, "neutral": neutral_btn}
        for name, btn in self._status_buttons.items():
            btn.setStyleSheet(_STATUS_BUTTON_STYLES[name])
        exp_down_btn = QPushButton("−", self)
        exp_up_btn = QPushButton("+", self)
        # Elements - prominent while active (its own accent-bordered style,
        # see _ELEMENTS_BUTTON_STYLE below) per the redesign's own "Clear
        # Elements control" priority. Independently toggleable from Boxes
        # (see _refresh_detection_overlay) - both may be on at once.
        self._elements_btn = QPushButton("Elements", self)
        self._elements_btn.setCheckable(True)
        self._elements_btn.setStyleSheet(_ELEMENTS_BUTTON_STYLE)
        self._elements_btn.setToolTip(
            "Show what the algorithm detected - Left Eye, Right Eye, and Head - as labeled "
            "boxes with each one's own confidence"
        )
        self._boxes_btn = QPushButton("Boxes", self)
        self._boxes_btn.setCheckable(True)
        self._boxes_btn.setChecked(self._show_boxes)
        self._boxes_btn.setToolTip(
            "Show the AI's detected-subject bounding box and, when Classic Vision has run, "
            "the eye it measured, on this image"
        )
        save_jpeg_btn = QPushButton("Save JPEG", self)
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
            exp_down_btn, exp_up_btn, self._elements_btn, self._boxes_btn, save_jpeg_btn,
        ):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        prev_btn.clicked.connect(self._go_prev)
        next_btn.clicked.connect(lambda: self._go_next())
        keep_btn.clicked.connect(lambda: self._apply_status("keep"))
        reject_btn.clicked.connect(lambda: self._apply_status("reject"))
        neutral_btn.clicked.connect(lambda: self._apply_status("neutral"))
        exp_down_btn.clicked.connect(lambda: self._adjust_exposure(-1))
        exp_up_btn.clicked.connect(lambda: self._adjust_exposure(1))
        self._elements_btn.toggled.connect(self._on_elements_toggled)
        self._boxes_btn.toggled.connect(self._on_boxes_toggled)
        save_jpeg_btn.clicked.connect(self._save_jpeg)
        # No Close button - see the module docstring: the window's own
        # native controls (and Escape - see keyPressEvent) already close it,
        # and a large "Close" button here was consuming bar space for
        # something every window already offers for free.

        bottom_bar = QWidget(self)
        bottom_bar.setObjectName("loupeBottomBar")
        bottom_bar.setStyleSheet(_BOTTOM_BAR_STYLESHEET)

        # Row 1 - the "what am I looking at, and how did it score" glance:
        # counter/filename/status on the left, the one prominent score
        # centered, this burst's own rank on the right when relevant. See
        # the module docstring's Row 1/Row 2 split.
        row1 = QWidget(bottom_bar)
        row1.setObjectName("loupeRow1")
        row1_layout = QGridLayout(row1)
        row1_layout.setContentsMargins(16, 10, 16, 8)
        row1_layout.setHorizontalSpacing(14)
        row1_layout.setColumnStretch(0, 1)
        row1_layout.setColumnStretch(1, 0)
        row1_layout.setColumnStretch(2, 1)

        row1_info_group = QWidget(row1)
        row1_info_layout = QHBoxLayout(row1_info_group)
        row1_info_layout.setContentsMargins(0, 0, 0, 0)
        row1_info_layout.setSpacing(12)
        row1_info_layout.addWidget(self._counter_label)
        row1_info_layout.addWidget(self._name_label)
        row1_info_layout.addWidget(self._status_label)
        row1_info_layout.addWidget(self._ai_badge_label)

        # Burst rank info AND the Color Source/Ranking Status detail live
        # here together (row1's own right-hand group), not split across
        # both rows - row 2's actions group (Reason/Prev/Keep/Reject/
        # Neutral/Next, ~666px on its own) is already the single widest
        # thing in the bar, so anything that can move off row 2 without
        # hurting "Row 1 = info, Row 2 = controls" should, to keep row 2's
        # own total comfortably under a MacBook's width - see this fix's
        # own measurement (row 2 needed 1555px before this move).
        row1_rank_group = QWidget(row1)
        row1_rank_layout = QHBoxLayout(row1_rank_group)
        row1_rank_layout.setContentsMargins(0, 0, 0, 0)
        row1_rank_layout.setSpacing(12)
        row1_rank_layout.addWidget(self._burst_id_label)
        row1_rank_layout.addWidget(self._burst_rank_label)
        row1_rank_layout.addWidget(self._burst_best_label)
        row1_rank_layout.addWidget(self._burst_color_source_label)
        row1_rank_layout.addWidget(self._burst_ranking_status_label)

        row1_layout.addWidget(row1_info_group, 0, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row1_layout.addWidget(self._primary_score_label, 0, 1, Qt.AlignmentFlag.AlignCenter)
        row1_layout.addWidget(row1_rank_group, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Row 2 - secondary metadata (left) and every control (center:
        # Reason/Prev/Keep/Reject/Neutral/Next, right: zoom/exposure/
        # Elements/Boxes/Save JPEG). Same "outer columns share the leftover
        # space, center column keeps its own size" trick as Row 1, so the
        # core review workflow stays visually centered regardless of how
        # wide the metadata/tools columns end up.
        row2 = QWidget(bottom_bar)
        row2.setObjectName("loupeRow2")
        row2_layout = QGridLayout(row2)
        row2_layout.setContentsMargins(16, 8, 16, 10)
        row2_layout.setHorizontalSpacing(14)
        row2_layout.setColumnStretch(0, 1)
        row2_layout.setColumnStretch(1, 0)
        row2_layout.setColumnStretch(2, 1)

        row2_meta_group = QWidget(row2)
        row2_meta_layout = QHBoxLayout(row2_meta_group)
        row2_meta_layout.setContentsMargins(0, 0, 0, 0)
        row2_meta_layout.setSpacing(12)
        row2_meta_layout.addWidget(self._score_label)

        row2_actions_group = QWidget(row2)
        row2_actions_layout = QHBoxLayout(row2_actions_group)
        row2_actions_layout.setContentsMargins(0, 0, 0, 0)
        row2_actions_layout.setSpacing(10)
        row2_actions_layout.addWidget(QLabel("Reason:"))
        row2_actions_layout.addWidget(self._reason_combo)
        row2_actions_layout.addWidget(self._reason_note)
        row2_actions_layout.addWidget(prev_btn)
        row2_actions_layout.addWidget(keep_btn)
        row2_actions_layout.addWidget(reject_btn)
        row2_actions_layout.addWidget(neutral_btn)
        row2_actions_layout.addWidget(next_btn)

        row2_tools_group = QWidget(row2)
        row2_tools_layout = QHBoxLayout(row2_tools_group)
        row2_tools_layout.setContentsMargins(0, 0, 0, 0)
        row2_tools_layout.setSpacing(10)
        row2_tools_layout.addWidget(self._zoom_label)
        row2_tools_layout.addWidget(exp_down_btn)
        row2_tools_layout.addWidget(self._exposure_label)
        row2_tools_layout.addWidget(exp_up_btn)
        row2_tools_layout.addWidget(self._elements_btn)
        row2_tools_layout.addWidget(self._boxes_btn)
        row2_tools_layout.addWidget(save_jpeg_btn)

        row2_layout.addWidget(row2_meta_group, 0, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row2_layout.addWidget(row2_actions_group, 0, 1, Qt.AlignmentFlag.AlignCenter)
        row2_layout.addWidget(row2_tools_group, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        bottom_bar_layout = QVBoxLayout(bottom_bar)
        bottom_bar_layout.setContentsMargins(0, 0, 0, 0)
        bottom_bar_layout.setSpacing(0)
        bottom_bar_layout.addWidget(row1)
        bottom_bar_layout.addWidget(row2)

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
        if pixmap.isNull():
            # QPixmap fails silently (no exception - only a Qt stderr
            # warning) on a corrupt/truncated cached preview JPEG, which
            # `preview_path` alone cannot detect since it only checks that
            # the file exists, not that it decodes. A stale, pre-atomic-write
            # cache entry left over from before this cache was written
            # atomically (see review.thumbnails.review_preview) is the known
            # cause - delete it and regenerate once before giving up, so a
            # single corrupt cache file heals itself on the next visit
            # instead of showing a permanently blank Loupe for that image.
            try:
                Path(preview).unlink(missing_ok=True)
                preview = self.service.preview_path(path)
                pixmap = QPixmap(str(preview))
            except Exception:  # noqa: BLE001 - regeneration failing must not crash the loupe either
                pixmap = QPixmap()
        if pixmap.isNull():
            QMessageBox.warning(
                self, "PeakPic - Loupe",
                f"Could not display this image (the preview failed to decode):\n{path}",
            )
        self._current_raw_pixmap = pixmap
        self._view.set_pixmap(self._exposed_pixmap())
        self._refresh_detection_overlay()
        self._update_info_labels()

    def _refresh_detection_overlay(self) -> None:
        """Boxes and Elements are independently toggleable - either, both,
        or neither can be active at once (Manual QA: Elements must never
        accidentally hide Boxes). One shared clear up front, then each
        active mode draws additively on top - see `_ZoomView.
        clear_detection_overlay`'s own docstring for why neither
        `set_detection_overlay` nor `set_elements_overlay` clears on its
        own anymore."""
        self._view.clear_detection_overlay()
        if not self._show_boxes and not self._show_elements:
            return
        try:
            eye = self.service.eye_keypoints(self._current_path())
        except Exception:  # noqa: BLE001 - a missing/unreadable eye record must not break the loupe
            eye = None
        if self._show_boxes:
            try:
                boxes = self.service.detection_boxes(self._current_path())
            except Exception:  # noqa: BLE001 - a missing/unreadable detection record must not break the loupe
                boxes = None
            self._view.set_detection_overlay(boxes, eye)
        if self._show_elements:
            self._view.set_elements_overlay(eye)

    def _on_boxes_toggled(self, checked: bool) -> None:
        """Independent of Elements (see _refresh_detection_overlay) - both
        can be on, off, or any combination; toggling one never touches the
        other's own checked state."""
        self._show_boxes = checked
        self._refresh_detection_overlay()

    def _on_elements_toggled(self, checked: bool) -> None:
        self._show_elements = checked
        self._refresh_detection_overlay()

    def _exposed_pixmap(self) -> QPixmap:
        if self._current_raw_pixmap is None:
            return QPixmap()
        if self._exposure_steps == 0:
            return self._current_raw_pixmap
        return _apply_brightness(self._current_raw_pixmap, self._exposure_steps * EXPOSURE_STEP)

    def _primary_strategy_id(self) -> str | None:
        """Whichever ranking strategy is "the" algorithm right now - the
        same one Burst Analysis/the gallery's Color Source picker/"Color by
        Algorithm" coloring already treat as current (ReviewSession.
        burst_strategy). The one score this dialog gives a single,
        prominent, unambiguous readout to - see _primary_score_text."""
        return getattr(self.service.session, "burst_strategy", None)

    def _primary_score_text(self, info: dict) -> str:
        """"IMAGE SCORE: 0.384" - the normalized [0, 1] score, exactly three
        decimal places, never the *100 "38.4"-style reading the burst row
        used to show (_update_burst_info_labels, before this - the same
        number, just multiplied for a percent-style glance that turned out
        to read as a totally different, wrong-looking scale)."""
        strategy_id = self._primary_strategy_id()
        entry = (info.get("ranking_results") or {}).get(strategy_id) or {} if strategy_id else {}
        score = entry.get("score")
        return f"IMAGE SCORE: {score:.3f}" if score is not None else "IMAGE SCORE: —"

    def _update_info_labels(self) -> None:
        path = self._current_path()
        info = self._current_item_info()

        self._counter_label.setText(f"Image {self.index + 1} of {len(self.image_paths)}")
        self._name_label.setText(Path(path).name)
        self._primary_score_label.setText(self._primary_score_text(info))
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
        """Burst ID / Burst Rank #X of N / Best Image - only populated for a
        burst-scoped session (see `_burst_scoped`); the labels stay empty
        text otherwise so a non-burst Loupe session shows nothing where
        this row would be. The score that produced burst_rank is the same
        number _primary_score_text already shows prominently (both read
        the same self._primary_strategy_id()) - no longer repeated here."""
        if not self._burst_scoped or self.items is None:
            return
        item = self.items[self.index]
        self._burst_id_label.setText(_format_burst_label(item.burst_id) if item.burst_id else "")
        self._burst_rank_label.setText(f"Burst Rank #{item.burst_rank} of {item.burst_size}")
        self._burst_best_label.setText(f"Best Image: {'Yes' if item.burst_best else 'No'}")


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
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            # Key_Equal, not just Key_Plus: on a standard US keyboard "+" is
            # Shift+= , and Qt reports the unshifted key (Key_Equal) for a
            # plain "=" press - accepting both means the shortcut works
            # whether or not the photographer holds Shift, matching how
            # every other zoom-by-keyboard app (e.g. a browser's Ctrl+=)
            # already behaves.
            self._view.zoom_by(1.15)
        elif key == Qt.Key.Key_Minus:
            self._view.zoom_by(1 / 1.15)
        elif key == Qt.Key.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)
