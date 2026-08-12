"""Delegate for rendering PeakPick's redesigned thumbnail cards (see
`docs/UX Design/20260810/Ver1.0/05_Thumbnail.svg`).

Card anatomy, top to bottom: the photo (cover-scaled) with a translucent
score badge over its bottom-left corner; a status-colored 2px card border
(not a full-card color wash - the border alone carries the review-status
signal, so the photograph itself stays the dominant content, per the
design spec's own first principle); "{rank}  {filename}" and capture time;
a status/domain line; a slim row of compact Keep/Reject/Neutral controls
(kept - see this module's own docstring below - but visually minimized
relative to the pre-redesign giant pill buttons).

Colors are pulled from `theme.current_palette()` on every paint rather than
cached, so a theme switch takes effect on the next repaint with no extra
notification wiring. Button geometry lives in one method (button_rects)
used by both paint() and GalleryView's mouse handling, so the drawn buttons
and their click targets can never drift apart.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractItemDelegate, QStyle

from ... import theme
from ...widgets.design_system import (
    ALGORITHM_SCORED,
    RADIUS_MD,
    STATUS_LABELS,
    format_score,
    resolve_status,
    score_ramp_color,
    status_bg,
    status_color,
)

STATUS_ORDER = ("keep", "reject", "neutral")
STATUS_SYMBOLS = {"keep": "✓", "reject": "✗", "neutral": "○"}

# Strategy ids each "domain" group covers - a static, UI-level fact about
# which detector/species-domain each registered strategy targets (see
# `eyes/domains.py`'s own DomainProfile.display_name for "Birds"/"Mammals"),
# not a per-image classification. Used only to label a thumbnail's optional
# domain indicator and to group the Grid's Domain filter/Dashboard's Domain
# charts - never to change what a strategy actually scores.
DOMAIN_BY_STRATEGY: dict[str, str] = {
    "classic-vision": "Birds",
    "classic-vision-eyepose-v0": "Birds",
    "classic-vision-fusion-birds": "Birds",
    "classic-vision-fusion-mammals": "Mammals",
    "classic-vision-fusion-combined": "Birds + Mammals",
}


class ThumbnailCardDelegate(QAbstractItemDelegate):
    """Renders thumbnail cards with image, filename, one selected-source
    score, rank, review status, and clickable Keep/Reject/Neutral controls.
    """

    CARD_WIDTH = 224
    THUMBNAIL_HEIGHT = 132
    PADDING = 8
    SPACING = 4
    CORNER_RADIUS = RADIUS_MD
    BUTTON_HEIGHT = 20
    BUTTON_GAP = 4
    NAME_ROW_HEIGHT = 18
    META_ROW_HEIGHT = 16
    CARD_HEIGHT = (
        PADDING + THUMBNAIL_HEIGHT + SPACING + NAME_ROW_HEIGHT + META_ROW_HEIGHT
        + SPACING + BUTTON_HEIGHT + PADDING
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._name_font = QFont()
        self._name_font.setBold(True)
        self._name_font.setPointSize(9)
        self._meta_font = QFont()
        self._meta_font.setPointSize(8)
        self._score_font = QFont()
        self._score_font.setPointSize(11)
        self._score_font.setBold(True)
        self._button_font = QFont()
        self._button_font.setPointSize(9)
        self._button_font.setBold(True)
        # None (the default) means "User Decision" - color every card by the
        # photographer's own Keep/Reject/Undecided and nothing else.
        # Otherwise a strategy id, and the card is colored by THAT strategy's
        # own score (see _resolve_status / _score_fraction).
        self._color_source: str | None = None
        # The lowest/highest score the selected strategy produced across the
        # currently visible set - the ends of the color ramp, so the tint
        # spreads over the range actually on screen rather than over a
        # theoretical 0..1 nothing occupies. None until a set is measured
        # (see set_score_range), in which case every scored card is drawn at
        # the top of the ramp rather than guessing a range.
        self._score_range: tuple[float, float] | None = None
        self._show_burst_badge = False

    def set_color_source(self, strategy_id: str | None) -> None:
        self._color_source = strategy_id

    def set_score_range(self, score_range: tuple[float, float] | None) -> None:
        """The (min, max) of the selected strategy's scores over the visible
        cards - recomputed by MainWindow whenever the visible set or the
        Color mode changes."""
        self._score_range = score_range

    def set_show_burst_badge(self, enabled: bool) -> None:
        self._show_burst_badge = enabled

    def sizeHint(self, option, index) -> QSize:  # noqa: ARG002
        return QSize(self.CARD_WIDTH, self.CARD_HEIGHT)

    def button_rects(self, card_rect: QRect) -> dict[str, QRect]:
        """Keep/Reject/Neutral button rects within a card, in the same
        coordinate space paint() receives - GalleryView's mouse handling
        uses this exact method to hit-test clicks against what was drawn."""
        x0 = card_rect.x() + self.PADDING
        width = card_rect.width() - 2 * self.PADDING
        y0 = card_rect.bottom() - self.PADDING - self.BUTTON_HEIGHT
        btn_width = (width - 2 * self.BUTTON_GAP) // 3
        last_width = width - 2 * (btn_width + self.BUTTON_GAP)
        return {
            "keep": QRect(x0, y0, btn_width, self.BUTTON_HEIGHT),
            "reject": QRect(x0 + btn_width + self.BUTTON_GAP, y0, btn_width, self.BUTTON_HEIGHT),
            "neutral": QRect(x0 + 2 * (btn_width + self.BUTTON_GAP), y0, last_width, self.BUTTON_HEIGHT),
        }

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        item = index.data(Qt.UserRole)
        if item is None:
            painter.restore()
            return

        palette = theme.current_palette()
        rect = option.rect
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        card_rect = rect.adjusted(2, 2, -2, -2)
        path = QPainterPath()
        path.addRoundedRect(float(card_rect.x()), float(card_rect.y()), float(card_rect.width()), float(card_rect.height()), self.CORNER_RADIUS, self.CORNER_RADIUS)

        # The card body stays a plain panel surface - the photograph, not a
        # flat color wash, is what communicates content; a light hover tint
        # is the only background variation (see the design spec's "the
        # photograph is always the dominant content" principle).
        bg_color = QColor(palette.hover_bg) if is_hovered else QColor(palette.panel_bg)
        painter.fillPath(path, bg_color)

        if is_selected:
            wash = QColor(palette.selection_border)
            wash.setAlpha(45)
            painter.fillPath(path, wash)

        status = self._resolve_status(item)
        status_paint = self._status_color(palette, item, status)
        border_color = QColor(palette.selection_border) if is_selected else QColor(status_paint)
        border_width = 3 if is_selected else 2
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inset = border_width / 2.0
        painter.drawRoundedRect(
            card_rect.adjusted(int(inset), int(inset), -int(inset), -int(inset)),
            self.CORNER_RADIUS, self.CORNER_RADIUS,
        )

        thumb_rect = card_rect.adjusted(self.PADDING, self.PADDING, -self.PADDING, 0)
        thumb_rect.setHeight(self.THUMBNAIL_HEIGHT)
        thumbnail = index.data(Qt.DecorationRole)
        if thumbnail is not None and not thumbnail.isNull():
            scaled = thumbnail.scaled(
                thumb_rect.width(), thumb_rect.height(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            crop_x = max(0, (scaled.width() - thumb_rect.width()) // 2)
            crop_y = max(0, (scaled.height() - thumb_rect.height()) // 2)
            cropped = scaled.copy(crop_x, crop_y, thumb_rect.width(), thumb_rect.height())
            image_path = QPainterPath()
            image_path.addRoundedRect(QRectF(thumb_rect), self.CORNER_RADIUS - 2, self.CORNER_RADIUS - 2)
            painter.save()
            painter.setClipPath(image_path)
            painter.drawPixmap(thumb_rect.topLeft(), cropped)
            painter.restore()
            if self._show_burst_badge and item.burst_size > 1:
                self._draw_burst_badge(painter, palette, thumb_rect, item.burst_size)

        self._draw_score_badge(painter, palette, thumb_rect, item)

        text_y = thumb_rect.bottom() + self.SPACING
        name_rect = QRect(card_rect.x() + self.PADDING, text_y, card_rect.width() - 2 * self.PADDING, self.NAME_ROW_HEIGHT)
        painter.setFont(self._name_font)
        painter.setPen(QColor(palette.text_primary))
        # The rank shown must belong to the SAME strategy as the score badge
        # below it - previously this was always the AI model's own rank while
        # the badge showed the selected strategy's score, so a folder ranked
        # by anything else showed a Crop Sharpness number next to an AI-model
        # position (or, far more often, no number at all). User Decision mode
        # selects no algorithm, so there is no rank to show either.
        rank = item.rank_for(self._color_source) if self._color_source else None
        rank_prefix = f"{rank:03d}  " if rank else ""
        elided_name = QFontMetrics(self._name_font).elidedText(
            f"{rank_prefix}{item.display_name}", Qt.TextElideMode.ElideRight, name_rect.width()
        )
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter, elided_name)

        meta_rect = QRect(name_rect)
        meta_rect.moveTop(name_rect.bottom())
        meta_rect.setHeight(self.META_ROW_HEIGHT)
        self._draw_meta_row(painter, palette, meta_rect, item, status, status_paint)

        self._draw_buttons(painter, palette, card_rect, item)

        painter.restore()

    def _resolve_status(self, item) -> str:
        """Which category this card belongs to UNDER THE SELECTED COLOR MODE,
        and only that mode - see `design_system.resolve_status`.

        User Decision mode (`self._color_source is None`): Keep / Reject /
        Undecided, from `item.user_decision`. An image nobody has reviewed is
        Undecided; no score, suggestion, cutoff or recorded algorithm
        decision can make it anything else.

        Algorithm mode (a strategy id): Scored / Filtered Out / Skipped, from
        that strategy's own result. Scored cards are additionally tinted by
        their actual score (`_status_color`). The photographer's decisions do
        not participate - that is what having two separate modes means.

        A thin wrapper around the shared implementation so the Grid and the
        Analytics Dashboard can never disagree about a card.
        """
        return resolve_status(item, self._color_source)

    def _status_color(self, palette, item, status: str) -> str:
        """The card's border color. In an algorithm mode a Scored card is
        painted from its own score's position in the visible range (see
        `design_system.score_ramp_color`), which is what makes the color
        actually correspond to the number in the badge; every other case is
        the category's flat color."""
        if self._color_source is not None and status == ALGORITHM_SCORED:
            return score_ramp_color(palette, self._score_fraction(item))
        return status_color(palette, status)

    def _score_fraction(self, item) -> float:
        """Where this card's score sits between the lowest and highest the
        selected strategy produced across the visible set - 1.0 when the
        range is unknown or degenerate (every visible card scored the same),
        since a flat range has no ordering to show."""
        score = item.score_for(self._color_source)
        if score is None or self._score_range is None:
            return 1.0
        low, high = self._score_range
        if high <= low:
            return 1.0
        return (score - low) / (high - low)

    def _selected_score_text(self, item) -> str:
        """The ONE score this card shows - the currently selected Color
        mode's own strategy, at the shared three-decimal score precision
        (`design_system.format_score`) - never every module's score at once,
        per the design spec's own "do not show every algorithm score on each
        thumbnail" rule. "User Decision" mode selects no algorithm, so there
        is no score to draw."""
        if self._color_source is None:
            return "—"
        return format_score(item.score_for(self._color_source))

    def _draw_score_badge(self, painter: QPainter, palette: theme.Palette, thumb_rect: QRect, item) -> None:
        text = self._selected_score_text(item)
        metrics = QFontMetrics(self._score_font)
        badge_width = metrics.horizontalAdvance(text) + 18
        badge_height = metrics.height() + 8
        badge_rect = QRect(thumb_rect.x() + 6, thumb_rect.bottom() - badge_height - 6, badge_width, badge_height)
        path = QPainterPath()
        path.addRoundedRect(QRectF(badge_rect), 6, 6)
        fill = QColor(palette.window_bg)
        fill.setAlpha(210)
        painter.fillPath(path, fill)
        painter.setFont(self._score_font)
        painter.setPen(QColor(palette.text_primary))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_meta_row(
        self, painter: QPainter, palette: theme.Palette, rect: QRect, item, status: str, status_paint: str
    ) -> None:
        painter.setFont(self._meta_font)
        metrics = QFontMetrics(self._meta_font)

        time_text = (item.captured_at or "").split("T")[-1][:8] if item.captured_at else ""
        painter.setPen(QColor(palette.text_muted))
        time_rect = QRect(rect)
        time_rect.setWidth(rect.width() // 2)
        painter.drawText(time_rect, Qt.AlignmentFlag.AlignVCenter, time_text)

        domain = DOMAIN_BY_STRATEGY.get(self._color_source or "")
        if domain:
            domain_rect = QRect(rect)
            domain_rect.setX(rect.x() + rect.width() // 2)
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(domain_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, domain)

        status_text = STATUS_LABELS[status]
        status_rect = QRect(rect)
        status_rect.setX(rect.x() + time_rect.width())
        status_rect.setWidth(rect.width() - time_rect.width() - (metrics.horizontalAdvance(domain) + 8 if domain else 0))
        painter.setPen(QColor(status_paint))
        font = QFont(self._meta_font)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(status_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter, status_text)

    def _draw_burst_badge(self, painter: QPainter, palette: theme.Palette, thumb_rect: QRect, burst_size: int) -> None:
        text = f"+{burst_size - 1}"
        metrics = QFontMetrics(self._meta_font)
        painter.setFont(self._meta_font)
        width = metrics.horizontalAdvance(text) + 12
        height = metrics.height() + 4
        badge_rect = QRect(thumb_rect.right() - width - 4, thumb_rect.top() + 4, width, height)
        path = QPainterPath()
        path.addRoundedRect(QRectF(badge_rect), height / 2.0, height / 2.0)
        fill = QColor(palette.window_bg)
        fill.setAlpha(210)
        painter.fillPath(path, fill)
        painter.setPen(QColor(palette.text_primary))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_buttons(self, painter: QPainter, palette: theme.Palette, card_rect: QRect, item) -> None:
        """A slim, low-visual-weight Keep/Reject/Neutral row - the design
        spec shows no per-card decision buttons at all (only a status text
        label), but removing single-card click-to-decide would drop real,
        heavily-used functionality (`GalleryView.decisionRequested`) with
        nothing replacing it. Kept, deliberately minimized (20px, muted
        outline until active) rather than the pre-redesign's large pills,
        so the card stays photo-dominant while the capability survives."""
        rects = self.button_rects(card_rect)
        active_colors = {"keep": palette.keep_fg, "reject": palette.reject_fg, "neutral": palette.neutral_fg}
        painter.setFont(self._button_font)
        for status in STATUS_ORDER:
            btn_rect = rects[status]
            is_active = item.review_status == status
            btn_path = QPainterPath()
            radius = self.BUTTON_HEIGHT / 2.0
            btn_path.addRoundedRect(QRectF(btn_rect), radius, radius)
            if is_active:
                painter.fillPath(btn_path, QColor(active_colors[status]))
                painter.setPen(QColor(palette.window_bg))
            else:
                painter.setPen(QPen(QColor(palette.border), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(btn_path)
                painter.setPen(QColor(palette.text_muted))
            painter.drawText(btn_rect, Qt.AlignmentFlag.AlignCenter, STATUS_SYMBOLS[status])
