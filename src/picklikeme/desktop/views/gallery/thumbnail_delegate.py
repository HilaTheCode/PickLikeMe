"""Delegate for rendering rich, clickable thumbnail cards in the gallery."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractItemDelegate, QStyle

from ... import theme

STATUS_ORDER = ("keep", "reject", "neutral")
STATUS_SYMBOLS = {"keep": "✓", "reject": "✗", "neutral": "○"}


class ThumbnailCardDelegate(QAbstractItemDelegate):
    """Renders thumbnail cards with image, filename, score, rank, AI
    suggestion, and clickable Keep/Reject/Neutral buttons.

    Colors are pulled from `theme.current_palette()` on every paint rather
    than cached, so a theme switch takes effect on the next repaint with no
    extra notification wiring. Button geometry lives in one method
    (button_rects) used by both paint() and GalleryView's mouse handling,
    so the drawn buttons and their click targets can never drift apart.
    """

    # Geometry constants - independent of theme. CARD_WIDTH is wide enough
    # for "Score -0.1234 · Rank 999" plus an "AI Reject" pill on the same
    # line without eliding - narrower cards truncated the score text
    # noticeably often.
    CARD_WIDTH = 220
    CARD_HEIGHT = 252
    THUMBNAIL_SIZE = 160
    PADDING = 8
    SPACING = 4
    CORNER_RADIUS = 8
    BUTTON_HEIGHT = 24
    BUTTON_GAP = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._name_font = QFont()
        self._name_font.setBold(True)
        self._small_font = QFont()
        self._small_font.setPointSize(8)
        self._tiny_font = QFont()
        self._tiny_font.setPointSize(7)
        self._button_font = QFont()
        self._button_font.setPointSize(9)
        self._button_font.setBold(True)

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
        """Paint a single thumbnail card."""
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

        bg_color = self._get_background_color(palette, item, is_hovered)
        painter.fillPath(path, bg_color)

        if is_selected:
            # Selection needs to read at a glance even for a card that's
            # also Keep/Reject-tinted - a translucent wash across the
            # whole card, on top of any status tint, plus a heavier
            # border, gives a much stronger combined cue (a pixel-sampling
            # check during development showed a border difference alone
            # was too weak to reliably tell apart in a dense grid).
            wash = QColor(palette.selection_border)
            wash.setAlpha(45)
            painter.fillPath(path, wash)

        # Frame color communicates review status at a glance - the single
        # biggest ask behind this redesign (green/red/neutral, matching the
        # web review UI's card borders), overridden by the selection color
        # when the card is selected.
        border_color = QColor(palette.selection_border) if is_selected else self._status_color(palette, item.review_status)
        border_width = 3 if is_selected else 2
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inset = border_width / 2.0
        painter.drawRoundedRect(
            card_rect.adjusted(int(inset), int(inset), -int(inset), -int(inset)),
            self.CORNER_RADIUS, self.CORNER_RADIUS,
        )

        # Thumbnail image. review_thumbnail()'s cached files are letterboxed
        # onto a square canvas (contactsheets.build_thumbnail pastes each
        # photo, centered, onto a fixed-color square) - fine at full size,
        # but at this card's much smaller display area the letterbox bars
        # became a visually prominent dark band. "Cover" scaling (like CSS
        # object-fit: cover) fills the target box and center-crops the
        # overflow, cropping the letterboxing out - the same effect the
        # web review UI gets from displaying these same cached files in an
        # <img> with object-fit: cover.
        thumbnail = index.data(Qt.DecorationRole)
        if thumbnail is not None and not thumbnail.isNull():
            thumb_rect = card_rect.adjusted(self.PADDING, self.PADDING, -self.PADDING, -self.PADDING)
            thumb_rect.setHeight(self.THUMBNAIL_SIZE)
            scaled = thumbnail.scaled(
                thumb_rect.width(), thumb_rect.height(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            crop_x = max(0, (scaled.width() - thumb_rect.width()) // 2)
            crop_y = max(0, (scaled.height() - thumb_rect.height()) // 2)
            cropped = scaled.copy(crop_x, crop_y, thumb_rect.width(), thumb_rect.height())
            painter.drawPixmap(thumb_rect.topLeft(), cropped)

        text_y = card_rect.y() + self.PADDING + self.THUMBNAIL_SIZE + self.SPACING
        text_rect = QRect(card_rect.x() + self.PADDING, text_y, card_rect.width() - 2 * self.PADDING, 18)

        # Filename - bold, so it reads as the card's primary label at a
        # glance rather than competing visually with the metadata below.
        painter.setFont(self._name_font)
        painter.setPen(QColor(palette.text_primary))
        elided_name = QFontMetrics(self._name_font).elidedText(item.display_name, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, elided_name)

        # Score/rank (left) and AI-suggestion pill (right) on one line.
        # moveTop(), not setY(): QRect.setY() moves the top edge while
        # holding the *bottom* edge fixed (it's meant for resizing a rect
        # from its top, not translating one) - it collapsed this row's
        # 18px height down to ~-1px, silently clamped, so nothing drew.
        # moveTop() translates the whole rect and preserves its size.
        score_rect = QRect(text_rect)
        score_rect.moveTop(score_rect.y() + 19)
        self._draw_score_and_ai_badge(painter, palette, score_rect, item)

        # Keep/Reject/Neutral buttons
        self._draw_buttons(painter, palette, card_rect, item)

        painter.restore()

    def _draw_score_and_ai_badge(self, painter: QPainter, palette: theme.Palette, rect: QRect, item) -> None:
        parts = []
        if item.score is not None:
            parts.append(f"Score {item.score:.4f}")
        if item.rank is not None:
            parts.append(f"Rank {item.rank}")
        score_text = " · ".join(parts) if parts else "Unranked"

        painter.setFont(self._small_font)
        painter.setPen(QColor(palette.text_muted))
        metrics = QFontMetrics(self._small_font)

        ai_text = ""
        if item.ai_suggestion and item.ai_suggestion != item.review_status:
            ai_text = f"AI {item.ai_suggestion.capitalize()}"
        ai_width = metrics.horizontalAdvance(ai_text) + 14 if ai_text else 0

        score_box = QRect(rect)
        score_box.setWidth(max(0, rect.width() - ai_width - (6 if ai_text else 0)))
        elided_score = metrics.elidedText(score_text, Qt.TextElideMode.ElideRight, score_box.width())
        painter.drawText(score_box, Qt.AlignmentFlag.AlignVCenter, elided_score)

        if ai_text:
            pill_rect = QRect(rect.right() - ai_width, rect.y(), ai_width, rect.height())
            pill_path = QPainterPath()
            pill_path.addRoundedRect(float(pill_rect.x()), float(pill_rect.y()), float(pill_rect.width()), float(pill_rect.height()), rect.height() / 2.0, rect.height() / 2.0)
            fill = QColor(palette.accent)
            fill.setAlpha(40)
            painter.fillPath(pill_path, fill)
            painter.setPen(QColor(palette.accent))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, ai_text)

    def _draw_buttons(self, painter: QPainter, palette: theme.Palette, card_rect: QRect, item) -> None:
        rects = self.button_rects(card_rect)
        active_colors = {"keep": palette.keep_fg, "reject": palette.reject_fg, "neutral": palette.neutral_fg}
        painter.setFont(self._button_font)
        for status in STATUS_ORDER:
            btn_rect = rects[status]
            is_active = item.review_status == status
            btn_path = QPainterPath()
            radius = self.BUTTON_HEIGHT / 2.0
            btn_path.addRoundedRect(float(btn_rect.x()), float(btn_rect.y()), float(btn_rect.width()), float(btn_rect.height()), radius, radius)
            if is_active:
                painter.fillPath(btn_path, QColor(active_colors[status]))
                painter.setPen(QColor("#ffffff") if status != "neutral" else QColor(palette.window_bg))
            else:
                painter.setPen(QPen(QColor(palette.border), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(btn_path)
                painter.setPen(QColor(palette.text_muted))
            painter.drawText(btn_rect, Qt.AlignmentFlag.AlignCenter, STATUS_SYMBOLS[status])

    @staticmethod
    def _get_background_color(palette: theme.Palette, item, is_hovered: bool) -> QColor:
        if item.review_status == "keep":
            return QColor(palette.keep_bg)
        if item.review_status == "reject":
            return QColor(palette.reject_bg)
        if is_hovered:
            return QColor(palette.hover_bg)
        return QColor(palette.neutral_bg)

    @staticmethod
    def _status_color(palette: theme.Palette, status: str) -> QColor:
        if status == "keep":
            return QColor(palette.keep_fg)
        if status == "reject":
            return QColor(palette.reject_fg)
        return QColor(palette.border)
