"""Delegate for rendering rich thumbnail cards in the gallery."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QAbstractItemDelegate, QStyle


class ThumbnailCardDelegate(QAbstractItemDelegate):
    """Renders thumbnail cards with image, filename, score, rank, and review status."""

    # Geometry constants
    CARD_WIDTH = 200
    CARD_HEIGHT = 200
    THUMBNAIL_SIZE = 160
    PADDING = 8
    SPACING = 4

    # Color palette
    COLOR_BG_NEUTRAL = QColor(245, 245, 245)
    COLOR_BG_KEEP = QColor(220, 255, 220)
    COLOR_BG_REJECT = QColor(255, 220, 220)
    COLOR_BG_HOVER = QColor(240, 240, 240)
    COLOR_TEXT_NORMAL = QColor(50, 50, 50)
    COLOR_TEXT_MUTED = QColor(120, 120, 120)
    COLOR_BORDER_NORMAL = QColor(200, 200, 200)
    COLOR_BORDER_SELECTED = QColor(50, 150, 255)
    COLOR_BADGE_KEEP = QColor(76, 175, 80)
    COLOR_BADGE_REJECT = QColor(244, 67, 54)
    COLOR_BADGE_AI = QColor(33, 150, 243)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._small_font = QFont()
        self._small_font.setPointSize(8)
        self._tiny_font = QFont()
        self._tiny_font.setPointSize(7)

    def sizeHint(self, option, index) -> QSize:  # noqa: ARG002
        return QSize(self.CARD_WIDTH, self.CARD_HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        """Paint a single thumbnail card."""
        painter.save()

        # Get item data
        item = index.data(Qt.UserRole)
        if item is None:
            painter.restore()
            return

        rect = option.rect
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        # Draw background
        bg_color = self._get_background_color(item, is_hovered)
        painter.fillRect(rect, bg_color)

        # Draw border
        border_color = self.COLOR_BORDER_SELECTED if is_selected else self.COLOR_BORDER_NORMAL
        border_pen = QPen(border_color, 2 if is_selected else 1)
        painter.setPen(border_pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        # Draw thumbnail image
        thumbnail = index.data(Qt.DecorationRole)
        if thumbnail is not None:
            thumb_rect = rect.adjusted(self.PADDING, self.PADDING, -self.PADDING, -self.PADDING)
            thumb_rect.setHeight(self.THUMBNAIL_SIZE)
            pixmap = thumbnail
            if not pixmap.isNull():
                scaled = pixmap.scaledToWidth(thumb_rect.width(), Qt.TransformationMode.SmoothTransformation)
                y_offset = (self.THUMBNAIL_SIZE - scaled.height()) // 2
                thumb_rect.setY(thumb_rect.y() + y_offset)
                painter.drawPixmap(thumb_rect.topLeft(), scaled)

        # Draw text section (below thumbnail)
        text_y = rect.y() + self.PADDING + self.THUMBNAIL_SIZE + self.SPACING
        text_rect = QRect(rect.x() + self.PADDING, text_y, rect.width() - 2 * self.PADDING, rect.height() - text_y - self.PADDING)

        # Draw filename
        filename = item.display_name
        name_font = QFont()
        painter.setFont(name_font)
        painter.setPen(self.COLOR_TEXT_NORMAL)
        name_rect = QRect(text_rect)
        name_rect.setHeight(20)
        elided_name = QFontMetrics(name_font).elidedText(filename, Qt.TextElideMode.ElideRight, name_rect.width())
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignTop, elided_name)

        # Draw score and rank
        score_text = ""
        if item.score is not None:
            score_text = f"Score: {item.score:.3f}"
        if item.rank is not None:
            rank_text = f"Rank: {item.rank}"
            score_text = f"{score_text} | {rank_text}" if score_text else rank_text

        if score_text:
            painter.setFont(self._small_font)
            painter.setPen(self.COLOR_TEXT_MUTED)
            score_rect = QRect(text_rect)
            score_rect.setY(score_rect.y() + 20)
            score_rect.setHeight(14)
            elided_score = QFontMetrics(self._small_font).elidedText(score_text, Qt.TextElideMode.ElideRight, score_rect.width())
            painter.drawText(score_rect, Qt.AlignmentFlag.AlignTop, elided_score)

        # Draw review status and AI suggestion badges
        painter.setFont(self._tiny_font)
        badge_y = text_rect.y() + 36
        self._draw_status_badge(painter, text_rect.x(), badge_y, item)

        painter.restore()

    def _get_background_color(self, item, is_hovered: bool) -> QColor:
        """Determine background color based on review status."""
        if item.review_status == "keep":
            return self.COLOR_BG_KEEP
        if item.review_status == "reject":
            return self.COLOR_BG_REJECT
        if is_hovered:
            return self.COLOR_BG_HOVER
        return self.COLOR_BG_NEUTRAL

    def _draw_status_badge(self, painter: QPainter, x: int, y: int, item) -> None:
        """Draw review status and AI suggestion badges."""
        status_text = ""
        status_color = None

        if item.review_status == "keep":
            status_text = "✓ Keep"
            status_color = self.COLOR_BADGE_KEEP
        elif item.review_status == "reject":
            status_text = "✗ Reject"
            status_color = self.COLOR_BADGE_REJECT
        elif item.review_status == "neutral":
            status_text = "? Neutral"
            status_color = self.COLOR_TEXT_MUTED

        # Draw status badge with background
        if status_text and status_color:
            painter.setPen(status_color)
            painter.drawText(x, y, status_text)

        # Draw AI suggestion if it differs from review status
        if item.ai_suggestion and item.ai_suggestion != item.review_status:
            ai_text = f"AI: {item.ai_suggestion.capitalize()}"
            painter.setPen(self.COLOR_BADGE_AI)
            painter.drawText(x + 100, y, ai_text)
