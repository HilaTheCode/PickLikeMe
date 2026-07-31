"""Delegate for rendering rich thumbnail cards in the gallery."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QAbstractItemDelegate, QStyle

from ... import theme


class ThumbnailCardDelegate(QAbstractItemDelegate):
    """Renders thumbnail cards with image, filename, score, rank, and review status.

    Colors are pulled from `theme.current_palette()` on every paint rather
    than cached, so a theme switch takes effect on the next repaint with no
    extra notification wiring.
    """

    # Geometry constants - independent of theme. CARD_HEIGHT must be tall
    # enough for PADDING + THUMBNAIL_SIZE + SPACING + filename line (20) +
    # score line (14) + badge line (~18) + PADDING, or the badge line
    # paints past the card's bottom edge into the next grid row (it did,
    # at the original CARD_HEIGHT=200 - GalleryView's setGridSize uses
    # these constants directly, so the overflow wasn't just cosmetic).
    CARD_WIDTH = 200
    CARD_HEIGHT = 232
    THUMBNAIL_SIZE = 160
    PADDING = 8
    SPACING = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._name_font = QFont()
        self._name_font.setBold(True)
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

        palette = theme.current_palette()
        rect = option.rect
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        # Draw background
        bg_color = self._get_background_color(palette, item, is_hovered)
        painter.fillRect(rect, bg_color)

        # Draw border
        border_color = QColor(palette.selection_border) if is_selected else QColor(palette.border)
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

        # Draw filename - bold, so it reads as the card's primary label at
        # a glance rather than competing visually with the metadata below.
        filename = item.display_name
        painter.setFont(self._name_font)
        painter.setPen(QColor(palette.text_primary))
        name_rect = QRect(text_rect)
        name_rect.setHeight(20)
        elided_name = QFontMetrics(self._name_font).elidedText(filename, Qt.TextElideMode.ElideRight, name_rect.width())
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
            painter.setPen(QColor(palette.text_muted))
            score_rect = QRect(text_rect)
            score_rect.setY(score_rect.y() + 20)
            score_rect.setHeight(14)
            elided_score = QFontMetrics(self._small_font).elidedText(score_text, Qt.TextElideMode.ElideRight, score_rect.width())
            painter.drawText(score_rect, Qt.AlignmentFlag.AlignTop, elided_score)

        # Draw review status and AI suggestion badges
        painter.setFont(self._tiny_font)
        badge_rect = QRect(text_rect)
        badge_rect.setY(badge_rect.y() + 36)
        badge_rect.setHeight(16)
        self._draw_status_badge(painter, palette, badge_rect, item)

        painter.restore()

    @staticmethod
    def _get_background_color(palette: theme.Palette, item, is_hovered: bool) -> QColor:
        """Determine background color based on review status."""
        if item.review_status == "keep":
            return QColor(palette.keep_bg)
        if item.review_status == "reject":
            return QColor(palette.reject_bg)
        if is_hovered:
            return QColor(palette.hover_bg)
        return QColor(palette.neutral_bg)

    @staticmethod
    def _draw_status_badge(painter: QPainter, palette: theme.Palette, rect: QRect, item) -> None:
        """Draw review status and AI suggestion badges.

        Uses the rect+alignment drawText overload throughout (matching the
        filename/score rows above), not the point+baseline overload - mixing
        the two previously put the badge's text *baseline* where the score
        row's text *top* was, so the two visibly overlapped.
        """
        status_text = ""
        status_color: QColor | None = None

        if item.review_status == "keep":
            status_text = "✓ Keep"
            status_color = QColor(palette.keep_fg)
        elif item.review_status == "reject":
            status_text = "✗ Reject"
            status_color = QColor(palette.reject_fg)
        elif item.review_status == "neutral":
            status_text = "? Neutral"
            status_color = QColor(palette.neutral_fg)

        # Draw status badge, then the AI suggestion right after it (not at
        # a fixed offset - "Neutral"/"AI: Neutral" are both longer than
        # "Keep"/"AI: Keep", and a fixed gap either clipped the longer
        # combinations or left an oddly wide gap for the shorter ones).
        next_x = rect.x()
        if status_text and status_color is not None:
            painter.setPen(status_color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, status_text)
            next_x = rect.x() + QFontMetrics(painter.font()).horizontalAdvance(status_text) + 8

        if item.ai_suggestion and item.ai_suggestion != item.review_status:
            ai_text = f"AI: {item.ai_suggestion.capitalize()}"
            ai_rect = QRect(rect)
            ai_rect.setX(next_x)
            painter.setPen(QColor(palette.accent))
            painter.drawText(ai_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, ai_text)
