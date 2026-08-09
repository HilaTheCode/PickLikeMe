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
    THUMBNAIL_SIZE = 160
    PADDING = 8
    SPACING = 4
    CORNER_RADIUS = 8
    BUTTON_HEIGHT = 24
    BUTTON_GAP = 4

    # One line per analysis module's score. Cards are a fixed size (the view
    # is a uniform grid), so the height reserves room for this many rows and
    # `_score_rows` shows at most that many - a card cannot grow taller for
    # one image without every card growing, and a folder scored by an unusual
    # number of modules must not push the buttons off the bottom.
    SCORE_ROW_HEIGHT = 15
    MAX_SCORE_ROWS = 3
    # 252 was the height with a single score row; the rest is the extra rows.
    CARD_HEIGHT = 252 + (MAX_SCORE_ROWS - 1) * SCORE_ROW_HEIGHT

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
        # None (the default) means "no algorithm fallback for Priority #2
        # of the coloring policy" - see MainWindow.color_source_options and
        # _get_background_color's own docstring. Otherwise a strategy id -
        # set via set_color_source, kept in sync with ReviewSession.
        # burst_strategy by MainWindow._on_color_source_changed, which is
        # what ImageItem.algorithm_suggestion is itself computed against.
        self._color_source: str | None = None
        # Off by default - see set_show_burst_badge / GalleryView.set_show_burst_badges.
        self._show_burst_badge = False

    def set_color_source(self, strategy_id: str | None) -> None:
        self._color_source = strategy_id

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
            if self._show_burst_badge and item.burst_size > 1:
                self._draw_burst_badge(painter, palette, thumb_rect, item.burst_size)

        text_y = card_rect.y() + self.PADDING + self.THUMBNAIL_SIZE + self.SPACING
        text_rect = QRect(card_rect.x() + self.PADDING, text_y, card_rect.width() - 2 * self.PADDING, 18)

        # Filename - bold, so it reads as the card's primary label at a
        # glance rather than competing visually with the metadata below.
        painter.setFont(self._name_font)
        painter.setPen(QColor(palette.text_primary))
        elided_name = QFontMetrics(self._name_font).elidedText(item.display_name, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, elided_name)

        # One line per analysis module's score, the first of which also
        # carries the AI-suggestion pill on its right.
        # moveTop(), not setY(): QRect.setY() moves the top edge while
        # holding the *bottom* edge fixed (it's meant for resizing a rect
        # from its top, not translating one) - it collapsed this row's
        # 18px height down to ~-1px, silently clamped, so nothing drew.
        # moveTop() translates the whole rect and preserves its size.
        score_rect = QRect(text_rect)
        score_rect.moveTop(score_rect.y() + 19)
        for row, (label, text) in enumerate(self._score_rows(item)):
            row_rect = QRect(score_rect)
            row_rect.moveTop(score_rect.y() + row * self.SCORE_ROW_HEIGHT)
            self._draw_score_row(painter, palette, row_rect, label, text, item, badge=row == 0)

        # Keep/Reject/Neutral buttons
        self._draw_buttons(painter, palette, card_rect, item)

        painter.restore()

    @staticmethod
    def _score_rows(item) -> list[tuple[str, str]]:
        """(label, text) per analysis module that either scored this image
        or explicitly filtered it out.

        Driven entirely by what is in `item.ranking_results`/`.filter_reasons`,
        so a future module shows up without touching this delegate. Labels
        come from the ranking registry when it knows the strategy, and fall
        back to the raw id when it does not - a folder scored by a module
        PeakPic no longer ships still displays its numbers rather than
        dropping them. A module that FILTERED this image (no score, but a
        recorded reason - see `ranking.classic`'s filter phase) shows that
        reason instead of being silently omitted, e.g. "Classic: No visible
        eye" - the at-a-glance answer to "why isn't this ranked".

        An image nothing has scored or filtered yields one "Unranked" row,
        so the card keeps a consistent shape instead of the filename jumping
        down.
        """
        from ....ranking import score_labels
        from ....ranking.filters import REJECT_REASON_LABELS

        labels = score_labels()
        strategy_ids = set(item.ranking_results) | set(item.filter_reasons)
        rows: list[tuple[str, str]] = []
        for strategy_id in sorted(strategy_ids, key=lambda sid: (sid != "ai-model", sid)):
            entry = item.ranking_results.get(strategy_id) or {}
            score = entry.get("score")
            label = labels.get(strategy_id, strategy_id)
            if score is not None:
                rank = entry.get("rank")
                text = f"{score:.4f}" + (f" · #{rank}" if rank else "")
            else:
                reason = item.filter_reasons.get(strategy_id)
                if reason is None:
                    continue
                text = REJECT_REASON_LABELS.get(reason, reason)
            rows.append((label, text))
            if len(rows) == ThumbnailCardDelegate.MAX_SCORE_ROWS:
                break
        return rows or [("", "Unranked")]

    def _draw_score_row(
        self, painter: QPainter, palette: theme.Palette, rect: QRect,
        label: str, text: str, item, *, badge: bool,
    ) -> None:
        painter.setFont(self._small_font)
        metrics = QFontMetrics(self._small_font)

        ai_text = ""
        if badge and item.ai_suggestion and item.ai_suggestion != item.review_status:
            ai_text = f"AI {item.ai_suggestion.capitalize()}"
        ai_width = metrics.horizontalAdvance(ai_text) + 14 if ai_text else 0

        score_box = QRect(rect)
        score_box.setWidth(max(0, rect.width() - ai_width - (6 if ai_text else 0)))

        # The module's name in the accent colour and its number in the muted
        # one, so several stacked rows stay scannable - which module a score
        # belongs to is the thing being read, not the digits.
        label_width = 0
        if label:
            painter.setPen(QColor(palette.accent))
            label_text = f"{label} "
            label_width = metrics.horizontalAdvance(label_text)
            painter.drawText(score_box, Qt.AlignmentFlag.AlignVCenter, label_text)

        value_box = QRect(score_box)
        value_box.setX(value_box.x() + label_width)
        painter.setPen(QColor(palette.text_muted))
        painter.drawText(
            value_box, Qt.AlignmentFlag.AlignVCenter,
            metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(0, value_box.width())),
        )

        if ai_text:
            pill_rect = QRect(rect.right() - ai_width, rect.y(), ai_width, rect.height())
            pill_path = QPainterPath()
            pill_path.addRoundedRect(float(pill_rect.x()), float(pill_rect.y()), float(pill_rect.width()), float(pill_rect.height()), rect.height() / 2.0, rect.height() / 2.0)
            fill = QColor(palette.accent)
            fill.setAlpha(40)
            painter.fillPath(pill_path, fill)
            painter.setPen(QColor(palette.accent))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, ai_text)

    def _draw_burst_badge(self, painter: QPainter, palette: theme.Palette, thumb_rect: QRect, burst_size: int) -> None:
        """"+N" in the thumbnail's top-right corner - N other images share
        this burst_best card's burst (see MainWindow._on_toggle_collapse_bursts).
        Only ever drawn in Collapse Bursts mode, never in the gallery's
        default one-card-per-image view."""
        text = f"+{burst_size - 1}"
        metrics = QFontMetrics(self._small_font)
        painter.setFont(self._small_font)
        width = metrics.horizontalAdvance(text) + 12
        height = metrics.height() + 4
        badge_rect = QRect(thumb_rect.right() - width - 4, thumb_rect.top() + 4, width, height)
        path = QPainterPath()
        path.addRoundedRect(float(badge_rect.x()), float(badge_rect.y()), float(badge_rect.width()), float(badge_rect.height()), height / 2.0, height / 2.0)
        fill = QColor(palette.window_bg)
        fill.setAlpha(210)
        painter.fillPath(path, fill)
        painter.setPen(QColor(palette.text_primary))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)

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

    def _get_background_color(self, palette: theme.Palette, item, is_hovered: bool) -> QColor:
        """The explicit, deterministic coloring policy: the photographer's
        own review_status always wins - Keep/Reject render in their fixed
        colors regardless of any algorithm score, exactly as if no Color
        Source were selected at all. Only once an image has no User
        Decision (still Neutral) does the currently selected Color Source
        get a say, and even then as a binary keep/reject call - not a
        score gradient - via ImageItem.algorithm_suggestion, which is
        already computed against whichever strategy the Color Source
        picker currently selects, at the current keep-percent threshold
        (see ReviewSession.suggestions_for's own docstring). This is why a
        moved threshold recolors the gallery automatically on the next
        refresh with no separate logic here: algorithm_suggestion already
        reflects it.

        A THIRD Priority-#2 outcome, alongside keep/reject: algorithm_
        suggestion is None whenever the selected strategy never scored this
        image at all (ReviewSession.suggestions_for only ever assigns keep/
        reject to images it actually scored - see that method's own
        docstring) - filtered out by that module (no visible eye, no
        subject, ...) or simply never ranked by it. That is not the same
        claim as Neutral ("scored, no decision yet") or Reject ("scored,
        didn't clear the cutoff"), so it gets its own "Skipped" color rather
        than silently falling through to plain Neutral, where it would be
        indistinguishable from an image that DID get a fair look from this
        algorithm.

        "Review Status" as the Color Source (self._color_source is None)
        has no algorithm to fall back to for Priority #2 - an undecided
        image simply stays neutral, exactly as it always has.
        """
        if item.review_status == "keep":
            return QColor(palette.keep_bg)
        if item.review_status == "reject":
            return QColor(palette.reject_bg)
        if self._color_source is not None:
            if item.algorithm_suggestion == "keep":
                return QColor(palette.keep_bg)
            if item.algorithm_suggestion == "reject":
                return QColor(palette.reject_bg)
            if item.algorithm_suggestion is None:
                return QColor(palette.skipped_bg)
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
