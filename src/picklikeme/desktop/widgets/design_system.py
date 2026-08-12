"""Shared PeakPick design-system components (2026-08 redesign).

One place the Grid, Loupe and Analytics Dashboard all build their chrome
from, so the three screens read as one product (docs/UX Design/20260810/
Ver1.0/PeakPick_UI_Design_Spec.md's own "create reusable components instead
of implementing each screen independently" rule) rather than three
independently-styled layouts that happen to share a color import.

Every color/spacing/radius value here is read from `theme.current_palette()`
at construction time (matching `ThumbnailCardDelegate`'s own "read fresh on
paint" convention) - nothing is hard-coded twice. `SPACING`/`RADIUS_*`/
`PRIMARY_HEIGHT`/`SECONDARY_HEIGHT` are the "8px rhythm / 7-12px radii /
44px-primary-36px-secondary" rules from the design spec, defined once here
so a screen's own layout code asks for `SPACING * 2` rather than a bare
"16" whose relationship to the base rhythm is not obvious from reading it.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import theme

# Every ranking score and score component is displayed with exactly three
# decimals - 0.000 / 0.237 / 0.812 / 1.000 - wherever the user can see one:
# the Grid card badge, the Loupe header and diagnostics line, and the
# Analytics Dashboard's score/metric readouts. Defined once so the same
# number cannot appear at one precision on one screen and another elsewhere.
# Stored values stay full-precision floats; this is presentation only.
SCORE_FORMAT = ".3f"


def format_score(value: float | None, *, empty: str = "—") -> str:
    """A score as the user sees it: three decimals, or `empty` when there is
    no score to show. The one place that formatting decision is made."""
    return empty if value is None else f"{value:{SCORE_FORMAT}}"


def format_metric_value(value: object, *, absent: str = "not measured") -> str:
    """One raw per-metric measurement from a strategy's metrics report, as a
    diagnostics line shows it.

    A metrics report is a dict of whatever that strategy chose to record, so
    a value here is NOT guaranteed to be a float. Two cases this must
    survive, both real:

    - `None` - the metric genuinely was not measured for this image, which is
      a different fact from measuring zero. Crop Sharpness records
      `relative_subject_size: null` for a full-frame-fallback image, because
      no subject was ever located to measure (see
      `ranking.crop_sharpness.ImageMetrics`). Formatting that with `:.3f`
      raises `TypeError: unsupported format string passed to
      NoneType.__format__`, and because this runs inside the Loupe's own
      construction, the exception escaped through the `doubleClicked` slot -
      where Qt prints it to a stderr a windowed app never shows and then
      simply does not open the dialog. The symptom was a card that silently
      did nothing when double-clicked, on the 1,582 of 5,986 images with no
      detected subject.
    - a bool/str flag such as `has_subject_detection` - happens to survive
      `:.3f` (bool is an int), but "1.000" is a nonsense rendering of True.

    Anything numeric renders at the shared three-decimal score precision;
    everything else renders as itself.
    """
    if value is None:
        return absent
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:{SCORE_FORMAT}}"
    return str(value)


SPACING = 8
RADIUS_SM = 7
RADIUS_MD = 9
RADIUS_LG = 12
PRIMARY_HEIGHT = 44
SECONDARY_HEIGHT = 36

# ---------------------------------------------------------------------------
# Grid coloring: TWO independent modes, never one blended answer.
#
# The Color selector picks exactly one of them, and the mode it picks is the
# ONLY input to a card's color:
#
#   User Decision (color_source is None)  -> USER_DECISION_CATEGORIES
#       Keep / Reject / Undecided, read from `item.user_decision` and nothing
#       else. An image with no decision is Undecided, full stop - no score, no
#       suggestion, no filter verdict, no cutoff and no recorded algorithm
#       decision can color it Keep or Reject.
#
#   An algorithm (color_source is a strategy id) -> ALGORITHM_CATEGORIES
#       Scored / Filtered Out / Skipped, read from that strategy's own result
#       for the image. A Scored card is additionally tinted along a low->high
#       ramp by its ACTUAL score (see `score_ramp_color`), which is what
#       "the color corresponds to that algorithm's score" means. The
#       photographer's own decisions do not enter into it at all.
#
# These used to be one function that answered in one five-value vocabulary,
# with User Decision winning and, failing that, the algorithm's binary
# keep/reject-at-a-threshold suggestion borrowing the SAME Keep/Reject colors.
# That made the two modes mutually contaminating in both directions: an
# algorithm-colored grid was tinted by whatever had been reviewed, and it
# showed a threshold verdict rather than the score it claimed to be showing.
# ---------------------------------------------------------------------------

USER_DECISION_KEEP = "keep"
USER_DECISION_REJECT = "reject"
USER_DECISION_UNDECIDED = "undecided"
USER_DECISION_CATEGORIES: tuple[str, ...] = (
    USER_DECISION_KEEP,
    USER_DECISION_REJECT,
    USER_DECISION_UNDECIDED,
)

ALGORITHM_SCORED = "scored"
ALGORITHM_FILTERED = "filtered"
ALGORITHM_SKIPPED = "skipped"
ALGORITHM_CATEGORIES: tuple[str, ...] = (ALGORITHM_SCORED, ALGORITHM_FILTERED, ALGORITHM_SKIPPED)

STATUS_LABELS: dict[str, str] = {
    USER_DECISION_KEEP: "Keep",
    USER_DECISION_REJECT: "Reject",
    USER_DECISION_UNDECIDED: "Undecided",
    ALGORITHM_SCORED: "Scored",
    ALGORITHM_FILTERED: "Filtered Out",
    ALGORITHM_SKIPPED: "Skipped",
}

# The one user-facing name for the non-algorithm Color mode. Lives here,
# next to the categories it selects, so the Color combo, the Analytics
# Dashboard's header card and the legend can never call it three things.
# It replaced "Review Status", which read as a property of the review
# process rather than as the photographer's own verdict - exactly the
# distinction that got lost.
USER_DECISION_LABEL = "User Decision"


def categories_for_color_source(color_source: str | None) -> tuple[str, ...]:
    """Which category vocabulary the selected Color mode speaks - the one
    place a legend, a KPI row or a count decides that, so no consumer can
    display Keep/Reject counters for an algorithm mode that never produces
    them."""
    return USER_DECISION_CATEGORIES if color_source is None else ALGORITHM_CATEGORIES


def status_color(palette: theme.Palette, status: str) -> str:
    """The one foreground/border color for a category - shared by
    StatusLegend, the thumbnail delegate's card border, and the Loupe's own
    status readouts, so "what color is Reject" has exactly one answer."""
    return {
        USER_DECISION_KEEP: palette.keep_fg,
        USER_DECISION_REJECT: palette.reject_fg,
        USER_DECISION_UNDECIDED: palette.filtered_fg,
        ALGORITHM_SCORED: palette.accent,
        ALGORITHM_FILTERED: palette.filtered_fg,
        ALGORITHM_SKIPPED: palette.skipped_fg,
    }.get(status, palette.text_muted)


def status_bg(palette: theme.Palette, status: str) -> str:
    return {
        USER_DECISION_KEEP: palette.keep_bg,
        USER_DECISION_REJECT: palette.reject_bg,
        USER_DECISION_UNDECIDED: palette.filtered_bg,
        ALGORITHM_SCORED: palette.neutral_bg,
        ALGORITHM_FILTERED: palette.filtered_bg,
        ALGORITHM_SKIPPED: palette.skipped_bg,
    }.get(status, palette.panel_bg_secondary)


def resolve_user_decision(item) -> str:
    """The photographer's own verdict for this card: Keep, Reject, or
    Undecided. The complete rule - there is no second clause.

    `item.user_decision` is already the three-state value the session
    computed from DECISION_SOURCE_USER rows alone (see `review.session.
    ReviewImage.user_decision`); this reads it defensively via
    `review.user_decision.normalize` so an item type that only carries the
    legacy `review_status` spelling still lands on Undecided rather than on
    something unexpected.
    """
    from ...review.user_decision import normalize

    return normalize(getattr(item, "user_decision", None) or getattr(item, "review_status", None))


def resolve_algorithm_state(item, strategy_id: str) -> str:
    """What `strategy_id` itself did with this image - never a keep/reject.

    Scored: it produced a score (`score_for`), so the card is colored by
    that score. Filtered Out: it examined the image and explicitly excluded
    it, recording a reason. Skipped: it never touched the image at all -
    a genuinely different thing from being filtered, and the distinction the
    design spec calls out by name.
    """
    if item.score_for(strategy_id) is not None:
        return ALGORITHM_SCORED
    if strategy_id in item.filter_reasons:
        return ALGORITHM_FILTERED
    return ALGORITHM_SKIPPED


def resolve_status(item, color_source: str | None) -> str:
    """Dispatch to whichever of the two modes `color_source` names - the one
    entry point the Grid delegate and the Analytics Dashboard share, so they
    can never disagree about a card."""
    if color_source is None:
        return resolve_user_decision(item)
    return resolve_algorithm_state(item, color_source)


def score_ramp_color(palette: theme.Palette, fraction: float) -> str:
    """Low-to-high score tint for an algorithm color mode: `fraction` 0.0 is
    the lowest score currently on screen, 1.0 the highest.

    Interpolated between the palette's own Reject and Keep hues so the ramp
    reads the way every other color in the app does (bad -> good), while
    staying a continuous gradient rather than the two-bucket threshold
    verdict this replaced - the point is to see the ordering, including how
    close together the middle of it is.
    """
    fraction = 0.0 if fraction < 0.0 else 1.0 if fraction > 1.0 else fraction
    low = _rgb(palette.reject_fg)
    high = _rgb(palette.keep_fg)
    blended = tuple(round(a + (b - a) * fraction) for a, b in zip(low, high))
    return "#%02X%02X%02X" % blended


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def app_font(*, bold: bool = False, size: int | None = None) -> QFont:
    """"Inter if available, otherwise a clean system sans-serif" (design
    spec) - QFont's own family-fallback already does exactly this: asking
    for "Inter" on a machine without it installed silently substitutes the
    platform default sans rather than erroring, so no availability check is
    needed here."""
    font = QFont("Inter")
    font.setStyleHint(QFont.StyleHint.SansSerif)
    if bold:
        font.setBold(True)
        font.setWeight(QFont.Weight.DemiBold)
    if size is not None:
        font.setPointSize(size)
    return font


def button_qss(palette: theme.Palette, *, height: int, border_color: str | None = None, radius: int = RADIUS_SM) -> str:
    border = border_color or palette.border
    return (
        f"QPushButton {{ background-color: {palette.panel_bg_secondary}; color: {palette.text_primary}; "
        f"border: 1px solid {border}; border-radius: {radius}px; min-height: {height}px; "
        f"padding: 0 {SPACING * 2}px; text-align: left; font-weight: 600; }}"
        f"QPushButton:hover {{ background-color: {palette.hover_bg}; }}"
        f"QPushButton:pressed {{ background-color: {palette.border}; }}"
        f"QPushButton:disabled {{ color: {palette.text_muted}; border-color: {palette.border}; }}"
    )


class PrimaryButton(QPushButton):
    """A toolbar-weight action button - ~44px, an optional small caption
    under the title (e.g. "Rank" / "Run Algorithm"), matching
    `04_Toolbar.svg`'s two-line button anatomy. `accent_color` borders the
    button in that color instead of the plain divider - used for the
    handful of primary actions the toolbar mockup itself borders (Rank,
    Apply Cutoff, Keep=green, Reject=red)."""

    def __init__(self, title: str, subtitle: str | None = None, *, accent_color: str | None = None, parent=None) -> None:
        super().__init__(parent)
        palette = theme.current_palette()
        self.setMinimumHeight(PRIMARY_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._title = title
        self._subtitle = subtitle
        text = f"{title}\n{subtitle}" if subtitle else title
        self.setText(text)
        border = accent_color or palette.border
        self.setStyleSheet(
            f"QPushButton {{ background-color: {palette.panel_bg_secondary}; color: {palette.text_primary}; "
            f"border: 1px solid {border}; border-radius: {RADIUS_SM}px; padding: 4px {SPACING * 2}px; "
            f"text-align: left; font-weight: 600; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: {palette.hover_bg}; }}"
            f"QPushButton:pressed {{ background-color: {palette.border}; }}"
            f"QPushButton:disabled {{ color: {palette.text_muted}; }}"
        )

    def set_subtitle(self, subtitle: str | None) -> None:
        """Update the small caption line (e.g. Apply Cutoff's own "Top 5%"
        reflecting the currently chosen percent) without rebuilding the
        button or its style."""
        self._subtitle = subtitle
        self.setText(f"{self._title}\n{subtitle}" if subtitle else self._title)


class SecondaryButton(QPushButton):
    """A ~36px control-weight button, for anything not promoted to primary
    toolbar weight - Clear Selection, per-tool buttons, dialog actions."""

    def __init__(self, title: str, *, checkable: bool = False, parent=None) -> None:
        super().__init__(title, parent)
        palette = theme.current_palette()
        self.setMinimumHeight(SECONDARY_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCheckable(checkable)
        self.setStyleSheet(
            f"QPushButton {{ background-color: {palette.panel_bg_secondary}; color: {palette.text_primary}; "
            f"border: 1px solid {palette.border}; border-radius: {RADIUS_SM}px; padding: 0 {SPACING * 2}px; "
            f"font-weight: 600; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: {palette.hover_bg}; }}"
            f"QPushButton:checked {{ border: 2px solid {palette.accent}; color: {palette.accent}; }}"
            f"QPushButton:disabled {{ color: {palette.text_muted}; }}"
        )


class LabeledCombo(QWidget):
    """A small caption label stacked above a combo box - the
    "Filter"/"Domain"/"Search"/"Burst"/"View" anatomy from `01_Grid.svg`'s
    secondary toolbar, as one reusable unit instead of a bare QComboBox with
    a sibling QLabel a caller has to remember to add every time."""

    def __init__(self, caption: str, *, combo: QComboBox | None = None, parent=None) -> None:
        super().__init__(parent)
        palette = theme.current_palette()
        # `combo`, when given, is an ALREADY-CONSTRUCTED QComboBox this
        # widget only wraps/styles/captions - reparented here rather than
        # built fresh, so a caller with existing wiring on that exact
        # instance (signal connections, other code holding a reference to
        # it - e.g. MainWindow's own `_filter_combo`) keeps working
        # unchanged; only where it visually lives moves.
        self.combo = combo if combo is not None else QComboBox(self)
        if combo is not None:
            self.combo.setParent(self)
        self.combo.setMinimumHeight(SECONDARY_HEIGHT - 6)
        self.combo.setStyleSheet(
            f"QComboBox {{ background-color: {palette.panel_bg_secondary}; color: {palette.text_primary}; "
            f"border: 1px solid {palette.border}; border-radius: {RADIUS_SM}px; padding: 2px {SPACING}px; "
            f"font-size: 12px; }}"
            f"QComboBox:hover {{ border-color: {palette.accent}; }}"
            f"QComboBox QAbstractItemView {{ background-color: {palette.panel_bg_secondary}; "
            f"color: {palette.text_primary}; selection-background-color: {palette.hover_bg}; }}"
        )
        caption_label = QLabel(caption, self)
        caption_label.setStyleSheet(f"color: {palette.text_muted}; font-size: 10px; font-weight: 600;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(caption_label)
        layout.addWidget(self.combo)

    @staticmethod
    def cap_width(combo: QComboBox, *, min_chars: int, max_width: int) -> None:
        """Same fix as `main_window._make_toolbar_combo_compact` - a combo
        must not grow to fit its single longest item (a ranking strategy's
        full display name can run 50+ characters)."""
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(min_chars)
        combo.setMaximumWidth(max_width)
        combo.currentTextChanged.connect(combo.setToolTip)
        combo.setToolTip(combo.currentText())


class LabeledSearch(QWidget):
    """The "Search" secondary-toolbar control - a captioned QLineEdit,
    the text-input equivalent of LabeledCombo above."""

    def __init__(self, caption: str, placeholder: str, *, parent=None) -> None:
        super().__init__(parent)
        palette = theme.current_palette()
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setMinimumHeight(SECONDARY_HEIGHT - 6)
        self.edit.setStyleSheet(
            f"QLineEdit {{ background-color: {palette.panel_bg_secondary}; color: {palette.text_primary}; "
            f"border: 1px solid {palette.border}; border-radius: {RADIUS_SM}px; padding: 2px {SPACING}px; "
            f"font-size: 12px; }}"
            f"QLineEdit:focus {{ border-color: {palette.accent}; }}"
        )
        caption_label = QLabel(caption, self)
        caption_label.setStyleSheet(f"color: {palette.text_muted}; font-size: 10px; font-weight: 600;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(caption_label)
        layout.addWidget(self.edit)


class Panel(QFrame):
    """A rounded, `panel_bg`-filled container - the base surface every
    grouped block of controls/content sits on (the sidebar, the secondary
    toolbar row, a Dashboard chart card, the Loupe's side panels). Plain
    QSS on a QFrame, the same mechanism `SummaryCard`
    (analytics_dashboard.py) already used, generalized so it is not
    redefined per screen."""

    def __init__(self, *, radius: int = RADIUS_LG, bordered: bool = True, parent=None) -> None:
        super().__init__(parent)
        palette = theme.current_palette()
        self.setObjectName("dsPanel")
        border = f"1px solid {palette.border}" if bordered else "none"
        self.setStyleSheet(
            f"#dsPanel {{ background-color: {palette.panel_bg}; border-radius: {radius}px; border: {border}; }}"
        )


class StatusLegend(QWidget):
    """A horizontal colored-dot legend for the categories the CURRENTLY
    SELECTED Color mode actually produces - the Grid's bottom legend
    (`01_Grid.svg`).

    Rebuilt by `set_color_source` whenever the mode changes, because the two
    modes have genuinely different vocabularies (see
    `categories_for_color_source`). A legend that always listed Keep/Reject
    was itself part of the confusion: it promised those colors meant a
    decision even while an algorithm mode was painting them from a cutoff.
    """

    def __init__(self, *, color_source: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self._color_source = color_source
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(SPACING * 3)
        self._rebuild()

    def set_color_source(self, color_source: str | None) -> None:
        if color_source == self._color_source:
            return
        self._color_source = color_source
        self._rebuild()

    def categories(self) -> tuple[str, ...]:
        return categories_for_color_source(self._color_source)

    def _rebuild(self) -> None:
        while self._layout.count():
            entry = self._layout.takeAt(0)
            widget = entry.widget()
            if widget is not None:
                widget.deleteLater()
        palette = theme.current_palette()
        self._layout.addStretch(1)
        for status in self.categories():
            color = status_color(palette, status)
            dot = QLabel("●", self)
            dot.setStyleSheet(f"color: {color}; font-size: 12px;")
            label = QLabel(STATUS_LABELS[status], self)
            label.setStyleSheet(f"color: {palette.text_muted}; font-size: 11px;")
            entry_layout = QHBoxLayout()
            entry_layout.setSpacing(6)
            entry_layout.addWidget(dot)
            entry_layout.addWidget(label)
            wrapper = QWidget(self)
            wrapper.setLayout(entry_layout)
            self._layout.addWidget(wrapper)
        self._layout.addStretch(1)


class KpiCard(Panel):
    """One glanceable "803 / Total Images"-style number - the Dashboard's
    KPI row (`03_Analytics_Dashboard.svg`) anatomy: a large bold value over
    a small muted caption, optionally tinted a status color."""

    def __init__(self, title: str, *, value_color: str | None = None, parent=None) -> None:
        super().__init__(parent=parent)
        palette = theme.current_palette()
        color = value_color or palette.text_primary
        self._value_label = QLabel("—", self)
        self._value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: 700; border: none;")
        title_label = QLabel(title, self)
        title_label.setStyleSheet(f"color: {palette.text_muted}; font-size: 11px; font-weight: 600; border: none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING * 2, SPACING * 2, SPACING * 2, SPACING * 2)
        layout.setSpacing(4)
        layout.addWidget(self._value_label)
        layout.addWidget(title_label)

    def set_value(self, value: str) -> None:
        self._value_label.setText(value)


class ChartCard(Panel):
    """A titled panel holding one matplotlib chart - the Dashboard's chart
    tiles (Score Distribution, Top Algorithms, ...). Styled to match the
    dark design system (transparent figure/axes background, palette-driven
    text/grid colors) once, here, so every chart in the dashboard looks
    like it belongs to the same system rather than matplotlib's own
    default light theme leaking through.
    """

    def __init__(self, title: str, *, parent=None) -> None:
        super().__init__(parent=parent)
        palette = theme.current_palette()
        title_label = QLabel(title, self)
        title_label.setStyleSheet(f"color: {palette.text_primary}; font-size: 13px; font-weight: 700; border: none;")

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(4, 2.6), dpi=100)
        self.figure.patch.set_alpha(0.0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet("background: transparent;")
        self.canvas.setMinimumHeight(180)

        self._empty_label = QLabel("No data yet", self)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(f"color: {palette.text_muted}; font-size: 11px; border: none;")
        self._empty_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING * 2, SPACING * 2, SPACING * 2, SPACING * 2)
        layout.setSpacing(SPACING)
        layout.addWidget(title_label)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self._empty_label)

    def style_axes(self, ax) -> None:
        """Apply the dashboard's dark chart theme to one Axes - text/tick/
        spine colors from the current palette, transparent background so
        the surrounding Panel's own fill shows through."""
        palette = theme.current_palette()
        ax.set_facecolor("none")
        ax.tick_params(colors=palette.text_muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(palette.border)
        ax.xaxis.label.set_color(palette.text_muted)
        ax.yaxis.label.set_color(palette.text_muted)
        ax.title.set_color(palette.text_primary)

    def clear(self) -> None:
        self.figure.clear()

    def set_empty(self, empty: bool) -> None:
        self.canvas.setVisible(not empty)
        self._empty_label.setVisible(empty)

    def redraw(self) -> None:
        self.canvas.draw_idle()


class ConfidenceBar(QWidget):
    """A labeled horizontal confidence meter - "Head  0.98" over a filled
    bar (`02_Loupe.svg`'s Elements panel). `color` is the bar's own fill
    (Head/Left Eye/Right Eye each get a distinct color in the Loupe, mirroring
    the overlay markers drawn on the photograph itself)."""

    def __init__(self, label: str, *, parent=None) -> None:
        super().__init__(parent)
        palette = theme.current_palette()
        self._palette = palette
        self._label_widget = QLabel(label, self)
        self._label_widget.setStyleSheet(f"color: {palette.text_muted}; font-size: 11px; font-weight: 500;")
        self._value_label = QLabel("—", self)
        self._value_label.setStyleSheet(f"color: {palette.text_primary}; font-size: 12px; font-weight: 700;")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._label_widget)
        header.addStretch(1)
        header.addWidget(self._value_label)

        self._track = QFrame(self)
        self._track.setFixedHeight(8)
        self._track.setStyleSheet(f"background-color: {palette.panel_bg_secondary}; border-radius: 4px;")
        track_layout = QHBoxLayout(self._track)
        track_layout.setContentsMargins(0, 0, 0, 0)
        track_layout.setSpacing(0)
        self._fill = QFrame(self._track)
        self._fill.setStyleSheet(f"background-color: {palette.secondary_accent}; border-radius: 4px;")
        track_layout.addWidget(self._fill)
        track_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self._track)

    def set_value(self, value: float | None, *, color: str | None = None) -> None:
        """`value` in [0, 1], or None to show the bar as empty/unknown
        ("—", zero-width fill) - never a fabricated 0.0 that would read as
        "confidently zero" instead of "no data"."""
        fill_color = color or self._palette.secondary_accent
        self._fill.setStyleSheet(f"background-color: {fill_color}; border-radius: 4px;")
        if value is None:
            self._value_label.setText("—")
            self._fill.setFixedWidth(0)
            return
        clamped = max(0.0, min(1.0, value))
        self._value_label.setText(f"{clamped:.2f}")
        # Deferred to the next event-loop turn so self._track has its real
        # laid-out width by the time this reads it (at construction/first
        # call the track may still be at its pre-layout size hint).
        from PySide6.QtCore import QTimer

        def _apply() -> None:
            track_width = max(0, self._track.width())
            self._fill.setFixedWidth(int(track_width * clamped))

        QTimer.singleShot(0, _apply)
        _apply()


class AlgorithmResultRow(QFrame):
    """One selectable row in the Loupe's Algorithm Results panel - name +
    score, highlighted when it is the current Elements Source. Clicking
    anywhere on the row selects it (`selected` signal), matching
    `02_Loupe.svg`'s "click an algorithm to switch Elements" interaction."""

    selected = Signal(str)  # strategy_id

    def __init__(self, strategy_id: str, label: str, *, parent=None) -> None:
        super().__init__(parent)
        self.strategy_id = strategy_id
        self._is_active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("algoResultRow")

        self._name_label = QLabel(label, self)
        self._score_label = QLabel("—", self)
        self._score_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._hint_label = QLabel("select", self)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addWidget(self._name_label)
        top_row.addStretch(1)
        top_row.addWidget(self._hint_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING * 2, SPACING, SPACING * 2, SPACING)
        layout.setSpacing(2)
        layout.addLayout(top_row)
        layout.addWidget(self._score_label)
        self._apply_style()

    def set_score_text(self, text: str) -> None:
        self._score_label.setText(text)

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self._apply_style()

    def _apply_style(self) -> None:
        palette = theme.current_palette()
        if self._is_active:
            self.setStyleSheet(
                f"#algoResultRow {{ background-color: {palette.hover_bg}; border: 2px solid {palette.accent}; "
                f"border-radius: {RADIUS_MD}px; }}"
            )
            self._hint_label.setVisible(False)
            self._name_label.setStyleSheet(f"color: {palette.text_primary}; font-size: 13px; font-weight: 700;")
            self._score_label.setStyleSheet(f"color: {palette.text_primary}; font-size: 18px; font-weight: 700; border: none;")
        else:
            self.setStyleSheet(
                f"#algoResultRow {{ background-color: {palette.panel_bg}; border: 1px solid {palette.border}; "
                f"border-radius: {RADIUS_MD}px; }}"
            )
            self._hint_label.setVisible(True)
            self._hint_label.setStyleSheet(f"color: {palette.text_muted}; font-size: 9px; border: none;")
            self._name_label.setStyleSheet(f"color: {palette.text_primary}; font-size: 13px; font-weight: 600; border: none;")
            self._score_label.setStyleSheet(f"color: {palette.text_primary}; font-size: 18px; font-weight: 700; border: none;")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.strategy_id)
        super().mousePressEvent(event)


class NavigationControl(QWidget):
    """"← Previous | N / Total | Next →" - the Loupe's top navigation bar
    (`02_Loupe.svg`). Purely presentational; a caller wires `previous`/
    `next` clicks and calls `set_position`."""

    previous = Signal()
    next = Signal()

    def __init__(self, *, parent=None) -> None:
        super().__init__(parent)
        palette = theme.current_palette()
        self._prev_btn = QPushButton("‹  Previous", self)
        self._next_btn = QPushButton("Next  ›", self)
        for btn in (self._prev_btn, self._next_btn):
            btn.setFlat(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(
                f"QPushButton {{ color: {palette.text_primary}; border: none; font-size: 13px; font-weight: 500; "
                f"background: transparent; padding: {SPACING}px {SPACING * 2}px; }}"
                f"QPushButton:hover {{ color: {palette.accent}; }}"
            )
        self._prev_btn.clicked.connect(self.previous.emit)
        self._next_btn.clicked.connect(self.next.emit)

        self._counter_label = QLabel("0 / 0", self)
        self._counter_label.setStyleSheet(f"color: {palette.text_primary}; font-size: 14px; font-weight: 700;")
        self._counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING * 2, 0, SPACING * 2, 0)
        layout.addWidget(self._prev_btn, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        layout.addWidget(self._counter_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        layout.addWidget(self._next_btn, 0, Qt.AlignmentFlag.AlignRight)

    def set_position(self, index: int, total: int) -> None:
        self._counter_label.setText(f"{index} / {total}")
        self._prev_btn.setEnabled(index > 1)
        self._next_btn.setEnabled(index < total)
