"""Semantic color palette and stylesheet for the desktop shell.

One `Palette` per theme (dark, light) so Keep/Reject/Neutral/Selection stay
the same green/red/gray/blue *hue family* everywhere - gallery cards, status
labels, badges - while each theme tunes the exact shades for contrast
against its own background. `current_palette()` is read fresh by anything
that paints (ThumbnailCardDelegate.paint, dialogs at construction time), so
switching themes takes effect without restarting the app - no per-widget
notification wiring needed.

The Loupe's image viewer intentionally stays dark-chrome regardless of the
app theme (see loupe_dialog.py) - the same convention as Lightroom, Capture
One and DxO PhotoLab, where the image canvas needs a neutral dark backdrop
to judge exposure/color regardless of the surrounding UI's theme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtGui import QPalette


@dataclass(frozen=True)
class Palette:
    name: str
    window_bg: str
    panel_bg: str
    text_primary: str
    text_muted: str
    border: str
    hover_bg: str
    selection_border: str
    accent: str
    keep_bg: str
    keep_fg: str
    reject_bg: str
    reject_fg: str
    neutral_bg: str
    neutral_fg: str
    # "Skipped" (see thumbnail_delegate.py's _get_background_color) - an
    # image the CURRENTLY SELECTED Color Source never even touched (no
    # score, no recorded filter reason), distinct from "Filtered Out"
    # (that module DID look at it and explicitly excluded it - see
    # filtered_bg/filtered_fg below), from ordinary Neutral/Review (scored,
    # just not yet decided) and from Reject (scored and explicitly not in
    # the keep cutoff). Purple - a hue family not otherwise used by any
    # other status, matching this file's own "perceptibly separated by hue
    # AND saturation" rule above.
    skipped_bg: str
    skipped_fg: str
    # "Filtered Out" - the selected Color Source's own module DID examine
    # this image and explicitly excluded it (a recorded filter reason - see
    # `ImageItem.filter_reasons`), as opposed to "Skipped" above (never
    # touched at all). Muted gray - deliberately desaturated relative to
    # every other status color, since "the algorithm looked and passed" is
    # informational, not a Keep/Reject/Review verdict needing the same
    # visual weight.
    filtered_bg: str = "#2a2f36"
    filtered_fg: str = "#8B95A0"
    # A second control-surface tone, one step lighter than panel_bg - the
    # PeakPick design system's "Secondary panel/control" token, used for
    # buttons/combos/input chrome that sits ON TOP of a panel_bg surface
    # (so the two remain visually distinct, not one flat plane). Defaults
    # to panel_bg's own DARK/LIGHT values via __post_init__ below for any
    # Palette that does not set it explicitly, so this stays optional.
    panel_bg_secondary: str = ""
    # The design system's secondary "information" accent (distinct from
    # `accent`, the one Keep/Select/primary-action color) - used for
    # secondary data series (a second confidence bar, a second chart
    # series) that must read as related-but-different from the primary
    # accent, never confusable with a review-status color.
    secondary_accent: str = "#5AA7FF"

    def __post_init__(self) -> None:
        if not self.panel_bg_secondary:
            object.__setattr__(self, "panel_bg_secondary", self.panel_bg)


DARK = Palette(
    name="dark",
    # PeakPick design system tokens (docs/UX Design/20260810/Ver1.0/
    # PeakPick_UI_Design_Spec.md) - background/primary panel/secondary
    # panel/divider/text/accent, applied here rather than duplicated so
    # every dialog that already reads `theme.current_palette()` (not just
    # Grid/Loupe/Dashboard) picks up the same coherent system.
    window_bg="#0B1014",
    panel_bg="#141B21",
    panel_bg_secondary="#1B242C",
    text_primary="#F2F5F7",
    text_muted="#9AA8B2",
    border="#2B3740",
    hover_bg="#202A32",
    selection_border="#F5C542",
    accent="#F5C542",
    secondary_accent="#5AA7FF",
    # Kept perceptibly separated from each other and from filtered_bg/
    # skipped_bg by hue AND saturation, not just luminance - see this
    # dataclass's own docstring/comments above.
    keep_bg="#173327",
    keep_fg="#42CC8E",
    reject_bg="#3A1E1E",
    reject_fg="#EF4444",
    # "Review" (undecided, no algorithm opinion, or Color Source = Review
    # Status) - the same gold as `accent`, per the design spec's five-
    # category legend (Keep/Review/Reject/Filtered Out/Skipped).
    neutral_bg="#332B14",
    neutral_fg="#F5C542",
    skipped_bg="#241B33",
    skipped_fg="#9B6BDB",
    filtered_bg="#242A30",
    filtered_fg="#8B95A0",
)

LIGHT = Palette(
    name="light",
    window_bg="#f5f5f5",
    panel_bg="#ffffff",
    text_primary="#212121",
    text_muted="#757575",
    border="#d0d0d0",
    hover_bg="#eeeeee",
    selection_border="#2196f3",
    accent="#2196f3",
    keep_bg="#e0f2e9",
    keep_fg="#2e7d32",
    reject_bg="#fde0e0",
    reject_fg="#c62828",
    neutral_bg="#f0f0f0",
    neutral_fg="#757575",
    skipped_bg="#fdf0e0",
    skipped_fg="#e65100",
)

PALETTES: dict[str, Palette] = {"dark": DARK, "light": LIGHT}
DEFAULT_THEME = "dark"

_current_theme_name = DEFAULT_THEME


def set_theme(name: str) -> None:
    global _current_theme_name
    _current_theme_name = name if name in PALETTES else DEFAULT_THEME


def current_theme_name() -> str:
    return _current_theme_name


def current_palette() -> Palette:
    return PALETTES[_current_theme_name]


def build_qpalette(palette: Palette) -> "QPalette":
    """A real QPalette matching `palette`'s colors - applied alongside
    `build_stylesheet`'s QSS (see MainWindow._apply_theme), not instead of
    it. QSS alone only reaches widget types a rule explicitly names;
    `QTableWidget`, `QHeaderView`, `QListWidget`, `QTabWidget` and friends
    have no rule in `build_stylesheet` and were rendering with Qt's default
    (light) palette even while the app's own chrome went dark - exactly the
    "hardcoded white backgrounds" the Analytics Dashboard's tables showed.
    A QPalette is inherited by every widget, styled or not, including ones
    added later with no QSS rule of their own - the actual fix "Qt palette
    everywhere" asks for, not a per-widget stylesheet patch.
    """
    from PySide6.QtGui import QColor, QPalette

    qp = QPalette()
    window = QColor(palette.window_bg)
    base = QColor(palette.panel_bg)
    text = QColor(palette.text_primary)
    muted = QColor(palette.text_muted)
    accent = QColor(palette.accent)
    border = QColor(palette.border)

    qp.setColor(QPalette.ColorRole.Window, window)
    qp.setColor(QPalette.ColorRole.WindowText, text)
    qp.setColor(QPalette.ColorRole.Base, base)
    qp.setColor(QPalette.ColorRole.AlternateBase, window)
    qp.setColor(QPalette.ColorRole.Text, text)
    qp.setColor(QPalette.ColorRole.Button, base)
    qp.setColor(QPalette.ColorRole.ButtonText, text)
    qp.setColor(QPalette.ColorRole.ToolTipBase, base)
    qp.setColor(QPalette.ColorRole.ToolTipText, text)
    qp.setColor(QPalette.ColorRole.PlaceholderText, muted)
    qp.setColor(QPalette.ColorRole.Highlight, accent)
    qp.setColor(QPalette.ColorRole.HighlightedText, window if palette.name == "dark" else QColor("#ffffff"))
    qp.setColor(QPalette.ColorRole.Link, accent)
    qp.setColor(QPalette.ColorRole.Mid, border)
    qp.setColor(QPalette.ColorRole.Dark, border)

    disabled_text = muted
    qp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    qp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    qp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
    return qp


def build_stylesheet(palette: Palette) -> str:
    """Application-wide QSS covering the main shell's chrome. Dialog- or
    widget-specific styling (e.g. the Loupe's dark overlay bar) layers its
    own stylesheet on top via objectName selectors, unaffected by this."""
    return f"""
    QMainWindow, QDialog {{
        background-color: {palette.window_bg};
        color: {palette.text_primary};
    }}
    QWidget {{
        color: {palette.text_primary};
    }}
    QToolBar {{
        background-color: {palette.panel_bg};
        border: none;
        spacing: 4px;
        padding: 4px;
    }}
    QToolBar QLabel {{
        color: {palette.text_muted};
    }}
    QToolButton {{
        background-color: transparent;
        color: {palette.text_primary};
        border: none;
        padding: 4px 8px;
        border-radius: 3px;
    }}
    QToolButton:hover {{
        background-color: {palette.hover_bg};
    }}
    QMenuBar {{
        background-color: {palette.panel_bg};
        color: {palette.text_primary};
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 4px 10px;
    }}
    QMenuBar::item:selected {{
        background-color: {palette.hover_bg};
    }}
    QMenu {{
        background-color: {palette.panel_bg};
        color: {palette.text_primary};
        border: 1px solid {palette.border};
    }}
    QMenu::item:selected {{
        background-color: {palette.hover_bg};
    }}
    QStatusBar {{
        background-color: {palette.panel_bg};
        color: {palette.text_muted};
    }}
    QPushButton {{
        background-color: {palette.panel_bg};
        color: {palette.text_primary};
        border: 1px solid {palette.border};
        border-radius: 3px;
        padding: 4px 10px;
    }}
    QPushButton:hover {{
        background-color: {palette.hover_bg};
    }}
    QPushButton:pressed {{
        background-color: {palette.border};
    }}
    QComboBox, QLineEdit, QDoubleSpinBox {{
        background-color: {palette.panel_bg};
        color: {palette.text_primary};
        border: 1px solid {palette.border};
        border-radius: 3px;
        padding: 2px 6px;
    }}
    QListView {{
        background-color: {palette.window_bg};
        border: none;
    }}
    QProgressBar {{
        background-color: {palette.panel_bg};
        color: {palette.text_primary};
        border: 1px solid {palette.border};
        border-radius: 3px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {palette.accent};
        border-radius: 2px;
    }}
    QTableWidget, QTableView, QListWidget, QTreeWidget {{
        background-color: {palette.panel_bg};
        alternate-background-color: {palette.window_bg};
        color: {palette.text_primary};
        gridline-color: {palette.border};
        border: 1px solid {palette.border};
        border-radius: 3px;
    }}
    QTableWidget::item:selected, QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {palette.selection_border};
        color: {palette.window_bg};
    }}
    QHeaderView::section {{
        background-color: {palette.window_bg};
        color: {palette.text_primary};
        border: none;
        border-right: 1px solid {palette.border};
        border-bottom: 1px solid {palette.border};
        padding: 4px 6px;
        font-weight: 600;
    }}
    QTabWidget::pane {{
        border: 1px solid {palette.border};
        background-color: {palette.window_bg};
    }}
    QTabBar::tab {{
        background-color: {palette.panel_bg};
        color: {palette.text_muted};
        border: 1px solid {palette.border};
        border-bottom: none;
        padding: 6px 14px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {palette.window_bg};
        color: {palette.text_primary};
    }}
    QSplitter::handle {{
        background-color: {palette.border};
    }}
    """
