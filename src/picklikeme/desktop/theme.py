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


DARK = Palette(
    name="dark",
    window_bg="#1e1e1e",
    panel_bg="#252526",
    text_primary="#e0e0e0",
    text_muted="#9e9e9e",
    border="#3c3c3c",
    hover_bg="#2d2d30",
    selection_border="#4fc3f7",
    accent="#4fc3f7",
    # Kept perceptibly separated from neutral_bg and from each other by hue
    # AND saturation, not just luminance - a screenshot of the first pass
    # (keep_bg #1b3a1f / reject_bg #3a1c1c / neutral_bg #2a2a2a) showed all
    # three clustering into the same dark-gray band at a glance, defeating
    # "immediately recognizable" review status on the gallery cards.
    keep_bg="#1d4a27",
    keep_fg="#66bb6a",
    reject_bg="#4a1f1f",
    reject_fg="#ef5350",
    neutral_bg="#242424",
    neutral_fg="#9e9e9e",
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
    """
