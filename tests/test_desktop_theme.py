"""theme.build_qpalette - the fix for the Analytics Dashboard's tables/lists
rendering with a hardcoded-looking white background even under the dark
app stylesheet: QSS alone (build_stylesheet) only reaches widget types it
explicitly names, but a QPalette is inherited by every widget, styled or
not. See theme.py's own docstring on build_qpalette for the full story.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


def test_dark_palette_uses_dark_colors_for_base_and_window() -> None:
    from PySide6.QtGui import QColor, QPalette

    from picklikeme.desktop import theme

    app = QApplication.instance() or QApplication([])
    qp = theme.build_qpalette(theme.DARK)

    assert qp.color(QPalette.ColorRole.Window) == QColor(theme.DARK.window_bg)
    assert qp.color(QPalette.ColorRole.Base) == QColor(theme.DARK.panel_bg)
    assert qp.color(QPalette.ColorRole.Text) == QColor(theme.DARK.text_primary)
    # The whole point: Base is the role QTableWidget/QListWidget/QTreeWidget
    # paint their background from - it must not be left at Qt's default
    # (light) value just because no QSS rule named the widget type.
    assert qp.color(QPalette.ColorRole.Base).lightness() < 128


def test_light_palette_uses_light_colors_for_base_and_window() -> None:
    from PySide6.QtGui import QColor, QPalette

    from picklikeme.desktop import theme

    app = QApplication.instance() or QApplication([])
    qp = theme.build_qpalette(theme.LIGHT)

    assert qp.color(QPalette.ColorRole.Window) == QColor(theme.LIGHT.window_bg)
    assert qp.color(QPalette.ColorRole.Base) == QColor(theme.LIGHT.panel_bg)
    assert qp.color(QPalette.ColorRole.Base).lightness() > 128


def test_dark_and_light_palettes_differ() -> None:
    from PySide6.QtGui import QPalette

    from picklikeme.desktop import theme

    app = QApplication.instance() or QApplication([])
    dark = theme.build_qpalette(theme.DARK)
    light = theme.build_qpalette(theme.LIGHT)

    assert dark.color(QPalette.ColorRole.Base) != light.color(QPalette.ColorRole.Base)
