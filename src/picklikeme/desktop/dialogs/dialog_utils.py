"""Small polish helpers shared by every workflow dialog (Manual QA Phase
12 - General UX): a maximize/minimize window button and remembered window
geometry - the same two things AnalyticsDashboard and LoupeDialog already
did for themselves individually, factored out once here so the smaller
Rank/Species/Preferences/AutoCrop/Set-User-Decisions dialogs get them too
without duplicating the same few lines six times.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QDialog


def polish_dialog(dialog: QDialog, *, geometry_key: str, settings: QSettings | None = None) -> None:
    """Call once, after a dialog has finished building its layout.

    Adds the maximize/minimize hints Qt hides by default for a `QDialog`
    on some platforms (Windows in particular - the exact issue
    AnalyticsDashboard/LoupeDialog's own module docstrings already
    describe), restores any previously-saved geometry for `geometry_key`,
    and arranges for the current geometry to be saved back whenever the
    dialog closes.

    Hooked to `finished` (emitted for accept/reject/Escape/the window's own
    close button alike), not `closeEvent` - one connection here covers
    every way a `QDialog` can close without each of the six dialogs needing
    its own override.
    """
    dialog.setWindowFlags(
        dialog.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint
    )
    store = settings if settings is not None else QSettings("PeakPic", "PeakPicDesktop")
    geometry = store.value(geometry_key)
    if geometry is not None:
        dialog.restoreGeometry(geometry)
    dialog.finished.connect(lambda _result: store.setValue(geometry_key, dialog.saveGeometry()))
