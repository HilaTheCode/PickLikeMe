"""desktop/dialogs/dialog_utils.py - the shared maximize-button + remembered-
geometry polish (Manual QA Phase 12) applied to every small workflow
dialog (Rank/AlgorithmParameters/SpeciesLanguage/Preferences/AutoCrop/
SetUserDecisionsBySubfolders), factored out once here rather than
duplicated six times.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication, QDialog
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


def _settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_polish_dialog_adds_the_maximize_and_minimize_hints(tmp_path) -> None:
    from picklikeme.desktop.dialogs.dialog_utils import polish_dialog

    QApplication.instance() or QApplication([])
    dialog = QDialog()
    polish_dialog(dialog, geometry_key="test/geometry", settings=_settings(tmp_path))

    flags = dialog.windowFlags()
    assert bool(flags & Qt.WindowType.WindowMaximizeButtonHint)
    assert bool(flags & Qt.WindowType.WindowMinimizeButtonHint)


def test_geometry_is_restored_from_a_previous_session(tmp_path) -> None:
    from picklikeme.desktop.dialogs.dialog_utils import polish_dialog

    QApplication.instance() or QApplication([])
    settings = _settings(tmp_path)

    first = QDialog()
    first.resize(640, 480)
    polish_dialog(first, geometry_key="test/geometry", settings=settings)
    first.done(QDialog.DialogCode.Accepted)  # emits finished -> saves geometry

    second = QDialog()
    polish_dialog(second, geometry_key="test/geometry", settings=settings)

    assert second.size().width() == 640
    assert second.size().height() == 480


def test_geometry_saves_on_reject_and_on_the_windows_own_close_alike(tmp_path) -> None:
    """`finished` covers accept/reject/Escape/the title-bar close button -
    not just an explicit OK click."""
    from picklikeme.desktop.dialogs.dialog_utils import polish_dialog

    QApplication.instance() or QApplication([])
    settings = _settings(tmp_path)

    dialog = QDialog()
    dialog.resize(555, 333)
    polish_dialog(dialog, geometry_key="test/geometry", settings=settings)
    dialog.reject()  # the Escape-key / Cancel path

    assert settings.value("test/geometry") is not None


def test_a_dialog_that_has_never_been_shown_before_gets_no_restore_call(tmp_path) -> None:
    """No stored geometry for this key yet - restoreGeometry must simply
    never be called, not called with None (which Qt would reject)."""
    from unittest import mock

    from picklikeme.desktop.dialogs.dialog_utils import polish_dialog

    QApplication.instance() or QApplication([])
    dialog = QDialog()
    with mock.patch.object(dialog, "restoreGeometry") as restore_spy:
        polish_dialog(dialog, geometry_key="test/never_saved", settings=_settings(tmp_path))

    restore_spy.assert_not_called()


def test_different_geometry_keys_never_cross_talk(tmp_path) -> None:
    from picklikeme.desktop.dialogs.dialog_utils import polish_dialog

    QApplication.instance() or QApplication([])
    settings = _settings(tmp_path)

    rank_dialog = QDialog()
    rank_dialog.resize(500, 400)
    polish_dialog(rank_dialog, geometry_key="dialogs/rank_geometry", settings=settings)
    rank_dialog.accept()

    prefs_dialog = QDialog()
    polish_dialog(prefs_dialog, geometry_key="dialogs/preferences_geometry", settings=settings)

    # The Preferences dialog never had its own geometry saved - it must not
    # pick up Rank's, even though both share one QSettings store.
    assert prefs_dialog.size().width() != 500 or prefs_dialog.size().height() != 400
