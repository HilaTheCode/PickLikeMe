"""Desktop smoke test - catches import-time regressions (a renamed/removed
function one module still imports by its old name, a circular import, a
missing dependency) before they reach manual testing.

This is deliberately shallow and fast: every other file in tests/ already
covers behaviour in depth. This file's only job is "does the app, and its
two most important dialogs, even come up" - exactly the class of bug that
slipped through in the past (Organize by Species failing with
`cannot import name 'read_species_list' from picklikeme.species.classifier`)
without ever failing a narrower, behaviour-focused test.
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


def test_every_desktop_facing_module_imports_cleanly() -> None:
    """Importing these must never raise - the exact failure mode a stale
    function name (moved/renamed/made private without updating callers)
    produces. Listed explicitly rather than walked via pkgutil, so a
    failure here names the actual broken module directly."""
    import picklikeme.desktop.app  # noqa: F401
    import picklikeme.desktop.application  # noqa: F401
    import picklikeme.desktop.dialogs.analytics_dashboard  # noqa: F401
    import picklikeme.desktop.dialogs.loupe_dialog  # noqa: F401
    import picklikeme.desktop.dialogs.workflow_dialogs  # noqa: F401
    import picklikeme.desktop.main_window  # noqa: F401
    import picklikeme.desktop.services  # noqa: F401
    import picklikeme.species.bioclip_classifier  # noqa: F401
    import picklikeme.species.classifier  # noqa: F401


def test_desktop_launches(tmp_path) -> None:
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    app = QApplication.instance() or QApplication([])
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(), service=service, worker_manager=WorkerManager(),
    )
    window.initialize()
    try:
        assert window.isVisible() is False  # never shown - offscreen smoke test, not a UI test
    finally:
        window.close()
        service.close()


def test_analytics_dashboard_opens(tmp_path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    app = QApplication.instance() or QApplication([])
    dashboard = AnalyticsDashboard(
        analytics_db=tmp_path / "analytics.sqlite", annotations_db=tmp_path / "annotations.sqlite",
        species_db=tmp_path / "species.db", parent=None,
    )
    try:
        assert dashboard is not None
    finally:
        dashboard.close()


def test_organize_by_species_dialog_opens_with_built_in_and_external_lists(tmp_path) -> None:
    """Covers exactly the two configurations regression #1 broke: the
    built-in species list (no path given) and an external one (Browse...
    equivalent - a path given directly to the constructor)."""
    from picklikeme.desktop.dialogs.workflow_dialogs import SpeciesLanguageDialog

    app = QApplication.instance() or QApplication([])

    built_in = SpeciesLanguageDialog()
    try:
        assert built_in.species_list_path() is None
    finally:
        built_in.close()

    species_file = tmp_path / "species.txt"
    species_file.write_text("Kingfisher\nOsprey\n", encoding="utf-8")
    external = SpeciesLanguageDialog(default_species_list_path=str(species_file))
    try:
        assert external.species_list_path() == str(species_file)
    finally:
        external.close()
