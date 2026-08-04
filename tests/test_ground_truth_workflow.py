"""Desktop wiring for "Set User Decisions by Subfolders..." - the full
MainWindow flow: open the dialog, scan the folders (preview), confirm, and
apply. `ground_truth.build_plan`/`apply_plan` themselves are tested in
isolation in test_ground_truth.py; this file covers only the desktop layer
on top: the confirmation prompt, the progress-bar-backed background run,
and the result summary/state refresh.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication, QMessageBox
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _window(tmp_path):
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(), service=service, worker_manager=WorkerManager(),
    )
    window.initialize()
    return window, service


def _make_jpeg(path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color="blue").save(path, format="JPEG")


def _use_synchronous_run_with_progress(monkeypatch) -> None:
    """Bypasses the real background QThread run_with_progress starts -
    irrelevant to what these tests check, and it cannot run headless
    without its own event loop (same rationale as test_ranking_ui.py's own
    fake_run_with_progress)."""
    import picklikeme.desktop.main_window as main_window_module

    def fake_run_with_progress(parent, title, func, *, on_success, on_error=None):
        del parent, title, on_error
        on_success(func())

        class _FakeThread:
            class _Signal:
                def connect(self, *_args, **_kwargs) -> None:
                    pass

            finished = _Signal()

        return _FakeThread()

    monkeypatch.setattr(main_window_module, "run_with_progress", fake_run_with_progress)


def test_confirm_and_apply_writes_decisions_and_refreshes_state(app, tmp_path, monkeypatch) -> None:
    from picklikeme.analyzer.annotations import REVIEW_KEEP

    keep_folder = tmp_path / "Keep"
    _make_jpeg(keep_folder / "a.jpg")

    window, service = _window(tmp_path)
    try:
        _use_synchronous_run_with_progress(monkeypatch)
        monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
        info_calls = []
        monkeypatch.setattr(
            QMessageBox, "information", staticmethod(lambda *a, **k: info_calls.append(a) or QMessageBox.StandardButton.Ok)
        )

        preview = service.preview_ground_truth_import(keep_folder=keep_folder)
        window._confirm_and_apply_ground_truth_import(preview)

        decisions = {row["image_path"]: row["decision"] for row in service.store.review_decisions()}
        assert decisions[str(keep_folder / "a.jpg")] == REVIEW_KEEP
        assert info_calls  # the result summary was shown
    finally:
        window.close()
        service.close()


def test_declining_the_confirmation_writes_nothing(app, tmp_path, monkeypatch) -> None:
    keep_folder = tmp_path / "Keep"
    _make_jpeg(keep_folder / "a.jpg")

    window, service = _window(tmp_path)
    try:
        _use_synchronous_run_with_progress(monkeypatch)
        monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))

        preview = service.preview_ground_truth_import(keep_folder=keep_folder)
        window._confirm_and_apply_ground_truth_import(preview)

        assert service.store.review_decision_count() == 0
    finally:
        window.close()
        service.close()


def test_nothing_to_change_shows_information_without_asking_to_confirm(app, tmp_path, monkeypatch) -> None:
    from picklikeme.analyzer.annotations import REVIEW_KEEP

    keep_folder = tmp_path / "Keep"
    path = keep_folder / "a.jpg"
    _make_jpeg(path)

    window, service = _window(tmp_path)
    try:
        _use_synchronous_run_with_progress(monkeypatch)
        service.store.set_review_decision(path, REVIEW_KEEP)  # already matches
        question_calls = []
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: question_calls.append(a) or QMessageBox.StandardButton.Yes),
        )
        info_calls = []
        monkeypatch.setattr(
            QMessageBox, "information", staticmethod(lambda *a, **k: info_calls.append(a) or QMessageBox.StandardButton.Ok)
        )

        preview = service.preview_ground_truth_import(keep_folder=keep_folder)
        window._confirm_and_apply_ground_truth_import(preview)

        assert question_calls == []  # nothing to change - never asks to confirm
        assert info_calls  # but still tells the photographer so
    finally:
        window.close()
        service.close()


def test_ground_truth_menu_action_exists_and_works_without_an_open_folder(app, tmp_path, monkeypatch) -> None:
    """Unlike every other Tools action, this one must not require
    self.state.current_folder - it operates entirely on the folders chosen
    inside its own dialog."""
    from picklikeme.desktop.dialogs.workflow_dialogs import SetUserDecisionsBySubfoldersDialog

    keep_folder = tmp_path / "Keep"
    _make_jpeg(keep_folder / "a.jpg")

    window, service = _window(tmp_path)
    try:
        assert window.state.current_folder is None

        class _FakeDialog:
            DialogCode = SetUserDecisionsBySubfoldersDialog.DialogCode

            def __init__(self, **kwargs):
                pass

            def exec(self):
                return self.DialogCode.Accepted

            def keep_folder(self):
                return str(keep_folder)

            def reject_folder(self):
                return None

            def neutral_folder(self):
                return None

        import picklikeme.desktop.main_window as main_window_module

        _use_synchronous_run_with_progress(monkeypatch)
        monkeypatch.setattr(main_window_module, "SetUserDecisionsBySubfoldersDialog", _FakeDialog)
        monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
        monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

        window._set_user_decisions_by_subfolders()

        assert service.store.review_decision_count() == 1
    finally:
        window.close()
        service.close()
