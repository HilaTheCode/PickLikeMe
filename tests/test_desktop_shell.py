import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from picklikeme.desktop.app import DesktopApplication
from picklikeme.desktop.application import ApplicationState, WorkerManager
from picklikeme.desktop.main_window import MainWindow
from picklikeme.desktop.services import ReviewService
from picklikeme.desktop.settings import DesktopSettings

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]


def test_application_state_tracks_folder_and_status() -> None:
    state = ApplicationState()
    state.current_folder = "/tmp/shoot"
    state.status_message = "Folder loaded"
    state.image_count = 42

    assert state.current_folder == "/tmp/shoot"
    assert state.image_count == 42
    assert state.status_message == "Folder loaded"


def test_worker_manager_tracks_jobs() -> None:
    manager = WorkerManager()
    job_id = manager.register_job("folder-load")

    assert job_id in manager.jobs
    assert manager.jobs[job_id]["name"] == "folder-load"


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_main_window_can_be_created() -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=":memory:")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())
    window.initialize()

    assert window.isVisible() is False
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_desktop_application_can_initialize_and_open_folder(tmp_path) -> None:
    app = DesktopApplication(db_path=tmp_path / "annotations.sqlite")
    app.initialize()

    folder = tmp_path / "shoot"
    folder.mkdir()
    (folder / "demo.jpg").write_bytes(b"fake")

    result = app.open_folder(str(folder))
    assert result["input_folder"] == str(folder.resolve())
    assert app.state.current_folder == str(folder)
    app.window.close()
    app.service.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_main_window_can_apply_review_status(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    folder = tmp_path / "review"
    folder.mkdir()
    image_path = folder / "demo.jpg"
    image_path.write_bytes(b"fake")

    window.open_folder(str(folder))
    window.apply_review_status("keep")

    assert state.current_selection == [str(image_path.resolve())]
    assert service.session.images[0].review_status == "keep"

    window.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_open_folder_cancel_restores_previous_session(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    previous_folder = tmp_path / "previous"
    previous_folder.mkdir()
    (previous_folder / "old.jpg").write_bytes(b"old")
    window.open_folder(str(previous_folder))

    new_folder = tmp_path / "new"
    new_folder.mkdir()
    (new_folder / "new.jpg").write_bytes(b"new")

    window._folder_load_snapshot = {"folder": str(previous_folder)}
    window._open_folder_in_progress = True
    window._cancel_open_folder()

    assert state.current_folder == str(previous_folder)
    assert window._open_folder_in_progress is False
    window.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_recent_folders_are_persisted_in_qsettings(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    window._save_recent_folder(str(tmp_path / "one"))
    window._save_recent_folder(str(tmp_path / "two"))

    persisted = window._settings.value("recent_folders", [])
    assert persisted[0] == str((tmp_path / "two").resolve())
    assert persisted[1] == str((tmp_path / "one").resolve())

    window.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_open_folder_failure_restores_previous_session(monkeypatch, tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    previous_folder = tmp_path / "previous"
    previous_folder.mkdir()
    (previous_folder / "old.jpg").write_bytes(b"old")
    window.open_folder(str(previous_folder))

    class DummyThread:
        def __init__(self) -> None:
            self.finished = SimpleNamespace(connect=lambda *args, **kwargs: None)

    def fake_run_in_background(parent, func, *, on_finished=None, on_failed=None):
        on_failed("boom")
        return DummyThread()

    monkeypatch.setattr("picklikeme.desktop.main_window.run_in_background", fake_run_in_background)
    # _start_open_folder now clears _open_folder_in_progress synchronously
    # (P1 fix) instead of waiting for the loading timer to tick, so this
    # second call actually reaches _handle_open_folder_failure's
    # QMessageBox.warning() instead of being silently dropped by the
    # "already loading" guard. That warning() is a real, correct modal for
    # an interactive session, but exec()s a nested event loop that never
    # returns under the offscreen test platform - stub it out, same as
    # test_desktop_workflow.py already does for loupe_dialog's QMessageBox.
    monkeypatch.setattr("picklikeme.desktop.main_window.QMessageBox.warning", staticmethod(lambda *a, **k: None))
    window._folder_load_snapshot = {"folder": str(previous_folder)}
    window._start_open_folder(str(tmp_path / "new"))

    assert state.current_folder == str(previous_folder)
    window.close()
    service.close()
    app.quit()
