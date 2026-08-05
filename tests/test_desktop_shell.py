import os
import time
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
def test_about_dialog_shows_the_actual_running_git_commit_and_version(monkeypatch) -> None:
    """The concrete, verifiable "am I running the build I think I am" check
    - reads the real git commit fresh (never a value baked in at process
    start), so a photographer can compare it against `git rev-parse HEAD`
    themselves rather than trusting a visual impression of the UI."""
    import subprocess

    from PySide6.QtWidgets import QMessageBox

    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=":memory:")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())
    window.initialize()

    shown = {}
    monkeypatch.setattr(
        QMessageBox, "about",
        staticmethod(lambda parent, title, text: shown.update(title=title, text=text)),
    )
    window._show_about()

    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert actual_commit in shown["text"]
    assert "Git commit:" in shown["text"]
    assert "Application version:" in shown["text"]
    # Manual QA Phase 13: Build Timestamp, Python Version, Source Path -
    # the two fields the About dialog was previously missing.
    import platform

    assert "Build timestamp" in shown["text"]
    assert "Python version:" in shown["text"]
    assert platform.python_version() in shown["text"]
    assert "Running from:" in shown["text"]

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


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_changing_a_review_decision_never_moves_the_gallery_scroll_position(tmp_path) -> None:
    """Regression test: a decision change (a single card's Keep/Reject
    button, or the keyboard/toolbar bulk path - both funnel through
    apply_review_status) used to jump the gallery straight back to the top
    every time, because _apply_filter's ImageModel.set_items() always does
    a full beginResetModel()/endResetModel() - Qt has no way to know "this
    is the same content, just redecorated" across a full reset, and drops
    scroll position to 0. Reviewing anything past the first screenful of a
    large folder was unusable as a result. The fix restores the scrollbar
    value on the next event-loop turn (the reset's layout recalculation is
    not necessarily synchronous, so restoring immediately can be clamped
    against a stale range and silently discarded)."""
    from PIL import Image

    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())
    window.resize(500, 400)
    window.show()

    folder = tmp_path / "review"
    folder.mkdir()
    paths = []
    for i in range(80):  # enough rows that the grid genuinely scrolls
        path = folder / f"img_{i:03d}.jpg"
        Image.new("RGB", (8, 8), color="blue").save(path, format="JPEG")
        paths.append(path)

    window.open_folder(str(folder))
    app.processEvents()

    scrollbar = window._gallery_view.verticalScrollBar()
    assert scrollbar.maximum() > 0, "the seeded gallery must actually be tall enough to scroll"
    scrollbar.setValue(scrollbar.maximum() // 2)
    app.processEvents()
    scrolled_to = scrollbar.value()
    assert scrolled_to > 0

    window.apply_review_status("reject", paths=[str(paths[5])])
    app.processEvents()  # lets the deferred restore (QTimer.singleShot(0, ...)) fire

    assert service.session._image_for(str(paths[5])).review_status == "reject"
    assert scrollbar.value() == scrolled_to

    window.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_moving_the_keep_threshold_spinner_immediately_recolors_without_moving_scroll(tmp_path) -> None:
    """Manual QA Phase 11: "Threshold changes should immediately recolor
    the gallery." Moving the Keep Threshold spinner must update
    ReviewSession.keep_percent right away (so ImageItem.algorithm_suggestion
    - and therefore Priority #2 of the coloring policy - reflects it on the
    very next paint) without requiring the separate, confirmed "Apply
    Cutoff" action, and without moving the gallery scroll position -
    exactly the same scroll-preserving refresh Issue 1 already established
    for a decision change."""
    from PIL import Image

    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())
    window.resize(500, 400)
    window.show()

    folder = tmp_path / "review"
    folder.mkdir()
    for i in range(80):
        Image.new("RGB", (8, 8), color="blue").save(folder / f"img_{i:03d}.jpg", format="JPEG")

    window.open_folder(str(folder))
    app.processEvents()

    scrollbar = window._gallery_view.verticalScrollBar()
    assert scrollbar.maximum() > 0
    scrollbar.setValue(scrollbar.maximum() // 2)
    app.processEvents()
    scrolled_to = scrollbar.value()

    assert service.session.keep_percent != 42.0
    window._cutoff_spin.setValue(42.0)  # triggers _on_cutoff_preview_changed
    app.processEvents()

    assert service.session.keep_percent == 42.0
    assert scrollbar.value() == scrolled_to

    window.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_moving_the_keep_threshold_spinner_before_a_folder_is_open_does_not_crash(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    window._cutoff_spin.setValue(20.0)  # must not raise - no folder open yet

    window.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_changing_color_source_never_moves_the_gallery_scroll_position(tmp_path) -> None:
    """Same Issue-1 scroll-preservation guarantee, extended to switching
    Color Source (Phase 11): recoloring the gallery this way also goes
    through a full ImageModel reset, so it must not jump scroll either."""
    from PIL import Image

    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())
    window.resize(500, 400)
    window.show()

    folder = tmp_path / "review"
    folder.mkdir()
    for i in range(80):
        Image.new("RGB", (8, 8), color="blue").save(folder / f"img_{i:03d}.jpg", format="JPEG")

    window.open_folder(str(folder))
    app.processEvents()

    scrollbar = window._gallery_view.verticalScrollBar()
    assert scrollbar.maximum() > 0
    scrollbar.setValue(scrollbar.maximum() // 2)
    app.processEvents()
    scrolled_to = scrollbar.value()

    classic_index = window._color_combo.findData("classic-vision")
    assert classic_index >= 0
    window._color_combo.setCurrentIndex(classic_index)
    app.processEvents()

    assert scrollbar.value() == scrolled_to

    window.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_open_folder_dialog_starts_at_the_last_opened_folder_not_home(tmp_path, monkeypatch) -> None:
    """Manual QA Issue 2: Open Folder must never default to Desktop/
    Documents/the project folder - it must reopen wherever Open Folder was
    last actually used, via _default_folder_for_dialog(), which already
    existed and read the right QSettings key but was never wired into the
    dialog call itself."""
    from unittest import mock

    from picklikeme.desktop import main_window as main_window_module

    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    last_folder = tmp_path / "previously_opened"
    last_folder.mkdir()
    window._settings.setValue("last_opened_folder", str(last_folder))

    with mock.patch.object(
        main_window_module.QFileDialog, "getExistingDirectory", return_value=""
    ) as dialog_spy:
        window._open_folder_dialog()

    dialog_spy.assert_called_once_with(window, "Open Folder", str(last_folder))
    window.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_open_folder_dialog_falls_back_to_home_before_any_folder_was_ever_opened(tmp_path) -> None:
    from unittest import mock

    from picklikeme.desktop import main_window as main_window_module

    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())
    # window._settings is the real, process-wide QSettings (see
    # MainWindow.__init__) - shared with every other test in this file, so
    # a prior test's "last_opened_folder" would otherwise leak in here.
    # Remove it explicitly to simulate "before any folder was ever opened".
    window._settings.remove("last_opened_folder")

    with mock.patch.object(
        main_window_module.QFileDialog, "getExistingDirectory", return_value=""
    ) as dialog_spy:
        window._open_folder_dialog()

    dialog_spy.assert_called_once_with(window, "Open Folder", str(Path.home()))
    window.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_thumbnail_decode_failure_can_be_retried(tmp_path) -> None:
    """Regression: a thumbnail that fails to decode (here, a file that is
    not a real image, so review_thumbnail/QPixmap load fails) used to get
    permanently stuck in _thumbnails_loading, since ThumbnailLoadTask.run()
    emitted nothing at all on failure - not even the fact that it had
    failed - so _load_thumbnail's own dedup guard could never let it be
    requested again for the rest of the session."""
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    folder = tmp_path / "review"
    folder.mkdir()
    bad_path = folder / "not_really_an_image.jpg"
    bad_path.write_bytes(b"fake")

    window.open_folder(str(folder))
    resolved = str(bad_path.resolve())
    key = (resolved, False)

    assert window._load_thumbnail(resolved) is None  # cache miss - queues the decode
    assert key in window._thumbnails_loading

    deadline = time.monotonic() + 5.0
    while key in window._thumbnails_loading and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert key not in window._thumbnails_loading, "a failed decode is stuck and can never be retried"
    assert window._load_thumbnail(resolved) is None
    assert key in window._thumbnails_loading, "a retry after failure was not queued"

    window.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_closing_the_window_drains_in_flight_thumbnail_decodes(tmp_path) -> None:
    """Regression: closing the main window used to hang the whole process
    indefinitely (python -m picklikeme.desktop never returned to the shell)
    whenever a QThreadPool.globalInstance() thumbnail decode was still
    running at the time. Confirmed by direct instrumentation against the
    real DesktopApplication - threading.enumerate(), QThreadPool
    .activeThreadCount(), and QApplication.aboutToQuit - that no
    Python-visible thread was ever the cause (QThreadPool's workers are
    native Qt threads, invisible to Python's own threading module); a
    worker finishing after the window - and _thumbnail_signal, one of its
    children - was torn down raised "RuntimeError: Signal source has been
    deleted" from inside QRunnable::run(), and the process never recovered.
    closeEvent() must leave the pool fully drained before returning."""
    from PIL import Image
    from PySide6.QtCore import QThreadPool

    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    folder = tmp_path / "review"
    folder.mkdir()
    paths = []
    for i in range(8):
        p = folder / f"IMG_{i:02d}.jpg"
        Image.new("RGB", (32, 24), color=(i * 20 % 255, 50, 100)).save(p)
        paths.append(p)

    # Slow enough that several decodes are still genuinely in flight - not
    # just queued - at the moment the window closes, matching the real
    # regression (a large folder of real RAW files, not instant JPEGs).
    real_thumbnail_path = ReviewService.thumbnail_path

    def slow_thumbnail_path(self, image_path, *, with_boxes=False):
        time.sleep(0.3)
        return real_thumbnail_path(self, image_path, with_boxes=with_boxes)

    ReviewService.thumbnail_path = slow_thumbnail_path
    try:
        window.open_folder(str(folder))
        for path in paths:
            window._load_thumbnail(str(path.resolve()))
        time.sleep(0.05)
        app.processEvents()

        assert QThreadPool.globalInstance().activeThreadCount() > 0, "test setup did not get any decode running"

        window.close()

        # closeEvent()'s clear()+waitForDone() is synchronous: by the time
        # close() returns, nothing may still be running against a signal
        # object this now-closed window owns.
        assert QThreadPool.globalInstance().activeThreadCount() == 0
    finally:
        ReviewService.thumbnail_path = real_thumbnail_path

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

    window._recent_folders_menu.remember(str(tmp_path / "one"))
    window._recent_folders_menu.remember(str(tmp_path / "two"))

    persisted = window._settings.value("recent_folders", [])
    assert persisted[0] == str((tmp_path / "two").resolve())
    assert persisted[1] == str((tmp_path / "one").resolve())

    window.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_recent_folders_menu_keeps_only_the_five_most_recent(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    for i in range(8):
        window._recent_folders_menu.remember(str(tmp_path / f"folder-{i}"))

    assert len(window._recent_folders_menu.items()) == 5
    assert window._recent_folders_menu.items()[0] == str((tmp_path / "folder-7").resolve())

    window.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_opening_a_missing_recent_folder_warns_and_removes_it(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    missing_folder = str(tmp_path / "gone")
    window._recent_folders_menu.remember(missing_folder)

    # See the sibling test above for why QMessageBox.warning must be stubbed
    # under the offscreen test platform.
    warnings: list[tuple] = []
    monkeypatch.setattr(
        "picklikeme.desktop.main_window.QMessageBox.warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    opened: list[str] = []
    monkeypatch.setattr(window, "_start_open_folder", lambda folder: opened.append(folder))

    window._open_recent_folder(missing_folder)

    assert opened == []  # never attempted to open a folder that isn't there
    assert len(warnings) == 1
    assert missing_folder not in window._recent_folders_menu.items()

    window.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_opening_an_existing_recent_folder_opens_it_without_warning(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    state = ApplicationState()
    settings = DesktopSettings()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(state=state, settings=settings, service=service, worker_manager=WorkerManager())

    real_folder = tmp_path / "shoot"
    real_folder.mkdir()
    window._recent_folders_menu.remember(str(real_folder))

    warnings: list[tuple] = []
    monkeypatch.setattr(
        "picklikeme.desktop.main_window.QMessageBox.warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    opened: list[str] = []
    monkeypatch.setattr(window, "_start_open_folder", lambda folder: opened.append(folder))

    window._open_recent_folder(str(real_folder))

    assert opened == [str(real_folder)]
    assert warnings == []
    assert str(real_folder) in window._recent_folders_menu.items()

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
