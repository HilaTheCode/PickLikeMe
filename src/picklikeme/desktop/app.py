"""Native desktop shell for PeakPic built on top of the existing review backend."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .application import ApplicationState, WorkerManager
from .core.caching import CacheManager
from .core.events import EventBus
from .core.jobs import JobManager
from .main_window import MainWindow
from .services import ReviewService
from .settings import SettingsStore

logger = logging.getLogger("picklikeme.desktop")


class DesktopApplication:
    """Application controller for the desktop shell."""

    def __init__(self, *, db_path: str | Path | None = None) -> None:
        self._app = QApplication.instance() or QApplication([])

        # ------------------------------------------------------------------
        # Application icon
        # ------------------------------------------------------------------
        project_root = Path(__file__).resolve().parents[2]
        icon_path = project_root / "assets" / "peakpic.ico"

        if icon_path.exists():
            icon = QIcon(str(icon_path))
            self._app.setWindowIcon(icon)

        # ------------------------------------------------------------------
        # Application identity
        # ------------------------------------------------------------------
        self._app.setApplicationName("PeakPic")
        self._app.setApplicationDisplayName("PeakPic Desktop")

        self.state = ApplicationState()
        self.service = ReviewService(db_path=db_path)
        self.settings = SettingsStore()
        self.worker_manager = WorkerManager()
        self.event_bus = EventBus()
        self.cache_manager = CacheManager()
        self.job_manager = JobManager()

        self.window = MainWindow(
            state=self.state,
            settings=self.settings.settings,
            service=self.service,
            worker_manager=self.worker_manager,
            event_bus=self.event_bus,
            cache_manager=self.cache_manager,
            job_manager=self.job_manager,
        )

        # Give the main window the same icon
        if icon_path.exists():
            self.window.setWindowIcon(icon)

    def initialize(self) -> None:
        self.window.initialize()
        self.window.show()
        logger.info("PeakPic desktop shell initialized")

    def open_folder(self, folder: str) -> dict[str, Any]:
        self.state.current_folder = folder
        return self.window.open_folder(folder)

    def run(self) -> int:
        try:
            self.initialize()
            logger.info("Desktop application ready")
            return self._app.exec()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Desktop application failed: %s", exc)
            return 1
        finally:
            self.service.close()


def main(argv: list[str] | None = None) -> int:
    del argv
    return DesktopApplication().run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
