"""Main window for PeakPic Desktop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QModelIndex, Qt, QSettings, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QWidget,
)

from .application import ApplicationState, WorkerManager
from .core.caching import CacheManager
from .core.events import EventBus
from .core.jobs import JobManager, JobSpec, run_in_background as _real_run_in_background

run_in_background = _real_run_in_background
from .dialogs.loupe_dialog import LoupeDialog
from .dialogs.progress import run_with_progress
from .dialogs.workflow_dialogs import AutoCropDialog, RankDialog, SpeciesLanguageDialog
from .models.image_item import ImageItem
from .models.image_model import ImageModel
from .settings import DesktopSettings
from .services import ReviewService
from .views.gallery.gallery_view import GalleryView

FILTERS = ("all", "keep", "reject", "neutral")
KEEP_PERCENT_PRESETS = (5.0, 10.0, 20.0, 25.0, 35.0)


class MainWindow(QMainWindow):
    """Qt main window for the desktop shell."""

    def __init__(
        self,
        *,
        state: ApplicationState,
        settings: DesktopSettings,
        service: ReviewService,
        worker_manager: WorkerManager,
        event_bus: EventBus | None = None,
        cache_manager: CacheManager | None = None,
        job_manager: JobManager | None = None,
    ) -> None:
        super().__init__()
        self.state = state
        self.settings = settings
        self.service = service
        self.worker_manager = worker_manager
        self.event_bus = event_bus or EventBus()
        self.cache_manager = cache_manager or CacheManager()
        self.job_manager = job_manager or JobManager()
        self._initialized = False
        self._settings = QSettings("PeakPic", "PeakPicDesktop")
        self._all_items: list[ImageItem] = []
        self._current_filter = "all"
        self._active_threads: list[Any] = []  # keeps background QThreads alive while running
        self._open_folder_in_progress = False
        self._open_folder_generation = 0
        self._open_folder_thread: Any | None = None
        self._folder_load_dialog: QProgressDialog | None = None
        self._folder_load_cancelled = False
        self._folder_load_snapshot: dict[str, Any] | None = None
        self._recent_folders: list[str] = []
        self._recent_folder_actions: list[QAction] = []
        self._thumbnail_generation_count = 0
        self._metadata_loaded_count = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(200)
        self._loading_timer.timeout.connect(self._poll_loading_state)

        self._folder_label = QLabel("No folder open")
        self._image_count_label = QLabel("Images: 0")
        self._status_label = QLabel("Ready")
        self._status_message_label = QLabel("")
        self._loading_progress = QProgressBar(self)
        self._loading_progress.setMaximumWidth(160)
        self._loading_progress.setVisible(False)

        self._central_widget = QWidget(self)
        self._gallery_model = ImageModel()
        self._gallery_model.set_thumbnail_provider(self._load_thumbnail)
        self._gallery_view = GalleryView(self._central_widget)
        self._gallery_view.setModel(self._gallery_model)
        self._gallery_view.doubleClicked.connect(self._open_loupe_for_index)
        self._gallery_view.keyPressSignal.connect(self._on_gallery_key_press)

        self._filter_combo = QComboBox(self)
        self._filter_combo.addItems([f.capitalize() for f in FILTERS])
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)

        self._cutoff_combo = QComboBox(self)
        for preset in KEEP_PERCENT_PRESETS:
            self._cutoff_combo.addItem(f"{preset:g}%", preset)
        self._cutoff_combo.addItem("Custom…", None)
        self._cutoff_combo.currentIndexChanged.connect(self._on_cutoff_preset_changed)

        self._cutoff_spin = QDoubleSpinBox(self)
        self._cutoff_spin.setRange(0.0, 100.0)
        self._cutoff_spin.setSuffix("%")
        self._cutoff_spin.setValue(KEEP_PERCENT_PRESETS[0])
        self._cutoff_spin.setEnabled(False)

        self._build_ui()
        self._load_recent_folders()
        self._refresh_recent_folders_menu()

    def _build_ui(self) -> None:
        self.setWindowTitle("PeakPic Desktop")
        self.resize(1200, 800)
        self.setDockOptions(self.dockOptions() | self.DockOption.AnimatedDocks)
        self.setCentralWidget(self._central_widget)
        self._central_widget.setLayout(self._build_central_layout())
        self._build_menu_bar()
        self._build_tool_bar()
        self._build_status_bar()
        self._build_docks()
        self._restore_state()

    def _build_central_layout(self) -> Any:
        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self._central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._gallery_view)
        return layout

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open Folder…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_folder_dialog)
        file_menu.addAction(open_action)
        self._recent_menu = file_menu.addMenu("Recent Folders")
        self._recent_menu.setEnabled(False)
        self._refresh_recent_folders_menu()
        file_menu.addSeparator()
        import_action = QAction("Import Selected…", self)
        import_action.triggered.connect(self._import_selected)
        file_menu.addAction(import_action)
        file_menu.addSeparator()
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._show_settings)
        file_menu.addAction(settings_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(exit_action)

        review_menu = menu_bar.addMenu("Review")
        keep_action = QAction("Keep", self)
        keep_action.setShortcut("K")
        keep_action.triggered.connect(lambda: self.apply_review_status("keep"))
        review_menu.addAction(keep_action)
        reject_action = QAction("Reject", self)
        reject_action.setShortcut("R")
        reject_action.triggered.connect(lambda: self.apply_review_status("reject"))
        review_menu.addAction(reject_action)
        neutral_action = QAction("Neutral", self)
        neutral_action.setShortcut("N")
        neutral_action.triggered.connect(lambda: self.apply_review_status("neutral"))
        review_menu.addAction(neutral_action)
        review_menu.addSeparator()
        loupe_action = QAction("Open in Loupe…", self)
        loupe_action.triggered.connect(self._open_loupe_for_selection)
        review_menu.addAction(loupe_action)

        view_menu = menu_bar.addMenu("View")
        zoom_in = QAction("Zoom In", self)
        zoom_in.triggered.connect(lambda: self._set_status("Zoom In placeholder"))
        view_menu.addAction(zoom_in)
        zoom_out = QAction("Zoom Out", self)
        zoom_out.triggered.connect(lambda: self._set_status("Zoom Out placeholder"))
        view_menu.addAction(zoom_out)

        tools_menu = menu_bar.addMenu("Tools")
        rank_action = QAction("Rank by AI…", self)
        rank_action.triggered.connect(self._rank_by_ai)
        tools_menu.addAction(rank_action)
        organize_action = QAction("Organize (Selected/Rejected)…", self)
        organize_action.triggered.connect(self._organize)
        tools_menu.addAction(organize_action)
        species_action = QAction("Organize by Species…", self)
        species_action.triggered.connect(self._organize_by_species)
        tools_menu.addAction(species_action)
        crop_action = QAction("Auto Crop…", self)
        crop_action.triggered.connect(self._auto_crop)
        tools_menu.addAction(crop_action)

        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_tool_bar(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        toolbar.addAction("Open Folder", self._open_folder_dialog)
        toolbar.addSeparator()
        toolbar.addAction("Keep", lambda: self.apply_review_status("keep"))
        toolbar.addAction("Reject", lambda: self.apply_review_status("reject"))
        toolbar.addAction("Neutral", lambda: self.apply_review_status("neutral"))
        toolbar.addAction("Loupe", self._open_loupe_for_selection)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Filter: "))
        toolbar.addWidget(self._filter_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" AI cutoff: "))
        toolbar.addWidget(self._cutoff_combo)
        toolbar.addWidget(self._cutoff_spin)
        toolbar.addAction("Apply Cutoff", self._apply_cutoff)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar(self)
        status_bar.addWidget(self._folder_label)
        status_bar.addWidget(self._image_count_label)
        status_bar.addWidget(self._status_label)
        status_bar.addWidget(self._loading_progress)
        status_bar.addPermanentWidget(self._status_message_label)
        self.setStatusBar(status_bar)

    def _build_docks(self) -> None:
        pass

    def _restore_state(self) -> None:
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._settings.value("window/state")
        if state is not None:
            self.restoreState(state)

    def _save_state(self) -> None:
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())

    def initialize(self) -> None:
        self._initialized = True
        self.state.status_message = "Desktop shell ready"
        self._status_label.setText("Ready")
        self._status_message_label.setText(self.state.status_message)
        self._load_recent_folders()
        self._refresh_recent_folders_menu()

    # -- folder loading, with a progress indicator while it runs ------------

    def _load_recent_folders(self) -> None:
        raw = self._settings.value("recent_folders", [])
        if isinstance(raw, str):
            raw = [raw]
        self._recent_folders = [folder for folder in raw if isinstance(folder, str) and folder]
        self._recent_folders = list(dict.fromkeys(self._recent_folders))

    def _save_recent_folder(self, folder: str) -> None:
        normalized = str(Path(folder).resolve())
        self._recent_folders = [entry for entry in self._recent_folders if entry != normalized]
        self._recent_folders.insert(0, normalized)
        self._recent_folders = self._recent_folders[:10]
        self._settings.setValue("recent_folders", self._recent_folders)
        self._refresh_recent_folders_menu()

    def _refresh_recent_folders_menu(self) -> None:
        if not hasattr(self, "_recent_menu"):
            return
        for action in self._recent_folder_actions:
            self._recent_menu.removeAction(action)
            action.deleteLater()
        self._recent_folder_actions = []
        if self._recent_folders:
            self._recent_menu.setEnabled(True)
            for folder in self._recent_folders:
                action = QAction(folder, self)
                action.triggered.connect(lambda checked=False, selected=folder: self._start_open_folder(str(selected)))
                self._recent_menu.addAction(action)
                self._recent_folder_actions.append(action)
            clear_action = QAction("Clear Recent Folders", self)
            clear_action.triggered.connect(self._clear_recent_folders)
            self._recent_menu.addSeparator()
            self._recent_menu.addAction(clear_action)
            self._recent_folder_actions.append(clear_action)
        else:
            self._recent_menu.setEnabled(False)
            action = QAction("No recent folders yet", self)
            action.setEnabled(False)
            self._recent_menu.addAction(action)
            self._recent_folder_actions.append(action)

    def _clear_recent_folders(self) -> None:
        self._recent_folders = []
        self._settings.setValue("recent_folders", [])
        self._refresh_recent_folders_menu()

    def _default_folder_for_dialog(self) -> str:
        last_folder = self._settings.value("last_opened_folder", "")
        if isinstance(last_folder, str) and last_folder:
            return last_folder
        return str(Path.home())

    def _setup_folder_load_dialog(self) -> None:
        self._folder_load_dialog = QProgressDialog("Opening folder…", "Cancel", 0, 100, self)
        self._folder_load_dialog.setWindowModality(Qt.WindowModal)
        self._folder_load_dialog.setMinimumDuration(0)
        self._folder_load_dialog.setAutoClose(False)
        self._folder_load_dialog.setAutoReset(False)
        self._folder_load_dialog.setValue(0)
        self._folder_load_dialog.canceled.connect(self._cancel_open_folder)

    def _show_folder_load_dialog(self, folder: str) -> None:
        if self._folder_load_dialog is None:
            self._setup_folder_load_dialog()
        detail = "Current operation: Scanning folder…\nFiles discovered: 0\nThumbnails generated: 0\nMetadata loaded: 0\nEstimated completion: 0%"
        self._folder_load_dialog.setLabelText(f"Opening {Path(folder).name}\n{detail}")
        self._folder_load_dialog.setValue(0)
        self._folder_load_dialog.show()

    def _hide_folder_load_dialog(self) -> None:
        if self._folder_load_dialog is not None:
            self._folder_load_dialog.close()
            self._folder_load_dialog.deleteLater()
            self._folder_load_dialog = None

    def open_folder(self, folder: str) -> dict[str, Any]:
        return self._start_open_folder(folder)

    def _start_open_folder(self, folder: str) -> dict[str, Any]:
        if self._open_folder_in_progress:
            self._set_status("A folder is already loading. Please wait for it to finish.")
            return self.service.load_session()

        folder_path = str(Path(folder).resolve())
        self._folder_load_snapshot = {"folder": self.state.current_folder, "state": self.service.load_session()}
        self._folder_load_cancelled = False
        self._open_folder_generation += 1
        generation = self._open_folder_generation
        self._open_folder_in_progress = True
        self._thumbnail_generation_count = 0
        self._metadata_loaded_count = 0
        self._loading_timer.start()
        self._show_folder_load_dialog(folder_path)
        self._set_status(f"Opening {folder_path}")
        self._settings.setValue("last_opened_folder", folder_path)
        self._last_opened_folder = folder_path

        try:
            result = self.service.open_folder(folder_path)
        except Exception as exc:  # noqa: BLE001 - surface errors without crashing the shell
            self._set_status(f"Could not open {folder_path}: {exc}")
            self._restore_previous_session()
            self._finish_open_folder(generation)
            QMessageBox.warning(self, "Open Folder", f"Could not open {folder_path}:\n{exc}")
            return self.service.load_session()

        if self._folder_load_cancelled or generation != self._open_folder_generation:
            self._restore_previous_session()
            self._finish_open_folder(generation)
            return self.service.load_session()

        self._save_recent_folder(folder_path)
        self.state.current_folder = result.get("input_folder") or self.state.current_folder
        self.state.image_count = result.get("counts", {}).get("total", 0)
        self._refresh_from_state(result)

        if run_in_background is not _real_run_in_background:
            self._open_folder_thread = run_in_background(
                self,
                lambda: result,
                on_finished=lambda _: self._finish_open_folder(generation, result=result),
                on_failed=lambda message: self._handle_open_folder_failure(generation, folder_path, message),
            )
            return result

        return result

    def _handle_open_folder_failure(self, generation: int, folder_path: str, message: str) -> None:
        self._set_status(f"Could not open {folder_path}: {message}")
        self._restore_previous_session()
        self._finish_open_folder(generation)
        QMessageBox.warning(self, "Open Folder", f"Could not open {folder_path}:\n{message}")

    def _finish_open_folder(self, generation: int, *, result: dict[str, Any] | None = None) -> None:
        if generation != self._open_folder_generation:
            return
        self._loading_timer.stop()
        self._open_folder_in_progress = False
        self._open_folder_thread = None
        self._hide_folder_load_dialog()
        if result is not None:
            state = self.service.load_session()
            self.state.current_folder = state.get("input_folder") or self.state.current_folder
            self.state.image_count = state.get("counts", {}).get("total", 0)
            self._refresh_from_state(state)
            self.event_bus.publish("folder-opened", {"folder": self.state.current_folder, "count": self.state.image_count})
            self.job_manager.submit(
                JobSpec(
                    name="folder-opened",
                    func=lambda: {"folder": self.state.current_folder, "count": self.state.image_count},
                    description="Notify infrastructure of a folder open",
                    on_finished=lambda job_result: self.event_bus.publish("folder-load-complete", job_result.payload),
                )
            )
            self._set_status(f"Opened {self.state.current_folder}")
        else:
            self._refresh_from_state(self.service.load_session())

    def _cancel_open_folder(self) -> None:
        if not self._open_folder_in_progress:
            return
        self._folder_load_cancelled = True
        self._open_folder_in_progress = False
        self._loading_timer.stop()
        self.service.session._loading_generation += 1
        self._restore_previous_session()
        self._hide_folder_load_dialog()
        self._set_status("Folder open cancelled; previous session restored")

    def _restore_previous_session(self) -> None:
        if self._folder_load_snapshot is None:
            self.service.session._clear()
            self._refresh_from_state(self.service.load_session())
            return
        previous_folder = self._folder_load_snapshot.get("folder")
        if previous_folder:
            self.service.session._loading_generation += 1
            self.service.open_folder(previous_folder)
            self.state.current_folder = previous_folder
            self._refresh_from_state(self.service.load_session())
            return
        self.service.session._clear()
        self._refresh_from_state(self.service.load_session())

    def _poll_loading_state(self) -> None:
        if not self._open_folder_in_progress:
            return
        state = self.service.load_session()
        loading = state.get("loading", {})
        self._loading_progress.setValue(int(loading.get("percent", 0)))
        self._status_message_label.setText(loading.get("message", ""))
        self._refresh_from_state(state)
        self._update_folder_load_dialog(state)
        if loading.get("complete", True):
            self._finish_open_folder(self._open_folder_generation)

    def _update_folder_load_dialog(self, state: dict[str, Any]) -> None:
        if self._folder_load_dialog is None:
            return
        loading = state.get("loading", {})
        images = state.get("images", [])
        percent = int(loading.get("percent", 0))
        operation = loading.get("message", "Preparing folder")
        discovered = state.get("counts", {}).get("total", len(images))
        metadata = sum(1 for image in images if image.get("captured_at") or image.get("detected_category"))
        detail = (
            f"Current operation: {operation}\n"
            f"Files discovered: {discovered}\n"
            f"Thumbnails generated: {self._thumbnail_generation_count}\n"
            f"Metadata loaded: {metadata}\n"
            f"Estimated completion: {percent}%"
        )
        self._folder_load_dialog.setLabelText(f"Opening folder\n{detail}")
        self._folder_load_dialog.setValue(percent)

    def _refresh_from_state(self, state: dict[str, Any]) -> None:
        self.state.image_count = state.get("counts", {}).get("total", 0)
        self._image_count_label.setText(f"Images: {self.state.image_count}")
        self._all_items = [
            ImageItem(
                path=image.get("image_path") or "",
                file_name=Path(image.get("image_path") or "").name,
                score=image.get("score"),
                review_status=image.get("review_status", "neutral"),
                ai_suggestion=image.get("ai_suggestion"),
            )
            for image in state.get("images", [])
        ]
        self._apply_filter()

    def _open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder", str(Path.home()))
        if folder:
            self.open_folder(folder)

    def _load_thumbnail(self, path: str) -> QPixmap | None:
        if not path:
            return None
        cached = self.cache_manager.get_thumbnail(path)
        if cached is not None:
            return cached
        try:
            thumbnail_path = self.service.thumbnail_path(path)
        except Exception:  # noqa: BLE001 - a bad frame must not break the gallery
            return None
        if thumbnail_path is None:
            return None
        pixmap = QPixmap(str(thumbnail_path))
        if pixmap.isNull():
            return None
        self.cache_manager.put_thumbnail(path, pixmap)
        return pixmap

    # -- filtering ------------------------------------------------------------

    def _on_filter_changed(self, index: int) -> None:
        self._current_filter = FILTERS[index] if 0 <= index < len(FILTERS) else "all"
        self._apply_filter()

    def _apply_filter(self) -> None:
        if self._current_filter == "all":
            filtered = list(self._all_items)
        else:
            filtered = [item for item in self._all_items if item.review_status == self._current_filter]
        self._gallery_model.set_items(filtered)
        if filtered and not self._gallery_view.currentIndex().isValid():
            self._gallery_view.setCurrentIndex(self._gallery_model.index(0, 0))

    def _filtered_paths(self) -> list[str]:
        return [item.path for item in self._gallery_model.items()]

    # -- cutoff percentage ------------------------------------------------------

    def _on_cutoff_preset_changed(self, index: int) -> None:
        preset = self._cutoff_combo.itemData(index)
        if preset is None:
            self._cutoff_spin.setEnabled(True)
        else:
            self._cutoff_spin.setEnabled(False)
            self._cutoff_spin.setValue(preset)

    def _apply_cutoff(self) -> None:
        if not self.state.current_folder:
            self._set_status("Open a folder before setting the AI cutoff")
            return
        percent = self._cutoff_spin.value()
        state = self.service.set_keep_percent(percent)
        self._refresh_from_state(state)
        self._set_status(f"AI cutoff set to {percent:g}%")

    # -- review actions -------------------------------------------------------

    def _selected_image_path(self) -> str | None:
        index = self._gallery_view.currentIndex()
        if not index.isValid():
            return None
        item = self._gallery_model.item_at(index.row())
        return item.path if item else None

    def apply_review_status(self, status: str) -> None:
        if not self.state.current_folder:
            self._set_status("Open a folder before applying review decisions")
            return

        image_path = self._selected_image_path()
        if image_path is None:
            self._set_status("Select an image in the gallery first")
            return

        self.state.current_selection = [image_path]
        self.service.set_review_status(image_path, status)
        self._set_status(f"Marked {Path(image_path).name} as {status}")
        self._refresh_from_state(self.service.load_session())

    # -- loupe / zoom review ---------------------------------------------------

    def _open_loupe_for_index(self, index: QModelIndex) -> None:
        self._open_loupe(start_row=index.row())

    def _open_loupe_for_selection(self) -> None:
        index = self._gallery_view.currentIndex()
        self._open_loupe(start_row=index.row() if index.isValid() else 0)

    def _open_loupe(self, *, start_row: int) -> None:
        items = self._gallery_model.items()
        if not items:
            self._set_status("No images to review in the current filter")
            return
        paths = [item.path for item in items]
        start_row = max(0, min(start_row, len(paths) - 1))
        dialog = LoupeDialog(service=self.service, image_paths=paths, items=items, start_index=start_row, parent=self)
        dialog.exec()
        self._refresh_from_state(self.service.load_session())

    # -- rank by AI -------------------------------------------------------------

    def _rank_by_ai(self) -> None:
        if not self.state.current_folder:
            self._set_status("Open a folder before ranking it")
            return
        dialog = RankDialog(parent=self)
        if dialog.exec() != RankDialog.DialogCode.Accepted:
            return
        checkpoint = dialog.checkpoint_path()
        crop_birds = dialog.crop_birds()

        def _run(on_progress=None, on_stage=None):
            return self.service.rank_folder(
                checkpoint=checkpoint, crop_birds=crop_birds, on_stage=on_stage, on_progress=on_progress
            )

        def _on_success(result: dict[str, Any]) -> None:
            self._refresh_from_state(result["state"])
            self._set_status(f"Ranked {result.get('image_count', 0)} images")

        thread = run_with_progress(self, "Rank by AI", _run, on_success=_on_success)
        self._active_threads.append(thread)
        thread.finished.connect(lambda: self._active_threads.remove(thread) if thread in self._active_threads else None)

    # -- organize (Selected/Rejected) -------------------------------------------

    def _organize(self) -> None:
        if not self.state.current_folder:
            self._set_status("Open a folder before organizing it")
            return
        preview = self.service.arrange(dry_run=True)
        message = (
            f"This will move:\n"
            f"  {preview['selected']} image(s) -> {preview['selected_dir']}\n"
            f"  {preview['rejected']} image(s) -> {preview['rejected_dir']}\n\n"
            "Neutral images are left where they are. Continue?"
        )
        confirm = QMessageBox.question(self, "Organize", message)
        if confirm != QMessageBox.StandardButton.Yes:
            return
        result = self.service.arrange(dry_run=False)
        self._refresh_from_state(self.service.load_session())
        self._set_status(f"Organized {result['moved']} image(s); {result['errors']} error(s)")

    # -- import selected ---------------------------------------------------------

    def _import_selected(self) -> None:
        if not self.state.current_folder:
            self._set_status("Open a folder before importing from it")
            return
        destination = QFileDialog.getExistingDirectory(self, "Import Selected — choose destination", str(Path.home()))
        if not destination:
            return

        def _run(on_progress=None, on_stage=None):
            return self.service.import_selected(destination, on_progress=on_progress)

        def _on_success(result: dict[str, Any]) -> None:
            self._set_status(f"Imported {result.get('copied', 0)} image(s) into {destination}")
            QMessageBox.information(self, "Import Selected", f"Copied {result.get('copied', 0)} image(s) to:\n{destination}")

        thread = run_with_progress(self, "Importing Selected Images", _run, on_success=_on_success)
        self._active_threads.append(thread)
        thread.finished.connect(lambda: self._active_threads.remove(thread) if thread in self._active_threads else None)

    # -- organize by species ------------------------------------------------------

    def _organize_by_species(self) -> None:
        if not self.state.current_folder:
            self._set_status("Open a folder before organizing it by species")
            return
        default_language = self._settings.value("review/species_language", "en")
        dialog = SpeciesLanguageDialog(default_language=default_language, parent=self)
        if dialog.exec() != SpeciesLanguageDialog.DialogCode.Accepted:
            return
        language = dialog.language()
        self._settings.setValue("review/species_language", language)

        def _run(on_progress=None, on_stage=None):
            return self.service.organize_by_species(language=language, on_progress=on_progress)

        def _on_success(result: dict[str, Any]) -> None:
            self._refresh_from_state(self.service.load_session())
            self._set_status(f"Organized {result.get('moved', 0)} image(s) into species folders")

        def _on_error(message: str) -> None:
            QMessageBox.warning(
                self,
                "Organize by Species",
                "Could not organize by species. The species classifier may not be installed:\n" + message,
            )

        thread = run_with_progress(self, "Organizing by Species", _run, on_success=_on_success, on_error=_on_error)
        self._active_threads.append(thread)
        thread.finished.connect(lambda: self._active_threads.remove(thread) if thread in self._active_threads else None)

    # -- auto crop --------------------------------------------------------------

    def _auto_crop(self) -> None:
        if not self.state.current_folder:
            self._set_status("Open a folder before running auto crop")
            return
        dialog = AutoCropDialog(parent=self)
        if dialog.exec() != AutoCropDialog.DialogCode.Accepted:
            return
        margin_percent = dialog.margin_percent()

        def _run(on_progress=None, on_stage=None):
            return self.service.auto_crop(margin_percent=margin_percent, on_progress=on_progress)

        def _on_success(result: dict[str, Any]) -> None:
            self._set_status(result.get("message", "Auto crop finished"))

        thread = run_with_progress(self, "Auto Crop", _run, on_success=_on_success)
        self._active_threads.append(thread)
        thread.finished.connect(lambda: self._active_threads.remove(thread) if thread in self._active_threads else None)

    # -- misc -----------------------------------------------------------------

    def _show_settings(self) -> None:
        QMessageBox.information(self, "Settings", "Settings dialog will be implemented in a later phase.")

    def _show_about(self) -> None:
        QMessageBox.about(self, "About PeakPic", "PeakPic Desktop\nNative desktop shell powered by the existing backend.")

    def _set_status(self, message: str) -> None:
        self.state.status_message = message
        self._status_message_label.setText(message)

    def _on_gallery_key_press(self, key: int) -> None:
        """Handle keyboard shortcuts in gallery view."""
        if key == Qt.Key.Key_K:
            self.apply_review_status("keep")
        elif key == Qt.Key.Key_R:
            self.apply_review_status("reject")
        elif key == Qt.Key.Key_N:
            self.apply_review_status("neutral")
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            self._open_loupe_for_selection()

    def closeEvent(self, event: Any) -> None:
        self._save_state()
        super().closeEvent(event)

