"""Main window for PeakPic Desktop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QModelIndex, QSize, Qt, QSettings, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QPixmap
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
    QPushButton,
    QStatusBar,
    QStyle,
    QStyleFactory,
    QTextEdit,
    QToolBar,
    QWidget,
)

from . import theme
from .application import ApplicationState, WorkerManager
from .core.caching import CacheManager
from .core.events import EventBus
from .core.jobs import JobManager, JobSpec, run_in_background as _real_run_in_background
from .core.thumbnail_loader import ThumbnailLoadTask, ThumbnailReadySignal

run_in_background = _real_run_in_background
from .dialogs.loupe_dialog import LoupeDialog
from .dialogs.progress import run_with_progress
from .dialogs.workflow_dialogs import AutoCropDialog, PreferencesDialog, RankDialog, SpeciesLanguageDialog
from .models.image_item import ImageItem
from .models.image_model import ImageModel
from .settings import DesktopSettings
from .services import ReviewService
from .views.gallery.gallery_view import GalleryView

FILTERS = ("all", "keep", "reject", "neutral")
KEEP_PERCENT_PRESETS = (5.0, 10.0, 20.0, 25.0, 35.0)
SORT_FIELDS = ("score", "filename", "captured_at")
SORT_FIELD_LABELS = {"score": "Model Score", "filename": "File Name", "captured_at": "Capture Time"}


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
        self._sort_field = "score"  # matches ReviewSession.load()'s own default ordering
        self._sort_ascending = False
        self._last_counts: dict[str, Any] = {}
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
        self._last_loading_stage: str | None = None
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(200)
        self._loading_timer.timeout.connect(self._poll_loading_state)

        # Thumbnails decode off the UI thread (see core/thumbnail_loader.py);
        # this signal is how a finished background decode gets back to the
        # GUI thread to repaint just its one row. _thumbnails_loading tracks
        # in-flight paths so Qt re-asking for the same not-yet-ready cell
        # (which it does, repeatedly, while scrolling/repainting) doesn't
        # queue duplicate decode jobs for it.
        self._thumbnail_signal = ThumbnailReadySignal()
        self._thumbnail_signal.ready.connect(self._on_thumbnail_ready)
        self._thumbnails_loading: set[str] = set()

        self._folder_label = QLabel("No folder open")
        self._image_count_label = QLabel("Images: 0")
        self._counts_label = QLabel("")
        self._status_label = QLabel("Ready")
        self._status_message_label = QLabel("")
        self._gpu_status_label = QLabel("")
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
        self._gallery_view.decisionRequested.connect(self._on_card_decision)

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

        self._sort_combo = QComboBox(self)
        for field in SORT_FIELDS:
            self._sort_combo.addItem(SORT_FIELD_LABELS[field], field)
        self._sort_combo.setCurrentIndex(SORT_FIELDS.index(self._sort_field))
        self._sort_combo.currentIndexChanged.connect(self._on_sort_field_changed)

        self._sort_direction_btn = QPushButton(self)
        self._sort_direction_btn.setCheckable(True)
        self._sort_direction_btn.setMaximumWidth(28)
        self._sort_direction_btn.clicked.connect(self._on_sort_direction_toggled)
        self._update_sort_direction_button()

        self._build_ui()
        self._load_recent_folders()
        self._refresh_recent_folders_menu()

    def _build_ui(self) -> None:
        self.setWindowTitle("PeakPic Desktop")
        self.resize(1200, 800)
        self.setDockOptions(self.dockOptions() | self.DockOption.AnimatedDocks)
        self._apply_theme(self._settings.value("theme", theme.DEFAULT_THEME))
        self.setCentralWidget(self._central_widget)
        self._central_widget.setLayout(self._build_central_layout())
        self._build_menu_bar()
        self._build_tool_bar()
        self._build_status_bar()
        self._build_docks()
        self._restore_state()

    def _apply_theme(self, name: str) -> None:
        """Apply a theme app-wide and repaint the gallery so the color
        change is visible immediately - no restart required."""
        theme.set_theme(name)
        app = QApplication.instance()
        if app is not None:
            # Windows' native "windows11"/"windowsvista" style renders its
            # own chrome for QMenuBar/QToolBar/QStatusBar and ignores most
            # QSS background-color rules on them - confirmed by screenshot,
            # not just style-string inspection: switching themes visibly
            # reskinned the gallery cards but left the toolbar/menu/status
            # bar dark in both themes. Fusion is the standard fix: it's a
            # QStyle Qt fully implements against stylesheets, and it also
            # renders identically across Windows/macOS/Linux.
            if app.style().objectName().lower() != "fusion":
                fusion = QStyleFactory.create("Fusion")
                if fusion is not None:
                    app.setStyle(fusion)
            app.setStyleSheet(theme.build_stylesheet(theme.current_palette()))
        self._gallery_view.viewport().update()
        if hasattr(self, "_dark_theme_action"):
            self._dark_theme_action.setChecked(theme.current_theme_name() == "dark")
            self._light_theme_action.setChecked(theme.current_theme_name() == "light")
        if hasattr(self, "_counts_label"):
            self._update_status_counts(self._last_counts)

    def _set_theme(self, name: str) -> None:
        self._apply_theme(name)
        self._settings.setValue("theme", theme.current_theme_name())

    def _build_central_layout(self) -> Any:
        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self._central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._gallery_view)
        return layout

    def _std_icon(self, pixmap: QStyle.StandardPixmap) -> QIcon:
        """A platform-native stand-in icon. Cheap and always available;
        a bespoke icon set is future polish, not a blocker for a clear,
        professional toolbar."""
        return self.style().standardIcon(pixmap)

    def _make_action(
        self,
        text: str,
        *,
        icon: QStyle.StandardPixmap | None = None,
        shortcut: str | QKeySequence | None = None,
        tooltip: str | None = None,
        triggered=None,
    ) -> QAction:
        action = QAction(self._std_icon(icon), text, self) if icon is not None else QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        hint = tooltip or text
        if shortcut is not None:
            sequence = QKeySequence(shortcut) if not isinstance(shortcut, QKeySequence) else shortcut
            hint = f"{hint} ({sequence.toString()})"
        action.setToolTip(hint)
        action.setStatusTip(hint)
        if triggered is not None:
            action.triggered.connect(triggered)
        return action

    def _build_menu_bar(self) -> None:
        # Actions are built once here and reused verbatim on the toolbar
        # (see _build_tool_bar) so every shortcut has exactly one QAction
        # behind it - two QActions bound to the same shortcut string would
        # make Qt refuse to fire either ("ambiguous shortcut overload").
        SP = QStyle.StandardPixmap

        self._open_action = self._make_action(
            "Open Folder…", icon=SP.SP_DirOpenIcon, shortcut=QKeySequence.Open,
            tooltip="Open a folder of images to review", triggered=self._open_folder_dialog,
        )
        self._import_action = self._make_action(
            "Import Selected…", icon=SP.SP_ArrowDown,
            tooltip="Copy Keep-marked images to another folder", triggered=self._import_selected,
        )
        self._settings_action = self._make_action(
            "Preferences…", icon=SP.SP_FileDialogInfoView,
            tooltip="Application settings", triggered=self._show_settings,
        )

        self._keep_action = self._make_action(
            "Keep", icon=SP.SP_DialogApplyButton, shortcut="K",
            tooltip="Mark the selected image Keep", triggered=lambda: self.apply_review_status("keep"),
        )
        self._reject_action = self._make_action(
            "Reject", icon=SP.SP_DialogCancelButton, shortcut="R",
            tooltip="Mark the selected image Reject", triggered=lambda: self.apply_review_status("reject"),
        )
        self._neutral_action = self._make_action(
            "Neutral", icon=SP.SP_DialogResetButton, shortcut="N",
            tooltip="Clear the review decision", triggered=lambda: self.apply_review_status("neutral"),
        )
        self._loupe_action = self._make_action(
            "Loupe", icon=SP.SP_FileDialogContentsView, shortcut="Return",
            tooltip="Open the selected image in the Loupe", triggered=self._open_loupe_for_selection,
        )
        self._select_all_action = self._make_action(
            "Select All", icon=SP.SP_FileDialogListView, shortcut=QKeySequence.StandardKey.SelectAll,
            tooltip="Select every currently visible (filtered) image", triggered=self._select_all_visible,
        )

        self._rank_action = self._make_action(
            "Rank by AI…", icon=SP.SP_BrowserReload,
            tooltip="Score every image in the folder with the AI model", triggered=self._rank_by_ai,
        )
        self._apply_cutoff_action = self._make_action(
            "Apply Cutoff", icon=SP.SP_DialogOkButton,
            tooltip="Apply the AI keep-percent cutoff to the current folder", triggered=self._apply_cutoff,
        )
        self._organize_action = self._make_action(
            "Organize…", icon=SP.SP_DirIcon,
            tooltip="Move Keep/Reject images into Selected/Rejected folders", triggered=self._organize,
        )
        self._species_action = self._make_action(
            "Organize by Species…", icon=SP.SP_FileDialogListView,
            tooltip="Group images into per-species folders", triggered=self._organize_by_species,
        )
        self._crop_action = self._make_action(
            "Auto Crop…", icon=SP.SP_FileDialogDetailedView,
            tooltip="Generate Lightroom crop metadata around detected subjects", triggered=self._auto_crop,
        )

        exit_action = self._make_action(
            "Exit", icon=SP.SP_DialogCloseButton, triggered=QApplication.instance().quit,
        )
        about_action = self._make_action("About", triggered=self._show_about)

        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self._open_action)
        self._recent_menu = file_menu.addMenu("Recent Folders")
        self._recent_menu.setEnabled(False)
        self._refresh_recent_folders_menu()
        file_menu.addSeparator()
        file_menu.addAction(self._import_action)
        file_menu.addSeparator()
        file_menu.addAction(self._settings_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        review_menu = menu_bar.addMenu("Review")
        review_menu.addAction(self._select_all_action)
        review_menu.addSeparator()
        review_menu.addAction(self._keep_action)
        review_menu.addAction(self._reject_action)
        review_menu.addAction(self._neutral_action)
        review_menu.addSeparator()
        review_menu.addAction(self._loupe_action)

        view_menu = menu_bar.addMenu("View")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self._dark_theme_action = QAction("Dark Theme", self, checkable=True)
        self._dark_theme_action.triggered.connect(lambda: self._set_theme("dark"))
        self._light_theme_action = QAction("Light Theme", self, checkable=True)
        self._light_theme_action.triggered.connect(lambda: self._set_theme("light"))
        theme_group.addAction(self._dark_theme_action)
        theme_group.addAction(self._light_theme_action)
        view_menu.addAction(self._dark_theme_action)
        view_menu.addAction(self._light_theme_action)
        self._dark_theme_action.setChecked(theme.current_theme_name() == "dark")
        self._light_theme_action.setChecked(theme.current_theme_name() == "light")

        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(self._rank_action)
        tools_menu.addAction(self._apply_cutoff_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self._organize_action)
        tools_menu.addAction(self._species_action)
        tools_menu.addAction(self._crop_action)

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(about_action)

    def _build_tool_bar(self) -> None:
        """Groups mirror the workflow: Open -> Review decisions/Loupe ->
        AI ranking/cutoff -> Organize/Import/Species/Crop -> Preferences."""
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(20, 20))

        toolbar.addAction(self._open_action)
        toolbar.addSeparator()

        toolbar.addAction(self._select_all_action)
        toolbar.addAction(self._keep_action)
        toolbar.addAction(self._reject_action)
        toolbar.addAction(self._neutral_action)
        toolbar.addAction(self._loupe_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Filter: "))
        toolbar.addWidget(self._filter_combo)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Sort: "))
        toolbar.addWidget(self._sort_combo)
        toolbar.addWidget(self._sort_direction_btn)
        toolbar.addSeparator()

        toolbar.addAction(self._rank_action)
        toolbar.addWidget(QLabel(" AI cutoff: "))
        toolbar.addWidget(self._cutoff_combo)
        toolbar.addWidget(self._cutoff_spin)
        toolbar.addAction(self._apply_cutoff_action)
        toolbar.addSeparator()

        toolbar.addAction(self._organize_action)
        toolbar.addAction(self._import_action)
        toolbar.addAction(self._species_action)
        toolbar.addAction(self._crop_action)
        toolbar.addSeparator()

        toolbar.addAction(self._settings_action)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar(self)
        status_bar.addWidget(self._folder_label)
        status_bar.addWidget(self._image_count_label)
        status_bar.addWidget(self._counts_label)
        status_bar.addWidget(self._status_label)
        status_bar.addWidget(self._loading_progress)
        status_bar.addPermanentWidget(self._gpu_status_label)
        status_bar.addPermanentWidget(self._status_message_label)
        self.setStatusBar(status_bar)

    def _update_status_counts(self, counts: dict[str, Any]) -> None:
        """Keep/Reject/Neutral breakdown, color-coded to match the gallery
        cards - the same at-a-glance density the web review header gives."""
        self._last_counts = counts
        palette = theme.current_palette()
        keep = counts.get("keep", 0)
        reject = counts.get("reject", 0)
        neutral = counts.get("neutral", 0)
        self._counts_label.setText(
            f'<span style="color:{palette.keep_fg}">Keep {keep}</span>&nbsp;&nbsp;'
            f'<span style="color:{palette.reject_fg}">Reject {reject}</span>&nbsp;&nbsp;'
            f'<span style="color:{palette.neutral_fg}">Neutral {neutral}</span>'
        )

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
        # QProgressDialog's first constructor argument is the label text
        # shown inside the dialog, not its window title - without an
        # explicit setWindowTitle() the title bar falls back to the
        # executable name ("python"), not the app name.
        self._folder_load_dialog.setWindowTitle("PeakPic - Opening Folder")
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
        self._last_loading_stage = None
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
            QMessageBox.warning(self, "PeakPic - Open Folder", f"Could not open {folder_path}:\n{exc}")
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
        QMessageBox.warning(self, "PeakPic - Open Folder", f"Could not open {folder_path}:\n{message}")

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
        # loading_state() is a cheap dict read; load_session() rebuilds the
        # full images list (as_dict() over every image, O(n)) and feeds
        # _refresh_from_state's model reset. A large folder's background
        # metadata pass fires this timer every 200ms for the whole duration
        # of loading-categories/-metadata (which can be many seconds for
        # thousands of real RAW files), and neither of the fields that
        # phase fills in (captured_at, detected_category) is even shown on
        # a gallery card - so doing the expensive rebuild+reset on every
        # tick was pure waste, and on a large folder was slow enough on its
        # own to make the whole app look hung. Only do it when the stage
        # actually changes (or loading finishes); the progress bar/dialog
        # still update smoothly every tick from the cheap read.
        loading = self.service.loading_state()
        percent = int(loading.get("percent", 0))
        self._loading_progress.setValue(percent)
        self._status_message_label.setText(loading.get("message", ""))

        stage = loading.get("stage")
        complete = loading.get("complete", True)
        if stage != self._last_loading_stage or complete:
            self._last_loading_stage = stage
            state = self.service.load_session()
            self._refresh_from_state(state)
            self._update_folder_load_dialog(state)
        else:
            self._update_folder_load_dialog_progress(loading)

        if complete:
            self._finish_open_folder(self._open_folder_generation)

    def _update_folder_load_dialog_progress(self, loading: dict[str, Any]) -> None:
        """Cheap per-tick refresh of just the progress dialog's percent and
        operation text - see the comment in _poll_loading_state for why the
        fuller per-file detail counts (_update_folder_load_dialog) only
        refresh on a stage change instead of every 200ms tick."""
        if self._folder_load_dialog is None:
            return
        percent = int(loading.get("percent", 0))
        operation = loading.get("message", "Preparing folder")
        self._folder_load_dialog.setLabelText(f"Opening folder\nCurrent operation: {operation}\nEstimated completion: {percent}%")
        self._folder_load_dialog.setValue(percent)

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
        input_folder = state.get("input_folder")
        self._folder_label.setText(f"Folder: {Path(input_folder).name}" if input_folder else "No folder open")
        self.state.image_count = state.get("counts", {}).get("total", 0)
        self._image_count_label.setText(f"Images: {self.state.image_count}")
        self._all_items = [
            ImageItem(
                path=image.get("image_path") or "",
                file_name=Path(image.get("image_path") or "").name,
                score=image.get("score"),
                rank=image.get("rank"),
                review_status=image.get("review_status", "neutral"),
                ai_suggestion=image.get("ai_suggestion"),
                captured_at=image.get("captured_at"),
            )
            for image in state.get("images", [])
        ]
        self._apply_filter()
        self._update_status_counts(state.get("counts", {}))

    def _open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder", str(Path.home()))
        if folder:
            self.open_folder(folder)

    def _load_thumbnail(self, path: str) -> QPixmap | None:
        """Called from ImageModel.data(Qt.DecorationRole) - i.e. from Qt's
        paint path, on the UI thread. Must never decode here: a cache miss
        starts a background decode (see core/thumbnail_loader.py) and
        returns None immediately; the delegate paints a blank card until
        _on_thumbnail_ready repaints this one row."""
        if not path:
            return None
        cached = self.cache_manager.get_thumbnail(path)
        if cached is not None:
            return cached
        if path not in self._thumbnails_loading:
            self._thumbnails_loading.add(path)
            task = ThumbnailLoadTask(path, self.service.thumbnail_path, self._thumbnail_signal)
            QThreadPool.globalInstance().start(task)
        return None

    def _on_thumbnail_ready(self, path: str, pixmap: QPixmap) -> None:
        self._thumbnails_loading.discard(path)
        self.cache_manager.put_thumbnail(path, pixmap)
        self._gallery_model.notify_thumbnail_ready(path)

    # -- filtering ------------------------------------------------------------

    def _on_filter_changed(self, index: int) -> None:
        self._current_filter = FILTERS[index] if 0 <= index < len(FILTERS) else "all"
        self._apply_filter()

    def _apply_filter(self) -> None:
        if self._current_filter == "all":
            filtered = list(self._all_items)
        else:
            filtered = [item for item in self._all_items if item.review_status == self._current_filter]
        filtered = self._sort_items(filtered)
        self._gallery_model.set_items(filtered)
        if filtered and not self._gallery_view.currentIndex().isValid():
            self._gallery_view.setCurrentIndex(self._gallery_model.index(0, 0))
        self._gallery_view.set_empty_message(self._empty_message_for_current_state())

    # -- sorting ----------------------------------------------------------------

    def _on_sort_field_changed(self, index: int) -> None:
        field = self._sort_combo.itemData(index)
        if field:
            self._sort_field = field
            self._apply_filter()

    def _on_sort_direction_toggled(self) -> None:
        self._sort_ascending = not self._sort_ascending
        self._update_sort_direction_button()
        self._apply_filter()

    def _update_sort_direction_button(self) -> None:
        if self._sort_ascending:
            self._sort_direction_btn.setText("↑")
            self._sort_direction_btn.setToolTip("Ascending - click for descending")
        else:
            self._sort_direction_btn.setText("↓")
            self._sort_direction_btn.setToolTip("Descending - click for ascending")

    def _sort_items(self, items: list[ImageItem]) -> list[ImageItem]:
        """Sort by the selected field, ascending or descending. Items
        missing a value for the field (no score, no capture-time metadata)
        always sort to the end regardless of direction - reversing which
        end unranked/undated images land on depending on ascending vs.
        descending would be more confusing than useful."""
        field = self._sort_field

        def value_of(item: ImageItem):
            if field == "filename":
                return item.display_name.lower()
            if field == "captured_at":
                return item.captured_at
            return item.score

        with_value = [item for item in items if value_of(item) is not None]
        without_value = [item for item in items if value_of(item) is None]
        with_value.sort(key=value_of, reverse=not self._sort_ascending)
        without_value.sort(key=lambda item: item.display_name.lower())
        return with_value + without_value

    def _empty_message_for_current_state(self) -> str:
        if not self.state.current_folder:
            return "Open a folder to begin reviewing images"
        if not self._all_items:
            return "No images found in this folder"
        if self._current_filter != "all":
            return f"No images match the '{self._current_filter.capitalize()}' filter"
        return ""

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

    def _select_all_visible(self) -> None:
        """Select every image in the current filter/sort view - the first
        step of "filter by status, select all, apply one decision to all
        of them": _apply_filter() already put only the filtered set into
        the model, so selectAll() naturally only selects that set."""
        self._gallery_view.selectAll()
        count = len(self._gallery_model.items())
        self._set_status(f"Selected {count} image(s)" if count else "No images to select")

    def _selected_image_paths(self) -> list[str]:
        """Every multi-selected image, or the single current one when
        nothing was explicitly multi-selected (mouse click, arrow-key
        navigation, or the fallback selection _apply_filter sets on the
        first row of a freshly filtered gallery)."""
        indexes = self._gallery_view.selectionModel().selectedIndexes()
        if not indexes:
            current = self._gallery_view.currentIndex()
            indexes = [current] if current.isValid() else []
        paths: list[str] = []
        for index in indexes:
            item = self._gallery_model.item_at(index.row())
            if item is not None and item.path:
                paths.append(item.path)
        return paths

    def apply_review_status(self, status: str, *, paths: list[str] | None = None) -> None:
        """paths=None (the keyboard/toolbar path) acts on the current
        multi-selection. An explicit paths list (a single-card button
        click, see _on_card_decision) acts on just that image and leaves
        the view's own multi-selection untouched."""
        if not self.state.current_folder:
            self._set_status("Open a folder before applying review decisions")
            return

        if paths is None:
            paths = self._selected_image_paths()
        if not paths:
            self._set_status("Select an image in the gallery first")
            return

        self.state.current_selection = paths
        for image_path in paths:
            self.service.set_review_status(image_path, status)

        if len(paths) == 1:
            self._set_status(f"Marked {Path(paths[0]).name} as {status}")
        else:
            self._set_status(f"Marked {len(paths)} images as {status}")
        self._refresh_from_state(self.service.load_session())

    def _on_card_decision(self, path: str, status: str) -> None:
        """A card's own Keep/Reject/Neutral button was clicked directly."""
        self.apply_review_status(status, paths=[path])

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
            device = result.get("device")
            if device:
                self.state.gpu_status = device
                self._gpu_status_label.setText(f"Device: {device}")

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
        confirm = QMessageBox.question(self, "PeakPic - Organize", message)
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
            QMessageBox.information(self, "PeakPic - Import Selected", f"Copied {result.get('copied', 0)} image(s) to:\n{destination}")

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
                "PeakPic - Organize by Species",
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
        paths = self._selected_image_paths()
        if not paths:
            self._set_status("Select one or more images before running Auto Crop")
            return
        dialog = AutoCropDialog(parent=self)
        if dialog.exec() != AutoCropDialog.DialogCode.Accepted:
            return
        margin_percent = dialog.margin_percent()

        def _run(on_progress=None, on_stage=None):
            return self.service.auto_crop(margin_percent=margin_percent, on_progress=on_progress, image_paths=paths)

        def _on_success(result: dict[str, Any]) -> None:
            self._set_status(result.get("message", "Auto crop finished"))

        thread = run_with_progress(self, f"Auto Crop ({len(paths)} selected)", _run, on_success=_on_success)
        self._active_threads.append(thread)
        thread.finished.connect(lambda: self._active_threads.remove(thread) if thread in self._active_threads else None)

    # -- misc -----------------------------------------------------------------

    def _show_settings(self) -> None:
        current_language = self._settings.value("review/species_language", "en")
        dialog = PreferencesDialog(default_theme=theme.current_theme_name(), default_language=current_language, parent=self)
        if dialog.exec() != PreferencesDialog.DialogCode.Accepted:
            return
        self._set_theme(dialog.theme_name())
        self._settings.setValue("review/species_language", dialog.species_language())

    def _show_about(self) -> None:
        QMessageBox.about(self, "About PeakPic", "PeakPic Desktop\nNative desktop shell powered by the existing backend.")

    def _set_status(self, message: str) -> None:
        self.state.status_message = message
        self._status_message_label.setText(message)

    def _on_gallery_key_press(self, key: int) -> None:
        """Open the Loupe for the current gallery selection. K/R/N are
        handled by the Review menu's QActions (shared with the toolbar),
        not here - see GalleryView.keyPressEvent."""
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._open_loupe_for_selection()

    def closeEvent(self, event: Any) -> None:
        self._save_state()
        super().closeEvent(event)

