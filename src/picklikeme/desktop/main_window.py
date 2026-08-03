"""Main window for PeakPic Desktop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QModelIndex, QSize, Qt, QSettings, QThreadPool
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStatusBar,
    QStyle,
    QStyleFactory,
    QTextEdit,
    QToolBar,
    QToolButton,
    QWidget,
)

from ..ranking import DEFAULT_STRATEGY_ID
from ..ranking.filters import REJECT_REASON_LABELS
from . import theme
from .application import ApplicationState, WorkerManager
from .core.caching import CacheManager
from .core.events import EventBus
from .core.jobs import JobManager, JobSpec, run_in_background as _real_run_in_background
from .core.thumbnail_loader import ThumbnailLoadTask, ThumbnailReadySignal
from .widgets.recent_items import DEFAULT_RECENT_ITEMS_LIMIT, RecentItemsMenu

run_in_background = _real_run_in_background
from .dialogs.analytics_dashboard import AnalyticsDashboard
from .dialogs.loupe_dialog import LoupeDialog
from .dialogs.progress import run_with_progress
from .dialogs.workflow_dialogs import (
    AlgorithmParametersDialog,
    AutoCropDialog,
    PreferencesDialog,
    RankDialog,
    SpeciesLanguageDialog,
)
from .models.image_item import ImageItem
from .models.image_model import ImageModel
from .settings import DesktopSettings
from .services import ReviewService
from .views.gallery.gallery_view import GalleryView

FILTERS = (
    "all", "keep", "reject", "neutral",
    "ai_keep", "ai_reject", "ai_keep_user_reject", "ai_reject_user_keep",
    "algorithm_keep", "algorithm_reject", "algorithm_keep_user_reject", "algorithm_reject_user_keep",
    "filtered",
)
# Matches the web review UI's filterImages() exactly (review/page.py) -
# ai_keep/ai_reject are "the AI currently suggests this" regardless of
# the photographer's own decision; the two conflict filters narrow that
# to specifically where the two disagree, one direction each (a single
# combined "conflicts" filter can't tell a photographer which images
# need which kind of attention). "filtered" is Desktop-only (the web UI
# predates non-AI strategies): any image at least one analysis module
# explicitly excluded from scoring (see ImageItem.filter_reasons) - the
# tool for a photographer investigating "why didn't Classic Vision rank
# this", regardless of which strategy or reason.
#
# algorithm_* are the generalized form of the ai_* filters above - they
# read item.algorithm_suggestion (ReviewSession.suggestions_for, computed
# against whichever strategy the Color Source picker currently selects -
# see _on_color_source_changed) instead of always the AI model
# specifically. The ai_* filters are deliberately left unchanged rather
# than being redefined in terms of the current strategy: "AI Keep" should
# always mean the AI model, exactly as it always has, even after a
# photographer switches Color Source to Classic Vision - algorithm_* is
# the new, separate way to ask "whichever strategy I'm currently looking
# at, where does it disagree with me."
FILTER_LABELS = {
    "all": "All",
    "keep": "Keep",
    "reject": "Reject",
    "neutral": "Neutral",
    "ai_keep": "AI Keep",
    "ai_reject": "AI Reject",
    "ai_keep_user_reject": "Conflict: AI Keep / You Reject",
    "ai_reject_user_keep": "Conflict: AI Reject / You Keep",
    "algorithm_keep": "Algorithm Keep (current Color Source)",
    "algorithm_reject": "Algorithm Reject (current Color Source)",
    "algorithm_keep_user_reject": "Conflict: Algorithm Keep / You Reject",
    "algorithm_reject_user_keep": "Conflict: Algorithm Reject / You Keep",
    "filtered": "Filtered (Skipped by an analysis module)",
}
KEEP_PERCENT_PRESETS = (5.0, 10.0, 20.0, 25.0, 35.0)
# Sorting by any analysis module's score, plus the two intrinsic file
# properties. A module's field is "score:<strategy_id>"; the bare "score" is
# kept as the default because it is what the window opens on and what
# ReviewSession's own load order already matches.
SORT_SCORE_PREFIX = "score:"
SORT_FIELDS = ("score", "filename", "captured_at")
SORT_FIELD_LABELS = {"score": "AI Score", "filename": "File Name", "captured_at": "Capture Time"}


def sort_options() -> list[tuple[str, str]]:
    """(field, label) for the Sort combo, one entry per analysis module.

    Built from the ranking registry rather than listed, so a new module
    becomes sortable at the same moment it becomes runnable. The AI model is
    already covered by the default "score" field, so it is not repeated.
    """
    from ..ranking import DEFAULT_STRATEGY_ID, available_strategies

    options = [("score", SORT_FIELD_LABELS["score"])]
    for info in available_strategies():
        if info.strategy_id == DEFAULT_STRATEGY_ID:
            continue
        options.append((f"{SORT_SCORE_PREFIX}{info.strategy_id}", f"{info.display_name} Score"))
    options.append(("filename", SORT_FIELD_LABELS["filename"]))
    options.append(("captured_at", SORT_FIELD_LABELS["captured_at"]))
    return options


def color_source_options() -> list[tuple[str | None, str]]:
    """(strategy_id_or_None, label) for the Color combo, one entry per
    analysis module plus the default.

    `None` means "tint a card's background by review status" - Keep/Reject
    green/red, Neutral the plain background - exactly today's behavior.
    Anything else tints the background by that strategy's own score instead
    (low to high, across whatever is currently visible), for scanning a
    Classic Vision-ranked folder's ordering at a glance without needing to
    sort by it first. Built from the registry, like `sort_options`, so a
    future module is colorable the moment it is runnable.
    """
    from ..ranking import available_strategies

    options: list[tuple[str | None, str]] = [(None, "Review Status")]
    for info in available_strategies():
        options.append((info.strategy_id, f"{info.display_name} Score"))
    return options


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
        self._show_detector_boxes = False
        # See color_source_options() docstring below, and the "Collapse
        # Bursts" View menu action - when true, the gallery shows only each
        # burst's top-ranked (burst_best) image instead of every member.
        self._collapse_bursts = False
        # None (the default) means "tint by review status" - see
        # color_source_options(). Anything else is a strategy id whose score
        # tints the background instead.
        self._color_source: str | None = None
        self._last_counts: dict[str, Any] = {}
        # Last-used parameters per ranking strategy, so re-running one starts
        # from what was chosen before rather than the defaults. Session-scoped
        # on purpose: these are per-shoot experiments, not a preference.
        self._ranking_params: dict[str, Any] = {}
        self._active_threads: list[Any] = []  # keeps background QThreads alive while running
        self._open_folder_in_progress = False
        self._open_folder_generation = 0
        self._open_folder_thread: Any | None = None
        self._folder_load_dialog: QProgressDialog | None = None
        self._folder_load_cancelled = False
        self._folder_load_snapshot: dict[str, Any] | None = None

        # Thumbnails decode off the UI thread (see core/thumbnail_loader.py);
        # this signal is how a finished background decode gets back to the
        # GUI thread to repaint just its one row. _thumbnails_loading tracks
        # in-flight paths so Qt re-asking for the same not-yet-ready cell
        # (which it does, repeatedly, while scrolling/repainting) doesn't
        # queue duplicate decode jobs for it.
        self._thumbnail_signal = ThumbnailReadySignal()
        self._thumbnail_signal.ready.connect(self._on_thumbnail_ready)
        self._thumbnail_signal.failed.connect(self._on_thumbnail_failed)
        self._thumbnails_loading: set[tuple[str, bool]] = set()

        self._folder_label = QLabel("No folder open")
        self._image_count_label = QLabel("Images: 0")
        self._counts_label = QLabel("")
        self._status_label = QLabel("Ready")
        self._status_message_label = QLabel("")
        self._gpu_status_label = QLabel("")

        self._central_widget = QWidget(self)
        self._gallery_model = ImageModel()
        self._gallery_model.set_thumbnail_provider(self._load_thumbnail)
        self._gallery_view = GalleryView(self._central_widget)
        self._gallery_view.setModel(self._gallery_model)
        self._gallery_view.doubleClicked.connect(self._open_loupe_for_index)
        self._gallery_view.keyPressSignal.connect(self._on_gallery_key_press)
        self._gallery_view.decisionRequested.connect(self._on_card_decision)

        self._filter_combo = QComboBox(self)
        self._filter_combo.addItems([FILTER_LABELS[f] for f in FILTERS])
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
        for field, label in sort_options():
            self._sort_combo.addItem(label, field)
        self._sort_combo.setCurrentIndex(max(0, self._sort_combo.findData(self._sort_field)))
        self._sort_combo.currentIndexChanged.connect(self._on_sort_field_changed)

        self._sort_direction_btn = QPushButton(self)
        self._sort_direction_btn.setCheckable(True)
        self._sort_direction_btn.setMaximumWidth(28)
        self._sort_direction_btn.clicked.connect(self._on_sort_direction_toggled)
        self._update_sort_direction_button()

        self._color_combo = QComboBox(self)
        for source, label in color_source_options():
            self._color_combo.addItem(label, source)
        self._color_combo.currentIndexChanged.connect(self._on_color_source_changed)

        self._build_ui()

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

        # One entry per registered ranking strategy (picklikeme.ranking), built
        # from the registry rather than listed here, so a new strategy appears
        # in both the Tools menu and the toolbar without touching this file.
        # The default strategy stays the AI model: clicking the toolbar button
        # itself (rather than opening its dropdown) runs that, so the
        # long-standing one-click "rank this folder" gesture is unchanged.
        self._rank_menu = QMenu("Rank", self)
        self._rank_strategy_actions: dict[str, QAction] = {}
        for info in self.service.ranking_strategies():
            action = self._make_action(
                f"{info.display_name}…", tooltip=info.description,
                triggered=lambda _checked=False, sid=info.strategy_id: self._rank_with_strategy(sid),
            )
            self._rank_menu.addAction(action)
            self._rank_strategy_actions[info.strategy_id] = action
        self._rank_action = self._make_action(
            "Rank…", icon=SP.SP_BrowserReload,
            tooltip="Score every image in the folder — choose a ranking method",
            triggered=lambda: self._rank_with_strategy(DEFAULT_STRATEGY_ID),
        )
        self._rank_action.setMenu(self._rank_menu)
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
        self._analytics_dashboard_action = self._make_action(
            "Analytics Dashboard…", icon=SP.SP_FileDialogDetailedView,
            tooltip="Browse past ranking/species-classification experiments and their metrics",
            triggered=self._show_analytics_dashboard,
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
        self._recent_folders_menu = RecentItemsMenu(
            self._recent_menu,
            self._settings,
            settings_key="recent_folders",
            on_select=self._open_recent_folder,
            empty_label="No recent folders yet",
            clear_label="Clear Recent Folders",
            limit=DEFAULT_RECENT_ITEMS_LIMIT,
        )
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
        view_menu.addSeparator()
        self._detector_boxes_action = QAction("Detector Boxes", self, checkable=True)
        self._detector_boxes_action.setToolTip(
            "Show the AI's detected-subject bounding boxes, and Classic Vision's measured eye, "
            "on gallery thumbnails and in the Loupe"
        )
        self._detector_boxes_action.toggled.connect(self._on_toggle_detector_boxes)
        view_menu.addAction(self._detector_boxes_action)
        self._collapse_bursts_action = QAction("Collapse Bursts", self, checkable=True)
        self._collapse_bursts_action.setToolTip(
            "Show only the top-ranked image of each burst; open one to flip through its "
            "burst mates in rank order"
        )
        self._collapse_bursts_action.toggled.connect(self._on_toggle_collapse_bursts)
        view_menu.addAction(self._collapse_bursts_action)

        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addMenu(self._rank_menu)
        tools_menu.addAction(self._apply_cutoff_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self._organize_action)
        tools_menu.addAction(self._species_action)
        tools_menu.addAction(self._crop_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self._analytics_dashboard_action)

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

        toolbar.addWidget(QLabel(" Filter: "))
        toolbar.addWidget(self._filter_combo)
        toolbar.addSeparator()

        toolbar.addAction(self._select_all_action)
        toolbar.addAction(self._keep_action)
        toolbar.addAction(self._reject_action)
        toolbar.addAction(self._neutral_action)
        toolbar.addAction(self._loupe_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Sort: "))
        toolbar.addWidget(self._sort_combo)
        toolbar.addWidget(self._sort_direction_btn)
        toolbar.addWidget(QLabel(" Color: "))
        toolbar.addWidget(self._color_combo)
        toolbar.addAction(self._detector_boxes_action)
        toolbar.addAction(self._collapse_bursts_action)
        toolbar.addSeparator()

        toolbar.addAction(self._rank_action)
        # MenuButtonPopup, not InstantPopup: the button's own half still runs
        # the default strategy on a single click (see _build_actions), and only
        # the arrow opens the list of the others.
        #
        # The menu comes from the QAction (set in _build_actions), not from
        # setMenu() on the button: QToolBar builds its button with
        # setDefaultAction(), and the button paints and popups its default
        # action's menu. Note that `QToolButton.menu()` still returns None in
        # that arrangement - it only reports an explicitly-set menu - so it is
        # a misleading thing to assert on; the style option's HasMenu feature
        # is what actually decides whether the arrow is drawn.
        rank_button = toolbar.widgetForAction(self._rank_action)
        if isinstance(rank_button, QToolButton):
            rank_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
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
        self._recent_folders_menu.reload()

    # -- folder loading, with a progress indicator while it runs ------------

    def _open_recent_folder(self, folder: str) -> None:
        """Click handler for a Recent Folders entry. A folder can vanish
        between being remembered and being reopened - moved, renamed, or on
        a drive that isn't mounted right now - so this checks before handing
        off to _start_open_folder, rather than letting that fail deeper in
        the stack with a less useful error."""
        if not Path(folder).is_dir():
            self._recent_folders_menu.remove(folder)
            QMessageBox.warning(
                self, "PeakPic - Open Folder", f"This folder could no longer be found:\n{folder}"
            )
            return
        self._start_open_folder(folder)

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
        # Spans only the brief synchronous portion of open_folder() (file
        # enumeration, ranked-file merge, thumbnail model population) -
        # closed again in _start_open_folder the moment the gallery has
        # real data, the same span the web review UI's own blocking overlay
        # covers (setLoading(true) before its fetch, off right after).
        # Nothing here tracks the background metadata pass that follows;
        # see _start_open_folder's comment for why that is no longer this
        # dialog's job.
        if self._folder_load_dialog is None:
            self._setup_folder_load_dialog()
        self._folder_load_dialog.setLabelText(f"Opening {Path(folder).name}\nScanning folder…")
        self._folder_load_dialog.setValue(0)
        self._folder_load_dialog.show()

    def _hide_folder_load_dialog(self) -> None:
        # QProgressDialog emits `canceled` from close()/cancel() whenever the
        # dialog hasn't reached its maximum value yet - true for every
        # programmatic close in this codebase now that the dialog is closed
        # as soon as the gallery is usable, not once the background load
        # hits 100%. Left connected, that would re-enter
        # _cancel_open_folder() and wrongly restore the previous session
        # right after a successful open. Clear the reference and disconnect
        # before close() so: (a) a reentrant call here sees None and no-ops
        # instead of hitting a deleted dialog, and (b) close() cannot fire
        # _cancel_open_folder() at all.
        dialog = self._folder_load_dialog
        if dialog is None:
            return
        self._folder_load_dialog = None
        dialog.canceled.disconnect(self._cancel_open_folder)
        dialog.close()
        dialog.deleteLater()

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

        self._recent_folders_menu.remember(folder_path)
        self.state.current_folder = result.get("input_folder") or self.state.current_folder
        self.state.image_count = result.get("counts", {}).get("total", 0)
        self._refresh_from_state(result)

        # The gallery now has real data - scores, ranks, thumbnails, review
        # status - and is fully usable. What's still running in the
        # background (ReviewSession's own thread) only fills in captured_at
        # and detected_category. detected_category has no Desktop consumer
        # at all today (only the web review UI's category chip/filter reads
        # it); captured_at feeds one optional sort mode ("Capture Time",
        # see _on_sort_field_changed) that pulls a fresh refresh on demand
        # the moment it's actually selected. Nothing currently visible or
        # interactive is waiting on this pass, so - deliberately, not an
        # oversight - the status message says only "Opened", never "...
        # loading in the background": that clause would be accurate for a
        # moment and then silently stale for however long the photographer
        # doesn't happen to trigger an unrelated refresh, which is a worse
        # message than none at all. Matches the web page exactly as far as
        # it matters: nothing here tracks this pass to completion.
        self._hide_folder_load_dialog()
        self._open_folder_in_progress = False
        self._set_status(f"Opened {self.state.current_folder}")

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
        """Reset open-folder bookkeeping after a run that didn't end in the
        ordinary success path in _start_open_folder - a failure, a cancel,
        or (only under the test-seam that monkeypatches run_in_background)
        an async completion. There is no longer a "the background metadata
        pass just finished" case here: nothing polls for that anymore (see
        _start_open_folder's comment), so this is purely open/cancel/fail
        cleanup."""
        if generation != self._open_folder_generation:
            return
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

    def _cancel_open_folder(self) -> None:
        if not self._open_folder_in_progress:
            return
        self._folder_load_cancelled = True
        self._open_folder_in_progress = False
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

    def _refresh_from_state(self, state: dict[str, Any]) -> None:
        input_folder = state.get("input_folder")
        self._folder_label.setText(f"Folder: {Path(input_folder).name}" if input_folder else "No folder open")
        self.state.image_count = state.get("counts", {}).get("total", 0)
        self._image_count_label.setText(f"Images: {self.state.image_count}")
        self._all_items = [
            ImageItem(
                path=image.get("image_path") or "",
                file_name=Path(image.get("image_path") or "").name,
                review_status=image.get("review_status", "neutral"),
                ai_suggestion=image.get("ai_suggestion"),
                algorithm_suggestion=image.get("algorithm_suggestion"),
                captured_at=image.get("captured_at"),
                # ranking_results already carries the AI model's own
                # score/rank under "ai-model" - ImageItem.score/.rank read it
                # from there, so there is nothing separate to pass here.
                ranking_results=image.get("ranking_results") or {},
                filter_reasons=image.get("filter_reasons") or {},
                metrics=image.get("metrics") or {},
                burst_id=image.get("burst_id"),
                burst_size=image.get("burst_size", 1),
                burst_rank=image.get("burst_rank", 1),
                burst_best=image.get("burst_best", True),
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
        _on_thumbnail_ready repaints this one row.

        Cached and requested under a (path, with_boxes) key - the same
        path needs two independent cache slots since review_thumbnail()
        returns a different file for each (a separate overlaid copy, not
        the plain thumbnail modified in place), and toggling the Detector
        Boxes view must not show a stale plain/overlaid thumbnail from
        before the toggle."""
        if not path:
            return None
        with_boxes = self._show_detector_boxes
        cache_key = (path, with_boxes)
        cached = self.cache_manager.get_thumbnail(cache_key)
        if cached is not None:
            return cached
        if cache_key not in self._thumbnails_loading:
            self._thumbnails_loading.add(cache_key)
            task = ThumbnailLoadTask(
                path, with_boxes,
                lambda p=path, wb=with_boxes: self.service.thumbnail_path(p, with_boxes=wb),
                self._thumbnail_signal,
            )
            QThreadPool.globalInstance().start(task)
        return None

    def _on_thumbnail_ready(self, path: str, with_boxes: bool, pixmap: QPixmap) -> None:
        self._thumbnails_loading.discard((path, with_boxes))
        self.cache_manager.put_thumbnail((path, with_boxes), pixmap)
        if with_boxes == self._show_detector_boxes:
            self._gallery_model.notify_thumbnail_ready(path)

    def _on_thumbnail_failed(self, path: str, with_boxes: bool) -> None:
        """A decode that raised, returned nothing, or produced a null pixmap
        (a locked file, a transient RAW-reader error) - not cached, since
        there is nothing to cache, but the in-flight marker must still be
        cleared. Without this, _load_thumbnail's `if cache_key not in
        self._thumbnails_loading` guard would never let this image be
        requested again for the rest of the session; that card would stay
        blank forever regardless of scrolling, resorting, or reopening the
        filter - the one path a bad frame could look identical to "still
        loading" with no way to tell the two apart."""
        self._thumbnails_loading.discard((path, with_boxes))

    def _on_toggle_detector_boxes(self, checked: bool) -> None:
        """Every visible cell's decoration is re-requested on the next
        repaint; _load_thumbnail keys its cache by (path, with_boxes), so
        this naturally fetches (or reuses an already-cached) overlaid or
        plain thumbnail per the new state rather than showing a stale one."""
        self._show_detector_boxes = checked
        self._gallery_view.viewport().update()

    def _on_toggle_collapse_bursts(self, checked: bool) -> None:
        """Disabled (the default) leaves the gallery exactly as it has
        always behaved - every image, individually. Enabled narrows the
        visible set to each burst's own burst_best image (see
        _filter_items) and shows a "+N" badge on any card whose burst has
        other members, so opening one (see _open_loupe) can offer them."""
        self._collapse_bursts = checked
        self._gallery_view.set_show_burst_badges(checked)
        self._apply_filter()

    # -- filtering ------------------------------------------------------------

    def _on_filter_changed(self, index: int) -> None:
        self._current_filter = FILTERS[index] if 0 <= index < len(FILTERS) else "all"
        self._apply_filter()

    def _apply_filter(self) -> None:
        filtered = self._filter_items(self._all_items, self._current_filter)
        if self._collapse_bursts:
            filtered = [item for item in filtered if item.burst_best]
        filtered = self._sort_items(filtered)
        self._gallery_model.set_items(filtered)
        self._update_color_source(filtered)
        if filtered and not self._gallery_view.currentIndex().isValid():
            self._gallery_view.setCurrentIndex(self._gallery_model.index(0, 0))
        self._gallery_view.set_empty_message(self._empty_message_for_current_state())

    def _on_color_source_changed(self, index: int) -> None:
        self._color_source = self._color_combo.itemData(index)
        # Burst Analysis ranks each burst's members by this same "selected
        # ranking strategy" (see ReviewSession.set_burst_strategy) - "Review
        # Status" (None) is not a ranking strategy, so that case falls back
        # to the AI model, same as burst_strategy's own default.
        self.service.set_burst_strategy(self._color_source or DEFAULT_STRATEGY_ID)
        self._refresh_from_state(self.service.load_session())

    def _update_color_source(self, items: list[ImageItem]) -> None:
        """Recompute the score range the gallery tints card backgrounds
        against, whenever the visible set or the chosen source changes -
        see color_source_options(). None (Review Status) needs no range at
        all; anything else is scaled against the low/high of that strategy's
        score among only the images actually visible right now, so the
        gradient always spans the full range of what's on screen instead of
        being skewed by images a filter has hidden."""
        if self._color_source is None:
            self._gallery_view.set_color_source(None, None)
            return
        scores = [s for s in (item.score_for(self._color_source) for item in items) if s is not None]
        score_range = (min(scores), max(scores)) if scores else None
        self._gallery_view.set_color_source(self._color_source, score_range)

    @staticmethod
    def _filter_items(items: list[ImageItem], current_filter: str) -> list[ImageItem]:
        """Matches the web review UI's filterImages() (review/page.py)
        exactly, including the two conflict directions and the AI-only
        filters, so a photographer moving between the two apps finds the
        same filter names doing the same thing."""
        if current_filter == "all":
            return list(items)
        if current_filter == "ai_keep":
            return [item for item in items if item.ai_suggestion == "keep"]
        if current_filter == "ai_reject":
            return [item for item in items if item.ai_suggestion == "reject"]
        if current_filter == "ai_keep_user_reject":
            return [item for item in items if item.ai_suggestion == "keep" and item.review_status == "reject"]
        if current_filter == "ai_reject_user_keep":
            return [item for item in items if item.ai_suggestion == "reject" and item.review_status == "keep"]
        if current_filter == "algorithm_keep":
            return [item for item in items if item.algorithm_suggestion == "keep"]
        if current_filter == "algorithm_reject":
            return [item for item in items if item.algorithm_suggestion == "reject"]
        if current_filter == "algorithm_keep_user_reject":
            return [item for item in items if item.algorithm_suggestion == "keep" and item.review_status == "reject"]
        if current_filter == "algorithm_reject_user_keep":
            return [item for item in items if item.algorithm_suggestion == "reject" and item.review_status == "keep"]
        if current_filter == "filtered":
            # Desktop-only, no web-UI equivalent (see FILTER_LABELS): any
            # image at least one analysis module explicitly excluded from
            # scoring, regardless of which strategy or reason.
            return [item for item in items if item.filter_reasons]
        return [item for item in items if item.review_status == current_filter]

    # -- sorting ----------------------------------------------------------------

    def _on_sort_field_changed(self, index: int) -> None:
        field = self._sort_combo.itemData(index)
        if not field:
            return
        self._sort_field = field
        if field == "captured_at" and not self.service.loading_state().get("complete", True):
            # captured_at is filled in lazily, in the background, after
            # Open Folder already returned (see _start_open_folder) - this
            # sort is the first moment a Desktop session actually needs it,
            # so pull whatever the background pass has finished so far right
            # now instead of showing a stale/partial view. loading_state()
            # is a cheap dict read; load_session() is the real refresh, and
            # it only runs here, once, on demand - not on any timer.
            self._refresh_from_state(self.service.load_session())
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
            if field.startswith(SORT_SCORE_PREFIX):
                # An image no module scored has no value for that module and
                # sorts to the end, exactly as an unranked image already does
                # for the AI score - see this method's docstring.
                return item.score_for(field[len(SORT_SCORE_PREFIX):])
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
            label = FILTER_LABELS.get(self._current_filter, self._current_filter.capitalize())
            return f"No images match the '{label}' filter"
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
        """Set the AI cutoff threshold, then apply the resulting AI
        suggestions to the gallery - mirrors the web review UI's "Apply AI
        Suggestions" behavior (review/session.py's apply_ai_suggestions,
        server.py's /api/review/apply-ai-suggestions), not previously wired
        into the desktop at all. A Neutral image has nothing manual at
        risk and is always updated. An image already marked Keep/Reject
        that disagrees with the new threshold is only overridden after an
        explicit confirmation showing exactly how many go each direction -
        "3 conflicts" doesn't tell a photographer whether they're about to
        lose 3 Keeps or 3 Rejects, which matters."""
        if not self.state.current_folder:
            self._set_status("Open a folder before setting the AI cutoff")
            return
        percent = self._cutoff_spin.value()
        state = self.service.set_keep_percent(percent)
        self._refresh_from_state(state)

        keep_to_reject = 0
        reject_to_keep = 0
        for image in state.get("images", []):
            status = image.get("review_status")
            suggestion = image.get("ai_suggestion")
            if suggestion is None or status not in ("keep", "reject") or suggestion == status:
                continue
            if status == "keep":
                keep_to_reject += 1
            else:
                reject_to_keep += 1
        conflicts = keep_to_reject + reject_to_keep

        include_decided = False
        if conflicts:
            message = (
                f"At this {percent:g}% cutoff, the AI disagrees with {conflicts} image(s) "
                "you already decided:\n\n"
                f"    {keep_to_reject} marked Keep would become Reject\n"
                f"    {reject_to_keep} marked Reject would become Keep\n\n"
                "Override these manual decisions to match the AI? Neutral images are "
                "always updated to the AI's suggestion regardless."
            )
            confirm = QMessageBox.question(
                self, "PeakPic - Apply Cutoff", message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            include_decided = confirm == QMessageBox.StandardButton.Yes

        result = self.service.apply_ai_suggestions(include_decided=include_decided)
        self._refresh_from_state(result["state"])
        if include_decided:
            self._set_status(
                f"AI cutoff set to {percent:g}% - applied to {result['applied']} neutral image(s), "
                f"overrode {result['overridden']} manual decision(s)"
            )
        elif conflicts:
            self._set_status(
                f"AI cutoff set to {percent:g}% - applied to {result['applied']} neutral image(s); "
                f"{conflicts} manual decision(s) left unchanged"
            )
        else:
            self._set_status(f"AI cutoff set to {percent:g}% - applied to {result['applied']} neutral image(s)")

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
        item = self._gallery_model.item_at(index.row()) if index.isValid() else None
        self._open_loupe_for_item(item)

    def _open_loupe_for_selection(self) -> None:
        index = self._gallery_view.currentIndex()
        item = self._gallery_model.item_at(index.row()) if index.isValid() else None
        self._open_loupe_for_item(item)

    def _open_loupe_for_item(self, item: ImageItem | None) -> None:
        """Opening a card normally scopes the Loupe to the current gallery's
        own visible order (unchanged from before Burst Analysis existed).

        In Collapse Bursts mode a visible card is one burst's burst_best
        image standing in for the whole group, so opening it instead scopes
        the Loupe to that burst's own members, ordered by burst_rank - "the
        existing review workflow while allowing navigation through the
        burst members" the feature asks for. Pulled from `self._all_items`,
        not the collapsed, filtered gallery model: the other members are
        deliberately not in that model's rows at all.
        """
        if self._collapse_bursts and item is not None and item.burst_id is not None:
            members = sorted(
                (i for i in self._all_items if i.burst_id == item.burst_id),
                key=lambda i: i.burst_rank,
            )
            self._open_loupe(items=members, start_row=0)
            return
        items = self._gallery_model.items()
        start_row = items.index(item) if item is not None and item in items else 0
        self._open_loupe(items=items, start_row=start_row)

    def _open_loupe(self, *, items: list[ImageItem], start_row: int) -> None:
        if not items:
            self._set_status("No images to review in the current filter")
            return
        paths = [item.path for item in items]
        start_row = max(0, min(start_row, len(paths) - 1))
        dialog = LoupeDialog(
            service=self.service, image_paths=paths, items=items, start_index=start_row,
            show_boxes=self._show_detector_boxes, parent=self,
        )
        dialog.exec()
        self._refresh_from_state(self.service.load_session())

    # -- ranking ----------------------------------------------------------------

    def _rank_with_strategy(self, strategy_id: str) -> None:
        """Collect the chosen strategy's parameters, then run it.

        Only the parameter-collection step differs per strategy, and only
        because the AI model's parameters (a checkpoint path, a checkbox) are
        not the numeric knobs `AlgorithmParametersDialog` generates itself.
        Everything after that - the background run, the progress dialog, the
        session refresh, the status line - is shared, because a ranking is a
        ranking whatever produced it.
        """
        if not self.state.current_folder:
            self._set_status("Open a folder before ranking it")
            return

        info = next(
            (i for i in self.service.ranking_strategies() if i.strategy_id == strategy_id), None
        )
        if info is None:
            self._set_status(f"Unknown ranking strategy: {strategy_id}")
            return

        kwargs = self._collect_ranking_parameters(strategy_id, info)
        if kwargs is None:  # the photographer cancelled
            return

        def _run(on_progress=None, on_stage=None):
            return self.service.rank_folder(
                strategy=strategy_id, on_stage=on_stage, on_progress=on_progress, **kwargs
            )

        def _on_success(result: dict[str, Any]) -> None:
            # A ranking run may change what a detector-box overlay shows for
            # any image (new/updated detections, or - Classic Vision - eye
            # data that did not exist the last time an overlaid thumbnail
            # was cached), so every cached pixmap must be dropped before the
            # gallery repaints, not just the session state refreshed.
            self.cache_manager.clear_thumbnails()
            self._refresh_from_state(result["state"])
            self._set_status(self._ranking_summary(result))
            device = result.get("device")
            if device:
                self.state.gpu_status = device
                self._gpu_status_label.setText(f"Device: {device}")

        thread = run_with_progress(self, info.display_name, _run, on_success=_on_success)
        self._active_threads.append(thread)
        thread.finished.connect(lambda: self._active_threads.remove(thread) if thread in self._active_threads else None)

    def _collect_ranking_parameters(self, strategy_id: str, info) -> dict[str, Any] | None:
        """The keyword arguments for `ReviewService.rank_folder`, or None if
        the photographer cancelled the parameter dialog.

        A strategy that declares numeric `ParamSpec`s gets a dialog generated
        from them, so adding one needs no code here. The single exception is
        the AI model, whose parameters are a file path and a checkbox rather
        than numbers - it keeps its own long-standing `RankDialog`, unchanged.
        """
        params_cls = self.service.ranking_params_class(strategy_id)
        if params_cls is not None and params_cls.specs():
            dialog = AlgorithmParametersDialog(
                params_cls=params_cls,
                title=f"{info.display_name} — Parameters",
                initial=self._ranking_params.get(strategy_id),
                parent=self,
            )
            if dialog.exec() != AlgorithmParametersDialog.DialogCode.Accepted:
                return None
            # Remembered for the session, so re-running with a tweak starts
            # from what was used last rather than from the defaults each time.
            params = dialog.parameters()
            self._ranking_params[strategy_id] = params
            return {"params": params}

        if strategy_id == DEFAULT_STRATEGY_ID:
            dialog = RankDialog(parent=self)
            if dialog.exec() != RankDialog.DialogCode.Accepted:
                return None
            return {"checkpoint": dialog.checkpoint_path(), "crop_birds": dialog.crop_birds()}

        return {}  # nothing to configure - run with the strategy's own defaults

    @staticmethod
    def _ranking_summary(result: dict[str, Any]) -> str:
        """One status line covering both a plain ranking and a filtered one."""
        ranked = result.get("image_count", 0)
        filtered = result.get("filtered") or {}
        if not filtered:
            return f"Ranked {ranked} images"
        breakdown = ", ".join(
            f"{REJECT_REASON_LABELS.get(reason, reason)}: {count}"
            for reason, count in sorted(filtered.items(), key=lambda item: -item[1])
        )
        skipped = sum(filtered.values())
        return f"Ranked {ranked} images; skipped {skipped} ({breakdown})"

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
        default_backend = self._settings.value("review/species_backend", "bioclip2")
        dialog = SpeciesLanguageDialog(default_language=default_language, default_backend=default_backend, parent=self)
        if dialog.exec() != SpeciesLanguageDialog.DialogCode.Accepted:
            return
        language = dialog.language()
        backend = dialog.backend()
        self._settings.setValue("review/species_language", language)
        self._settings.setValue("review/species_backend", backend)

        def _run(on_progress=None, on_stage=None):
            return self.service.organize_by_species(backend=backend, language=language, on_progress=on_progress)

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
        current_backend = self._settings.value("review/species_backend", "bioclip2")
        dialog = PreferencesDialog(
            default_theme=theme.current_theme_name(),
            default_language=current_language,
            default_backend=current_backend,
            parent=self,
        )
        if dialog.exec() != PreferencesDialog.DialogCode.Accepted:
            return
        self._set_theme(dialog.theme_name())
        self._settings.setValue("review/species_language", dialog.species_language())
        self._settings.setValue("review/species_backend", dialog.species_backend())

    def _show_analytics_dashboard(self) -> None:
        """Non-modal (`.show()`, not `.exec()`) so a photographer can keep
        the dashboard open while switching back to Review/the Loupe -
        browsing past experiments is a reference task, not a blocking
        workflow step, unlike Organize by Species's own dialog."""
        dialog = AnalyticsDashboard(parent=self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()

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
        # ReviewSession._background_load (the daemon thread filling in
        # captured_at/detected_category after Open Folder already returned)
        # checks this generation counter every 25 images and bails out the
        # moment it no longer matches - the same mechanism that already
        # stops it when a different folder is opened. Bumping it here means
        # a slow pass on a large folder does not keep running, and does not
        # keep writing to the annotations database, after the window it was
        # started for is already gone.
        self.service.session._loading_generation += 1

        # Every gallery thumbnail decode runs on QThreadPool.globalInstance()
        # - a process-wide singleton this window does not own and cannot
        # just walk away from. Confirmed by direct instrumentation (logging
        # threading.enumerate(), QThreadPool.activeThreadCount(), and every
        # QTimer/QThread/job this window tracks, at QApplication.aboutToQuit)
        # that closing this window while decodes are still in flight left
        # activeThreadCount() non-zero and the process hanging indefinitely
        # after app.exec() returned - never a Python-visible thread (daemon
        # or otherwise; threading.enumerate() only ever showed MainThread and
        # ReviewSession's own daemon thread), because QThreadPool's workers
        # are native Qt threads outside Python's threading module entirely.
        # A worker still running when this window - and _thumbnail_signal,
        # one of its children - gets torn down finishes later and tries to
        # emit onto an already-deleted QObject, confirmed to raise
        # "RuntimeError: Signal source has been deleted" from inside
        # QRunnable::run() on its own thread with nothing positioned to ever
        # recover from that. clear() drops every not-yet-started decode (the
        # bulk of a large folder's backlog - nothing left to wait for or
        # crash on); waitForDone() then blocks only for whatever handful of
        # decodes were already mid-flight, bounded by a timeout so a single
        # unusually slow RAW read cannot hang the close indefinitely by
        # itself (ThumbnailLoadTask's own isValid() check is the actual
        # safety net for that remaining sliver either way).
        QThreadPool.globalInstance().clear()
        QThreadPool.globalInstance().waitForDone(5000)

        super().closeEvent(event)

