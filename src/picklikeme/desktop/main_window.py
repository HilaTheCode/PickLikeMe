"""Main window for PeakPic Desktop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QModelIndex, Qt, QSettings, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
    QToolButton,
    QVBoxLayout,
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
from .filtering import FilterableRecord, apply_filters
from .views.gallery.thumbnail_delegate import DOMAIN_BY_STRATEGY
from .widgets.advanced_filters_panel import AdvancedFiltersPanel
from .widgets.design_system import (
    PRIMARY_HEIGHT,
    RADIUS_LG,
    RADIUS_SM,
    SPACING,
    USER_DECISION_LABEL,
    LabeledCombo,
    LabeledSearch,
    Panel,
    PrimaryButton,
    SecondaryButton,
    StatusLegend,
)
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
    SetUserDecisionsBySubfoldersDialog,
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
    "neutral": "Undecided",
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
# Burst member order for a burst-scoped Loupe session (opened from a
# collapsed burst card - see _open_loupe_for_item). Owned entirely by the
# Main Grid, not the Loupe: the Loupe used to carry its own Capture Time /
# Burst Score toggle and re-sort itself on every change, which is what
# caused the repeated Capture-Time/Score synchronization bugs (one sort mode
# silently breaking the other, or a mode change not actually reordering
# navigation). The Loupe now only ever receives an already-ordered member
# list and walks it - see LoupeDialog's own module docstring. Burst Score
# reuses burst_rank as-is (already score-descending - see
# burst_analysis.py's own docstring); Capture Time re-sorts by
# ImageItem.captured_at. Persisted via QSettings under
# BURST_SORT_SETTINGS_KEY so the photographer only picks it once - same key
# the old Loupe-level toggle used, so an existing preference carries over.
BURST_SORT_CAPTURE_TIME = "capture_time"
BURST_SORT_BURST_SCORE = "burst_score"
BURST_SORT_SETTINGS_KEY = "review/burst_sort_mode"
DEFAULT_BURST_SORT_MODE = BURST_SORT_BURST_SCORE
# Sorting by any analysis module's score, plus the two intrinsic file
# properties. A module's field is "score:<strategy_id>"; the bare "score" is
# kept as the default because it is what the window opens on and what
# ReviewSession's own load order already matches.
SORT_SCORE_PREFIX = "score:"
# "Whichever algorithm the Color selector currently names" - resolved on
# every sort (_sort_items) through the same _resolve_color_source the Grid's
# own coloring and score badge use, so the number shown on a card and the
# order the cards are in always come from ONE strategy.
#
# It is the default because the alternative was worse in exactly the case
# that matters: the bare "score" field below is the AI model's score
# specifically, so a folder ranked only by Crop Sharpness opened showing
# Crop Sharpness scores on every card while sorting them by an AI score
# none of them had - i.e. not sorted at all, and with no rank number, while
# appearing to be "sorted by score".
SORT_SELECTED_ALGORITHM = "score:__selected__"
SORT_FIELDS = (SORT_SELECTED_ALGORITHM, "score", "filename", "captured_at")
SORT_FIELD_LABELS = {
    SORT_SELECTED_ALGORITHM: "Selected Algorithm Score",
    "score": "AI Score",
    "filename": "File Name",
    "captured_at": "Capture Time",
}


def sort_options() -> list[tuple[str, str]]:
    """(field, label) for the Sort combo, one entry per analysis module.

    Built from the ranking registry rather than listed, so a new module
    becomes sortable at the same moment it becomes runnable. The AI model is
    already covered by the "score" field, so it is not repeated.
    `SORT_SELECTED_ALGORITHM` leads: see its own comment above.
    """
    from ..ranking import DEFAULT_STRATEGY_ID, available_strategies

    options = [
        (SORT_SELECTED_ALGORITHM, SORT_FIELD_LABELS[SORT_SELECTED_ALGORITHM]),
        ("score", SORT_FIELD_LABELS["score"]),
    ]
    for info in available_strategies():
        if info.strategy_id == DEFAULT_STRATEGY_ID:
            continue
        options.append((f"{SORT_SCORE_PREFIX}{info.strategy_id}", f"{info.display_name} Score"))
    options.append(("filename", SORT_FIELD_LABELS["filename"]))
    options.append(("captured_at", SORT_FIELD_LABELS["captured_at"]))
    return options


# Sentinel Color Source value meaning "Algorithm Ran Last" - see
# color_source_options's own docstring and MainWindow._resolve_color_source
# for how this differs from picking a specific strategy by name: this one
# re-resolves to whichever strategy actually ran most recently EVERY time
# it is used, rather than freezing to whatever that happened to be at
# selection time. Not a real strategy_id (never registered in `ranking`),
# so it can never collide with one - deliberately not `None`, which already
# means "User Decision" (see below).
ALGORITHM_RAN_LAST = "__algorithm_ran_last__"

# USER_DECISION_LABEL ("User Decision") is defined in design_system, next to
# the categories that mode selects - imported above.


def color_source_options() -> list[tuple[str | None, str]]:
    """(strategy_id_or_sentinel_or_None, label) for the Color combo - one
    entry per analysis module, plus "Algorithm Ran Last" and "User
    Decision".

    Two genuinely different KINDS of mode, and the selected one is the ONLY
    input to a card's color (see `design_system.resolve_status`):

    `None` -> "User Decision": Keep / Reject / Undecided, from the
    photographer's own decisions alone. An image nobody has reviewed is
    Undecided and stays neutral no matter what any algorithm scored it, what
    the cutoff would suggest, or whether an algorithm cutoff was ever
    recorded for it.

    A real strategy_id -> that strategy's own SCORE, tinted low to high
    across whatever is currently visible, for scanning a folder's ordering
    at a glance without sorting by it first. `ALGORITHM_RAN_LAST` (listed
    first - the default a freshly opened folder starts on, see
    `ReviewSession.latest_run_strategy`) is the same thing for "whichever
    strategy actually produced this folder's most recent completed run",
    re-resolved every time it is used (`MainWindow._resolve_color_source`)
    rather than pinned at selection time. The per-strategy entries are built
    from the registry, like `sort_options`, so a future module is colorable
    the moment it is runnable.
    """
    from ..ranking import available_strategies

    options: list[tuple[str | None, str]] = [
        (ALGORITHM_RAN_LAST, "Algorithm Ran Last"),
        (None, USER_DECISION_LABEL),
    ]
    for info in available_strategies():
        options.append((info.strategy_id, f"{info.display_name} Score"))
    return options


# The Grid's secondary-toolbar "Domain" filter (01_Grid.svg) - a UI-level
# grouping of the registered strategies' own DOMAIN_BY_STRATEGY label (see
# thumbnail_delegate.py), not a new backend concept: "Birds" narrows to
# images at least one Birds-domain strategy actually scored, and so on.
# Built from the same DOMAIN_BY_STRATEGY dict the thumbnail card's own
# domain indicator reads, so the filter option list and a card's own label
# can never name a domain differently from each other.
DOMAIN_ALL = "all"


def domain_options() -> list[tuple[str, str]]:
    labels = dict.fromkeys(DOMAIN_BY_STRATEGY.values())  # de-duplicated, insertion order
    return [(DOMAIN_ALL, "All Domains")] + [(label, label) for label in labels]


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
        # Best-effort per-path species lookup for Advanced Filters' Species
        # control (see _refresh_species_cache) - keyed by path, value is a
        # species.classifier.SpeciesPrediction or None once that path has
        # been checked and nothing was on record for it. Absence of the key
        # itself (as opposed to a stored None) means "not looked up yet" -
        # the incremental-lookup marker _refresh_species_cache relies on.
        self._species_by_path: dict[str, Any] = {}
        self._current_filter = "all"
        # Follows the Color selector's own strategy, so the score shown on a
        # card and the order the cards are in always come from the same
        # algorithm - see SORT_SELECTED_ALGORITHM.
        self._sort_field = SORT_SELECTED_ALGORITHM
        self._sort_ascending = False
        self._show_detector_boxes = False
        # See color_source_options() docstring below, and the "Collapse
        # Bursts" View menu action - when true, the gallery shows only each
        # burst's top-ranked (burst_best) image instead of every member.
        self._collapse_bursts = False
        # The Main Grid's own Burst Order preference - see BURST_SORT_*
        # above and the "Burst Order" View menu submenu built in
        # _build_menu_bar. Read once at startup; _set_burst_sort_mode keeps
        # both this attribute and QSettings in sync from then on.
        stored_burst_sort_mode = self._settings.value(BURST_SORT_SETTINGS_KEY, DEFAULT_BURST_SORT_MODE)
        self._burst_sort_mode = (
            stored_burst_sort_mode
            if stored_burst_sort_mode in (BURST_SORT_CAPTURE_TIME, BURST_SORT_BURST_SCORE)
            else DEFAULT_BURST_SORT_MODE
        )
        # ALGORITHM_RAN_LAST (the default, matching the combo's own first
        # item - see color_source_options()) dynamically resolves to
        # whichever strategy most recently ran (_resolve_color_source);
        # None means "tint by review status"; anything else is a specific
        # strategy id whose score tints the background instead.
        self._color_source: str | None = ALGORITHM_RAN_LAST
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
        self._make_toolbar_combo_compact(self._filter_combo, min_chars=13, max_width=150)

        # Product Direction: "the highest priority is now to move the
        # advanced filtering capabilities into the Review Window" - narrows
        # the same gallery model the simple Filter combo above and Collapse
        # Bursts already narrow (see _apply_filter), through the one shared
        # filtering.py engine (desktop/filtering.py's own docstring: "the
        # same engine should drive Main Review Window / Analytics Dashboard
        # / Loupe navigation").
        self._advanced_filters_panel = AdvancedFiltersPanel(self)
        self._advanced_filters_panel.criteriaChanged.connect(self._apply_filter)

        self._cutoff_combo = QComboBox(self)
        for preset in KEEP_PERCENT_PRESETS:
            self._cutoff_combo.addItem(f"{preset:g}%", preset)
        self._cutoff_combo.addItem("Custom…", None)
        self._cutoff_combo.currentIndexChanged.connect(self._on_cutoff_preset_changed)
        self._make_toolbar_combo_compact(self._cutoff_combo, min_chars=8, max_width=100)

        self._cutoff_spin = QDoubleSpinBox(self)
        self._cutoff_spin.setRange(0.0, 100.0)
        self._cutoff_spin.setSuffix("%")
        self._cutoff_spin.setValue(KEEP_PERCENT_PRESETS[0])
        self._cutoff_spin.setEnabled(False)
        # Manual QA Phase 11: "Threshold changes should immediately recolor
        # the gallery." Deliberately separate from _apply_cutoff (the
        # "Apply Cutoff" action, which bulk-writes review_status with its
        # own conflict-confirmation dialog) - this only moves
        # ReviewSession.keep_percent, which set_keep_percent's own docstring
        # already guarantees "never changes anyone's review_status", so a
        # live preview here is always safe to fire on every value change.
        self._cutoff_spin.valueChanged.connect(self._on_cutoff_preview_changed)

        self._sort_combo = QComboBox(self)
        for field, label in sort_options():
            self._sort_combo.addItem(label, field)
        self._sort_combo.setCurrentIndex(max(0, self._sort_combo.findData(self._sort_field)))
        self._sort_combo.currentIndexChanged.connect(self._on_sort_field_changed)
        self._make_toolbar_combo_compact(self._sort_combo, min_chars=13, max_width=148)

        self._sort_direction_btn = QPushButton(self)
        self._sort_direction_btn.setCheckable(True)
        self._sort_direction_btn.setMaximumWidth(28)
        self._sort_direction_btn.clicked.connect(self._on_sort_direction_toggled)
        self._update_sort_direction_button()

        self._color_combo = QComboBox(self)
        for source, label in color_source_options():
            self._color_combo.addItem(label, source)
        self._color_combo.currentIndexChanged.connect(self._on_color_source_changed)
        self._make_toolbar_combo_compact(self._color_combo, min_chars=12, max_width=138)

        # A toolbar-level, always-visible way to see/change the Burst Order
        # preference (see BURST_SORT_* and _set_burst_sort_mode) - the View
        # menu's own "Burst Order" submenu still exists and stays in sync
        # (both read/write the exact same self._burst_sort_mode /
        # _set_burst_sort_mode, no second sorting mechanism). Two items only
        # ("Time"/"Score"), so a compact combo reads cleanly without a label
        # explaining it further - "Burst:" is enough context.
        self._burst_sort_combo = QComboBox(self)
        self._burst_sort_combo.addItem("Time", BURST_SORT_CAPTURE_TIME)
        self._burst_sort_combo.addItem("Score", BURST_SORT_BURST_SCORE)
        self._burst_sort_combo.setToolTip(
            "Burst Order - the order Loupe navigation uses when opened from a collapsed burst card"
        )
        index = self._burst_sort_combo.findData(self._burst_sort_mode)
        if index >= 0:
            self._burst_sort_combo.setCurrentIndex(index)
        self._burst_sort_combo.currentIndexChanged.connect(self._on_burst_sort_combo_changed)
        self._make_toolbar_combo_compact(self._burst_sort_combo, min_chars=5, max_width=85)

        # Grid redesign secondary toolbar (01_Grid.svg) - Domain narrows to
        # images at least one strategy in that domain group actually scored
        # (see thumbnail_delegate.DOMAIN_BY_STRATEGY/domain_options above);
        # Search is a plain filename substring filter, both composed with
        # the existing Filter combo/Collapse Bursts/Advanced Filters in
        # _apply_filter - one more AND-combined narrowing, not a second
        # filtering mechanism.
        self._domain_combo = QComboBox(self)
        for value, label in domain_options():
            self._domain_combo.addItem(label, value)
        self._domain_combo.currentIndexChanged.connect(self._on_domain_changed)
        self._make_toolbar_combo_compact(self._domain_combo, min_chars=12, max_width=140)
        self._domain_filter = DOMAIN_ALL

        self._search_widget = LabeledSearch("Search", "Filename or tag…", parent=self)
        self._search_widget.edit.textChanged.connect(self._on_search_changed)
        self._search_text = ""

        self._build_ui()

    @staticmethod
    def _make_toolbar_combo_compact(combo: QComboBox, *, min_chars: int, max_width: int) -> None:
        """Cap a toolbar combo box's width instead of letting it grow to fit
        its longest item (e.g. Filter's "Filtered (Skipped by an analysis
        module)" or Sort/Color's "<strategy display name> Score", which can
        run to 50+ characters for a ranking strategy with a long name). Left
        at Qt's default AdjustToContentsOnFirstShow, the combo box sizes
        itself to that longest label, which on a MacBook-width window was
        pushing the Color and Collapse Bursts controls off the visible
        toolbar (see top-toolbar layout fix).

        minimumContentsLength sets a floor so the box never gets so narrow
        the selected value is unreadable; maximumWidth caps how far it can
        grow. The dropdown popup itself is unaffected and still shows each
        item at full width - only the closed box's current-value display is
        constrained, so a tooltip mirrors the full text for when it elides.
        """
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(min_chars)
        combo.setMaximumWidth(max_width)
        combo.currentTextChanged.connect(combo.setToolTip)
        combo.setToolTip(combo.currentText())

    def _build_ui(self) -> None:
        self.setWindowTitle("PeakPic Desktop")
        # 1470px-class MacBook screen (the design spec's own target width)
        # plus a little slack, so the redesigned chrome never opens already
        # cramped on a first launch with no saved geometry yet. A user's own
        # resize is remembered afterwards via _save_state/_restore_state.
        self.resize(1520, 900)
        self._apply_theme(self._settings.value("theme", theme.DEFAULT_THEME))
        self._build_menu_bar()
        self.setCentralWidget(self._central_widget)
        self._central_widget.setLayout(self._build_central_layout())
        self._build_status_bar()
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
            # A QPalette alongside the stylesheet - QSS alone only reaches
            # widget types build_stylesheet names explicitly; a QPalette is
            # inherited by every widget, including QTableWidget/QListWidget/
            # QTabWidget in dialogs like the Analytics Dashboard that would
            # otherwise keep rendering with Qt's default light palette even
            # under a dark stylesheet - see theme.build_qpalette.
            app.setPalette(theme.build_qpalette(theme.current_palette()))
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
        """The Grid screen's full chrome (`01_Grid.svg`): a primary toolbar
        (the high-value actions), a secondary toolbar (filter/sort/color/
        domain/search/view), a left sidebar (Recent Folders/Collections),
        the thumbnail grid, and a bottom status legend - built from the
        design system's own reusable components (`widgets/design_system.py`)
        rather than as a one-off layout, per the redesign's own "build
        reusable components instead of implementing each screen
        independently" rule."""
        layout = QVBoxLayout(self._central_widget)
        layout.setContentsMargins(SPACING * 2, SPACING * 2, SPACING * 2, SPACING * 2)
        layout.setSpacing(SPACING)
        layout.addWidget(self._build_primary_bar())
        layout.addWidget(self._build_secondary_bar())

        body = QHBoxLayout()
        body.setSpacing(SPACING * 2)
        body.addWidget(self._build_sidebar())

        main_column = QVBoxLayout()
        main_column.setSpacing(SPACING)
        main_column.addWidget(self._advanced_filters_panel)
        main_column.addWidget(self._gallery_view, 1)
        legend_panel = Panel(radius=RADIUS_LG, parent=self._central_widget)
        legend_layout = QVBoxLayout(legend_panel)
        legend_layout.setContentsMargins(SPACING * 2, SPACING, SPACING * 2, SPACING)
        # The legend lists the categories the CURRENTLY selected Color mode
        # produces (Keep/Reject/Undecided, or Scored/Filtered Out/Skipped) -
        # kept on the window so _update_color_source can re-point it.
        self._status_legend = StatusLegend(color_source=self._resolve_color_source(), parent=legend_panel)
        legend_layout.addWidget(self._status_legend)
        main_column.addWidget(legend_panel)
        main_column_widget = QWidget(self._central_widget)
        main_column_widget.setLayout(main_column)
        body.addWidget(main_column_widget, 1)

        layout.addLayout(body, 1)
        return layout

    def _build_primary_bar(self) -> QWidget:
        """The Grid's primary toolbar (`04_Toolbar.svg`) - Rank/Apply
        Cutoff/Keep/Reject/Clear Selection/Export on the left (the
        high-value, one-click actions), Color Source/Sort on the right
        (the two controls that change what the whole grid displays) -
        matching the SVG's own single-row arrangement at 1470px width."""
        palette = theme.current_palette()
        bar = Panel(radius=RADIUS_LG, parent=self._central_widget)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACING * 2, SPACING, SPACING * 2, SPACING)
        layout.setSpacing(SPACING)

        brand = QLabel("△ PeakPick", bar)
        brand.setStyleSheet(f"color: {palette.text_primary}; font-size: 18px; font-weight: 700; border: none;")
        layout.addWidget(brand)
        layout.addSpacing(SPACING * 2)

        self._rank_button = QToolButton(bar)
        self._rank_button.setObjectName("rankButton")
        self._rank_button.setDefaultAction(self._rank_action)
        self._rank_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._rank_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._rank_button.setText("Rank")
        self._rank_button.setMinimumHeight(PRIMARY_HEIGHT)
        self._rank_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rank_button.setStyleSheet(
            f"QToolButton {{ background-color: {palette.panel_bg_secondary}; color: {palette.text_primary}; "
            f"border: 1px solid {palette.accent}; border-radius: {RADIUS_SM}px; "
            f"padding: 4px {SPACING * 2}px; font-weight: 600; font-size: 12px; }}"
            f"QToolButton:hover {{ background-color: {palette.hover_bg}; }}"
            f"QToolButton::menu-button {{ border: none; width: 16px; }}"
        )
        layout.addWidget(self._rank_button)

        self._apply_cutoff_button = PrimaryButton("Apply Cutoff", self._cutoff_subtitle(), accent_color=palette.accent, parent=bar)
        self._apply_cutoff_button.clicked.connect(self._apply_cutoff_action.trigger)
        layout.addWidget(self._apply_cutoff_button)
        self._cutoff_combo.setParent(bar)
        self._cutoff_spin.setParent(bar)
        layout.addWidget(self._cutoff_combo)
        layout.addWidget(self._cutoff_spin)
        self._cutoff_spin.valueChanged.connect(lambda _v: self._apply_cutoff_button.set_subtitle(self._cutoff_subtitle()))

        keep_btn = PrimaryButton("Keep", "Selected", accent_color=palette.keep_fg, parent=bar)
        keep_btn.clicked.connect(self._keep_action.trigger)
        layout.addWidget(keep_btn)

        reject_btn = PrimaryButton("Reject", "Selected", accent_color=palette.reject_fg, parent=bar)
        reject_btn.clicked.connect(self._reject_action.trigger)
        layout.addWidget(reject_btn)

        clear_btn = PrimaryButton("Clear", "Selection", parent=bar)
        clear_btn.clicked.connect(self._clear_selection_action.trigger)
        layout.addWidget(clear_btn)

        export_btn = PrimaryButton("Export", "Keep-marked", parent=bar)
        export_btn.setToolTip(self._import_action.toolTip())
        export_btn.clicked.connect(self._import_action.trigger)
        layout.addWidget(export_btn)

        # Sits with the other one-click actions rather than in a menu: after
        # anything moves files behind the app's back (Finder, another tool,
        # an Arrange run from a different open folder) this is the one
        # control that makes the grid agree with the disk again.
        self._refresh_button = PrimaryButton("Refresh", "Folder", parent=bar)
        self._refresh_button.setToolTip(self._refresh_action.toolTip())
        self._refresh_button.clicked.connect(self._refresh_action.trigger)
        layout.addWidget(self._refresh_button)

        layout.addStretch(1)

        self._color_combo.setParent(bar)
        color_widget = LabeledCombo("Color Source", combo=self._color_combo, parent=bar)
        layout.addWidget(color_widget)

        self._sort_combo.setParent(bar)
        sort_widget = LabeledCombo("Sort", combo=self._sort_combo, parent=bar)
        sort_row = QHBoxLayout()
        sort_row.setContentsMargins(0, 0, 0, 0)
        sort_row.setSpacing(4)
        sort_row.addWidget(sort_widget)
        self._sort_direction_btn.setParent(bar)
        sort_row.addWidget(self._sort_direction_btn, 0, Qt.AlignmentFlag.AlignBottom)
        sort_container = QWidget(bar)
        sort_container.setLayout(sort_row)
        layout.addWidget(sort_container)

        return bar

    def _cutoff_subtitle(self) -> str:
        return f"Top {self._cutoff_spin.value():g}%"

    def _build_secondary_bar(self) -> QWidget:
        """The Grid's secondary toolbar - Filter/Domain/Search/Burst/View,
        plus the Detector Boxes/Collapse Bursts toggles."""
        palette = theme.current_palette()
        bar = Panel(radius=RADIUS_LG, parent=self._central_widget)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACING * 2, SPACING, SPACING * 2, SPACING)
        layout.setSpacing(SPACING * 2)

        self._filter_combo.setParent(bar)
        layout.addWidget(LabeledCombo("Filter", combo=self._filter_combo, parent=bar))

        self._domain_combo.setParent(bar)
        layout.addWidget(LabeledCombo("Domain", combo=self._domain_combo, parent=bar))

        self._search_widget.setParent(bar)
        layout.addWidget(self._search_widget)

        self._burst_sort_combo.setParent(bar)
        layout.addWidget(LabeledCombo("Burst", combo=self._burst_sort_combo, parent=bar))

        view_combo = QComboBox(bar)
        view_combo.addItem("Grid")
        view_combo.setEnabled(False)
        view_combo.setToolTip("Only Grid view is currently available.")
        LabeledCombo.cap_width(view_combo, min_chars=6, max_width=90)
        view_widget = LabeledCombo("View", combo=view_combo, parent=bar)
        layout.addWidget(view_widget)

        layout.addStretch(1)

        boxes_btn = SecondaryButton("Detector Boxes", checkable=True, parent=bar)
        boxes_btn.setToolTip(self._detector_boxes_action.toolTip())
        boxes_btn.toggled.connect(self._detector_boxes_action.setChecked)
        self._detector_boxes_action.toggled.connect(boxes_btn.setChecked)
        layout.addWidget(boxes_btn)

        collapse_btn = SecondaryButton("Collapse Bursts", checkable=True, parent=bar)
        collapse_btn.setToolTip(self._collapse_bursts_action.toolTip())
        collapse_btn.toggled.connect(self._collapse_bursts_action.setChecked)
        self._collapse_bursts_action.toggled.connect(collapse_btn.setChecked)
        layout.addWidget(collapse_btn)

        return bar

    def _build_sidebar(self) -> QWidget:
        """The Grid's left sidebar (`01_Grid.svg`) - Recent Folders (real,
        backed by the same QSettings-persisted list the File menu's Recent
        Folders submenu already uses - see RecentItemsMenu) and a
        Collections section. Collections has no backend concept anywhere in
        this codebase yet (no persisted "collection" of images exists) - shown
        as a labeled placeholder rather than a non-functional button, per
        the design spec's own "do not add controls merely because there is
        empty space" rule; a real Collections feature is future work, not
        something this visual redesign pass should fabricate."""
        palette = theme.current_palette()
        sidebar = Panel(radius=RADIUS_LG, parent=self._central_widget)
        sidebar.setFixedWidth(200)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(SPACING * 2, SPACING * 2, SPACING * 2, SPACING * 2)
        layout.setSpacing(SPACING)

        libraries_label = QLabel("LIBRARIES", sidebar)
        libraries_label.setStyleSheet(f"color: {palette.text_muted}; font-size: 11px; font-weight: 700; border: none;")
        layout.addWidget(libraries_label)

        recent_label = QLabel("Recent Folders", sidebar)
        recent_label.setStyleSheet(f"color: {palette.text_muted}; font-size: 12px; border: none;")
        layout.addWidget(recent_label)

        self._recent_folders_list = QListWidget(sidebar)
        self._recent_folders_list.setFrameShape(QListWidget.Shape.NoFrame)
        self._recent_folders_list.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; color: {palette.text_primary}; font-size: 12px; }}"
            f"QListWidget::item {{ padding: 4px 6px; border-radius: 6px; }}"
            f"QListWidget::item:selected {{ background-color: {palette.hover_bg}; color: {palette.accent}; }}"
            f"QListWidget::item:hover {{ background-color: {palette.hover_bg}; }}"
        )
        self._recent_folders_list.itemClicked.connect(
            lambda item: self._open_recent_folder(item.data(Qt.ItemDataRole.UserRole))
        )
        layout.addWidget(self._recent_folders_list)
        self._refresh_recent_sidebar()

        layout.addSpacing(SPACING * 2)
        collections_label = QLabel("COLLECTIONS", sidebar)
        collections_label.setStyleSheet(f"color: {palette.text_muted}; font-size: 11px; font-weight: 700; border: none;")
        layout.addWidget(collections_label)
        collections_placeholder = QLabel("Coming in a future update", sidebar)
        collections_placeholder.setWordWrap(True)
        collections_placeholder.setStyleSheet(f"color: {palette.text_muted}; font-size: 11px; border: none;")
        layout.addWidget(collections_placeholder)

        layout.addStretch(1)
        return sidebar

    def _refresh_recent_sidebar(self) -> None:
        """Keeps the sidebar's Recent Folders list in sync with
        `self._recent_folders_menu` - called whenever that list changes
        (a folder opened, Recent Folders cleared)."""
        if not hasattr(self, "_recent_folders_list"):
            return
        self._recent_folders_list.clear()
        for folder in self._recent_folders_menu.items():
            item = QListWidgetItem(Path(folder).name)
            item.setData(Qt.ItemDataRole.UserRole, folder)
            item.setToolTip(folder)
            self._recent_folders_list.addItem(item)

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
        # Refresh Folder is a READ-ONLY resync of the grid with the disk (see
        # ReviewSession.refresh): it adds files that appeared, drops files
        # that are gone, and touches no score, decision, crop or ranking. F5
        # because that is what "reload what I am looking at" is bound to
        # everywhere else a photographer works.
        self._refresh_action = self._make_action(
            "Refresh Folder", icon=SP.SP_BrowserReload, shortcut=QKeySequence.Refresh,
            tooltip="Rescan the open folder for added or removed files",
            triggered=self._refresh_folder,
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
        self._clear_selection_action = self._make_action(
            "Clear Selection", icon=SP.SP_DialogResetButton,
            tooltip="Deselect every image", triggered=self._clear_selection,
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
            tooltip="Record the selected algorithm's keep-percent cutoff — an algorithm decision, "
                    "not a User Decision",
            triggered=self._apply_cutoff,
        )
        self._clear_algorithm_decisions_action = self._make_action(
            "Clear Algorithm Decisions", icon=SP.SP_DialogResetButton,
            tooltip="Discard every recorded algorithm cutoff — your own Keep/Reject decisions are kept",
            triggered=self._clear_algorithm_decisions,
        )
        self._organize_action = self._make_action(
            "Organize…", icon=SP.SP_DirIcon,
            tooltip="Move your Keep/Reject images into Selected/Rejected folders — "
                    "undecided images are left alone",
            triggered=self._organize,
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
        self._ground_truth_action = self._make_action(
            "Set User Decisions by Subfolders…", icon=SP.SP_DialogApplyButton,
            tooltip="Bulk-set Keep/Reject/Neutral from folders of already-sorted images, for Ground Truth",
            triggered=self._set_user_decisions_by_subfolders,
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
            is_valid=lambda path: Path(path).is_dir(),
        )
        file_menu.addSeparator()
        file_menu.addAction(self._refresh_action)
        file_menu.addAction(self._import_action)
        file_menu.addSeparator()
        file_menu.addAction(self._settings_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        review_menu = menu_bar.addMenu("Review")
        review_menu.addAction(self._select_all_action)
        review_menu.addAction(self._clear_selection_action)
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
            "burst mates in the order set by Burst Order below"
        )
        self._collapse_bursts_action.toggled.connect(self._on_toggle_collapse_bursts)
        view_menu.addAction(self._collapse_bursts_action)

        # The Main Grid's own Burst Order choice - the ONLY place burst
        # member order is decided (see BURST_SORT_* above). The Loupe
        # receives whichever order this produces and just navigates it;
        # changing it here requires closing and reopening the Loupe to take
        # effect, by design - see _open_loupe_for_item.
        burst_order_menu = view_menu.addMenu("Burst Order")
        burst_order_menu.setToolTip(
            "Order burst members are navigated in when the Loupe is opened from a collapsed "
            "burst card. Change here, then reopen the Loupe to see the new order."
        )
        burst_order_group = QActionGroup(self)
        burst_order_group.setExclusive(True)
        self._burst_order_capture_time_action = QAction("Capture Time", self, checkable=True)
        self._burst_order_score_action = QAction("Score (highest first)", self, checkable=True)
        burst_order_group.addAction(self._burst_order_capture_time_action)
        burst_order_group.addAction(self._burst_order_score_action)
        burst_order_menu.addAction(self._burst_order_capture_time_action)
        burst_order_menu.addAction(self._burst_order_score_action)
        self._burst_order_capture_time_action.setChecked(self._burst_sort_mode == BURST_SORT_CAPTURE_TIME)
        self._burst_order_score_action.setChecked(self._burst_sort_mode == BURST_SORT_BURST_SCORE)
        self._burst_order_capture_time_action.triggered.connect(
            lambda: self._set_burst_sort_mode(BURST_SORT_CAPTURE_TIME)
        )
        self._burst_order_score_action.triggered.connect(lambda: self._set_burst_sort_mode(BURST_SORT_BURST_SCORE))

        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addMenu(self._rank_menu)
        tools_menu.addAction(self._apply_cutoff_action)
        tools_menu.addAction(self._clear_algorithm_decisions_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self._organize_action)
        tools_menu.addAction(self._species_action)
        tools_menu.addAction(self._crop_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self._analytics_dashboard_action)
        tools_menu.addAction(self._ground_truth_action)

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(about_action)

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
        """The USER DECISION breakdown - Keep / Reject / Undecided - color-
        coded to match the gallery cards. Counts what the photographer has
        actually decided (ReviewSession.counts reads `user_decision`), so a
        freshly ranked folder reads "Undecided <everything>" rather than
        claiming a review that never happened."""
        self._last_counts = counts
        palette = theme.current_palette()
        keep = counts.get("keep", 0)
        reject = counts.get("reject", 0)
        undecided = counts.get("undecided", counts.get("neutral", 0))
        self._counts_label.setText(
            f'<span style="color:{palette.keep_fg}">Keep {keep}</span>&nbsp;&nbsp;'
            f'<span style="color:{palette.reject_fg}">Reject {reject}</span>&nbsp;&nbsp;'
            f'<span style="color:{palette.filtered_fg}">Undecided {undecided}</span>'
        )

    def _restore_state(self) -> None:
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _save_state(self) -> None:
        self._settings.setValue("window/geometry", self.saveGeometry())

    def initialize(self) -> None:
        self._initialized = True
        self.state.status_message = "Desktop shell ready"
        self._status_label.setText("Ready")
        self._status_message_label.setText(self.state.status_message)
        self._recent_folders_menu.reload()
        self._refresh_recent_sidebar()

    # -- folder loading, with a progress indicator while it runs ------------

    def _open_recent_folder(self, folder: str) -> None:
        """Click handler for a Recent Folders entry. A folder can vanish
        between being remembered and being reopened - moved, renamed, or on
        a drive that isn't mounted right now - so this checks before handing
        off to _start_open_folder, rather than letting that fail deeper in
        the stack with a less useful error."""
        if not Path(folder).is_dir():
            self._recent_folders_menu.remove(folder)
            self._refresh_recent_sidebar()
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
        self._refresh_recent_sidebar()
        self.state.current_folder = result.get("input_folder") or self.state.current_folder
        self.state.image_count = result.get("counts", {}).get("total", 0)
        # A genuinely new folder - species lookups keyed by the old folder's
        # paths are no longer useful and would otherwise accumulate forever
        # across however many folders a session opens.
        self._species_by_path = {}
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
            self._sync_color_source_from_session()
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
                algorithm_decision=image.get("algorithm_decision"),
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
        self._refresh_species_cache(self._all_items)
        self._refresh_advanced_filter_options()
        self._apply_filter()
        self._update_status_counts(state.get("counts", {}))

    def _refresh_species_cache(self, items: list[ImageItem]) -> None:
        """Best-effort per-path species lookup for Advanced Filters'
        Species control - mirrors AnalyticsDashboard's ImageExplorerTab.
        _populate_candidates, except with no single run/classifier_id to
        scope to (Organize by Species is a separate, file-moving-only
        workflow not tied to ReviewSession) - so this asks SpeciesCache for
        "the most recent prediction from any classifier" (get_any) instead.

        Incremental: only paths not already a key in self._species_by_path
        are queried, so _refresh_from_state's frequent re-invocation (every
        color-source change, every cutoff-preview tick) costs nothing once
        a folder's images have all been looked up once. _start_open_folder
        resets the dict when a genuinely new folder is opened.
        """
        missing = [item.path for item in items if item.path and item.path not in self._species_by_path]
        if not missing:
            return
        try:
            from ..species.cache import DEFAULT_SPECIES_DB, SpeciesCache

            cache = SpeciesCache(DEFAULT_SPECIES_DB)
            try:
                for path in missing:
                    self._species_by_path[path] = cache.get_any(path)
            finally:
                cache.close()
        except Exception:  # noqa: BLE001 - species lookup is best-effort, must never block Review
            for path in missing:
                self._species_by_path.setdefault(path, None)

    def _build_filterable_records(self) -> list[FilterableRecord]:
        """Adapts the Review Window's own live `ImageItem` list into the
        shared filtering engine's generic shape - see filtering.py's own
        docstring for why this adapter, rather than a data-model
        unification, is what "one shared engine" means here."""
        strategy_id = self._resolve_color_source() or DEFAULT_STRATEGY_ID
        records: list[FilterableRecord] = []
        for item in self._all_items:
            metrics = item.metrics.get(strategy_id) or {}
            if not metrics:
                # The current Color Source strategy (e.g. the AI model)
                # never produces these detailed per-metric measurements
                # (only Classic Vision/EyePose do) - fall back to whichever
                # strategy's metrics ARE present rather than showing
                # everything as unmeasured.
                metrics = next((m for m in item.metrics.values() if m), {})
            reject_reason = item.filter_reasons.get(strategy_id) or next(
                iter(item.filter_reasons.values()), None
            )
            prediction = self._species_by_path.get(item.path)
            records.append(
                FilterableRecord(
                    path=item.path,
                    folder=str(Path(item.path).parent) if item.path else "",
                    filename=item.display_name,
                    user_decision=item.review_status,
                    algorithm_decision=item.algorithm_suggestion,
                    reject_reason=reject_reason,
                    species=prediction.species if prediction is not None else None,
                    species_confidence=prediction.confidence if prediction is not None else None,
                    score=item.score,
                    eye_confidence=metrics.get("eye_confidence"),
                    head_confidence=metrics.get("head_confidence"),
                    subject_size=metrics.get("subject_size"),
                    eye_sharpness=metrics.get("eye_sharpness"),
                    subject_sharpness=metrics.get("subject_sharpness"),
                    burst_id=item.burst_id,
                    burst_size=item.burst_size,
                    burst_rank=item.burst_rank,
                    burst_best=item.burst_best,
                )
            )
        return records

    def _refresh_advanced_filter_options(self) -> None:
        """Keeps the panel's Folder/Species/Reject Reason/Burst Rank combos
        in sync with whatever is actually in the currently open folder -
        called whenever self._all_items is rebuilt."""
        folders = sorted({str(Path(item.path).parent) for item in self._all_items if item.path})
        species = sorted({
            prediction.species
            for prediction in self._species_by_path.values()
            if prediction is not None and prediction.species
        })
        reject_reasons = [(code, REJECT_REASON_LABELS.get(code, code)) for code in REJECT_REASON_LABELS]
        max_burst_size = max((item.burst_size for item in self._all_items), default=1)
        self._advanced_filters_panel.set_available_options(
            folders=folders, species=species, reject_reasons=reject_reasons, max_burst_size=max_burst_size,
        )

    def _open_folder_dialog(self) -> None:
        # Manual QA Issue 2: must always start from the last folder actually
        # opened via Open Folder - never Desktop/Documents/the project
        # folder/an output folder. _default_folder_for_dialog() already
        # existed and reads exactly that (falling back to home only the
        # very first time this app has ever opened a folder); it just
        # wasn't wired up here yet.
        folder = QFileDialog.getExistingDirectory(self, "Open Folder", self._default_folder_for_dialog())
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
        other members, so opening one (see _open_loupe) can offer them.

        The action's own label always names the CURRENT action, not the
        current state - "Collapse Bursts" while expanded (clicking it
        collapses them), "Uncollapse Bursts" once collapsed (clicking it
        expands them again), so the toolbar/menu text never silently
        disagrees with what a click actually does right now."""
        self._collapse_bursts = checked
        self._collapse_bursts_action.setText("Uncollapse Bursts" if checked else "Collapse Bursts")
        self._gallery_view.set_show_burst_badges(checked)
        self._apply_filter()

    def _set_burst_sort_mode(self, mode: str) -> None:
        """The single place Burst Order is decided - see BURST_SORT_* and
        _open_loupe_for_item, the only place this value is READ. Reachable
        from two UI surfaces (the toolbar's _burst_sort_combo and the View
        menu's "Burst Order" submenu, see _build_tool_bar/_build_menu_bar) -
        both call this same setter rather than writing self._burst_sort_mode
        directly, so this is also the one place that keeps them showing the
        same value as each other. Persisted immediately, same as every other
        QSettings-backed preference in this window (e.g. _set_theme), so it
        survives a restart without the photographer needing to touch it
        twice."""
        self._burst_sort_mode = mode
        self._settings.setValue(BURST_SORT_SETTINGS_KEY, mode)
        self._burst_order_capture_time_action.setChecked(mode == BURST_SORT_CAPTURE_TIME)
        self._burst_order_score_action.setChecked(mode == BURST_SORT_BURST_SCORE)
        combo_index = self._burst_sort_combo.findData(mode)
        if combo_index >= 0 and self._burst_sort_combo.currentIndex() != combo_index:
            self._burst_sort_combo.blockSignals(True)
            self._burst_sort_combo.setCurrentIndex(combo_index)
            self._burst_sort_combo.blockSignals(False)

    def _on_burst_sort_combo_changed(self, index: int) -> None:
        mode = self._burst_sort_combo.itemData(index)
        if mode is not None and mode != self._burst_sort_mode:
            self._set_burst_sort_mode(mode)

    # -- filtering ------------------------------------------------------------

    def _on_filter_changed(self, index: int) -> None:
        self._current_filter = FILTERS[index] if 0 <= index < len(FILTERS) else "all"
        self._apply_filter()

    def _on_domain_changed(self, index: int) -> None:
        self._domain_filter = self._domain_combo.itemData(index) or DOMAIN_ALL
        self._apply_filter()

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text.strip().lower()
        self._apply_filter()

    def _apply_filter(self) -> None:
        filtered = self._filter_items(self._all_items, self._current_filter)
        if self._collapse_bursts:
            filtered = [item for item in filtered if item.burst_best]
        if self._domain_filter != DOMAIN_ALL:
            domain_strategies = {sid for sid, label in DOMAIN_BY_STRATEGY.items() if label == self._domain_filter}
            filtered = [item for item in filtered if domain_strategies & set(item.ranking_results)]
        if self._search_text:
            filtered = [item for item in filtered if self._search_text in item.display_name.lower()]
        criteria = self._advanced_filters_panel.criteria
        if criteria.is_active():
            # AND the Advanced Filters panel's own narrowing on top of the
            # simple Filter combo + Collapse Bursts above - both routes
            # into the one gallery model, exactly as the product direction
            # asks ("filtering must affect the main image grid ... support
            # multiple simultaneous filters"). Records are only built from
            # what's already survived the cheaper filters, not the full
            # folder, since apply_filters() only needs to see candidates.
            by_path = {item.path: item for item in filtered}
            records = [r for r in self._build_filterable_records() if r.path in by_path]
            matching_paths = {r.path for r in apply_filters(records, criteria)}
            filtered = [item for item in filtered if item.path in matching_paths]
        filtered = self._sort_items(filtered)
        self._gallery_model.set_items(filtered)
        self._update_color_source(filtered)
        if filtered and not self._gallery_view.currentIndex().isValid():
            self._gallery_view.setCurrentIndex(self._gallery_model.index(0, 0))
        self._gallery_view.set_empty_message(self._empty_message_for_current_state())

    def _sync_color_source_from_session(self) -> None:
        """Make the Color Source combo (and everything it drives - Grid
        coloring, Apply Cutoff, Burst ranking, Filtering) reflect THIS
        folder's own "Algorithm Ran Last" (`ReviewSession.
        latest_run_strategy`) rather than whatever the combo happened
        to be showing from a previously-open folder, or its own first-item
        default ("User Decision" - see `color_source_options`).

        Before this existed, `_color_source` (a plain UI-local variable, set
        ONLY by `_on_color_source_changed`) was never synchronised FROM
        `ReviewSession.burst_strategy` - only ever pushed the other
        direction, combo -> session. On a freshly opened folder that had
        been ranked only by a non-default strategy, that meant the combo
        kept showing "User Decision" while the session had already
        (correctly, after `latest_run_strategy`) selected the strategy
        that actually has data - a real, visible mismatch between what the
        photographer sees selected and what the Grid/Cutoff/Filter actually
        use, which is exactly the class of bug this whole pass exists to
        close. Called once per folder open (`_finish_open_folder`) and once
        per completed ranking run (`_run_ranking`'s `_on_success`) - a fresh
        ranking result IS the new "Algorithm Ran Last" the moment it
        finishes, so the Grid should reflect it immediately without the
        photographer having to touch the Color Source combo by hand.
        """
        self._color_source = ALGORITHM_RAN_LAST
        index = self._color_combo.findData(ALGORITHM_RAN_LAST)
        self._color_combo.blockSignals(True)
        self._color_combo.setCurrentIndex(max(0, index))
        self._color_combo.blockSignals(False)
        resolved = self._resolve_color_source()
        self.service.set_burst_strategy(resolved or DEFAULT_STRATEGY_ID)
        self._update_color_source(self._gallery_model.items())

    def _resolve_color_source(self) -> str | None:
        """What `self._color_source` actually means right now: a real
        strategy_id, `None` (User Decision), or - if it is the
        `ALGORITHM_RAN_LAST` sentinel - whichever strategy
        `ReviewSession.latest_run_strategy` currently resolves to, re-checked
        on every call rather than cached, so it always tracks the true
        latest run even as new rankings complete. Every caller that used to
        read `self._color_source` directly to decide what to score/color/
        filter by now goes through this instead - see `color_source_options`
        for why the sentinel exists.
        """
        if self._color_source == ALGORITHM_RAN_LAST:
            return self.service.session.latest_run_strategy()
        return self._color_source

    def _on_color_source_changed(self, index: int) -> None:
        self._color_source = self._color_combo.itemData(index)
        # Burst Analysis ranks each burst's members by this same "selected
        # ranking strategy" (see ReviewSession.set_burst_strategy) - "User
        # Decision" (None) is not a ranking strategy at all, so that case
        # falls back to the AI model, same as burst_strategy's own default.
        self.service.set_burst_strategy(self._resolve_color_source() or DEFAULT_STRATEGY_ID)
        # Scroll-preserving (see _refresh_preserving_scroll's own docstring,
        # Manual QA Issue 1): switching Color Source recolors the whole
        # gallery through the same full model reset a decision change does,
        # so it must not jump the scroll position back to the top either.
        self._refresh_preserving_scroll(self.service.load_session())

    def _on_cutoff_preview_changed(self, percent: float) -> None:
        """Manual QA Phase 11: moving the Keep Threshold spinner recolors
        the gallery immediately - a pure display preview via
        ReviewSession.set_keep_percent (never touches review_status), not
        the explicit, confirmed "Apply Cutoff" action below."""
        if not self.state.current_folder:
            return
        self._refresh_preserving_scroll(self.service.set_keep_percent(percent))

    def _update_color_source(self, items: list[ImageItem]) -> None:
        """Propagate the chosen Color mode - and, for an algorithm mode, the
        score range it spans - to the gallery.

        `items` is the currently VISIBLE set, and the range is measured over
        exactly that: an algorithm-colored card is tinted by where its own
        score sits between the lowest and highest on screen (see
        `ThumbnailCardDelegate._score_fraction`), which is what makes the
        color correspond to the score rather than to a keep/reject verdict
        at some threshold. User Decision mode has no range at all.
        """
        strategy_id = self._resolve_color_source()
        self._gallery_view.set_color_source(strategy_id, score_range=self._score_range_for(items, strategy_id))
        self._status_legend.set_color_source(strategy_id)

    @staticmethod
    def _score_range_for(items: list[ImageItem], strategy_id: str | None) -> tuple[float, float] | None:
        """(min, max) of `strategy_id`'s own scores across `items`, or None
        when there is no algorithm selected or nothing it scored."""
        if strategy_id is None:
            return None
        scores = [score for item in items if (score := item.score_for(strategy_id)) is not None]
        return (min(scores), max(scores)) if scores else None

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

        if field == SORT_SELECTED_ALGORITHM:
            # Resolved here, once per sort, rather than stored: the Color
            # selector's "Algorithm Ran Last" is itself dynamic, so freezing
            # a strategy id at the moment Sort was chosen would drift out of
            # step with the scores the cards are actually showing.
            selected = self._resolve_color_source()
            field = f"{SORT_SCORE_PREFIX}{selected}" if selected else "filename"

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
        if self._advanced_filters_panel.criteria.is_active():
            return "No images match the current Advanced Filters"
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
        """Record the selected algorithm's current cutoff for this folder -
        as an ALGORITHM decision, never as a User Decision.

        This action is what taught the whole codebase the difference. It used
        to write the cutoff's keep/reject straight into `review_decisions`
        through the same call a Grid button click uses, confirming only the
        handful of CONFLICTS with already-decided images and silently
        sweeping every undecided one. One click on a 5,986-image folder
        therefore produced 5,986 rows that were, from that moment on,
        indistinguishable from images the photographer had reviewed by hand:
        the Grid colored them all as decisions, the counts claimed them, and
        Arrange would have filed all of them into _Selected/_Rejected.

        Now every row it writes carries DECISION_SOURCE_ALGORITHM (see
        ReviewSession._apply_suggestions), so none of them is ever read as a
        User Decision; an image the photographer HAS decided is not touched
        at all; and the confirmation states up front how many images this is
        about to record something for, rather than mentioning only the
        conflicts. Applies whichever strategy the Color picker currently
        shows, same fallback-to-AI-model rule as _on_color_source_changed.
        """
        if not self.state.current_folder:
            self._set_status("Open a folder before applying an algorithm cutoff")
            return
        strategy_id = self._resolve_color_source() or DEFAULT_STRATEGY_ID
        strategy_label = self._color_combo.currentText() if self._color_source else "AI Model"
        percent = self._cutoff_spin.value()
        state = self.service.set_keep_percent(percent)
        self._refresh_preserving_scroll(state)

        # Two different populations, counted separately because they mean
        # different things: images this will WRITE something for (no user
        # decision), and images it will leave alone but whose owner disagrees
        # with the algorithm (a real user decision - reported, never touched).
        to_write = 0
        conflicts = 0
        for image in state.get("images", []):
            suggestion = image.get("algorithm_suggestion")
            if suggestion is None:
                continue
            decision = image.get("review_status")
            if decision in ("keep", "reject"):
                conflicts += suggestion != decision
            else:
                to_write += 1

        message = (
            f"Record {strategy_label}'s {percent:g}% cutoff for {to_write} image(s)?\n\n"
            "This is an ALGORITHM decision, shown when you color the grid by this algorithm. "
            "It is not a User Decision: it does not color the grid in User Decision mode, and "
            "Arrange will not file these images.\n\n"
            f"{conflicts} image(s) you decided yourself disagree with this cutoff; those are left "
            "exactly as you set them."
        )
        confirm = QMessageBox.question(
            self, "PeakPic - Apply Cutoff", message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            self._set_status("Apply Cutoff cancelled - no decisions recorded")
            return

        result = self.service.apply_algorithm_suggestions(strategy_id, include_decided=True)
        self._refresh_from_state(result["state"])
        self._set_status(
            f"{strategy_label} cutoff at {percent:g}% recorded for "
            f"{result['applied'] + result['overridden']} image(s) as algorithm decisions; "
            f"{conflicts} of your own decision(s) left unchanged"
        )

    def _clear_algorithm_decisions(self) -> None:
        """Undo every recorded algorithm cutoff for this folder, leaving the
        photographer's own Keep/Reject untouched - the escape hatch for an
        "Apply Cutoff" that was not what was wanted (see _apply_cutoff)."""
        if not self.state.current_folder:
            self._set_status("Open a folder first")
            return
        recorded = self._last_counts.get("algorithm_decisions", 0)
        if not recorded:
            self._set_status("No algorithm decisions recorded for this folder")
            return
        confirm = QMessageBox.question(
            self, "PeakPic - Clear Algorithm Decisions",
            f"Discard {recorded} recorded algorithm decision(s)?\n\n"
            "Your own Keep/Reject decisions are not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        removed = self.service.clear_algorithm_decisions()
        self._refresh_preserving_scroll(self.service.load_session())
        self._set_status(f"Cleared {removed} algorithm decision(s)")

    # -- review actions -------------------------------------------------------

    def _select_all_visible(self) -> None:
        """Select every image in the current filter/sort view - the first
        step of "filter by status, select all, apply one decision to all
        of them": _apply_filter() already put only the filtered set into
        the model, so selectAll() naturally only selects that set."""
        self._gallery_view.selectAll()
        count = len(self._gallery_model.items())
        self._set_status(f"Selected {count} image(s)" if count else "No images to select")

    def _clear_selection(self) -> None:
        self._gallery_view.clearSelection()
        self._set_status("Selection cleared")

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
        self._refresh_preserving_scroll(self.service.load_session())

    def _refresh_preserving_scroll(self, state: dict[str, Any]) -> None:
        """Same as `_refresh_from_state`, except the gallery's scroll
        position survives it. `_apply_filter` (which `_refresh_from_state`
        always ends up calling) rebuilds the gallery model via
        `ImageModel.set_items` - a full `beginResetModel`/`endResetModel` -
        every time, because the filtered/sorted set can genuinely change
        shape (a filter that excludes on review_status, a sort keyed by
        it). Qt has no way to know "this is the same content, just
        redecorated" across a full reset, so it drops the scroll position
        to 0 - previously true even for the single most common action in
        the whole app, clicking one card's Keep/Reject button, making
        reviewing anything past the first screenful of a large folder
        actively painful (see the regression test for the exact repro).

        The restore is deferred one event-loop turn (`QTimer.singleShot`)
        rather than applied immediately: the scrollbar's range is not
        necessarily recomputed synchronously inside `endResetModel()` -
        setting the value immediately can be clamped against the OLD
        range and silently discarded once the real layout catches up.
        """
        scrollbar = self._gallery_view.verticalScrollBar()
        saved_scroll = scrollbar.value()
        self._refresh_from_state(state)
        QTimer.singleShot(0, lambda: scrollbar.setValue(saved_scroll))

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
        the Loupe to that burst's own members, ordered by the Main Grid's
        own Burst Order setting (self._burst_sort_mode - see the View menu's
        "Burst Order" submenu and BURST_SORT_* above) - "the existing review
        workflow while allowing navigation through the burst members" the
        feature asks for. Pulled from `self._all_items`, not the collapsed,
        filtered gallery model: the other members are deliberately not in
        that model's rows at all.

        The Loupe itself never re-sorts - it receives exactly this order and
        walks it (see LoupeDialog's own module docstring). This is the fix
        for the repeated Capture-Time/Score synchronization bugs: there is
        now exactly one place a burst's member order is decided, not two
        that could disagree or desync.
        """
        if self._collapse_bursts and item is not None and item.burst_id is not None:
            unsorted_members = [i for i in self._all_items if i.burst_id == item.burst_id]
            mode = self._burst_sort_mode
            if mode == BURST_SORT_BURST_SCORE and not self._burst_score_available(unsorted_members):
                # Never silently show a Burst Score order that is secretly
                # just Capture Time (burst_rank degenerates to capture order
                # when no member has been scored by the active Color Source
                # - see burst_analysis.py) - fall back for THIS burst only,
                # without touching the photographer's saved preference.
                mode = BURST_SORT_CAPTURE_TIME
                self._set_status(
                    "Burst Score unavailable for the current Color Source - opening Loupe in Capture Time order"
                )
            members = self._sort_burst_members(unsorted_members, mode)
            self._open_loupe(items=members, start_row=0, burst_scoped=True)
            return
        items = self._gallery_model.items()
        start_row = items.index(item) if item is not None and item in items else 0
        self._open_loupe(items=items, start_row=start_row)

    @staticmethod
    def _sort_burst_members(members: list[ImageItem], mode: str) -> list[ImageItem]:
        """Always derives from the given list via `sorted()` (never mutates
        it in place) - so this is a pure function of (members, mode) with no
        hidden state, the exact property that made the old Loupe-side
        version's "sort fresh from an immutable original every time" fix
        work: picking the same mode twice always reproduces the same
        sequence, regardless of what was requested in between."""
        if mode == BURST_SORT_CAPTURE_TIME:
            return sorted(members, key=lambda i: i.captured_at or "")
        return sorted(members, key=lambda i: i.burst_rank)

    def _burst_score_available(self, members: list[ImageItem]) -> bool:
        """Whether the active Color Source (ReviewSession.burst_strategy)
        has actually scored at least one member of this specific burst - see
        _sort_burst_members's caller. When it hasn't, burst_rank carries no
        real signal (analyze_bursts's tie-break falls back to capture
        order), so Burst Score would silently coincide with Capture Time."""
        strategy_id = getattr(self.service.session, "burst_strategy", None)
        return strategy_id is not None and any(item.score_for(strategy_id) is not None for item in members)

    def _open_loupe(self, *, items: list[ImageItem], start_row: int, burst_scoped: bool = False) -> None:
        if not items:
            self._set_status("No images to review in the current filter")
            return
        paths = [item.path for item in items]
        start_row = max(0, min(start_row, len(paths) - 1))
        dialog = LoupeDialog(
            service=self.service, image_paths=paths, items=items, start_index=start_row,
            show_boxes=self._show_detector_boxes, burst_scoped=burst_scoped, parent=self,
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
        self._run_ranking(strategy_id, info, kwargs, force_preprocess=False)

    def _run_ranking(self, strategy_id: str, info, kwargs: dict[str, Any], *, force_preprocess: bool) -> None:
        """The actual background run, factored out of `_rank_with_strategy`
        so a `CropCacheVersionMismatch` (see `_on_error` below) can retry
        itself with `force_preprocess=True` through the exact same path,
        rather than duplicating the thread-launch wiring."""

        def _run(on_progress=None, on_stage=None):
            return self.service.rank_folder(
                strategy=strategy_id, on_stage=on_stage, on_progress=on_progress,
                force_preprocess=force_preprocess, **kwargs
            )

        def _on_success(result: dict[str, Any]) -> None:
            # A ranking run may change what a detector-box overlay shows for
            # any image (new/updated detections, or - Classic Vision - eye
            # data that did not exist the last time an overlaid thumbnail
            # was cached), so every cached pixmap must be dropped before the
            # gallery repaints, not just the session state refreshed.
            self.cache_manager.clear_thumbnails()
            # The ranking that just finished is now this folder's own
            # "Algorithm Ran Last" (ReviewService.rank_folder already
            # re-opened the session, which recomputed
            # ReviewSession.burst_strategy from it - see
            # ReviewSession.latest_run_strategy) - sync the Color
            # Source combo to match before repainting the gallery, so it
            # shows this run's results without the photographer needing to
            # touch the combo themselves.
            self._sync_color_source_from_session()
            self._refresh_from_state(result["state"])
            self._set_status(self._ranking_summary(result))
            device = result.get("device")
            if device:
                self.state.gpu_status = device
                self._gpu_status_label.setText(f"Device: {device}")

        def _on_error(message: str) -> None:
            # CropCacheVersionMismatch's own message shape (preprocess.py) -
            # matched by text rather than exception type because run_with_
            # progress's worker only ever reports str(exc) across the thread
            # boundary (see core/jobs.py), never the exception object itself.
            # The message text is already a pinned, tested contract (see
            # test_preprocess_pipeline.py's own assertIn("--force", ...)), so
            # this is a stable thing to match on, not a fragile guess.
            if not force_preprocess and "different parameters" in message and "--force" in message:
                answer = QMessageBox.question(
                    self,
                    f"PeakPic - {info.display_name}",
                    "The subject-crop cache was built with different detection/crop settings and "
                    "needs to be rebuilt before this ranking can continue - this re-runs subject "
                    "detection for every image in the folder, which can take a while.\n\n"
                    "Rebuild the cache now and retry?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self._run_ranking(strategy_id, info, kwargs, force_preprocess=True)
                    return
            QMessageBox.warning(self, f"PeakPic - {info.display_name}", f"{info.display_name} failed:\n{message}")

        thread = run_with_progress(self, info.display_name, _run, on_success=_on_success, on_error=_on_error)
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
        """File by USER DECISION only - see ReviewSession.arrange. The
        preview counts come from `keep_paths`/`reject_paths`, which read
        `user_decision`, so the numbers in this dialog are exactly the images
        the photographer has decided and nothing else: a ranking result, an
        algorithm cutoff recorded by Apply Cutoff, or a high score is not a
        reason to move a file."""
        if not self.state.current_folder:
            self._set_status("Open a folder before organizing it")
            return
        preview = self.service.arrange(dry_run=True)
        undecided = self._last_counts.get("undecided", self._last_counts.get("neutral", 0))
        message = (
            f"This will move the images you decided yourself:\n"
            f"  {preview['selected']} Keep -> {preview['selected_dir']}\n"
            f"  {preview['rejected']} Reject -> {preview['rejected_dir']}\n\n"
            f"{undecided} undecided image(s) are left exactly where they are. Continue?"
        )
        confirm = QMessageBox.question(self, "PeakPic - Organize", message)
        if confirm != QMessageBox.StandardButton.Yes:
            return
        result = self.service.arrange(dry_run=False)
        # An explicit rescan, not `load_session()`. Arrange has just moved
        # files out of this folder, so the grid it was showing a moment ago
        # is stale by construction - and `load_session` only re-serialises
        # whatever the session already holds, it does not look at the disk.
        # Going through refresh_folder means the "files that moved are gone
        # from the grid" guarantee is this call site's, not a side effect of
        # arrange happening to reload when it found something to move.
        self._refresh_from_state(self.service.refresh_folder())
        self._set_status(f"Organized {result['moved']} image(s); {result['errors']} error(s)")

    # -- refresh -----------------------------------------------------------------

    def _refresh_folder(self) -> None:
        """Resync the grid with the folder on disk (Refresh Folder / F5).

        Read-only by construction: everything it can change comes from
        re-reading the folder, the strategy CSVs and the annotation store, so
        no score, decision, crop or ranking can be altered by pressing it -
        see `ReviewSession.refresh`.

        Reports the delta rather than a bare "done", because the whole reason
        to press this is to find out whether the disk still matches what is
        on screen; "no change" is as useful an answer as "3 removed".
        """
        if not self.state.current_folder:
            self._set_status("Open a folder before refreshing it")
            return
        before = {item.path for item in self._all_items}
        self._refresh_from_state(self.service.refresh_folder())
        after = {item.path for item in self._all_items}
        added, removed = len(after - before), len(before - after)
        if added or removed:
            self._set_status(f"Refreshed: {added} added, {removed} removed ({len(after)} images)")
        else:
            self._set_status(f"Refreshed: no change ({len(after)} images)")

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
        default_species_list_path = self._settings.value("review/species_list_path", "") or None
        dialog = SpeciesLanguageDialog(
            default_language=default_language,
            default_backend=default_backend,
            default_species_list_path=default_species_list_path,
            parent=self,
        )
        if dialog.exec() != SpeciesLanguageDialog.DialogCode.Accepted:
            return
        language = dialog.language()
        backend = dialog.backend()
        species_list_path = dialog.species_list_path()
        self._settings.setValue("review/species_language", language)
        self._settings.setValue("review/species_backend", backend)
        # "" (not None) so QSettings.value's own "" default above reads back
        # cleanly next time - QSettings round-trips None inconsistently
        # across platforms, "" does not.
        self._settings.setValue("review/species_list_path", species_list_path or "")

        def _run(on_progress=None, on_stage=None):
            return self.service.organize_by_species(
                backend=backend, language=language, species_list_path=species_list_path, on_progress=on_progress
            )

        def _on_success(result: dict[str, Any]) -> None:
            # Same reasoning as _organize: this moved files, so the grid has
            # to be rebuilt from the disk rather than re-serialised.
            self._refresh_from_state(self.service.refresh_folder())
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

    # -- set user decisions by subfolders (Ground Truth) -------------------------

    def _set_user_decisions_by_subfolders(self) -> None:
        """Version 2 workflow (see ground_truth.py's own module docstring):
        Keep/Reject each accept multiple subfolders of the Root Folder;
        Neutral is inferred automatically for everything else under it.
        The Root Folder is always the currently open review folder - one
        walk of it is what makes "everything not in Keep/Reject" a
        well-defined set at all, so unlike the rest of this method's
        sibling Tools actions, this one now requires a folder to be open."""
        if not self.state.current_folder:
            self._set_status("Open a folder before using Set User Decisions by Subfolders")
            return
        dialog = SetUserDecisionsBySubfoldersDialog(root_folder=self.state.current_folder, parent=self)
        if dialog.exec() != SetUserDecisionsBySubfoldersDialog.DialogCode.Accepted:
            return
        root_folder = dialog.root_folder()
        keep_folders = dialog.keep_folders()
        reject_folders = dialog.reject_folders()

        def _preview_run(on_progress=None, on_stage=None):
            return self.service.preview_ground_truth_import(
                root_folder=root_folder, keep_folders=keep_folders, reject_folders=reject_folders,
                on_progress=on_progress,
            )

        def _on_preview_error(message: str) -> None:
            QMessageBox.warning(
                self, "PeakPic - Set User Decisions by Subfolders", f"Could not scan the folders:\n{message}",
            )

        thread = run_with_progress(
            self, "Scanning Folders", _preview_run,
            on_success=self._confirm_and_apply_ground_truth_import, on_error=_on_preview_error,
        )
        self._active_threads.append(thread)
        thread.finished.connect(lambda: self._active_threads.remove(thread) if thread in self._active_threads else None)

    def _confirm_and_apply_ground_truth_import(self, preview: dict[str, Any]) -> None:
        totals = preview["totals"]
        lines = [
            f"Keep\t{totals['keep']}",
            f"Reject\t{totals['reject']}",
            f"Neutral\t{totals['neutral']}",
            "",
            f"Already matching\t{totals['already_matching']}",
            f"Will change\t{totals['will_change']}",
        ]
        if totals["conflicts"]:
            lines.append(f"Conflicts (found in more than one folder - will be skipped)\t{totals['conflicts']}")
        if totals["will_change"] == 0:
            QMessageBox.information(
                self, "PeakPic - Set User Decisions by Subfolders", "\n".join(lines) + "\n\nNothing to change.",
            )
            return
        choice = QMessageBox.question(
            self, "PeakPic - Set User Decisions by Subfolders",
            "\n".join(lines) + f"\n\nApply these {totals['will_change']} change(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            self._set_status("Set User Decisions by Subfolders cancelled")
            return

        def _apply_run(on_progress=None, on_stage=None):
            return self.service.apply_ground_truth_import()

        def _on_apply_success(result: dict[str, Any]) -> None:
            self._refresh_from_state(result["state"])
            summary = (
                f"Updated Keep: {result['updated_keep']}\n"
                f"Updated Reject: {result['updated_reject']}\n"
                f"Updated Neutral: {result['updated_neutral']}\n"
                f"Skipped: {len(result['skipped'])}\n"
                f"Conflicts: {len(result['conflicts'])}"
            )
            QMessageBox.information(self, "PeakPic - Set User Decisions by Subfolders", summary)
            total_updated = result["updated_keep"] + result["updated_reject"] + result["updated_neutral"]
            self._set_status(f"Updated {total_updated} user decision(s)")

        def _on_apply_error(message: str) -> None:
            QMessageBox.warning(
                self, "PeakPic - Set User Decisions by Subfolders", f"Could not apply changes:\n{message}",
            )

        thread = run_with_progress(
            self, "Applying User Decisions", _apply_run, on_success=_on_apply_success, on_error=_on_apply_error,
        )
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
        workflow step, unlike Organize by Species's own dialog.

        The live Review context (Root Folder, Color Source, Keep
        Threshold) is a snapshot of "right now" - a fresh dashboard is
        constructed every time this is opened (never reused), so it is
        never more than one click stale."""
        dialog = AnalyticsDashboard(
            settings=self._settings,
            root_folder=self.state.current_folder,
            color_source=self._resolve_color_source(),
            keep_percent=self.service.session.keep_percent,
            items=self._all_items,
            parent=self,
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()

    def _show_about(self) -> None:
        """Answers "am I actually running the build I think I am" without
        guessing - the exact git commit (and its own commit date, the
        honest proxy this project has for a "build timestamp" - see
        resolve_git_commit_timestamp's own docstring), installed package
        version, interpreter version, and source path this running process
        was imported from - read fresh every time this is opened (never
        cached at startup), so it reflects the actual code currently
        executing, not whatever was true when the process began. Reuses
        analytics.environment's resolvers rather than re-implementing "how
        do I find the git commit" a second time (Manual QA Phase 13)."""
        from pathlib import Path

        from ..analytics.environment import (
            resolve_application_version,
            resolve_git_commit,
            resolve_git_commit_timestamp,
            resolve_python_version,
        )

        commit = resolve_git_commit()
        commit_timestamp = resolve_git_commit_timestamp()
        version = resolve_application_version()
        python_version = resolve_python_version()
        module_path = Path(__file__).resolve().parents[1]  # .../src/picklikeme
        QMessageBox.about(
            self,
            "About PeakPic",
            "PeakPic Desktop\nNative desktop shell powered by the existing backend.\n\n"
            f"Application version: {version or 'unknown'}\n"
            f"Git commit: {commit or 'unknown (not a git checkout, or git unavailable)'}\n"
            f"Build timestamp (commit date): {commit_timestamp or 'unknown'}\n"
            f"Python version: {python_version}\n"
            f"Running from: {module_path}",
        )

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

