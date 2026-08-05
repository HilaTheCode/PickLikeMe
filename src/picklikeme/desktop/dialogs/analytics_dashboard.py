"""The Diagnostics & Analytics Dashboard.

Phase 2's own mandate: the goal is no longer displaying statistics, it is
understanding the algorithm - "why did it decide this", "where does it
disagree with the photographer", "what should improve next". User vs
Algorithm (see UserVsAlgorithmTab) is the primary page for that reason -
agreement with the photographer, not the score itself, is the actual
measure of success - shown first, ahead of Run Summary, Species Analysis
and Burst Analytics. Implemented as one master-detail dialog - an
Experiment Browser (a list) on the left, and the detail views as tabs on
the right that redraw for whichever experiment is currently selected.

Deliberately reads only `AnalyticsStore`'s already-generic accessors
(`list_runs`, `get_run`, `category_counts`, `summary_metrics`,
`image_paths`, `image_metrics`) - nothing here assumes a species-
classification run specifically. A ranking run selected in the Experiment
Browser renders exactly as well (its "species distribution" table shows
reject reasons instead, its image metrics show eye/subject sharpness
instead of confidences) - the same category-counts/image-metrics tables
underlie both, see analytics/store.py's own docstring. This is what "avoid
hardcoded backend names" (Part 7) means applied to the dashboard itself,
not just to the classifier registry.

Product Direction (the "NEW PRODUCT DIRECTION" pivot): this dashboard no
longer contains its own per-image browser/investigation tool
(ImageExplorerTab - Original/Crop panels, Visual Debug overlays, Score
Explanation - was removed entirely). "The Analytics Dashboard should
remain responsible only for: statistics, analytics, trends, confusion
matrices, run summaries, species analytics, burst analytics. It should NOT
become another image browser." Per-image investigation now belongs to the
Review Window (Advanced Filters + the main grid) and the Loupe (per-image
debugging) - see AnalyticsDashboard's own class docstring for how
Advanced Filters and drill-downs still let a photographer narrow WHICH
images every table/chart here reflects, without ever showing the images
themselves.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...analyzer.annotations import DEFAULT_ANNOTATIONS_DB, AnnotationStore
from ...analytics.agreement import (
    AgreementReport,
    algorithm_decisions_for_run,
    compare_run_to_user_decisions,
    user_decisions_for_paths,
)
from ...analytics.reports import metric_statistics, run_statistics
from ...analytics.store import DEFAULT_ANALYTICS_DB, AnalyticsStore
from ...burst_analysis import BurstInfo, ScoredImage, analyze_bursts
from ...ranking.classic import read_filter_report
from ...ranking.filters import REJECT_REASON_LABELS
from ...species.classifier import UNKNOWN_SPECIES
from ..filtering import FilterableRecord, apply_filters
from ..widgets.advanced_filters_panel import AdvancedFiltersPanel

# run_id/folder/started_at (etc.) are not "algorithm identity" facts a
# photographer reads the Experiment Metadata panel to answer "what produced
# this" - these are, in the order the panel shows them when present. Not
# every run has every key (a species run has no "algorithm_version"; a
# ranking run has no "species_count") - see ExperimentMetadataPanel.
_METADATA_PARAM_FIELDS: tuple[tuple[str, str], ...] = (
    ("algorithm_version", "Algorithm Version"),
    ("classifier_version", "Classifier Version"),
    ("model_id", "Model"),
    ("model_version", "Model Version"),
    ("eye_detector", "EyePose Model"),
    ("detector", "Detector"),
    ("classifier_backend", "Species Backend"),
    ("species_list_filename", "Species List Filename"),
    ("species_list_hash", "Species List Hash"),
    ("species_count", "Species Count"),
    ("open_clip_version", "open_clip Version"),
    ("application_version", "Application Version"),
    ("git_commit", "Git Commit"),
    ("gpu_name", "GPU"),
    ("cuda_available", "CUDA Available"),
)


def _friendly_strategy_label(strategy_id: str) -> str:
    """"ai-model" -> "AI Model", "bioclip2" -> "BioCLIP 2 (recommended)" -
    whichever registry (ranking or species) recognises this id, falling
    back to the raw id verbatim for anything neither does (an older run
    from a backend since removed, or a typo) - never fabricated."""
    try:
        from ...ranking import score_labels
        labels = score_labels()
        if strategy_id in labels:
            return labels[strategy_id]
    except Exception:  # noqa: BLE001 - a label lookup must never break the dashboard
        pass
    try:
        from ...species.classifier import available_classifiers
        for info in available_classifiers():
            if info.classifier_id == strategy_id:
                return info.display_name
    except Exception:  # noqa: BLE001
        pass
    return strategy_id


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def _fill_two_column_table(table: QTableWidget, rows: list[tuple[str, str]]) -> None:
    table.setSortingEnabled(False)  # avoid re-sorting while (re)populating
    table.setRowCount(len(rows))
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(["Field", "Value"])
    for row_index, (field, value) in enumerate(rows):
        table.setItem(row_index, 0, QTableWidgetItem(field))
        table.setItem(row_index, 1, QTableWidgetItem(value))
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    table.setSortingEnabled(True)


def _style_table(table: QTableWidget) -> None:
    """The shared look/behaviour every table in this dashboard gets: click a
    header to sort, alternating row shading for readability on a long list,
    the last column stretches to fill remaining width rather than leaving a
    dead gap."""
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(True)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.verticalHeader().setVisible(False)


class SummaryCard(QFrame):
    """One glanceable number - "Accepted: 42" - not a table row. A row of
    these at the top of Run Summary is the "immediately know how this
    experiment went" the dashboard is meant to answer before any table.

    Optionally clickable (Phase 7 - "every KPI must be clickable"): pass
    `clickable=True` and connect to `clicked` - emitted on a left mouse
    press, with a pointing-hand cursor and a subtle hover highlight as the
    only visual difference from a non-clickable card, so a plain summary
    number (e.g. Run Summary's own cards) is not mistaken for a button.
    """

    clicked = Signal()

    def __init__(self, title: str, parent=None, *, clickable: bool = False) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("summaryCard")
        self._clickable = clickable
        border_rule = "#summaryCard { border: 1px solid palette(mid); border-radius: 6px; background-color: palette(base); }"
        if clickable:
            border_rule += (
                "\n#summaryCard:hover { border: 1px solid palette(highlight); background-color: palette(alternate-base); }"
            )
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip("Click to see these images in the Image Explorer")
        self.setStyleSheet(border_rule)
        self._value_label = QLabel("—", self)
        self._value_label.setStyleSheet("font-size: 20pt; font-weight: 600;")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title, self)
        title_label.setStyleSheet("color: palette(mid); font-size: 9pt;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        layout.addWidget(self._value_label)
        layout.addWidget(title_label)

    def set_value(self, value: str) -> None:
        self._value_label.setText(value)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class DashboardHeaderPanel(QWidget):
    """"The dashboard should immediately communicate what dataset is
    currently being analyzed" (Phase 3's own mandate) - a single, always-
    visible banner combining the LIVE Review context (Root Folder, Color
    Source, Keep Threshold - whatever MainWindow currently has set,
    independent of any experiment selection) with facts about whichever
    experiment is currently selected (Algorithm, Ground Truth Coverage,
    Number of Images). Distinct from ExperimentMetadataPanel below it,
    which is the full per-run detail table - this is the glanceable
    summary above it, always visible even before any run is selected.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root_folder: str | None = None
        self._scope_label_text = "Entire Analytics Database"

        self._context_label = QLabel(self)
        self._context_label.setWordWrap(True)
        self._context_label.setStyleSheet("color: palette(mid);")

        self._dataset_card = SummaryCard("Dataset", self)
        self._algorithm_card = SummaryCard("Algorithm", self)
        self._color_source_card = SummaryCard("Color Source", self)
        self._threshold_card = SummaryCard("Keep Threshold", self)
        self._coverage_card = SummaryCard("Ground Truth Coverage", self)
        self._image_count_card = SummaryCard("Number of Images", self)
        cards_row = QHBoxLayout()
        for card in (
            self._dataset_card, self._algorithm_card, self._color_source_card,
            self._threshold_card, self._coverage_card, self._image_count_card,
        ):
            cards_row.addWidget(card)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._context_label)
        layout.addLayout(cards_row)

    def set_live_context(
        self, *, root_folder: str | None, color_source: str | None, keep_percent: float | None, scope_label: str,
    ) -> None:
        """The part that never depends on which experiment (if any) is
        selected - called once at dialog construction and again whenever
        the Scope selector changes, so the header is accurate even before
        the photographer has clicked an experiment yet."""
        self._root_folder = root_folder
        self._scope_label_text = scope_label
        self._dataset_card.set_value(Path(root_folder).name if root_folder else "(none)")
        self._color_source_card.set_value(_strategy_label_or_review_status(color_source))
        self._threshold_card.set_value(f"{keep_percent:.0f}%" if keep_percent is not None else "n/a")
        self._update_context_label()
        # An experiment may no longer be selected after a Scope change -
        # the run-specific cards must not keep showing stale data.
        self._algorithm_card.set_value("—")
        self._coverage_card.set_value("—")
        self._image_count_card.set_value("—")

    def _update_context_label(self) -> None:
        root = self._root_folder or "(no folder open)"
        self._context_label.setText(f"Root Folder: {root}      |      Analytics Scope: {self._scope_label_text}")

    def show_run(
        self, analytics_store: AnalyticsStore, annotation_store: AnnotationStore, run_id: str,
    ) -> None:
        run = analytics_store.get_run(run_id) or {}
        self._algorithm_card.set_value(_friendly_strategy_label(run.get("strategy_id", "")))
        self._image_count_card.set_value(str(run.get("considered", 0)))

        from ...analytics.agreement import compare_run_to_user_decisions

        report = compare_run_to_user_decisions(analytics_store, annotation_store, run_id)
        covered = report.compared
        total = report.compared + report.neutral
        self._coverage_card.set_value(f"{100 * covered / total:.0f}%" if total else "n/a")


def _strategy_label_or_review_status(strategy_id: str | None) -> str:
    return "Review Status" if strategy_id is None else _friendly_strategy_label(strategy_id)


class ExperimentMetadataPanel(QWidget):
    """"The user should immediately know exactly how the experiment was
    produced" - shown above the tabs (not buried inside one of them) so it
    stays visible regardless of which tab is open. Curated, not exhaustive -
    RunSummaryTab's own generic table already dumps every recorded param;
    this shows only the well-known "what produced this" fields, in a fixed
    reading order, and only the ones this particular run actually has (see
    _METADATA_PARAM_FIELDS).

    Collapsible, defaulting to collapsed - manual QA on DashboardHeaderPanel
    found that Device/Images Processed/Experiment Date and the rest of this
    table's technical, rarely-changing metadata was eating vertical space
    the analysis tabs need more (Run Summary, User vs Algorithm - the
    actual "understand the algorithm" content this dashboard exists for).
    The header panel above already surfaces the frequently-useful subset
    (Algorithm, Number of Images) as prominent cards; this stays available
    one click away for the rest, rather than disappearing into a separate
    dialog - still "above the tabs", just not competing with them for
    space by default.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._toggle_button = QPushButton("▸ Experiment Details", self)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(False)
        self._toggle_button.setFlat(True)
        self._toggle_button.setStyleSheet("text-align: left; border: none; padding: 2px 0;")
        self._toggle_button.toggled.connect(self._on_toggled)

        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(150)
        self._table.setVisible(False)
        _style_table(self._table)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._table)

    def _on_toggled(self, expanded: bool) -> None:
        self._table.setVisible(expanded)
        self._toggle_button.setText(("▾" if expanded else "▸") + " Experiment Details")

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        run = store.get_run(run_id) or {}
        params = dict(run.get("params") or {})
        summary = store.summary_metrics(run_id)

        rows: list[tuple[str, str]] = [
            ("Algorithm", _friendly_strategy_label(run.get("strategy_id", ""))),
            ("Experiment Date", run.get("started_at", "")),
            ("Images Processed", str(run.get("considered", 0))),
            ("Device", run.get("device") or "n/a"),
        ]
        if "runtime_seconds" in summary:
            rows.append(("Experiment Duration", _format_duration(summary["runtime_seconds"])))
        for key, label in _METADATA_PARAM_FIELDS:
            if key in params and params[key] not in (None, ""):
                rows.append((label, str(params[key])))

        _fill_two_column_table(self._table, rows)


# Friendly labels for well-known per-image metric names in the Metric
# Statistics table - any OTHER metric name a future strategy records still
# gets a row, just titled with its raw name (see RunSummaryTab.show_run).
_METRIC_LABELS: dict[str, str] = {
    "score": "Score",
    "eye_confidence": "Eye Confidence",
    "head_confidence": "Head Confidence",
    "eye_sharpness": "Eye Sharpness",
    "subject_sharpness": "Subject Sharpness",
    "subject_size": "Subject Size",
    "top1_confidence": "Top-1 Confidence",
    "top2_confidence": "Top-2 Confidence",
}


def _metric_statistics_for_paths(
    store: AnalyticsStore, run_id: str, metric_name: str, paths: list[str]
) -> dict | None:
    """The same mean/median/min/max/count shape as
    `analytics.reports.metric_statistics`, computed only over `paths` -
    Advanced Filters' equivalent for RunSummaryTab, which otherwise reads
    that whole-run SQL aggregate directly. Built from `store.image_metrics`
    per path (already used for exactly this purpose elsewhere in this
    module, e.g. BurstAnalyticsTab/ImageExplorerTab) rather than a new
    path-scoped query on AnalyticsStore itself."""
    values = [v for path in paths if (v := store.image_metrics(run_id, path).get(metric_name)) is not None]
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "count": len(values),
    }


class UserVsAlgorithmTab(QWidget):
    """The primary dashboard page (see this module's own Phase 2 mandate):
    not "how good is the score", but "how well does the algorithm agree
    with the photographer" - a full confusion matrix, precision/recall/F1,
    and the two things a live review session can never show because it
    only ever exists in the present: comparing a PAST run's own decisions
    against whatever the photographer has since decided, and doing that
    for a threshold the photographer can move right here rather than
    reopening the folder in Review.

    Drilling down (`drillDownRequested`) hands the AnalyticsDashboard the
    exact list of image paths behind whichever confusion-matrix cell OR
    summary KPI card was clicked, so it can filter the Image Explorer to
    just that category - "false positives", not "every image this run
    touched". Phase 7's own requirement ("every KPI must be clickable"): all
    six cards above the matrix, not only the four matrix cells.
    """

    drillDownRequested = Signal(list, str)  # (image_paths, category label)

    _CELL_LABELS = {
        "algo_keep_user_keep": "Algorithm Keep / User Keep",
        "algo_keep_user_reject": "Algorithm Keep / User Reject  (False Positive)",
        "algo_reject_user_keep": "Algorithm Reject / User Keep  (False Negative)",
        "algo_reject_user_reject": "Algorithm Reject / User Reject",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._analytics_store: AnalyticsStore | None = None
        self._annotation_store: AnnotationStore | None = None
        self._run_id: str | None = None
        self._report: AgreementReport | None = None
        self._paths: list[str] | None = None

        self._threshold_spin = QDoubleSpinBox(self)
        self._threshold_spin.setRange(0.0, 100.0)
        self._threshold_spin.setSuffix("%")
        self._threshold_spin.setDecimals(1)
        self._threshold_spin.setToolTip(
            "What fraction of scored images counts as Algorithm Keep - defaults to this "
            "run's own accepted/considered ratio, the same threshold concept as Review's own "
            "keep-percent. Move it to see how agreement changes at a different cutoff."
        )
        self._threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self._reset_threshold_btn = QPushButton("Reset to run default", self)
        self._reset_threshold_btn.clicked.connect(self._reset_threshold)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Keep top:", self))
        threshold_row.addWidget(self._threshold_spin)
        threshold_row.addWidget(self._reset_threshold_btn)
        threshold_row.addStretch(1)

        self._user_keep_card = SummaryCard("User Keep", self, clickable=True)
        self._user_reject_card = SummaryCard("User Reject", self, clickable=True)
        self._algo_keep_card = SummaryCard("Algorithm Keep", self, clickable=True)
        self._algo_reject_card = SummaryCard("Algorithm Reject", self, clickable=True)
        self._agree_card = SummaryCard("Agreement %", self, clickable=True)
        self._disagree_card = SummaryCard("Disagreement %", self, clickable=True)
        self._user_keep_card.clicked.connect(lambda: self._drill_down_from_kpi("user_keep"))
        self._user_reject_card.clicked.connect(lambda: self._drill_down_from_kpi("user_reject"))
        self._algo_keep_card.clicked.connect(lambda: self._drill_down_from_kpi("algo_keep"))
        self._algo_reject_card.clicked.connect(lambda: self._drill_down_from_kpi("algo_reject"))
        self._agree_card.clicked.connect(lambda: self._drill_down_from_kpi("agree"))
        self._disagree_card.clicked.connect(lambda: self._drill_down_from_kpi("disagree"))
        cards_row = QHBoxLayout()
        for card in (
            self._user_keep_card, self._user_reject_card, self._algo_keep_card,
            self._algo_reject_card, self._agree_card, self._disagree_card,
        ):
            cards_row.addWidget(card)

        self._matrix_table = QTableWidget(self)
        self._matrix_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._matrix_table.setRowCount(2)
        self._matrix_table.setColumnCount(2)
        self._matrix_table.setHorizontalHeaderLabels(["Algorithm Keep", "Algorithm Reject"])
        self._matrix_table.setVerticalHeaderLabels(["User Keep", "User Reject"])
        self._matrix_table.cellClicked.connect(self._on_matrix_cell_clicked)
        self._matrix_table.setToolTip("Click a cell to see those images in the Image Explorer")

        self._metrics_table = QTableWidget(self)
        self._metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._metrics_table)

        self._coverage_label = QLabel(self)
        self._coverage_label.setWordWrap(True)
        self._coverage_label.setStyleSheet("color: palette(mid);")

        layout = QVBoxLayout(self)
        layout.addLayout(threshold_row)
        layout.addLayout(cards_row)
        layout.addWidget(QLabel("Confusion Matrix (click a cell to drill down)", self))
        layout.addWidget(self._matrix_table)
        layout.addWidget(QLabel("Precision / Recall / Overrides", self))
        layout.addWidget(self._metrics_table)
        layout.addWidget(self._coverage_label)

    def show_run(
        self, analytics_store: AnalyticsStore, annotation_store: AnnotationStore, run_id: str,
        *, paths: list[str] | None = None,
    ) -> None:
        """`paths=None` (the default) reports on every image this run
        scored. Otherwise (Advanced Filters active) every card, the
        confusion matrix, and precision/recall/F1 are scoped to exactly
        `paths` - see `compare_run_to_user_decisions`'s own docstring for
        why the Keep/Reject cut itself never changes, only what is
        reported."""
        self._analytics_store = analytics_store
        self._annotation_store = annotation_store
        self._run_id = run_id
        self._paths = paths

        run = analytics_store.get_run(run_id) or {}
        considered = run.get("considered") or 0
        default_percent = 100.0 * (run.get("accepted") or 0) / considered if considered else 0.0
        self._default_threshold = default_percent
        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(default_percent)
        self._threshold_spin.blockSignals(False)

        self._recompute(keep_percent=default_percent)

    def _reset_threshold(self) -> None:
        self._threshold_spin.setValue(self._default_threshold)  # triggers _on_threshold_changed

    def _on_threshold_changed(self, value: float) -> None:
        self._recompute(keep_percent=value)

    def _recompute(self, *, keep_percent: float) -> None:
        if self._analytics_store is None or self._annotation_store is None or self._run_id is None:
            return
        report = compare_run_to_user_decisions(
            self._analytics_store, self._annotation_store, self._run_id,
            keep_percent=keep_percent, paths=self._paths,
        )
        self._report = report

        self._user_keep_card.set_value(str(report.user_keep))
        self._user_reject_card.set_value(str(report.user_reject))
        self._algo_keep_card.set_value(str(report.algorithm_keep))
        self._algo_reject_card.set_value(str(report.algorithm_reject))
        d = report.to_dict()
        self._agree_card.set_value(f"{d['agree_percent']:.1f}%" if d["agree_percent"] is not None else "n/a")
        self._disagree_card.set_value(f"{d['disagree_percent']:.1f}%" if d["disagree_percent"] is not None else "n/a")

        self._matrix_table.setItem(0, 0, QTableWidgetItem(str(report.algo_keep_user_keep)))
        self._matrix_table.setItem(0, 1, QTableWidgetItem(str(report.algo_reject_user_keep)))
        self._matrix_table.setItem(1, 0, QTableWidgetItem(str(report.algo_keep_user_reject)))
        self._matrix_table.setItem(1, 1, QTableWidgetItem(str(report.algo_reject_user_reject)))
        self._matrix_table.resizeColumnsToContents()

        def _fmt(value: float | None, *, percent: bool = False) -> str:
            if value is None:
                return "n/a"
            return f"{value * 100:.1f}%" if percent else f"{value:.4f}"

        rows = [
            ("Precision", _fmt(report.precision)),
            ("Recall", _fmt(report.recall)),
            ("F1", _fmt(report.f1)),
            ("False Positives (Algorithm Keep, User Reject)", str(report.algo_keep_user_reject)),
            ("False Negatives (Algorithm Reject, User Keep)", str(report.algo_reject_user_keep)),
            ("Override Rate", f"{report.override_rate:.1f}%" if report.override_rate is not None else "n/a"),
            ("Avg score - User Keep", _fmt(report.mean_score_user_kept)),
            ("Avg score - User Reject", _fmt(report.mean_score_user_rejected)),
        ]
        _fill_two_column_table(self._metrics_table, rows)

        self._coverage_label.setText(
            f"{report.compared} image(s) compared  |  {report.neutral} not yet reviewed (Neutral)  |  "
            f"{report.unmatched} could not be matched to a current file (moved or deleted since this run)"
        )

    def _on_matrix_cell_clicked(self, row: int, column: int) -> None:
        if self._report is None:
            return
        key = {(0, 0): "algo_keep_user_keep", (0, 1): "algo_reject_user_keep",
               (1, 0): "algo_keep_user_reject", (1, 1): "algo_reject_user_reject"}[(row, column)]
        paths = [path for path, user_status, algo_status in self._report.pairs
                 if f"algo_{algo_status}_user_{user_status}" == key]
        self.drillDownRequested.emit(paths, self._CELL_LABELS[key])

    _KPI_LABELS = {
        "user_keep": "User Keep", "user_reject": "User Reject",
        "algo_keep": "Algorithm Keep", "algo_reject": "Algorithm Reject",
        "agree": "Agreement", "disagree": "Disagreement",
    }

    def _drill_down_from_kpi(self, kpi: str) -> None:
        """Phase 7 - every KPI card is clickable, not only the confusion-
        matrix cells: filters to exactly the images behind whichever number
        was clicked, using the same `report.pairs` the matrix cells already
        drill down from, so a card's count and its drill-down result can
        never disagree with each other."""
        if self._report is None:
            return
        predicates = {
            "user_keep": lambda user, algo: user == "keep",
            "user_reject": lambda user, algo: user == "reject",
            "algo_keep": lambda user, algo: algo == "keep",
            "algo_reject": lambda user, algo: algo == "reject",
            "agree": lambda user, algo: user == algo,
            "disagree": lambda user, algo: user != algo,
        }
        predicate = predicates[kpi]
        paths = [path for path, user_status, algo_status in self._report.pairs if predicate(user_status, algo_status)]
        self.drillDownRequested.emit(paths, self._KPI_LABELS[kpi])


class RunSummaryTab(QWidget):
    """Everything "how did this experiment go, in numbers" for one selected
    experiment: glanceable summary cards, a curated Score & Quality summary
    (Phase 8's own named list), per-metric statistics (mean/median/min/max,
    generic - for whatever ELSE this run recorded), and every recorded
    field/param in full.

    Known limitation: "Average Runtime" is the run's own total
    `runtime_seconds` divided by images considered - a per-image average,
    since AnalyticsStore only ever records one total runtime per run, never
    a per-image timing.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._images_processed_card = SummaryCard("Images Processed", self)
        self._accepted_card = SummaryCard("Accepted", self)
        self._rejected_card = SummaryCard("Rejected", self)
        self._acceptance_card = SummaryCard("Acceptance %", self)
        self._score_card = SummaryCard("Avg Score", self)
        cards_row = QHBoxLayout()
        for card in (
            self._images_processed_card, self._accepted_card, self._rejected_card,
            self._acceptance_card, self._score_card,
        ):
            cards_row.addWidget(card)

        self._summary_stats_table = QTableWidget(self)
        self._summary_stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._summary_stats_table)

        self._metric_stats_table = QTableWidget(self)
        self._metric_stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._metric_stats_table)

        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._table)

        layout = QVBoxLayout(self)
        layout.addLayout(cards_row)
        layout.addWidget(QLabel("Score & Quality Summary", self))
        layout.addWidget(self._summary_stats_table)
        layout.addWidget(QLabel("Metric Statistics", self))
        layout.addWidget(self._metric_stats_table)
        layout.addWidget(QLabel("Full Run Detail", self))
        layout.addWidget(self._table)

    def show_run(self, store: AnalyticsStore, run_id: str, *, records: list[FilterableRecord] | None = None) -> None:
        """`records=None` (the default) reports on the whole run, exactly
        as before Advanced Filters existed - the whole-run SQL aggregates
        (`run_statistics`, `metric_statistics`) this tab otherwise reads
        directly. Otherwise every card/table is recomputed from `records`
        (already narrowed to the active filters/drill-down) instead:
        Considered is simply `len(records)`; Accepted/Rejected come from
        each record's own `algorithm_decision` (already computed once, in
        `_build_run_records`, from the run's FULL ranking - see
        `algorithm_decisions_for_run`'s own docstring for why that must
        never be re-derived from just the filtered subset). Per-metric
        statistics still read `store.image_metrics` per path
        (`_metric_statistics_for_paths`) rather than each record's own
        fixed field set, since a run can record metrics (e.g.
        top1_confidence, inference_seconds) FilterableRecord has no field
        for at all.

        Known limitation: Accepted/Rejected has no meaning for a filtered
        SPECIES-classification run (`algorithm_decision` is a ranking-run
        concept only - species runs report "accepted" as "successfully
        classified, not Unknown" instead, a fact `run_statistics` already
        captures correctly for the UNFILTERED case but that this tab has no
        per-image equivalent for once filtered) - it reads 0/0 rather than
        a wrong number in that case, never a fabricated guess.
        """
        run = store.get_run(run_id) or {}
        summary = store.summary_metrics(run_id)
        # Which metrics EXIST at all is a run-wide fact (what this
        # strategy records), independent of which images are filtered in.
        metric_names = store.metric_names(run_id)

        if records is None:
            stats = run_statistics(store, run_id)
            considered = stats.get("considered", 0)
            accepted = stats.get("accepted", 0)
            rejected = stats.get("rejected", 0)
            stat_fn = lambda name: metric_statistics(store, run_id, name)  # noqa: E731
            folder, strategy_id = stats.get("folder", ""), stats.get("strategy_id", "")
            started_at, device = stats.get("started_at", ""), stats.get("device") or "n/a"
            params = stats.get("params", {})
        else:
            considered = len(records)
            accepted = sum(1 for r in records if r.algorithm_decision == "keep")
            rejected = considered - accepted
            paths = [r.path for r in records]
            stat_fn = lambda name: _metric_statistics_for_paths(store, run_id, name, paths)  # noqa: E731
            folder, strategy_id = run.get("folder", ""), run.get("strategy_id", "")
            started_at, device = run.get("started_at", ""), run.get("device") or "n/a"
            params = dict(run.get("params") or {})

        all_metric_stats = {name: stat_fn(name) for name in metric_names}

        self._images_processed_card.set_value(str(considered))
        self._accepted_card.set_value(str(accepted))
        self._rejected_card.set_value(str(rejected))
        self._acceptance_card.set_value(f"{100.0 * accepted / considered:.1f}%" if considered else "n/a")
        score_stats = all_metric_stats.get("score") or stat_fn("score") or {}
        self._score_card.set_value(f"{score_stats['mean']:.4f}" if score_stats else "n/a")

        def _fmt(value: float | None) -> str:
            return f"{value:.4f}" if value is not None else "n/a"

        eye_confidence_stats = all_metric_stats.get("eye_confidence") or {}
        head_confidence_stats = all_metric_stats.get("head_confidence") or {}
        eye_sharpness_stats = all_metric_stats.get("eye_sharpness") or {}
        subject_sharpness_stats = all_metric_stats.get("subject_sharpness") or {}
        subject_size_stats = all_metric_stats.get("subject_size") or {}
        runtime_seconds = summary.get("runtime_seconds")
        average_runtime = runtime_seconds / considered if runtime_seconds is not None and considered else None

        _fill_two_column_table(self._summary_stats_table, [
            ("Median Score", _fmt(score_stats.get("median"))),
            ("Highest Score", _fmt(score_stats.get("max"))),
            ("Lowest Score", _fmt(score_stats.get("min"))),
            ("Images / Second", _fmt(summary.get("images_per_second"))),
            ("Average Runtime (seconds/image)", _fmt(average_runtime)),
            ("Average Eye Confidence", _fmt(eye_confidence_stats.get("mean"))),
            ("Average Head Confidence", _fmt(head_confidence_stats.get("mean"))),
            ("Average Eye Sharpness", _fmt(eye_sharpness_stats.get("mean"))),
            ("Average Subject Sharpness", _fmt(subject_sharpness_stats.get("mean"))),
            ("Average Subject Size", _fmt(subject_size_stats.get("mean"))),
        ])

        self._metric_stats_table.setSortingEnabled(False)
        self._metric_stats_table.setRowCount(len(metric_names))
        self._metric_stats_table.setColumnCount(5)
        self._metric_stats_table.setHorizontalHeaderLabels(["Metric", "Mean", "Median", "Min", "Max"])
        for row_index, name in enumerate(metric_names):
            metric_stats = all_metric_stats.get(name) or {}
            self._metric_stats_table.setItem(row_index, 0, QTableWidgetItem(_METRIC_LABELS.get(name, name)))
            for col_index, key in enumerate(("mean", "median", "min", "max"), start=1):
                value = metric_stats.get(key)
                self._metric_stats_table.setItem(
                    row_index, col_index, QTableWidgetItem(f"{value:.4f}" if value is not None else "n/a")
                )
        self._metric_stats_table.resizeColumnsToContents()
        self._metric_stats_table.horizontalHeader().setStretchLastSection(True)
        self._metric_stats_table.setSortingEnabled(True)

        rows: list[tuple[str, str]] = [
            ("Experiment ID", run_id),
            ("Folder", folder),
            ("Backend / Strategy", strategy_id),
            ("Started at", started_at),
            ("Device", device),
            ("Considered", str(considered)),
            ("Accepted", str(accepted)),
            ("Rejected", str(rejected)),
        ]
        for name, value in sorted(summary.items()):
            rows.append((f"Runtime: {name}", f"{value:.4f}" if isinstance(value, float) else str(value)))
        for name in metric_names:
            metric_stats = all_metric_stats.get(name)
            if metric_stats:
                rows.append((f"Mean {name}", f"{metric_stats['mean']:.4f}"))
        # The full experiment record (model id/version, species list hash,
        # GPU, thresholds, ...) lives in params for a species run - see
        # species.experiment.ExperimentMetadata.to_dict(). Shown generically
        # so a ranking run's differently-shaped params render just as well.
        for key, value in sorted(params.items()):
            if isinstance(value, dict):
                for sub_key, sub_value in sorted(value.items()):
                    rows.append((f"{key}.{sub_key}", str(sub_value)))
            else:
                rows.append((key, str(value)))

        _fill_two_column_table(self._table, rows)


class SpeciesAnalysisTab(QWidget):
    """The category-count breakdown for one experiment - a species
    distribution (including "Unknown") for a species run, or reject
    reasons for a ranking run - same underlying table, see this module's
    own docstring. Phase 9 adds Average Confidence (mean top1_confidence,
    when this run recorded one), a Top 5 Predictions highlight, and makes
    every species/category row clickable - filters the Image Explorer to
    that species via `speciesDrillDownRequested`.

    Known limitation: "Top 5 Predictions" means the five most COMMON
    predicted species in this run (the distribution table's own top rows) -
    AnalyticsStore never records a per-image top-5 candidate LIST, only
    top1_confidence (and top2..5_confidence as bare numbers, with no species
    name attached) - see species.experiment_capture's own docstring.
    """

    speciesDrillDownRequested = Signal(str)  # species/category name

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel(self)

        self._confidence_card = SummaryCard("Average Confidence", self)
        self._unknown_rate_card = SummaryCard("Unknown Rate", self)
        cards_row = QHBoxLayout()
        cards_row.addWidget(self._confidence_card)
        cards_row.addWidget(self._unknown_rate_card)

        self._top5_table = QTableWidget(self)
        self._top5_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._top5_table.setMaximumHeight(160)
        _style_table(self._top5_table)

        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setToolTip("Click a species/category to see those images in the Image Explorer")
        self._table.cellClicked.connect(self._on_row_clicked)
        _style_table(self._table)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addLayout(cards_row)
        layout.addWidget(QLabel("Top 5 Predictions", self))
        layout.addWidget(self._top5_table)
        layout.addWidget(QLabel("Full Distribution (click a row to filter the Image Explorer)", self))
        layout.addWidget(self._table)

    def show_run(self, store: AnalyticsStore, run_id: str, *, records: list[FilterableRecord] | None = None) -> None:
        """`records=None` (the default) reports on the whole run, from the
        same whole-run `category_counts` SQL aggregate as before Advanced
        Filters existed. Otherwise (Advanced Filters active) the
        distribution is recomputed from each filtered record's own
        `species`/`reject_reason` field instead - `category_counts` is a
        run-wide total with no per-image breakdown to filter (see this
        module's own docstring), unlike `FilterableRecord`, which already
        carries one classification per image."""
        run = store.get_run(run_id) or {}

        if records is None:
            considered = run.get("considered", 0) or 0
            accepted = run.get("accepted", 0) or 0
            counts = store.category_counts(run_id)
            unknown_rate = store.summary_metrics(run_id).get("unknown_rate")
            confidence_stats = metric_statistics(store, run_id, "top1_confidence")
            confidence_mean = confidence_stats["mean"] if confidence_stats else None
        else:
            considered = len(records)
            accepted = sum(1 for r in records if r.algorithm_decision == "keep")
            counts = {}
            for r in records:
                category = r.species
                if category is None and r.reject_reason:
                    category = REJECT_REASON_LABELS.get(r.reject_reason, r.reject_reason)
                if category:
                    counts[category] = counts.get(category, 0) + 1
            unknown = sum(1 for r in records if r.species == UNKNOWN_SPECIES)
            unknown_rate = unknown / considered if considered else None
            confidences = [r.species_confidence for r in records if r.species_confidence is not None]
            confidence_mean = statistics.fmean(confidences) if confidences else None

        label_text = f"{len(counts)} distinct outcome(s) across {considered} image(s)"
        self._label.setText(label_text)
        self._unknown_rate_card.set_value(f"{unknown_rate:.1%}" if unknown_rate is not None else "n/a")
        self._confidence_card.set_value(f"{confidence_mean:.4f}" if confidence_mean is not None else "n/a")

        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        _fill_ranked_table(self._top5_table, ranked[:5], considered)

        self._table.setSortingEnabled(False)
        # "Accepted" is not itself a row in category_counts (it is the
        # complement of every reject/category count, tracked separately on
        # the run itself) - shown first so the table reads as a complete
        # outcome breakdown, not only the rejected/categorized slice. Not
        # clickable (see _on_row_clicked) - it is not a species/category
        # name the Review Window's Species filter could ever match.
        rows = [("Accepted", accepted)] + ranked
        _fill_ranked_table(self._table, rows, considered)

    def _on_row_clicked(self, row: int, _column: int) -> None:
        if row == 0:  # "Accepted" - not a real species/category, see show_run
            return
        item = self._table.item(row, 0)
        if item is not None and item.text():
            self.speciesDrillDownRequested.emit(item.text())


def _fill_ranked_table(table: QTableWidget, rows: list[tuple[str, int]], considered: int) -> None:
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(["Species / Category", "Count", "% of considered"])
    for row_index, (name, count) in enumerate(rows):
        percent = f"{100.0 * count / considered:.1f}%" if considered else "n/a"
        table.setItem(row_index, 0, QTableWidgetItem(name))
        item_count = QTableWidgetItem()
        item_count.setData(Qt.ItemDataRole.DisplayRole, count)
        table.setItem(row_index, 1, item_count)
        table.setItem(row_index, 2, QTableWidgetItem(percent))
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    table.setSortingEnabled(True)


def _compute_burst_map(
    store: AnalyticsStore, annotation_store: AnnotationStore, run_id: str, paths: list[str]
) -> dict[str, BurstInfo]:
    """Burst membership/rank/winner for `paths`, recomputed locally from
    each image's own EXIF capture time and this run's own recorded score -
    best-effort (never crashes the dashboard) since neither input is
    guaranteed: a path may have no readable EXIF, and burst_analysis itself
    is never persisted anywhere (see BurstAnalyticsTab's own Known
    Limitations). Shared by ImageExplorerTab (its Burst filter) and
    BurstAnalyticsTab so the two can never disagree about one image's rank.
    """
    try:
        scored_images = [
            ScoredImage(
                path=path,
                captured_at=annotation_store.capture_timestamp_of(path),
                score=store.image_metrics(run_id, path).get("score"),
            )
            for path in paths
        ]
        return analyze_bursts(scored_images)
    except Exception:  # noqa: BLE001 - burst rank is best-effort
        return {}


class BurstAnalyticsTab(QWidget):
    """Phase 10 - burst-level analytics for one experiment: how many bursts,
    how big, and which image won each one. Reuses `_compute_burst_map` (the
    same recomputation `ImageExplorerTab`'s own Burst filter already uses)
    against every image this run scored, so the two never disagree.

    Known limitations:
    - Burst membership/rank is recomputed here, never read from a persisted
      table - burst_analysis.py's own docstring is explicit that burst
      ranking is deliberately never persisted (it is a function of
      whichever score the CALLER passes in, not a fixed fact about an
      image). Re-opening this tab after a Ground Truth import or a
      threshold change elsewhere therefore reflects this run's own score
      every time, not a frozen snapshot.
    - An image with no readable EXIF capture time becomes its own
      singleton burst (burst_size=1) - indistinguishable here from a photo
      that was genuinely alone, see burst_analysis.py's own docstring.
    """

    def __init__(self, *, annotation_store: AnnotationStore, parent=None) -> None:
        super().__init__(parent)
        self._annotation_store = annotation_store
        self._store: AnalyticsStore | None = None
        self._run_id: str | None = None

        self._burst_count_card = SummaryCard("Bursts", self)
        self._average_size_card = SummaryCard("Average Burst Size", self)
        self._singleton_card = SummaryCard("Singleton Images", self)
        self._multi_image_card = SummaryCard("Multi-Image Bursts", self)
        cards_row = QHBoxLayout()
        for card in (
            self._burst_count_card, self._average_size_card, self._singleton_card, self._multi_image_card,
        ):
            cards_row.addWidget(card)

        self._size_distribution_table = QTableWidget(self)
        self._size_distribution_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._size_distribution_table.setMaximumHeight(160)
        _style_table(self._size_distribution_table)

        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._table)

        layout = QVBoxLayout(self)
        layout.addLayout(cards_row)
        layout.addWidget(QLabel("Burst Size Distribution", self))
        layout.addWidget(self._size_distribution_table)
        layout.addWidget(QLabel("Every Image", self))
        layout.addWidget(self._table)

    def show_run(self, store: AnalyticsStore, run_id: str, *, paths: list[str] | None = None) -> None:
        """`paths=None` (the default) reports on every image this run
        scored. Burst membership/rank/winner is always computed from the
        run's FULL image set (never just `paths`) - narrowing the INPUT
        would silently redefine "which image won this burst" whenever a
        filter happens to exclude the actual winner, the same principle
        `algorithm_decisions_for_run`'s own docstring explains for Algorithm
        Decision. `paths`, when given, only narrows which of the already-
        computed bursts/images are actually reported."""
        self._store = store
        self._run_id = run_id
        all_paths = store.image_paths(run_id)
        burst_by_path = _compute_burst_map(store, self._annotation_store, run_id, all_paths)
        display_paths = list(paths) if paths is not None else all_paths
        allowed = set(display_paths)
        filtered_burst_by_path = {p: info for p, info in burst_by_path.items() if p in allowed}

        burst_ids = {info.burst_id for info in filtered_burst_by_path.values()}
        # One BurstInfo per distinct burst actually represented in the
        # filtered set - any member works, since burst_size is a fixed fact
        # about the whole burst, not just its (possibly filtered-out) winner.
        distinct_bursts = {info.burst_id: info for info in filtered_burst_by_path.values()}.values()
        sizes = [info.burst_size for info in distinct_bursts]
        singleton_images = sum(1 for info in filtered_burst_by_path.values() if info.burst_size == 1)
        multi_image_bursts = sum(1 for info in distinct_bursts if info.burst_size > 1)

        self._burst_count_card.set_value(str(len(burst_ids)))
        self._average_size_card.set_value(f"{sum(sizes) / len(sizes):.2f}" if sizes else "n/a")
        self._singleton_card.set_value(str(singleton_images))
        self._multi_image_card.set_value(str(multi_image_bursts))

        size_counts: dict[int, int] = {}
        for size in sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        self._size_distribution_table.setSortingEnabled(False)
        self._size_distribution_table.setRowCount(len(size_counts))
        self._size_distribution_table.setColumnCount(2)
        self._size_distribution_table.setHorizontalHeaderLabels(["Burst Size", "Number of Bursts"])
        for row_index, size in enumerate(sorted(size_counts)):
            self._size_distribution_table.setItem(row_index, 0, QTableWidgetItem(str(size)))
            count_item = QTableWidgetItem()
            count_item.setData(Qt.ItemDataRole.DisplayRole, size_counts[size])
            self._size_distribution_table.setItem(row_index, 1, count_item)
        self._size_distribution_table.resizeColumnsToContents()
        self._size_distribution_table.horizontalHeader().setStretchLastSection(True)
        self._size_distribution_table.setSortingEnabled(True)

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(display_paths))
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Image", "Burst Size", "Burst Rank", "Burst Winner"])
        for row_index, path in enumerate(display_paths):
            info = filtered_burst_by_path.get(path)
            self._table.setItem(row_index, 0, QTableWidgetItem(Path(path).name))
            self._table.setItem(row_index, 1, QTableWidgetItem(str(info.burst_size) if info else "n/a"))
            self._table.setItem(row_index, 2, QTableWidgetItem(str(info.burst_rank) if info else "n/a"))
            self._table.setItem(row_index, 3, QTableWidgetItem("Yes" if info and info.burst_best else "No"))
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSortingEnabled(True)


_GEOMETRY_SETTINGS_KEY = "analytics/dashboard_geometry"


def _build_run_records(
    store: AnalyticsStore, annotation_store: AnnotationStore, run_id: str,
    *, species_db: str | Path | None = None,
) -> list[FilterableRecord]:
    """Every image this run touched, adapted into the shared filtering
    engine's generic shape (see desktop/filtering.py) - the Analytics
    Dashboard's own equivalent of MainWindow._build_filterable_records.
    Product Direction Phase B: "use the same engine everywhere" - this is
    the ONE place a run's images become FilterableRecords, reused by
    Advanced Filters, the Species-row drill-down, and (indirectly, via
    _apply_filters_to_tabs) every detail tab.

    Replaces the per-path lookups the now-removed ImageExplorerTab's own
    `_populate_candidates` used to build purely for its own filter bar
    (algorithm/user decision, species, burst, reject reason) - same
    best-effort sources, same Known Limitations (species/burst are
    recomputed, never a persisted per-image fact; reject reasons depend on
    `.picklikeme/classic_vision_filters.json` still existing next to the
    run's own folder).
    """
    run = store.get_run(run_id) or {}
    folder = run.get("folder")
    strategy_id = run.get("strategy_id", "")

    scored_paths = store.image_paths(run_id)
    reject_reason_by_path: dict[str, str | None] = {path: None for path in scored_paths}
    filtered_paths: list[str] = []
    if folder:
        try:
            report = read_filter_report(folder, strategy_id)
            for path, reason in (report.get("images") or {}).items():
                if path not in reject_reason_by_path:
                    filtered_paths.append(path)
                reject_reason_by_path[path] = reason
        except Exception:  # noqa: BLE001 - best-effort; see this function's own docstring
            pass
    all_paths = list(scored_paths) + filtered_paths

    try:
        algo_decision_by_path = algorithm_decisions_for_run(store, run_id)
    except Exception:  # noqa: BLE001 - filtering must never crash the dashboard
        algo_decision_by_path = {}
    try:
        user_decision_by_path = user_decisions_for_paths(annotation_store, all_paths)
    except Exception:  # noqa: BLE001
        user_decision_by_path = {}

    species_by_path: dict[str, object] = {}
    try:
        from ...species.cache import DEFAULT_SPECIES_DB, SpeciesCache

        cache = SpeciesCache(species_db if species_db is not None else DEFAULT_SPECIES_DB)
        try:
            for path in all_paths:
                prediction = cache.get(path, strategy_id)
                if prediction is not None:
                    species_by_path[path] = prediction
        finally:
            cache.close()
    except Exception:  # noqa: BLE001 - species prediction is best-effort
        pass

    burst_by_path = _compute_burst_map(store, annotation_store, run_id, all_paths)

    records: list[FilterableRecord] = []
    for path in all_paths:
        metrics = store.image_metrics(run_id, path)
        prediction = species_by_path.get(path)
        burst = burst_by_path.get(path)
        records.append(FilterableRecord(
            path=path,
            folder=str(Path(path).parent),
            filename=Path(path).name,
            user_decision=user_decision_by_path.get(path, "neutral"),
            algorithm_decision=algo_decision_by_path.get(path),
            reject_reason=reject_reason_by_path.get(path),
            species=prediction.species if prediction is not None else None,
            species_confidence=prediction.confidence if prediction is not None else None,
            score=metrics.get("score"),
            eye_confidence=metrics.get("eye_confidence"),
            head_confidence=metrics.get("head_confidence"),
            subject_size=metrics.get("subject_size"),
            eye_sharpness=metrics.get("eye_sharpness"),
            subject_sharpness=metrics.get("subject_sharpness"),
            burst_id=burst.burst_id if burst else None,
            burst_size=burst.burst_size if burst else 1,
            burst_rank=burst.burst_rank if burst else 1,
            burst_best=burst.burst_best if burst else True,
        ))
    return records


class AnalyticsDashboard(QDialog):
    """The Experiment Browser (left) plus an Experiment Metadata header,
    Advanced Filters, and User vs Algorithm / Run Summary / Species
    Analysis / Burst Analytics (right, as tabs) - the priority order
    specified for this phase.

    Product Direction (the "NEW PRODUCT DIRECTION" pivot): "the Dashboard
    should no longer contain Image Explorer functionality... it should
    support the same filtering engine [as the Review Window]. Changing
    filters should immediately update: KPIs, Statistics, Tables, Charts,
    Confusion Matrix, Species Analytics, Burst Analytics, Run Summary."
    ImageExplorerTab (its own per-image browser, Original/Crop panels,
    Visual Debug overlays, and Score Explanation table) was removed
    entirely - browsing/investigating one image now belongs to the Review
    Window (Advanced Filters + the main grid) and the Loupe (per-image
    debugging), never a second image browser living inside the Dashboard.

    Every detail tab's own `show_run` takes an optional `paths`/`records`
    filter argument, always computed here from `self._all_run_records`
    (see `_build_run_records`) narrowed by `self._advanced_filters_panel`
    AND `self._drill_down_paths` together (`_apply_filters_to_tabs`) -
    exactly the two-source AND pattern the Review Window's own simple
    Filter combo + Advanced Filters already establishes, so a KPI/matrix-
    cell/species-row click (a "drill-down", pinned until explicitly
    cleared) and a manually-set Advanced Filters criterion can combine
    rather than fight each other.
    """

    def __init__(
        self,
        *,
        analytics_db: str | Path = DEFAULT_ANALYTICS_DB,
        annotations_db: str | Path = DEFAULT_ANNOTATIONS_DB,
        species_db: str | Path | None = None,
        settings: QSettings | None = None,
        root_folder: str | None = None,
        color_source: str | None = None,
        keep_percent: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PeakPic - Analytics Dashboard")
        # QDialog hides the maximize/minimize buttons by default on some
        # platforms (Windows in particular) and stays a fixed size until
        # explicitly resized - the same fix already applied to LoupeDialog
        # (see its own module docstring/__init__).
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(1200, 780)
        self._settings = settings if settings is not None else QSettings("PeakPic", "PeakPicDesktop")
        geometry = self._settings.value(_GEOMETRY_SETTINGS_KEY)
        if geometry is not None:
            self.restoreGeometry(geometry)

        self._store = AnalyticsStore(analytics_db)
        self._annotation_store = AnnotationStore(annotations_db)
        # Injectable so a test never touches the real project-wide species
        # cache (cache/species.db) just by selecting an experiment - see
        # _build_run_records. None (the default) means "the real one".
        self._species_db = species_db
        # The live Review context this dashboard was opened with - never
        # re-read afterward (MainWindow constructs a fresh dashboard each
        # time it is opened, see _show_analytics_dashboard), so these stay
        # exactly as accurate as "when I clicked Analytics Dashboard".
        self._root_folder = root_folder
        self._color_source = color_source
        self._keep_percent = keep_percent
        # Every image the currently-selected run touched, as FilterableRecords
        # (see _build_run_records) - rebuilt once per experiment selection,
        # never per filter change (_apply_filters_to_tabs only re-filters
        # this already-built list, the same "compute once, filter repeatedly"
        # pattern MainWindow's own Advanced Filters wiring uses).
        self._all_run_records: list[FilterableRecord] = []
        self._current_run_id: str | None = None
        # A KPI card / confusion-matrix cell / species-row click pins the
        # detail tabs to exactly that path list until explicitly cleared
        # (see _set_drill_down) - ANDed with Advanced Filters, never
        # replacing it, so a photographer can narrow a drill-down further
        # ("false positives" + "Score > 0.8") without losing either.
        self._drill_down_paths: list[str] | None = None

        # Phase 2 - Analytics Scope: narrows the Experiment Browser to only
        # runs recorded against the current Root Folder, or shows every run
        # ever recorded. "Every analytics page must respect this selection"
        # is satisfied structurally, not by separate logic per tab: every
        # tab only ever shows data for whichever experiment is SELECTED,
        # and Scope controls which experiments can be selected at all - see
        # _refresh_experiment_list.
        self._scope_current_root_radio = QRadioButton("Current Root Folder", self)
        self._scope_entire_db_radio = QRadioButton("Entire Analytics Database", self)
        scope_group = QButtonGroup(self)
        scope_group.addButton(self._scope_current_root_radio)
        scope_group.addButton(self._scope_entire_db_radio)
        if root_folder:
            self._scope_current_root_radio.setChecked(True)
        else:
            self._scope_current_root_radio.setEnabled(False)
            self._scope_current_root_radio.setToolTip("No folder is currently open in Review")
            self._scope_entire_db_radio.setChecked(True)
        self._scope_current_root_radio.toggled.connect(self._on_scope_changed)
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:", self))
        scope_row.addWidget(self._scope_current_root_radio)
        scope_row.addWidget(self._scope_entire_db_radio)

        self._experiment_search = QLineEdit(self)
        self._experiment_search.setPlaceholderText("Search experiments (folder or algorithm)...")
        self._experiment_search.textChanged.connect(self._apply_experiment_filter)

        self._experiment_list = QListWidget(self)
        self._experiment_list.setAlternatingRowColors(True)
        self._experiment_list.currentItemChanged.connect(self._on_experiment_selected)

        refresh_button = QPushButton("Refresh", self)
        refresh_button.clicked.connect(self._refresh_experiment_list)

        browser_column = QVBoxLayout()
        browser_column.addLayout(scope_row)
        browser_column.addWidget(QLabel("Experiments (most recent first)", self))
        browser_column.addWidget(self._experiment_search)
        browser_column.addWidget(self._experiment_list)
        browser_column.addWidget(refresh_button)
        browser_container = QWidget(self)
        browser_container.setLayout(browser_column)

        self._header_panel = DashboardHeaderPanel(self)
        self._header_panel.set_live_context(
            root_folder=root_folder, color_source=color_source, keep_percent=keep_percent,
            scope_label=self._current_scope_label(),
        )
        self._metadata_panel = ExperimentMetadataPanel(self)

        # Phase B/C - the same shared filtering engine the Review Window
        # uses (desktop/filtering.py), populated from _all_run_records
        # rather than ImageItem - see _refresh_advanced_filter_options.
        self._advanced_filters_panel = AdvancedFiltersPanel(self)
        self._advanced_filters_panel.criteriaChanged.connect(self._apply_filters_to_tabs)

        self._drill_down_label = QLabel(self)
        self._drill_down_label.setStyleSheet("color: palette(mid);")
        self._clear_drill_down_button = QPushButton("Clear", self)
        self._clear_drill_down_button.setVisible(False)
        self._clear_drill_down_button.clicked.connect(self._clear_drill_down)
        drill_down_row = QHBoxLayout()
        drill_down_row.addWidget(self._drill_down_label, 1)
        drill_down_row.addWidget(self._clear_drill_down_button)

        self._user_vs_algorithm_tab = UserVsAlgorithmTab(self)
        self._user_vs_algorithm_tab.drillDownRequested.connect(self._on_drill_down)
        self._run_summary_tab = RunSummaryTab(self)
        self._species_analysis_tab = SpeciesAnalysisTab(self)
        self._species_analysis_tab.speciesDrillDownRequested.connect(self._on_species_drill_down)
        self._burst_analytics_tab = BurstAnalyticsTab(annotation_store=self._annotation_store, parent=self)

        self._tabs = QTabWidget(self)
        # User vs Algorithm first - "the primary dashboard page" (see this
        # module's own Phase 2 mandate): the purpose of PickLikeMe is
        # agreement with the photographer, not maximizing a score.
        self._tabs.addTab(self._user_vs_algorithm_tab, "User vs Algorithm")
        self._tabs.addTab(self._run_summary_tab, "Run Summary")
        self._tabs.addTab(self._species_analysis_tab, "Species Analysis")
        self._tabs.addTab(self._burst_analytics_tab, "Burst Analytics")

        detail_column = QVBoxLayout()
        detail_column.addWidget(self._metadata_panel)
        detail_column.addWidget(self._advanced_filters_panel)
        detail_column.addLayout(drill_down_row)
        detail_column.addWidget(self._tabs, 1)
        detail_with_metadata = QWidget(self)
        detail_with_metadata.setLayout(detail_column)

        # Shown instead of the (otherwise blank-looking) tabs when there is
        # nothing recorded yet - a dashboard with 1161 tests behind it but
        # zero real runs still needs to say so in plain language, not just
        # render an empty table that looks like something broke.
        self._empty_state_label = QLabel(
            "No experiments recorded yet.\n\n"
            "Run “Rank…” or “Organize by Species…” from the Tools menu to record one - "
            "this dashboard only shows runs completed after analytics recording was added.\n\n"
            "Already ran one and still see this? Click Refresh.",
            self,
        )
        self._empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state_label.setWordWrap(True)
        self._empty_state_label.setStyleSheet("padding: 32px; color: palette(mid);")

        self._detail_stack = QStackedWidget(self)
        self._detail_stack.addWidget(detail_with_metadata)
        self._detail_stack.addWidget(self._empty_state_label)

        splitter = QSplitter(self)
        splitter.addWidget(browser_container)
        splitter.addWidget(self._detail_stack)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._header_panel)
        layout.addWidget(splitter, 1)

        self._refresh_experiment_list()
        if self._experiment_list.count():
            self._experiment_list.setCurrentRow(0)

    def _update_empty_state(self) -> None:
        has_experiments = self._experiment_list.count() > 0
        self._detail_stack.setCurrentIndex(0 if has_experiments else 1)

    def _scope_is_current_root(self) -> bool:
        return self._scope_current_root_radio.isChecked() and bool(self._root_folder)

    def _current_scope_label(self) -> str:
        if self._scope_is_current_root():
            return "Current Root Folder"
        return "Entire Analytics Database"

    def _on_scope_changed(self) -> None:
        self._header_panel.set_live_context(
            root_folder=self._root_folder, color_source=self._color_source, keep_percent=self._keep_percent,
            scope_label=self._current_scope_label(),
        )
        self._refresh_experiment_list()

    def _refresh_experiment_list(self) -> None:
        # Re-selects whatever run was showing before the rebuild (see
        # refresh_current_run's own docstring) - clicking Refresh after a
        # Ground Truth import must not leave the detail tabs holding stale
        # data just because the experiment LIST itself did not change.
        previously_selected = self._experiment_list.currentItem()
        previous_run_id = previously_selected.data(Qt.ItemDataRole.UserRole) if previously_selected else None

        # Phase 2 - Analytics Scope: "Current Root Folder" narrows the list
        # to runs recorded against exactly this folder (resolved, so a
        # trailing slash or relative-vs-absolute spelling difference never
        # silently hides a matching run) - every tab downstream only ever
        # shows whichever run is selected, so narrowing the SELECTABLE set
        # here is what makes "every analytics page respects this
        # selection" true everywhere at once, not a per-tab filter.
        scope_folder = str(Path(self._root_folder).resolve()) if self._scope_is_current_root() else None
        self._experiment_list.clear()
        restore_row = -1
        for row, run in enumerate(self._store.list_runs(folder=scope_folder)):
            label = f"{run['started_at']}  |  {_friendly_strategy_label(run['strategy_id'])}  |  {Path(run['folder']).name}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, run["run_id"])
            self._experiment_list.addItem(item)
            if run["run_id"] == previous_run_id:
                restore_row = row
        self._apply_experiment_filter(self._experiment_search.text())
        self._update_empty_state()
        if restore_row >= 0:
            self._experiment_list.setCurrentRow(restore_row)  # triggers _on_experiment_selected

    def _apply_experiment_filter(self, text: str) -> None:
        """A search box over the (usually short, but not always) experiment
        list - filters in place rather than rebuilding it, so the current
        selection survives typing a query that still matches it."""
        needle = text.strip().lower()
        for row in range(self._experiment_list.count()):
            item = self._experiment_list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _on_experiment_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            return
        run_id = current.data(Qt.ItemDataRole.UserRole)
        # A genuinely different run's own path list makes any pinned
        # drill-down meaningless (re-selecting the SAME run, e.g. via
        # refresh_current_run, must NOT lose it - that is exactly the
        # "still looking at the same thing after a refresh" case).
        if run_id != self._current_run_id:
            self._clear_drill_down(refresh=False)
        self._current_run_id = run_id

        self._header_panel.show_run(self._store, self._annotation_store, run_id)
        self._metadata_panel.show_run(self._store, run_id)
        self._all_run_records = _build_run_records(
            self._store, self._annotation_store, run_id, species_db=self._species_db,
        )
        self._refresh_advanced_filter_options()
        self._apply_filters_to_tabs()

    def _refresh_advanced_filter_options(self) -> None:
        """Keeps the panel's Folder/Species/Reject Reason/Burst Rank combos
        in sync with whichever run is currently selected - mirrors
        MainWindow's own _refresh_advanced_filter_options exactly."""
        records = self._all_run_records
        folders = sorted({r.folder for r in records if r.folder})
        species = sorted({r.species for r in records if r.species})
        reject_codes = sorted({r.reject_reason for r in records if r.reject_reason})
        reject_reasons = [(code, REJECT_REASON_LABELS.get(code, code)) for code in reject_codes]
        max_burst_size = max((r.burst_size for r in records), default=1)
        self._advanced_filters_panel.set_available_options(
            folders=folders, species=species, reject_reasons=reject_reasons, max_burst_size=max_burst_size,
        )

    def _apply_filters_to_tabs(self) -> None:
        """The single choke point every detail tab's data flows through -
        Advanced Filters' own criteria AND whichever drill-down is
        currently pinned (see the class docstring), applied to
        self._all_run_records, and the resulting path/record subset handed
        to every tab. Runs on: an experiment being selected, any Advanced
        Filters control changing (criteriaChanged), and any drill-down
        being set or cleared - the same "no Apply button, the grid updates
        immediately" requirement the Review Window's own Advanced Filters
        satisfies."""
        if self._current_run_id is None:
            return
        run_id = self._current_run_id
        records = self._all_run_records
        if self._drill_down_paths is not None:
            allowed = set(self._drill_down_paths)
            records = [r for r in records if r.path in allowed]

        criteria = self._advanced_filters_panel.criteria
        active = criteria.is_active() or self._drill_down_paths is not None
        if active:
            filtered_records = apply_filters(records, criteria)
            paths: list[str] | None = [r.path for r in filtered_records]
        else:
            filtered_records = None
            paths = None

        self._user_vs_algorithm_tab.show_run(self._store, self._annotation_store, run_id, paths=paths)
        self._run_summary_tab.show_run(self._store, run_id, paths=paths)
        self._species_analysis_tab.show_run(self._store, run_id, records=filtered_records)
        self._burst_analytics_tab.show_run(self._store, run_id, paths=paths)

    def _on_drill_down(self, paths: list, label: str) -> None:
        """A User vs Algorithm confusion-matrix cell or KPI card was
        clicked - pin every detail tab to exactly those images, so
        "drilling down into every category" (see the Phase 2 mandate)
        actually shows the photographer the images behind a number, not
        just the count itself."""
        self._set_drill_down(paths, label)

    def _on_species_drill_down(self, species: str) -> None:
        """Species Analysis's distribution table (Phase 9) - pins every
        detail tab to every image this run recorded under exactly that
        species/category. Computed from self._all_run_records (already
        built for the currently-selected run), the same source Advanced
        Filters' own Species control reads."""
        matching = [r.path for r in self._all_run_records if r.species == species]
        self._set_drill_down(matching, f"Species: {species}")

    def _set_drill_down(self, paths: list[str], label: str) -> None:
        self._drill_down_paths = list(paths)
        self._drill_down_label.setText(f"Showing: {label} ({len(paths)} image(s))")
        self._clear_drill_down_button.setVisible(True)
        self._apply_filters_to_tabs()

    def _clear_drill_down(self, *, refresh: bool = True) -> None:
        self._drill_down_paths = None
        self._drill_down_label.setText("")
        self._clear_drill_down_button.setVisible(False)
        if refresh:
            self._apply_filters_to_tabs()

    def refresh_current_run(self) -> None:
        """Re-runs whichever experiment is currently selected through every
        tab again - the explicit "immediately refresh Agreement/Confusion
        Matrix/Precision/Recall/F1" requirement after a Ground Truth import
        changes review decisions out from under an already-open dashboard.
        Advanced Filters and any pinned drill-down survive this (see
        _on_experiment_selected) - only the underlying data is recomputed."""
        self._on_experiment_selected(self._experiment_list.currentItem(), None)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._settings.setValue(_GEOMETRY_SETTINGS_KEY, self.saveGeometry())
        self._store.close()
        self._annotation_store.close()
        super().closeEvent(event)
