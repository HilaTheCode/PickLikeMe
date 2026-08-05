"""The Diagnostics & Analytics Dashboard.

Phase 2's own mandate: the goal is no longer displaying statistics, it is
understanding the algorithm - "why did it decide this", "where does it
disagree with the photographer", "what should improve next". User vs
Algorithm (see UserVsAlgorithmTab) is the primary page for that reason -
agreement with the photographer, not the score itself, is the actual
measure of success - shown first, ahead of Run Summary, Species Analysis
and Image Explorer. Implemented as one master-detail dialog - an
Experiment Browser (a list) on the left, and the four detail views as tabs
on the right that redraw for whichever experiment is currently selected.

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

The Image Explorer's per-image detail panel lists whatever
`store.image_metrics(run_id, image_path)` returns as generic rows, rather
than a fixed "top1_confidence, top2_confidence, ..." layout - so EyePose
metrics (eye_confidence, head_confidence, ...) recorded by a future run
appear here with no UI change, per the explicit "design accordingly"
instruction.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...analyzer.annotations import DEFAULT_ANNOTATIONS_DB, AnnotationStore
from ...analyzer.contactsheets import EYE_BOX_ACCEPTED, EYE_BOX_REJECTED, OTHER_BOX, SELECTED_BOX
from ...analytics.agreement import (
    AgreementReport,
    algorithm_decisions_for_run,
    compare_run_to_user_decisions,
    user_decisions_for_paths,
)
from ...analytics.reports import metric_statistics, run_statistics
from ...analytics.score_explanation import explain_score
from ...analytics.store import DEFAULT_ANALYTICS_DB, AnalyticsStore
from ...bird_crop import crop_cache_path
from ...burst_analysis import BurstInfo, ScoredImage, analyze_bursts
from ...config import DEFAULT_CROP_CACHE_DIR
from ...ranking.classic import read_filter_report
from ...ranking.filters import REJECT_REASON_LABELS
from ...review.thumbnails import detection_boxes_for, eye_keypoints_for, eye_keypoints_in_crop_for

_THUMBNAIL_SIZE = 320
# Manual QA Issue 3: the Original/Crop panels in ImageExplorerTab
# specifically (every other tab's thumbnails stay at _THUMBNAIL_SIZE) - the
# landmark overlay was unreadable at 320px, so both panels render
# significantly larger there.
_INSPECTOR_IMAGE_SIZE = 460

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

    def show_run(self, analytics_store: AnalyticsStore, annotation_store: AnnotationStore, run_id: str) -> None:
        self._analytics_store = analytics_store
        self._annotation_store = annotation_store
        self._run_id = run_id

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
            self._analytics_store, self._annotation_store, self._run_id, keep_percent=keep_percent,
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

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        stats = run_statistics(store, run_id)
        run = store.get_run(run_id) or {}
        summary = store.summary_metrics(run_id)

        considered = stats.get("considered", 0)
        accepted = stats.get("accepted", 0)
        rejected = stats.get("rejected", 0)
        self._images_processed_card.set_value(str(considered))
        self._accepted_card.set_value(str(accepted))
        self._rejected_card.set_value(str(rejected))
        self._acceptance_card.set_value(f"{100.0 * accepted / considered:.1f}%" if considered else "n/a")
        score_stats = metric_statistics(store, run_id, "score")
        self._score_card.set_value(f"{score_stats['mean']:.4f}" if score_stats else "n/a")

        def _fmt(value: float | None) -> str:
            return f"{value:.4f}" if value is not None else "n/a"

        score_stats = score_stats or {}
        eye_confidence_stats = metric_statistics(store, run_id, "eye_confidence") or {}
        head_confidence_stats = metric_statistics(store, run_id, "head_confidence") or {}
        eye_sharpness_stats = metric_statistics(store, run_id, "eye_sharpness") or {}
        subject_sharpness_stats = metric_statistics(store, run_id, "subject_sharpness") or {}
        subject_size_stats = metric_statistics(store, run_id, "subject_size") or {}
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
        metric_names = store.metric_names(run_id)
        self._metric_stats_table.setRowCount(len(metric_names))
        self._metric_stats_table.setColumnCount(5)
        self._metric_stats_table.setHorizontalHeaderLabels(["Metric", "Mean", "Median", "Min", "Max"])
        for row_index, name in enumerate(metric_names):
            metric_stats = metric_statistics(store, run_id, name) or {}
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
            ("Folder", stats.get("folder", "")),
            ("Backend / Strategy", stats.get("strategy_id", "")),
            ("Started at", stats.get("started_at", "")),
            ("Device", stats.get("device") or "n/a"),
            ("Considered", str(considered)),
            ("Accepted", str(accepted)),
            ("Rejected", str(rejected)),
        ]
        for name, value in sorted(summary.items()):
            rows.append((f"Runtime: {name}", f"{value:.4f}" if isinstance(value, float) else str(value)))
        for name, value in sorted(stats.get("metric_means", {}).items()):
            rows.append((f"Mean {name}", f"{value:.4f}"))
        # The full experiment record (model id/version, species list hash,
        # GPU, thresholds, ...) lives in params for a species run - see
        # species.experiment.ExperimentMetadata.to_dict(). Shown generically
        # so a ranking run's differently-shaped params render just as well.
        params = stats.get("params", {})
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

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        run = store.get_run(run_id) or {}
        considered = run.get("considered", 0) or 0
        accepted = run.get("accepted", 0) or 0
        counts = store.category_counts(run_id)
        summary = store.summary_metrics(run_id)

        unknown_rate = summary.get("unknown_rate")
        label_text = f"{len(counts)} distinct outcome(s) across {considered} image(s)"
        self._label.setText(label_text)
        self._unknown_rate_card.set_value(f"{unknown_rate:.1%}" if unknown_rate is not None else "n/a")

        confidence_stats = metric_statistics(store, run_id, "top1_confidence")
        self._confidence_card.set_value(f"{confidence_stats['mean']:.4f}" if confidence_stats else "n/a")

        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        _fill_ranked_table(self._top5_table, ranked[:5], considered)

        self._table.setSortingEnabled(False)
        # "Accepted" is not itself a row in category_counts (it is the
        # complement of every reject/category count, tracked separately on
        # the run itself) - shown first so the table reads as a complete
        # outcome breakdown, not only the rejected/categorized slice. Not
        # clickable (see _on_row_clicked) - it is not a species/category
        # name the Image Explorer's species filter could ever match.
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

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        self._store = store
        self._run_id = run_id
        paths = store.image_paths(run_id)
        burst_by_path = _compute_burst_map(store, self._annotation_store, run_id, paths)

        burst_ids = {info.burst_id for info in burst_by_path.values()}
        sizes = [info.burst_size for info in burst_by_path.values() if info.burst_rank == 1]
        singleton_images = sum(1 for info in burst_by_path.values() if info.burst_size == 1)
        multi_image_bursts = sum(1 for size in sizes if size > 1)

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
        self._table.setRowCount(len(paths))
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Image", "Burst Size", "Burst Rank", "Burst Winner"])
        for row_index, path in enumerate(paths):
            info = burst_by_path.get(path)
            self._table.setItem(row_index, 0, QTableWidgetItem(Path(path).name))
            self._table.setItem(row_index, 1, QTableWidgetItem(str(info.burst_size) if info else "n/a"))
            self._table.setItem(row_index, 2, QTableWidgetItem(str(info.burst_rank) if info else "n/a"))
            self._table.setItem(row_index, 3, QTableWidgetItem("Yes" if info and info.burst_best else "No"))
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSortingEnabled(True)


# ---------------------------------------------------------------------------
# Visual Debug (Phase 5): every processing-stage rectangle/marker the Image
# Explorer can draw, each its own checkbox and its own colour/style, never
# merged into another stage's rectangle - see ImageExplorerTab's own
# docstring for which stages share a colour family and why, and its Known
# Limitations for the two pairs this codebase's own data model cannot tell
# apart (Detection Boxes vs its own Selected/Candidate breakdown; Expanded
# Crop vs Final Crop).
# ---------------------------------------------------------------------------

# The BOX family - drawn on the Original (a full-frame concept).
_UNION_DETECTION_PEN = QPen(QColor(148, 163, 184), 2, Qt.PenStyle.DotLine)
_CANDIDATE_DETECTION_PEN = QPen(QColor(*OTHER_BOX), 2, Qt.PenStyle.DashLine)
_SELECTED_DETECTION_PEN = QPen(QColor(*SELECTED_BOX), 2)
_REJECTED_DETECTION_PEN = QPen(QColor(*EYE_BOX_REJECTED), 2, Qt.PenStyle.DashDotLine)
_EXPANDED_CROP_PEN = QPen(QColor(59, 130, 246), 2, Qt.PenStyle.DashLine)
_FINAL_CROP_PEN = QPen(QColor(59, 130, 246), 2, Qt.PenStyle.DotLine)
_HEAD_BOX_PEN = QPen(QColor(249, 115, 22), 2)

# The LANDMARK family - drawn on the Crop when one is cached, else falls
# back to the Original (see Manual QA Issue 3). Name, the
# eye_keypoints_for()/eye_keypoints_in_crop_for() dict key holding it, and a
# colour distinct from every box colour above so boxes and landmarks never
# get confused when both overlay families are on at once.
_LANDMARK_STYLE: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("Left Eye", "left", (56, 189, 248)),
    ("Right Eye", "right", (251, 146, 60)),
    ("Beak", "beak", (250, 204, 21)),
    ("Head", "head_top", (226, 232, 240)),
    ("Left Shoulder", "left_shoulder", (167, 139, 250)),
    ("Right Shoulder", "right_shoulder", (192, 132, 252)),
)
# Only these four (never the shoulders - a body/torso landmark, not a head
# one) contribute to the approximate "Head Box" overlay.
_HEAD_BOX_LANDMARK_KEYS: tuple[str, ...] = ("left", "right", "beak", "head_top")

# Every independent overlay checkbox Visual Debug supports (key, label) -
# key doubles as the dict key in the checked-state map and the preset sets
# below. Order here is the order checkboxes render in the grid.
_OVERLAY_CHECKBOX_DEFS: tuple[tuple[str, str], ...] = (
    ("detection_boxes", "Detection Boxes"),
    ("candidate_detections", "Candidate Detection Boxes"),
    ("selected_detection", "Selected Detection"),
    ("rejected_detections", "Rejected Detections"),
    ("expanded_crop", "Expanded Crop"),
    ("final_crop", "Final Crop"),
    ("eye_roi", "Eye ROI"),
    ("head_box", "Head Box"),
    ("landmarks", "Landmarks"),
    ("landmark_labels", "Landmark Labels"),
    ("confidence_values", "Confidence Values"),
)
_ALL_OVERLAY_KEYS = frozenset(key for key, _label in _OVERLAY_CHECKBOX_DEFS)
_BOX_FAMILY_KEYS = frozenset({
    "detection_boxes", "candidate_detections", "selected_detection", "rejected_detections",
    "expanded_crop", "final_crop", "eye_roi",
})
_LANDMARK_FAMILY_KEYS = frozenset({"head_box", "landmarks", "landmark_labels", "confidence_values"})

# Preset -> exact checked-set. "Custom" has no fixed set - it means "keep
# whatever is currently checked" - see _on_overlay_checkbox_toggled.
_OVERLAY_PRESETS: dict[str, frozenset[str]] = {
    "Detection": frozenset({"detection_boxes", "candidate_detections", "selected_detection", "rejected_detections"}),
    "Crop": frozenset({"expanded_crop", "final_crop"}),
    "EyePose": frozenset({"eye_roi", "head_box", "landmarks", "landmark_labels", "confidence_values"}),
    "Everything": frozenset(_ALL_OVERLAY_KEYS),
}
_PRESET_NAMES: tuple[str, ...] = ("Detection", "Crop", "EyePose", "Everything", "Custom")


class ImageExplorerTab(QWidget):
    """Phase 4/5/6 - the primary investigation tool: search/filter over
    whichever run (or drill-down subset) is currently shown, full per-image
    detail (Original, Crop, Ground Truth, Algorithm Decision, User Decision,
    generic metrics, Score Explanation), and Visual Debug - every ranking-
    pipeline processing stage as its own independently-toggleable overlay,
    with presets for the common combinations. Respects the dashboard's
    Analytics Scope structurally, the same way every other tab does: it only
    ever shows images from whichever experiment Scope allowed to be
    SELECTED (see AnalyticsDashboard's own docstring) - there is no separate
    scope check to duplicate here.

    Two independent overlay families, restated from Manual QA Issue 3's own
    fix: the BOX family (_BOX_FAMILY_KEYS - detection/crop/eye-ROI
    rectangles, full-frame concepts) draws on the Original; the LANDMARK
    family (_LANDMARK_FAMILY_KEYS - markers, labels, head box) draws on the
    Crop when one is cached (the head fills most of the frame there, not a
    small fraction of it) with a fallback to the Original otherwise.

    Known limitations:
    - Landmarks (and Eye ROI, Head Box, Rejected Detections) are only
      available for images a Classic Vision run scored with EyePose-v0
      specifically - SuperAnimal-Bird locates only the eye, so
      beak/head/shoulders are never available for it.
    - "Expanded Crop" and "Final Crop" render as the same rectangle, and
      "Detection Boxes" (every detected box, selected and candidates alike,
      in one neutral style) necessarily overlaps "Selected Detection" +
      "Candidate Detection Boxes" (the same boxes, split into two more
      specific toggles) - this codebase's DetectionRecord only ever tracked
      one margin-grown crop rectangle and one selected/others split, never a
      four-way or three-way breakdown, so exposing all of Phase 5's named
      checkboxes means some of them necessarily draw the same underlying
      rectangle a differently-scoped checkbox also draws.
    - "Reject Reason" filtering (and the images it would surface) depends on
      `.picklikeme/classic_vision_filters.json` still existing next to the
      run's own folder - AnalyticsStore itself only ever persisted AGGREGATE
      reject-reason counts, never a per-image reason. That sidecar is
      overwritten by every new Classic Vision run for the same folder, so
      inspecting an OLDER historical run's filtered-out images only works
      until a newer run against the same folder replaces it.
    - "Species Prediction" and "Burst Rank" are both best-effort: species
      comes from `species.cache.SpeciesCache`, keyed by this run's own
      strategy_id as classifier_id (empty for a ranking run, which never
      wrote to that cache); burst rank is recomputed locally from each
      image's own EXIF capture time and this run's own score, not read from
      any persisted burst table (none exists - see burst_analysis.py's own
      docstring on why burst ranking is never persisted).
    - "Algorithm Decision"/"Conflict Type" always use this run's own default
      keep-percent threshold (accepted/considered ratio), independent of any
      threshold adjustment made in the User vs Algorithm tab - the two do
      not live-sync.
    - A pre-v3 cached eye record (see eyes.cache.EYE_CACHE_VERSION) has no
      beak/head/shoulder landmarks on disk yet; re-running Classic Vision
      regenerates it with them.
    """

    def __init__(
        self, *, crop_cache_dir: Path, annotation_store: AnnotationStore,
        species_db: str | Path | None = None, parent=None,
    ) -> None:
        super().__init__(parent)
        self._crop_cache_dir = crop_cache_dir
        self._annotation_store = annotation_store
        # Injectable so a test never touches the real project-wide species
        # cache (cache/species.db) just by populating the image list - the
        # same isolation crop_cache_dir/annotation_store already get from
        # their own callers. None (the default, used by AnalyticsDashboard)
        # means "the real one" - species.cache.SpeciesCache's own default.
        self._species_db = species_db
        self._store: AnalyticsStore | None = None
        self._run_id: str | None = None

        self._all_paths: list[str] = []
        self._algo_decision_by_path: dict[str, str] = {}
        self._user_decision_by_path: dict[str, str] = {}
        self._reject_reason_by_path: dict[str, str | None] = {}
        self._species_by_path: dict[str, str] = {}
        self._burst_by_path: dict[str, BurstInfo] = {}

        self._current_image_path: str | None = None
        self._current_full_pixmap: QPixmap | None = None
        self._current_crop_full_pixmap: QPixmap | None = None

        # ---- Filter bar (Phase 4) ------------------------------------
        self._search_edit = QLineEdit(self)
        self._search_edit.setPlaceholderText("Search filename...")
        self._search_edit.textChanged.connect(self._apply_filters)

        self._folder_combo = QComboBox(self)
        self._species_combo = QComboBox(self)
        self._burst_combo = QComboBox(self)
        self._burst_combo.addItems(["All", "Burst Winners", "Burst Losers"])
        self._reject_reason_combo = QComboBox(self)
        self._conflict_combo = QComboBox(self)
        self._conflict_combo.addItems(["All", "Agree", "False Positive", "False Negative", "N/A"])
        self._user_decision_combo = QComboBox(self)
        self._user_decision_combo.addItems(["All", "Keep", "Reject", "Neutral"])
        self._algo_decision_combo = QComboBox(self)
        self._algo_decision_combo.addItems(["All", "Keep", "Reject"])
        for combo in (
            self._folder_combo, self._species_combo, self._burst_combo, self._reject_reason_combo,
            self._conflict_combo, self._user_decision_combo, self._algo_decision_combo,
        ):
            combo.currentIndexChanged.connect(self._apply_filters)

        self._score_min_spin = QDoubleSpinBox(self)
        self._score_max_spin = QDoubleSpinBox(self)
        for spin in (self._score_min_spin, self._score_max_spin):
            spin.setDecimals(4)
            spin.setRange(-1000.0, 1000.0)
            spin.valueChanged.connect(self._apply_filters)

        filter_grid = QGridLayout()
        filter_grid.addWidget(self._search_edit, 0, 0, 1, 4)
        for row, (label, widget) in enumerate((
            ("Folder", self._folder_combo), ("Species", self._species_combo),
            ("Burst", self._burst_combo), ("Reject Reason", self._reject_reason_combo),
            ("Conflict Type", self._conflict_combo), ("User Decision", self._user_decision_combo),
            ("Algorithm Decision", self._algo_decision_combo),
        ), start=1):
            filter_grid.addWidget(QLabel(label, self), row, 0)
            filter_grid.addWidget(widget, row, 1)
        score_row = filter_grid.rowCount()
        filter_grid.addWidget(QLabel("Score range", self), score_row, 0)
        score_range_row = QHBoxLayout()
        score_range_row.addWidget(self._score_min_spin)
        score_range_row.addWidget(QLabel("to", self))
        score_range_row.addWidget(self._score_max_spin)
        filter_grid.addLayout(score_range_row, score_row, 1)

        self._image_list = QListWidget(self)
        self._image_list.setAlternatingRowColors(True)
        self._image_list.currentItemChanged.connect(self._on_image_selected)

        list_container = QWidget(self)
        list_layout = QVBoxLayout(list_container)
        list_layout.addWidget(QLabel("Filters", self))
        list_layout.addLayout(filter_grid)
        list_layout.addWidget(QLabel("Images", self))
        list_layout.addWidget(self._image_list, 1)

        # ---- Original / Crop panels (Manual QA Issue 3) ----------------
        self._original_label = QLabel("Original", self)
        self._original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._original_label.setMinimumSize(_INSPECTOR_IMAGE_SIZE, _INSPECTOR_IMAGE_SIZE)
        self._crop_label = QLabel("Crop", self)
        self._crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_label.setMinimumSize(_INSPECTOR_IMAGE_SIZE, _INSPECTOR_IMAGE_SIZE)
        images_row = QHBoxLayout()
        images_row.addWidget(self._original_label)
        images_row.addWidget(self._crop_label)

        # ---- Visual Debug overlay presets + checkboxes (Phase 5) -------
        self._overlay_preset_combo = QComboBox(self)
        self._overlay_preset_combo.addItems(_PRESET_NAMES)
        self._overlay_preset_combo.setCurrentText("Custom")
        self._overlay_preset_combo.currentTextChanged.connect(self._on_overlay_preset_changed)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Overlay preset:", self))
        preset_row.addWidget(self._overlay_preset_combo)
        preset_row.addStretch(1)

        self._overlay_checkboxes: dict[str, QCheckBox] = {}
        overlay_grid = QGridLayout()
        for index, (key, label) in enumerate(_OVERLAY_CHECKBOX_DEFS):
            checkbox = QCheckBox(label, self)
            checkbox.toggled.connect(self._on_overlay_checkbox_toggled)
            self._overlay_checkboxes[key] = checkbox
            overlay_grid.addWidget(checkbox, index // 3, index % 3)

        self._metrics_table = QTableWidget(self)
        self._metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._metrics_table)

        # ---- Score Explanation (Phase 6) --------------------------------
        self._score_explanation_summary_label = QLabel(self)
        self._score_explanation_summary_label.setStyleSheet("font-weight: 600;")
        self._score_table = QTableWidget(self)
        self._score_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._score_table)

        self._landmarks_table = QTableWidget(self)
        self._landmarks_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._landmarks_table)

        detail_column = QVBoxLayout()
        detail_column.addLayout(images_row)
        detail_column.addLayout(preset_row)
        detail_column.addLayout(overlay_grid)
        detail_column.addWidget(QLabel("Details (Metrics / Ground Truth / Decisions)", self))
        detail_column.addWidget(self._metrics_table)
        detail_column.addWidget(QLabel("Score Explanation", self))
        detail_column.addWidget(self._score_explanation_summary_label)
        detail_column.addWidget(self._score_table)
        detail_column.addWidget(QLabel("Landmarks", self))
        detail_column.addWidget(self._landmarks_table)

        # The detail column is tall (large images + preset grid + three
        # tables) - a scroll area keeps it usable on anything smaller than a
        # maximized window, matching Phase 12's own "responsive layouts"
        # requirement rather than fighting it later.
        detail_content = QWidget(self)
        detail_content.setLayout(detail_column)
        detail_scroll = QScrollArea(self)
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setWidget(detail_content)

        splitter = QSplitter(self)
        splitter.addWidget(list_container)
        splitter.addWidget(detail_scroll)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # ---- Populating the candidate set -----------------------------------

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        self._store = store
        self._run_id = run_id
        self._populate_candidates(store.image_paths(run_id), include_filtered=True)

    def show_paths(self, store: AnalyticsStore, run_id: str, paths: list[str]) -> None:
        """Populate the image list from exactly `paths`, not every image
        this run recorded - the drill-down target for User vs Algorithm's
        confusion-matrix cells and KPI cards ("show me only the false
        positives"), and for any other future caller that wants a filtered
        view rather than the whole run. Unlike `show_run`, never extended
        with filtered-out images from the run's own sidecar - a drill-down
        caller asked for an EXACT set."""
        self._store = store
        self._run_id = run_id
        self._populate_candidates(list(paths), include_filtered=False)

    def _populate_candidates(self, scored_paths: list[str], *, include_filtered: bool) -> None:
        assert self._store is not None and self._run_id is not None  # noqa: S101 - only called from show_run/show_paths
        run = self._store.get_run(self._run_id) or {}
        folder = run.get("folder")
        strategy_id = run.get("strategy_id", "")

        self._reject_reason_by_path = {path: None for path in scored_paths}
        filtered_paths: list[str] = []
        if include_filtered and folder:
            try:
                report = read_filter_report(folder, strategy_id)
                for path, reason in (report.get("images") or {}).items():
                    if path not in self._reject_reason_by_path:
                        filtered_paths.append(path)
                    self._reject_reason_by_path[path] = reason
            except Exception:  # noqa: BLE001 - best-effort; see class docstring's Known Limitations
                pass
        self._all_paths = list(scored_paths) + filtered_paths

        try:
            self._algo_decision_by_path = algorithm_decisions_for_run(self._store, self._run_id)
        except Exception:  # noqa: BLE001 - filtering must never crash the dashboard
            self._algo_decision_by_path = {}
        try:
            self._user_decision_by_path = user_decisions_for_paths(self._annotation_store, self._all_paths)
        except Exception:  # noqa: BLE001
            self._user_decision_by_path = {}

        self._species_by_path = {}
        try:
            from ...species.cache import DEFAULT_SPECIES_DB, SpeciesCache

            cache = SpeciesCache(self._species_db if self._species_db is not None else DEFAULT_SPECIES_DB)
            try:
                for path in self._all_paths:
                    prediction = cache.get(path, strategy_id)
                    if prediction is not None and prediction.species:
                        self._species_by_path[path] = prediction.species
            finally:
                cache.close()
        except Exception:  # noqa: BLE001 - species prediction is best-effort (see Known Limitations)
            pass

        self._burst_by_path = _compute_burst_map(self._store, self._annotation_store, self._run_id, self._all_paths)

        self._refresh_filter_options()
        self._apply_filters()

    def _refresh_filter_options(self) -> None:
        def _reset_combo(combo: QComboBox, options: list[str]) -> None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(options)
            combo.blockSignals(False)

        folders = sorted({str(Path(path).parent) for path in self._all_paths})
        _reset_combo(self._folder_combo, ["All Folders"] + folders)

        species = sorted({name for name in self._species_by_path.values() if name})
        _reset_combo(self._species_combo, ["All Species"] + species)

        reasons = sorted({reason for reason in self._reject_reason_by_path.values() if reason})
        _reset_combo(self._reject_reason_combo, ["All"] + [REJECT_REASON_LABELS.get(r, r) for r in reasons])

        scores = []
        if self._store is not None and self._run_id is not None:
            scores = [
                value for path in self._all_paths
                if (value := self._store.image_metrics(self._run_id, path).get("score")) is not None
            ]
        low, high = (min(scores), max(scores)) if scores else (0.0, 1.0)
        for spin, value in ((self._score_min_spin, low), (self._score_max_spin, high)):
            spin.blockSignals(True)
            spin.setRange(min(low, -1000.0), max(high, 1000.0))
            spin.setValue(value)
            spin.blockSignals(False)

    def _conflict_type(self, path: str) -> str:
        algo_status = self._algo_decision_by_path.get(path)
        user_status = self._user_decision_by_path.get(path)
        if algo_status is None or user_status is None or user_status == "neutral":
            return "N/A"
        if algo_status == user_status:
            return "Agree"
        return "False Positive" if algo_status == "keep" else "False Negative"

    def _apply_filters(self) -> None:
        """Every filter combines with every other (AND), matching Phase 4's
        own "allow combining multiple filters" requirement - a candidate
        image must clear all of them, not just one."""
        search = self._search_edit.text().strip().lower()
        folder_choice = self._folder_combo.currentText()
        species_choice = self._species_combo.currentText()
        burst_choice = self._burst_combo.currentText()
        score_min, score_max = self._score_min_spin.value(), self._score_max_spin.value()
        reason_choice = self._reject_reason_combo.currentText()
        conflict_choice = self._conflict_combo.currentText()
        user_choice = self._user_decision_combo.currentText()
        algo_choice = self._algo_decision_combo.currentText()
        previous_path = self._current_image_path

        visible: list[str] = []
        for path in self._all_paths:
            if search and search not in Path(path).name.lower():
                continue
            if folder_choice not in ("", "All Folders") and str(Path(path).parent) != folder_choice:
                continue
            if species_choice not in ("", "All Species") and self._species_by_path.get(path) != species_choice:
                continue
            burst = self._burst_by_path.get(path)
            if burst_choice == "Burst Winners" and not (burst and burst.burst_best):
                continue
            if burst_choice == "Burst Losers" and not (burst and not burst.burst_best):
                continue
            score = None
            if self._store is not None and self._run_id is not None:
                score = self._store.image_metrics(self._run_id, path).get("score")
            if score is not None and not (score_min <= score <= score_max):
                continue
            if reason_choice != "All":
                reason = self._reject_reason_by_path.get(path)
                reason_label = REJECT_REASON_LABELS.get(reason, reason) if reason else None
                if reason_label != reason_choice:
                    continue
            if algo_choice != "All" and (self._algo_decision_by_path.get(path) or "").capitalize() != algo_choice:
                continue
            if user_choice != "All" and (self._user_decision_by_path.get(path) or "").capitalize() != user_choice:
                continue
            if conflict_choice != "All" and self._conflict_type(path) != conflict_choice:
                continue
            visible.append(path)

        self._image_list.blockSignals(True)
        self._image_list.clear()
        restore_row = -1
        for row, path in enumerate(visible):
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._image_list.addItem(item)
            if path == previous_path:
                restore_row = row
        self._image_list.blockSignals(False)

        if visible:
            self._image_list.setCurrentRow(restore_row if restore_row >= 0 else 0)  # triggers _on_image_selected
        else:
            self._current_image_path = None
            self._current_full_pixmap = None
            self._current_crop_full_pixmap = None
            self._original_label.setText("(no images match the current filters)")
            self._crop_label.setText("")
            self._metrics_table.setRowCount(0)
            self._landmarks_table.setRowCount(0)
            self._score_table.setRowCount(0)
            self._score_explanation_summary_label.setText("")

    # ---- Per-image detail --------------------------------------------

    def _on_image_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None or self._store is None or self._run_id is None:
            return
        image_path = current.data(Qt.ItemDataRole.UserRole)
        run = self._store.get_run(self._run_id) or {}

        self._current_image_path = image_path
        self._current_full_pixmap = self._load_full_pixmap(self._original_pixmap_path(image_path))
        crop_path = crop_cache_path(self._crop_cache_dir, image_path)
        self._current_crop_full_pixmap = self._load_full_pixmap(crop_path) if crop_path.is_file() else None
        self._refresh_overlays()

        metrics = self._store.image_metrics(self._run_id, image_path)
        user_status = self._user_decision_by_path.get(image_path)
        algo_status = self._algo_decision_by_path.get(image_path)
        rows = [
            ("Image", image_path),
            ("Experiment ID", self._run_id),
            ("Backend", run.get("strategy_id", "")),
            # Ground Truth and User Decision render from the same underlying
            # review-decision record - this app has no separate ground-truth
            # store distinct from a manual review decision (see the class
            # docstring's Known Limitations) - shown as two rows anyway since
            # Phase 4 lists them as two distinct display fields.
            ("Ground Truth", (user_status or "not reviewed").capitalize()),
            ("User Decision", (user_status or "not reviewed").capitalize()),
            ("Algorithm Decision", (algo_status or "n/a").capitalize()),
        ]
        species = self._species_by_path.get(image_path)
        if species:
            rows.append(("Species Prediction", species))
        burst = self._burst_by_path.get(image_path)
        if burst:
            winner = " (winner)" if burst.burst_best else ""
            rows.append(("Burst Rank", f"{burst.burst_rank} of {burst.burst_size}{winner}"))
        for name, value in sorted(metrics.items()):
            rows.append((name, f"{value:.4f}"))
        _fill_two_column_table(self._metrics_table, rows)

        self._fill_score_explanation(image_path)

    def _fill_score_explanation(self, image_path: str) -> None:
        table = self._score_table
        try:
            explanation = explain_score(self._store, self._run_id, image_path)
        except Exception:  # noqa: BLE001 - must never crash the dashboard
            explanation = None

        table.setSortingEnabled(False)
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["Metric", "Raw Value", "Normalized Value", "Weight", "Contribution", "Running Total"]
        )
        if explanation is None or not explanation.rows:
            table.setRowCount(0)
            self._score_explanation_summary_label.setText(
                "No per-metric breakdown recorded for this run/image "
                "(e.g. an AI-model or species-classification run)."
            )
            table.setSortingEnabled(True)
            return

        table.setRowCount(len(explanation.rows))
        for row_index, row in enumerate(explanation.rows):
            values = (
                row.label, f"{row.raw_value:.4f}", f"{row.normalized_value:.4f}",
                f"{row.weight:.3f}", f"{row.contribution:.4f}", f"{row.running_total:.4f}",
            )
            for col_index, value in enumerate(values):
                table.setItem(row_index, col_index, QTableWidgetItem(value))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        table.setSortingEnabled(True)

        final = explanation.final_score
        summary = f"Final Score: {final:.4f}" if final is not None else "Final Score: n/a"
        if explanation.recomputed_score is not None:
            summary += f"   (recomputed from breakdown: {explanation.recomputed_score:.4f})"
        self._score_explanation_summary_label.setText(summary)

    # ---- Visual Debug (Phase 5) ------------------------------------------

    def _on_overlay_preset_changed(self, name: str) -> None:
        preset_keys = _OVERLAY_PRESETS.get(name)
        if preset_keys is not None:  # not "Custom" - Custom keeps whatever is already checked
            for key, checkbox in self._overlay_checkboxes.items():
                checkbox.blockSignals(True)
                checkbox.setChecked(key in preset_keys)
                checkbox.blockSignals(False)
        self._refresh_overlays()

    def _on_overlay_checkbox_toggled(self, _checked: bool) -> None:
        current_keys = frozenset(key for key, cb in self._overlay_checkboxes.items() if cb.isChecked())
        matching_preset = next((name for name, keys in _OVERLAY_PRESETS.items() if keys == current_keys), "Custom")
        self._overlay_preset_combo.blockSignals(True)
        self._overlay_preset_combo.setCurrentText(matching_preset)
        self._overlay_preset_combo.blockSignals(False)
        self._refresh_overlays()

    def _refresh_overlays(self) -> None:
        """Redraws both the Original and Crop panels from their already-
        loaded full-resolution pixmaps plus whichever overlay checkboxes are
        currently checked - runs on image selection and on every checkbox/
        preset change, without re-reading either image file each time.

        The BOX family draws on the Original; the LANDMARK family draws on
        the Crop when one is cached, else falls back to the Original - see
        the class docstring.
        """
        image_path = self._current_image_path
        original_pixmap = self._current_full_pixmap
        if image_path is None or original_pixmap is None or original_pixmap.isNull():
            if image_path is not None:
                self._original_label.setText(f"Original not available\n({Path(image_path).name} may have moved)")
            self._crop_label.setText("")
            self._landmarks_table.setRowCount(0)
            return

        checked = {key: cb.isChecked() for key, cb in self._overlay_checkboxes.items()}
        original_display = original_pixmap.scaled(
            _INSPECTOR_IMAGE_SIZE, _INSPECTOR_IMAGE_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        crop_pixmap = self._current_crop_full_pixmap
        crop_display = (
            crop_pixmap.scaled(
                _INSPECTOR_IMAGE_SIZE, _INSPECTOR_IMAGE_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            )
            if crop_pixmap is not None and not crop_pixmap.isNull()
            else None
        )

        if any(checked[key] for key in _BOX_FAMILY_KEYS):
            boxes = detection_boxes_for(image_path)
            eye_full = eye_keypoints_for(image_path, crop_cache_dir=self._crop_cache_dir)
            original_display = self._draw_box_family(original_display, boxes, eye_full, checked)

        if any(checked[key] for key in _LANDMARK_FAMILY_KEYS):
            if crop_display is not None:
                eye_crop = eye_keypoints_in_crop_for(image_path, crop_cache_dir=self._crop_cache_dir)
                crop_display = self._draw_landmark_family(crop_display, eye_crop, checked)
                self._fill_landmark_table(eye_crop)
                # "Keep optional overlay on the original image if useful"
                # (Issue 3's own wording): markers only, never the text that
                # made the original unreadable in the first place - the crop
                # above is the detailed view.
                eye_full = eye_keypoints_for(image_path, crop_cache_dir=self._crop_cache_dir)
                context_only = {**checked, "landmark_labels": False, "confidence_values": False}
                original_display = self._draw_landmark_family(original_display, eye_full, context_only)
            else:
                eye_full = eye_keypoints_for(image_path, crop_cache_dir=self._crop_cache_dir)
                original_display = self._draw_landmark_family(original_display, eye_full, checked)
                self._fill_landmark_table(eye_full)
        else:
            self._landmarks_table.setRowCount(0)

        self._original_label.setPixmap(original_display)
        if crop_display is not None:
            self._crop_label.setPixmap(crop_display)
        else:
            self._crop_label.setText("No crop cached for this image\n(this backend may classify the full frame)")

    def _draw_box_family(
        self, display: QPixmap, boxes: dict | None, eye: dict | None, checked: dict[str, bool]
    ) -> QPixmap:
        source_size = (boxes or {}).get("source_size") or (eye or {}).get("source_size")
        if source_size is None or not source_size[0] or not source_size[1]:
            return display
        scale_x = display.width() / source_size[0]
        scale_y = display.height() / source_size[1]

        def to_display(box) -> QRectF:
            x1, y1, x2, y2 = box
            return QRectF(x1 * scale_x, y1 * scale_y, (x2 - x1) * scale_x, (y2 - y1) * scale_y)

        composed = QPixmap(display)
        painter = QPainter(composed)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            if checked["detection_boxes"] and boxes:
                painter.setPen(_UNION_DETECTION_PEN)
                for box in ([boxes["selected"]] if boxes.get("selected") else []) + list(boxes.get("others") or []):
                    painter.drawRect(to_display(box["box"]))
            if checked["candidate_detections"] and boxes:
                painter.setPen(_CANDIDATE_DETECTION_PEN)
                for box in boxes.get("others") or []:
                    painter.drawRect(to_display(box["box"]))
            if checked["selected_detection"] and boxes and boxes.get("selected"):
                painter.setPen(_SELECTED_DETECTION_PEN)
                painter.drawRect(to_display(boxes["selected"]["box"]))
            if checked["rejected_detections"] and eye and eye.get("box") and not eye.get("accepted"):
                painter.setPen(_REJECTED_DETECTION_PEN)
                painter.drawRect(to_display(eye["box"]))
            if checked["expanded_crop"] and boxes and boxes.get("expanded_box"):
                painter.setPen(_EXPANDED_CROP_PEN)
                painter.drawRect(to_display(boxes["expanded_box"]))
            if checked["final_crop"] and boxes and boxes.get("expanded_box"):
                painter.setPen(_FINAL_CROP_PEN)
                painter.drawRect(to_display(boxes["expanded_box"]))
            if checked["eye_roi"] and eye and eye.get("box"):
                colour = EYE_BOX_ACCEPTED if eye.get("accepted") else EYE_BOX_REJECTED
                pen = QPen(QColor(*colour), 2)
                if not eye.get("accepted"):
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(to_display(eye["box"]))
        finally:
            painter.end()
        return composed

    def _draw_landmark_family(self, display: QPixmap, eye: dict | None, checked: dict[str, bool]) -> QPixmap:
        if not eye or not eye.get("source_size"):
            return display
        source_size = eye["source_size"]
        scale_x = display.width() / source_size[0]
        scale_y = display.height() / source_size[1]

        composed = QPixmap(display)
        painter = QPainter(composed)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = 4.0
        try:
            if checked["head_box"]:
                points = [eye.get(key) for key in _HEAD_BOX_LANDMARK_KEYS]
                points = [point for point in points if point]
                if points:
                    xs = [point["x"] * scale_x for point in points]
                    ys = [point["y"] * scale_y for point in points]
                    painter.setPen(_HEAD_BOX_PEN)
                    painter.drawRect(QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
            if checked["landmarks"]:
                for name, key, colour in _LANDMARK_STYLE:
                    point = eye.get(key)
                    if not point:
                        continue
                    x, y = point["x"] * scale_x, point["y"] * scale_y
                    painter.setPen(QPen(QColor(*colour), 2))
                    painter.setBrush(QColor(*colour))
                    painter.drawEllipse(QRectF(x - radius, y - radius, radius * 2, radius * 2))
                    label_parts = []
                    if checked["landmark_labels"]:
                        label_parts.append(name)
                    if checked["confidence_values"]:
                        label_parts.append(f"{point['confidence']:.2f}")
                    if label_parts:
                        painter.drawText(int(x) + 6, int(y) - 6, " ".join(label_parts))
        finally:
            painter.end()
        return composed

    def _fill_landmark_table(self, eye: dict | None) -> None:
        """Name / Confidence / Pixel Coordinates / Normalized Coordinates for
        every landmark that is actually present - "if available" (Manual QA
        Issue 3's own wording): a landmark this image's detector never
        computed simply does not get a row, rather than a fabricated
        placeholder one."""
        rows: list[tuple[str, str, str, str]] = []
        source_size = (eye or {}).get("source_size")
        if eye and source_size and source_size[0] and source_size[1]:
            width, height = source_size
            for name, key, _colour in _LANDMARK_STYLE:
                point = eye.get(key)
                if not point:
                    continue
                rows.append((
                    name, f"{point['confidence']:.3f}",
                    f"({point['x']:.1f}, {point['y']:.1f})",
                    f"({point['x'] / width:.3f}, {point['y'] / height:.3f})",
                ))
        table = self._landmarks_table
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Landmark", "Confidence", "Pixel Coordinates", "Normalized Coordinates"])
        for row_index, values in enumerate(rows):
            for col_index, value in enumerate(values):
                table.setItem(row_index, col_index, QTableWidgetItem(value))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        table.setSortingEnabled(True)

    @staticmethod
    def _original_pixmap_path(image_path: str) -> Path | None:
        try:
            from ...review.thumbnails import review_preview
            return review_preview(image_path)
        except Exception:  # noqa: BLE001 - a missing/unreadable source image must not crash the dashboard
            return None

    @staticmethod
    def _load_full_pixmap(path: Path | None) -> QPixmap:
        """Unscaled, so overlay coordinates (in the source frame's own pixel
        space) can be mapped using a single width/height ratio computed
        against the ACTUAL displayed size - both Original and Crop are
        loaded this way and scaled down in `_refresh_overlays` right before
        display, rather than guessing what a pre-scaled pixmap's own
        KeepAspectRatio scale happened to produce."""
        if path is None or not Path(path).is_file():
            return QPixmap()
        return QPixmap(str(path))


_GEOMETRY_SETTINGS_KEY = "analytics/dashboard_geometry"


class AnalyticsDashboard(QDialog):
    """The Experiment Browser (left) plus an Experiment Metadata header and
    User vs Algorithm / Run Summary / Species Analysis / Burst Analytics /
    Image Explorer (right, as tabs) - the priority order specified for this
    phase."""

    def __init__(
        self,
        *,
        analytics_db: str | Path = DEFAULT_ANALYTICS_DB,
        annotations_db: str | Path = DEFAULT_ANNOTATIONS_DB,
        crop_cache_dir: str | Path = DEFAULT_CROP_CACHE_DIR,
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
        # The live Review context this dashboard was opened with - never
        # re-read afterward (MainWindow constructs a fresh dashboard each
        # time it is opened, see _show_analytics_dashboard), so these stay
        # exactly as accurate as "when I clicked Analytics Dashboard".
        self._root_folder = root_folder
        self._color_source = color_source
        self._keep_percent = keep_percent

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
        self._user_vs_algorithm_tab = UserVsAlgorithmTab(self)
        self._user_vs_algorithm_tab.drillDownRequested.connect(self._on_drill_down)
        self._run_summary_tab = RunSummaryTab(self)
        self._species_analysis_tab = SpeciesAnalysisTab(self)
        self._species_analysis_tab.speciesDrillDownRequested.connect(self._on_species_drill_down)
        self._burst_analytics_tab = BurstAnalyticsTab(annotation_store=self._annotation_store, parent=self)
        self._image_explorer_tab = ImageExplorerTab(
            crop_cache_dir=Path(crop_cache_dir), annotation_store=self._annotation_store,
            species_db=species_db, parent=self,
        )

        self._tabs = QTabWidget(self)
        # User vs Algorithm first - "the primary dashboard page" (see this
        # module's own Phase 2 mandate): the purpose of PickLikeMe is
        # agreement with the photographer, not maximizing a score.
        self._tabs.addTab(self._user_vs_algorithm_tab, "User vs Algorithm")
        self._tabs.addTab(self._run_summary_tab, "Run Summary")
        self._tabs.addTab(self._species_analysis_tab, "Species Analysis")
        self._tabs.addTab(self._burst_analytics_tab, "Burst Analytics")
        self._tabs.addTab(self._image_explorer_tab, "Image Explorer")

        detail_column = QVBoxLayout()
        detail_column.addWidget(self._metadata_panel)
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
        self._header_panel.show_run(self._store, self._annotation_store, run_id)
        self._metadata_panel.show_run(self._store, run_id)
        self._user_vs_algorithm_tab.show_run(self._store, self._annotation_store, run_id)
        self._run_summary_tab.show_run(self._store, run_id)
        self._species_analysis_tab.show_run(self._store, run_id)
        self._burst_analytics_tab.show_run(self._store, run_id)
        self._image_explorer_tab.show_run(self._store, run_id)

    def _on_drill_down(self, paths: list, label: str) -> None:
        """A User vs Algorithm confusion-matrix cell was clicked - filter
        the Image Explorer to exactly those images and switch to it, so
        "drilling down into every category" (see the Phase 2 mandate)
        actually shows the photographer the images, not just a count."""
        run_id = self._user_vs_algorithm_tab._run_id
        if run_id is None:
            return
        self._image_explorer_tab.show_paths(self._store, run_id, paths)
        self._tabs.setCurrentWidget(self._image_explorer_tab)
        self._image_explorer_tab.setToolTip(f"Filtered: {label} ({len(paths)} image(s))")

    def _on_species_drill_down(self, species: str) -> None:
        """Species Analysis's distribution table (Phase 9) - filters the
        Image Explorer to every image this run recorded under exactly that
        species/category, via its own species filter combo, rather than a
        separate paths-based drill-down: the Explorer already has every
        image for this run loaded (show_run, not show_paths), so narrowing
        it in place preserves every OTHER filter the photographer may
        already have set."""
        current = self._experiment_list.currentItem()
        run_id = current.data(Qt.ItemDataRole.UserRole) if current else None
        if run_id is None:
            return
        self._tabs.setCurrentWidget(self._image_explorer_tab)
        if self._image_explorer_tab._run_id != run_id:
            self._image_explorer_tab.show_run(self._store, run_id)
        self._image_explorer_tab._species_combo.setCurrentText(species)

    def refresh_current_run(self) -> None:
        """Re-runs whichever experiment is currently selected through every
        tab again - the explicit "immediately refresh Agreement/Confusion
        Matrix/Precision/Recall/F1" requirement after a Ground Truth import
        changes review decisions out from under an already-open dashboard."""
        self._on_experiment_selected(self._experiment_list.currentItem(), None)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._settings.setValue(_GEOMETRY_SETTINGS_KEY, self.saveGeometry())
        self._store.close()
        self._annotation_store.close()
        super().closeEvent(event)
