"""The Diagnostics & Analytics Dashboard.

Phase 2's own mandate: the goal is no longer displaying statistics, it is
understanding the algorithm - "why did it decide this", "where does it
disagree with the photographer", "what should improve next". User vs
Algorithm (see UserVsAlgorithmTab) is the primary page for that reason -
agreement with the photographer, not the score itself, is the actual
measure of success - shown first, ahead of Run Summary, Species Analysis
and Image Inspector. Implemented as one master-detail dialog - an
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

The Image Inspector's per-image detail panel lists whatever
`store.image_metrics(run_id, image_path)` returns as generic rows, rather
than a fixed "top1_confidence, top2_confidence, ..." layout - so EyePose
metrics (eye_confidence, head_confidence, ...) recorded by a future run
appear here with no UI change, per the explicit "design accordingly"
instruction.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
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
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...analyzer.annotations import DEFAULT_ANNOTATIONS_DB, AnnotationStore
from ...analytics.agreement import AgreementReport, compare_run_to_user_decisions
from ...analytics.reports import metric_statistics, run_statistics
from ...analytics.store import DEFAULT_ANALYTICS_DB, AnalyticsStore
from ...bird_crop import crop_cache_path
from ...config import DEFAULT_CROP_CACHE_DIR

_THUMBNAIL_SIZE = 320

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
    experiment went" the dashboard is meant to answer before any table."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("summaryCard")
        self.setStyleSheet(
            "#summaryCard { border: 1px solid palette(mid); border-radius: 6px; background-color: palette(base); }"
        )
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


class ExperimentMetadataPanel(QWidget):
    """"The user should immediately know exactly how the experiment was
    produced" - shown above the tabs (not buried inside one of them) so it
    stays visible regardless of which tab is open. Curated, not exhaustive -
    RunSummaryTab's own generic table already dumps every recorded param;
    this shows only the well-known "what produced this" fields, in a fixed
    reading order, and only the ones this particular run actually has (see
    _METADATA_PARAM_FIELDS)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(150)
        _style_table(self._table)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Experiment Metadata", self))
        layout.addWidget(self._table)

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
    exact list of image paths behind whichever confusion-matrix cell was
    clicked, so it can filter the Image Inspector to just that category -
    "false positives", not "every image this run touched".
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

        self._user_keep_card = SummaryCard("User Keep", self)
        self._user_reject_card = SummaryCard("User Reject", self)
        self._algo_keep_card = SummaryCard("Algorithm Keep", self)
        self._algo_reject_card = SummaryCard("Algorithm Reject", self)
        self._agree_card = SummaryCard("Agreement %", self)
        self._disagree_card = SummaryCard("Disagreement %", self)
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
        self._matrix_table.setToolTip("Click a cell to see those images in the Image Inspector")

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


class RunSummaryTab(QWidget):
    """Everything "how did this experiment go, in numbers" for one selected
    experiment: glanceable summary cards, per-metric statistics (mean/
    median/min/max, for whatever this run actually recorded), and every
    recorded field/param in full."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._accepted_card = SummaryCard("Accepted", self)
        self._rejected_card = SummaryCard("Rejected", self)
        self._acceptance_card = SummaryCard("Acceptance %", self)
        self._score_card = SummaryCard("Avg Score", self)
        cards_row = QHBoxLayout()
        for card in (self._accepted_card, self._rejected_card, self._acceptance_card, self._score_card):
            cards_row.addWidget(card)

        self._metric_stats_table = QTableWidget(self)
        self._metric_stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._metric_stats_table)

        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._table)

        layout = QVBoxLayout(self)
        layout.addLayout(cards_row)
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
        self._accepted_card.set_value(str(accepted))
        self._rejected_card.set_value(str(rejected))
        self._acceptance_card.set_value(f"{100.0 * accepted / considered:.1f}%" if considered else "n/a")
        score_stats = metric_statistics(store, run_id, "score")
        self._score_card.set_value(f"{score_stats['mean']:.4f}" if score_stats else "n/a")

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
    own docstring."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel(self)
        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._table)
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._table)

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        run = store.get_run(run_id) or {}
        considered = run.get("considered", 0) or 0
        accepted = run.get("accepted", 0) or 0
        counts = store.category_counts(run_id)
        summary = store.summary_metrics(run_id)

        unknown_rate = summary.get("unknown_rate")
        label_text = f"{len(counts)} distinct outcome(s) across {considered} image(s)"
        if unknown_rate is not None:
            label_text += f"  |  Unknown rate: {unknown_rate:.1%}"
        self._label.setText(label_text)

        self._table.setSortingEnabled(False)
        # "Accepted" is not itself a row in category_counts (it is the
        # complement of every reject/category count, tracked separately on
        # the run itself) - shown first so the table reads as a complete
        # outcome breakdown, not only the rejected/categorized slice.
        rows = [("Accepted", accepted)] + sorted(counts.items(), key=lambda kv: -kv[1])
        self._table.setRowCount(len(rows))
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Species / Category", "Count", "% of considered"])
        for row_index, (name, count) in enumerate(rows):
            percent = f"{100.0 * count / considered:.1f}%" if considered else "n/a"
            self._table.setItem(row_index, 0, QTableWidgetItem(name))
            item_count = QTableWidgetItem()
            item_count.setData(Qt.ItemDataRole.DisplayRole, count)
            self._table.setItem(row_index, 1, item_count)
            self._table.setItem(row_index, 2, QTableWidgetItem(percent))
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSortingEnabled(True)


class ImageInspectorTab(QWidget):
    """For every classified image: original, crop (best-effort - not every
    backend uses one, see docs/Species_Classification_Investigation.md),
    every recorded metric generically, backend, experiment ID, runtime."""

    def __init__(self, *, crop_cache_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._crop_cache_dir = crop_cache_dir
        self._store: AnalyticsStore | None = None
        self._run_id: str | None = None

        self._image_list = QListWidget(self)
        self._image_list.setAlternatingRowColors(True)
        self._image_list.currentItemChanged.connect(self._on_image_selected)

        self._original_label = QLabel("Original", self)
        self._original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._original_label.setMinimumSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)
        self._crop_label = QLabel("Crop", self)
        self._crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_label.setMinimumSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)

        images_row = QHBoxLayout()
        images_row.addWidget(self._original_label)
        images_row.addWidget(self._crop_label)

        self._metrics_table = QTableWidget(self)
        self._metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _style_table(self._metrics_table)

        detail_column = QVBoxLayout()
        detail_column.addLayout(images_row)
        detail_column.addWidget(self._metrics_table)

        splitter = QSplitter(self)
        list_container = QWidget(self)
        list_layout = QVBoxLayout(list_container)
        list_layout.addWidget(QLabel("Images", self))
        list_layout.addWidget(self._image_list)
        splitter.addWidget(list_container)
        detail_container = QWidget(self)
        detail_container.setLayout(detail_column)
        splitter.addWidget(detail_container)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        self.show_paths(store, run_id, store.image_paths(run_id))

    def show_paths(self, store: AnalyticsStore, run_id: str, paths: list[str]) -> None:
        """Populate the image list from exactly `paths`, not every image
        this run recorded - the drill-down target for User vs Algorithm's
        confusion-matrix cells ("show me only the false positives"), and
        for any other future caller that wants a filtered view rather than
        the whole run."""
        self._store = store
        self._run_id = run_id
        self._image_list.clear()
        for image_path in paths:
            item = QListWidgetItem(Path(image_path).name)
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self._image_list.addItem(item)
        if self._image_list.count():
            self._image_list.setCurrentRow(0)
        else:
            self._original_label.setText("(no images in this selection)")
            self._crop_label.setText("")
            self._metrics_table.setRowCount(0)

    def _on_image_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None or self._store is None or self._run_id is None:
            return
        image_path = current.data(Qt.ItemDataRole.UserRole)
        run = self._store.get_run(self._run_id) or {}

        original_pixmap = self._load_pixmap(self._original_pixmap_path(image_path))
        if original_pixmap.isNull():
            # Most often: this run also moved the file (Organize by Species
            # relocates every image it classifies) - the path recorded at
            # classification time no longer points at anything, a known
            # limitation, not a crash - see docs/BioCLIP_Backend_
            # Infrastructure_Deliverables.md's "remaining technical debt".
            self._original_label.setText(f"Original not available\n({Path(image_path).name} may have moved)")
        else:
            self._original_label.setPixmap(original_pixmap)
        crop_path = crop_cache_path(self._crop_cache_dir, image_path)
        if crop_path.is_file():
            self._crop_label.setPixmap(self._load_pixmap(crop_path))
        else:
            self._crop_label.setText("No crop cached for this image\n(this backend may classify the full frame)")

        metrics = self._store.image_metrics(self._run_id, image_path)
        rows = [
            ("Image", image_path),
            ("Experiment ID", self._run_id),
            ("Backend", run.get("strategy_id", "")),
        ]
        for name, value in sorted(metrics.items()):
            rows.append((name, f"{value:.4f}"))
        _fill_two_column_table(self._metrics_table, rows)

    @staticmethod
    def _original_pixmap_path(image_path: str) -> Path | None:
        try:
            from ...review.thumbnails import review_preview
            return review_preview(image_path)
        except Exception:  # noqa: BLE001 - a missing/unreadable source image must not crash the dashboard
            return None

    @staticmethod
    def _load_pixmap(path: Path | None) -> QPixmap:
        if path is None or not Path(path).is_file():
            return QPixmap()
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return pixmap
        return pixmap.scaled(
            _THUMBNAIL_SIZE, _THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )


_GEOMETRY_SETTINGS_KEY = "analytics/dashboard_geometry"


class AnalyticsDashboard(QDialog):
    """The Experiment Browser (left) plus an Experiment Metadata header and
    Run Summary / Species Analysis / Image Inspector (right, as tabs) - the
    priority order specified for this phase."""

    def __init__(
        self,
        *,
        analytics_db: str | Path = DEFAULT_ANALYTICS_DB,
        annotations_db: str | Path = DEFAULT_ANNOTATIONS_DB,
        crop_cache_dir: str | Path = DEFAULT_CROP_CACHE_DIR,
        settings: QSettings | None = None,
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

        self._experiment_search = QLineEdit(self)
        self._experiment_search.setPlaceholderText("Search experiments (folder or algorithm)...")
        self._experiment_search.textChanged.connect(self._apply_experiment_filter)

        self._experiment_list = QListWidget(self)
        self._experiment_list.setAlternatingRowColors(True)
        self._experiment_list.currentItemChanged.connect(self._on_experiment_selected)

        refresh_button = QPushButton("Refresh", self)
        refresh_button.clicked.connect(self._refresh_experiment_list)

        browser_column = QVBoxLayout()
        browser_column.addWidget(QLabel("Experiments (most recent first)", self))
        browser_column.addWidget(self._experiment_search)
        browser_column.addWidget(self._experiment_list)
        browser_column.addWidget(refresh_button)
        browser_container = QWidget(self)
        browser_container.setLayout(browser_column)

        self._metadata_panel = ExperimentMetadataPanel(self)
        self._user_vs_algorithm_tab = UserVsAlgorithmTab(self)
        self._user_vs_algorithm_tab.drillDownRequested.connect(self._on_drill_down)
        self._run_summary_tab = RunSummaryTab(self)
        self._species_analysis_tab = SpeciesAnalysisTab(self)
        self._image_inspector_tab = ImageInspectorTab(crop_cache_dir=Path(crop_cache_dir), parent=self)

        self._tabs = QTabWidget(self)
        # User vs Algorithm first - "the primary dashboard page" (see this
        # module's own Phase 2 mandate): the purpose of PickLikeMe is
        # agreement with the photographer, not maximizing a score.
        self._tabs.addTab(self._user_vs_algorithm_tab, "User vs Algorithm")
        self._tabs.addTab(self._run_summary_tab, "Run Summary")
        self._tabs.addTab(self._species_analysis_tab, "Species Analysis")
        self._tabs.addTab(self._image_inspector_tab, "Image Inspector")

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
        layout.addWidget(splitter)

        self._refresh_experiment_list()
        if self._experiment_list.count():
            self._experiment_list.setCurrentRow(0)

    def _update_empty_state(self) -> None:
        has_experiments = self._experiment_list.count() > 0
        self._detail_stack.setCurrentIndex(0 if has_experiments else 1)

    def _refresh_experiment_list(self) -> None:
        # Re-selects whatever run was showing before the rebuild (see
        # refresh_current_run's own docstring) - clicking Refresh after a
        # Ground Truth import must not leave the detail tabs holding stale
        # data just because the experiment LIST itself did not change.
        previously_selected = self._experiment_list.currentItem()
        previous_run_id = previously_selected.data(Qt.ItemDataRole.UserRole) if previously_selected else None

        self._experiment_list.clear()
        restore_row = -1
        for row, run in enumerate(self._store.list_runs()):
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
        self._metadata_panel.show_run(self._store, run_id)
        self._user_vs_algorithm_tab.show_run(self._store, self._annotation_store, run_id)
        self._run_summary_tab.show_run(self._store, run_id)
        self._species_analysis_tab.show_run(self._store, run_id)
        self._image_inspector_tab.show_run(self._store, run_id)

    def _on_drill_down(self, paths: list, label: str) -> None:
        """A User vs Algorithm confusion-matrix cell was clicked - filter
        the Image Inspector to exactly those images and switch to it, so
        "drilling down into every category" (see the Phase 2 mandate)
        actually shows the photographer the images, not just a count."""
        run_id = self._user_vs_algorithm_tab._run_id
        if run_id is None:
            return
        self._image_inspector_tab.show_paths(self._store, run_id, paths)
        self._tabs.setCurrentWidget(self._image_inspector_tab)
        self._image_inspector_tab.setToolTip(f"Filtered: {label} ({len(paths)} image(s))")

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
