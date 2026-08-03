"""The Diagnostics & Analytics Dashboard - Part 5/6 of the BioCLIP
multi-backend infrastructure work (docs/Analytics_Dashboard_Plan.md Phase
1's Desktop surface).

Priority order, as specified: Run Summary, Experiment Browser, Species
Analysis, Image Inspector. Implemented as one master-detail dialog - an
Experiment Browser (a list) on the left, and the other three views as tabs
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

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...analytics.reports import run_statistics
from ...analytics.store import DEFAULT_ANALYTICS_DB, AnalyticsStore
from ...bird_crop import crop_cache_path
from ...config import DEFAULT_CROP_CACHE_DIR

_THUMBNAIL_SIZE = 320


def _fill_two_column_table(table: QTableWidget, rows: list[tuple[str, str]]) -> None:
    table.setRowCount(len(rows))
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(["Field", "Value"])
    for row_index, (field, value) in enumerate(rows):
        table.setItem(row_index, 0, QTableWidgetItem(field))
        table.setItem(row_index, 1, QTableWidgetItem(value))
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)


class RunSummaryTab(QWidget):
    """All recorded metadata, runtime, classifier, species list,
    configuration, statistics - for one selected experiment."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout = QVBoxLayout(self)
        layout.addWidget(self._table)

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        stats = run_statistics(store, run_id)
        run = store.get_run(run_id) or {}
        summary = store.summary_metrics(run_id)

        rows: list[tuple[str, str]] = [
            ("Experiment ID", run_id),
            ("Folder", stats.get("folder", "")),
            ("Backend / Strategy", stats.get("strategy_id", "")),
            ("Started at", stats.get("started_at", "")),
            ("Device", stats.get("device") or "n/a"),
            ("Considered", str(stats.get("considered", 0))),
            ("Accepted", str(stats.get("accepted", 0))),
            ("Rejected", str(stats.get("rejected", 0))),
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
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._table)

    def show_run(self, store: AnalyticsStore, run_id: str) -> None:
        run = store.get_run(run_id) or {}
        considered = run.get("considered", 0) or 0
        counts = store.category_counts(run_id)
        summary = store.summary_metrics(run_id)

        unknown_rate = summary.get("unknown_rate")
        label_text = f"{len(counts)} distinct outcome(s) across {considered} image(s)"
        if unknown_rate is not None:
            label_text += f"  |  Unknown rate: {unknown_rate:.1%}"
        self._label.setText(label_text)

        self._table.setRowCount(len(counts))
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Species / Category", "Count", "% of considered"])
        for row_index, (name, count) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])):
            percent = f"{100.0 * count / considered:.1f}%" if considered else "n/a"
            self._table.setItem(row_index, 0, QTableWidgetItem(name))
            self._table.setItem(row_index, 1, QTableWidgetItem(str(count)))
            self._table.setItem(row_index, 2, QTableWidgetItem(percent))
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)


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
        self._store = store
        self._run_id = run_id
        self._image_list.clear()
        for image_path in store.image_paths(run_id):
            item = QListWidgetItem(Path(image_path).name)
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self._image_list.addItem(item)
        if self._image_list.count():
            self._image_list.setCurrentRow(0)
        else:
            self._original_label.setText("(no per-image data recorded for this run)")
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


class AnalyticsDashboard(QDialog):
    """The Experiment Browser (left) plus Run Summary / Species Analysis /
    Image Inspector (right, as tabs) - the priority order specified for
    this phase."""

    def __init__(
        self,
        *,
        analytics_db: str | Path = DEFAULT_ANALYTICS_DB,
        crop_cache_dir: str | Path = DEFAULT_CROP_CACHE_DIR,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PeakPic - Analytics Dashboard")
        self.resize(1100, 700)

        self._store = AnalyticsStore(analytics_db)

        self._experiment_list = QListWidget(self)
        self._experiment_list.currentItemChanged.connect(self._on_experiment_selected)

        refresh_button = QPushButton("Refresh", self)
        refresh_button.clicked.connect(self._refresh_experiment_list)

        browser_column = QVBoxLayout()
        browser_column.addWidget(QLabel("Experiments (most recent first)", self))
        browser_column.addWidget(self._experiment_list)
        browser_column.addWidget(refresh_button)
        browser_container = QWidget(self)
        browser_container.setLayout(browser_column)

        self._run_summary_tab = RunSummaryTab(self)
        self._species_analysis_tab = SpeciesAnalysisTab(self)
        self._image_inspector_tab = ImageInspectorTab(crop_cache_dir=Path(crop_cache_dir), parent=self)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self._run_summary_tab, "Run Summary")
        self._tabs.addTab(self._species_analysis_tab, "Species Analysis")
        self._tabs.addTab(self._image_inspector_tab, "Image Inspector")

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
        self._detail_stack.addWidget(self._tabs)
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
        self._detail_stack.setCurrentWidget(self._tabs if has_experiments else self._empty_state_label)

    def _refresh_experiment_list(self) -> None:
        self._experiment_list.clear()
        for run in self._store.list_runs():
            label = f"{run['started_at']}  |  {run['strategy_id']}  |  {Path(run['folder']).name}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, run["run_id"])
            self._experiment_list.addItem(item)
        self._update_empty_state()

    def _on_experiment_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            return
        run_id = current.data(Qt.ItemDataRole.UserRole)
        self._run_summary_tab.show_run(self._store, run_id)
        self._species_analysis_tab.show_run(self._store, run_id)
        self._image_inspector_tab.show_run(self._store, run_id)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._store.close()
        super().closeEvent(event)
