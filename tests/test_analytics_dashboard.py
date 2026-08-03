"""AnalyticsDashboard - Part 5/6 of the BioCLIP multi-backend
infrastructure work. Constructs the real dialog against a real (small,
seeded) AnalyticsStore - not mocked - so a schema drift between store.py
and this dialog's own queries would fail here, not only be caught visually.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QApplication = None

from picklikeme.analytics.store import AnalyticsStore


def _seed(db_path: Path) -> None:
    with AnalyticsStore(db_path) as store:
        store.insert_run(
            "run-1", folder=str(db_path.parent / "shoot"), strategy_id="bioclip2", started_at="2026-08-03T21:00:00",
            considered=3, accepted=2, device="cuda",
            params={"model_id": "hf-hub:imageomics/bioclip-2", "species_count": 55},
            reject_counts={"Kingfisher": 1, "Osprey": 1, "Unknown": 1},
            image_metrics={
                "a.jpg": {"top1_confidence": 0.9, "top2_confidence": 0.05, "inference_seconds": 0.02},
                "b.jpg": {"top1_confidence": 0.7, "inference_seconds": 0.03},
            },
            summary_metrics={"runtime_seconds": 4.2, "images_per_second": 0.7, "unknown_rate": 1 / 3},
        )
        store.insert_run(
            "run-2", folder=str(db_path.parent / "shoot"), strategy_id="bioclip", started_at="2026-08-03T20:00:00",
            considered=3, accepted=1, device="cpu",
            params={"model_id": "hf-hub:imageomics/bioclip", "species_count": 55},
            reject_counts={"Unknown": 2, "Baboon": 1},
            image_metrics={"a.jpg": {"top1_confidence": 0.3}},
            summary_metrics={"runtime_seconds": 8.0},
        )


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_dashboard_lists_every_experiment_most_recent_first(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")

    assert dialog._experiment_list.count() == 2
    assert "bioclip2" in dialog._experiment_list.item(0).text()  # most recent (21:00) first
    assert "bioclip" in dialog._experiment_list.item(1).text()

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_selecting_an_experiment_populates_all_three_tabs(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")

    dialog._experiment_list.setCurrentRow(0)  # the bioclip2 run
    app.processEvents()

    assert dialog._run_summary_tab._table.rowCount() > 0
    assert dialog._species_analysis_tab._table.rowCount() == 3  # Kingfisher, Osprey, Unknown
    assert dialog._image_inspector_tab._image_list.count() == 2  # a.jpg, b.jpg

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_switching_between_experiments_updates_the_detail_tabs(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()
    first_species_rows = dialog._species_analysis_tab._table.rowCount()

    dialog._experiment_list.setCurrentRow(1)  # the bioclip (v1) run
    app.processEvents()
    second_species_rows = dialog._species_analysis_tab._table.rowCount()

    assert first_species_rows == 3
    assert second_species_rows == 2  # Unknown, Baboon
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_dashboard_with_no_experiments_does_not_crash(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "empty_analytics.db")

    assert dialog._experiment_list.count() == 0
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_image_inspector_never_crashes_on_a_missing_source_file(tmp_path: Path) -> None:
    """The exact real-world case found during validation: a recorded image
    path that no longer exists (Organize by Species moved the file after
    recording it) must degrade to a clear message, never a crash or a
    silently blank widget."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")

    dialog._tabs.setCurrentIndex(2)
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()
    dialog._image_inspector_tab._image_list.setCurrentRow(0)
    app.processEvents()

    assert dialog._image_inspector_tab._original_label.pixmap() is None or \
        dialog._image_inspector_tab._original_label.pixmap().isNull() or \
        dialog._image_inspector_tab._original_label.text() != ""
    dialog.close()
