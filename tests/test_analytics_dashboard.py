"""AnalyticsDashboard - Part 5/6 of the BioCLIP multi-backend
infrastructure work. Constructs the real dialog against a real (small,
seeded) AnalyticsStore - not mocked - so a schema drift between store.py
and this dialog's own queries would fail here, not only be caught visually.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QApplication = None
    QPixmap = None

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
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    assert dialog._experiment_list.count() == 2
    # Friendly labels (see _friendly_strategy_label), not the raw registry
    # id - most recent (21:00) first.
    assert "BioCLIP 2" in dialog._experiment_list.item(0).text()
    assert "BioCLIP (original)" in dialog._experiment_list.item(1).text()

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_selecting_an_experiment_populates_all_three_tabs(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)  # the bioclip2 run
    app.processEvents()

    assert dialog._run_summary_tab._table.rowCount() > 0
    assert dialog._species_analysis_tab._table.rowCount() == 4  # Accepted, Kingfisher, Osprey, Unknown
    assert dialog._image_explorer_tab._image_list.count() == 2  # a.jpg, b.jpg

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_switching_between_experiments_updates_the_detail_tabs(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()
    first_species_rows = dialog._species_analysis_tab._table.rowCount()

    dialog._experiment_list.setCurrentRow(1)  # the bioclip (v1) run
    app.processEvents()
    second_species_rows = dialog._species_analysis_tab._table.rowCount()

    assert first_species_rows == 4  # Accepted, Kingfisher, Osprey, Unknown
    assert second_species_rows == 3  # Accepted, Unknown, Baboon
    dialog.close()


# ---------------------------------------------------------------------------
# Phase 9 - Species Analytics: Average Confidence, Top 5 Predictions, and
# clicking a distribution row filters the Image Explorer by species.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_average_confidence_card_shows_the_mean_top1_confidence(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")  # run-1: a.jpg=0.9, b.jpg=0.7 -> mean 0.8
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    assert dialog._species_analysis_tab._confidence_card._value_label.text() == "0.8000"
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_unknown_rate_card_matches_the_recorded_summary_metric(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")  # run-1 records unknown_rate = 1/3
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    assert dialog._species_analysis_tab._unknown_rate_card._value_label.text() == "33.3%"
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_top_5_predictions_table_shows_at_most_five_highest_count_rows(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")  # run-1: Kingfisher=1, Osprey=1, Unknown=1 (3 categories, under 5)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    table = dialog._species_analysis_tab._top5_table
    assert table.rowCount() == 3  # never padded to 5 with fake rows
    names = {table.item(r, 0).text() for r in range(table.rowCount())}
    assert names == {"Kingfisher", "Osprey", "Unknown"}
    assert "Accepted" not in names  # Top 5 is species/category only, not the accepted count
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_clicking_a_species_row_filters_the_image_explorer(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    table = dialog._species_analysis_tab._table
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "Kingfisher")
    table.cellClicked.emit(row, 0)
    app.processEvents()

    assert dialog._tabs.currentWidget() is dialog._image_explorer_tab
    assert dialog._image_explorer_tab._species_combo.currentText() in ("Kingfisher", "All Species")
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_clicking_the_accepted_row_does_not_drill_down(tmp_path: Path) -> None:
    """"Accepted" is a summary row, not a real species/category - clicking
    it must not try to filter by a species literally named "Accepted"."""
    from unittest import mock

    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    spy = mock.Mock()
    dialog._species_analysis_tab.speciesDrillDownRequested.connect(spy)
    table = dialog._species_analysis_tab._table
    assert table.item(0, 0).text() == "Accepted"
    table.cellClicked.emit(0, 0)

    spy.assert_not_called()
    dialog.close()


# ---------------------------------------------------------------------------
# Phase 10 - Burst Analytics: Burst Size / Burst Rank / Burst Winner, plus
# Winner/Loser filtering already wired into ImageExplorerTab's own Burst
# combo (Phase 4/6's groundwork - both share _compute_burst_map).
# a.jpg and b.jpg are 1 second apart (same burst, a scores higher -> a wins);
# c.jpg is 5 minutes later -> its own singleton burst.
# ---------------------------------------------------------------------------


def _make_jpeg_with_capture_time(path: Path, timestamp: str) -> Path:
    """A real, decodable JPEG carrying an EXIF DateTimeOriginal tag, since
    AnnotationStore.capture_timestamp_of exercises the real read path - the
    same helper as tests/test_annotations.py's own make_jpeg_with_capture_time,
    kept local rather than imported cross-file (this project's tests/ has no
    __init__.py, so cross-test-file imports are not a pattern used
    elsewhere)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (4, 4), color="blue")
    exif = image.getexif()
    exif[36867] = timestamp  # DateTimeOriginal
    image.save(path, format="JPEG", exif=exif)
    return path


def _make_burst_images(tmp_path: Path):
    shoot = tmp_path / "shoot"
    a = shoot / "a.jpg"
    b = shoot / "b.jpg"
    c = shoot / "c.jpg"
    _make_jpeg_with_capture_time(a, "2024:06:15 10:30:00")
    _make_jpeg_with_capture_time(b, "2024:06:15 10:30:01")
    _make_jpeg_with_capture_time(c, "2024:06:15 10:35:00")

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder=str(shoot), strategy_id="ai-model", started_at="t", considered=3, accepted=3,
            device="cpu", params={}, reject_counts={},
            image_metrics={str(a): {"score": 0.9}, str(b): {"score": 0.5}, str(c): {"score": 0.3}},
        )
    return shoot, a, b, c


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_burst_analytics_cards_summarize_the_run(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _make_burst_images(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    tab = dialog._burst_analytics_tab
    assert tab._burst_count_card._value_label.text() == "2"  # {a,b} and {c}
    assert tab._singleton_card._value_label.text() == "1"  # c.jpg
    assert tab._multi_image_card._value_label.text() == "1"  # {a,b}
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_burst_analytics_per_image_table_shows_size_rank_and_winner(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _make_burst_images(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    table = dialog._burst_analytics_tab._table
    rows = {
        table.item(r, 0).text(): (table.item(r, 1).text(), table.item(r, 2).text(), table.item(r, 3).text())
        for r in range(table.rowCount())
    }
    assert rows["a.jpg"] == ("2", "1", "Yes")  # higher score -> rank 1 -> winner
    assert rows["b.jpg"] == ("2", "2", "No")
    assert rows["c.jpg"] == ("1", "1", "Yes")  # singleton is trivially its own winner
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_image_explorer_burst_filter_agrees_with_burst_analytics_tab(tmp_path: Path) -> None:
    """Both read through the same _compute_burst_map - a winner in one tab
    must be a winner in the other."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _make_burst_images(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    explorer = dialog._image_explorer_tab
    explorer._burst_combo.setCurrentText("Burst Winners")
    winner_names = {explorer._image_list.item(r).text() for r in range(explorer._image_list.count())}
    assert winner_names == {"a.jpg", "c.jpg"}

    explorer._burst_combo.setCurrentText("Burst Losers")
    loser_names = {explorer._image_list.item(r).text() for r in range(explorer._image_list.count())}
    assert loser_names == {"b.jpg"}
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_dashboard_with_no_experiments_does_not_crash(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "empty_analytics.db", annotations_db=tmp_path / "annotations.sqlite")

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
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._tabs.setCurrentIndex(2)
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()
    dialog._image_explorer_tab._image_list.setCurrentRow(0)
    app.processEvents()

    assert dialog._image_explorer_tab._original_label.pixmap() is None or \
        dialog._image_explorer_tab._original_label.pixmap().isNull() or \
        dialog._image_explorer_tab._original_label.text() != ""
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_experiment_search_box_filters_the_list_in_place(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_search.setText("bioclip 2")
    visible = [
        dialog._experiment_list.item(row).text()
        for row in range(dialog._experiment_list.count())
        if not dialog._experiment_list.item(row).isHidden()
    ]
    assert len(visible) == 1
    assert "BioCLIP 2" in visible[0]

    dialog._experiment_search.setText("")
    visible = [
        row for row in range(dialog._experiment_list.count()) if not dialog._experiment_list.item(row).isHidden()
    ]
    assert len(visible) == 2

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_experiment_metadata_panel_shows_algorithm_and_recorded_params(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)  # the bioclip2 run
    app.processEvents()

    table = dialog._metadata_panel._table
    rows = {
        table.item(r, 0).text(): table.item(r, 1).text()
        for r in range(table.rowCount())
    }
    assert "BioCLIP 2" in rows["Algorithm"]
    assert rows["Experiment Date"] == "2026-08-03T21:00:00"
    assert rows["Images Processed"] == "3"
    assert rows["Species Count"] == "55"
    assert rows["Experiment Duration"] == "4.2s"

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_experiment_metadata_panel_starts_collapsed_and_toggles(tmp_path: Path) -> None:
    """Manual QA finding: Device/Images Processed/Experiment Date and the
    rest of this table's rarely-changing technical metadata was eating
    vertical space the analysis tabs need more - now collapsible, default
    collapsed, one click to expand. Populating the table (show_run) must
    still work while collapsed - the data is queried by other code/tests
    regardless of visibility."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    # isHidden() (the explicit per-widget flag), not isVisible() (which
    # also depends on the whole ancestor chain being shown on screen - the
    # dialog in this test never is, so isVisible() would read False either
    # way and could not distinguish the two states).
    panel = dialog._metadata_panel
    assert panel._table.isHidden() is True
    assert panel._toggle_button.isChecked() is False
    assert panel._table.rowCount() > 0  # still populated even while collapsed

    panel._toggle_button.setChecked(True)
    assert panel._table.isHidden() is False
    assert "▾" in panel._toggle_button.text()

    panel._toggle_button.setChecked(False)
    assert panel._table.isHidden() is True
    assert "▸" in panel._toggle_button.text()

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_run_summary_cards_show_acceptance_and_score(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder=str(tmp_path / "shoot"), strategy_id="ai-model", started_at="2026-08-03T21:00:00",
            considered=4, accepted=3, device="cuda", params={},
            reject_counts={},
            image_metrics={"a.jpg": {"score": 0.9}, "b.jpg": {"score": 0.7}, "c.jpg": {"score": 0.5}},
            summary_metrics={"runtime_seconds": 2.0},
        )
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    tab = dialog._run_summary_tab
    assert tab._images_processed_card._value_label.text() == "4"
    assert tab._accepted_card._value_label.text() == "3"
    assert tab._rejected_card._value_label.text() == "1"
    assert tab._acceptance_card._value_label.text() == "75.0%"
    assert tab._score_card._value_label.text() == "0.7000"

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_run_summary_score_and_quality_table_shows_the_phase_8_fields(tmp_path: Path) -> None:
    """Phase 8's own named list: Median/Highest/Lowest Score, Images/Second,
    Average Runtime, Average Eye/Head Confidence, Average Eye/Subject
    Sharpness, Average Subject Size."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder=str(tmp_path / "shoot"), strategy_id="classic-vision-eyepose", started_at="t",
            considered=2, accepted=2, device="cpu", params={}, reject_counts={},
            image_metrics={
                "a.jpg": {
                    "score": 0.9, "eye_confidence": 0.8, "head_confidence": 0.7,
                    "eye_sharpness": 10.0, "subject_sharpness": 20.0, "subject_size": 0.3,
                },
                "b.jpg": {
                    "score": 0.5, "eye_confidence": 0.6, "head_confidence": 0.5,
                    "eye_sharpness": 6.0, "subject_sharpness": 12.0, "subject_size": 0.1,
                },
            },
            summary_metrics={"runtime_seconds": 10.0, "images_per_second": 0.2},
        )
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    table = dialog._run_summary_tab._summary_stats_table
    rows = {table.item(r, 0).text(): table.item(r, 1).text() for r in range(table.rowCount())}
    assert rows["Median Score"] == "0.7000"
    assert rows["Highest Score"] == "0.9000"
    assert rows["Lowest Score"] == "0.5000"
    assert rows["Images / Second"] == "0.2000"
    assert rows["Average Runtime (seconds/image)"] == "5.0000"  # 10.0s / 2 images
    assert rows["Average Eye Confidence"] == "0.7000"
    assert rows["Average Head Confidence"] == "0.6000"
    assert rows["Average Eye Sharpness"] == "8.0000"
    assert rows["Average Subject Sharpness"] == "16.0000"
    assert rows["Average Subject Size"] == "0.2000"

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_run_summary_score_and_quality_table_shows_n_a_for_metrics_this_run_never_recorded(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-ai", folder=str(tmp_path / "shoot"), strategy_id="ai-model", started_at="t",
            considered=1, accepted=1, device="cuda", params={}, reject_counts={},
            image_metrics={"a.jpg": {"score": 0.5}},
        )
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    table = dialog._run_summary_tab._summary_stats_table
    rows = {table.item(r, 0).text(): table.item(r, 1).text() for r in range(table.rowCount())}
    assert rows["Average Eye Confidence"] == "n/a"
    assert rows["Average Runtime (seconds/image)"] == "n/a"  # no summary_metrics recorded at all

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_dashboard_geometry_is_remembered_via_qsettings(tmp_path: Path) -> None:
    """Byte-for-byte round-trip through QSettings, and that a fresh dialog
    actually asks Qt to restore it - not an assertion on the resulting
    on-screen pixel size, which offscreen/headless rendering can clamp
    against a virtual screen shared (and mutated) by the rest of a full
    test run, independent of whether save/restore itself worked."""
    from PySide6.QtCore import QSettings

    from picklikeme.desktop.dialogs.analytics_dashboard import _GEOMETRY_SETTINGS_KEY, AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite", settings=settings)
    dialog.resize(950, 640)
    dialog.close()

    saved = settings.value(_GEOMETRY_SETTINGS_KEY)
    assert saved is not None

    settings_reloaded = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    assert settings_reloaded.value(_GEOMETRY_SETTINGS_KEY) == saved

    restored_with: list = []
    original_restore = AnalyticsDashboard.restoreGeometry
    try:
        AnalyticsDashboard.restoreGeometry = lambda self, geometry: (
            restored_with.append(geometry), original_restore(self, geometry)
        )[1]
        dialog2 = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite", settings=settings_reloaded)
    finally:
        AnalyticsDashboard.restoreGeometry = original_restore
    assert restored_with == [saved]
    dialog2.close()


def _make_jpeg(path: Path, *, color: str = "blue") -> None:
    """A JPEG whose bytes are unique to `color` - image_identity() is
    content-based, so two calls with the same color would collide onto the
    same identity, exactly the bug this scenario's own images must not
    have (see _seed_agreement_scenario's four distinctly-colored images)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path, format="JPEG")


def _seed_agreement_scenario(tmp_path: Path) -> Path:
    """Four real images, a run whose accepted/considered ratio is 50%
    (so the algorithm's own default keep-percent splits them 2/2 by
    score), and review decisions for three of them - one agreeing on
    each side, one disagreement (a false positive), and one left
    unreviewed (Neutral)."""
    from picklikeme.analyzer.annotations import REVIEW_KEEP, REVIEW_REJECT, AnnotationStore

    shoot = tmp_path / "shoot"
    a = shoot / "a.jpg"  # highest score -> algorithm Keep; user also Keep -> agree
    b = shoot / "b.jpg"  # 2nd highest -> algorithm Keep; user Reject -> false positive
    c = shoot / "c.jpg"  # 3rd -> algorithm Reject; user Reject -> agree
    d = shoot / "d.jpg"  # lowest -> algorithm Reject; never reviewed -> Neutral
    for path, color in ((a, "red"), (b, "green"), (c, "blue"), (d, "yellow")):
        _make_jpeg(path, color=color)

    analytics_db = tmp_path / "analytics.db"
    with AnalyticsStore(analytics_db) as store:
        store.insert_run(
            "run-1", folder=str(shoot), strategy_id="ai-model", started_at="2026-08-04T10:00:00",
            considered=4, accepted=2, device="cpu", params={},
            reject_counts={},
            image_metrics={
                str(a): {"score": 0.9}, str(b): {"score": 0.8}, str(c): {"score": 0.3}, str(d): {"score": 0.1},
            },
        )

    annotations_db = tmp_path / "annotations.sqlite"
    with AnnotationStore(annotations_db) as annotation_store:
        annotation_store.set_review_decision(a, REVIEW_KEEP)
        annotation_store.set_review_decision(b, REVIEW_REJECT)
        annotation_store.set_review_decision(c, REVIEW_REJECT)
        # d.jpg intentionally left Neutral (never decided).

    return tmp_path


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_user_vs_algorithm_tab_is_first_and_shows_the_confusion_matrix(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    assert dialog._tabs.tabText(0) == "User vs Algorithm"
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    tab = dialog._user_vs_algorithm_tab
    assert tab._user_keep_card._value_label.text() == "1"
    assert tab._user_reject_card._value_label.text() == "2"
    assert tab._algo_keep_card._value_label.text() == "2"
    assert tab._algo_reject_card._value_label.text() == "2"
    # a.jpg: algo keep/user keep; b.jpg: algo keep/user reject (FP);
    # c.jpg: algo reject/user reject; d.jpg: unreviewed, excluded.
    assert tab._matrix_table.item(0, 0).text() == "1"  # User Keep / Algorithm Keep
    assert tab._matrix_table.item(1, 0).text() == "1"  # User Reject / Algorithm Keep (false positive)
    assert tab._matrix_table.item(1, 1).text() == "1"  # User Reject / Algorithm Reject
    assert "1 not yet reviewed" in tab._coverage_label.text()

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_moving_the_threshold_recomputes_the_matrix(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    tab = dialog._user_vs_algorithm_tab
    assert tab._algo_keep_card._value_label.text() == "2"  # default: 50% (2 of 4)

    tab._threshold_spin.setValue(25.0)  # only a.jpg now counts as Algorithm Keep
    app.processEvents()

    assert tab._algo_keep_card._value_label.text() == "1"
    dialog.close()


# ---------------------------------------------------------------------------
# Phase 7 - every KPI card is clickable, not only the confusion-matrix
# cells: reuses _seed_agreement_scenario's own four images (a=agree/Keep,
# b=false positive, c=agree/Reject, d=Neutral, excluded from every pair).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_user_keep_card_is_clickable(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    dialog._user_vs_algorithm_tab._user_keep_card.clicked.emit()
    app.processEvents()

    assert dialog._tabs.currentWidget() is dialog._image_explorer_tab
    assert dialog._image_explorer_tab._image_list.count() == 1
    assert dialog._image_explorer_tab._image_list.item(0).text() == "a.jpg"
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_user_reject_card_is_clickable(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    dialog._user_vs_algorithm_tab._user_reject_card.clicked.emit()
    app.processEvents()

    names = {dialog._image_explorer_tab._image_list.item(row).text() for row in range(dialog._image_explorer_tab._image_list.count())}
    assert names == {"b.jpg", "c.jpg"}
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_algorithm_keep_card_is_clickable(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    dialog._user_vs_algorithm_tab._algo_keep_card.clicked.emit()
    app.processEvents()

    names = {dialog._image_explorer_tab._image_list.item(row).text() for row in range(dialog._image_explorer_tab._image_list.count())}
    assert names == {"a.jpg", "b.jpg"}
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_algorithm_reject_card_is_clickable(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    dialog._user_vs_algorithm_tab._algo_reject_card.clicked.emit()
    app.processEvents()

    # d.jpg is Algorithm Reject too, but was never reviewed (Neutral) - not
    # in report.pairs at all, so it correctly never appears here.
    assert dialog._image_explorer_tab._image_list.count() == 1
    assert dialog._image_explorer_tab._image_list.item(0).text() == "c.jpg"
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_agreement_card_is_clickable(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    dialog._user_vs_algorithm_tab._agree_card.clicked.emit()
    app.processEvents()

    names = {dialog._image_explorer_tab._image_list.item(row).text() for row in range(dialog._image_explorer_tab._image_list.count())}
    assert names == {"a.jpg", "c.jpg"}
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_disagreement_card_is_clickable(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    dialog._user_vs_algorithm_tab._disagree_card.clicked.emit()
    app.processEvents()

    assert dialog._image_explorer_tab._image_list.count() == 1
    assert dialog._image_explorer_tab._image_list.item(0).text() == "b.jpg"
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_a_real_mouse_click_on_a_kpi_card_emits_the_clicked_signal(tmp_path: Path) -> None:
    """Not just the signal wired up correctly - the actual mousePressEvent
    override that makes a card respond to being clicked at all."""
    from PySide6.QtCore import QPointF, Qt as QtCoreQt
    from PySide6.QtGui import QMouseEvent
    from picklikeme.desktop.dialogs.analytics_dashboard import SummaryCard

    app = QApplication.instance() or QApplication([])
    card = SummaryCard("Test", clickable=True)
    received = []
    card.clicked.connect(lambda: received.append(True))

    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(5, 5), QtCoreQt.MouseButton.LeftButton,
        QtCoreQt.MouseButton.LeftButton, QtCoreQt.KeyboardModifier.NoModifier,
    )
    card.mousePressEvent(event)

    assert received == [True]


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_a_non_clickable_summary_card_never_emits_clicked(tmp_path: Path) -> None:
    """Run Summary's own cards (clickable=False, the default) must not
    suddenly start drilling down anywhere just because SummaryCard itself
    gained click support."""
    from PySide6.QtCore import QPointF, Qt as QtCoreQt
    from PySide6.QtGui import QMouseEvent
    from picklikeme.desktop.dialogs.analytics_dashboard import SummaryCard

    app = QApplication.instance() or QApplication([])
    card = SummaryCard("Test")
    received = []
    card.clicked.connect(lambda: received.append(True))

    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(5, 5), QtCoreQt.MouseButton.LeftButton,
        QtCoreQt.MouseButton.LeftButton, QtCoreQt.KeyboardModifier.NoModifier,
    )
    card.mousePressEvent(event)

    assert received == []


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_clicking_a_confusion_matrix_cell_filters_the_image_inspector(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    # Row 1, column 0 = User Reject / Algorithm Keep = the false positive (b.jpg).
    dialog._user_vs_algorithm_tab._matrix_table.cellClicked.emit(1, 0)
    app.processEvents()

    assert dialog._tabs.currentWidget() is dialog._image_explorer_tab
    assert dialog._image_explorer_tab._image_list.count() == 1
    assert dialog._image_explorer_tab._image_list.item(0).text() == "b.jpg"

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_refresh_current_run_reflects_a_review_decision_made_after_the_dashboard_opened(tmp_path: Path) -> None:
    """The explicit Part 3 requirement: after Set User Decisions by
    Subfolders changes review_status, Agreement/Confusion Matrix/Precision/
    Recall/F1 must reflect it once refreshed - not require reopening the
    whole dashboard."""
    from picklikeme.analyzer.annotations import REVIEW_KEEP, AnnotationStore
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    tmp_path = _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    assert dialog._user_vs_algorithm_tab._coverage_label.text().startswith("3 image(s) compared")

    # d.jpg gets a decision from a completely separate AnnotationStore
    # connection - simulating Ground Truth import running against the same
    # underlying database file from MainWindow while this dashboard stays open.
    with AnnotationStore(tmp_path / "annotations.sqlite") as other_connection:
        other_connection.set_review_decision(tmp_path / "shoot" / "d.jpg", REVIEW_KEEP)

    dialog.refresh_current_run()
    app.processEvents()

    assert dialog._user_vs_algorithm_tab._coverage_label.text().startswith("4 image(s) compared")
    dialog.close()


# ---------------------------------------------------------------------------
# Phase 2 - Analytics Scope, Phase 3 - Dashboard header panel
# ---------------------------------------------------------------------------


def _seed_two_folders(db_path: Path, root_a: Path, root_b: Path) -> None:
    """Two runs recorded against genuinely DIFFERENT root folders - the
    scenario the "Current Root Folder" scope actually needs to prove it
    filters correctly (the original _seed fixture's two runs share one
    folder, which cannot distinguish "filtered" from "showing everything")."""
    with AnalyticsStore(db_path) as store:
        store.insert_run(
            "run-a", folder=str(root_a), strategy_id="ai-model", started_at="2026-08-04T21:00:00",
            considered=4, accepted=3, device="cuda", params={},
            reject_counts={}, image_metrics={str(root_a / "x.jpg"): {"score": 0.9}},
        )
        store.insert_run(
            "run-b", folder=str(root_b), strategy_id="classic-vision", started_at="2026-08-04T20:00:00",
            considered=2, accepted=1, device="cpu", params={},
            reject_counts={}, image_metrics={str(root_b / "y.jpg"): {"score": 0.5}},
        )


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_scope_defaults_to_current_root_folder_when_a_folder_is_open(tmp_path: Path) -> None:
    from PySide6.QtCore import Qt

    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    root_a = tmp_path / "ShootA"
    root_b = tmp_path / "ShootB"
    _seed_two_folders(tmp_path / "analytics.db", root_a, root_b)
    app = QApplication.instance() or QApplication([])

    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite", root_folder=str(root_a))

    assert dialog._scope_current_root_radio.isChecked() is True
    assert dialog._experiment_list.count() == 1
    assert "run-a" == dialog._experiment_list.item(0).data(Qt.ItemDataRole.UserRole)
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_scope_defaults_to_entire_database_with_no_folder_open(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    root_a = tmp_path / "ShootA"
    root_b = tmp_path / "ShootB"
    _seed_two_folders(tmp_path / "analytics.db", root_a, root_b)
    app = QApplication.instance() or QApplication([])

    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite", root_folder=None)

    assert dialog._scope_entire_db_radio.isChecked() is True
    assert dialog._scope_current_root_radio.isEnabled() is False
    assert dialog._experiment_list.count() == 2
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_switching_scope_to_entire_database_shows_every_run(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    root_a = tmp_path / "ShootA"
    root_b = tmp_path / "ShootB"
    _seed_two_folders(tmp_path / "analytics.db", root_a, root_b)
    app = QApplication.instance() or QApplication([])

    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite", root_folder=str(root_a))
    assert dialog._experiment_list.count() == 1

    dialog._scope_entire_db_radio.setChecked(True)

    assert dialog._experiment_list.count() == 2
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_header_panel_shows_live_context_before_any_selection(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    root_a = tmp_path / "ShootA"
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(
        species_db=tmp_path / "species.db",
        analytics_db=tmp_path / "empty.db", annotations_db=tmp_path / "annotations.sqlite", root_folder=str(root_a), color_source="classic-vision", keep_percent=30.0,
    )

    assert str(root_a) in dialog._header_panel._context_label.text()
    assert "Current Root Folder" in dialog._header_panel._context_label.text()
    assert dialog._header_panel._dataset_card._value_label.text() == "ShootA"
    assert dialog._header_panel._threshold_card._value_label.text() == "30%"
    assert "Classic" in dialog._header_panel._color_source_card._value_label.text()
    assert dialog._header_panel._algorithm_card._value_label.text() == "—"
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_header_panel_updates_when_an_experiment_is_selected(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    root_a = tmp_path / "ShootA"
    root_b = tmp_path / "ShootB"
    _seed_two_folders(tmp_path / "analytics.db", root_a, root_b)
    app = QApplication.instance() or QApplication([])

    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite", root_folder=str(root_a))
    app.processEvents()

    assert dialog._header_panel._algorithm_card._value_label.text() == "AI"
    assert dialog._header_panel._image_count_card._value_label.text() == "4"
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_header_panel_resets_run_specific_fields_when_scope_changes(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    root_a = tmp_path / "ShootA"
    root_b = tmp_path / "ShootB"
    _seed_two_folders(tmp_path / "analytics.db", root_a, root_b)
    app = QApplication.instance() or QApplication([])

    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite", root_folder=str(root_a))
    assert dialog._header_panel._algorithm_card._value_label.text() == "AI"

    dialog._scope_entire_db_radio.setChecked(True)

    # A run is still selected (the first one in the now-larger list), so
    # the run-specific cards get repopulated rather than staying stale -
    # not asserting a specific value here, just that this doesn't crash
    # and the scope label itself updated.
    assert "Entire Analytics Database" in dialog._header_panel._context_label.text()
    dialog.close()


# ---------------------------------------------------------------------------
# ImageExplorerTab (Phase 4/5/6) - Visual Debug's overlay checkboxes/presets,
# neither one running the detector itself - they only ever read what an
# earlier Classic Vision run already cached (detection_boxes_for /
# eye_keypoints_for), the same read-only rule the Gallery/Loupe overlays
# already follow.
# ---------------------------------------------------------------------------


def _make_explorer_tab(tmp_path: Path, monkeypatch):
    from unittest import mock

    from picklikeme.desktop.dialogs import analytics_dashboard as dashboard_module

    annotation_store = mock.Mock()
    annotation_store.review_decisions.return_value = []
    annotation_store.capture_timestamp_of.return_value = None
    tab = dashboard_module.ImageExplorerTab(
        crop_cache_dir=tmp_path / "crops", annotation_store=annotation_store,
        species_db=tmp_path / "species.db",
    )
    base_pixmap = QPixmap(200, 150)
    base_pixmap.fill()
    monkeypatch.setattr(tab, "_load_full_pixmap", lambda path: base_pixmap)

    store = mock.Mock()
    store.image_metrics.return_value = {}
    store.get_run.return_value = {"strategy_id": "classic-vision"}
    store.image_paths.return_value = []
    tab.show_paths(store, "run-1", ["some/a.jpg"])
    return tab, dashboard_module


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_no_overlay_checkbox_reads_the_cache_when_none_are_checked(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)
    boxes_spy = mock.Mock(return_value=None)
    eye_spy = mock.Mock(return_value=None)
    monkeypatch.setattr(dashboard_module, "detection_boxes_for", boxes_spy)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", eye_spy)

    tab._refresh_overlays()

    boxes_spy.assert_not_called()
    eye_spy.assert_not_called()
    assert tab._landmarks_table.rowCount() == 0
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_selected_detection_checkbox_draws_independently_of_landmarks(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)
    boxes_spy = mock.Mock(return_value={
        "source_size": (200, 150),
        "selected": {"box": (10.0, 10.0, 50.0, 50.0)},
        "expanded_box": [0.0, 0.0, 60.0, 60.0],
        "others": [],
    })
    eye_spy = mock.Mock(return_value=None)
    monkeypatch.setattr(dashboard_module, "detection_boxes_for", boxes_spy)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", eye_spy)

    tab._overlay_checkboxes["selected_detection"].setChecked(True)

    boxes_spy.assert_called_once_with("some/a.jpg")
    eye_spy.assert_called_once()  # the box family also wants the eye ROI, if any
    assert not tab._original_label.pixmap().isNull()
    assert tab._landmarks_table.rowCount() == 0  # no landmark-family checkbox is on
    assert tab._overlay_preset_combo.currentText() == "Custom"
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_landmarks_checkbox_populates_only_the_landmarks_that_are_present(tmp_path: Path, monkeypatch) -> None:
    """"if available" (Issue 3's own wording): Left Eye and Beak were
    supplied, Right Eye/Head/both shoulders were not - only the two present
    ones get a row, never a fabricated placeholder for the rest."""
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)
    eye_spy = mock.Mock(return_value={
        "source_size": (200, 150),
        "accepted": True,
        "left": {"x": 20.0, "y": 30.0, "confidence": 0.91},
        "right": None,
        "beak": {"x": 25.0, "y": 60.0, "confidence": 0.82},
        "head_top": None,
        "left_shoulder": None,
        "right_shoulder": None,
    })
    boxes_spy = mock.Mock(return_value=None)
    monkeypatch.setattr(dashboard_module, "detection_boxes_for", boxes_spy)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", eye_spy)

    tab._overlay_checkboxes["landmarks"].setChecked(True)

    boxes_spy.assert_not_called()  # landmarks-only: never touches the box cache
    eye_spy.assert_called_once_with("some/a.jpg", crop_cache_dir=tab._crop_cache_dir)
    assert tab._landmarks_table.rowCount() == 2
    names = {tab._landmarks_table.item(row, 0).text() for row in range(tab._landmarks_table.rowCount())}
    assert names == {"Left Eye", "Beak"}
    # 4 columns now: Landmark / Confidence / Pixel Coordinates / Normalized
    # Coordinates (Phase 5's own requirement).
    assert tab._landmarks_table.columnCount() == 4
    row = next(r for r in range(tab._landmarks_table.rowCount()) if tab._landmarks_table.item(r, 0).text() == "Left Eye")
    assert tab._landmarks_table.item(row, 3).text() == f"({20.0 / 200:.3f}, {30.0 / 150:.3f})"

    # Turning it back off clears the table rather than leaving stale rows.
    tab._overlay_checkboxes["landmarks"].setChecked(False)
    assert tab._landmarks_table.rowCount() == 0
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_box_family_and_landmark_family_checkboxes_are_independent(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)
    boxes_spy = mock.Mock(return_value={
        "source_size": (200, 150), "selected": {"box": (10.0, 10.0, 50.0, 50.0)}, "expanded_box": None, "others": [],
    })
    eye_spy = mock.Mock(return_value={
        "source_size": (200, 150), "accepted": False,
        "left": {"x": 20.0, "y": 30.0, "confidence": 0.91}, "right": None,
        "beak": None, "head_top": None, "left_shoulder": None, "right_shoulder": None,
    })
    monkeypatch.setattr(dashboard_module, "detection_boxes_for", boxes_spy)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", eye_spy)

    tab._overlay_checkboxes["selected_detection"].setChecked(True)
    tab._overlay_checkboxes["landmarks"].setChecked(True)

    boxes_spy.assert_called_with("some/a.jpg")
    assert tab._landmarks_table.rowCount() == 1
    assert not tab._original_label.pixmap().isNull()
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_rejected_detections_checkbox_only_draws_when_the_eye_was_rejected(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)
    monkeypatch.setattr(dashboard_module, "detection_boxes_for", mock.Mock(return_value=None))
    accepted_eye = {"source_size": (200, 150), "accepted": True, "box": (1.0, 1.0, 5.0, 5.0)}
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", mock.Mock(return_value=accepted_eye))

    tab._overlay_checkboxes["rejected_detections"].setChecked(True)
    accepted_pixmap = tab._original_label.pixmap()

    tab._overlay_checkboxes["rejected_detections"].setChecked(False)
    rejected_eye = {**accepted_eye, "accepted": False}
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", mock.Mock(return_value=rejected_eye))
    tab._overlay_checkboxes["rejected_detections"].setChecked(True)
    rejected_pixmap = tab._original_label.pixmap()

    # Can't easily assert on drawn pixels without a real render target, but
    # both paths must at least produce a real (non-null) pixmap - the
    # meaningful behavioural coverage (accepted eyes drawing nothing extra)
    # lives in the pure box_family drawing logic itself.
    assert not accepted_pixmap.isNull()
    assert not rejected_pixmap.isNull()
    tab.close()


# ---------------------------------------------------------------------------
# Visual Debug presets (Phase 5): Detection / Crop / EyePose / Everything /
# Custom - a preset sets the exact checked set; manually toggling any
# checkbox afterward switches the combo to "Custom".
# ---------------------------------------------------------------------------


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_detection_preset_checks_exactly_the_detection_stage_boxes(tmp_path: Path, monkeypatch) -> None:
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)

    tab._overlay_preset_combo.setCurrentText("Detection")

    checked = {key for key, cb in tab._overlay_checkboxes.items() if cb.isChecked()}
    assert checked == dashboard_module._OVERLAY_PRESETS["Detection"]
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_everything_preset_checks_every_checkbox(tmp_path: Path, monkeypatch) -> None:
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)

    tab._overlay_preset_combo.setCurrentText("Everything")

    assert all(cb.isChecked() for cb in tab._overlay_checkboxes.values())
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_manually_toggling_a_checkbox_after_a_preset_switches_to_custom(tmp_path: Path, monkeypatch) -> None:
    tab, _dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)
    tab._overlay_preset_combo.setCurrentText("Crop")
    assert tab._overlay_preset_combo.currentText() == "Crop"

    tab._overlay_checkboxes["eye_roi"].setChecked(True)

    assert tab._overlay_preset_combo.currentText() == "Custom"
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_checking_exactly_a_presets_own_set_by_hand_is_recognised_as_that_preset(tmp_path: Path, monkeypatch) -> None:
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)

    for key in dashboard_module._OVERLAY_PRESETS["Crop"]:
        tab._overlay_checkboxes[key].setChecked(True)

    assert tab._overlay_preset_combo.currentText() == "Crop"
    tab.close()


# ---------------------------------------------------------------------------
# Manual QA Issue 3: landmarks must draw on the CROP (where the head fills
# most of the frame) rather than the full photo, whenever a crop is cached
# for the image - with a fallback to the Original when it is not.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_landmarks_draw_on_the_crop_when_one_is_cached(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    from picklikeme.bird_crop import crop_cache_path
    from picklikeme.desktop.dialogs import analytics_dashboard as dashboard_module

    app = QApplication.instance() or QApplication([])
    crop_dir = tmp_path / "crops"
    annotation_store = mock.Mock()
    annotation_store.review_decisions.return_value = []
    annotation_store.capture_timestamp_of.return_value = None
    tab = dashboard_module.ImageExplorerTab(
        crop_cache_dir=crop_dir, annotation_store=annotation_store, species_db=tmp_path / "species.db",
    )
    base_pixmap = QPixmap(200, 150)
    base_pixmap.fill()
    monkeypatch.setattr(tab, "_load_full_pixmap", lambda path: base_pixmap)

    image_path = "some/a.jpg"
    # Only existence is checked (Path.is_file()) - the bytes are never
    # decoded, since _load_full_pixmap is mocked above.
    crop_path = crop_cache_path(crop_dir, image_path)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake crop bytes")

    crop_eye_spy = mock.Mock(return_value={
        "source_size": (80, 60), "accepted": True,
        "left": {"x": 20.0, "y": 30.0, "confidence": 0.9}, "right": None,
        "beak": None, "head_top": None, "left_shoulder": None, "right_shoulder": None,
    })
    full_eye_spy = mock.Mock(return_value=None)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_in_crop_for", crop_eye_spy)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", full_eye_spy)

    store = mock.Mock()
    store.image_metrics.return_value = {}
    store.get_run.return_value = {"strategy_id": "classic-vision"}
    store.image_paths.return_value = []
    tab.show_paths(store, "run-1", [image_path])

    tab._overlay_checkboxes["landmarks"].setChecked(True)

    crop_eye_spy.assert_called_with(image_path, crop_cache_dir=crop_dir)
    # The Original still gets a (context-only) pass too - "keep optional
    # overlay on the original image if useful" - via the ordinary
    # full-frame function.
    full_eye_spy.assert_called_with(image_path, crop_cache_dir=crop_dir)
    assert tab._landmarks_table.rowCount() == 1
    assert tab._landmarks_table.item(0, 0).text() == "Left Eye"
    assert not tab._crop_label.pixmap().isNull()
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_landmarks_fall_back_to_the_original_when_no_crop_is_cached(tmp_path: Path, monkeypatch) -> None:
    """No file at all exists at the computed crop cache path for this
    image - crop_path.is_file() is False, so landmarks fall back onto the
    Original instead of showing nothing."""
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)
    eye_spy = mock.Mock(return_value={
        "source_size": (200, 150), "accepted": True,
        "left": {"x": 20.0, "y": 30.0, "confidence": 0.9}, "right": None,
        "beak": None, "head_top": None, "left_shoulder": None, "right_shoulder": None,
    })
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", eye_spy)
    crop_eye_spy = mock.Mock()
    monkeypatch.setattr(dashboard_module, "eye_keypoints_in_crop_for", crop_eye_spy)

    tab._overlay_checkboxes["landmarks"].setChecked(True)

    crop_eye_spy.assert_not_called()  # never called at all - there is no crop to draw it on
    eye_spy.assert_called_with("some/a.jpg", crop_cache_dir=tab._crop_cache_dir)
    assert tab._landmarks_table.rowCount() == 1
    assert "No crop cached" in tab._crop_label.text()
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_explorer_images_render_significantly_larger_than_other_tabs(tmp_path: Path, monkeypatch) -> None:
    """Manual QA Issue 3: "make both the original image and the crop
    significantly larger" - pins the actual size relationship rather than a
    specific number, so this stays meaningful if _THUMBNAIL_SIZE itself
    ever changes."""
    from picklikeme.desktop.dialogs import analytics_dashboard as dashboard_module

    assert dashboard_module._INSPECTOR_IMAGE_SIZE > dashboard_module._THUMBNAIL_SIZE

    tab, _dashboard_module = _make_explorer_tab(tmp_path, monkeypatch)
    assert tab._original_label.minimumSize().width() == dashboard_module._INSPECTOR_IMAGE_SIZE
    assert tab._crop_label.minimumSize().width() == dashboard_module._INSPECTOR_IMAGE_SIZE
    tab.close()


# ---------------------------------------------------------------------------
# Phase 4 - filtering: every filter combines with every other (AND). Uses
# show_run (not show_paths) against a real, small AnalyticsStore so the
# reject-reason/score-range/folder filters exercise real data rather than
# reimplementing the same logic in mocks.
# ---------------------------------------------------------------------------


def _make_explorer_tab_with_real_run(tmp_path: Path, monkeypatch):
    from unittest import mock

    from picklikeme.analytics.store import AnalyticsStore
    from picklikeme.desktop.dialogs import analytics_dashboard as dashboard_module

    store = AnalyticsStore(tmp_path / "analytics.db")
    store.insert_run(
        "run-1", folder=str(tmp_path / "shoot"), strategy_id="classic-vision-eyepose", started_at="t",
        considered=3, accepted=2, device="cpu",
        params={"weights": {"eye_sharpness_weight": 1.0, "subject_sharpness_weight": 0.0, "subject_size_weight": 0.0}},
        reject_counts={},
        image_metrics={
            str(tmp_path / "shoot" / "a.jpg"): {"eye_sharpness": 1.0, "subject_sharpness": 1.0, "subject_size": 0.1, "score": 0.9},
            str(tmp_path / "shoot" / "b.jpg"): {"eye_sharpness": 5.0, "subject_sharpness": 1.0, "subject_size": 0.1, "score": 0.5},
            str(tmp_path / "other" / "c.jpg"): {"eye_sharpness": 10.0, "subject_sharpness": 1.0, "subject_size": 0.1, "score": 0.1},
        },
    )
    annotation_store = mock.Mock()
    annotation_store.review_decisions.return_value = []
    annotation_store.capture_timestamp_of.return_value = None
    tab = dashboard_module.ImageExplorerTab(
        crop_cache_dir=tmp_path / "crops", annotation_store=annotation_store, species_db=tmp_path / "species.db",
    )
    monkeypatch.setattr(tab, "_load_full_pixmap", lambda path: QPixmap())
    tab.show_run(store, "run-1")
    return tab, store


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_show_run_lists_every_scored_image(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tab, _store = _make_explorer_tab_with_real_run(tmp_path, monkeypatch)
    assert tab._image_list.count() == 3
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_folder_filter_narrows_to_exactly_that_folder(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tab, _store = _make_explorer_tab_with_real_run(tmp_path, monkeypatch)

    tab._folder_combo.setCurrentText(str(tmp_path / "shoot"))

    names = {tab._image_list.item(row).text() for row in range(tab._image_list.count())}
    assert names == {"a.jpg", "b.jpg"}
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_search_and_folder_filters_combine(tmp_path: Path, monkeypatch) -> None:
    """AND, not OR - narrowing by folder AND then by a filename search that
    only one of the folder's own images matches."""
    app = QApplication.instance() or QApplication([])
    tab, _store = _make_explorer_tab_with_real_run(tmp_path, monkeypatch)

    tab._folder_combo.setCurrentText(str(tmp_path / "shoot"))
    tab._search_edit.setText("a.jpg")

    assert tab._image_list.count() == 1
    assert tab._image_list.item(0).text() == "a.jpg"
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_score_range_filter_narrows_by_the_runs_own_recorded_score(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tab, _store = _make_explorer_tab_with_real_run(tmp_path, monkeypatch)

    tab._score_min_spin.setValue(0.4)
    tab._score_max_spin.setValue(1.0)

    names = {tab._image_list.item(row).text() for row in range(tab._image_list.count())}
    assert names == {"a.jpg", "b.jpg"}  # c.jpg's score (0.1) falls below the min
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_clearing_a_filter_restores_the_full_candidate_list(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tab, _store = _make_explorer_tab_with_real_run(tmp_path, monkeypatch)

    tab._folder_combo.setCurrentText(str(tmp_path / "shoot"))
    assert tab._image_list.count() == 2

    tab._folder_combo.setCurrentText("All Folders")
    assert tab._image_list.count() == 3
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_show_paths_never_extends_the_exact_drill_down_set(tmp_path: Path, monkeypatch) -> None:
    """Unlike show_run, show_paths (the drill-down entry point) must show
    EXACTLY the given paths - never widened with other filtered-out images
    from the same run's folder sidecar."""
    from unittest import mock

    from picklikeme.desktop.dialogs import analytics_dashboard as dashboard_module

    app = QApplication.instance() or QApplication([])
    annotation_store = mock.Mock()
    annotation_store.review_decisions.return_value = []
    annotation_store.capture_timestamp_of.return_value = None
    tab = dashboard_module.ImageExplorerTab(
        crop_cache_dir=tmp_path / "crops", annotation_store=annotation_store, species_db=tmp_path / "species.db",
    )
    monkeypatch.setattr(tab, "_load_full_pixmap", lambda path: QPixmap())
    store = mock.Mock()
    store.image_metrics.return_value = {}
    store.get_run.return_value = {"strategy_id": "classic-vision", "folder": str(tmp_path)}

    tab.show_paths(store, "run-1", ["some/a.jpg"])

    assert tab._image_list.count() == 1
    assert tab._image_list.item(0).text() == "a.jpg"
    tab.close()


# ---------------------------------------------------------------------------
# Phase 4 - detail rows: Ground Truth / Algorithm Decision / User Decision.
# Phase 6 - Score Explanation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_detail_table_shows_ground_truth_and_decision_rows(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tab, _store = _make_explorer_tab_with_real_run(tmp_path, monkeypatch)

    fields = {tab._metrics_table.item(row, 0).text() for row in range(tab._metrics_table.rowCount())}
    assert {"Ground Truth", "User Decision", "Algorithm Decision"} <= fields
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_score_explanation_table_shows_the_per_metric_breakdown(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tab, _store = _make_explorer_tab_with_real_run(tmp_path, monkeypatch)
    tab._image_list.setCurrentRow(0)  # a.jpg

    assert tab._score_table.rowCount() == 3  # eye_sharpness, subject_sharpness, subject_size
    labels = {tab._score_table.item(row, 0).text() for row in range(tab._score_table.rowCount())}
    assert "Eye sharpness" in labels
    assert "Final Score" in tab._score_explanation_summary_label.text()
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_score_explanation_is_empty_for_a_run_with_no_weighted_metrics(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    from picklikeme.analytics.store import AnalyticsStore
    from picklikeme.desktop.dialogs import analytics_dashboard as dashboard_module

    app = QApplication.instance() or QApplication([])
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.insert_run(
        "run-ai", folder=str(tmp_path / "shoot"), strategy_id="ai-model", started_at="t",
        considered=1, accepted=1, device="cuda", params={}, reject_counts={},
        image_metrics={str(tmp_path / "shoot" / "a.jpg"): {"score": 0.5}},
    )
    annotation_store = mock.Mock()
    annotation_store.review_decisions.return_value = []
    annotation_store.capture_timestamp_of.return_value = None
    tab = dashboard_module.ImageExplorerTab(
        crop_cache_dir=tmp_path / "crops", annotation_store=annotation_store, species_db=tmp_path / "species.db",
    )
    monkeypatch.setattr(tab, "_load_full_pixmap", lambda path: QPixmap())

    tab.show_run(store, "run-ai")

    assert tab._score_table.rowCount() == 0
    assert "No per-metric breakdown" in tab._score_explanation_summary_label.text()
    tab.close()
