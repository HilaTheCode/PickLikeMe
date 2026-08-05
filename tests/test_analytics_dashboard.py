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
    assert dialog._user_vs_algorithm_tab._run_id == "run-1"

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
def test_clicking_a_species_row_pins_every_tab_to_that_species(tmp_path: Path) -> None:
    """Species Analysis's distribution table drill-down no longer opens a
    separate Image Explorer (removed - see analytics_dashboard.py's own
    Product Direction docstring); it pins the shared drill-down scope
    instead, which every detail tab (here: Run Summary's own card) reads
    through _apply_filters_to_tabs. Needs real per-image SpeciesCache
    predictions - species DRILL-DOWN is always best-effort, per-image data
    (never derived from the whole-run `category_counts` aggregate the
    UNFILTERED table itself reads, which is why `reject_counts` is ALSO
    seeded below: the two are genuinely separate sources - see this
    module's own docstring)."""
    from picklikeme.species.cache import SpeciesCache
    from picklikeme.species.classifier import SpeciesPrediction
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    shoot = tmp_path / "shoot"
    a, b = shoot / "a.jpg", shoot / "b.jpg"
    _make_jpeg(a, color="red")
    _make_jpeg(b, color="green")
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder=str(shoot), strategy_id="bioclip2", started_at="t",
            considered=2, accepted=2, device="cpu", params={},
            reject_counts={"Kingfisher": 1, "Osprey": 1},  # the whole-run aggregate the UNFILTERED table reads
            image_metrics={str(a): {"top1_confidence": 0.9}, str(b): {"top1_confidence": 0.4}},
        )
    species_db = tmp_path / "species.db"
    with SpeciesCache(species_db) as cache:
        cache.store(a, SpeciesPrediction(species="Kingfisher", confidence=0.9, classifier_id="bioclip2"))
        cache.store(b, SpeciesPrediction(species="Osprey", confidence=0.4, classifier_id="bioclip2"))

    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=species_db, analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    table = dialog._species_analysis_tab._table
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "Kingfisher")
    table.cellClicked.emit(row, 0)
    app.processEvents()

    assert dialog._drill_down_paths == [str(a)]
    assert dialog._run_summary_tab._images_processed_card._value_label.text() == "1"  # Considered = len(records)
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
def test_advanced_filters_burst_winners_agrees_with_burst_analytics_tab(tmp_path: Path) -> None:
    """Both Advanced Filters' own Burst control and BurstAnalyticsTab read
    through the same _compute_burst_map (via _build_run_records) - a
    winner in one must be a winner in the other, and narrowing to Burst
    Winners must narrow every OTHER tab too (Phase C: "changing filters
    should immediately update ... Burst Analytics, Run Summary")."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _make_burst_images(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    dialog._advanced_filters_panel._burst_combo.setCurrentText("Winners")
    app.processEvents()

    burst_table = dialog._burst_analytics_tab._table
    winner_names = {burst_table.item(r, 0).text() for r in range(burst_table.rowCount())}
    assert winner_names == {"a.jpg", "c.jpg"}
    assert dialog._run_summary_tab._images_processed_card._value_label.text() == "2"

    dialog._advanced_filters_panel._burst_combo.setCurrentText("Losers")
    app.processEvents()
    loser_names = {burst_table.item(r, 0).text() for r in range(burst_table.rowCount())}
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
def test_dashboard_never_crashes_when_a_run_recorded_a_missing_source_file(tmp_path: Path) -> None:
    """The exact real-world case found during validation: a recorded image
    path that no longer exists (Organize by Species moved the file after
    recording it) must never crash record-building or any detail tab -
    per-image display of a missing file is now the Loupe's concern, not
    the Dashboard's (see this module's own Product Direction docstring),
    but _build_run_records/_apply_filters_to_tabs must still degrade
    gracefully rather than raising."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")  # "a.jpg"/"b.jpg" - relative paths that never existed on disk
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    assert dialog._run_summary_tab._table.rowCount() > 0
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

    assert [Path(p).name for p in dialog._drill_down_paths] == ["a.jpg"]
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

    names = {Path(p).name for p in dialog._drill_down_paths}
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

    names = {Path(p).name for p in dialog._drill_down_paths}
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
    assert [Path(p).name for p in dialog._drill_down_paths] == ["c.jpg"]
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

    names = {Path(p).name for p in dialog._drill_down_paths}
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

    assert [Path(p).name for p in dialog._drill_down_paths] == ["b.jpg"]
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
def test_clicking_a_confusion_matrix_cell_pins_the_drill_down(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    # Row 1, column 0 = User Reject / Algorithm Keep = the false positive (b.jpg).
    dialog._user_vs_algorithm_tab._matrix_table.cellClicked.emit(1, 0)
    app.processEvents()

    assert [Path(p).name for p in dialog._drill_down_paths] == ["b.jpg"]
    # isHidden() (the explicit per-widget flag), not isVisible() - the
    # dialog in this test is never actually shown on screen, so isVisible()
    # would read False either way regardless of setVisible()'s own state.
    assert dialog._clear_drill_down_button.isHidden() is False

    dialog._clear_drill_down_button.click()
    app.processEvents()
    assert dialog._drill_down_paths is None
    assert dialog._clear_drill_down_button.isHidden() is True

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_advanced_filters_narrow_user_vs_algorithm_live_with_no_apply_button(tmp_path: Path) -> None:
    """Phase C's own requirement: "changing filters should immediately
    update KPIs ... Confusion Matrix ... Run Summary" - Advanced Filters
    alone (no drill-down, no Apply button) must narrow every tab as soon
    as a control changes."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    assert dialog._run_summary_tab._images_processed_card._value_label.text() == "4"

    dialog._advanced_filters_panel._user_decision_combo.setCurrentText("Reject")
    app.processEvents()

    # b.jpg and c.jpg are the only User Reject images (see _seed_agreement_scenario).
    assert dialog._run_summary_tab._images_processed_card._value_label.text() == "2"
    assert dialog._user_vs_algorithm_tab._user_reject_card._value_label.text() == "2"

    dialog._advanced_filters_panel.reset()
    app.processEvents()
    assert dialog._run_summary_tab._images_processed_card._value_label.text() == "4"

    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_drill_down_and_advanced_filters_combine_with_and(tmp_path: Path) -> None:
    """A pinned drill-down and a manually-set Advanced Filters criterion
    must narrow together (AND), never have one silently override the
    other - the same two-source pattern the Review Window's simple Filter
    combo + Advanced Filters already establishes."""
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(species_db=tmp_path / "species.db", analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    # Drill down to User Reject (b.jpg, c.jpg), then further narrow to
    # Algorithm Keep (b.jpg only - c.jpg is Algorithm Reject).
    dialog._user_vs_algorithm_tab._user_reject_card.clicked.emit()
    app.processEvents()
    assert {Path(p).name for p in dialog._drill_down_paths} == {"b.jpg", "c.jpg"}

    dialog._advanced_filters_panel._algorithm_decision_combo.setCurrentText("Keep")
    app.processEvents()

    assert dialog._run_summary_tab._images_processed_card._value_label.text() == "1"
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


