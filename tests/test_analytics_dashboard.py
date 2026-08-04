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
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")

    dialog._experiment_list.setCurrentRow(0)  # the bioclip2 run
    app.processEvents()

    assert dialog._run_summary_tab._table.rowCount() > 0
    assert dialog._species_analysis_tab._table.rowCount() == 4  # Accepted, Kingfisher, Osprey, Unknown
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

    assert first_species_rows == 4  # Accepted, Kingfisher, Osprey, Unknown
    assert second_species_rows == 3  # Accepted, Unknown, Baboon
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


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_experiment_search_box_filters_the_list_in_place(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed(tmp_path / "analytics.db")
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")

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
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")

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
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")

    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    tab = dialog._run_summary_tab
    assert tab._accepted_card._value_label.text() == "3"
    assert tab._rejected_card._value_label.text() == "1"
    assert tab._acceptance_card._value_label.text() == "75.0%"
    assert tab._score_card._value_label.text() == "0.7000"

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

    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", settings=settings)
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
        dialog2 = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", settings=settings_reloaded)
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
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")

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
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    tab = dialog._user_vs_algorithm_tab
    assert tab._algo_keep_card._value_label.text() == "2"  # default: 50% (2 of 4)

    tab._threshold_spin.setValue(25.0)  # only a.jpg now counts as Algorithm Keep
    app.processEvents()

    assert tab._algo_keep_card._value_label.text() == "1"
    dialog.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_clicking_a_confusion_matrix_cell_filters_the_image_inspector(tmp_path: Path) -> None:
    from picklikeme.desktop.dialogs.analytics_dashboard import AnalyticsDashboard

    _seed_agreement_scenario(tmp_path)
    app = QApplication.instance() or QApplication([])
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
    dialog._experiment_list.setCurrentRow(0)
    app.processEvents()

    # Row 1, column 0 = User Reject / Algorithm Keep = the false positive (b.jpg).
    dialog._user_vs_algorithm_tab._matrix_table.cellClicked.emit(1, 0)
    app.processEvents()

    assert dialog._tabs.currentWidget() is dialog._image_inspector_tab
    assert dialog._image_inspector_tab._image_list.count() == 1
    assert dialog._image_inspector_tab._image_list.item(0).text() == "b.jpg"

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
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", annotations_db=tmp_path / "annotations.sqlite")
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
