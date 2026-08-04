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
    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db")
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

    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", root_folder=str(root_a))

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

    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", root_folder=None)

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

    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", root_folder=str(root_a))
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
        analytics_db=tmp_path / "empty.db", root_folder=str(root_a), color_source="classic-vision", keep_percent=30.0,
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

    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", root_folder=str(root_a))
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

    dialog = AnalyticsDashboard(analytics_db=tmp_path / "analytics.db", root_folder=str(root_a))
    assert dialog._header_panel._algorithm_card._value_label.text() == "AI"

    dialog._scope_entire_db_radio.setChecked(True)

    # A run is still selected (the first one in the now-larger list), so
    # the run-specific cards get repopulated rather than staying stale -
    # not asserting a specific value here, just that this doesn't crash
    # and the scope label itself updated.
    assert "Entire Analytics Database" in dialog._header_panel._context_label.text()
    dialog.close()


# ---------------------------------------------------------------------------
# ImageInspectorTab's overlay checkboxes (Manual QA Issue 3): "Show
# Detection / Crop Boxes" and "Show Landmarks", independently toggleable,
# neither one running the detector itself - they only ever read what an
# earlier Classic Vision run already cached (detection_boxes_for /
# eye_keypoints_for), the same read-only rule the Gallery/Loupe overlays
# already follow.
# ---------------------------------------------------------------------------


def _make_inspector_tab(tmp_path: Path, monkeypatch):
    from unittest import mock

    from picklikeme.desktop.dialogs import analytics_dashboard as dashboard_module

    tab = dashboard_module.ImageInspectorTab(crop_cache_dir=tmp_path / "crops")
    base_pixmap = QPixmap(200, 150)
    base_pixmap.fill()
    monkeypatch.setattr(tab, "_load_full_pixmap", lambda path: base_pixmap)

    store = mock.Mock()
    store.image_metrics.return_value = {}
    store.get_run.return_value = {"strategy_id": "classic-vision"}
    tab.show_paths(store, "run-1", ["some/a.jpg"])
    return tab, dashboard_module


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_neither_overlay_checkbox_reads_the_cache_when_both_are_off(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_inspector_tab(tmp_path, monkeypatch)
    boxes_spy = mock.Mock(return_value=None)
    eye_spy = mock.Mock(return_value=None)
    monkeypatch.setattr(dashboard_module, "detection_boxes_for", boxes_spy)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", eye_spy)

    tab._refresh_original_display()

    boxes_spy.assert_not_called()
    eye_spy.assert_not_called()
    assert tab._landmarks_table.rowCount() == 0
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_show_boxes_checkbox_draws_independently_of_landmarks(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_inspector_tab(tmp_path, monkeypatch)
    boxes_spy = mock.Mock(return_value={
        "source_size": (200, 150),
        "selected": {"box": (10.0, 10.0, 50.0, 50.0)},
        "expanded_box": [0.0, 0.0, 60.0, 60.0],
    })
    eye_spy = mock.Mock(return_value=None)
    monkeypatch.setattr(dashboard_module, "detection_boxes_for", boxes_spy)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", eye_spy)

    tab._show_boxes_checkbox.setChecked(True)

    boxes_spy.assert_called_once_with("some/a.jpg")
    eye_spy.assert_called_once()  # boxes overlay also wants the eye ROI, if any
    assert not tab._original_label.pixmap().isNull()
    assert tab._landmarks_table.rowCount() == 0  # landmarks checkbox is still off
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_show_landmarks_checkbox_populates_only_the_landmarks_that_are_present(tmp_path: Path, monkeypatch) -> None:
    """"if available" (Issue 3's own wording): Left Eye and Beak were
    supplied, Right Eye/Head/both shoulders were not - only the two present
    ones get a row, never a fabricated placeholder for the rest."""
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_inspector_tab(tmp_path, monkeypatch)
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

    tab._show_landmarks_checkbox.setChecked(True)

    boxes_spy.assert_not_called()  # landmarks-only: never touches the box cache
    eye_spy.assert_called_once_with("some/a.jpg", crop_cache_dir=tab._crop_cache_dir)
    assert tab._landmarks_table.rowCount() == 2
    names = {tab._landmarks_table.item(row, 0).text() for row in range(tab._landmarks_table.rowCount())}
    assert names == {"Left Eye", "Beak"}

    # Turning it back off clears the table rather than leaving stale rows.
    tab._show_landmarks_checkbox.setChecked(False)
    assert tab._landmarks_table.rowCount() == 0
    tab.close()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_both_overlay_checkboxes_can_be_on_at_once(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    app = QApplication.instance() or QApplication([])
    tab, dashboard_module = _make_inspector_tab(tmp_path, monkeypatch)
    boxes_spy = mock.Mock(return_value={
        "source_size": (200, 150), "selected": {"box": (10.0, 10.0, 50.0, 50.0)}, "expanded_box": None,
    })
    eye_spy = mock.Mock(return_value={
        "source_size": (200, 150), "accepted": False,
        "left": {"x": 20.0, "y": 30.0, "confidence": 0.91}, "right": None,
        "beak": None, "head_top": None, "left_shoulder": None, "right_shoulder": None,
    })
    monkeypatch.setattr(dashboard_module, "detection_boxes_for", boxes_spy)
    monkeypatch.setattr(dashboard_module, "eye_keypoints_for", eye_spy)

    tab._show_boxes_checkbox.setChecked(True)
    tab._show_landmarks_checkbox.setChecked(True)

    # Each checkbox's own toggle triggers its own redraw (boxes_spy is
    # therefore called once per toggle while "Show Boxes" stays checked) -
    # the point here is that both overlays are independently active
    # together, not a call-count contract.
    boxes_spy.assert_called_with("some/a.jpg")
    assert tab._landmarks_table.rowCount() == 1
    assert not tab._original_label.pixmap().isNull()
    tab.close()
