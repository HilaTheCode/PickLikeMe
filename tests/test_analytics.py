"""Analytics foundation: AnalyticsStore's schema, record_run's generic
capture contract, and the Phase 1 reports (Run Statistics, Rejection
Analysis, confidence distributions) built on top of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picklikeme.analytics.capture import record_run
from picklikeme.analytics.reports import (
    confidence_distribution,
    export_rejection_analysis_csv,
    export_run_statistics_csv,
    rejection_analysis,
    run_statistics,
)
from picklikeme.analytics.store import AnalyticsStore


def test_insert_and_read_back_a_run(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.db"
    with AnalyticsStore(db_path) as store:
        store.insert_run(
            "run-1",
            folder="/photos/shoot1",
            strategy_id="classic-vision-eyepose",
            started_at="2026-08-03T10:00:00",
            considered=10,
            accepted=7,
            device="cuda",
            params={"eye_confidence_threshold": 0.5},
            reject_counts={"NO_SUBJECT": 2, "LOW_HEAD_CONFIDENCE": 1},
            image_metrics={
                "a.jpg": {"eye_confidence": 0.9, "subject_size": 0.1},
                "b.jpg": {"eye_confidence": 0.8, "subject_size": 0.2},
            },
        )

        run = store.get_run("run-1")
        assert run["folder"] == "/photos/shoot1"
        assert run["considered"] == 10
        assert run["accepted"] == 7
        assert run["params"] == {"eye_confidence_threshold": 0.5}
        assert store.reject_counts("run-1") == {"NO_SUBJECT": 2, "LOW_HEAD_CONFIDENCE": 1}
        assert store.metric_names("run-1") == ["eye_confidence", "subject_size"]
        assert sorted(store.metric_values("run-1", "eye_confidence")) == [0.8, 0.9]


def test_get_run_returns_none_for_an_unknown_id(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        assert store.get_run("does-not-exist") is None


def test_reinserting_the_same_run_id_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    """insert_run is idempotent per run_id - a caller retrying after a
    partial failure must not end up with doubled reject counts or metrics."""
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="f", strategy_id="s", started_at="t", considered=5, accepted=5,
            device=None, params={}, reject_counts={"NO_SUBJECT": 1}, image_metrics={"a.jpg": {"x": 1.0}},
        )
        store.insert_run(
            "run-1", folder="f", strategy_id="s", started_at="t", considered=5, accepted=4,
            device=None, params={}, reject_counts={"NO_SUBJECT": 3}, image_metrics={"a.jpg": {"x": 2.0}},
        )
        assert store.reject_counts("run-1") == {"NO_SUBJECT": 3}
        assert store.metric_values("run-1", "x") == [2.0]


def test_list_runs_filters_by_folder_and_strategy(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "r1", folder="/a", strategy_id="classic", started_at="2026-08-01T00:00:00",
            considered=1, accepted=1, device=None, params={}, reject_counts={}, image_metrics={},
        )
        store.insert_run(
            "r2", folder="/b", strategy_id="ai-model", started_at="2026-08-02T00:00:00",
            considered=1, accepted=1, device=None, params={}, reject_counts={}, image_metrics={},
        )
        assert [r["run_id"] for r in store.list_runs(folder="/a")] == ["r1"]
        assert [r["run_id"] for r in store.list_runs(strategy_id="ai-model")] == ["r2"]
        assert {r["run_id"] for r in store.list_runs()} == {"r1", "r2"}


def test_record_run_is_generic_and_never_imports_a_ranking_module() -> None:
    """The whole point of capture.py: it accepts plain dicts/counts, never a
    strategy-specific dataclass - checked here by inspecting its own
    imports rather than trusting the module docstring."""
    import picklikeme.analytics.capture as capture_module

    assert "ranking" not in capture_module.__file__  # sanity: this IS capture.py
    source = Path(capture_module.__file__).read_text(encoding="utf-8")
    assert "from ..ranking" not in source
    assert "from .ranking" not in source


def test_record_run_writes_a_retrievable_run(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.db"
    run_id = record_run(
        "/photos/shoot1",
        "classic-vision-eyepose",
        considered=4,
        accepted=3,
        reject_counts={"NO_VISIBLE_EYE": 1},
        image_metrics={"a.jpg": {"eye_confidence": 0.95}},
        params={"eye_confidence_threshold": 0.5},
        device="cpu",
        db_path=db_path,
    )
    assert run_id is not None
    with AnalyticsStore(db_path) as store:
        run = store.get_run(run_id)
        assert run["considered"] == 4
        assert run["accepted"] == 3
        assert store.reject_counts(run_id) == {"NO_VISIBLE_EYE": 1}


def test_record_run_never_raises_even_if_the_db_path_is_unwritable(tmp_path: Path, monkeypatch) -> None:
    """Analytics recording must never break a ranking run - see the module
    docstring's 'failure to record is never fatal' contract."""
    import picklikeme.analytics.capture as capture_module

    def _boom(*args, **kwargs):
        raise OSError("disk is full")

    monkeypatch.setattr(capture_module, "AnalyticsStore", _boom)
    result = record_run("/photos", "classic", considered=1, accepted=1, db_path=tmp_path / "x.db")
    assert result is None


def test_run_statistics_summarizes_counts_and_metric_means(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="classic", started_at="2026-08-03T10:00:00",
            considered=4, accepted=2, device="cuda", params={"threshold": 0.5},
            reject_counts={"NO_SUBJECT": 1, "NO_VISIBLE_EYE": 1},
            image_metrics={"a.jpg": {"eye_confidence": 0.9}, "b.jpg": {"eye_confidence": 0.7}},
        )
        stats = run_statistics(store, "run-1")

    assert stats["considered"] == 4
    assert stats["accepted"] == 2
    assert stats["rejected"] == 2
    assert stats["reject_counts"] == {"NO_SUBJECT": 1, "NO_VISIBLE_EYE": 1}
    assert stats["metric_means"] == {"eye_confidence": 0.8}


def test_run_statistics_is_empty_for_an_unknown_run(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        assert run_statistics(store, "nope") == {}


def test_rejection_analysis_reports_counts_and_percentages_sorted_descending(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="classic", started_at="t",
            considered=10, accepted=5, device=None, params={},
            reject_counts={"NO_SUBJECT": 1, "LOW_HEAD_CONFIDENCE": 4},
            image_metrics={},
        )
        rows = rejection_analysis(store, "run-1")

    assert rows[0]["reason"] == "LOW_HEAD_CONFIDENCE"
    assert rows[0]["count"] == 4
    assert rows[0]["percent_of_considered"] == pytest.approx(40.0)
    assert rows[1]["reason"] == "NO_SUBJECT"
    assert rows[1]["percent_of_considered"] == pytest.approx(10.0)


def test_metric_statistics_reports_mean_median_min_max(tmp_path: Path) -> None:
    from picklikeme.analytics.reports import metric_statistics

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="classic", started_at="t",
            considered=4, accepted=4, device=None, params={}, reject_counts={},
            image_metrics={
                "a.jpg": {"score": 0.5}, "b.jpg": {"score": 0.9}, "c.jpg": {"score": 0.6}, "d.jpg": {"score": 0.2},
            },
        )
        stats = metric_statistics(store, "run-1", "score")

    assert stats == {"mean": 0.55, "median": 0.55, "min": 0.2, "max": 0.9, "count": 4}


def test_metric_statistics_is_none_for_a_metric_this_run_never_recorded(tmp_path: Path) -> None:
    from picklikeme.analytics.reports import metric_statistics

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="classic", started_at="t",
            considered=1, accepted=1, device=None, params={}, reject_counts={}, image_metrics={},
        )
        assert metric_statistics(store, "run-1", "score") is None


def test_confidence_distribution_returns_the_raw_values_for_tuning(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="classic", started_at="t",
            considered=3, accepted=3, device=None, params={}, reject_counts={},
            image_metrics={"a.jpg": {"eye_confidence": 0.5}, "b.jpg": {"eye_confidence": 0.9}, "c.jpg": {"eye_confidence": 0.6}},
        )
        values = confidence_distribution(store, "run-1", "eye_confidence")

    assert sorted(values) == [0.5, 0.6, 0.9]


def test_confidence_distribution_is_empty_for_a_metric_this_run_never_recorded(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="classic", started_at="t",
            considered=1, accepted=1, device=None, params={}, reject_counts={}, image_metrics={},
        )
        assert confidence_distribution(store, "run-1", "wing_pose_score") == []


def test_export_run_statistics_csv_writes_a_readable_file(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="classic", started_at="t",
            considered=2, accepted=1, device="cpu", params={},
            reject_counts={"NO_SUBJECT": 1}, image_metrics={"a.jpg": {"eye_confidence": 0.9}},
        )
        out = export_run_statistics_csv(store, "run-1", tmp_path / "stats.csv")

    text = out.read_text(encoding="utf-8")
    assert "considered,2" in text
    assert "reject_count:NO_SUBJECT,1" in text
    assert "metric_mean:eye_confidence,0.9" in text


def test_export_rejection_analysis_csv_writes_a_readable_file(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="classic", started_at="t",
            considered=4, accepted=2, device=None, params={},
            reject_counts={"NO_SUBJECT": 2}, image_metrics={},
        )
        out = export_rejection_analysis_csv(store, "run-1", tmp_path / "rejects.csv")

    text = out.read_text(encoding="utf-8")
    assert "reason,count,percent_of_considered" in text
    assert "NO_SUBJECT,2,50.0" in text


# ---------------------------------------------------------------------------
# Extensions added for species-classification runs (Part 4 of the
# BioCLIP multi-backend infrastructure work) - summary_metrics, the
# category_counts generic alias, and per-image lookups for the Image
# Inspector.
# ---------------------------------------------------------------------------


def test_summary_metrics_round_trip(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="bioclip2", started_at="t",
            considered=10, accepted=8, device="cuda", params={},
            reject_counts={}, image_metrics={},
            summary_metrics={"runtime_seconds": 12.5, "images_per_second": 0.8},
        )
        metrics = store.summary_metrics("run-1")

    assert metrics == {"runtime_seconds": 12.5, "images_per_second": 0.8}


def test_summary_metrics_are_replaced_not_accumulated_on_reinsert(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="bioclip2", started_at="t",
            considered=1, accepted=1, device=None, params={}, reject_counts={}, image_metrics={},
            summary_metrics={"runtime_seconds": 5.0},
        )
        store.insert_run(
            "run-1", folder="/f", strategy_id="bioclip2", started_at="t",
            considered=1, accepted=1, device=None, params={}, reject_counts={}, image_metrics={},
            summary_metrics={"runtime_seconds": 9.0},
        )
        assert store.summary_metrics("run-1") == {"runtime_seconds": 9.0}


def test_category_counts_is_the_generic_name_for_reject_counts(tmp_path: Path) -> None:
    """category_counts and reject_counts must always agree - species runs
    use category_counts to store a species distribution (including
    "Unknown"), ranking runs use reject_counts for reject reasons, same
    underlying table - see store.py's own docstring."""
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="bioclip2", started_at="t",
            considered=5, accepted=4, device=None, params={},
            reject_counts={"Kingfisher": 3, "Unknown": 1, "Osprey": 1},
            image_metrics={},
        )
        assert store.category_counts("run-1") == store.reject_counts("run-1")
        assert store.category_counts("run-1")["Unknown"] == 1


def test_image_paths_and_image_metrics_for_the_image_inspector(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-1", folder="/f", strategy_id="bioclip2", started_at="t",
            considered=2, accepted=2, device=None, params={}, reject_counts={},
            image_metrics={
                "a.jpg": {"top1_confidence": 0.9, "inference_ms": 12.0},
                "b.jpg": {"top1_confidence": 0.4},
            },
        )
        assert store.image_paths("run-1") == ["a.jpg", "b.jpg"]
        assert store.image_metrics("run-1", "a.jpg") == {"top1_confidence": 0.9, "inference_ms": 12.0}
        assert store.image_metrics("run-1", "b.jpg") == {"top1_confidence": 0.4}
        assert store.image_metrics("run-1", "does-not-exist.jpg") == {}


def test_record_run_forwards_summary_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.db"
    run_id = record_run(
        "/photos", "bioclip2", considered=3, accepted=3,
        summary_metrics={"runtime_seconds": 4.2}, db_path=db_path,
    )
    with AnalyticsStore(db_path) as store:
        assert store.summary_metrics(run_id) == {"runtime_seconds": 4.2}
