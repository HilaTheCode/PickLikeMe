"""analytics/score_explanation.py - Phase 6 (Score Explanation): every
number recomputed from AnalyticsStore's own recorded raw metrics and run
params, using the exact same normalize/weighted-sum arithmetic
ranking.classic.combine() applies at ranking time - nothing here is
invented, so these tests pin that the recomputation is actually faithful,
not just plausible-looking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picklikeme.analytics.score_explanation import WEIGHTED_METRICS, explain_score
from picklikeme.analytics.store import AnalyticsStore
from picklikeme.ranking.metrics import robust_normalize


def _seed_ranking_run(store: AnalyticsStore, *, weights: dict[str, float] | None = None) -> None:
    weights = weights or {
        "eye_sharpness_weight": 0.5, "subject_sharpness_weight": 0.3, "subject_size_weight": 0.2,
    }
    store.insert_run(
        "run-1", folder="/shoot", strategy_id="classic-vision-eyepose", started_at="2026-08-03T10:00:00",
        considered=3, accepted=3, device="cpu", params={"weights": weights},
        reject_counts={},
        image_metrics={
            "a.jpg": {
                "eye_sharpness": 0.0, "subject_sharpness": 10.0, "subject_size": 0.1,
                "eye_confidence": 0.9, "score": 0.111,
            },
            "b.jpg": {
                "eye_sharpness": 5.0, "subject_sharpness": 20.0, "subject_size": 0.2,
                "eye_confidence": 0.7, "score": 0.222,
            },
            "c.jpg": {
                "eye_sharpness": 10.0, "subject_sharpness": 30.0, "subject_size": 0.3,
                "eye_confidence": 0.5, "score": 0.333,
            },
        },
    )


def test_no_metrics_recorded_for_this_image_returns_none(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        _seed_ranking_run(store)
        assert explain_score(store, "run-1", "not_in_this_run.jpg") is None


def test_an_ai_model_run_with_only_a_bare_score_returns_none(tmp_path: Path) -> None:
    """ranking.ai_model records only {"score": ...} - none of the three
    weighted metrics - so there is nothing truthful to explain."""
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-ai", folder="/shoot", strategy_id="ai-model", started_at="t", considered=1, accepted=1,
            device="cuda", params={}, reject_counts={}, image_metrics={"a.jpg": {"score": 0.87}},
        )
        assert explain_score(store, "run-ai", "a.jpg") is None


def test_rows_match_the_exact_metrics_combine_itself_weights(tmp_path: Path) -> None:
    """Deliberately excludes eye_confidence, even though it IS recorded for
    this image - ranking.classic.combine() never weights it into the final
    score, so showing a row for it would misrepresent how the score was
    actually produced."""
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        _seed_ranking_run(store)
        explanation = explain_score(store, "run-1", "a.jpg")

    assert explanation is not None
    assert [row.metric for row in explanation.rows] == list(WEIGHTED_METRICS)
    assert "eye_confidence" not in [row.metric for row in explanation.rows]


def test_normalized_values_match_robust_normalize_for_the_same_distribution(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        _seed_ranking_run(store)
        explanation_a = explain_score(store, "run-1", "a.jpg")
        explanation_b = explain_score(store, "run-1", "b.jpg")
        explanation_c = explain_score(store, "run-1", "c.jpg")

    expected_eye = robust_normalize([0.0, 5.0, 10.0])  # a, b, c's own eye_sharpness values
    for explanation, expected in zip((explanation_a, explanation_b, explanation_c), expected_eye):
        row = next(r for r in explanation.rows if r.metric == "eye_sharpness")
        assert row.normalized_value == pytest.approx(expected)


def test_weight_comes_from_the_runs_own_recorded_params(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        _seed_ranking_run(store, weights={
            "eye_sharpness_weight": 0.6, "subject_sharpness_weight": 0.25, "subject_size_weight": 0.15,
        })
        explanation = explain_score(store, "run-1", "b.jpg")

    weights_by_metric = {row.metric: row.weight for row in explanation.rows}
    assert weights_by_metric == {
        "eye_sharpness": 0.6, "subject_sharpness": 0.25, "subject_size": 0.15,
    }


def test_contribution_is_weight_times_normalized_and_running_total_accumulates(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        _seed_ranking_run(store)
        explanation = explain_score(store, "run-1", "b.jpg")

    running = 0.0
    for row in explanation.rows:
        assert row.contribution == pytest.approx(row.weight * row.normalized_value)
        running += row.contribution
        assert row.running_total == pytest.approx(running)
    assert explanation.recomputed_score == pytest.approx(running)


def test_final_score_is_read_verbatim_from_the_recorded_metric(tmp_path: Path) -> None:
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        _seed_ranking_run(store)
        explanation = explain_score(store, "run-1", "c.jpg")

    assert explanation.final_score == pytest.approx(0.333)


def test_a_run_missing_the_score_metric_still_yields_the_metric_breakdown(tmp_path: Path) -> None:
    """final_score is None (never recorded), but the three weighted metrics
    were - the breakdown itself is still worth showing."""
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-2", folder="/shoot", strategy_id="classic-vision-eyepose", started_at="t",
            considered=1, accepted=1, device="cpu",
            params={"weights": {"eye_sharpness_weight": 1.0, "subject_sharpness_weight": 0.0, "subject_size_weight": 0.0}},
            reject_counts={}, image_metrics={"a.jpg": {"eye_sharpness": 5.0, "subject_sharpness": 1.0, "subject_size": 0.1}},
        )
        explanation = explain_score(store, "run-2", "a.jpg")

    assert explanation.final_score is None
    assert explanation.recomputed_score is not None


def test_a_run_missing_the_weights_param_defaults_every_weight_to_zero(tmp_path: Path) -> None:
    """No fabricated weight when a run's params never recorded one - 0.0,
    not a guessed default, so the contribution truthfully shows as zero
    rather than pretending to know a weight that was never recorded."""
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-3", folder="/shoot", strategy_id="classic-vision-eyepose", started_at="t",
            considered=1, accepted=1, device="cpu", params={},
            reject_counts={}, image_metrics={"a.jpg": {"eye_sharpness": 5.0, "subject_sharpness": 1.0, "subject_size": 0.1}},
        )
        explanation = explain_score(store, "run-3", "a.jpg")

    assert all(row.weight == 0.0 for row in explanation.rows)
    assert explanation.recomputed_score == 0.0


def test_only_the_metrics_actually_present_produce_rows(tmp_path: Path) -> None:
    """A run that recorded only two of the three weighted metrics for an
    image still explains those two, rather than returning nothing at all."""
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        store.insert_run(
            "run-4", folder="/shoot", strategy_id="classic-vision-eyepose", started_at="t",
            considered=1, accepted=1, device="cpu",
            params={"weights": {"eye_sharpness_weight": 0.7, "subject_sharpness_weight": 0.3, "subject_size_weight": 0.0}},
            reject_counts={}, image_metrics={"a.jpg": {"eye_sharpness": 5.0, "subject_sharpness": 1.0}},
        )
        explanation = explain_score(store, "run-4", "a.jpg")

    assert [row.metric for row in explanation.rows] == ["eye_sharpness", "subject_sharpness"]
