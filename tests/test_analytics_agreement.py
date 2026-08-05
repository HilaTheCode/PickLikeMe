"""analytics.agreement - User vs Algorithm, computed from a PAST recorded
run (AnalyticsStore) joined against AnnotationStore's review decisions by
content identity. The live-session equivalent (ReviewSession.
agreement_stats) is tested in test_review_session.py; this file covers
only the historical-run version and its own extra concerns: the
keep_percent-from-accepted-ratio default, and the unmatched/neutral split
a live session never has to make (every image in a live session is, by
definition, currently on disk at a known path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picklikeme.analyzer.annotations import REVIEW_KEEP, REVIEW_REJECT, AnnotationStore
from picklikeme.analytics.agreement import (
    algorithm_decisions_for_run,
    compare_run_to_user_decisions,
    user_decisions_for_paths,
)
from picklikeme.analytics.store import AnalyticsStore


@pytest.fixture
def analytics_store(tmp_path):
    s = AnalyticsStore(tmp_path / "analytics.db")
    yield s
    s.close()


@pytest.fixture
def annotation_store(tmp_path):
    s = AnnotationStore(tmp_path / "annotations.sqlite")
    yield s
    s.close()


def _seed_run(analytics_store, tmp_path, *, considered: int, accepted: int, scores: dict[str, float]) -> str:
    analytics_store.insert_run(
        "run-1", folder=str(tmp_path), strategy_id="ai-model", started_at="t",
        considered=considered, accepted=accepted, device="cpu", params={},
        reject_counts={}, image_metrics={path: {"score": score} for path, score in scores.items()},
    )
    return "run-1"


def _write(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_algorithm_decisions_default_to_the_runs_own_accepted_ratio(tmp_path, analytics_store) -> None:
    scores = {f"img_{i}.jpg": 1.0 - i * 0.1 for i in range(4)}  # img_0 highest, img_3 lowest
    run_id = _seed_run(analytics_store, tmp_path, considered=4, accepted=2, scores=scores)

    decisions = algorithm_decisions_for_run(analytics_store, run_id)

    assert decisions["img_0.jpg"] == "keep"
    assert decisions["img_1.jpg"] == "keep"
    assert decisions["img_2.jpg"] == "reject"
    assert decisions["img_3.jpg"] == "reject"


def test_algorithm_decisions_accept_an_explicit_keep_percent_override(tmp_path, analytics_store) -> None:
    scores = {f"img_{i}.jpg": 1.0 - i * 0.1 for i in range(4)}
    run_id = _seed_run(analytics_store, tmp_path, considered=4, accepted=2, scores=scores)

    decisions = algorithm_decisions_for_run(analytics_store, run_id, keep_percent=25.0)

    assert decisions["img_0.jpg"] == "keep"
    assert decisions["img_1.jpg"] == "reject"


def test_algorithm_decisions_for_an_unknown_run_is_empty(tmp_path, analytics_store) -> None:
    assert algorithm_decisions_for_run(analytics_store, "nope") == {}


def test_algorithm_decisions_paths_narrows_the_result_without_changing_the_cut(tmp_path, analytics_store) -> None:
    """Product Direction (Analytics Dashboard filtering): scoping to a path
    subset must never redefine what "keep" means for an image - the cut
    stays the top keep_percent% of the FULL run, only the reported set
    narrows. img_1 is "keep" here (top 2 of 4) exactly as it is unfiltered,
    even though restricted to only the bottom three images."""
    scores = {f"img_{i}.jpg": 1.0 - i * 0.1 for i in range(4)}  # img_0 highest, img_3 lowest
    run_id = _seed_run(analytics_store, tmp_path, considered=4, accepted=2, scores=scores)

    decisions = algorithm_decisions_for_run(
        analytics_store, run_id, paths=["img_1.jpg", "img_2.jpg", "img_3.jpg"],
    )

    assert set(decisions) == {"img_1.jpg", "img_2.jpg", "img_3.jpg"}
    assert decisions["img_1.jpg"] == "keep"
    assert decisions["img_2.jpg"] == "reject"
    assert decisions["img_3.jpg"] == "reject"


def test_user_decisions_distinguishes_neutral_from_unmatched(tmp_path, annotation_store) -> None:
    kept = _write(tmp_path / "a.jpg", b"a")
    rejected = _write(tmp_path / "b.jpg", b"b")
    never_decided = _write(tmp_path / "c.jpg", b"c")
    annotation_store.set_review_decision(kept, REVIEW_KEEP)
    annotation_store.set_review_decision(rejected, REVIEW_REJECT)
    moved_away = str(tmp_path / "does-not-exist-anymore.jpg")

    decisions = user_decisions_for_paths(
        annotation_store, [str(kept), str(rejected), str(never_decided), moved_away],
    )

    assert decisions[str(kept)] == "keep"
    assert decisions[str(rejected)] == "reject"
    assert decisions[str(never_decided)] == "neutral"
    assert moved_away not in decisions  # unmatched - identity could not be resolved at all


def test_compare_run_to_user_decisions_perfect_agreement(tmp_path, analytics_store, annotation_store) -> None:
    a = _write(tmp_path / "a.jpg", b"a")
    b = _write(tmp_path / "b.jpg", b"b")
    annotation_store.set_review_decision(a, REVIEW_KEEP)
    annotation_store.set_review_decision(b, REVIEW_REJECT)
    run_id = _seed_run(analytics_store, tmp_path, considered=2, accepted=1, scores={str(a): 0.9, str(b): 0.1})

    report = compare_run_to_user_decisions(analytics_store, annotation_store, run_id)

    assert report.compared == 2
    assert report.agree == 2
    assert report.disagree == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.override_rate == 0.0
    assert report.mean_score_user_kept == pytest.approx(0.9)
    assert report.mean_score_user_rejected == pytest.approx(0.1)


def test_compare_run_to_user_decisions_disagreement_shows_up_as_false_positive_and_negative(
    tmp_path, analytics_store, annotation_store
) -> None:
    """Algorithm keeps a.jpg, user rejects it (false positive). Algorithm
    rejects b.jpg, user keeps it (false negative)."""
    a = _write(tmp_path / "a.jpg", b"a")
    b = _write(tmp_path / "b.jpg", b"b")
    annotation_store.set_review_decision(a, REVIEW_REJECT)
    annotation_store.set_review_decision(b, REVIEW_KEEP)
    run_id = _seed_run(analytics_store, tmp_path, considered=2, accepted=1, scores={str(a): 0.9, str(b): 0.1})

    report = compare_run_to_user_decisions(analytics_store, annotation_store, run_id)

    assert report.algo_keep_user_reject == 1  # false positive
    assert report.algo_reject_user_keep == 1  # false negative
    assert report.to_dict()["false_positives"] == 1
    assert report.to_dict()["false_negatives"] == 1
    assert report.agree == 0
    assert report.disagree == 2
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.override_rate == 100.0


def test_compare_run_to_user_decisions_paths_scopes_the_report(tmp_path, analytics_store, annotation_store) -> None:
    """Analytics Dashboard Advanced Filters: narrowing to `paths` narrows
    every KPI/confusion-matrix count to just that subset, matching-run
    context (the algorithm's own accepted ratio) unchanged."""
    a = _write(tmp_path / "a.jpg", b"a")
    b = _write(tmp_path / "b.jpg", b"b")
    annotation_store.set_review_decision(a, REVIEW_REJECT)  # false positive
    annotation_store.set_review_decision(b, REVIEW_KEEP)  # false negative
    run_id = _seed_run(analytics_store, tmp_path, considered=2, accepted=1, scores={str(a): 0.9, str(b): 0.1})

    report = compare_run_to_user_decisions(analytics_store, annotation_store, run_id, paths=[str(a)])

    assert report.compared == 1
    assert report.algo_keep_user_reject == 1
    assert report.algo_reject_user_keep == 0
    assert report.pairs == [(str(a), "reject", "keep")]


def test_neutral_and_unmatched_images_are_excluded_from_comparison_but_counted(
    tmp_path, analytics_store, annotation_store
) -> None:
    decided = _write(tmp_path / "a.jpg", b"a")
    never_decided = _write(tmp_path / "b.jpg", b"b")
    annotation_store.set_review_decision(decided, REVIEW_KEEP)
    # "c.jpg" is recorded in the run but was never written to disk here -
    # its identity cannot be resolved (moved/deleted since ranking).
    run_id = _seed_run(
        analytics_store, tmp_path, considered=3, accepted=2,
        scores={str(decided): 0.9, str(never_decided): 0.5, str(tmp_path / "c.jpg"): 0.1},
    )

    report = compare_run_to_user_decisions(analytics_store, annotation_store, run_id)

    assert report.compared == 1
    assert report.neutral == 1
    assert report.unmatched == 1


def test_algorithm_keep_reject_totals_include_unmatched_and_neutral_images(
    tmp_path, analytics_store, annotation_store
) -> None:
    """Algorithm Keep/Reject totals describe what the algorithm did for
    every image it scored - independent of whether the photographer's
    decision on that image could be compared at all."""
    decided = _write(tmp_path / "a.jpg", b"a")
    annotation_store.set_review_decision(decided, REVIEW_KEEP)
    run_id = _seed_run(
        analytics_store, tmp_path, considered=2, accepted=1,
        scores={str(decided): 0.9, str(tmp_path / "gone.jpg"): 0.1},
    )

    report = compare_run_to_user_decisions(analytics_store, annotation_store, run_id)

    assert report.algorithm_keep == 1
    assert report.algorithm_reject == 1
