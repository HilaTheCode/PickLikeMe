"""User vs Algorithm - the historical-run counterpart to
`review.session.ReviewSession.agreement_stats()`, which only ever looks at
a LIVE, currently-open review session. This module answers the same
question - confusion matrix, precision/recall/F1 - for a PAST recorded
experiment (`AnalyticsStore`), joined against the photographer's actual
review decisions (`AnnotationStore`) by content identity, long after the
review session that set those decisions has closed.

**Why the algorithm's own Keep/Reject isn't already a stored fact.** A
ranking run records a continuous score per image, never a Keep/Reject
label - the split is `ReviewSession.keep_percent`, a review-time threshold
the photographer can move at any time (see its own docstring: "moving this
never changes anyone's review_status"). This module treats a run's own
recorded `accepted`/`considered` ratio as the default threshold (the split
that run's own summary implies), but `keep_percent` is always overridable
by the caller - the Dashboard's own "Keep top N%" control - for exactly
the same reason it is adjustable in Review.

**Why some images cannot be compared at all.** `AnnotationStore.identity_of`
needs the file to exist at the given path RIGHT NOW to establish identity
(see identity.py) - unlike a live ReviewSession, whose images the
photographer is looking at in the current folder, a historical run's
recorded paths may point at files `Organize`/`Arrange` has since moved.
Those images are reported as `unmatched`, never guessed at - the same
"explicit unknown, never fabricated" standard the rest of this project's
analytics already holds to (see species.experiment's own docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..analyzer.annotations import DECISION_SOURCE_USER, REVIEW_KEEP, REVIEW_REJECT, AnnotationStore
from ..identity import IdentityUnavailable
from .store import AnalyticsStore

KEEP = REVIEW_KEEP
REJECT = REVIEW_REJECT


def algorithm_decisions_for_run(
    store: AnalyticsStore, run_id: str, *, keep_percent: float | None = None, metric_name: str = "score",
    paths: list[str] | None = None,
) -> dict[str, str]:
    """image_path -> "keep"/"reject" for every image this run recorded
    `metric_name` for - ranked by that metric, split at `keep_percent`
    (top `keep_percent`% is "keep"). `keep_percent=None` (the default)
    uses the run's own recorded accepted/considered ratio - what the
    algorithm's own accept/reject split for this run actually was, rather
    than an arbitrary guess.

    `paths=None` (the default) returns every scored image. Otherwise the
    RESULT is narrowed to `paths` after the cut is computed - never before:
    the cut (and therefore what "keep" means for any one image) is always
    the top `keep_percent`% of the run's OWN full ranking, so a photographer
    narrowing the Analytics Dashboard with Advanced Filters (e.g. to one
    species) sees the same Algorithm Decision for an image whether or not a
    filter happens to be active. Ranking only that species' own images
    against each other would silently redefine "keep" per-filter, which
    would make an image's own Algorithm Decision change depending on what
    else was on screen - never desired.
    """
    run = store.get_run(run_id)
    if run is None:
        return {}
    if keep_percent is None:
        considered = run.get("considered") or 0
        keep_percent = 100.0 * (run.get("accepted") or 0) / considered if considered else 0.0

    scored = [(path, store.image_metrics(run_id, path).get(metric_name)) for path in store.image_paths(run_id)]
    scored = [(path, value) for path, value in scored if value is not None]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    cut = round(len(scored) * keep_percent / 100.0)
    decisions = {path: (KEEP if index < cut else REJECT) for index, (path, _value) in enumerate(scored)}
    if paths is not None:
        allowed = set(paths)
        decisions = {path: status for path, status in decisions.items() if path in allowed}
    return decisions


NEUTRAL = "neutral"


def user_decisions_for_paths(annotation_store: AnnotationStore, paths: list[str]) -> dict[str, str]:
    """image_path -> "keep"/"reject"/"neutral", for every path whose
    identity could be resolved. A path is simply ABSENT from the result
    when identity cannot be established at all (the file has since moved
    or been deleted - see this module's own docstring) - distinguished on
    purpose from "neutral" (identity resolved fine, the photographer just
    has not decided yet): one is a data-availability gap, the other is a
    real, current fact about the photographer's own review progress, and a
    report that conflated them would misrepresent which of the two is
    actually true. One bulk read (`review_decisions()`), not one query per
    path - the same discipline `ReviewSession`'s own fast-path identity
    matching already uses.

    DECISION_SOURCE_USER rows only. This function's entire purpose is
    comparing an algorithm against a HUMAN, so a decision an algorithm's own
    cutoff recorded (see `AnnotationStore.set_review_decision`'s `source`)
    is not evidence here - counting it would have the run's agreement
    statistics quietly measuring the cutoff against itself and reporting
    near-perfect agreement for a folder nobody had reviewed."""
    current_decision = {
        row["image_hash"]: row["decision"]
        for row in annotation_store.review_decisions()
        if (row.get("source") or DECISION_SOURCE_USER) == DECISION_SOURCE_USER
    }
    result: dict[str, str] = {}
    for path in paths:
        try:
            identity = annotation_store.identity_of(path)
        except IdentityUnavailable:
            continue
        result[path] = current_decision.get(identity) or NEUTRAL
    return result


@dataclass
class AgreementReport:
    compared: int = 0
    unmatched: int = 0  # algorithm scored it, but its identity could not be resolved (file moved/deleted)
    neutral: int = 0  # identity resolved fine, but the photographer has not decided yet
    user_keep: int = 0
    user_reject: int = 0
    algorithm_keep: int = 0
    algorithm_reject: int = 0
    agree: int = 0
    disagree: int = 0
    algo_keep_user_keep: int = 0
    algo_keep_user_reject: int = 0
    algo_reject_user_keep: int = 0
    algo_reject_user_reject: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    override_rate: float | None = None  # == disagree / compared; see module docstring
    mean_score_user_kept: float | None = None
    mean_score_user_rejected: float | None = None
    # Every compared image, for drill-down - the Dashboard's own confusion-
    # matrix cells filter the image list against this.
    pairs: list[tuple[str, str, str]] = field(default_factory=list)  # (image_path, user_status, algo_status)

    def to_dict(self) -> dict:
        return {
            "compared": self.compared,
            "unmatched": self.unmatched,
            "neutral": self.neutral,
            "user_keep": self.user_keep,
            "user_reject": self.user_reject,
            "algorithm_keep": self.algorithm_keep,
            "algorithm_reject": self.algorithm_reject,
            "agree": self.agree,
            "disagree": self.disagree,
            "agree_percent": round(100 * self.agree / self.compared, 1) if self.compared else None,
            "disagree_percent": round(100 * self.disagree / self.compared, 1) if self.compared else None,
            "algo_keep_user_keep": self.algo_keep_user_keep,
            "algo_keep_user_reject": self.algo_keep_user_reject,
            "algo_reject_user_keep": self.algo_reject_user_keep,
            "algo_reject_user_reject": self.algo_reject_user_reject,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positives": self.algo_keep_user_reject,
            "false_negatives": self.algo_reject_user_keep,
            "override_rate": self.override_rate,
            "mean_score_user_kept": self.mean_score_user_kept,
            "mean_score_user_rejected": self.mean_score_user_rejected,
        }


def compare_run_to_user_decisions(
    analytics_store: AnalyticsStore,
    annotation_store: AnnotationStore,
    run_id: str,
    *,
    keep_percent: float | None = None,
    metric_name: str = "score",
    paths: list[str] | None = None,
) -> AgreementReport:
    """The full User vs Algorithm report for one run. Treats Keep as the
    positive class and the photographer's own decision as ground truth -
    precision is "of what the algorithm suggested keeping, how much did the
    photographer also keep"; recall is "of what the photographer kept, how
    much did the algorithm also suggest keeping" - the exact same framing
    `ReviewSession.agreement_stats()` already uses, so the two can never
    disagree about what these words mean.

    `paths=None` (the default) reports on every image this run scored - see
    `algorithm_decisions_for_run`'s own docstring for what `paths` narrows
    to and, just as importantly, what it deliberately does NOT change
    (the meaning of "keep" itself)."""
    algo_decisions = algorithm_decisions_for_run(
        analytics_store, run_id, keep_percent=keep_percent, metric_name=metric_name, paths=paths,
    )
    user_decisions = user_decisions_for_paths(annotation_store, list(algo_decisions))

    report = AgreementReport()
    kept_scores: list[float] = []
    rejected_scores: list[float] = []
    for path, algo_status in algo_decisions.items():
        if algo_status == KEEP:
            report.algorithm_keep += 1
        else:
            report.algorithm_reject += 1

        user_status = user_decisions.get(path)
        if user_status is None:
            report.unmatched += 1
            continue
        if user_status == NEUTRAL:
            report.neutral += 1
            continue
        report.compared += 1
        report.pairs.append((path, user_status, algo_status))
        if user_status == KEEP:
            report.user_keep += 1
            score = analytics_store.image_metrics(run_id, path).get(metric_name)
            if score is not None:
                kept_scores.append(score)
        else:
            report.user_reject += 1
            score = analytics_store.image_metrics(run_id, path).get(metric_name)
            if score is not None:
                rejected_scores.append(score)

        if algo_status == KEEP and user_status == KEEP:
            report.algo_keep_user_keep += 1
        elif algo_status == KEEP:
            report.algo_keep_user_reject += 1
        elif user_status == KEEP:
            report.algo_reject_user_keep += 1
        else:
            report.algo_reject_user_reject += 1

    report.agree = report.algo_keep_user_keep + report.algo_reject_user_reject
    report.disagree = report.algo_keep_user_reject + report.algo_reject_user_keep
    predicted_keep = report.algo_keep_user_keep + report.algo_keep_user_reject
    actual_keep = report.algo_keep_user_keep + report.algo_reject_user_keep
    report.precision = report.algo_keep_user_keep / predicted_keep if predicted_keep else None
    report.recall = report.algo_keep_user_keep / actual_keep if actual_keep else None
    report.f1 = (
        2 * report.precision * report.recall / (report.precision + report.recall)
        if report.precision and report.recall else None
    )
    report.override_rate = round(100 * report.disagree / report.compared, 1) if report.compared else None
    report.mean_score_user_kept = sum(kept_scores) / len(kept_scores) if kept_scores else None
    report.mean_score_user_rejected = sum(rejected_scores) / len(rejected_scores) if rejected_scores else None
    return report
