"""The one shared filtering engine (Product Direction: "the same engine
should drive Main Review Window / Analytics Dashboard / Loupe navigation")
behind the Advanced Filters panel (`desktop/widgets/advanced_filters_panel.py`).

Deliberately Qt-free and deliberately ignorant of where its data comes
from: `FilterableRecord` is a small, generic per-image shape both the live
Review Window (built from `ImageItem`) and the Analytics Dashboard (built
from `AnalyticsStore` rows plus `species.cache`/`_compute_burst_map`) can
adapt their own, structurally different data into - see
`main_window.py`'s `_build_filterable_records` and
`analytics_dashboard.py`'s equivalent. The MATCHING logic
(`matches`/`apply_filters`) is the single place "does this image pass the
current filters" is decided; neither caller re-implements it.

Loupe navigation needs no adapter of its own at all: it already opens
scoped to whatever `ImageModel` the Review Window's gallery is currently
showing (`main_window.py::_open_loupe_for_item` reads
`self._gallery_model.items()`) - so once Advanced Filters narrows that
model via this module, Loupe's Prev/Next automatically narrows with it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterableRecord:
    """One image, reduced to exactly the fields Advanced Filters can ask
    about - nothing else. `None` means "not available for this image"
    (never scored, no species prediction, not part of a multi-image
    burst's own distinguishing rank, ...) and a range filter on a field
    that is `None` for a given image always excludes it (see `matches`) -
    the same "explicit unknown, never fabricated" standard this project
    applies everywhere else.
    """

    path: str
    folder: str
    filename: str
    user_decision: str = "neutral"  # "keep" / "reject" / "neutral"
    algorithm_decision: str | None = None  # "keep" / "reject" / None (unscored)
    reject_reason: str | None = None
    species: str | None = None
    species_confidence: float | None = None
    score: float | None = None
    eye_confidence: float | None = None
    head_confidence: float | None = None
    subject_size: float | None = None
    eye_sharpness: float | None = None
    subject_sharpness: float | None = None
    burst_id: str | None = None
    burst_size: int = 1
    burst_rank: int = 1
    burst_best: bool = True


# The four conflict categories a photographer can filter by - see
# `compute_conflict_type`. "n/a" covers both "never reviewed" (still
# Neutral) and "never scored" (no algorithm_decision) - the filter engine
# never guesses at a conflict verdict from incomplete information.
CONFLICT_AGREE = "agree"
CONFLICT_FALSE_POSITIVE = "false_positive"  # algorithm said Keep, you said Reject
CONFLICT_FALSE_NEGATIVE = "false_negative"  # algorithm said Reject, you said Keep
CONFLICT_NA = "n/a"

CONFLICT_LABELS = {
    CONFLICT_AGREE: "Agree",
    CONFLICT_FALSE_POSITIVE: "False Positive (Algorithm Keep / You Reject)",
    CONFLICT_FALSE_NEGATIVE: "False Negative (Algorithm Reject / You Keep)",
    CONFLICT_NA: "N/A",
}


def compute_conflict_type(user_decision: str | None, algorithm_decision: str | None) -> str:
    """The single, shared definition of "conflict" - reused by every
    adapter so the Review Window's Conflict Type filter, the Dashboard's
    Agreement tab, and anything else that asks this question can never
    quietly define it two different ways."""
    if not algorithm_decision or not user_decision or user_decision == "neutral":
        return CONFLICT_NA
    if algorithm_decision == user_decision:
        return CONFLICT_AGREE
    return CONFLICT_FALSE_POSITIVE if algorithm_decision == "keep" else CONFLICT_FALSE_NEGATIVE


@dataclass(frozen=True)
class FilterCriteria:
    """Every Advanced Filters control's current value, all AND-combined
    (see `matches`) - a control left at its "no filter" value (empty
    string / `None`) contributes nothing. Frozen and comparable, so a
    widget can cheaply check "did anything actually change" before
    re-filtering (`is_active`, `__eq__` via dataclass equality)."""

    search: str = ""
    folder: str | None = None
    species: str | None = None
    burst: str = "all"  # "all" / "winners" / "losers"
    burst_rank: int | None = None
    user_decision: str | None = None
    algorithm_decision: str | None = None
    conflict_type: str | None = None
    reject_reason: str | None = None
    score_min: float | None = None
    score_max: float | None = None
    eye_confidence_min: float | None = None
    eye_confidence_max: float | None = None
    head_confidence_min: float | None = None
    head_confidence_max: float | None = None
    subject_size_min: float | None = None
    subject_size_max: float | None = None
    eye_sharpness_min: float | None = None
    eye_sharpness_max: float | None = None
    subject_sharpness_min: float | None = None
    subject_sharpness_max: float | None = None
    species_confidence_min: float | None = None
    species_confidence_max: float | None = None

    def is_active(self) -> bool:
        """Whether any filter is actually narrowing the set - lets a
        caller skip recomputation work (e.g. the Dashboard's per-tab
        statistics) when the answer is just "show me everything", the
        common case."""
        return self != FilterCriteria()


# (record field, criteria-min-field, criteria-max-field) for every ranged
# numeric filter - one shared loop in `matches` rather than six near-
# identical if-blocks that could quietly drift out of sync with each other.
_RANGE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("score", "score_min", "score_max"),
    ("eye_confidence", "eye_confidence_min", "eye_confidence_max"),
    ("head_confidence", "head_confidence_min", "head_confidence_max"),
    ("subject_size", "subject_size_min", "subject_size_max"),
    ("eye_sharpness", "eye_sharpness_min", "eye_sharpness_max"),
    ("subject_sharpness", "subject_sharpness_min", "subject_sharpness_max"),
    ("species_confidence", "species_confidence_min", "species_confidence_max"),
)


def matches(record: FilterableRecord, criteria: FilterCriteria) -> bool:
    """True when `record` clears every active filter in `criteria`. Every
    check is independent and AND-combined - "Species = Kingfisher AND
    User Reject AND Score > 0.80 AND Eye Confidence < 0.90" is simply four
    of these checks all needing to pass, exactly as the product direction
    describes it."""
    if criteria.search and criteria.search.lower() not in record.filename.lower():
        return False
    if criteria.folder and record.folder != criteria.folder:
        return False
    if criteria.species and record.species != criteria.species:
        return False
    if criteria.burst == "winners" and not record.burst_best:
        return False
    if criteria.burst == "losers" and record.burst_best:
        return False
    if criteria.burst_rank is not None and record.burst_rank != criteria.burst_rank:
        return False
    if criteria.user_decision and record.user_decision != criteria.user_decision:
        return False
    if criteria.algorithm_decision and record.algorithm_decision != criteria.algorithm_decision:
        return False
    if criteria.conflict_type and compute_conflict_type(record.user_decision, record.algorithm_decision) != criteria.conflict_type:
        return False
    if criteria.reject_reason and record.reject_reason != criteria.reject_reason:
        return False
    for field_name, min_field, max_field in _RANGE_FIELDS:
        lo = getattr(criteria, min_field)
        hi = getattr(criteria, max_field)
        if lo is None and hi is None:
            continue
        value = getattr(record, field_name)
        if value is None:
            return False  # a range filter on an unmeasured field excludes it, never guesses
        if lo is not None and value < lo:
            return False
        if hi is not None and value > hi:
            return False
    return True


def apply_filters(records: list[FilterableRecord], criteria: FilterCriteria) -> list[FilterableRecord]:
    """The whole engine, end to end: every record that survives every
    active filter, order preserved."""
    if not criteria.is_active():
        return list(records)
    return [record for record in records if matches(record, criteria)]
