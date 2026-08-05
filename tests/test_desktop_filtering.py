"""desktop/filtering.py - the one shared filter engine behind Advanced
Filters, used identically by the Review Window, the Analytics Dashboard,
and (implicitly, via the Review Window's own gallery model) Loupe
navigation. Pure Python, no Qt - these tests never construct a
QApplication.
"""

from __future__ import annotations

from picklikeme.desktop.filtering import (
    CONFLICT_AGREE,
    CONFLICT_FALSE_NEGATIVE,
    CONFLICT_FALSE_POSITIVE,
    CONFLICT_NA,
    FilterCriteria,
    FilterableRecord,
    apply_filters,
    compute_conflict_type,
    matches,
)


def _record(**overrides) -> FilterableRecord:
    defaults = dict(path="/shoot/a.jpg", folder="/shoot", filename="a.jpg")
    defaults.update(overrides)
    return FilterableRecord(**defaults)


# ---------------------------------------------------------------------------
# compute_conflict_type - the shared definition every adapter reuses.
# ---------------------------------------------------------------------------


def test_conflict_agree_when_user_and_algorithm_match():
    assert compute_conflict_type("keep", "keep") == CONFLICT_AGREE
    assert compute_conflict_type("reject", "reject") == CONFLICT_AGREE


def test_conflict_false_positive_when_algorithm_keeps_but_user_rejects():
    assert compute_conflict_type("reject", "keep") == CONFLICT_FALSE_POSITIVE


def test_conflict_false_negative_when_algorithm_rejects_but_user_keeps():
    assert compute_conflict_type("keep", "reject") == CONFLICT_FALSE_NEGATIVE


def test_conflict_na_when_never_scored():
    assert compute_conflict_type("keep", None) == CONFLICT_NA


def test_conflict_na_when_still_neutral():
    assert compute_conflict_type("neutral", "keep") == CONFLICT_NA


# ---------------------------------------------------------------------------
# matches() - one filter at a time.
# ---------------------------------------------------------------------------


def test_default_criteria_matches_everything():
    record = _record()
    assert matches(record, FilterCriteria()) is True


def test_search_matches_filename_case_insensitively():
    record = _record(filename="DSC1234.jpg")
    assert matches(record, FilterCriteria(search="dsc")) is True
    assert matches(record, FilterCriteria(search="nomatch")) is False


def test_folder_filter_is_exact():
    record = _record(folder="/shoot/_Selected")
    assert matches(record, FilterCriteria(folder="/shoot/_Selected")) is True
    assert matches(record, FilterCriteria(folder="/shoot")) is False


def test_species_filter():
    record = _record(species="Kingfisher")
    assert matches(record, FilterCriteria(species="Kingfisher")) is True
    assert matches(record, FilterCriteria(species="Osprey")) is False
    # No species recorded at all - a species filter must exclude it, not
    # silently pass it through.
    assert matches(_record(species=None), FilterCriteria(species="Kingfisher")) is False


def test_burst_winners_and_losers():
    winner = _record(burst_best=True)
    loser = _record(burst_best=False)
    assert matches(winner, FilterCriteria(burst="winners")) is True
    assert matches(loser, FilterCriteria(burst="winners")) is False
    assert matches(loser, FilterCriteria(burst="losers")) is True
    assert matches(winner, FilterCriteria(burst="losers")) is False
    # "all" (the default) never filters by burst membership.
    assert matches(winner, FilterCriteria(burst="all")) is True
    assert matches(loser, FilterCriteria(burst="all")) is True


def test_burst_rank_filter():
    record = _record(burst_rank=2)
    assert matches(record, FilterCriteria(burst_rank=2)) is True
    assert matches(record, FilterCriteria(burst_rank=1)) is False


def test_user_decision_filter():
    record = _record(user_decision="reject")
    assert matches(record, FilterCriteria(user_decision="reject")) is True
    assert matches(record, FilterCriteria(user_decision="keep")) is False


def test_algorithm_decision_filter():
    record = _record(algorithm_decision="keep")
    assert matches(record, FilterCriteria(algorithm_decision="keep")) is True
    assert matches(record, FilterCriteria(algorithm_decision="reject")) is False
    # Never scored at all - an explicit Algorithm Decision filter excludes it.
    assert matches(_record(algorithm_decision=None), FilterCriteria(algorithm_decision="keep")) is False


def test_conflict_type_filter_reuses_compute_conflict_type():
    false_positive = _record(user_decision="reject", algorithm_decision="keep")
    agree = _record(user_decision="keep", algorithm_decision="keep")
    assert matches(false_positive, FilterCriteria(conflict_type=CONFLICT_FALSE_POSITIVE)) is True
    assert matches(agree, FilterCriteria(conflict_type=CONFLICT_FALSE_POSITIVE)) is False


def test_reject_reason_filter():
    record = _record(reject_reason="NO_VISIBLE_EYE")
    assert matches(record, FilterCriteria(reject_reason="NO_VISIBLE_EYE")) is True
    assert matches(record, FilterCriteria(reject_reason="NO_SUBJECT")) is False


def test_score_range_filter_is_inclusive_on_both_ends():
    record = _record(score=0.5)
    assert matches(record, FilterCriteria(score_min=0.5, score_max=0.5)) is True
    assert matches(record, FilterCriteria(score_min=0.51)) is False
    assert matches(record, FilterCriteria(score_max=0.49)) is False


def test_score_range_excludes_an_unscored_image_rather_than_passing_it():
    record = _record(score=None)
    assert matches(record, FilterCriteria(score_min=0.0)) is False


def test_every_range_field_is_wired_up():
    """One test per numeric field, confirming each is actually connected -
    not just eye_confidence/score, which the tests above already exercise
    more thoroughly."""
    cases = [
        ("eye_confidence", "eye_confidence_min", "eye_confidence_max"),
        ("head_confidence", "head_confidence_min", "head_confidence_max"),
        ("subject_size", "subject_size_min", "subject_size_max"),
        ("eye_sharpness", "eye_sharpness_min", "eye_sharpness_max"),
        ("subject_sharpness", "subject_sharpness_min", "subject_sharpness_max"),
        ("species_confidence", "species_confidence_min", "species_confidence_max"),
    ]
    for field, min_key, max_key in cases:
        record = _record(**{field: 10.0})
        assert matches(record, FilterCriteria(**{min_key: 10.0})) is True, field
        assert matches(record, FilterCriteria(**{min_key: 10.1})) is False, field
        assert matches(record, FilterCriteria(**{max_key: 9.9})) is False, field


# ---------------------------------------------------------------------------
# Combining multiple filters (AND) - the product direction's own example.
# ---------------------------------------------------------------------------


def test_multiple_filters_combine_with_and():
    """Species = Kingfisher AND User Reject AND Score > 0.80 AND Eye
    Confidence < 0.90 - the exact example from the product direction."""
    criteria = FilterCriteria(
        species="Kingfisher", user_decision="reject", score_min=0.80, eye_confidence_max=0.90,
    )
    matching = _record(
        species="Kingfisher", user_decision="reject", score=0.85, eye_confidence=0.88,
    )
    assert matches(matching, criteria) is True

    # Each one independently should be able to break the match.
    assert matches(_record(species="Osprey", user_decision="reject", score=0.85, eye_confidence=0.88), criteria) is False
    assert matches(_record(species="Kingfisher", user_decision="keep", score=0.85, eye_confidence=0.88), criteria) is False
    assert matches(_record(species="Kingfisher", user_decision="reject", score=0.79, eye_confidence=0.88), criteria) is False
    assert matches(_record(species="Kingfisher", user_decision="reject", score=0.85, eye_confidence=0.91), criteria) is False


def test_apply_filters_returns_only_matching_records_in_order():
    records = [
        _record(path="a.jpg", filename="a.jpg", user_decision="keep"),
        _record(path="b.jpg", filename="b.jpg", user_decision="reject"),
        _record(path="c.jpg", filename="c.jpg", user_decision="keep"),
    ]
    result = apply_filters(records, FilterCriteria(user_decision="keep"))
    assert [r.path for r in result] == ["a.jpg", "c.jpg"]


def test_apply_filters_with_no_active_criteria_is_a_cheap_passthrough():
    records = [_record(path="a.jpg"), _record(path="b.jpg")]
    result = apply_filters(records, FilterCriteria())
    assert result == records
    assert result is not records  # still a new list, never the caller's own


def test_is_active_distinguishes_default_from_any_set_filter():
    assert FilterCriteria().is_active() is False
    assert FilterCriteria(search="x").is_active() is True
    assert FilterCriteria(score_min=0.5).is_active() is True
    assert FilterCriteria(burst="winners").is_active() is True
