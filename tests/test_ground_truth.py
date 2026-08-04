"""ground_truth.build_plan/apply_plan - "Set User Decisions by Subfolders",
the bulk Ground-Truth-seeding workflow. Not an import: only
review_decisions rows change, matched by content identity (see
identity.py), never by path alone - a photographer's Keep/Reject/Neutral
folders may hold copies or renames of images tracked elsewhere under a
completely different path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picklikeme.analyzer.annotations import REVIEW_KEEP, REVIEW_REJECT, AnnotationStore
from picklikeme.ground_truth import build_plan, apply_plan


@pytest.fixture
def store(tmp_path):
    s = AnnotationStore(tmp_path / "annotations.sqlite")
    yield s
    s.close()


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_build_plan_counts_keep_folder_images_as_will_change_when_undecided(tmp_path, store) -> None:
    keep_folder = tmp_path / "Keep"
    _write(keep_folder / "a.jpg", b"image a")
    _write(keep_folder / "b.jpg", b"image b")

    plan = build_plan(store, keep_folder=keep_folder)

    assert len(plan.keep.paths) == 2
    assert plan.keep.will_change == 2
    assert plan.keep.already_matching == 0
    assert plan.reject.paths == []
    assert plan.neutral.paths == []
    assert plan.totals() == {
        "keep": 2, "reject": 0, "neutral": 0, "already_matching": 0, "will_change": 2, "conflicts": 0,
    }


def test_build_plan_recurses_into_subfolders(tmp_path, store) -> None:
    keep_folder = tmp_path / "Keep"
    _write(keep_folder / "a.jpg", b"a")
    _write(keep_folder / "2026-08-01" / "b.jpg", b"b")
    _write(keep_folder / "2026-08-01" / "burst1" / "c.jpg", b"c")

    plan = build_plan(store, keep_folder=keep_folder)

    assert len(plan.keep.paths) == 3


def test_build_plan_ignores_non_image_files(tmp_path, store) -> None:
    keep_folder = tmp_path / "Keep"
    _write(keep_folder / "a.jpg", b"a")
    _write(keep_folder / "notes.txt", b"not an image")
    _write(keep_folder / "Thumbs.db", b"not an image")

    plan = build_plan(store, keep_folder=keep_folder)

    assert [Path(p).name for p in plan.keep.paths] == ["a.jpg"]


def test_build_plan_recognises_an_already_matching_decision(tmp_path, store) -> None:
    keep_folder = tmp_path / "Keep"
    path = _write(keep_folder / "a.jpg", b"a")
    store.set_review_decision(path, REVIEW_KEEP)

    plan = build_plan(store, keep_folder=keep_folder)

    assert plan.keep.already_matching == 1
    assert plan.keep.will_change == 0


def test_build_plan_recognises_a_decision_that_will_change(tmp_path, store) -> None:
    keep_folder = tmp_path / "Keep"
    path = _write(keep_folder / "a.jpg", b"a")
    store.set_review_decision(path, REVIEW_REJECT)  # currently Reject, folder says Keep

    plan = build_plan(store, keep_folder=keep_folder)

    assert plan.keep.will_change == 1
    assert plan.keep.already_matching == 0


def test_neutral_folder_targets_clearing_the_decision(tmp_path, store) -> None:
    neutral_folder = tmp_path / "Neutral"
    already_neutral = _write(neutral_folder / "a.jpg", b"a")
    has_a_decision = _write(neutral_folder / "b.jpg", b"b")
    store.set_review_decision(has_a_decision, REVIEW_KEEP)

    plan = build_plan(store, neutral_folder=neutral_folder)

    assert plan.neutral.already_matching == 1  # a.jpg: never decided = already Neutral
    assert plan.neutral.will_change == 1  # b.jpg: has a Keep decision, must be cleared


def test_all_three_folders_together(tmp_path, store) -> None:
    keep = _write(tmp_path / "Keep" / "a.jpg", b"a")
    reject = _write(tmp_path / "Reject" / "b.jpg", b"b")
    neutral = _write(tmp_path / "Neutral" / "c.jpg", b"c")

    plan = build_plan(
        store, keep_folder=keep.parent, reject_folder=reject.parent, neutral_folder=neutral.parent,
    )

    # c.jpg (Neutral folder) was never decided, so it is already Neutral -
    # only a.jpg (Keep) and b.jpg (Reject) actually change anything.
    assert plan.totals() == {
        "keep": 1, "reject": 1, "neutral": 1, "already_matching": 1, "will_change": 2, "conflicts": 0,
    }


def test_folders_are_all_optional(tmp_path, store) -> None:
    plan = build_plan(store)
    assert plan.totals() == {
        "keep": 0, "reject": 0, "neutral": 0, "already_matching": 0, "will_change": 0, "conflicts": 0,
    }


def test_missing_folder_raises_file_not_found(tmp_path, store) -> None:
    with pytest.raises(FileNotFoundError):
        build_plan(store, keep_folder=tmp_path / "does-not-exist")


def test_the_same_image_under_two_folders_is_a_conflict_not_applied_to_either(tmp_path, store) -> None:
    """Matched by content identity, not path - a copy of the same bytes
    under both Keep and Reject is genuinely ambiguous and must never be
    silently resolved one way."""
    keep_folder = tmp_path / "Keep"
    reject_folder = tmp_path / "Reject"
    _write(keep_folder / "a.jpg", b"identical bytes")
    _write(reject_folder / "a_copy.jpg", b"identical bytes")  # same content, different name/location

    plan = build_plan(store, keep_folder=keep_folder, reject_folder=reject_folder)

    assert plan.keep.paths == []
    assert plan.reject.paths == []
    assert len(plan.conflicts) == 2
    assert plan.totals()["conflicts"] == 2


def test_an_empty_file_is_skipped_not_a_crash(tmp_path, store) -> None:
    keep_folder = tmp_path / "Keep"
    _write(keep_folder / "empty.jpg", b"")  # image_identity raises IdentityUnavailable for empty files
    _write(keep_folder / "a.jpg", b"a")

    plan = build_plan(store, keep_folder=keep_folder)

    assert len(plan.keep.paths) == 1
    assert plan.keep.skipped == [str(keep_folder / "empty.jpg")]


def test_apply_plan_writes_keep_reject_and_clears_neutral(tmp_path, store) -> None:
    keep = _write(tmp_path / "Keep" / "a.jpg", b"a")
    reject = _write(tmp_path / "Reject" / "b.jpg", b"b")
    neutral_target = _write(tmp_path / "Neutral" / "c.jpg", b"c")
    store.set_review_decision(neutral_target, REVIEW_KEEP)  # must be cleared

    plan = build_plan(
        store, keep_folder=keep.parent, reject_folder=reject.parent, neutral_folder=neutral_target.parent,
    )
    result = apply_plan(store, plan)

    assert result == {
        "updated_keep": 1, "updated_reject": 1, "updated_neutral": 1, "skipped": [], "conflicts": [],
    }
    decisions = {row["image_path"]: row["decision"] for row in store.review_decisions()}
    assert decisions[str(keep)] == REVIEW_KEEP
    assert decisions[str(reject)] == REVIEW_REJECT
    assert str(neutral_target) not in decisions  # cleared, not merely set to something else


def test_apply_plan_never_touches_conflicting_images(tmp_path, store) -> None:
    keep_folder = tmp_path / "Keep"
    reject_folder = tmp_path / "Reject"
    _write(keep_folder / "a.jpg", b"identical bytes")
    _write(reject_folder / "a_copy.jpg", b"identical bytes")

    plan = build_plan(store, keep_folder=keep_folder, reject_folder=reject_folder)
    result = apply_plan(store, plan)

    assert result["updated_keep"] == 0
    assert result["updated_reject"] == 0
    assert len(result["conflicts"]) == 2
    assert store.review_decision_count() == 0


def test_apply_plan_is_idempotent_for_already_matching_images(tmp_path, store) -> None:
    keep = _write(tmp_path / "Keep" / "a.jpg", b"a")
    store.set_review_decision(keep, REVIEW_KEEP)

    plan = build_plan(store, keep_folder=keep.parent)
    result = apply_plan(store, plan)

    assert result["updated_keep"] == 1
    decisions = {row["image_path"]: row["decision"] for row in store.review_decisions()}
    assert decisions[str(keep)] == REVIEW_KEEP
