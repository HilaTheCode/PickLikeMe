"""ground_truth.build_plan/apply_plan - "Set User Decisions by Subfolders"
Version 2: Keep and Reject each accept MULTIPLE subfolders; Neutral is
never folder-selected - every image under the Root Folder not inside a
selected Keep/Reject subfolder becomes Neutral automatically. Not an
import: only review_decisions rows change, matched by content identity
(see identity.py), never by path alone.
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


def test_images_under_a_keep_subfolder_become_keep(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    _write(root / "Selected" / "a.jpg", b"a")
    _write(root / "b.jpg", b"b")  # not under any Keep/Reject folder -> Neutral

    plan = build_plan(store, root_folder=root, keep_folders=[root / "Selected"])

    assert [Path(p).name for p in plan.keep.paths] == ["a.jpg"]
    assert [Path(p).name for p in plan.neutral.paths] == ["b.jpg"]
    assert plan.reject.paths == []


def test_multiple_keep_subfolders_all_contribute(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    _write(root / "Favorites" / "a.jpg", b"a")
    _write(root / "Portfolio" / "b.jpg", b"b")
    _write(root / "Print" / "c.jpg", b"c")

    plan = build_plan(
        store, root_folder=root,
        keep_folders=[root / "Favorites", root / "Portfolio", root / "Print"],
    )

    assert {Path(p).name for p in plan.keep.paths} == {"a.jpg", "b.jpg", "c.jpg"}


def test_multiple_reject_subfolders_all_contribute(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    _write(root / "Rejected" / "a.jpg", b"a")
    _write(root / "Trash" / "b.jpg", b"b")
    _write(root / "Delete" / "c.jpg", b"c")

    plan = build_plan(
        store, root_folder=root,
        reject_folders=[root / "Rejected", root / "Trash", root / "Delete"],
    )

    assert {Path(p).name for p in plan.reject.paths} == {"a.jpg", "b.jpg", "c.jpg"}


def test_neutral_is_never_folder_selected_and_covers_everything_else(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    _write(root / "Selected" / "a.jpg", b"a")
    _write(root / "Rejected" / "b.jpg", b"b")
    _write(root / "c.jpg", b"c")  # directly under root
    _write(root / "misc" / "d.jpg", b"d")  # an unrelated subfolder

    plan = build_plan(
        store, root_folder=root, keep_folders=[root / "Selected"], reject_folders=[root / "Rejected"],
    )

    assert {Path(p).name for p in plan.neutral.paths} == {"c.jpg", "d.jpg"}


def test_no_folders_selected_makes_everything_neutral(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    _write(root / "a.jpg", b"a")
    _write(root / "sub" / "b.jpg", b"b")

    plan = build_plan(store, root_folder=root)

    assert {Path(p).name for p in plan.neutral.paths} == {"a.jpg", "b.jpg"}
    assert plan.keep.paths == []
    assert plan.reject.paths == []


def test_a_folder_selected_for_both_keep_and_reject_is_a_conflict(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    overlap = root / "Ambiguous"
    _write(overlap / "a.jpg", b"a")

    plan = build_plan(store, root_folder=root, keep_folders=[overlap], reject_folders=[overlap])

    assert plan.keep.paths == []
    assert plan.reject.paths == []
    assert plan.neutral.paths == []
    assert [Path(p).name for p in plan.conflicts] == ["a.jpg"]


def test_a_reject_folder_nested_inside_a_keep_folder_conflicts_only_the_overlap(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    keep_folder = root / "Selected"
    nested_reject = keep_folder / "SecondLook_Reject"
    _write(keep_folder / "a.jpg", b"a")  # only under Keep
    _write(nested_reject / "b.jpg", b"b")  # under both -> conflict

    plan = build_plan(store, root_folder=root, keep_folders=[keep_folder], reject_folders=[nested_reject])

    assert [Path(p).name for p in plan.keep.paths] == ["a.jpg"]
    assert [Path(p).name for p in plan.conflicts] == ["b.jpg"]


def test_a_keep_folder_outside_the_root_folder_raises(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    root.mkdir()
    outside = tmp_path / "Elsewhere"
    outside.mkdir()

    with pytest.raises(ValueError, match="not the Root Folder or a subfolder"):
        build_plan(store, root_folder=root, keep_folders=[outside])


def test_a_missing_root_folder_raises_file_not_found(tmp_path, store) -> None:
    with pytest.raises(FileNotFoundError):
        build_plan(store, root_folder=tmp_path / "does-not-exist")


def test_a_missing_keep_folder_raises_file_not_found(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        build_plan(store, root_folder=root, keep_folders=[root / "NoSuchFolder"])


def test_ignores_non_image_files(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    _write(root / "a.jpg", b"a")
    _write(root / "notes.txt", b"not an image")

    plan = build_plan(store, root_folder=root)

    assert [Path(p).name for p in plan.neutral.paths] == ["a.jpg"]


def test_already_matching_vs_will_change(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    keep_path = _write(root / "Selected" / "a.jpg", b"a")
    reject_path = _write(root / "Rejected" / "b.jpg", b"b")
    neutral_path = _write(root / "c.jpg", b"c")
    store.set_review_decision(keep_path, REVIEW_KEEP)  # already matches
    store.set_review_decision(reject_path, REVIEW_KEEP)  # will change to reject
    store.set_review_decision(neutral_path, REVIEW_KEEP)  # will change to neutral (cleared)

    plan = build_plan(store, root_folder=root, keep_folders=[root / "Selected"], reject_folders=[root / "Rejected"])

    assert plan.keep.already_matching == 1 and plan.keep.will_change == 0
    assert plan.reject.already_matching == 0 and plan.reject.will_change == 1
    assert plan.neutral.already_matching == 0 and plan.neutral.will_change == 1


def test_an_empty_file_is_skipped_not_a_crash(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    _write(root / "Selected" / "empty.jpg", b"")  # image_identity raises for empty files
    _write(root / "Selected" / "a.jpg", b"a")

    plan = build_plan(store, root_folder=root, keep_folders=[root / "Selected"])

    assert [Path(p).name for p in plan.keep.paths] == ["a.jpg"]
    assert [Path(p).name for p in plan.keep.skipped] == ["empty.jpg"]


def test_apply_plan_writes_keep_reject_and_clears_neutral(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    keep_path = _write(root / "Selected" / "a.jpg", b"a")
    reject_path = _write(root / "Rejected" / "b.jpg", b"b")
    neutral_path = _write(root / "c.jpg", b"c")
    store.set_review_decision(neutral_path, REVIEW_KEEP)  # must be cleared

    plan = build_plan(store, root_folder=root, keep_folders=[root / "Selected"], reject_folders=[root / "Rejected"])
    result = apply_plan(store, plan)

    assert result == {
        "updated_keep": 1, "updated_reject": 1, "updated_neutral": 1, "skipped": [], "conflicts": [],
    }
    decisions = {row["image_path"]: row["decision"] for row in store.review_decisions()}
    assert decisions[str(keep_path)] == REVIEW_KEEP
    assert decisions[str(reject_path)] == REVIEW_REJECT
    assert str(neutral_path) not in decisions  # cleared, not merely set to something else


def test_apply_plan_never_touches_conflicting_images(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    overlap = root / "Ambiguous"
    _write(overlap / "a.jpg", b"a")

    plan = build_plan(store, root_folder=root, keep_folders=[overlap], reject_folders=[overlap])
    result = apply_plan(store, plan)

    assert result["updated_keep"] == 0
    assert result["updated_reject"] == 0
    assert result["updated_neutral"] == 0
    assert len(result["conflicts"]) == 1
    assert store.review_decision_count() == 0


def test_apply_plan_is_idempotent_for_already_matching_images(tmp_path, store) -> None:
    root = tmp_path / "Shoot"
    keep_path = _write(root / "Selected" / "a.jpg", b"a")
    store.set_review_decision(keep_path, REVIEW_KEEP)

    plan = build_plan(store, root_folder=root, keep_folders=[root / "Selected"])
    result = apply_plan(store, plan)

    assert result["updated_keep"] == 1
    decisions = {row["image_path"]: row["decision"] for row in store.review_decisions()}
    assert decisions[str(keep_path)] == REVIEW_KEEP
