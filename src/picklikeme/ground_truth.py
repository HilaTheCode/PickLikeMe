"""Set User Decisions by Subfolders - bulk-seeding `review_status` (Keep/
Reject/Neutral) from photographer-curated subfolders of the current Root
Folder, for building Ground Truth to evaluate the algorithm against.

**Not an import.** Nothing here reads image bytes for any purpose other than
computing identity - only rows in `review_decisions` change (see
`AnnotationStore.set_review_decision`/`clear_review_decision`). No file is
ever copied or moved.

**Version 2 workflow** (replaces the original one-folder-per-role design):
Keep and Reject each accept MULTIPLE subfolders (a photographer's Lightroom-
style organization is rarely a single folder - "Selected", "Favorites",
"Portfolio" might all mean Keep). Neutral is never folder-selected at all -
every image found anywhere under the Root Folder that is not inside a
selected Keep or Reject subfolder is automatically Neutral. This requires
exactly one recursive walk, of the Root Folder alone; Keep/Reject folders
are identified by membership (is this image's path inside one of them), not
walked separately.

Matched by content identity (`identity.image_identity`), never only by
path - the whole point of Ground Truth is comparing the photographer's
verdict against the algorithm's for images that may have been ranked,
reviewed, or renamed under one path and now live at a completely different
one inside these folders. See identity.py's own module docstring for why a
path-only match would be wrong here specifically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .analyzer.annotations import REVIEW_KEEP, REVIEW_REJECT, AnnotationStore
from .identity import IdentityUnavailable

# Broader than dataset.py's ALLOWED_RAW_EXTENSIONS (RAW-only, for "rank a
# card straight off the camera") - a Root Folder is photographer-curated
# output, exactly as likely to hold JPEG exports as the original RAWs.
IMAGE_EXTENSIONS = frozenset({
    ".nef", ".arw", ".cr2", ".cr3", ".dng", ".raf", ".orf", ".rw2",
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
})

KEEP = "keep"
REJECT = "reject"
NEUTRAL = "neutral"
CONFLICT = "conflict"
# "keep"/"reject" are AnnotationStore's own REVIEW_KEEP/REVIEW_REJECT
# decision strings; `None` means Neutral - i.e. no decision recorded at all
# (see `review.session.ReviewSession.set_review_status`'s own "Neutral is a
# real, explicit choice... clearing the stored decision" - the same meaning
# here).
_TARGET_DECISION: dict[str, str | None] = {KEEP: REVIEW_KEEP, REJECT: REVIEW_REJECT, NEUTRAL: None}


def _iter_image_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _is_within(path: Path, folder: Path) -> bool:
    """True if `path` is `folder` itself or lives anywhere under it - both
    already resolved (absolute, symlinks collapsed) by the caller, so this
    is a pure path-segment comparison, never a filesystem call."""
    return path == folder or folder in path.parents


@dataclass
class RoleAssessment:
    """One role's (Keep/Reject/Neutral) candidates, already compared
    against what AnnotationStore currently has on record for each - the
    material both the preview (counts only) and the apply step (the same
    paths, now written) need, computed once so the two can never disagree."""

    role: str
    paths: list[str] = field(default_factory=list)
    already_matching: int = 0
    will_change: int = 0
    skipped: list[str] = field(default_factory=list)  # unreadable / identity unavailable


@dataclass
class GroundTruthPlan:
    keep: RoleAssessment
    reject: RoleAssessment
    neutral: RoleAssessment
    # Images found under more than one selected folder role (e.g. a Reject
    # folder accidentally nested inside a Keep folder, or the same subtree
    # picked for both) - genuinely ambiguous, so never applied automatically;
    # the photographer must resolve the overlapping selection and re-run.
    conflicts: list[str] = field(default_factory=list)
    root_folder: str = ""

    def totals(self) -> dict[str, int]:
        return {
            "keep": len(self.keep.paths),
            "reject": len(self.reject.paths),
            "neutral": len(self.neutral.paths),
            "already_matching": self.keep.already_matching + self.reject.already_matching + self.neutral.already_matching,
            "will_change": self.keep.will_change + self.reject.will_change + self.neutral.will_change,
            "conflicts": len(self.conflicts),
        }


def build_plan(
    store: AnnotationStore,
    *,
    root_folder: str | Path,
    keep_folders: list[str | Path] = (),
    reject_folders: list[str | Path] = (),
    on_progress: Callable[[int, int], None] | None = None,
) -> GroundTruthPlan:
    """One recursive walk of `root_folder`. Every image found is classified
    by which selected folder (if any) contains it: a Keep folder -> Keep, a
    Reject folder -> Reject, both -> Conflict, neither -> Neutral
    automatically - no folder to pick for Neutral, ever.

    `keep_folders`/`reject_folders` must each be `root_folder` itself or a
    folder somewhere under it - otherwise an image inside one could never be
    found by the single walk this function does, and would silently end up
    Neutral instead of Keep/Reject. Raises ValueError up front rather than
    silently mis-classifying.

    Returns the full plan - `apply_plan` takes this exact object rather
    than re-walking, so preview and apply can never see a different set of
    files (a file appearing/disappearing between the two calls would
    otherwise silently change the outcome).
    """
    root = Path(root_folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist: {root_folder}")

    keep_roots = [Path(f).resolve() for f in keep_folders]
    reject_roots = [Path(f).resolve() for f in reject_folders]
    for label, folders in (("Keep", keep_roots), ("Reject", reject_roots)):
        for folder in folders:
            if not folder.is_dir():
                raise FileNotFoundError(f"{label} folder does not exist: {folder}")
            if not _is_within(folder, root):
                raise ValueError(
                    f"{label} folder {folder} is not the Root Folder or a subfolder of it ({root}) - "
                    "an image inside it would never be found by the Root Folder walk."
                )

    current_decision = {row["image_hash"]: row["decision"] for row in store.review_decisions()}
    assessments = {role: RoleAssessment(role=role) for role in (KEEP, REJECT, NEUTRAL)}
    conflicts: list[str] = []

    files = list(_iter_image_files(root))
    total = len(files)
    for index, path in enumerate(files, start=1):
        resolved = path.resolve()
        in_keep = any(_is_within(resolved, folder) for folder in keep_roots)
        in_reject = any(_is_within(resolved, folder) for folder in reject_roots)

        if in_keep and in_reject:
            conflicts.append(str(path))
        else:
            role = KEEP if in_keep else REJECT if in_reject else NEUTRAL
            assessment = assessments[role]
            try:
                # store.identity_of (not the bare identity.image_identity)
                # so the result is cached against this exact path - apply_plan's
                # later set_review_decision/clear_review_decision calls hit
                # that cache instead of re-hashing every file a second time.
                identity = store.identity_of(path)
            except IdentityUnavailable:
                assessment.skipped.append(str(path))
            else:
                assessment.paths.append(str(path))
                target = _TARGET_DECISION[role]
                if current_decision.get(identity) == target:
                    assessment.already_matching += 1
                else:
                    assessment.will_change += 1

        if on_progress is not None:
            on_progress(index, total)

    return GroundTruthPlan(
        keep=assessments[KEEP], reject=assessments[REJECT], neutral=assessments[NEUTRAL],
        conflicts=sorted(conflicts), root_folder=str(root),
    )


def apply_plan(store: AnnotationStore, plan: GroundTruthPlan) -> dict:
    """Write every role's paths per `_TARGET_DECISION` - Keep/Reject set a
    decision, Neutral clears one. Idempotent: an "already matching" image is
    still (re-)written, exactly the same as any other - a plain, uniform
    loop with no separate "skip if unchanged" branch to keep in sync with
    what `build_plan` already decided was "matching" (recomputing that
    condition twice from two different code paths is how they'd eventually
    disagree)."""
    updated = {KEEP: 0, REJECT: 0, NEUTRAL: 0}
    skipped: list[str] = list(plan.keep.skipped) + list(plan.reject.skipped) + list(plan.neutral.skipped)

    for role, assessment in ((KEEP, plan.keep), (REJECT, plan.reject), (NEUTRAL, plan.neutral)):
        target = _TARGET_DECISION[role]
        for path in assessment.paths:
            try:
                if target is None:
                    store.clear_review_decision(path)
                else:
                    store.set_review_decision(path, target)
            except IdentityUnavailable:
                skipped.append(path)
            else:
                updated[role] += 1

    return {
        "updated_keep": updated[KEEP],
        "updated_reject": updated[REJECT],
        "updated_neutral": updated[NEUTRAL],
        "skipped": skipped,
        "conflicts": list(plan.conflicts),
    }
