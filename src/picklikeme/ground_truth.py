"""Set User Decisions by Subfolders - bulk-seeding `review_status` (Keep/
Reject/Neutral) from three existing folders of already-sorted images, for
building Ground Truth to evaluate the algorithm against.

**Not an import.** Nothing here reads image bytes for any purpose other than
computing identity, copies a file, or moves a file - only rows in
`review_decisions` change (see `AnnotationStore.set_review_decision`/
`clear_review_decision`). A photographer who already sorted a shoot into
Lightroom collections/folders (Keep/Reject/Neutral) gets those judgements
into PickLikeMe without re-clicking through every image.

Matched by content identity (`identity.image_identity`), never only by
path - the whole point of Ground Truth is comparing the photographer's
verdict against the algorithm's for images that may have been ranked,
reviewed, or renamed under one path and now live at a completely different
one inside these three folders. See identity.py's own module docstring for
why a path-only match would be wrong here specifically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .analyzer.annotations import REVIEW_KEEP, REVIEW_REJECT, AnnotationStore
from .identity import IdentityUnavailable, image_identity

# Broader than dataset.py's ALLOWED_RAW_EXTENSIONS (RAW-only, for "rank a
# card straight off the camera") - these folders are photographer-curated
# output, exactly as likely to hold JPEG exports as the original RAWs.
IMAGE_EXTENSIONS = frozenset({
    ".nef", ".arw", ".cr2", ".cr3", ".dng", ".raf", ".orf", ".rw2",
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
})

# The three folder roles, and what `review_status` each one means - "keep"/
# "reject" are AnnotationStore's own REVIEW_KEEP/REVIEW_REJECT decision
# strings; `None` means Neutral, i.e. no decision recorded at all (see
# `review.session.ReviewSession.set_review_status`'s own "Neutral is a real,
# explicit choice... clearing the stored decision" - the same meaning here).
KEEP = "keep"
REJECT = "reject"
NEUTRAL = "neutral"
_TARGET_DECISION: dict[str, str | None] = {KEEP: REVIEW_KEEP, REJECT: REVIEW_REJECT, NEUTRAL: None}


def _iter_image_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


@dataclass
class RoleAssessment:
    """One folder role's candidates, already compared against what
    AnnotationStore currently has on record for each - the material both
    the preview (counts only) and the apply step (the same paths, now
    written) need, computed once so the two can never disagree."""

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
    # Images whose identity was found under more than one of the three
    # folders - genuinely ambiguous ("is this a Keep or a Reject?"), so
    # never applied automatically; the photographer must resolve these by
    # hand (move the file out of one of the folders) and re-run.
    conflicts: list[str] = field(default_factory=list)

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
    keep_folder: str | Path | None = None,
    reject_folder: str | Path | None = None,
    neutral_folder: str | Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> GroundTruthPlan:
    """Walk whichever of the three folders were given (each is optional -
    see the dialog's own "all folders are optional" requirement),
    recursively, and compare every image found against AnnotationStore's
    current decision for it - one bulk read (`review_decisions()`), not one
    query per image, the same discipline `ReviewSession`'s own fast-path
    identity matching already uses and for the same reason (this can easily
    be thousands of images).

    Returns the full plan - `apply_plan` takes this exact object rather
    than re-walking the folders, so preview and apply can never see a
    different set of files (a file appearing/disappearing between the two
    calls would otherwise silently change the outcome).
    """
    current_decision = {row["image_hash"]: row["decision"] for row in store.review_decisions()}

    roles: dict[str, tuple[Path | None, str | None]] = {
        KEEP: (Path(keep_folder) if keep_folder else None, REVIEW_KEEP),
        REJECT: (Path(reject_folder) if reject_folder else None, REVIEW_REJECT),
        NEUTRAL: (Path(neutral_folder) if neutral_folder else None, None),
    }

    identity_by_path: dict[str, str] = {}
    role_by_identity: dict[str, list[str]] = {}
    assessments: dict[str, RoleAssessment] = {}

    all_files: list[tuple[str, Path]] = []
    for role, (folder, _target) in roles.items():
        assessments[role] = RoleAssessment(role=role)
        if folder is None:
            continue
        if not folder.is_dir():
            raise FileNotFoundError(f"{role.capitalize()} folder does not exist: {folder}")
        for path in _iter_image_files(folder):
            all_files.append((role, path))

    total = len(all_files)
    for index, (role, path) in enumerate(all_files, start=1):
        try:
            identity = image_identity(path)
        except IdentityUnavailable:
            assessments[role].skipped.append(str(path))
        else:
            identity_by_path[str(path)] = identity
            role_by_identity.setdefault(identity, []).append(role)
        if on_progress is not None:
            on_progress(index, total)

    conflicting_identities = {identity for identity, seen_roles in role_by_identity.items() if len(set(seen_roles)) > 1}

    for role, path in all_files:
        identity = identity_by_path.get(str(path))
        if identity is None or identity in conflicting_identities:
            continue
        assessment = assessments[role]
        assessment.paths.append(str(path))
        target = roles[role][1]
        if current_decision.get(identity) == target:
            assessment.already_matching += 1
        else:
            assessment.will_change += 1

    conflicts = sorted({
        path for path, identity in identity_by_path.items() if identity in conflicting_identities
    })

    return GroundTruthPlan(
        keep=assessments[KEEP], reject=assessments[REJECT], neutral=assessments[NEUTRAL], conflicts=conflicts,
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
