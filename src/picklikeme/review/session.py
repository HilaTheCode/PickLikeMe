"""What the photographer is reviewing, and what would happen if they filed it.

All of the review application's decisions live here: which images exist, where
the keep-percentage cut falls, which manual overrides beat it, and what
`Arrange` would move where. No HTTP and no SQL beyond the store's own methods,
so every rule below is directly testable.

Two ideas do the work:

- **The gallery is the union of the ranking and the folder.** An image present
  on disk but absent from the ranking still appears, because the alternative is
  a photograph silently missing from a review of its own shoot.
- **A manual decision always beats the threshold.** Moving the keep percentage
  re-sorts everything the photographer has not personally ruled on, and touches
  nothing they have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..analyzer.annotations import REVIEW_KEEP, REVIEW_REJECT, REVIEW_REASON_OTHER, AnnotationStore
from ..identity import IdentityUnavailable
from ..organize import (
    DEFAULT_SELECTION_PERCENTAGE,
    OrganizeResult,
    organize_by_decision,
    selection_count,
    validate_selection_percentage,
)
from ..sidecar import ranking_path, read_run_metadata, rewrite_ranking_paths

logger = logging.getLogger(__name__)

# What an image's badge says, and why it is where it is.
STATE_MANUAL_KEEP = "manual_keep"
STATE_MANUAL_REJECT = "manual_reject"
STATE_AUTO_SELECTED = "auto_selected"
STATE_AUTO_REJECTED = "auto_rejected"
STATE_UNRANKED = "unranked"


@dataclass
class ReviewImage:
    """One row of the gallery."""

    image_path: str
    filename: str
    score: float | None = None
    rank: int | None = None
    decision: str | None = None  # REVIEW_KEEP / REVIEW_REJECT, or None
    # Why a manual decision overrides the model - one of REVIEW_REASONS, or
    # None. Meaningless without a decision, and always cleared alongside one.
    reason: str | None = None
    # Free text, only meaningful alongside REVIEW_REASON_OTHER - see
    # AnnotationStore.set_review_decision.
    reason_note: str | None = None
    # Set when the ranking lists an image that is no longer on disk. Shown as a
    # placeholder rather than dropped, so the photographer can see the gap.
    missing_file: bool = False

    @property
    def is_ranked(self) -> bool:
        return self.score is not None

    def as_dict(self, state: str) -> dict:
        return {
            "image_path": self.image_path,
            "filename": self.filename,
            "score": self.score,
            "rank": self.rank,
            "decision": self.decision,
            "reason": self.reason,
            "reason_note": self.reason_note,
            "state": state,
            "missing_file": self.missing_file,
        }


class ReviewSession:
    """The reviewed folder, its ranking, and the photographer's overrides."""

    def __init__(
        self,
        input_folder: str | Path,
        store: AnnotationStore,
        *,
        ranking_file: str | Path | None = None,
        keep_percent: float = DEFAULT_SELECTION_PERCENTAGE,
    ):
        self.input_folder = Path(input_folder).resolve()
        self.store = store
        self.ranking_file = Path(ranking_file) if ranking_file else ranking_path(self.input_folder)
        self.keep_percent = validate_selection_percentage(keep_percent)
        self.run_metadata = read_run_metadata(self.input_folder)
        self.warnings: list[str] = []
        self.images: list[ReviewImage] = []
        self.load()

    # -- loading ------------------------------------------------------------

    def load(self) -> None:
        """(Re)build the gallery from the ranking, the folder, and the store."""
        ranked = self._load_ranked()
        on_disk = self._enumerate_folder()

        images: list[ReviewImage] = []
        seen: set[str] = set()
        for image in ranked:
            images.append(image)
            seen.add(_key(image.image_path))
        for path in on_disk:
            if _key(str(path)) in seen:
                continue
            images.append(ReviewImage(image_path=str(path), filename=path.name))

        # Best first; unranked last, since they have no score to place them by.
        images.sort(key=lambda i: (i.score is None, -(i.score or 0.0), i.filename))
        self.images = images
        self._apply_decisions()

    def _load_ranked(self) -> list[ReviewImage]:
        if not self.ranking_file.is_file():
            self.warnings.append(f"No ranking at {self.ranking_file}; every image is unranked.")
            return []
        from ..analyzer.io import load_ranking

        try:
            ranking = load_ranking(self.ranking_file)
        except Exception as exc:  # noqa: BLE001 - a broken ranking must not end the session
            logger.warning("Could not read %s: %s", self.ranking_file, exc)
            self.warnings.append(f"Could not read the ranking ({exc}); every image is unranked.")
            return []

        self.warnings.extend(ranking.warnings)
        return [
            ReviewImage(
                image_path=image.image_path,
                filename=image.filename,
                score=image.score,
                rank=image.rank,
                missing_file=not Path(image.image_path).is_file(),
            )
            for image in ranking.images
        ]

    def _enumerate_folder(self) -> list[Path]:
        """Images actually on disk, including any already filed by a previous
        arrange - re-reviewing an organized shoot has to see its own output."""
        from ..analyzer.io import enumerate_ground_truth
        from ..sidecar import SIDECAR_DIRNAME

        try:
            found = enumerate_ground_truth(self.input_folder)
        except FileNotFoundError as exc:
            self.warnings.append(str(exc))
            return []
        return [p for p in found if SIDECAR_DIRNAME not in p.parts]

    def _apply_decisions(self) -> None:
        """Attach stored manual decisions to the gallery.

        Matched on path, which is one query for the whole session. Content
        identity is the authority and is what a write uses, but resolving it
        for every image just to discover most were never decided would cost
        minutes on a cold cache (see identity.py). `reconcile_by_identity()`
        closes the gap for the few rows a path match misses.
        """
        rows = self.store.review_decisions()
        by_hash = {
            row["image_hash"]: (row["decision"], row.get("reason"), row.get("reason_note")) for row in rows
        }
        by_path = {
            _key(row["image_path"]): (row["decision"], row.get("reason"), row.get("reason_note"))
            for row in rows
        }
        self._decisions_by_hash = by_hash
        matched: set[str] = set()
        for image in self.images:
            decision, reason, reason_note = by_path.get(_key(image.image_path), (None, None, None))
            image.decision = decision
            image.reason = reason
            image.reason_note = reason_note
            # A decision matched to a path whose file is gone is not really
            # matched: the image it describes has moved, and its new copy is
            # elsewhere in the gallery waiting to be found by identity.
            if decision is not None and not image.missing_file:
                matched.add(_key(image.image_path))
        self._unmatched_decisions = len(rows) - len(matched)

    def reconcile_by_identity(self) -> int:
        """Recover decisions for images whose file has moved since it was
        reviewed, by falling back to content identity.

        Only runs when a stored decision failed to match any gallery image by
        path, and only hashes gallery images that are currently undecided - so
        the common case (nothing moved) costs nothing at all. Returns the
        number of decisions recovered.
        """
        if not getattr(self, "_unmatched_decisions", 0):
            return 0
        recovered = 0
        for image in self.images:
            if image.decision is not None or image.missing_file:
                continue
            try:
                digest = self.store.identity_of(image.image_path)
            except IdentityUnavailable:
                continue
            decision, reason, reason_note = self._decisions_by_hash.get(digest, (None, None, None))
            if decision is not None:
                image.decision = decision
                image.reason = reason
                image.reason_note = reason_note
                recovered += 1
        if recovered:
            logger.info("Recovered %d review decision(s) by content identity", recovered)
        return recovered

    # -- classification -----------------------------------------------------

    def set_keep_percent(self, percent: float) -> float:
        self.keep_percent = validate_selection_percentage(percent)
        return self.keep_percent

    @property
    def cut(self) -> int:
        """How many ranked images the threshold alone would select."""
        return selection_count(sum(1 for i in self.images if i.is_ranked), self.keep_percent)

    def states(self) -> dict[str, str]:
        """image path -> why it is where it is.

        The single place the threshold and the overrides are combined; the
        gallery, the counts and `arrange()` all read this, so the badge on a
        card can never disagree with where its file would actually go.
        """
        cut = self.cut
        states: dict[str, str] = {}
        ranked_position = 0
        for image in self.images:
            if image.decision == REVIEW_KEEP:
                state = STATE_MANUAL_KEEP
            elif image.decision == REVIEW_REJECT:
                state = STATE_MANUAL_REJECT
            elif image.is_ranked:
                state = STATE_AUTO_SELECTED if ranked_position < cut else STATE_AUTO_REJECTED
            else:
                state = STATE_UNRANKED
            if image.is_ranked:
                ranked_position += 1
            states[image.image_path] = state
        return states

    def selected_paths(self) -> list[str]:
        states = self.states()
        return [i.image_path for i in self.images if states[i.image_path] in (STATE_MANUAL_KEEP, STATE_AUTO_SELECTED)]

    def rejected_paths(self) -> list[str]:
        states = self.states()
        return [i.image_path for i in self.images if states[i.image_path] in (STATE_MANUAL_REJECT, STATE_AUTO_REJECTED)]

    def untouched_paths(self) -> list[str]:
        """Unranked and undecided: no basis to file them, so they stay put."""
        states = self.states()
        return [i.image_path for i in self.images if states[i.image_path] == STATE_UNRANKED]

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.images),
            "selected": len(self.selected_paths()),
            "rejected": len(self.rejected_paths()),
            "untouched": len(self.untouched_paths()),
            "manual": sum(1 for i in self.images if i.decision is not None),
            "missing_file": sum(1 for i in self.images if i.missing_file),
        }

    def as_dict(self) -> dict:
        states = self.states()
        return {
            "input_folder": str(self.input_folder),
            "ranking_file": str(self.ranking_file),
            "has_ranking": self.ranking_file.is_file(),
            "keep_percent": self.keep_percent,
            "counts": self.counts(),
            "warnings": list(self.warnings),
            "run": self.run_metadata,
            "images": [image.as_dict(states[image.image_path]) for image in self.images],
        }

    # -- writes -------------------------------------------------------------

    def set_decision(
        self,
        image_path: str,
        decision: str | None,
        reason: str | None = None,
        reason_note: str | None = None,
    ) -> str:
        """Record (or clear) a manual Keep/Reject and update the gallery.

        `reason` says why the photographer overrode the model - meaningless
        without a decision, so clearing the decision always clears it too,
        and setting one without a reason clears any reason left over from a
        previous decision on this image. `reason_note` is free text and only
        means anything alongside REVIEW_REASON_OTHER; the store drops it
        otherwise, and this mirrors that here too.

        Persisted immediately - a review session's work must never exist only
        in a browser tab.
        """
        image = self._image_for(image_path)
        if decision is None:
            self.store.clear_review_decision(image.image_path)
            reason = None
            reason_note = None
        else:
            if reason != REVIEW_REASON_OTHER:
                reason_note = None
            self.store.set_review_decision(image.image_path, decision, reason=reason, reason_note=reason_note)
        image.decision = decision
        image.reason = reason
        image.reason_note = reason_note
        return self.states()[image.image_path]

    def _image_for(self, image_path: str) -> ReviewImage:
        key = _key(image_path)
        for image in self.images:
            if _key(image.image_path) == key:
                return image
        raise KeyError(f"{image_path} is not part of this review session")

    def arrange(self, *, dry_run: bool = False) -> OrganizeResult:
        """File the shoot: selected to `_Selected`, rejected to `_Rejected`.

        Unranked, undecided images are passed to neither list, so they are left
        exactly where they are - the app never moves a file it has no basis for
        classifying.

        On a real run the ranking and the stored decisions are both repointed
        at the new locations, so the folder can be reviewed again afterwards.
        """
        result = organize_by_decision(
            self.selected_paths(),
            self.rejected_paths(),
            self.input_folder,
            dry_run=dry_run,
            announce=False,
        )
        if not dry_run and result.moves:
            rewrite_ranking_paths(self.input_folder, result.moves)
            self.store.repoint_review_decisions(result.moves)
            self.load()
        return result


def _key(path: str | Path) -> str:
    """Compare paths the way the filesystem does, so a ranking written with a
    different spelling (case, separators) still matches what is on disk."""
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:  # pragma: no cover - unresolvable path
        return str(path).casefold()
