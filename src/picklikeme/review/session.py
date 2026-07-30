"""What the photographer is reviewing, and what would happen if they filed it.

All of the review application's decisions live here: which images exist, what
the AI ranking merely suggests, which review status the photographer has
actually given each image, and what `Arrange` would move where. No HTTP and no
SQL beyond the store's own methods, so every rule below is directly testable.

Three ideas do the work:

- **The gallery is the union of the ranking and the folder.** An image present
  on disk but absent from the ranking still appears, because the alternative is
  a photograph silently missing from a review of its own shoot.
- **AI metadata and the photographer's review status are completely separate.**
  A score, a rank, and the "AI suggests Keep/Reject" hint derived from them are
  read-only information the model produced. Every image's `review_status` is
  always exactly one of Keep, Reject or Neutral - the photographer's own,
  independent verdict - and nothing about the ranking ever changes it by
  itself. Clearing a decision (one image, or a whole bulk selection) always
  lands on Neutral, never silently on whatever the model would have picked.
- **Only Keep/Reject file anything.** `arrange()` reads `review_status` alone;
  a Neutral image - ranked highly or not ranked at all - is never moved, since
  "the photographer hasn't looked at it yet" is not a basis to sort it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..analyzer.annotations import REVIEW_KEEP, REVIEW_REASON_OTHER, REVIEW_REJECT, AnnotationStore
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

# The photographer's review status - always exactly one of these three,
# completely independent of the AI ranking. Reuses REVIEW_KEEP/REVIEW_REJECT
# (the values AnnotationStore's `review_decisions` table already stores) so
# writing one is a direct pass-through; REVIEW_STATUS_NEUTRAL has no row in
# that table at all - "no decision recorded" *is* Neutral, not a fourth state.
REVIEW_STATUS_KEEP = REVIEW_KEEP
REVIEW_STATUS_REJECT = REVIEW_REJECT
REVIEW_STATUS_NEUTRAL = "neutral"
REVIEW_STATUSES: frozenset[str] = frozenset({REVIEW_STATUS_KEEP, REVIEW_STATUS_REJECT, REVIEW_STATUS_NEUTRAL})


class InvalidReviewStatus(ValueError):
    """Raised for a review status outside REVIEW_STATUSES."""


@dataclass
class ReviewImage:
    """One row of the gallery."""

    image_path: str
    filename: str
    # AI metadata - read-only, never written by this application, never
    # affected by anything the photographer does in it.
    score: float | None = None
    rank: int | None = None
    # The photographer's own verdict: REVIEW_KEEP, REVIEW_REJECT, or None.
    # None *is* Neutral - see review_status - not "no opinion yet from
    # somewhere else". Meaningless without a decision, and always cleared
    # alongside one.
    decision: str | None = None
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

    @property
    def review_status(self) -> str:
        """The one true tri-state status this application ever shows or
        files by. Always Neutral until the photographer explicitly says
        otherwise - never inferred from the ranking."""
        return self.decision or REVIEW_STATUS_NEUTRAL

    def as_dict(self, ai_suggestion: str | None) -> dict:
        return {
            "image_path": self.image_path,
            "filename": self.filename,
            "score": self.score,
            "rank": self.rank,
            "review_status": self.review_status,
            # What the AI ranking alone would recommend at the current
            # threshold - "keep"/"reject", or None if unranked. Informational
            # only; see ReviewSession._ai_suggestions.
            "ai_suggestion": ai_suggestion,
            "reason": self.reason,
            "reason_note": self.reason_note,
            "missing_file": self.missing_file,
        }


class ReviewSession:
    """The reviewed folder, its ranking, and the photographer's own verdicts."""

    def __init__(
        self,
        input_folder: str | Path | None,
        store: AnnotationStore,
        *,
        ranking_file: str | Path | None = None,
        keep_percent: float = DEFAULT_SELECTION_PERCENTAGE,
    ):
        self.store = store
        self.keep_percent = validate_selection_percentage(keep_percent)
        if input_folder is None:
            # `picklikeme review` with no --input at all starts here: an
            # empty, folder-less gallery, so the page still has something to
            # render and the photographer picks a folder from inside it (see
            # review.server's /api/review/open-folder) rather than the
            # command failing to start.
            self._clear()
        else:
            self.open_folder(input_folder, ranking_file=ranking_file)

    def _clear(self) -> None:
        self.input_folder = None
        self.ranking_file = None
        self.run_metadata: dict = {}
        self.warnings: list[str] = ['No folder open yet. Click "Open Folder…" above to choose one.']
        self.images: list[ReviewImage] = []

    # -- loading ------------------------------------------------------------

    def open_folder(self, input_folder: str | Path, *, ranking_file: str | Path | None = None) -> None:
        """Point this session at a different folder - including one that has
        never been ranked at all. `_load_ranked` already treats a missing
        ranking as "every image unranked" rather than an error (see below);
        combined with every image always starting Neutral, that is exactly
        what browsing a fresh, un-ranked shoot needs - nothing but Keep/Reject
        to sort it, with no AI opinion in the mix at all.

        Manual decisions already on record for images under the new folder
        are picked up the same way any reload finds them, by path (see
        `_apply_decisions`) - opening a folder is not a reason to lose a
        photographer's prior work in it.
        """
        self.input_folder = Path(input_folder).resolve()
        self.ranking_file = Path(ranking_file) if ranking_file else ranking_path(self.input_folder)
        self.run_metadata = read_run_metadata(self.input_folder)
        self.warnings: list[str] = []
        self.load()

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
            self.warnings.append(f"No ranking at {self.ranking_file}; every image starts Neutral.")
            return []
        from ..analyzer.io import load_ranking

        try:
            ranking = load_ranking(self.ranking_file)
        except Exception as exc:  # noqa: BLE001 - a broken ranking must not end the session
            logger.warning("Could not read %s: %s", self.ranking_file, exc)
            self.warnings.append(f"Could not read the ranking ({exc}); every image starts Neutral.")
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

    # -- AI metadata (read-only) ---------------------------------------------

    def set_keep_percent(self, percent: float) -> float:
        """The AI suggestion threshold - what fraction of RANKED images the
        model's own ordering alone would flag as "keep". Purely informational
        (see _ai_suggestions): moving this never changes anyone's
        review_status, only what the AI-suggestion hint next to each image
        says, unless the photographer explicitly applies it (see
        apply_ai_suggestions)."""
        self.keep_percent = validate_selection_percentage(percent)
        return self.keep_percent

    @property
    def cut(self) -> int:
        """How many ranked images the AI threshold alone would flag as keep."""
        return selection_count(sum(1 for i in self.images if i.is_ranked), self.keep_percent)

    def _ai_suggestions(self) -> dict[str, str | None]:
        """What the AI ranking alone would recommend for each image at the
        current threshold - "keep"/"reject" for a ranked image, None for one
        the model never scored. Never written anywhere, never fed into
        review_status or arrange() by itself - purely the hint shown next to
        the photographer's own, independent verdict."""
        cut = self.cut
        suggestions: dict[str, str | None] = {}
        ranked_position = 0
        for image in self.images:
            if image.is_ranked:
                suggestions[image.image_path] = REVIEW_STATUS_KEEP if ranked_position < cut else REVIEW_STATUS_REJECT
                ranked_position += 1
            else:
                suggestions[image.image_path] = None
        return suggestions

    # -- the photographer's review status ------------------------------------

    def keep_paths(self) -> list[str]:
        return [i.image_path for i in self.images if i.review_status == REVIEW_STATUS_KEEP]

    def reject_paths(self) -> list[str]:
        return [i.image_path for i in self.images if i.review_status == REVIEW_STATUS_REJECT]

    def neutral_paths(self) -> list[str]:
        return [i.image_path for i in self.images if i.review_status == REVIEW_STATUS_NEUTRAL]

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.images),
            "keep": len(self.keep_paths()),
            "reject": len(self.reject_paths()),
            "neutral": len(self.neutral_paths()),
            "missing_file": sum(1 for i in self.images if i.missing_file),
        }

    def as_dict(self) -> dict:
        ai_suggestions = self._ai_suggestions()
        return {
            "input_folder": str(self.input_folder) if self.input_folder else None,
            "ranking_file": str(self.ranking_file) if self.ranking_file else None,
            "has_ranking": bool(self.ranking_file and self.ranking_file.is_file()),
            "keep_percent": self.keep_percent,
            "counts": self.counts(),
            "warnings": list(self.warnings),
            "run": self.run_metadata,
            "images": [image.as_dict(ai_suggestions.get(image.image_path)) for image in self.images],
        }

    # -- writes ---------------------------------------------------------------

    def set_review_status(
        self,
        image_path: str,
        status: str,
        *,
        reason: str | None = None,
        reason_note: str | None = None,
    ) -> str:
        """Record the photographer's Keep/Reject/Neutral for one image.

        `status` is always one of REVIEW_STATUSES - Neutral is a real,
        explicit choice here (clearing the stored decision), not the absence
        of one. `reason`/`reason_note` only mean anything alongside Keep or
        Reject - a reason is why an override was made, and Neutral is not an
        override of anything, so either is dropped when `status` is Neutral,
        and `reason_note` is dropped unless `reason` is REVIEW_REASON_OTHER
        (mirroring what the store itself enforces).

        Persisted immediately - a review session's work must never exist only
        in a browser tab. Returns the resulting review_status.
        """
        if status not in REVIEW_STATUSES:
            raise InvalidReviewStatus(f"status must be one of {sorted(REVIEW_STATUSES)}, got {status!r}")
        image = self._image_for(image_path)
        if status == REVIEW_STATUS_NEUTRAL:
            self.store.clear_review_decision(image.image_path)
            reason = None
            reason_note = None
        else:
            if reason != REVIEW_REASON_OTHER:
                reason_note = None
            self.store.set_review_decision(image.image_path, status, reason=reason, reason_note=reason_note)
        image.decision = None if status == REVIEW_STATUS_NEUTRAL else status
        image.reason = reason
        image.reason_note = reason_note
        return image.review_status

    def set_review_statuses(self, image_paths: list[str], status: str) -> dict:
        """Apply the same review status to many images at once - the
        multi-select bulk action's backend.

        No reason travels with a bulk action - reason fields exist to record
        a photographer's own judgement about ONE frame (eyes, focus), which
        does not generalise across an arbitrary batch. Reuses
        `set_review_status` per image, so every invariant it enforces still
        holds for each one; an invalid `status` raises immediately, before
        anything is written, since that is a caller bug rather than a
        per-image data issue.

        A path no longer in this gallery (the file moved, or a folder switch
        happened mid-selection) - or one whose identity can no longer be
        established (see IdentityUnavailable) - is skipped and reported
        rather than aborting the whole batch: a large multi-select is exactly
        the case likely to include a few stale ones, and everything already
        applied before that point has already been persisted (each
        `set_review_status` commits on its own), so an abort would silently
        throw away real, saved progress along with the report of what went
        wrong.
        """
        applied = 0
        failed: list[str] = []
        for image_path in image_paths:
            try:
                self.set_review_status(image_path, status)
            except (KeyError, IdentityUnavailable):
                failed.append(image_path)
            else:
                applied += 1
        return {"applied": applied, "failed": failed}

    def apply_ai_suggestions(self) -> dict:
        """Bulk-accept the AI's CURRENT suggestion for every ranked image
        still Neutral - a fast starting point for a very large shoot: review
        the exceptions by hand, let this handle the rest.

        Every image it touches becomes a real, independent, explicit Keep or
        Reject the moment this runs, indistinguishable afterwards from one
        set by hand - not an ongoing tie to the model or the threshold. Never
        touches an image the photographer has already ruled on (Keep or
        Reject), and never touches one the AI has no opinion about
        (unranked) - both are left exactly as they were.
        """
        suggestions = self._ai_suggestions()
        applied = 0
        for image in self.images:
            if image.review_status != REVIEW_STATUS_NEUTRAL:
                continue
            suggestion = suggestions.get(image.image_path)
            if suggestion is None:
                continue
            self.set_review_status(image.image_path, suggestion)
            applied += 1
        return {"applied": applied}

    def _image_for(self, image_path: str) -> ReviewImage:
        key = _key(image_path)
        for image in self.images:
            if _key(image.image_path) == key:
                return image
        raise KeyError(f"{image_path} is not part of this review session")

    def arrange(self, *, dry_run: bool = False) -> OrganizeResult:
        """File the shoot by the photographer's OWN review status alone: Keep
        to `_Selected`, Reject to `_Rejected`, Neutral left exactly where it
        is - ranked highly by the AI or not. The ranking never files anything
        by itself; only an explicit Keep or Reject does, whether set one
        image at a time or through a bulk action.

        On a real run the ranking and the stored decisions are both repointed
        at the new locations, so the folder can be reviewed again afterwards.
        """
        if self.input_folder is None:
            raise ValueError("No folder is open to arrange.")
        result = organize_by_decision(
            self.keep_paths(),
            self.reject_paths(),
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
