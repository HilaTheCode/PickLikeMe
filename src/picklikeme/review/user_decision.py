"""The photographer's own Keep / Reject / Undecided - one vocabulary, one file.

PeakPick carries three kinds of per-image information that look alike from a
distance and must never be substituted for one another:

1. **Algorithm result** - a ranking strategy's score/rank for an image, plus
   any keep/reject *suggestion* derived from it at the current cutoff. Read-
   only output of a model. Lives in `ReviewImage.ranking_results`.
2. **User Decision** - what the photographer said about this one photograph.
   Exactly one of the three values below, and *only* ever set by an explicit
   act in the review UI. This module.
3. **Crop/detection state** - what a detector found (or did not find) in the
   frame: a subject box, an eye, a filter reason. Lives in
   `ReviewImage.filter_reasons`/`metrics` and the detector caches.

`UNDECIDED` is a real, first-class value, not a missing one. "This image has
no row in `review_decisions`" and "this image is Undecided" are the same
statement, and neither of them is "Keep". Nothing may derive a decision from
a score, a rank, a cutoff, a filter verdict, the presence of a crop, an
image's position in a sort, or the mere fact that a strategy scored it - see
`analyzer.annotations.REVIEW_DECISION_SOURCES` for the persisted flag that
makes that impossible rather than merely discouraged.

The wire/storage spelling of "no decision" stays `"neutral"` (the value the
web review page, the desktop's own N button, and every existing test already
speak - see `review.session.REVIEW_STATUS_NEUTRAL`); `normalize` maps it onto
`UNDECIDED` so the three-state vocabulary has exactly one internal spelling
without a rename rippling through unrelated code.
"""

from __future__ import annotations

from ..analyzer.annotations import REVIEW_KEEP, REVIEW_REJECT

KEEP = REVIEW_KEEP
REJECT = REVIEW_REJECT
UNDECIDED = "undecided"

USER_DECISIONS: frozenset[str] = frozenset({KEEP, REJECT, UNDECIDED})

# In legend/reading order, for any UI that lists the three states.
USER_DECISION_ORDER: tuple[str, ...] = (KEEP, REJECT, UNDECIDED)
USER_DECISION_LABELS: dict[str, str] = {
    KEEP: "Keep",
    REJECT: "Reject",
    UNDECIDED: "Undecided",
}


def normalize(value: str | None) -> str:
    """Any spelling of "the photographer's verdict" -> one of the three.

    `None` and the legacy `"neutral"` both mean UNDECIDED; anything
    unrecognised does too, because the one thing this function must never do
    is turn an unknown into a Keep.
    """
    if value == KEEP:
        return KEEP
    if value == REJECT:
        return REJECT
    return UNDECIDED


def is_decided(value: str | None) -> bool:
    """True only for an explicit Keep or Reject - the single test every
    caller that files, colors or counts "decided" images should use."""
    return normalize(value) in (KEEP, REJECT)
