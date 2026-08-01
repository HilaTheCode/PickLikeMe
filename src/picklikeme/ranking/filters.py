"""Phase 1: deciding whether an image participates in scoring at all.

Filtering and scoring are kept completely separate. A filter answers one
yes/no question and, when the answer is no, says *why* with an explicit
reason code; it never computes a metric, and nothing in `metrics.py` ever
decides whether an image belongs in the ranking. That separation is what
lets a future strategy reuse one without the other.

Reason codes are plain strings in `REJECT_REASONS`, not an enum with
exhaustive matches anywhere, precisely so adding a filter is adding a class
and one constant - no caller has to grow a new branch, because every caller
treats a reason as an opaque label to count and display.

Adding a filter is:

    1. a new reason constant, added to REJECT_REASONS,
    2. a class with `reason` and `check`,
    3. one entry in the list the strategy builds (see `ranking.classic`).

The order filters run in matters and is the caller's choice: `FilterChain`
stops at the first rejection, so the cheapest and most fundamental questions
("is there a subject at all?") come first and the expensive model-backed
ones last.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from ..eyes import EyeDetection, EyeDetector

# Nothing the subject detector recognises is in the frame. The image is not
# "badly composed" - there is no animal in it to judge at all.
NO_SUBJECT = "NO_SUBJECT"

# A subject was found, but no eye on it was visible enough to locate. The
# canonical wildlife reject: the bird turned away, or motion blur smeared the
# head past the point where an eye can be resolved.
NO_VISIBLE_EYE = "NO_VISIBLE_EYE"

# A subject was found, but no eye detector available here covers that kind of
# animal - today the eye model is bird-specific (see
# eyes.superanimal_bird), and running it on a mammal produces a confident,
# wrong answer rather than an honest miss. Reported separately from
# NO_VISIBLE_EYE because the two mean genuinely different things: this one is
# a gap in PeakPic's coverage, not a judgement about the photograph, and a
# photographer looking at "why was my tiger shot skipped" deserves to be told
# which.
UNSUPPORTED_SUBJECT = "UNSUPPORTED_SUBJECT"

REJECT_REASONS: tuple[str, ...] = (NO_SUBJECT, NO_VISIBLE_EYE, UNSUPPORTED_SUBJECT)

REJECT_REASON_LABELS: dict[str, str] = {
    NO_SUBJECT: "No subject detected",
    NO_VISIBLE_EYE: "No visible eye",
    UNSUPPORTED_SUBJECT: "No eye detector for this subject",
}


@dataclass
class FilterCandidate:
    """Everything the filters may look at, for one image.

    Populated once by the strategy from data this project already has on
    disk - the cached subject crop and the detection record preprocessing
    wrote beside it - so no filter re-decodes a RAW or re-runs the subject
    detector.

    `eye` is the one mutable field: `EyeFilter` runs the eye detector (the
    only genuinely new, expensive work in the chain) and records its result
    here, so the scoring phase reuses it rather than detecting a second time.
    """

    image_path: str
    # The cached subject crop, RGB. None when the crop could not be read.
    subject_crop: "np.ndarray | None" = None
    # The selected detection's box in FULL-FRAME pixels, and the frame's own
    # (width, height) - both straight from the detection sidecar.
    subject_box: tuple[float, float, float, float] | None = None
    source_size: tuple[int, int] | None = None
    # The COCO class of the selected detection (see bird_crop), used to ask an
    # eye detector whether it covers this animal at all.
    subject_label: int | None = None
    # Filled in by EyeFilter whenever the eye detector actually ran, whether
    # or not the result was trusted (see EyeDetection.accepted) - None only
    # ever means "the detector never ran at all" (no subject, or an
    # unsupported one). The scoring phase requires `.accepted`; a debugging
    # overlay (review.thumbnails.eye_keypoints_for) does not, and reads this
    # even for a rejected image.
    eye: "EyeDetection | None" = None


class ImageFilter(Protocol):
    """One yes/no question about a candidate.

    `reason` is the code reported when `check` returns False - a fixed
    property of the filter, so the chain never has to be told what a given
    filter's rejection means.
    """

    @property
    def reason(self) -> str: ...

    def check(self, candidate: FilterCandidate) -> bool: ...


class SubjectFilter:
    """Filter 1: is there a detected subject at all?

    Reads the detection the crop cache already recorded rather than running
    the detector - by the time a candidate reaches here, `preprocess.build_cache`
    has already detected every image in the folder exactly once.
    """

    reason = NO_SUBJECT

    def check(self, candidate: FilterCandidate) -> bool:
        return candidate.subject_box is not None and candidate.subject_crop is not None


class EyeFilter:
    """Filter 2: is at least one eye visible?

    One eye is enough, deliberately - see
    `SuperAnimalBirdEyeDetector.detect` for why requiring both would reject
    most of a wildlife archive.

    Also the only filter that can reject for a *second* reason: a subject the
    configured detector does not cover is reported as UNSUPPORTED_SUBJECT, not
    as a missing eye. `reason` therefore reports the last rejection's cause
    rather than being a constant - which is exactly why `FilterChain` reads it
    after `check` rather than caching it up front.
    """

    def __init__(self, detector: "EyeDetector") -> None:
        self._detector = detector
        self._reason = NO_VISIBLE_EYE

    @property
    def reason(self) -> str:
        return self._reason

    def check(self, candidate: FilterCandidate) -> bool:
        self._reason = NO_VISIBLE_EYE
        if candidate.subject_crop is None:
            return False
        if candidate.subject_label is not None and not self._detector.supports(candidate.subject_label):
            self._reason = UNSUPPORTED_SUBJECT
            return False
        candidate.eye = self._detector.detect(candidate.subject_crop)
        return candidate.eye.accepted


class FilterChain:
    """Runs filters in order and stops at the first rejection.

    Short-circuiting is not just an optimisation here: the expensive filter
    (eye detection, a neural network forward pass) must never run for an
    image that has no subject to look at in the first place.
    """

    def __init__(self, filters: list[ImageFilter]) -> None:
        self.filters = filters

    def reject_reason(self, candidate: FilterCandidate) -> str | None:
        """The reason this candidate is excluded from scoring, or None if it
        passed every filter."""
        for image_filter in self.filters:
            if not image_filter.check(candidate):
                return image_filter.reason
        return None
