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

# A subject was found and an eye detector covers it, but the detector itself
# was not confident a real head instance was even in the crop - independent
# of, and checked BEFORE, whether any specific eye landmark looks trustworthy
# (see eyes.detector.EyeDetection.head_visible's own docstring and
# eyes.eyepose_v0.head_visible for the measured signal this reads). Reported
# separately from NO_VISIBLE_EYE because the two mean genuinely different
# things: this one is "there was nothing head-shaped here to judge an eye
# on at all" (a wing covering the head, the bird facing away, a bad crop),
# NO_VISIBLE_EYE is "a head was there, but this particular eye still wasn't
# trustworthy" - see docs/EyePose_Investigation_Phase_1.md's Part 2/3.
LOW_HEAD_CONFIDENCE = "LOW_HEAD_CONFIDENCE"

REJECT_REASONS: tuple[str, ...] = (NO_SUBJECT, LOW_HEAD_CONFIDENCE, NO_VISIBLE_EYE, UNSUPPORTED_SUBJECT)

REJECT_REASON_LABELS: dict[str, str] = {
    NO_SUBJECT: "No subject detected",
    LOW_HEAD_CONFIDENCE: "Head not confidently detected",
    # "reliable" is deliberate (EyePose Investigation Phase 1, Part 3): this
    # also covers a spatially-found-but-untrusted eye (confidence below
    # threshold, or anatomically implausible), not only a literal absence of
    # any eye channel - "No visible eye" alone underclaimed what the gate
    # actually checks.
    NO_VISIBLE_EYE: "No reliable visible eye",
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
    # The selected detection's TIGHT box in FULL-FRAME pixels (pre-margin),
    # and the frame's own (width, height) - both straight from the detection
    # sidecar. Used for the subject-size metric and the subject-box overlay,
    # where the tight box is exactly what should be shown/measured.
    subject_box: tuple[float, float, float, float] | None = None
    # The crop's own rectangle in FULL-FRAME pixels - the tight `subject_box`
    # grown by the margin `bird_crop.build_crop` actually applied before
    # cropping (`bird_crop.CropResult.expanded_box`). `subject_crop`'s pixels
    # span THIS rectangle, not `subject_box` - anything projecting a
    # crop-space coordinate (e.g. an eye keypoint) back onto the full frame
    # must scale/offset against `crop_box`, never `subject_box`. Conflating
    # the two was a real, proven bug - see
    # docs/EyePose_Investigation_Phase_1.md's Q1 finding.
    crop_box: tuple[float, float, float, float] | None = None
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
    """Filter 2: is at least one eye visible - and, before that question is
    even asked, was a real head instance actually here to look at?

    One eye is enough, deliberately - see
    `SuperAnimalBirdEyeDetector.detect` for why requiring both would reject
    most of a wildlife archive.

    This is the "Decision Engine" from the EyePose Investigation Phase 1
    report's Part 3: three genuinely independent questions, each with its
    own reason, evaluated in this order once the eye detector has run
    (`SubjectFilter`, upstream, already answered "is there a subject at
    all" - `NO_SUBJECT`):

    1. Does a detector even cover this subject's class? -> `UNSUPPORTED_SUBJECT`
    2. Is the detector confident a real head instance is in the crop at all,
       independent of any individual landmark (`EyeDetection.head_visible`)?
       -> `LOW_HEAD_CONFIDENCE`
    3. Given a head is there, is the specific eye trustworthy
       (`EyeDetection.accepted`)? -> `NO_VISIBLE_EYE`

    Question 2 matters as its own, separate check because it catches a
    failure question 3 alone cannot: a crop with no real head can still
    produce a confident-*looking* individual eye landmark (measured, not
    hypothetical - see `eyes.eyepose_v0`'s own module docstring for the
    example). Checking it first means a photographer sees "no head" rather
    than a misleadingly specific "eye not visible" for that case.

    Only ONE call to the (expensive) detector regardless of how many of
    these three checks end up mattering - `detect()` already computes
    `head_visible` and `accepted` together in a single forward pass, this
    filter simply reads both off the one result. `reason` therefore reports
    whichever of the three questions actually failed, not a constant -
    which is exactly why `FilterChain` reads it after `check` rather than
    caching it up front.

    `gate_by_subject_label` (default True, matching every backend's original
    behavior): whether question 1 is even asked. It exists because the COCO
    class label behind `candidate.subject_label` is reliable only for the
    classes COCO genuinely has (bird, in particular) - see
    `eyes.domains.DomainProfile.gate_by_subject_label`'s own docstring for
    the measured case this documents (a Colobus monkey recorded as COCO
    class 16/"bird" at 0.99 confidence). A caller that has ALREADY decided
    subject eligibility some other way before constructing this filter's
    detector - `ranking.combined.ClassicVisionCombinedStrategy` classifies
    the crop itself, per Burst, via `eyes.domain_detector.ClipDomainDetector`,
    before ever building the per-image detector - must not have that
    decision second-guessed by a COCO label the whole point was to stop
    trusting for this. Passing False turns question 1 into a no-op (every
    candidate proceeds to question 2), because the detector handed to this
    filter is already known to be domain-appropriate by construction.
    """

    def __init__(self, detector: "EyeDetector", gate_by_subject_label: bool = True) -> None:
        self._detector = detector
        self._gate_by_subject_label = gate_by_subject_label
        self._reason = NO_VISIBLE_EYE

    @property
    def reason(self) -> str:
        return self._reason

    def check(self, candidate: FilterCandidate) -> bool:
        self._reason = NO_VISIBLE_EYE
        if candidate.subject_crop is None:
            return False
        if (
            self._gate_by_subject_label
            and candidate.subject_label is not None
            and not self._detector.supports(candidate.subject_label)
        ):
            self._reason = UNSUPPORTED_SUBJECT
            return False
        candidate.eye = self._detector.detect(candidate.subject_crop)
        if not candidate.eye.head_visible:
            self._reason = LOW_HEAD_CONFIDENCE
            return False
        self._reason = NO_VISIBLE_EYE
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
