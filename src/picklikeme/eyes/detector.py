"""The pluggable eye-detection boundary.

Anything that needs "where is this animal's eye, and how sure are we" asks
for an `EyeDetector` and gets back an `EyeDetection` - never a heatmap, a
keypoint index, or a model handle. Which model answers, what species it
covers, whether it runs on CPU or GPU is that class's own business, exactly
the way `species.classifier.SpeciesClassifier` already draws this line for
species identification.

Deliberately a separate package from `bird_crop` (subject detection) rather
than another method on `BirdDetector`: the two answer different questions,
are backed by unrelated models, and are wanted independently - the Classic
Vision ranking strategy needs both, a future "sharpest eye in the burst"
tool would need only this one, and subject detection is already a hard
dependency of training, which eye detection must never become.

Two things every caller can rely on regardless of which detector produced
them:

- **A box, not just a point.** Most animal-eye models available today are
  *keypoint* models: they regress an (x, y) location, not an extent. A
  caller measuring eye sharpness needs a region, so deriving a box is
  someone's job - and doing it here, once, behind the interface, is what
  keeps every caller from inventing its own slightly different rule. See
  `EyeDetection.box` and `SuperAnimalBirdEyeDetector` for how the current
  one derives it.
- **`None` means "no usable eye".** An unsupported subject, a low-confidence
  guess, an eye facing away from the camera and a model that failed to load
  all collapse to the same answer, so a caller never has to separately
  inspect a confidence to decide whether it has something to work with.
  This mirrors `species.classifier.UNKNOWN_SPECIES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


@dataclass(frozen=True)
class EyeKeypoint:
    """One raw eye-channel prediction - what the keypoint model itself said,
    before any accept/reject decision is layered on top.

    Kept completely separate from whether it was ultimately trusted: the
    debugging overlay (see `review.thumbnails`) wants to show BOTH eye
    channels and their confidences regardless of whether `EyeDetection`
    accepted the image, precisely because seeing what a rejected (or
    wrongly-accepted) image's eye channels actually found is the whole point
    of a debugging tool - see `EyeDetection.accepted`.
    """

    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class EyeDetection:
    """One detector's answer for one subject crop - always returned, never
    `None`, so the raw keypoints remain inspectable even for an image the
    detector ultimately rejects (see `accepted`).

    Coordinates are pixels *in the image that was passed to `detect`* - the
    subject crop, not the full frame - because that is the only frame of
    reference the detector is given. A caller holding both (see
    `ranking.classic`) maps them back itself; the detector never guesses at
    a coordinate system it was not shown.
    """

    # (x1, y1, x2, y2) around the PRIMARY eye (whichever of left/right is
    # more confident) - the region `ranking.metrics.region_focus_measure`
    # crops for the eye-sharpness metric. Always a real region with
    # non-zero area, so a caller can crop it without a special case, even
    # when `accepted` is False - see the module docstring on why the
    # interface promises a box even when the underlying model is a keypoint
    # regressor, and see `superanimal_bird.py` for why this box is populated
    # from the detector's best guess regardless of acceptance (a rejected
    # image's "what it would have used" is exactly what a debugging overlay
    # needs to show).
    box: tuple[float, float, float, float]
    # The primary eye's confidence in [0, 1]. Comparable across images from
    # the same detector, not across different detectors.
    confidence: float
    # The primary eye's own keypoint, when the backing model is
    # keypoint-based (None if a future detector regresses a box directly).
    center: tuple[float, float] | None = None
    # Which detector produced this, e.g. "superanimal-bird". Stable and
    # short; recorded alongside cached results so an answer from one
    # detector is never served to a caller now using another - the same
    # rule `species.classifier.SpeciesClassifier.classifier_id` states.
    detector_id: str = ""
    # The raw left/right eye-channel predictions, independent of which one
    # (if either) became the "primary" box above. Both are populated
    # whenever the detector ran at all - a single forward pass already
    # computes every channel - regardless of confidence or agreement.
    # Used by the debugging overlay (see `review.thumbnails.eye_keypoints_for`)
    # to show "if both left and right eye are detected, display both", and
    # by `superanimal_bird.py`'s own agreement check as the signal that
    # decides `accepted`.
    left: EyeKeypoint | None = None
    right: EyeKeypoint | None = None
    # Whether the EYE itself should be trusted - confidence AND (depending on
    # the backend) a plausibility check, but NOT whether the subject's head
    # was visible/localised at all in the first place (see `head_visible`
    # below - a deliberately separate, independent question; EyeFilter
    # checks both, with its own rejection reason for each - see
    # docs/EyePose_Investigation_Phase_1.md's Part 2/3). `box`/`confidence`/
    # `center` are still populated even when this is False, specifically so
    # a debugging overlay can show what a REJECTED image's eye guess looked
    # like - that is the whole point of investigating a wrongly-accepted or
    # wrongly-rejected image after the fact.
    accepted: bool = False
    # A holistic "is a real head instance actually present here" signal,
    # independent of any single landmark's own confidence - None for a
    # backend that does not compute one (e.g. SuperAnimal-Bird has no
    # equivalent single scalar today), in which case `head_visible` below
    # stays at its default (True: never gates a backend that cannot answer
    # this question). For EyePose-v0 this is the winning anchor's own
    # pre-decode "is this a bird head instance" score - see
    # `eyepose_v0.head_visible`'s own docstring for why this catches a
    # failure mode per-landmark confidence (including the primary eye's own)
    # cannot: a crop containing no real bird head at all can still produce a
    # confident-looking guess for where "the eye" would be.
    head_confidence: float | None = None
    # Whether `head_confidence` cleared the configured threshold - checked by
    # EyeFilter as a gate independent of `accepted`, with its own rejection
    # reason (LOW_HEAD_CONFIDENCE). Defaults to True (never rejects) so a
    # backend that leaves `head_confidence` at None is never affected.
    head_visible: bool = True
    # The other four of EyePose-v0's own six-landmark set (see
    # eyepose_v0.KPT_NAMES) - computed on every forward pass alongside
    # left/right eye, previously discarded before detect() ever returned
    # them (found while building the Image Inspector's landmark overlay:
    # the data already existed, nothing read it). None for a backend that
    # does not predict body landmarks at all (SuperAnimal-Bird only ever
    # locates the eye) - never fabricated, matching left/right's own
    # "populated whenever the detector ran at all" contract for whichever
    # backend actually computes it.
    beak: EyeKeypoint | None = None
    head_top: EyeKeypoint | None = None
    left_shoulder: EyeKeypoint | None = None
    right_shoulder: EyeKeypoint | None = None


class EyeDetector(Protocol):
    """Anything that can locate an eye in a subject crop.

    `supports(label)` exists because eye detectors are, in practice,
    taxon-specific: the only free pretrained model good enough to use here
    covers birds (see superanimal_bird.py), and running it on a tiger
    produces a confident, completely wrong answer rather than an honest
    "I don't know" - verified on this project's own crop cache. A caller
    that knows what it detected (this project always does; `bird_crop`
    records the COCO class beside every cached crop) can therefore ask
    first, and the filter layer turns a False into its own explicit reject
    reason rather than silently mislabelling it as "no visible eye".
    """

    @property
    def detector_id(self) -> str: ...

    def supports(self, coco_label: int) -> bool: ...

    def detect(self, subject_crop_rgb: "np.ndarray") -> EyeDetection: ...


def derive_eye_box(
    center_x: float, center_y: float, width: int, height: int, frac: float, min_px: float
) -> tuple[float, float, float, float]:
    """A square region around a single (x, y) eye point, clamped inside the
    crop it was found in - the "keypoint -> region" derivation every current
    and future keypoint-based `EyeDetector` needs (see the module docstring's
    "A box, not just a point").

    Extracted from `superanimal_bird.SuperAnimalBirdEyeDetector.detect` so a
    second detector (e.g. `eyepose_v0`) shares the exact same clamping
    arithmetic rather than a second, potentially slightly different copy of
    it - both backends' boxes must mean the same thing to `ranking.metrics`
    and the overlay.

    `frac` is a fraction of the crop's *shorter* side (so the box scales with
    the subject, not a fixed pixel size); `min_px` floors it so a tiny crop
    still yields a region a sharpness measure can work with.
    """
    side = max(min_px, frac * min(width, height))
    half = side / 2.0
    x1 = max(0.0, min(center_x - half, width - 1.0))
    y1 = max(0.0, min(center_y - half, height - 1.0))
    x2 = min(float(width), max(x1 + 1.0, center_x + half))
    y2 = min(float(height), max(y1 + 1.0, center_y + half))
    return (x1, y1, x2, y2)


def build_eye_detector(name: str = "superanimal-bird", **kwargs) -> EyeDetector:
    """Construct a registered eye detector by name.

    A factory rather than a direct import for the same reason
    `species.classifier.build_classifier` is one: each concrete detector
    imports its own heavy ML runtime (torch/timm for SuperAnimal-Bird,
    onnxruntime for eyepose_v0) inside its own `__init__`, so merely
    importing this module - which the ranking registry does at startup to
    list the available strategies - never pays for either.

    Registering a third backend later is exactly this: one import line, one
    entry in `detectors` below. Classic Vision, the cache, the overlay and
    everything else consume only the `EyeDetector`/`EyeDetection` interface
    above and never need to change.
    """
    from .eyepose_v0 import EyePoseV0EyeDetector
    from .superanimal_bird import SuperAnimalBirdEyeDetector

    detectors: dict[str, type] = {
        SuperAnimalBirdEyeDetector.detector_id: SuperAnimalBirdEyeDetector,
        EyePoseV0EyeDetector.detector_id: EyePoseV0EyeDetector,
    }
    try:
        cls = detectors[name]
    except KeyError:
        raise ValueError(
            f"Unknown eye detector {name!r}. Available: {', '.join(sorted(detectors))}"
        ) from None
    return cls(**kwargs)
