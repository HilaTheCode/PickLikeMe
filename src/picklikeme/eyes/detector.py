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
    # Whether the caller (EyeFilter) should trust this detection at all -
    # the actual filtering decision, computed from confidence AND from
    # left/right agreement (see superanimal_bird.py's module docstring for
    # why confidence alone is not sufficient). `box`/`confidence`/`center`
    # are still populated even when this is False, specifically so a
    # debugging overlay can show what a REJECTED image's eye guess looked
    # like - that is the whole point of investigating a wrongly-accepted or
    # wrongly-rejected image after the fact.
    accepted: bool = False


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


def build_eye_detector(name: str = "superanimal-bird", **kwargs) -> EyeDetector:
    """Construct a registered eye detector by name.

    A factory rather than a direct import for the same reason
    `species.classifier.build_classifier` is one: the concrete detector
    imports torch/timm inside its own `__init__`, so merely importing this
    module - which the ranking registry does at startup to list the
    available strategies - never pays for a heavy ML import that a session
    ranking with the AI model would never use.
    """
    from .superanimal_bird import SuperAnimalBirdEyeDetector

    detectors: dict[str, type] = {
        SuperAnimalBirdEyeDetector.detector_id: SuperAnimalBirdEyeDetector,
    }
    try:
        cls = detectors[name]
    except KeyError:
        raise ValueError(
            f"Unknown eye detector {name!r}. Available: {', '.join(sorted(detectors))}"
        ) from None
    return cls(**kwargs)
