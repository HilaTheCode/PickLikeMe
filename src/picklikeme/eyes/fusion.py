"""The Fusion/Validation layer: combines EyePose-v0 with one or more
complementary eye detectors into a single, more robust prediction.

    Head crop
        |
    +---+---+
    |       |
 EyePose-v0  Model B (SuperAnimal-Bird by default)
    |       |
    +---+---+
        |
   Fusion Layer            (this module: agreement, per-model trust)
        |
  Geometry validation      (eyes.geometry: in-bounds, left/right plausibility)
        |
  [Burst-level evidence]   (eyes.burst_consistency - a separate, later step;
        |                   see that module's own docstring for why it is
        |                   not inside detect() itself)
        v
  Final eye prediction

Why a fusion layer, and why not just replace EyePose-v0
----------------------------------------------------------
Measured on this project's own real crop cache (1,324 crops with a cached
EyePose-v0 result, SuperAnimal-Bird run fresh on the same crops - see the
accompanying benchmark report): EyePose-v0 alone accepts 55% of crops,
SuperAnimal-Bird alone accepts 20%; only 14% of crops are accepted by BOTH.
Manual review of stratified samples found EyePose-v0 to be the more reliable
single opinion (see the report's "SuperAnimal accepted, EyePose rejected"
category - SuperAnimal's own confident accepts there were very often on
shoulder/back fur, not an eye), which is why it keeps the larger default
weight below - but agreement between the two, when it happens, correlates
strongly with correctness, and disagreement strongly with at least one of
them being wrong. Neither model should be discarded; what was missing was
a layer that uses agreement, per-model trust, and geometric plausibility
together rather than trusting either model's own confidence in isolation -
see the module-level `STATUS_*` constants for the range of verdicts this
layer can reach, deliberately including "no reliable answer" rather than
always producing a point.

Model weight vs prediction confidence
----------------------------------------
These are kept as two separate numbers everywhere in this module. A model's
`weight` (`FusionConfig.model_weights`) is a fixed, configurable statement of
how much this project trusts that BACKEND in general; a prediction's own
`confidence` is what THAT model said about THIS crop. `_trust_score` is the
only place they are combined, and it also folds in whether the model's own
internal accept/reject gate passed (`EyeDetection.accepted`) - which is
already each backend's own geometric+confidence judgement, not duplicated
here (see `eyes.geometry`'s module docstring for what this layer adds on
top instead: things neither backend's own gate checks by construction,
because it only ever sees itself, never the other model).

Disagreement handling
------------------------
Two accepted predictions that are geometrically close (within
`agreement_threshold`, in head-scale units - see `eyes.geometry.
normalized_distance`) are fused as a trust-weighted average. Two that are
NOT close are never averaged (see the module docstring's worked example in
the project brief: averaging two genuinely different guesses produces a
point that is probably nowhere real) - instead the higher-trust prediction
wins outright only if it clears `disagreement_dominance_ratio` against the
other; otherwise fusion reports `STATUS_DISAGREEMENT` and declines to invent
a location, exactly the "uncertain / disagreement" preference the project
brief asks for over a confidently-wrong point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import EyeDetection, EyeKeypoint, derive_eye_box
from .geometry import (
    build_head_frame,
    eye_pair_disagreement,
    normalized_distance,
    point_in_bounds,
)

FUSION_DETECTOR_ID = "fusion-v1"

# Both models agreed (or only one had a trustworthy opinion), and the result
# clears `min_fused_confidence` - the box/center/confidence above are a real,
# trusted answer.
STATUS_AGREE = "AGREE"
STATUS_SINGLE_MODEL = "SINGLE_MODEL"
# Both accepted predictions disagreed by more than `agreement_threshold` and
# neither dominated the other enough to pick a winner (see
# `disagreement_dominance_ratio`) - `accepted` is False, no location is
# invented, though the two raw candidates remain inspectable via
# `left`/`right` for debugging (see `detect`'s own docstring).
STATUS_DISAGREEMENT = "DISAGREEMENT"
# One model dominated a disagreement clearly enough to be trusted alone -
# functionally a single-model result, but the status records that a
# conflicting opinion existed and was overruled, which SINGLE_MODEL alone
# would not tell a photographer.
STATUS_DISAGREEMENT_RESOLVED = "DISAGREEMENT_RESOLVED"
# At least one model ran, but nothing cleared its own accept/reject gate -
# the same "no reliable visible eye" a single backend already reports, kept
# under its own name here because "disagreement" would misdescribe it (the
# two may have quietly agreed on nothing).
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
# No sub-detector produced a usable point at all (e.g. every one saw an
# empty crop) - see `ModelPrediction.point`.
STATUS_NO_DETECTION = "NO_DETECTION"


@dataclass(frozen=True)
class ModelWeight:
    """One backend's fixed, configurable trust weight - see the module
    docstring's "Model weight vs prediction confidence" section. Weights
    need not sum to 1; `_normalized_weights` rescales whichever subset of
    configured models actually produced a prediction for a given crop, so
    adding a third model later is one more `ModelWeight`, not a rebalancing
    of the existing two."""

    detector_id: str
    weight: float


# Informed by this project's own 1,324-crop paired benchmark (see the
# accompanying report): EyePose-v0 was the more reliable single opinion in
# manual review of the disagreement cases, so it keeps the larger share -
# not the project brief's own example ratio, which it explicitly says not
# to assume, but a similar magnitude arrived at independently from this
# project's real data.
DEFAULT_MODEL_WEIGHTS: tuple[ModelWeight, ...] = (
    ModelWeight("eyepose-v0", 0.65),
    ModelWeight("superanimal-bird", 0.35),
)

# Two accepted predictions "agree" at or below this normalised distance
# (head-scale units - see eyes.geometry.normalized_distance). Informed by
# the same benchmark: among the 190 real crops both backends accepted, 41%
# fell within 0.50 head-scale units and manual review of the closest pairs
# (down to 0.01) found them consistently looking at the same real feature,
# while the most-disagreeing pairs (norm_dist 2-6) were essentially never
# the same feature. 0.40 sits inside that gap. Not independently validated
# against ground-truth bird annotations - this project's only available
# real, timestamped photographs were a non-bird species (see the report's
# benchmark-dataset section) - a reasoned starting point in the same spirit
# as EyePose-v0's own thresholds, not an empirically fitted one.
DEFAULT_AGREEMENT_THRESHOLD = 0.4

# When two accepted predictions disagree (above the threshold above), the
# higher-trust one is used alone only if it out-trusts the other by at
# least this ratio - otherwise neither is picked and the result is
# STATUS_DISAGREEMENT. 2.0 requires a real, substantial margin (roughly
# "twice as trusted"), not a coin-flip-close edge, before overruling a
# conflicting opinion outright - a starting point, not empirically fitted,
# chosen to err toward reporting disagreement rather than confidently
# picking a side on a close call (see the module docstring's disagreement
# section and the project brief's explicit preference for "uncertain" over
# "confidently wrong").
DEFAULT_DISAGREEMENT_DOMINANCE_RATIO = 2.0

# The fused confidence (after trust-weighting and any agreement bonus)
# must clear this before `accepted` is True. Matches both backends' own
# DEFAULT_MIN_CONFIDENCE (0.80 each) in spirit - fusion's own confidence
# scale is not the same arithmetic quantity as either backend's raw
# per-landmark score (it already folds in trust weight), so this is kept as
# its own, separately configurable number rather than reusing 0.80 as if it
# meant the same thing here.
DEFAULT_MIN_FUSED_CONFIDENCE = 0.5

# Same defaults as both backends' own eye-box derivation - see
# eyes.detector.derive_eye_box. Not model-specific; kept identical so a
# fused box is comparably sized to a single-model one for a fair overlay/
# sharpness comparison.
DEFAULT_EYE_BOX_FRAC = 0.08
MIN_EYE_BOX_PX = 12.0

# A prediction whose own backend rejected it (EyeDetection.accepted is
# False) is not discarded outright - a debugging overlay benefits from
# seeing it, and it can still act as a weak corroborating signal - but its
# trust score is scaled down sharply relative to an accepted one from the
# same backend.
REJECTED_TRUST_PENALTY = 0.35


@dataclass(frozen=True)
class FusionConfig:
    """Every tunable of the fusion layer, in one place - see the module
    docstring for how each is used, and `ranking.classic.
    ClassicVisionFusionParams` for how a photographer adjusts these from the
    desktop app."""

    model_weights: tuple[ModelWeight, ...] = DEFAULT_MODEL_WEIGHTS
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD
    disagreement_dominance_ratio: float = DEFAULT_DISAGREEMENT_DOMINANCE_RATIO
    min_fused_confidence: float = DEFAULT_MIN_FUSED_CONFIDENCE
    eye_box_frac: float = DEFAULT_EYE_BOX_FRAC
    min_eye_box_px: float = MIN_EYE_BOX_PX
    rejected_trust_penalty: float = REJECTED_TRUST_PENALTY

    def weight_for(self, detector_id: str) -> float:
        for model_weight in self.model_weights:
            if model_weight.detector_id == detector_id:
                return model_weight.weight
        return 0.0


@dataclass(frozen=True)
class _Candidate:
    """One sub-detector's prediction, reduced to what fusion needs to reason
    about - kept internal (leading underscore) because `EyeDetection`, not
    this, is what every caller outside this module consumes."""

    detector_id: str
    detection: EyeDetection
    point: tuple[float, float] | None
    trust: float


def _trust_score(detection: EyeDetection, weight: float, config: FusionConfig, crop_width: int, crop_height: int) -> float:
    """`weight * confidence`, scaled down when this model's own accept/
    reject gate rejected the prediction or the point falls implausibly
    outside the crop it came from - see the module docstring's "Model
    weight vs prediction confidence" section for why weight and confidence
    are multiplied rather than either one deciding alone."""
    if detection.center is None:
        return 0.0
    trust = weight * max(0.0, min(1.0, detection.confidence))
    if not detection.accepted:
        trust *= config.rejected_trust_penalty
    if not point_in_bounds(detection.center, crop_width, crop_height):
        trust = 0.0
    return trust


def _weighted_point(candidates: list[_Candidate]) -> tuple[float, float]:
    total = sum(c.trust for c in candidates)
    if total <= 0.0:
        # Every candidate here was already filtered to trust > 0 by the
        # caller; this is only reachable if that invariant is ever relaxed,
        # and an unweighted average is a safe, unsurprising fallback rather
        # than a division by zero.
        xs = [c.point[0] for c in candidates]
        ys = [c.point[1] for c in candidates]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    x = sum(c.point[0] * c.trust for c in candidates) / total
    y = sum(c.point[1] * c.trust for c in candidates) / total
    return x, y


def _build_result(
    *,
    status: str,
    point: tuple[float, float] | None,
    confidence: float,
    source_ids: tuple[str, ...],
    width: int,
    height: int,
    config: FusionConfig,
    accepted: bool,
    left: EyeKeypoint | None,
    right: EyeKeypoint | None,
    head_confidence: float | None,
    head_visible: bool,
    beak: EyeKeypoint | None,
    head_top: EyeKeypoint | None,
    left_shoulder: EyeKeypoint | None,
    right_shoulder: EyeKeypoint | None,
) -> EyeDetection:
    if point is None:
        box = (0.0, 0.0, 1.0, min(1.0, float(height)) or 1.0)
        center = None
    else:
        box = derive_eye_box(point[0], point[1], width, height, config.eye_box_frac, config.min_eye_box_px)
        center = point
    return EyeDetection(
        box=box,
        confidence=confidence,
        center=center,
        detector_id=FUSION_DETECTOR_ID,
        left=left,
        right=right,
        accepted=accepted,
        head_confidence=head_confidence,
        head_visible=head_visible,
        beak=beak,
        head_top=head_top,
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        fusion_status=status,
        source_detectors=source_ids,
    )


class FusionEyeDetector:
    """Combines EyePose-v0 with one or more complementary eye detectors -
    implements `eyes.detector.EyeDetector`, so `ranking.classic`,
    `eyes.cache`, and the Gallery/Loupe overlay consume it exactly like any
    single-model backend (see `eyes.detector`'s own module docstring for
    that contract). `eyes.build_eye_detector("fusion-v1")` is how a caller
    constructs one; `ranking.classic.ClassicVisionFusionStrategy` is the
    ranking strategy that runs it.

    Adding a third model later is: construct it (lazily, inside `__init__`,
    the same pattern every other backend already follows) and add one entry
    to `sub_detectors`/`FusionConfig.model_weights` - `detect`'s own fusion
    logic reasons over however many candidates it is given, not a fixed two.
    """

    detector_id = FUSION_DETECTOR_ID

    def __init__(
        self,
        sub_detectors: list | None = None,
        config: FusionConfig | None = None,
        device: str = "cpu",
        **sub_detector_kwargs,
    ) -> None:
        self.config = config or FusionConfig()
        if sub_detectors is not None:
            self._sub_detectors = list(sub_detectors)
        else:
            # Imported lazily, same reasoning eyes.detector.build_eye_detector
            # documents: constructing a FusionEyeDetector should not cost
            # more than constructing both backends already costs separately.
            from .eyepose_v0 import EyePoseV0EyeDetector
            from .superanimal_bird import SuperAnimalBirdEyeDetector

            self._sub_detectors = [
                EyePoseV0EyeDetector(device=device, **sub_detector_kwargs.get("eyepose_v0", {})),
                SuperAnimalBirdEyeDetector(device=device, **sub_detector_kwargs.get("superanimal_bird", {})),
            ]

    def supports(self, coco_label: int) -> bool:
        """Covers a subject if ANY sub-detector does - today both bundled
        backends are bird-only, so this is equivalent to bird-only, but a
        future sub-detector covering a different class would extend this
        automatically."""
        return any(sub.supports(coco_label) for sub in self._sub_detectors)

    def detect(self, subject_crop_rgb) -> EyeDetection:
        """Run every configured sub-detector on the same crop, then fuse -
        see the module docstring for the algorithm. Always returns an
        `EyeDetection`, never `None`, matching every other backend's
        contract - even `STATUS_DISAGREEMENT`/`STATUS_NO_DETECTION` carry a
        real (if `accepted=False`) record, and `left`/`right` are populated
        from whichever sub-detector contributed the winning candidate (or
        the higher-trust one, on an unresolved disagreement) so a debugging
        overlay still has real keypoints to show."""
        height, width = (subject_crop_rgb.shape[:2] if subject_crop_rgb is not None else (0, 0))
        if subject_crop_rgb is None or subject_crop_rgb.size == 0:
            return _build_result(
                status=STATUS_NO_DETECTION, point=None, confidence=0.0, source_ids=(),
                width=width, height=height, config=self.config, accepted=False,
                left=None, right=None, head_confidence=None, head_visible=False,
                beak=None, head_top=None, left_shoulder=None, right_shoulder=None,
            )

        detections = {sub.detector_id: sub.detect(subject_crop_rgb) for sub in self._sub_detectors}

        candidates: list[_Candidate] = []
        for detector_id, detection in detections.items():
            weight = self.config.weight_for(detector_id)
            trust = _trust_score(detection, weight, self.config, width, height)
            if detection.center is not None and trust > 0.0:
                candidates.append(_Candidate(detector_id=detector_id, detection=detection, point=detection.center, trust=trust))

        # EyePose-v0's own landmark set is the only source of beak/head_top/
        # shoulders and a holistic head_confidence today - passed through
        # unchanged (never fabricated for a backend that has none, matching
        # EyeDetection's own established "None means this backend does not
        # compute it" convention) so the Loupe's Elements overlay and the
        # head-relative Burst frame both keep working on a fused result
        # exactly as they do on a plain EyePose-v0 one.
        eyepose_detection = detections.get("eyepose-v0")
        head_confidence = eyepose_detection.head_confidence if eyepose_detection else None
        head_visible = eyepose_detection.head_visible if eyepose_detection else True
        beak = eyepose_detection.beak if eyepose_detection else None
        head_top = eyepose_detection.head_top if eyepose_detection else None
        left_shoulder = eyepose_detection.left_shoulder if eyepose_detection else None
        right_shoulder = eyepose_detection.right_shoulder if eyepose_detection else None

        if not candidates:
            reason = STATUS_LOW_CONFIDENCE if detections else STATUS_NO_DETECTION
            return _build_result(
                status=reason, point=None, confidence=0.0, source_ids=(),
                width=width, height=height, config=self.config, accepted=False,
                left=None, right=None, head_confidence=head_confidence, head_visible=head_visible,
                beak=beak, head_top=head_top, left_shoulder=left_shoulder, right_shoulder=right_shoulder,
            )

        if len(candidates) == 1:
            only = candidates[0]
            confidence = min(1.0, only.trust / max(self.config.weight_for(only.detector_id), 1e-6))
            accepted = only.detection.accepted and confidence >= self.config.min_fused_confidence
            return _build_result(
                status=STATUS_SINGLE_MODEL, point=only.point, confidence=confidence, source_ids=(only.detector_id,),
                width=width, height=height, config=self.config, accepted=accepted,
                left=only.detection.left, right=only.detection.right,
                head_confidence=head_confidence, head_visible=head_visible,
                beak=beak, head_top=head_top, left_shoulder=left_shoulder, right_shoulder=right_shoulder,
            )

        # Two or more trusted candidates: agreement is judged pairwise in
        # head-scale units. EyePose-v0's own beak<->head_top axis is used as
        # the shared scale reference whenever it is available (the same
        # reference its own accept/reject gate already uses - see
        # eyepose_v0.py); a crop where EyePose-v0 did not report both
        # landmarks falls back to the crop's own shorter side as a coarser,
        # always-available scale.
        head_scale = None
        if beak is not None and head_top is not None:
            frame = build_head_frame((beak.x, beak.y), (head_top.x, head_top.y))
            if frame is not None:
                head_scale = frame.scale
        if head_scale is None:
            head_scale = max(3.0, min(width, height))

        candidates.sort(key=lambda c: -c.trust)
        best, second = candidates[0], candidates[1]
        distance = normalized_distance(best.point, second.point, head_scale)

        if distance <= self.config.agreement_threshold:
            point = _weighted_point(candidates)
            total_weight = sum(self.config.weight_for(c.detector_id) for c in candidates)
            confidence = min(1.0, sum(c.trust for c in candidates) / max(total_weight, 1e-6))
            accepted = confidence >= self.config.min_fused_confidence
            primary = best
            return _build_result(
                status=STATUS_AGREE, point=point, confidence=confidence,
                source_ids=tuple(c.detector_id for c in candidates),
                width=width, height=height, config=self.config, accepted=accepted,
                left=primary.detection.left, right=primary.detection.right,
                head_confidence=head_confidence, head_visible=head_visible,
                beak=beak, head_top=head_top, left_shoulder=left_shoulder, right_shoulder=right_shoulder,
            )

        # Disagreement: never average (see the module docstring). Only a
        # clear, substantial dominance picks a winner.
        if best.trust >= second.trust * self.config.disagreement_dominance_ratio:
            confidence = min(1.0, best.trust / max(self.config.weight_for(best.detector_id), 1e-6))
            accepted = best.detection.accepted and confidence >= self.config.min_fused_confidence
            return _build_result(
                status=STATUS_DISAGREEMENT_RESOLVED, point=best.point, confidence=confidence,
                source_ids=(best.detector_id,),
                width=width, height=height, config=self.config, accepted=accepted,
                left=best.detection.left, right=best.detection.right,
                head_confidence=head_confidence, head_visible=head_visible,
                beak=beak, head_top=head_top, left_shoulder=left_shoulder, right_shoulder=right_shoulder,
            )

        # No reliable winner - report disagreement rather than inventing a
        # location. The raw candidates are still surfaced via left/right
        # (the higher-trust model's own channels) for a debugging overlay.
        return _build_result(
            status=STATUS_DISAGREEMENT, point=None, confidence=0.0, source_ids=(),
            width=width, height=height, config=self.config, accepted=False,
            left=best.detection.left, right=best.detection.right,
            head_confidence=head_confidence, head_visible=head_visible,
            beak=beak, head_top=head_top, left_shoulder=left_shoulder, right_shoulder=right_shoulder,
        )
