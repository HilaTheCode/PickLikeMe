"""Ranking Mode: Birds + Mammals - the safari mode.

A photographer does not switch their photographic subject (bird vs mammal)
mid-Burst - a rapid sequence is one bird, or one mammal, never both. So
domain detection here runs ONCE PER BURST, on one representative frame, not
once per image (see `eyes.domain_detector`'s own module docstring for the
classifier itself).

    Burst
      |
  Domain Detector (eyes.domain_detector.ClipDomainDetector)
      |
  +---+----+--------+
  |        |        |
 BIRD   MAMMAL   UNCERTAIN
  |        |        |
 Birds  Mammals  every available
 models  models   model together
  |        |        |
  +--------+--------+
           |
   Shared Fusion Layer (eyes.fusion.FusionEyeDetector - unchanged)
           |
   Shared scoring (ranking.classic.measure/combine - unchanged)

Reuses `ranking.classic.ClassicVisionStrategy` entirely for filtering,
scoring, and CSV/report writing - see that class's own docstring. The ONE
thing this strategy overrides is `_build_eye_filter_router`, the hook that
method exists specifically for: everywhere else, a per-Burst-routed
detector looks exactly like the single shared detector every other Classic
Vision backend already uses.

Burst grouping here uses each file's own modification time as a stand-in
for capture time - not the real EXIF `DateTimeOriginal` the review layer's
`AnnotationStore.capture_timestamp_of` reads (see that call site in
`review.session`), because plumbing that dependency into this
workflow-blind analysis module (see `ranking.classic`'s own module
docstring: "never refuses to run because of workflow state") would be a
much larger change than this first working implementation calls for. File
mtime is a reasonable proxy for "were these frames handled together" and
is enough for the actual job this module needs it for - deciding which
frames share a photographic subject - but is a known limitation if a
folder's file timestamps do not reflect capture order (e.g. after certain
copy/export operations) - see the accompanying report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..bird_crop import CropParams, crop_cache_path
from ..burst import BurstEntry, reconstruct_bursts
from ..eyes import EyeDetector, build_eye_detector
from ..eyes.domain_detector import (
    DEFAULT_MIN_CONFIDENCE as DOMAIN_DEFAULT_MIN_CONFIDENCE,
)
from ..eyes.domain_detector import (
    DOMAIN_BIRD,
    DOMAIN_MAMMAL,
    ClipDomainDetector,
)
from ..eyes.domains import BIRDS_PROFILE, MAMMALS_PROFILE, build_domain_fusion_detector
from ..eyes.fusion import DEFAULT_AGREEMENT_THRESHOLD, DEFAULT_MIN_FUSED_CONFIDENCE, FusionConfig, ModelWeight
from .base import GROUP_THRESHOLDS, GROUP_WEIGHTS, ParamSpec, StrategyInfo, WeightedParams, use_subject_filter_spec
from .classic import (
    METRIC_LABELS,
    ClassicVisionStrategy,
    _detection_specs,
    _fusion_threshold_specs,
    _fusion_weight_specs,
    _scoring_weight_specs,
)

COMBINED_STRATEGY_ID = "classic-vision-fusion-combined"

# How far apart (in file-mtime seconds) two images may sit and still be
# treated as one Burst for domain-routing purposes - see the module
# docstring. Matches burst_analysis.DEFAULT_MAX_GAP_SECONDS, the same
# default the review layer's own (EXIF-based) Burst grouping already uses,
# so "what counts as one Burst" means the same thing here even though the
# timestamp source is different.
DEFAULT_MAX_GAP_SECONDS = 2.0


@dataclass(frozen=True)
class ClassicVisionCombinedParams(WeightedParams):
    """Ranking Mode: Birds + Mammals. Carries every model weight either
    single-domain Fusion strategy has (see `ClassicVisionBirdFusionParams`/
    `ClassicVisionMammalFusionParams`) plus the domain detector's own
    confidence threshold - one params dataclass, because a photographer
    tuning this mode should not need to separately open two other modes'
    dialogs to adjust the same underlying weights.
    """

    eye_sharpness_weight: float = 70.0
    subject_sharpness_weight: float = 10.0
    subject_size_weight: float = 20.0
    eyepose_v0_model_weight: float = BIRDS_PROFILE.default_model_weights[0].weight
    superanimal_bird_model_weight: float = BIRDS_PROFILE.default_model_weights[1].weight
    superanimal_quadruped_model_weight: float = MAMMALS_PROFILE.default_model_weights[0].weight
    domain_min_confidence: float = DOMAIN_DEFAULT_MIN_CONFIDENCE
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD
    min_fused_confidence: float = DEFAULT_MIN_FUSED_CONFIDENCE
    detection_confidence_threshold: float = CropParams.conf_threshold
    crop_confidence_threshold: float = CropParams.min_crop_confidence

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        return (
            *_scoring_weight_specs(),
            *_fusion_weight_specs(BIRDS_PROFILE, "Birds Bursts:"),
            *_fusion_weight_specs(MAMMALS_PROFILE, "Mammal Bursts:"),
            ParamSpec(
                name="domain_min_confidence",
                label="Domain detection confidence",
                default=DOMAIN_DEFAULT_MIN_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
                group=GROUP_THRESHOLDS,
                decimals=2,
                help=(
                    "Below this, a Burst's Bird-vs-Mammal classification is treated as "
                    "uncertain and every available model is tried together instead of "
                    "guessing the domain."
                ),
            ),
            *_fusion_threshold_specs(),
            *_detection_specs(),
            use_subject_filter_spec(),
        )


class ClassicVisionCombinedStrategy(ClassicVisionStrategy):
    """Ranking Mode: Birds + Mammals - see the module docstring. Filtering,
    scoring, and CSV/report writing are all inherited from
    `ClassicVisionStrategy` unchanged; only `_build_eye_filter_router` is
    overridden, to classify each Burst once and route a domain-appropriate
    Fusion detector to every image in it.
    """

    info = StrategyInfo(
        strategy_id=COMBINED_STRATEGY_ID,
        display_name="Classic Vision Ranking (Birds + Mammals)",
        description=(
            "Ranking Mode: Birds + Mammals (the safari mode). Classifies each Burst "
            "once as Bird/Mammal/Uncertain (eyes.domain_detector), then routes it "
            "through the matching domain's models via the shared Fusion/Validation "
            "layer - never a per-image guess, and never two independent scoring "
            "engines."
        ),
        score_label="Classic (Birds+Mammals)",
    )
    params_class = ClassicVisionCombinedParams
    param_specs = ClassicVisionCombinedParams.specs()
    metric_labels = METRIC_LABELS
    # Unused directly (see _build_eye_filter_router, which this strategy
    # overrides instead of relying on the base class's single-detector
    # construction) - kept as a descriptive default rather than None so
    # introspection/logging that reads _eye_detector_name still sees
    # something meaningful.
    _eye_detector_name = "fusion-birds"
    # This strategy already answers "is the crop's subject a bird or a
    # mammal" itself, per Burst, from the crop's own pixels
    # (_build_eye_filter_router -> _classify_burst -> ClipDomainDetector) -
    # before a detector is ever chosen for a given image. Re-checking the
    # crop's COCO class label at EyeFilter time (the base class's default)
    # would ask the exact question this architecture exists to stop trusting
    # COCO for: a real, measured case on this project's own data is a
    # Colobus monkey crop COCO itself recorded as class 16/"bird" at 0.99
    # confidence (see eyes.domains.MAMMALS_PROFILE's own docstring) - gating
    # on that label here could reject a crop this strategy already
    # correctly routed to a domain-appropriate detector. See EyeFilter's own
    # docstring for the general mechanism this flag controls.
    _gate_by_subject_label = False

    def _eye_detector_kwargs(self, params: ClassicVisionCombinedParams) -> dict:
        return {}

    def _eye_detector_metadata(self, params: ClassicVisionCombinedParams) -> dict:
        return {
            "eyepose_v0_model_weight": params.eyepose_v0_model_weight,
            "superanimal_bird_model_weight": params.superanimal_bird_model_weight,
            "superanimal_quadruped_model_weight": params.superanimal_quadruped_model_weight,
            "domain_min_confidence": params.domain_min_confidence,
            "agreement_threshold": params.agreement_threshold,
            "min_fused_confidence": params.min_fused_confidence,
        }

    def _build_eye_filter_router(
        self,
        params: ClassicVisionCombinedParams,
        resolved_device: str,
        image_paths: list[str],
        crop_cache_dir,
    ) -> Callable[[str], EyeDetector]:
        bursts = _group_by_file_mtime(image_paths)
        domain_detector = ClipDomainDetector(device=resolved_device, min_confidence=params.domain_min_confidence)

        path_domain: dict[str, str] = {}
        for burst in bursts:
            verdict = _classify_burst(burst, crop_cache_dir, domain_detector)
            for path in burst:
                path_domain[path] = verdict

        detector_cache: dict[str, EyeDetector] = {}

        def detector_for(image_path: str) -> EyeDetector:
            domain = path_domain.get(image_path, "uncertain")
            if domain not in detector_cache:
                detector_cache[domain] = _build_detector_for_domain(domain, params, resolved_device)
            return detector_cache[domain]

        return detector_for


def _group_by_file_mtime(image_paths: list[str], *, max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS) -> list[list[str]]:
    """Group `image_paths` into pseudo-Bursts by file modification time -
    see the module docstring for why mtime rather than real EXIF capture
    time. Reuses `burst.reconstruct_bursts` (the same clustering
    `burst_analysis` and `ingest.burst` already use) rather than a third
    implementation of "group by time gap"."""
    entries = []
    for path in image_paths:
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            mtime = 0.0
        timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        entries.append(BurstEntry(path=path, timestamp=timestamp))
    groups = reconstruct_bursts(entries, max_gap_seconds=max_gap_seconds)
    return [[entry.path for entry in group] for group in groups]


def _classify_burst(burst: list[str], crop_cache_dir, domain_detector: ClipDomainDetector) -> str:
    """One domain verdict for a whole Burst - the first member with a
    readable cached crop decides it for every member (see the module
    docstring: classified once per Burst, never per image)."""
    import cv2

    for path in burst:
        crop_path = crop_cache_path(crop_cache_dir, path)
        if not crop_path.is_file():
            continue
        image_bgr = cv2.imread(str(crop_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return domain_detector.predict(image_rgb).domain
    return "uncertain"


def _build_detector_for_domain(domain: str, params: ClassicVisionCombinedParams, device: str) -> EyeDetector:
    """The Fusion detector for one Burst's classified domain - BIRD/MAMMAL
    route through `eyes.domains.build_domain_fusion_detector` exactly like
    the single-domain Fusion strategies do; UNCERTAIN runs every available
    model together through the same, unchanged `FusionEyeDetector` rather
    than guessing a domain - agreement/geometric validity are left to sort
    it out, the same principle the whole Fusion layer is built on."""
    if domain == DOMAIN_BIRD:
        return build_domain_fusion_detector(
            "birds",
            device=device,
            config=FusionConfig(
                model_weights=(
                    ModelWeight("eyepose-v0", params.eyepose_v0_model_weight),
                    ModelWeight("superanimal-bird", params.superanimal_bird_model_weight),
                ),
                agreement_threshold=params.agreement_threshold,
                min_fused_confidence=params.min_fused_confidence,
            ),
        )
    if domain == DOMAIN_MAMMAL:
        return build_domain_fusion_detector(
            "mammals",
            device=device,
            config=FusionConfig(
                model_weights=(ModelWeight("superanimal-quadruped", params.superanimal_quadruped_model_weight),),
                agreement_threshold=params.agreement_threshold,
                min_fused_confidence=params.min_fused_confidence,
            ),
        )

    from ..eyes.fusion import FusionEyeDetector

    sub_detectors = [
        build_eye_detector("eyepose-v0", device=device),
        build_eye_detector("superanimal-bird", device=device),
        build_eye_detector("superanimal-quadruped", device=device),
    ]
    weights = (
        ModelWeight("eyepose-v0", params.eyepose_v0_model_weight),
        ModelWeight("superanimal-bird", params.superanimal_bird_model_weight),
        ModelWeight("superanimal-quadruped", params.superanimal_quadruped_model_weight),
    )
    return FusionEyeDetector(
        sub_detectors=sub_detectors,
        config=FusionConfig(
            model_weights=weights,
            agreement_threshold=params.agreement_threshold,
            min_fused_confidence=params.min_fused_confidence,
        ),
    )
