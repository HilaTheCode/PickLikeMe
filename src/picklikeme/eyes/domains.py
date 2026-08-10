"""Ranking-mode domain profiles - which eye-detection models participate in
the shared Fusion Layer for a given subject domain.

PeakPic's eye-detection models are taxon-specific by construction (see
`eyes.detector`'s own module docstring): EyePose-v0 and SuperAnimal-Bird were
both trained on bird-only data, and SuperAnimal-Quadruped on mammal-only
data. Neither generalises to the other domain in a way this project trusts
for scoring - see `superanimal_quadruped.py`'s own module docstring for why
a monkey-shaped point EyePose-v0 sometimes produces is not evidence it
understands mammal anatomy.

This module is the one place that says, for a given Ranking Mode, WHICH
detectors are appropriate and at what default weight - `fusion.
FusionEyeDetector` itself stays completely domain-agnostic (it fuses
whatever `EyeDetector` instances it is given; see that module's own
docstring), so a Ranking Mode is nothing more than "construct the Fusion
Layer with THIS set of detectors and THESE default weights" - one
`DomainProfile`, not a second fusion implementation.

    Ranking Mode
         |
    +----+----+
    |         |
  BIRDS    MAMMALS
    |         |
 (EyePose-v0,  (SuperAnimal-Quadruped,
  SuperAnimal-   ...)
  Bird)
    |         |
    +----+----+
         v
  fusion.FusionEyeDetector   (unchanged, domain-agnostic)

Adding a third Ranking Mode later - or a second model to either existing
one - is one more `DomainProfile` (or one more entry in an existing one's
`detector_ids`/`default_model_weights`), never a change to `fusion.py`
itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fusion import FusionConfig, FusionEyeDetector, ModelWeight

RANKING_MODE_BIRDS = "birds"
RANKING_MODE_MAMMALS = "mammals"
RANKING_MODES: tuple[str, ...] = (RANKING_MODE_BIRDS, RANKING_MODE_MAMMALS)


@dataclass(frozen=True)
class DomainProfile:
    """Which eye detectors - and at what default weight - participate in
    the shared Fusion Layer for one Ranking Mode, and how subject
    eligibility is decided for it. See the module docstring."""

    ranking_mode: str
    display_name: str
    # eyes.build_eye_detector names, in the order the Fusion Layer receives
    # them. A single-entry tuple is a completely valid profile - the Fusion
    # Layer already handles "exactly one trusted candidate" as
    # STATUS_SINGLE_MODEL (see fusion.py) - so a domain starts useful with
    # one real model and grows without any fusion-layer change.
    detector_ids: tuple[str, ...]
    default_model_weights: tuple[ModelWeight, ...]
    # True (Birds): subject eligibility is gated by the upstream subject
    # detector's own COCO-class label (bird_crop.COCO_BIRD_CLASS genuinely
    # exists and is reliable for this domain).
    # False (Mammals): eligibility is NOT gated by that label. COCO's
    # 80-class vocabulary has no class for most real safari species (no
    # monkey, lion, leopard, cheetah, antelope, buffalo...) - concretely,
    # confidently measured on this project's own real archive: a Colobus
    # monkey crop recorded as COCO class 16 ("bird") at 0.99 detector
    # confidence (see the accompanying report). Gating Mammal-mode
    # eligibility on that label would make it unusable on exactly the real
    # photographs it exists to help with. Selecting Mammals as the Ranking
    # Mode IS the domain declaration here, in place of a COCO label this
    # project's own upstream detector cannot supply for these species - see
    # `superanimal_quadruped.SuperAnimalQuadrupedEyeDetector.supports`'s own
    # docstring, which is what actually implements this for the mammal
    # detector.
    gate_by_subject_label: bool


# Weights unchanged from the bird-only fusion work (see fusion.py's module
# docstring for the 1,324-crop benchmark they were derived from) - Birds
# mode is the exact same two-model pairing already benchmarked, just now
# reached through an explicit domain profile instead of being fusion.py's
# only hard-coded option.
BIRDS_PROFILE = DomainProfile(
    ranking_mode=RANKING_MODE_BIRDS,
    display_name="Birds",
    detector_ids=("eyepose-v0", "superanimal-bird"),
    default_model_weights=(ModelWeight("eyepose-v0", 0.65), ModelWeight("superanimal-bird", 0.35)),
    gate_by_subject_label=True,
)

# A single model today - see DomainProfile.detector_ids's own docstring for
# why that is a complete, valid profile rather than a placeholder. No
# second mammal-domain model was found that met this project's own
# practicality bar (see the accompanying report's model-investigation
# section - AP-10K/Animal Kingdom both require MMPose, whose mmcv
# dependency has no prebuilt PyPI wheel at all); weight 1.0 is not a
# meaningful "trust ratio" the way Birds' 0.65/0.35 split is, only the
# placeholder a single-entry FusionConfig needs, kept in the same
# ModelWeight shape so adding a real second mammal model later is the
# same one-line change BIRDS_PROFILE's own two entries demonstrate.
MAMMALS_PROFILE = DomainProfile(
    ranking_mode=RANKING_MODE_MAMMALS,
    display_name="Mammals",
    detector_ids=("superanimal-quadruped",),
    default_model_weights=(ModelWeight("superanimal-quadruped", 1.0),),
    gate_by_subject_label=False,
)

_PROFILES: dict[str, DomainProfile] = {p.ranking_mode: p for p in (BIRDS_PROFILE, MAMMALS_PROFILE)}


def domain_profile(ranking_mode: str) -> DomainProfile:
    """The `DomainProfile` for a Ranking Mode id, or `ValueError` for an
    unknown one - the same "fail loudly, list what IS valid" shape
    `eyes.build_eye_detector` already uses for an unknown detector name."""
    try:
        return _PROFILES[ranking_mode]
    except KeyError:
        raise ValueError(f"Unknown ranking mode {ranking_mode!r}. Available: {', '.join(_PROFILES)}") from None


def available_ranking_modes() -> list[DomainProfile]:
    """Every registered Ranking Mode, in menu order - cheap, reads class
    attributes only, exactly like `ranking.available_strategies()`."""
    return [BIRDS_PROFILE, MAMMALS_PROFILE]


def build_domain_fusion_detector(
    ranking_mode: str,
    *,
    device: str = "cpu",
    config: FusionConfig | None = None,
) -> FusionEyeDetector:
    """The Fusion Layer, constructed for one Ranking Mode - the one place
    `eyes.build_eye_detector`'s domain-specific factory entries
    (`"fusion-birds"`/`"fusion-mammals"`) delegate to. Builds each of the
    profile's own `detector_ids` through the ordinary `build_eye_detector`
    factory (so a domain's sub-detectors are constructed exactly the way a
    caller would construct them standalone - no special-casing) and hands
    them to `FusionEyeDetector` unchanged; `config.model_weights` defaults
    to the profile's own weights when the caller does not override them,
    but a caller adjusting weights from the desktop app (see
    `ranking.classic`) passes its own `FusionConfig` through untouched.
    """
    from .detector import build_eye_detector

    profile = domain_profile(ranking_mode)
    sub_detectors = [build_eye_detector(detector_id, device=device) for detector_id in profile.detector_ids]
    if config is None:
        config = FusionConfig(model_weights=profile.default_model_weights)
    return FusionEyeDetector(sub_detectors=sub_detectors, config=config)
