"""The pluggable species-classification boundary.

`arrange_by_species` (see arrange.py) needs exactly one thing from a
classifier: given a decoded image, name the species and say how confident
it is. Everything else - which model that is, what candidate species it
even knows about, whether it runs on CPU or GPU - is that class's own
business, never arrange.py's. Swapping in a better classifier later (a
newer foundation model, one fine-tuned on this photographer's own species
mix, an ensemble) means adding one class here and registering it in
CLASSIFIERS - arrange.py, review, and the ranking pipeline never change.

UNKNOWN_SPECIES is the one answer every classifier's caller can rely on
regardless of which classifier produced it: an unsupported species, a
low-confidence guess, and "nothing to say at all" all collapse to the same
value, and Arrange by Species treats all three identically - the image goes
to the Unknown/ folder rather than into a folder named after a guess nobody
should trust.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

UNKNOWN_SPECIES = "Unknown"


@dataclass(frozen=True)
class SpeciesPrediction:
    """One classifier's answer for one image.

    `species` is UNKNOWN_SPECIES whenever the classifier has nothing
    trustworthy to report - callers (arrange.py, the cache) only ever need
    to compare `species` against UNKNOWN_SPECIES, never separately inspect
    `confidence` to decide what bucket an image belongs in.
    """

    species: str
    confidence: float | None
    classifier_id: str
    # Optional: the full ranked (species, confidence) list a classifier
    # computed on its way to this single answer - Top-5 analytics (see
    # analytics.species_capture) read this when present rather than
    # re-running inference a second time just to see more than the winner.
    # `None` for a cache-served answer (SpeciesCache only persists the
    # single winning species, not the full ranking - see its own module
    # docstring) or for a backend that never computes a full ranking at
    # all (e.g. a closed-set classifier that only ever reports one label).
    # Always populated for a FRESH classification from a BioCLIP-family
    # backend, at zero extra inference cost - the full similarity vector
    # already exists in memory before the single argmax answer is taken.
    top_predictions: tuple[tuple[str, float], ...] | None = None


class SpeciesClassifier(Protocol):
    """Anything arrange_by_species() (and SpeciesCache) can consume.

    `classifier_id` is a short, stable string identifying this exact model
    *and* configuration (e.g. "bioclip-2:aeb5a3073ad9" - the model name plus
    a digest of the exact species list, see BioClipSpeciesClassifier's own
    classifier_id property). Stored alongside every prediction (see
    cache.py) so a result computed by one classifier is never silently
    served to a caller now using a different one - the same principle
    analyzer.detections.DetectionCache's own cache_version already applies
    to detector output.

    KNOWN GAP (see docs/BioCLIP_Backend_Architecture_Review.md Section 5):
    SpeciesCache's storage is keyed by image identity alone, not by
    (image identity, classifier_id) together - reads are safe (a
    classifier_id mismatch is correctly treated as a cache miss), but two
    different classifiers' cached answers for the same image cannot
    currently coexist; the later one overwrites the earlier one. Not yet
    fixed - flagged here so it is not mistaken for already having the
    per-detector isolation this docstring's own principle implies.
    """

    @property
    def classifier_id(self) -> str: ...

    def classify(self, image: "Image.Image") -> SpeciesPrediction: ...


@dataclass(frozen=True)
class ClassifierInfo:
    """What a UI needs to list a species-classification backend in a menu -
    same shape as `ranking.base.StrategyInfo`, deliberately: a photographer
    choosing a species backend is the same kind of decision as choosing a
    ranking strategy, so it gets the same kind of pure-data, no-model-import
    listing."""

    classifier_id: str
    display_name: str
    description: str


# Every registered backend, in menu order - pure data, no model constructed
# or imported just by listing what is available (see build_classifier's own
# docstring for why that import stays lazy).
AVAILABLE_CLASSIFIERS: tuple[ClassifierInfo, ...] = (
    ClassifierInfo(
        classifier_id="bioclip2",
        display_name="BioCLIP 2 (recommended)",
        description="ViT-L/14, trained on TreeOfLife-200M (~952K taxa). The larger, newer BioCLIP release.",
    ),
    ClassifierInfo(
        classifier_id="bioclip",
        display_name="BioCLIP (original)",
        description="ViT-B/16, trained on TreeOfLife-10M (~450K taxa). Smaller and faster; kept for comparison against BioCLIP 2.",
    ),
)


def available_classifiers() -> tuple[ClassifierInfo, ...]:
    """Every registered species-classification backend, in menu order."""
    return AVAILABLE_CLASSIFIERS


def build_classifier(name: str, **kwargs) -> SpeciesClassifier:
    """Construct a registered classifier by name.

    A factory rather than importing concrete classes directly: the only
    concrete classifier implementation today (BioClipSpeciesClassifier)
    lazily imports torch and open_clip inside its own __init__, exactly
    like BirdDetector does for torch/torchvision - so nothing that merely
    imports this module (arrange.py, the cache, the CLI's --help) pays for
    a heavy ML import it may never use.

    "bioclip2" (the default, existing) and "bioclip" (the original BioCLIP,
    added for comparison - see `species.bioclip_classifier`'s module
    docstring) are both backed by the exact same `BioClipSpeciesClassifier`
    class: it was already parameterized by `model_id`, so a second backend
    is one more registration, `functools.partial`-bound to its own default
    `model_id`, not a second class. Callers never pass `model_id`
    themselves - which concrete model a name resolves to is this registry's
    decision alone, exactly like `eyes.build_eye_detector`'s own registry.
    """
    from .bioclip_classifier import BIOCLIP_V1_MODEL_ID, BioClipSpeciesClassifier

    classifiers: dict[str, Callable[..., SpeciesClassifier]] = {
        "bioclip2": BioClipSpeciesClassifier,
        "bioclip": functools.partial(BioClipSpeciesClassifier, model_id=BIOCLIP_V1_MODEL_ID),
    }
    try:
        factory = classifiers[name]
    except KeyError:
        raise ValueError(f"Unknown classifier {name!r}. Available: {', '.join(sorted(classifiers))}") from None
    return factory(**kwargs)
