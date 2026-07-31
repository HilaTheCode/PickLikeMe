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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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


class SpeciesClassifier(Protocol):
    """Anything arrange_by_species() (and SpeciesCache) can consume.

    `classifier_id` is a short, stable string identifying this exact model
    *and* configuration (e.g. "bioclip2:species-42"). Stored alongside every
    prediction (see cache.py) so a result computed by one classifier is
    never silently served to a caller now using a different one - the same
    principle analyzer.detections.DetectionCache's own cache_version already
    applies to detector output.
    """

    @property
    def classifier_id(self) -> str: ...

    def classify(self, image: "Image.Image") -> SpeciesPrediction: ...


def build_classifier(name: str, **kwargs) -> SpeciesClassifier:
    """Construct a registered classifier by name.

    A factory rather than importing concrete classes directly: the only
    concrete classifier today (BioClipSpeciesClassifier) lazily imports
    torch and open_clip inside its own __init__, exactly like BirdDetector
    does for torch/torchvision - so nothing that merely imports this module
    (arrange.py, the cache, the CLI's --help) pays for a heavy ML import it
    may never use.
    """
    from .bioclip_classifier import BioClipSpeciesClassifier

    classifiers: dict[str, type] = {
        "bioclip2": BioClipSpeciesClassifier,
    }
    try:
        cls = classifiers[name]
    except KeyError:
        raise ValueError(f"Unknown classifier {name!r}. Available: {', '.join(sorted(classifiers))}") from None
    return cls(**kwargs)
