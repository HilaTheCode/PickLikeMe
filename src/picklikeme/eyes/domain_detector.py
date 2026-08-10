"""Burst-level BIRD/MAMMAL/UNCERTAIN domain detection.

`eyes.domains` says WHICH models a Ranking Mode uses; this module is the one
piece of new machinery the "Birds + Mammals" combined mode needs on top of
that - deciding which mode applies to a given Burst in the first place.

Classified ONCE PER BURST, never per image: a photographer does not swap
their photographic subject mid-burst (see `ranking.combined`'s own module
docstring for where this gets used), so one representative frame's subject
crop is enough - this module has no opinion about *which* frame a caller
picks, only about classifying whichever crop it is given.

Deliberately coarse: BIRD vs MAMMAL vs UNCERTAIN, never a species. Reuses
this project's existing `open_clip_torch` dependency and the same zero-shot
image/text-embedding pattern `species.bioclip_classifier` already
established (`create_model_and_transforms` + `get_tokenizer` + cosine
similarity against text prompts) - but against a small, generic CLIP
checkpoint (`ViT-B-32`/`openai`, ~150 MB) rather than BioCLIP's own much
larger taxonomy-specific checkpoint (571 MB-1.6 GB): this task only needs a
3-way split, not species identification, so the smaller, faster, more
commonly-already-cached general-purpose model is the appropriate tool here,
not the specialised one BioCLIP's own module docstring picked for a
different job (naming a species).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

DOMAIN_BIRD = "bird"
DOMAIN_MAMMAL = "mammal"
DOMAIN_UNCERTAIN = "uncertain"

DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"

# Below this top-domain similarity, the verdict is UNCERTAIN rather than
# forced - see the module docstring's "do not force an unreliable domain
# assignment" requirement. A reasoned starting point (CLIP zero-shot
# softmax similarity over a small, unambiguous 2-way prompt set typically
# separates a clear photograph well above this), not empirically fitted
# against a labelled sample - configurable so it can be tuned once real
# mixed-domain Bursts have been reviewed. Nudged down from an initial 0.6
# after a small real-crop spot check (6 real subject crops from this
# project's own cache - see the accompanying report): genuine mammal crops
# clustered at 0.55-0.60, which a 0.6 cutoff was pushing into UNCERTAIN too
# often for a first working implementation.
DEFAULT_MIN_CONFIDENCE = 0.5

_BIRD_PROMPTS: tuple[str, ...] = ("a photo of a bird", "a close-up wildlife photo of a bird")
_MAMMAL_PROMPTS: tuple[str, ...] = ("a photo of a mammal", "a close-up wildlife photo of a mammal")


@dataclass(frozen=True)
class DomainPrediction:
    """One verdict for one (representative) crop - see the module
    docstring. `domain` is one of `DOMAIN_BIRD`/`DOMAIN_MAMMAL`/
    `DOMAIN_UNCERTAIN`; `confidence` is the zero-shot similarity behind it,
    always populated (including for `DOMAIN_UNCERTAIN`, so a caller can see
    *how* uncertain rather than just that it was)."""

    domain: str
    confidence: float


class ClipDomainDetector:
    """Zero-shot BIRD/MAMMAL/UNCERTAIN classification from a subject crop -
    see the module docstring for the model and why it is a different,
    smaller one than `species.bioclip_classifier` uses for actual species
    identification."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str = "cpu",
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.device = device
        self.min_confidence = min_confidence

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        prompts = list(_BIRD_PROMPTS) + list(_MAMMAL_PROMPTS)
        self._num_bird_prompts = len(_BIRD_PROMPTS)
        with torch.no_grad():
            tokens = tokenizer(prompts)
            text_features = self.model.encode_text(tokens.to(device))
            self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def predict(self, image_rgb: "np.ndarray") -> DomainPrediction:
        """One domain verdict for one crop - see the module docstring for
        why a caller should call this once per Burst, on one representative
        frame, rather than once per image."""
        from PIL import Image

        torch = self._torch
        if image_rgb is None or image_rgb.size == 0:
            return DomainPrediction(domain=DOMAIN_UNCERTAIN, confidence=0.0)

        pil_image = Image.fromarray(image_rgb)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)[0]

        scores = similarity.cpu().numpy()
        bird_score = float(scores[: self._num_bird_prompts].max())
        mammal_score = float(scores[self._num_bird_prompts:].max())

        if bird_score >= mammal_score:
            domain, confidence = DOMAIN_BIRD, bird_score
        else:
            domain, confidence = DOMAIN_MAMMAL, mammal_score

        if confidence < self.min_confidence:
            return DomainPrediction(domain=DOMAIN_UNCERTAIN, confidence=confidence)
        return DomainPrediction(domain=domain, confidence=confidence)
