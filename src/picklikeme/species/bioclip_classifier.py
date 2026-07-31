"""BioCLIP-2 - the default local, offline species classifier.

Why BioCLIP rather than the MegaDetector + SpeciesNet pairing suggested
earlier: SpeciesNet is trained overwhelmingly on Wildlife Insights' camera
trap imagery - fixed position, motion-triggered, often monochrome/IR,
animals rarely centred or well-composed. PickLikeMe's input here is the
opposite: a photographer's own handheld, composed, daylight wildlife
photography, already filtered down to their own Keep folder. BioCLIP's
training set (TreeOfLife-10M/200M) leans heavily on iNaturalist - citizen
science photography of exactly this character - so it is the closer domain
match for this specific input, not a strictly "better" model in the
abstract. See the final report for the fuller comparison.

A second, more structural reason: BioCLIP is PyTorch/open_clip based, so it
shares a framework with the existing detector rather than adding a second
one (SpeciesNet has historically been TensorFlow-centric). And it is
zero-shot against an open, text-prompted taxonomy rather than a fixed
trained classification head - adding a species later is adding one string
to a list, never retraining anything, which is exactly the "more species
later" extensibility this feature was asked to support.

Offline once installed: `open_clip`'s `create_model_and_transforms` downloads
the pretrained checkpoint once (from Hugging Face Hub) and caches it locally
- the same one-time-download-then-fully-local shape the existing detector's
torchvision COCO weights already use. No per-image network call, ever.

Not run end-to-end against real downloaded weights as part of this change -
see the module docstring in tests/test_species_bioclip_classifier.py for
why, and what is verified instead.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .classifier import UNKNOWN_SPECIES, SpeciesPrediction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

logger = logging.getLogger(__name__)

# Verify against the current model card before relying on this in
# production - open_clip/Hugging Face Hub tags for a specific release can
# change between BioCLIP versions.
DEFAULT_MODEL_ID = "hf-hub:imageomics/bioclip-2"

# CLIP-style zero-shot classification is sensitive to the prompt template;
# this is the standard generic CLIP convention, not BioCLIP's own
# fine-tuned template - a reasonable starting point, tunable later without
# touching anything outside this one constant.
DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"

# A starting vocabulary covering common subjects across the taxonomy this
# project already recognises structurally (bird_crop.DETECTION_CATEGORIES) -
# deliberately small and editable, not exhaustive. Extending species
# coverage is editing this list (or passing --species-list to point at a
# photographer's own), never retraining or changing code.
DEFAULT_SPECIES_LIST: tuple[str, ...] = (
    "Kingfisher", "White-winged Black Tern", "Common Tern", "Osprey", "Bald Eagle",
    "Golden Eagle", "Peregrine Falcon", "Red-tailed Hawk", "Great Blue Heron",
    "Great Egret", "Snowy Egret", "Sandhill Crane", "Mallard", "Canada Goose",
    "Snow Goose", "American Robin", "Northern Cardinal", "Blue Jay",
    "Barn Owl", "Great Horned Owl", "Snowy Owl", "Puffin", "Gannet",
    "Colobus Monkey", "Chimpanzee", "Gorilla", "Baboon",
    "Lion", "Tiger", "Leopard", "Cheetah", "Jaguar",
    "Elephant", "Rhinoceros", "Hippopotamus", "Giraffe", "Zebra",
    "African Buffalo", "Wildebeest", "Impala", "Gazelle",
    "Red Fox", "Gray Wolf", "Coyote", "Black Bear", "Grizzly Bear", "Polar Bear",
    "White-tailed Deer", "Elk", "Moose", "Bison",
    "Raccoon", "Otter", "Beaver", "Squirrel",
)


def _read_species_list(path: str | Path) -> tuple[str, ...]:
    """One species name per line, blank lines and '#' comments ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return tuple(
        stripped for line in lines if (stripped := line.strip()) and not stripped.startswith("#")
    )


class BioClipSpeciesClassifier:
    """Zero-shot species classification via BioCLIP-2.

    Text embeddings for every candidate species are computed once, at
    construction, and reused for every `classify()` call - the one-time
    cost of an open-vocabulary model, paid once per process rather than
    once per image, the same tradeoff BirdDetector already makes for its
    own model weights.
    """

    def __init__(
        self,
        species_list: tuple[str, ...] | None = None,
        *,
        species_list_path: str | Path | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        min_confidence: float = 0.5,
        device: str = "cpu",
    ):
        if species_list_path is not None:
            species_list = _read_species_list(species_list_path)
        self.species_list = species_list or DEFAULT_SPECIES_LIST
        if not self.species_list:
            raise ValueError("species_list must not be empty")
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.min_confidence = min_confidence
        self.device = device

        # Lazy, exactly like BirdDetector's torch/torchvision: importing
        # this module (or even constructing a classifier for a --help run)
        # must never require these to be installed unless species
        # classification is actually used.
        import open_clip
        import torch

        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)
        self._model = model.to(device).eval()
        self._preprocess = preprocess

        prompts = [prompt_template.format(species) for species in self.species_list]
        with torch.no_grad():
            tokens = tokenizer(prompts).to(device)
            text_features = self._model.encode_text(tokens)
            self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @property
    def classifier_id(self) -> str:
        """The model plus exactly which species it was asked to consider -
        both determine the answer, so both must invalidate a cached one.
        See SpeciesClassifier's own docstring for why this matters."""
        digest = hashlib.sha1("\n".join(self.species_list).encode("utf-8")).hexdigest()[:12]
        return f"bioclip2:{Path(self.model_id).name or self.model_id}:{digest}"

    def classify(self, image: "Image.Image") -> SpeciesPrediction:
        torch = self._torch
        with torch.no_grad():
            pixels = self._preprocess(image).unsqueeze(0).to(self.device)
            image_features = self._model.encode_image(pixels)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)[0]
            confidence, index = similarity.max(dim=-1)
            confidence = float(confidence.item())
            index = int(index.item())

        if confidence < self.min_confidence:
            return SpeciesPrediction(species=UNKNOWN_SPECIES, confidence=confidence, classifier_id=self.classifier_id)
        return SpeciesPrediction(
            species=self.species_list[index], confidence=confidence, classifier_id=self.classifier_id
        )
