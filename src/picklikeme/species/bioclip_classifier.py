"""BioCLIP-2 - the default local, offline species classifier.

Also backs the original BioCLIP (v1) as a second, independently-selectable
backend - see `BIOCLIP_V1_MODEL_ID` below. Both are the same `open_clip`
OpenCLIP-family model loaded through the identical
`create_model_and_transforms`/`get_tokenizer` API (confirmed directly
against both models' official Hugging Face model cards and by loading each
one locally, including on CUDA - not assumed), so `BioClipSpeciesClassifier`
already needed no code change to serve either: it was already parameterized
by `model_id`. What differs between the two, verified the same way:

|                | BioCLIP (v1)                  | BioCLIP 2 (default)            |
|----------------|--------------------------------|---------------------------------|
| `model_id`     | `hf-hub:imageomics/bioclip`    | `hf-hub:imageomics/bioclip-2`  |
| Architecture   | ViT-B/16                       | ViT-L/14                       |
| Training data  | TreeOfLife-10M, ~450K taxa     | TreeOfLife-200M, ~952K taxa    |
| Released       | Nov 2023 (CVPR 2024)           | newer, larger                  |
| Local weights  | ~571MB                         | ~1.6GB                         |
| Preprocessing  | identical - Resize(224, bicubic) -> CenterCrop(224,224) -> Normalize(CLIP mean/std) for both |

Same dependencies (`open_clip_torch`, already a project dependency), same
one-time-download-then-offline shape, same Hugging Face Hub cache location
(`~/.cache/huggingface/hub/models--imageomics--<name>`) - registering a
second model is exactly `species/classifier.py` binding a second `model_id`
default via `functools.partial`, not a new class.

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

# The original BioCLIP (v1) - see the module docstring's comparison table.
# A second, independently-selectable backend (`species/classifier.py`
# registers it as "bioclip"), not a replacement for the default above.
BIOCLIP_V1_MODEL_ID = "hf-hub:imageomics/bioclip"

# Bumped whenever THIS class's own preprocessing/prompt/decision logic
# changes in a way that could change a prediction - independent of the
# model checkpoint's own version (see species/experiment.py's
# ExperimentMetadata.model_version) and independent of the open_clip
# library's own version. An experiment record needs all three, separately,
# to answer "what exactly produced this result" - see experiment.py's
# module docstring.
CLASSIFIER_VERSION = "1"

# CLIP-style zero-shot classification is sensitive to the prompt template;
# this is the standard generic CLIP convention, not BioCLIP's own
# fine-tuned template - a reasonable starting point, tunable later without
# touching anything outside this one constant.
DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"

# How many ranked (species, confidence) pairs classify() keeps on
# SpeciesPrediction.top_predictions by default - see classify()'s own
# docstring on why this costs nothing extra (the full ranking already
# exists in memory before the single winning answer is taken).
DEFAULT_TOP_N = 5

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
        device: str | None = None,
    ):
        if species_list_path is not None:
            species_list = _read_species_list(species_list_path)
        self.species_list = species_list or DEFAULT_SPECIES_LIST
        if not self.species_list:
            raise ValueError("species_list must not be empty")
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.min_confidence = min_confidence

        # Resolved here, at the source, not left to every caller to get
        # right - `device=None` (or omitted entirely) auto-selects CUDA
        # when available, the same fallback chain every other model in
        # this project already uses. Previously this defaulted to a
        # hardcoded "cpu", and both current callers (desktop/services.py,
        # species/cli.py) also independently defaulted to "cpu" - meaning
        # species classification silently never used a GPU regardless of
        # availability. Fixing the default here means a *future* caller
        # that also forgets to resolve a device gets the right behaviour
        # anyway, not just today's two known call sites - see
        # docs/BioCLIP_Backend_Architecture_Review.md.
        from ..platform import resolve_torch_device

        self.device = resolve_torch_device(device)

        # Lazy, exactly like BirdDetector's torch/torchvision: importing
        # this module (or even constructing a classifier for a --help run)
        # must never require these to be installed unless species
        # classification is actually used.
        import open_clip
        import torch

        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess

        prompts = [prompt_template.format(species) for species in self.species_list]
        with torch.no_grad():
            tokens = tokenizer(prompts).to(self.device)
            text_features = self._model.encode_text(tokens)
            self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        gpu_name = None
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001 - a GPU name query must never break construction
                gpu_name = None
        logger.info(
            "Species classifier ready: backend model_id=%s | requested device=%r -> execution device=%s "
            "| CUDA available=%s | GPU=%s | species_count=%d",
            model_id, device, self.device, torch.cuda.is_available(), gpu_name or "n/a", len(self.species_list),
        )

    @property
    def classifier_id(self) -> str:
        """The model plus exactly which species it was asked to consider -
        both determine the answer, so both must invalidate a cached one.
        See SpeciesClassifier's own docstring for why this matters.

        Derived from `self.model_id` itself (e.g. "bioclip-2" or "bioclip"),
        not hardcoded - this class now backs more than one BioCLIP version
        (see the module docstring), and a hardcoded "bioclip2" prefix would
        mislabel a BioCLIP-v1-backed instance's cached predictions as if
        they came from v2, silently defeating any future benchmark that
        compares the two by classifier_id. This does change the exact
        string for the existing default model (previously
        "bioclip2:bioclip-2:<digest>", now "bioclip-2:<digest>") - safe:
        `SpeciesCache` treats any classifier_id it does not recognise as a
        cache miss, never a wrong answer, so already-organized folders just
        get re-classified once, the same safe-by-construction behaviour
        `bird_crop.CROP_CACHE_VERSION` and `eyes.cache.EYE_CACHE_VERSION`
        bumps already rely on elsewhere in this project.
        """
        digest = hashlib.sha1("\n".join(self.species_list).encode("utf-8")).hexdigest()[:12]
        model_name = Path(self.model_id).name or self.model_id
        return f"{model_name}:{digest}"

    def classify(self, image: "Image.Image", *, top_n: int = DEFAULT_TOP_N) -> SpeciesPrediction:
        """The single winning answer (`species`/`confidence`/`classifier_id`
        - unchanged contract, exactly as before), plus `top_predictions`:
        the `top_n` best (species, confidence) pairs from the SAME forward
        pass - no second inference call. `top_n` only controls how much of
        the already-computed similarity vector is kept, never how much
        work is done; pass 0 to omit `top_predictions` entirely (e.g. a
        hot loop that only ever needs the single answer)."""
        torch = self._torch
        with torch.no_grad():
            pixels = self._preprocess(image).unsqueeze(0).to(self.device)
            image_features = self._model.encode_image(pixels)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)[0]
            confidence, index = similarity.max(dim=-1)
            confidence = float(confidence.item())
            index = int(index.item())

            top_predictions = None
            if top_n > 0:
                ranked = torch.argsort(similarity, descending=True)[:top_n]
                top_predictions = tuple(
                    (self.species_list[int(i)], float(similarity[int(i)].item())) for i in ranked
                )

        species = UNKNOWN_SPECIES if confidence < self.min_confidence else self.species_list[index]
        return SpeciesPrediction(
            species=species, confidence=confidence, classifier_id=self.classifier_id,
            top_predictions=top_predictions,
        )
