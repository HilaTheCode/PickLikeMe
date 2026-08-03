"""BioClipSpeciesClassifier.classify()'s top_predictions field (Part 4 of
the multi-backend infrastructure work) - populated from the SAME forward
pass that produces the single winning answer, at zero extra inference
cost. Built by constructing a classifier via __new__ (bypassing __init__,
so no model is downloaded) and wiring in real, small torch tensors for
_model/_text_features - real tensor math is exercised, only the
multi-hundred-MB model load is skipped.
"""

from __future__ import annotations

import torch

from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier
from picklikeme.species.classifier import UNKNOWN_SPECIES


class _FakeVisionModel:
    """encode_image always returns the same fixed embedding - what
    "the image" looks like to this fake model is irrelevant; only the
    ranking math downstream of it is under test."""

    def __init__(self, embedding: torch.Tensor):
        self._embedding = embedding

    def encode_image(self, pixels):
        return self._embedding.unsqueeze(0)


def _build_classifier(species_list, text_features, image_embedding, min_confidence=0.5) -> BioClipSpeciesClassifier:
    clf = BioClipSpeciesClassifier.__new__(BioClipSpeciesClassifier)
    clf.species_list = species_list
    clf.min_confidence = min_confidence
    clf.model_id = "hf-hub:imageomics/bioclip-2"
    clf.prompt_template = "a photo of a {}"
    clf.device = "cpu"
    clf._torch = torch
    clf._model = _FakeVisionModel(image_embedding)
    clf._preprocess = lambda image: torch.zeros(3, 4, 4)  # unused by the fake model, just needs to exist
    clf._text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return clf


def test_top_predictions_are_ranked_by_confidence_descending() -> None:
    species = ("Kingfisher", "Osprey", "Egret")
    # Orthogonal-ish text embeddings so the image embedding clearly favors one.
    text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    image_embedding = torch.tensor([1.0, 0.0])

    clf = _build_classifier(species, text_features, image_embedding)
    prediction = clf.classify(object(), top_n=3)

    assert prediction.species == "Kingfisher"
    assert prediction.top_predictions is not None
    assert len(prediction.top_predictions) == 3
    names = [name for name, _confidence in prediction.top_predictions]
    assert names[0] == "Kingfisher"
    confidences = [confidence for _name, confidence in prediction.top_predictions]
    assert confidences == sorted(confidences, reverse=True)


def test_top_n_zero_omits_top_predictions_entirely() -> None:
    species = ("Kingfisher", "Osprey")
    text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    image_embedding = torch.tensor([1.0, 0.0])

    clf = _build_classifier(species, text_features, image_embedding)
    prediction = clf.classify(object(), top_n=0)

    assert prediction.top_predictions is None
    assert prediction.species == "Kingfisher"  # the single-answer contract is unaffected


def test_top_predictions_still_populate_even_when_the_winner_is_unknown() -> None:
    """Below-threshold results still expose the full ranking - useful for
    analytics precisely because these are the borderline cases."""
    species = ("Kingfisher", "Osprey")
    text_features = torch.tensor([[1.0, 0.0], [0.99, 0.01]])  # near-tied -> low winning confidence
    image_embedding = torch.tensor([1.0, 0.0])

    clf = _build_classifier(species, text_features, image_embedding, min_confidence=0.99)
    prediction = clf.classify(object(), top_n=2)

    assert prediction.species == UNKNOWN_SPECIES
    assert prediction.top_predictions is not None
    assert len(prediction.top_predictions) == 2


def test_default_top_n_is_five() -> None:
    from picklikeme.species.bioclip_classifier import DEFAULT_TOP_N

    assert DEFAULT_TOP_N == 5

    species = tuple(f"Species-{i}" for i in range(10))
    text_features = torch.eye(10)[:, :2].float()
    image_embedding = torch.tensor([1.0, 0.0])

    clf = _build_classifier(species, text_features, image_embedding)
    prediction = clf.classify(object())  # no explicit top_n - must use the default

    assert len(prediction.top_predictions) == DEFAULT_TOP_N
