"""The species-classification backend registry: `available_classifiers()`,
`build_classifier()`'s name-to-model dispatch, and `BioClipSpeciesClassifier.
classifier_id`'s derivation from `model_id` - the piece that makes two
different BioCLIP-family models produce distinguishable cache keys (see
`species/bioclip_classifier.py`'s own docstring on why a hardcoded prefix
would have silently mislabeled a BioCLIP-v1-backed instance's predictions).

None of these tests construct a real classifier (that would download and
run a multi-hundred-MB model) - `build_classifier`'s dispatch is checked by
capturing what `BioClipSpeciesClassifier.__init__` was called with, and
`classifier_id` is checked directly on an instance built via `__new__`
(bypassing `__init__` entirely), the same "test the pure logic, not the
model load" shape `test_eyepose_v0.py`/`test_eye_detector.py` already use
for their own heavy backends.
"""

from __future__ import annotations

import pytest

from picklikeme.species.classifier import (
    AVAILABLE_CLASSIFIERS,
    ClassifierInfo,
    available_classifiers,
    build_classifier,
)


def test_available_classifiers_lists_both_bioclip_versions() -> None:
    infos = available_classifiers()
    ids = {info.classifier_id for info in infos}
    assert ids == {"bioclip2", "bioclip"}
    assert all(isinstance(info, ClassifierInfo) for info in infos)
    assert all(info.display_name and info.description for info in infos)


def test_available_classifiers_returns_the_same_tuple_object() -> None:
    """Pure data, listable without constructing or importing a model - see
    build_classifier's own docstring on why the heavy import stays lazy."""
    assert available_classifiers() is AVAILABLE_CLASSIFIERS


def test_build_classifier_bioclip2_uses_the_bioclip2_model_id(monkeypatch) -> None:
    import picklikeme.species.bioclip_classifier as bioclip_module

    captured: dict = {}

    class _StubClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(bioclip_module, "BioClipSpeciesClassifier", _StubClassifier)

    build_classifier("bioclip2", device="cpu")

    assert captured.get("model_id", bioclip_module.DEFAULT_MODEL_ID) == bioclip_module.DEFAULT_MODEL_ID
    assert "model_id" not in captured  # bioclip2 relies on the class's own default, never passes one


def test_build_classifier_bioclip_v1_forces_the_v1_model_id(monkeypatch) -> None:
    """The whole point of the registry: "bioclip" must never silently
    resolve to the same model as "bioclip2" - that would make a benchmark
    comparing the two meaningless."""
    import picklikeme.species.bioclip_classifier as bioclip_module

    captured: dict = {}

    class _StubClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(bioclip_module, "BioClipSpeciesClassifier", _StubClassifier)

    build_classifier("bioclip", device="cpu")

    assert captured["model_id"] == bioclip_module.BIOCLIP_V1_MODEL_ID
    assert bioclip_module.BIOCLIP_V1_MODEL_ID != bioclip_module.DEFAULT_MODEL_ID


def test_build_classifier_forwards_other_kwargs_to_either_backend(monkeypatch) -> None:
    import picklikeme.species.bioclip_classifier as bioclip_module

    captured: dict = {}

    class _StubClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(bioclip_module, "BioClipSpeciesClassifier", _StubClassifier)

    build_classifier("bioclip", min_confidence=0.7, device="cuda", species_list_path="x.txt")

    assert captured["min_confidence"] == 0.7
    assert captured["device"] == "cuda"
    assert captured["species_list_path"] == "x.txt"


def test_build_classifier_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown classifier"):
        build_classifier("not-a-real-backend")


def test_classifier_id_is_derived_from_model_id_not_hardcoded() -> None:
    """Two instances backed by different BioCLIP models must never produce
    the same classifier_id, even with an identical species list - otherwise
    SpeciesCache would silently serve one model's cached answer to a caller
    using the other."""
    from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier

    v2 = BioClipSpeciesClassifier.__new__(BioClipSpeciesClassifier)
    v2.model_id = "hf-hub:imageomics/bioclip-2"
    v2.species_list = ("Kingfisher", "Osprey")

    v1 = BioClipSpeciesClassifier.__new__(BioClipSpeciesClassifier)
    v1.model_id = "hf-hub:imageomics/bioclip"
    v1.species_list = ("Kingfisher", "Osprey")

    assert v2.classifier_id != v1.classifier_id
    assert v2.classifier_id.startswith("bioclip-2:")
    assert v1.classifier_id.startswith("bioclip:")


def test_classifier_id_changes_with_the_species_list() -> None:
    from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier

    a = BioClipSpeciesClassifier.__new__(BioClipSpeciesClassifier)
    a.model_id = "hf-hub:imageomics/bioclip-2"
    a.species_list = ("Kingfisher",)

    b = BioClipSpeciesClassifier.__new__(BioClipSpeciesClassifier)
    b.model_id = "hf-hub:imageomics/bioclip-2"
    b.species_list = ("Kingfisher", "Osprey")

    assert a.classifier_id != b.classifier_id
