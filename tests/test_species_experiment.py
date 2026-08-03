"""ExperimentMetadata / build_experiment_metadata - the reproducibility
record captured once per species-classification run (Part 2 of the
BioCLIP multi-backend infrastructure work). No test here constructs a real
classifier (that downloads a model) - a minimal stand-in object exposing
just the public attributes `build_experiment_metadata` reads is used
instead, the same "test the pure logic, not the model load" shape already
used in test_species_classifier.py.
"""

from __future__ import annotations

from picklikeme.species.experiment import ExperimentMetadata, build_experiment_metadata


class _StubClassifier:
    def __init__(
        self,
        *,
        model_id="hf-hub:imageomics/bioclip-2",
        species_list=("Kingfisher", "Osprey"),
        device="cpu",
        min_confidence=0.5,
        prompt_template="a photo of a {}",
        classifier_id="bioclip-2:aeb5a3073ad9",
    ):
        self.model_id = model_id
        self.species_list = species_list
        self.device = device
        self.min_confidence = min_confidence
        self.prompt_template = prompt_template
        self.classifier_id = classifier_id


def test_build_experiment_metadata_captures_the_backend_and_model() -> None:
    meta = build_experiment_metadata(_StubClassifier(), "bioclip2")

    assert isinstance(meta, ExperimentMetadata)
    assert meta.classifier_backend == "bioclip2"
    assert meta.model_id == "hf-hub:imageomics/bioclip-2"
    assert meta.species_count == 2
    assert meta.device == "cpu"


def test_backend_name_is_never_derived_from_classifier_id() -> None:
    """The exact bug this design avoids: classifier_id's model-name segment
    ("bioclip-2") is a different string from the registry key ("bioclip2")
    - if backend were ever derived from classifier_id instead of passed in,
    this would silently mislabel the experiment."""
    stub = _StubClassifier(classifier_id="bioclip-2:aeb5a3073ad9")
    meta = build_experiment_metadata(stub, "bioclip2")
    assert meta.classifier_backend == "bioclip2"
    assert meta.classifier_backend != "bioclip-2"  # the classifier_id's own model-name segment


def test_two_different_backends_for_the_same_model_id_are_distinguishable() -> None:
    """Registering the same underlying class under two different backend
    names must still produce two different experiment records."""
    stub = _StubClassifier()
    meta_a = build_experiment_metadata(stub, "bioclip2")
    meta_b = build_experiment_metadata(stub, "some-other-registry-name")
    assert meta_a.classifier_backend != meta_b.classifier_backend


def test_species_list_hash_changes_with_the_species_list() -> None:
    meta_a = build_experiment_metadata(_StubClassifier(species_list=("Kingfisher",)), "bioclip2")
    meta_b = build_experiment_metadata(_StubClassifier(species_list=("Kingfisher", "Osprey")), "bioclip2")
    assert meta_a.species_list_hash != meta_b.species_list_hash


def test_configuration_hash_changes_with_thresholds() -> None:
    meta_a = build_experiment_metadata(_StubClassifier(min_confidence=0.5), "bioclip2")
    meta_b = build_experiment_metadata(_StubClassifier(min_confidence=0.7), "bioclip2")
    assert meta_a.configuration_hash != meta_b.configuration_hash


def test_configuration_hash_is_stable_for_identical_thresholds() -> None:
    meta_a = build_experiment_metadata(_StubClassifier(min_confidence=0.5), "bioclip2")
    meta_b = build_experiment_metadata(_StubClassifier(min_confidence=0.5), "bioclip2")
    assert meta_a.configuration_hash == meta_b.configuration_hash


def test_species_list_filename_defaults_to_built_in_marker() -> None:
    meta = build_experiment_metadata(_StubClassifier(), "bioclip2")
    assert meta.species_list_filename == "(built-in default)"


def test_species_list_filename_reflects_a_custom_path() -> None:
    meta = build_experiment_metadata(_StubClassifier(), "bioclip2", species_list_path="my_species.txt")
    assert meta.species_list_filename == "my_species.txt"


def test_missing_optional_attributes_degrade_to_none_not_a_crash() -> None:
    """A future, non-BioCLIP backend that lacks min_confidence/prompt_
    template entirely must not break metadata collection for the whole
    run - see build_experiment_metadata's own docstring."""

    class _MinimalStub:
        model_id = "some://other-model"
        species_list = ()
        device = "cpu"
        classifier_id = "other:digest"

    meta = build_experiment_metadata(_MinimalStub(), "future-backend")
    assert meta.thresholds == {}
    assert meta.species_count == 0


def test_experiment_id_is_unique_per_call() -> None:
    meta_a = build_experiment_metadata(_StubClassifier(), "bioclip2")
    meta_b = build_experiment_metadata(_StubClassifier(), "bioclip2")
    assert meta_a.experiment_id != meta_b.experiment_id


def test_to_dict_round_trips_every_field() -> None:
    meta = build_experiment_metadata(_StubClassifier(), "bioclip2")
    payload = meta.to_dict()
    assert payload["classifier_backend"] == "bioclip2"
    assert payload["species_count"] == 2
    assert "thresholds" in payload and isinstance(payload["thresholds"], dict)


def test_gpu_fields_are_consistent_with_cuda_availability() -> None:
    import torch

    meta = build_experiment_metadata(_StubClassifier(), "bioclip2")
    if not torch.cuda.is_available():
        assert meta.cuda_available is False
        assert meta.gpu_name is None
    else:
        assert meta.cuda_available is True
