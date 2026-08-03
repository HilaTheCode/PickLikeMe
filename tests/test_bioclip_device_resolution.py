"""BioClipSpeciesClassifier's device resolution - the actual root cause of
the "Desktop species classification silently defaults to CPU" finding from
docs/BioCLIP_Backend_Architecture_Review.md Section 7. Fixed at the source
(the class's own __init__), not only at its two known callers, so a future
caller that also forgets to pass a device still gets the right behaviour.

No real model is downloaded here - `open_clip.create_model_and_transforms`/
`get_tokenizer` are stubbed with minimal fakes, matching this project's own
"stub the heavy model, test the wiring" convention (test_eye_detector.py,
test_eyepose_v0.py). The device-resolution logic under test runs for real;
only the multi-hundred-MB model load is replaced.
"""

from __future__ import annotations

import inspect

import pytest
import torch


class _FakeTensor:
    def to(self, device):
        return self

    def norm(self, dim=-1, keepdim=True):
        return torch.ones(1)

    def __truediv__(self, other):
        return self

    def __getitem__(self, index):
        return self


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def encode_text(self, tokens):
        return _FakeTensor()


def _stub_open_clip(monkeypatch):
    import open_clip

    monkeypatch.setattr(
        open_clip, "create_model_and_transforms", lambda model_id: (_FakeModel(), None, lambda img: None)
    )
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda model_id: lambda prompts: _FakeTensor())


def test_device_defaults_to_none_not_a_hardcoded_cpu() -> None:
    """The regression check that needs no model load at all: the
    constructor's own signature must never again default to "cpu"."""
    from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier

    default = inspect.signature(BioClipSpeciesClassifier.__init__).parameters["device"].default
    assert default is None, (
        "device must default to None (auto-resolve), not a hardcoded device string - "
        "this is exactly the bug found in the architecture review"
    )


def test_omitting_device_resolves_to_cuda_when_available(monkeypatch) -> None:
    from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier

    _stub_open_clip(monkeypatch)
    classifier = BioClipSpeciesClassifier(species_list=("Kingfisher",))

    if torch.cuda.is_available():
        assert classifier.device == "cuda"
    else:
        assert classifier.device == "cpu"


def test_an_explicit_device_is_still_honoured(monkeypatch) -> None:
    from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier

    _stub_open_clip(monkeypatch)
    classifier = BioClipSpeciesClassifier(species_list=("Kingfisher",), device="cpu")
    assert classifier.device == "cpu"


def test_construction_logs_the_resolved_device_and_gpu_name(monkeypatch, caplog) -> None:
    """Part 3's explicit ask: session-start logging reporting the selected/
    execution device, GPU model, and that the model loaded."""
    import logging

    from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier

    _stub_open_clip(monkeypatch)
    with caplog.at_level(logging.INFO, logger="picklikeme.species.bioclip_classifier"):
        classifier = BioClipSpeciesClassifier(species_list=("Kingfisher",))

    messages = "\n".join(record.message for record in caplog.records)
    assert "execution device=" in messages
    assert "CUDA available=" in messages
    assert classifier.device in messages


def test_organize_by_species_service_no_longer_hardcodes_cpu() -> None:
    """The other half of the same finding: ReviewService.organize_by_species
    must default to auto (None), not "cpu" - see desktop/services.py."""
    from picklikeme.desktop.services import ReviewService

    default = inspect.signature(ReviewService.organize_by_species).parameters["device"].default
    assert default is None


def test_cli_no_longer_hardcodes_cpu() -> None:
    from picklikeme.species.cli import build_arrange_species_parser

    parser = build_arrange_species_parser()
    device_action = next(a for a in parser._actions if a.dest == "device")
    assert device_action.default is None
