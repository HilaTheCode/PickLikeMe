"""Ranking Mode domain profiles (eyes.domains) - which models each of
Birds/Mammals selects, and that EyePose-v0 stays exclusively a Bird model.
"""

from __future__ import annotations

import pytest

from picklikeme.eyes.domains import (
    BIRDS_PROFILE,
    MAMMALS_PROFILE,
    RANKING_MODE_BIRDS,
    RANKING_MODE_MAMMALS,
    available_ranking_modes,
    build_domain_fusion_detector,
    domain_profile,
)


def test_birds_profile_uses_eyepose_v0_and_superanimal_bird():
    assert "eyepose-v0" in BIRDS_PROFILE.detector_ids
    assert "superanimal-bird" in BIRDS_PROFILE.detector_ids
    assert BIRDS_PROFILE.gate_by_subject_label is True


def test_mammals_profile_does_not_use_eyepose_v0():
    """The core semantic requirement: EyePose-v0 sometimes placing a point
    on a monkey eye is not evidence it belongs in the Mammal pipeline."""
    assert "eyepose-v0" not in MAMMALS_PROFILE.detector_ids
    assert "superanimal-bird" not in MAMMALS_PROFILE.detector_ids
    assert "superanimal-quadruped" in MAMMALS_PROFILE.detector_ids


def test_mammals_profile_is_not_gated_by_the_upstream_coco_label():
    """COCO has no class for most real safari species - see the module's
    own docstring for the measured Colobus-monkey-as-"bird" example this is
    responding to."""
    assert MAMMALS_PROFILE.gate_by_subject_label is False


def test_domain_profile_resolves_known_modes():
    assert domain_profile(RANKING_MODE_BIRDS) is BIRDS_PROFILE
    assert domain_profile(RANKING_MODE_MAMMALS) is MAMMALS_PROFILE


def test_domain_profile_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="Unknown ranking mode"):
        domain_profile("reptiles")


def test_available_ranking_modes_lists_both():
    modes = {p.ranking_mode for p in available_ranking_modes()}
    assert modes == {RANKING_MODE_BIRDS, RANKING_MODE_MAMMALS}


def test_model_weights_are_configurable_per_domain():
    weights = {mw.detector_id: mw.weight for mw in BIRDS_PROFILE.default_model_weights}
    assert weights["eyepose-v0"] > 0.0
    assert weights["superanimal-bird"] > 0.0
    assert weights["eyepose-v0"] != weights["superanimal-bird"]


def test_build_domain_fusion_detector_uses_the_profiles_own_detectors(monkeypatch):
    """The Fusion Engine itself stays domain-agnostic (see fusion.py) -
    build_domain_fusion_detector is what supplies it the right sub-detector
    ids for a Ranking Mode. Verified here by monkeypatching the underlying
    factory rather than loading real models."""
    requested: list[str] = []

    class _Stub:
        def __init__(self, name):
            self.detector_id = name

        def supports(self, label):
            return True

        def detect(self, crop):
            raise AssertionError("not called in this test")

    def fake_build_eye_detector(name, **kwargs):
        requested.append(name)
        return _Stub(name)

    # build_domain_fusion_detector does `from .detector import
    # build_eye_detector` inside its own body (a lazy, per-call import), so
    # patching the attribute on the detector module - resolved at call time
    # - is enough; no real model is ever constructed here.
    import picklikeme.eyes.detector as detector_module

    monkeypatch.setattr(detector_module, "build_eye_detector", fake_build_eye_detector)

    fusion = build_domain_fusion_detector(RANKING_MODE_MAMMALS, device="cpu")
    assert requested == ["superanimal-quadruped"]
    assert fusion.config.model_weights[0].detector_id == "superanimal-quadruped"
