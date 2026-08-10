"""Ranking Mode wiring - Birds / Mammals / Birds+Mammals - and Burst-level
domain routing (ranking.combined). Uses stub detectors and a stub domain
detector throughout so no real model is loaded; the routing logic itself
is what is under test.
"""

from __future__ import annotations

import os
import time

import pytest

from picklikeme.ranking import (
    ClassicVisionBirdFusionStrategy,
    ClassicVisionCombinedStrategy,
    ClassicVisionMammalFusionStrategy,
    available_strategies,
    get_strategy,
)
from picklikeme.ranking.combined import (
    ClassicVisionCombinedParams,
    _build_detector_for_domain,
    _classify_burst,
    _group_by_file_mtime,
)


def test_birds_and_mammals_fusion_strategies_are_registered():
    ids = {s.strategy_id for s in available_strategies()}
    assert "classic-vision-fusion-birds" in ids
    assert "classic-vision-fusion-mammals" in ids
    assert "classic-vision-fusion-combined" in ids


def test_get_strategy_constructs_each_fusion_mode():
    assert isinstance(get_strategy("classic-vision-fusion-birds"), ClassicVisionBirdFusionStrategy)
    assert isinstance(get_strategy("classic-vision-fusion-mammals"), ClassicVisionMammalFusionStrategy)
    assert isinstance(get_strategy("classic-vision-fusion-combined"), ClassicVisionCombinedStrategy)


def test_birds_fusion_strategy_configures_eyepose_v0():
    strategy = ClassicVisionBirdFusionStrategy()
    params = strategy.params_class()
    kwargs = strategy._eye_detector_kwargs(params)
    detector_ids = {mw.detector_id for mw in kwargs["config"].model_weights}
    assert "eyepose-v0" in detector_ids


def test_mammals_fusion_strategy_never_configures_eyepose_v0():
    """The bird-specific model must never become the primary mammal
    detector - see eyes.domains's own module docstring."""
    strategy = ClassicVisionMammalFusionStrategy()
    params = strategy.params_class()
    kwargs = strategy._eye_detector_kwargs(params)
    detector_ids = {mw.detector_id for mw in kwargs["config"].model_weights}
    assert "eyepose-v0" not in detector_ids
    assert "superanimal-bird" not in detector_ids
    assert "superanimal-quadruped" in detector_ids


def test_model_weights_are_configurable_through_params():
    strategy = ClassicVisionBirdFusionStrategy()
    params = strategy.params_class(eyepose_v0_model_weight=0.9, superanimal_bird_model_weight=0.1)
    kwargs = strategy._eye_detector_kwargs(params)
    weights = {mw.detector_id: mw.weight for mw in kwargs["config"].model_weights}
    assert weights["eyepose-v0"] == pytest.approx(0.9)
    assert weights["superanimal-bird"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Burst-level domain routing (ranking.combined)
# ---------------------------------------------------------------------------


def test_group_by_file_mtime_splits_on_a_large_gap(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.jpg"
    for path in (a, b, c):
        path.write_bytes(b"x")

    now = time.time()
    os.utime(a, (now, now))
    os.utime(b, (now + 0.5, now + 0.5))  # same burst as a
    os.utime(c, (now + 30.0, now + 30.0))  # a clear gap - a separate burst

    bursts = _group_by_file_mtime([str(a), str(b), str(c)], max_gap_seconds=2.0)

    sizes = sorted(len(group) for group in bursts)
    assert sizes == [1, 2]


class _StubDomainDetector:
    """Returns a fixed verdict regardless of the crop it is given - the
    Burst-routing logic is what is under test here, not the real CLIP
    classifier (see eyes.domain_detector's own tests, if any real-model
    inference test exists) or any real image content."""

    def __init__(self, domain: str) -> None:
        self._domain = domain

    def predict(self, image_rgb):
        from picklikeme.eyes.domain_detector import DomainPrediction

        return DomainPrediction(domain=self._domain, confidence=0.9)


def test_classify_burst_returns_uncertain_when_no_crop_is_cached(tmp_path):
    """No cached crop for any member - classification cannot run, and the
    honest answer is uncertain, not a guess."""
    verdict = _classify_burst(["missing1.jpg", "missing2.jpg"], tmp_path, _StubDomainDetector("bird"))
    assert verdict == "uncertain"


def test_build_detector_for_domain_bird_excludes_mammal_models(monkeypatch):
    import picklikeme.eyes.detector as detector_module

    class _Stub:
        def __init__(self, name):
            self.detector_id = name

        def supports(self, label):
            return True

        def detect(self, crop):
            raise AssertionError("not called in this test")

    monkeypatch.setattr(detector_module, "build_eye_detector", lambda name, **kw: _Stub(name))

    params = ClassicVisionCombinedParams()
    detector = _build_detector_for_domain("bird", params, "cpu")

    sub_ids = {sub.detector_id for sub in detector._sub_detectors}
    assert sub_ids == {"eyepose-v0", "superanimal-bird"}


def test_build_detector_for_domain_mammal_excludes_eyepose_v0(monkeypatch):
    import picklikeme.eyes.detector as detector_module

    class _Stub:
        def __init__(self, name):
            self.detector_id = name

        def supports(self, label):
            return True

        def detect(self, crop):
            raise AssertionError("not called in this test")

    monkeypatch.setattr(detector_module, "build_eye_detector", lambda name, **kw: _Stub(name))

    params = ClassicVisionCombinedParams()
    detector = _build_detector_for_domain("mammal", params, "cpu")

    sub_ids = {sub.detector_id for sub in detector._sub_detectors}
    assert sub_ids == {"superanimal-quadruped"}
    assert "eyepose-v0" not in sub_ids


def test_build_detector_for_domain_uncertain_tries_every_model(monkeypatch):
    """OTHER/UNCERTAIN: rather than guessing the domain, every available
    model participates and the shared Fusion layer's own agreement/
    geometric-validity logic decides - see ranking.combined's own module
    docstring."""
    import picklikeme.eyes.detector as detector_module

    class _Stub:
        def __init__(self, name):
            self.detector_id = name

        def supports(self, label):
            return True

        def detect(self, crop):
            raise AssertionError("not called in this test")

    monkeypatch.setattr(detector_module, "build_eye_detector", lambda name, **kw: _Stub(name))

    params = ClassicVisionCombinedParams()
    detector = _build_detector_for_domain("uncertain", params, "cpu")

    sub_ids = {sub.detector_id for sub in detector._sub_detectors}
    assert sub_ids == {"eyepose-v0", "superanimal-bird", "superanimal-quadruped"}
