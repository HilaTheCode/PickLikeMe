"""Ranking Mode wiring - Birds / Mammals / Birds+Mammals - and Burst-level
domain routing (ranking.combined). Uses stub detectors and a stub domain
detector throughout so no real model is loaded; the routing logic itself
is what is under test.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from picklikeme.bird_crop import COCO_BIRD_CLASS, COCO_PERSON_CLASS, crop_cache_path
from picklikeme.eyes.domain_detector import DOMAIN_BIRD, DOMAIN_MAMMAL, DOMAIN_UNCERTAIN, DomainPrediction
from picklikeme.ranking import (
    ClassicVisionBirdFusionStrategy,
    ClassicVisionCombinedStrategy,
    ClassicVisionEyePoseStrategy,
    ClassicVisionMammalFusionStrategy,
    ClassicVisionStrategy,
    available_strategies,
    get_strategy,
)
from picklikeme.ranking.combined import (
    ClassicVisionCombinedParams,
    _build_detector_for_domain,
    _classify_burst,
    _group_by_file_mtime,
)
from picklikeme.ranking.filters import NO_SUBJECT, UNSUPPORTED_SUBJECT


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


# ---------------------------------------------------------------------------
# Architecture: the crop (not COCO's class label) is the first object-
# presence gate; domain classification runs on the crop, purely to pick the
# eye detector; COCO's own label must never re-reject a crop this strategy
# already routed to a domain-appropriate detector. See ranking.filters.
# EyeFilter's own docstring for the mechanism (gate_by_subject_label) and
# eyes.domains.MAMMALS_PROFILE's docstring for the measured Colobus/"bird"
# case this whole architecture responds to.
# ---------------------------------------------------------------------------


def test_combined_strategy_does_not_gate_by_the_crops_coco_label():
    assert ClassicVisionCombinedStrategy._gate_by_subject_label is False


def test_single_domain_strategies_are_unaffected_and_still_gate_by_coco_label():
    """Only the domain-routing (Combined) strategy changes - a photographer
    who explicitly picked a single-domain strategy still gets that domain's
    original, unchanged eligibility behavior."""
    assert ClassicVisionStrategy._gate_by_subject_label is True
    assert ClassicVisionEyePoseStrategy._gate_by_subject_label is True
    assert ClassicVisionBirdFusionStrategy._gate_by_subject_label is True
    assert ClassicVisionMammalFusionStrategy._gate_by_subject_label is True


def _write_cache_entry(cache_dir, image_path, *, label, box=(10, 10, 60, 60)):
    """Minimal stand-in for what preprocessing already wrote for one image -
    same shape as test_ranking_strategies.py's own helper, kept local so this
    file does not depend on another test module's private helpers."""
    crop = crop_cache_path(cache_dir, image_path)
    crop.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    cv2.imwrite(str(crop), (rng.integers(0, 256, size=(64, 64, 3))).astype("uint8"))
    payload = {
        "version": 1,
        "source_size": [800, 600],
        "selected": {"box": list(box), "score": 0.9, "label": label},
        "detections": [{"box": list(box), "score": 0.9, "label": label}],
        "expanded_box": [5, 5, 65, 65],
    }
    crop.with_name(crop.stem + ".detections.json").write_text(json.dumps(payload), encoding="utf-8")
    return crop


class _StubClipDomainDetector:
    """Replaces the real CLIP model with a fixed verdict - the routing
    architecture is what these tests exercise, never real image content."""

    def __init__(self, domain: str, confidence: float = 0.9, **_kwargs) -> None:
        self._domain = domain
        self._confidence = confidence

    def predict(self, image_rgb):
        return DomainPrediction(domain=self._domain, confidence=self._confidence)


class _RealisticStubDetector:
    """Mirrors the actual eligibility shape of the real backends: EyePose-v0/
    SuperAnimal-Bird only support COCO's bird class; SuperAnimal-Quadruped
    supports anything (see its own module docstring - COCO simply has no
    class for most real mammal subjects, so gating it would make it useless).
    A real, accepted eye is always returned so a successful run proves the
    detector was actually reached and used, not merely "not rejected"."""

    def __init__(self, name: str) -> None:
        self.detector_id = name
        self.detect_calls = 0

    def supports(self, coco_label: int) -> bool:
        if self.detector_id == "superanimal-quadruped":
            return True
        return coco_label == COCO_BIRD_CLASS

    def detect(self, subject_crop_rgb):
        from picklikeme.eyes.detector import EyeDetection

        self.detect_calls += 1
        return EyeDetection(box=(1.0, 1.0, 5.0, 5.0), confidence=0.95, center=(3.0, 3.0), accepted=True)


def _run_combined(folder, cache_dir, analytics_db, monkeypatch, *, domain, image_paths, labels):
    """Runs the real ClassicVisionCombinedStrategy.rank_folder end to end,
    with only the two genuinely expensive models stubbed out (the CLIP
    domain classifier and the eye-detector backends) - everything else
    (Burst grouping, per-domain detector construction, the filter chain,
    scoring, CSV writing) is the real, unmodified code path."""
    import picklikeme.eyes.detector as detector_module
    from picklikeme.ranking import classic as classic_module
    from picklikeme.ranking import combined as combined_module

    folder.mkdir(exist_ok=True)
    for path, label in zip(image_paths, labels):
        Path(path).write_bytes(b"not really a raw file")
        _write_cache_entry(cache_dir, path, label=label)

    monkeypatch.setattr(classic_module, "build_cache", lambda *a, **k: {})
    monkeypatch.setattr(
        combined_module, "ClipDomainDetector", lambda **kw: _StubClipDomainDetector(domain)
    )
    # Two separate name bindings need patching: build_domain_fusion_detector
    # (bird/mammal branches) re-imports build_eye_detector fresh on every call
    # (a local import inside eyes.domains.build_domain_fusion_detector), so
    # patching the detector module reaches it - but combined.py's own
    # uncertain-domain branch imported the name once at module load time
    # (`from ..eyes import ... build_eye_detector`), which that patch alone
    # does not reach.
    monkeypatch.setattr(detector_module, "build_eye_detector", lambda name, **kw: _RealisticStubDetector(name))
    monkeypatch.setattr(combined_module, "build_eye_detector", lambda name, **kw: _RealisticStubDetector(name))

    return ClassicVisionCombinedStrategy().rank_folder(
        folder,
        params=ClassicVisionCombinedParams(),
        crop_cache_dir=cache_dir,
        device="cpu",
        analytics_db=analytics_db,
    )


def test_a_mammal_coco_mislabels_as_bird_still_reaches_the_mammal_pipeline(tmp_path, monkeypatch):
    """The regression case this whole architecture exists for: a real
    Colobus-style mammal that COCO itself confidently recorded as class 16
    ("bird") - see eyes.domains.MAMMALS_PROFILE's own docstring for the
    measured case. The crop is valid, the crop-based CLIP classifier
    correctly says MAMMAL, and the image must still reach - and be scored
    by - the mammal eye pipeline despite COCO's wrong label."""
    result = _run_combined(
        tmp_path / "shoot", tmp_path / "crops", tmp_path / "analytics.db", monkeypatch,
        domain=DOMAIN_MAMMAL, image_paths=[str(tmp_path / "shoot" / "colobus.nef")], labels=[COCO_BIRD_CLASS],
    )
    assert result["filtered"] == {}
    assert result["image_count"] == 1


def test_a_bird_coco_mislabels_as_something_else_still_reaches_the_bird_pipeline(tmp_path, monkeypatch):
    """The mirror case: a real bird COCO recorded under some other class
    (simulated here with the generic 'person' class id - any non-bird label
    demonstrates the same bug). Before this task's fix, EyeFilter's own COCO-
    label check (self._detector.supports(label)) would have rejected this as
    UNSUPPORTED_SUBJECT even though the crop is valid and CLIP correctly
    classified it as BIRD - re-litigating a question domain routing had
    already answered."""
    result = _run_combined(
        tmp_path / "shoot", tmp_path / "crops", tmp_path / "analytics.db", monkeypatch,
        domain=DOMAIN_BIRD, image_paths=[str(tmp_path / "shoot" / "bird.nef")], labels=[COCO_PERSON_CLASS],
    )
    assert result["filtered"] == {}
    assert result["image_count"] == 1


def test_uncertain_domain_is_not_rejected_as_no_subject(tmp_path, monkeypatch):
    """UNCERTAIN must route to every available model rather than reject the
    image - see _build_detector_for_domain_uncertain_tries_every_model above
    for the detector-selection half of this; this is the filter-chain half:
    a valid crop must still be scored, never turned into NO_SUBJECT or
    UNSUPPORTED_SUBJECT just because the domain call was a coin flip."""
    result = _run_combined(
        tmp_path / "shoot", tmp_path / "crops", tmp_path / "analytics.db", monkeypatch,
        domain=DOMAIN_UNCERTAIN, image_paths=[str(tmp_path / "shoot" / "ambiguous.nef")], labels=[COCO_PERSON_CLASS],
    )
    assert result["filtered"] == {}
    assert result["image_count"] == 1


def test_no_valid_crop_is_still_no_subject_and_the_eye_detector_never_runs(tmp_path, monkeypatch):
    """Stage 1 (does a valid crop/localization result exist at all) is
    unchanged by this task - only the COCO-label re-check at Stage 3 was
    removed. An image preprocessing never found a subject for must still be
    NO_SUBJECT, and the (expensive) eye detector must never be asked about
    it."""
    import picklikeme.eyes.detector as detector_module
    from picklikeme.ranking import classic as classic_module
    from picklikeme.ranking import combined as combined_module

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    no_subject_path = folder / "empty.nef"
    no_subject_path.write_bytes(b"not really a raw file")
    # Deliberately no cache entry written for it at all.

    monkeypatch.setattr(classic_module, "build_cache", lambda *a, **k: {})
    monkeypatch.setattr(
        combined_module, "ClipDomainDetector", lambda **kw: _StubClipDomainDetector(DOMAIN_UNCERTAIN)
    )

    calls = {"detect": 0}

    class _NeverCalled(_RealisticStubDetector):
        def detect(self, subject_crop_rgb):
            calls["detect"] += 1
            raise AssertionError("eye detector must never run on a NO_SUBJECT image")

    monkeypatch.setattr(detector_module, "build_eye_detector", lambda name, **kw: _NeverCalled(name))

    result = ClassicVisionCombinedStrategy().rank_folder(
        folder,
        params=ClassicVisionCombinedParams(),
        crop_cache_dir=cache_dir,
        device="cpu",
        analytics_db=tmp_path / "analytics.db",
    )

    assert result["filtered"] == {NO_SUBJECT: 1}
    assert result["image_count"] == 0
    assert calls["detect"] == 0


def test_coco_classification_cannot_prevent_a_valid_crop_from_continuing(tmp_path, monkeypatch):
    """Direct end-to-end proof of the core architectural requirement: the
    SAME COCO label (bird) reaches a successful outcome whether the
    crop-based domain classifier agrees with it (BIRD) or overrules it
    (MAMMAL) - because COCO's label is no longer consulted for eligibility
    at all in this strategy, only the crop-based domain verdict is."""
    for index, domain in enumerate([DOMAIN_BIRD, DOMAIN_MAMMAL]):
        folder = tmp_path / f"shoot_{index}"
        result = _run_combined(
            folder, tmp_path / f"crops_{index}", tmp_path / f"analytics_{index}.db", monkeypatch,
            domain=domain, image_paths=[str(folder / "subject.nef")], labels=[COCO_BIRD_CLASS],
        )
        assert result["filtered"] == {}, f"domain={domain} was incorrectly filtered"
        assert result["image_count"] == 1, f"domain={domain} did not produce a ranked result"
