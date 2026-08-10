"""SuperAnimal-Quadruped's reimplemented architecture and accept/reject
gate - the mammal-domain analogue of test_eye_detector.py's SuperAnimal-Bird
coverage. Pins the contract that makes strict checkpoint loading work
without downloading the ~160 MB weights; the one test that needs the real
weights is skipped unless they are already cached (mirrors both existing
eye-detector test files' own convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from picklikeme.bird_crop import COCO_BIRD_CLASS
from picklikeme.eyes import build_eye_detector
from picklikeme.eyes.superanimal_quadruped import (
    BODYPARTS,
    DEFAULT_MAX_EYE_DISAGREEMENT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_WEIGHTS_DIR,
    LEFT_EYE_INDEX,
    NECK_BASE_INDEX,
    NOSE_INDEX,
    RIGHT_EYE_INDEX,
    WEIGHTS_FILENAME,
    SuperAnimalQuadrupedEyeDetector,
    _build_network,
)

pytest.importorskip("timm")
pytest.importorskip("torch")


def _stub_detector(**overrides) -> SuperAnimalQuadrupedEyeDetector:
    detector = SuperAnimalQuadrupedEyeDetector.__new__(SuperAnimalQuadrupedEyeDetector)
    detector.min_confidence = overrides.get("min_confidence", DEFAULT_MIN_CONFIDENCE)
    detector.eye_box_frac = overrides.get("eye_box_frac", 0.08)
    detector.max_eye_disagreement = overrides.get("max_eye_disagreement", DEFAULT_MAX_EYE_DISAGREEMENT)
    return detector


def _keypoints(*, left, right, nose=(50.0, 40.0, 0.9), neck_base=(50.0, 80.0, 0.9)) -> np.ndarray:
    keypoints = np.zeros((len(BODYPARTS), 3), dtype=np.float64)
    keypoints[LEFT_EYE_INDEX] = left
    keypoints[RIGHT_EYE_INDEX] = right
    keypoints[NOSE_INDEX] = nose
    keypoints[NECK_BASE_INDEX] = neck_base
    return keypoints


def test_the_bodypart_list_matches_the_published_checkpoint():
    assert len(BODYPARTS) == 39
    assert BODYPARTS[LEFT_EYE_INDEX] == "left_eye"
    assert BODYPARTS[RIGHT_EYE_INDEX] == "right_eye"
    assert (RIGHT_EYE_INDEX, LEFT_EYE_INDEX) == (5, 10)


def test_the_rebuilt_network_is_keyed_exactly_as_the_checkpoint_is():
    """These names are what let `load_state_dict(..., strict=True)`
    succeed - see the module docstring's empirical verification against the
    real published checkpoint."""
    state_dict = _build_network().state_dict()

    for key in (
        "backbone.model.conv1.weight",
        "backbone.model.bn1.weight",
        "backbone.model.stage4.0.branches.0.0.conv1.weight",
        "backbone.model.classifier.weight",
        "backbone.model.final_layer.0.weight",
        "backbone.model.downsamp_modules.0.0.weight",
        "heads.bodypart.heatmap_head.model.weight",
        "heads.bodypart.heatmap_head.model.bias",
    ):
        assert key in state_dict, f"missing {key}"

    # incre_modules was explicitly removed - the checkpoint does not have
    # it (see the module docstring) - so it must not appear here either, or
    # strict loading would fail with "missing keys".
    assert not any(key.startswith("backbone.model.incre_modules") for key in state_dict)


def test_the_heatmap_head_has_one_channel_per_bodypart():
    state_dict = _build_network().state_dict()
    weight = state_dict["heads.bodypart.heatmap_head.model.weight"]
    # ConvTranspose2d weight layout is (in_channels, out_channels, kH, kW).
    assert tuple(weight.shape) == (32, len(BODYPARTS), 1, 1)


def test_the_network_produces_a_heatmap_at_stride_four():
    import torch

    network = _build_network().eval()
    with torch.no_grad():
        heatmap = network(torch.zeros(1, 3, 256, 256))

    assert heatmap.shape == (1, len(BODYPARTS), 64, 64)  # 256 / 4 = 64


def test_the_detector_declares_support_for_any_subject():
    """Unlike EyePose-v0/SuperAnimal-Bird, this detector's own supports()
    always returns True - domain eligibility for Mammals is decided by the
    Ranking Mode (see eyes.domains), not by an unreliable upstream COCO
    label (see the module docstring's own worked example)."""
    detector = SuperAnimalQuadrupedEyeDetector.__new__(SuperAnimalQuadrupedEyeDetector)
    assert detector.supports(COCO_BIRD_CLASS) is True
    assert detector.supports(999) is True


def test_build_eye_detector_knows_this_backend():
    """Registered, even without constructing it here (that needs the real
    weights) - an unregistered name would not be listed in the "Available:"
    error message below."""
    with pytest.raises(ValueError, match="superanimal-quadruped"):
        build_eye_detector("definitely-not-registered")


def test_agreeing_eye_channels_are_accepted(monkeypatch):
    detector = _stub_detector(min_confidence=0.80, max_eye_disagreement=0.5)
    keypoints = _keypoints(left=(100.0, 100.0, 0.95), right=(101.0, 100.5, 0.90))
    monkeypatch.setattr(detector, "_predict_keypoints", lambda crop: keypoints)

    detection = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

    assert detection.accepted is True
    assert detection.confidence == pytest.approx(0.95)


def test_disagreeing_eye_channels_are_rejected_despite_high_confidence(monkeypatch):
    detector = _stub_detector(min_confidence=0.80, max_eye_disagreement=0.5)
    keypoints = _keypoints(
        left=(458.3, 67.6, 0.947), right=(430.3, 65.0, 0.863),
        nose=(440.5, 58.0, 0.90), neck_base=(441.2, 69.0, 0.95),
    )
    monkeypatch.setattr(detector, "_predict_keypoints", lambda crop: keypoints)

    detection = detector.detect(np.zeros((313, 808, 3), dtype=np.uint8))

    assert detection.confidence == pytest.approx(0.947)
    assert detection.accepted is False


def test_low_confidence_is_rejected_before_the_agreement_check_even_runs(monkeypatch):
    detector = _stub_detector(min_confidence=0.80)
    keypoints = _keypoints(left=(100.0, 100.0, 0.50), right=(100.0, 100.0, 0.40))
    monkeypatch.setattr(detector, "_predict_keypoints", lambda crop: keypoints)

    detection = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

    assert detection.accepted is False
    assert detection.confidence == pytest.approx(0.50)


def test_an_empty_crop_is_declined_rather_than_crashing():
    detector = SuperAnimalQuadrupedEyeDetector.__new__(SuperAnimalQuadrupedEyeDetector)
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)).accepted is False
    assert detector.detect(None).accepted is False


@pytest.mark.skipif(
    not (DEFAULT_WEIGHTS_DIR / WEIGHTS_FILENAME).is_file(),
    reason="SuperAnimal-Quadruped weights not downloaded; skipping the real-inference check",
)
def test_the_real_checkpoint_loads_strictly_and_returns_a_bounded_box():
    detector = build_eye_detector("superanimal-quadruped", device="cpu", min_confidence=0.0)
    crop = np.random.default_rng(7).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    detection = detector.detect(crop)

    x1, y1, x2, y2 = detection.box
    assert 0.0 <= x1 < x2 <= 300.0
    assert 0.0 <= y1 < y2 <= 200.0
    assert 0.0 <= detection.confidence <= 1.0
    assert detection.detector_id == "superanimal-quadruped"
    assert detection.left is not None
    assert detection.right is not None
