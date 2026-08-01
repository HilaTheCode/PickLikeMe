"""The SuperAnimal-Bird eye detector's reimplemented architecture.

PeakPic rebuilds the published DeepLabCut checkpoint's network from torch +
timm rather than depending on DeepLabCut itself (see
`eyes.superanimal_bird`'s module docstring for why). That reimplementation is
only correct as long as its module names and tensor shapes match the
checkpoint exactly - a mismatch would load partially and silently produce a
randomly-initialised head that still returns plausible-looking confidences.

These tests pin the contract that makes the strict load work, without
downloading the ~103 MB weights: the parameter names the checkpoint is keyed
by, the head shapes, and the body-part indices the eye lookup depends on.
The one test that does need the real weights is skipped unless they happen to
be cached already.
"""

from __future__ import annotations

import numpy as np
import pytest

from picklikeme.bird_crop import COCO_BIRD_CLASS, COCO_PERSON_CLASS
from picklikeme.eyes import EyeDetection, build_eye_detector
from picklikeme.eyes.superanimal_bird import (
    BILL_INDEX,
    BODYPARTS,
    CROWN_INDEX,
    DEFAULT_MAX_EYE_DISAGREEMENT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_WEIGHTS_DIR,
    INPUT_SIZE,
    LEFT_EYE_INDEX,
    MODEL_STRIDE,
    RIGHT_EYE_INDEX,
    WEIGHTS_FILENAME,
    SuperAnimalBirdEyeDetector,
    _build_network,
)


def _stub_detector(**overrides) -> SuperAnimalBirdEyeDetector:
    """A detector with no model, no torch, no weights - only the plain
    attributes `detect()` reads. `_predict_keypoints` is monkeypatched per
    test onto controlled (42, 3) keypoint arrays, so the accept/reject logic
    is tested directly without running the real network."""
    detector = SuperAnimalBirdEyeDetector.__new__(SuperAnimalBirdEyeDetector)
    detector.min_confidence = overrides.get("min_confidence", DEFAULT_MIN_CONFIDENCE)
    detector.eye_box_frac = overrides.get("eye_box_frac", 0.08)
    detector.max_eye_disagreement = overrides.get("max_eye_disagreement", DEFAULT_MAX_EYE_DISAGREEMENT)
    return detector


def _keypoints(*, left, right, crown=(50.0, 40.0, 0.9), bill=(50.0, 60.0, 0.9)) -> np.ndarray:
    """A (42, 3) array with every channel zeroed except the four this
    module's accept/reject logic actually reads - left_eye, right_eye,
    crown, bill (the head-scale reference)."""
    keypoints = np.zeros((len(BODYPARTS), 3), dtype=np.float64)
    keypoints[LEFT_EYE_INDEX] = left
    keypoints[RIGHT_EYE_INDEX] = right
    keypoints[CROWN_INDEX] = crown
    keypoints[BILL_INDEX] = bill
    return keypoints

pytest.importorskip("timm")
pytest.importorskip("torch")


def test_the_bodypart_list_matches_the_published_checkpoint() -> None:
    """42 parts, in the checkpoint's own channel order - the order IS the
    channel mapping, so the eye indices are only meaningful alongside it."""
    assert len(BODYPARTS) == 42
    assert BODYPARTS[LEFT_EYE_INDEX] == "left_eye"
    assert BODYPARTS[RIGHT_EYE_INDEX] == "right_eye"
    assert (LEFT_EYE_INDEX, RIGHT_EYE_INDEX) == (6, 11)


def test_the_rebuilt_network_is_keyed_exactly_as_the_checkpoint_is() -> None:
    """These names are what let `load_state_dict(..., strict=True)` succeed.
    Renaming any of them would break loading the published weights."""
    state_dict = _build_network().state_dict()

    for key in (
        "backbone.model.conv1.weight",
        "backbone.model.bn1.weight",
        "backbone.model.layer4.2.conv3.weight",
        "heads.bodypart.heatmap_head.deconv_layers.0.weight",
        "heads.bodypart.heatmap_head.deconv_layers.0.bias",
        "heads.bodypart.locref_head.deconv_layers.0.weight",
        "heads.bodypart.locref_head.deconv_layers.0.bias",
    ):
        assert key in state_dict, f"missing {key}"

    # GroupNorm, not BatchNorm: the checkpoint is resnet50_gn and carries no
    # running statistics at all. A BatchNorm backbone would have running_mean
    # keys the checkpoint cannot supply.
    assert not any("running_mean" in key for key in state_dict)
    assert not any("num_batches_tracked" in key for key in state_dict)


def test_the_heads_have_one_channel_per_bodypart_and_two_per_locref() -> None:
    state_dict = _build_network().state_dict()
    heatmap = state_dict["heads.bodypart.heatmap_head.deconv_layers.0.weight"]
    locref = state_dict["heads.bodypart.locref_head.deconv_layers.0.weight"]
    # ConvTranspose2d weights are (in_channels, out_channels, kH, kW).
    assert tuple(heatmap.shape) == (2048, len(BODYPARTS), 3, 3)
    assert tuple(locref.shape) == (2048, 2 * len(BODYPARTS), 3, 3)


def test_the_network_maps_the_expected_input_size_to_the_expected_stride() -> None:
    """Model stride is backbone output stride (16) halved by the head's
    transposed convolution - 8. The heatmap decode's coordinate maths depends
    on it, so a backbone or head change that moved it would be caught here."""
    import torch

    network = _build_network().eval()
    with torch.no_grad():
        heatmap, locref = network(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))

    assert MODEL_STRIDE == 8
    expected = INPUT_SIZE // MODEL_STRIDE
    # A kernel-3 stride-2 transposed convolution with no padding overshoots by
    # one, which the argmax decode tolerates; the point is the ~1/8 scale.
    assert abs(heatmap.shape[-1] - expected) <= 2
    assert heatmap.shape[1] == len(BODYPARTS)
    assert locref.shape[1] == 2 * len(BODYPARTS)


def test_the_detector_only_claims_to_understand_birds() -> None:
    """Running the bird model on a mammal produces a confident, wrong answer
    (verified on this project's crop cache - it put a tiger's "eyes" on its
    ear at 0.67/0.90). `supports` is what stops the caller trusting that."""
    supports = SuperAnimalBirdEyeDetector.supports
    assert supports(None, COCO_BIRD_CLASS) is True
    assert supports(None, COCO_PERSON_CLASS) is False
    assert supports(None, 21) is False  # cow


def test_build_eye_detector_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown eye detector"):
        build_eye_detector("no-such-detector")


# ---------------------------------------------------------------------------
# The left/right agreement check - the fix for the reported soaring-bird
# false positive (see the module docstring's "Confidence is not enough").
# Exercised directly against controlled keypoint arrays rather than the real
# model, so the accept/reject arithmetic is pinned independent of what the
# network happens to predict on any particular photo.
# ---------------------------------------------------------------------------


def test_agreeing_eye_channels_are_accepted(monkeypatch) -> None:
    """A profile shot: only one eye is real, but the occluded channel's
    prediction converges on the same pixel - exactly the DUCK case measured
    in the investigation. Small disagreement, high confidence -> accepted."""
    detector = _stub_detector(min_confidence=0.80, max_eye_disagreement=0.5)
    keypoints = _keypoints(left=(100.0, 100.0, 0.95), right=(101.0, 100.5, 0.90))
    monkeypatch.setattr(detector, "_predict_keypoints", lambda crop: keypoints)

    detection = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

    assert detection.accepted is True
    assert detection.confidence == pytest.approx(0.95)
    assert (detection.left.x, detection.left.y, detection.left.confidence) == pytest.approx((100.0, 100.0, 0.95))
    assert (detection.right.x, detection.right.y, detection.right.confidence) == pytest.approx((101.0, 100.5, 0.90))


def test_disagreeing_eye_channels_are_rejected_despite_high_confidence(monkeypatch) -> None:
    """The reported bug, reproduced directly: a confident (0.947) but
    hallucinated eye pair on a soaring bird with its head foreshortened
    between spread wings - measured on the real crop at a left/right
    separation of 28px against an 11px head scale (crown<->bill), giving a
    normalised disagreement of 2.56. Both numbers are used here as-is."""
    detector = _stub_detector(min_confidence=0.80, max_eye_disagreement=0.5)
    keypoints = _keypoints(
        left=(458.3, 67.6, 0.947), right=(430.3, 65.0, 0.863),
        crown=(440.5, 58.0, 0.90), bill=(441.2, 69.0, 0.95),
    )
    monkeypatch.setattr(detector, "_predict_keypoints", lambda crop: keypoints)

    detection = detector.detect(np.zeros((313, 808, 3), dtype=np.uint8))

    assert detection.confidence == pytest.approx(0.947)  # the confidence gate alone would pass this
    assert detection.accepted is False  # the agreement gate catches what confidence could not
    # The raw keypoints are still there for a debugging overlay to show, even
    # though this detection is rejected.
    assert detection.left is not None
    assert detection.right is not None


def test_low_confidence_is_rejected_before_the_agreement_check_even_runs(monkeypatch) -> None:
    detector = _stub_detector(min_confidence=0.80)
    keypoints = _keypoints(left=(100.0, 100.0, 0.50), right=(100.0, 100.0, 0.40))
    monkeypatch.setattr(detector, "_predict_keypoints", lambda crop: keypoints)

    detection = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

    assert detection.accepted is False
    assert detection.confidence == pytest.approx(0.50)


def test_a_degenerate_head_scale_does_not_explode_the_disagreement_ratio(monkeypatch) -> None:
    """crown and bill collapsing onto (almost) the same pixel must not turn
    a division into a spurious pass or a crash - MIN_HEAD_SCALE_PX floors it."""
    detector = _stub_detector(min_confidence=0.80, max_eye_disagreement=0.5)
    keypoints = _keypoints(
        left=(100.0, 100.0, 0.90), right=(100.5, 100.0, 0.85),
        crown=(50.0, 50.0, 0.9), bill=(50.0, 50.0, 0.9),  # coincide exactly
    )
    monkeypatch.setattr(detector, "_predict_keypoints", lambda crop: keypoints)

    detection = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))  # must not raise

    assert detection.accepted is True  # 0.5px separation is still tiny even against the floor


def test_the_default_threshold_matches_the_validated_investigation() -> None:
    """Pins the shipped default to the number the module docstring cites,
    so a future change to it is a deliberate edit, not an accidental one."""
    assert DEFAULT_MAX_EYE_DISAGREEMENT == 0.5


def test_an_eye_detection_carries_a_real_region_not_just_a_point() -> None:
    """Callers measure sharpness inside `box`, so it must always be a usable
    region even though the backing model regresses a keypoint."""
    detection = EyeDetection(box=(10.0, 12.0, 30.0, 32.0), confidence=0.8, center=(20.0, 22.0))
    x1, y1, x2, y2 = detection.box
    assert x2 > x1 and y2 > y1


@pytest.mark.skipif(
    not (DEFAULT_WEIGHTS_DIR / WEIGHTS_FILENAME).is_file(),
    reason="SuperAnimal-Bird weights not downloaded; skipping the real-inference check",
)
def test_the_real_checkpoint_loads_strictly_and_returns_a_bounded_box() -> None:
    """The end-to-end guarantee, when the weights are already present: the
    published checkpoint loads with nothing missing, and a detection's box is
    always clamped inside the crop it was found in."""
    detector = build_eye_detector("superanimal-bird", device="cpu", min_confidence=0.0)
    crop = np.random.default_rng(7).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    detection = detector.detect(crop)

    x1, y1, x2, y2 = detection.box
    assert 0.0 <= x1 < x2 <= 300.0
    assert 0.0 <= y1 < y2 <= 200.0
    assert 0.0 <= detection.confidence <= 1.0
    assert detection.detector_id == "superanimal-bird"
    # Both raw eye channels are always populated - one forward pass computes
    # every body part - regardless of whether the pair was accepted.
    assert detection.left is not None
    assert detection.right is not None


def test_an_empty_crop_is_declined_rather_than_crashing() -> None:
    """`detect` is called for every image in a folder; one unreadable crop
    must not end the run. It never returns None (see the module docstring -
    a debugging overlay needs raw data even for a rejected image), so an
    empty crop comes back as an explicitly unaccepted EyeDetection instead."""
    detector = SuperAnimalBirdEyeDetector.__new__(SuperAnimalBirdEyeDetector)
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)).accepted is False
    assert detector.detect(None).accepted is False
