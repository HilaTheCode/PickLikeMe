"""The shared Fusion/Validation layer - exercised against controlled stub
detectors (never a real model), so the fusion arithmetic itself is pinned
independent of what any particular backend happens to predict on a real
photo, the same "stub the model, test the decision logic" approach
test_eye_detector.py already uses for SuperAnimal-Bird's own accept/reject
gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from picklikeme.eyes.detector import EyeDetection
from picklikeme.eyes.fusion import (
    DEFAULT_MIN_FUSED_CONFIDENCE,
    STATUS_AGREE,
    STATUS_DISAGREEMENT,
    STATUS_DISAGREEMENT_RESOLVED,
    STATUS_LOW_CONFIDENCE,
    STATUS_NO_DETECTION,
    STATUS_SINGLE_MODEL,
    FusionConfig,
    FusionEyeDetector,
    ModelWeight,
)

CROP = np.zeros((200, 200, 3), dtype=np.uint8)


class _StubDetector:
    """A minimal `EyeDetector` returning a fixed, caller-supplied
    `EyeDetection` - exactly what `fusion.py`'s own algorithm needs to be
    tested against controlled inputs."""

    def __init__(self, detector_id: str, detection: EyeDetection, supports_result: bool = True) -> None:
        self.detector_id = detector_id
        self._detection = detection
        self._supports_result = supports_result

    def supports(self, coco_label: int) -> bool:
        return self._supports_result

    def detect(self, subject_crop_rgb) -> EyeDetection:
        return self._detection


def _detection(x, y, confidence, accepted, detector_id="stub") -> EyeDetection:
    return EyeDetection(
        box=(x - 5, y - 5, x + 5, y + 5),
        confidence=confidence,
        center=(x, y),
        detector_id=detector_id,
        accepted=accepted,
    )


def _fusion(detectors, **config_overrides) -> FusionEyeDetector:
    config = FusionConfig(
        model_weights=tuple(ModelWeight(d.detector_id, w) for d, w in detectors),
        **config_overrides,
    )
    return FusionEyeDetector(sub_detectors=[d for d, _ in detectors], config=config)


def test_two_close_confident_predictions_are_fused_and_accepted():
    a = _StubDetector("model-a", _detection(100.0, 100.0, 0.95, True, "model-a"))
    b = _StubDetector("model-b", _detection(102.0, 101.0, 0.90, True, "model-b"))
    fusion = _fusion([(a, 0.6), (b, 0.4)])

    result = fusion.detect(CROP)

    assert result.fusion_status == STATUS_AGREE
    assert result.accepted is True
    assert result.source_detectors == ("model-a", "model-b")
    # Fused point sits between the two, closer to the higher-weighted one.
    assert 100.0 <= result.center[0] <= 102.0


def test_two_wildly_disagreeing_predictions_are_not_averaged():
    """The project brief's own worked example: (420, 220) vs (570, 410) -
    averaging would land on neither. Equal weight/confidence -> no reliable
    winner -> DISAGREEMENT, not an invented midpoint."""
    a = _StubDetector("model-a", _detection(420.0, 220.0, 0.91, True, "model-a"))
    b = _StubDetector("model-b", _detection(570.0, 410.0, 0.88, True, "model-b"))
    fusion = _fusion([(a, 0.5), (b, 0.5)], agreement_threshold=0.4)

    result = fusion.detect(np.zeros((600, 700, 3), dtype=np.uint8))

    assert result.fusion_status == STATUS_DISAGREEMENT
    assert result.accepted is False
    assert result.center is None
    # The midpoint the naive average would have produced is nowhere near
    # either raw candidate.
    midpoint = ((420.0 + 570.0) / 2, (220.0 + 410.0) / 2)
    assert result.box != pytest.approx((midpoint[0] - 5, midpoint[1] - 5, midpoint[0] + 5, midpoint[1] + 5))


def test_a_clearly_dominant_prediction_wins_a_disagreement():
    a = _StubDetector("model-a", _detection(100.0, 100.0, 0.95, True, "model-a"))
    b = _StubDetector("model-b", _detection(300.0, 300.0, 0.95, True, "model-b"))
    # model-a's weight dominates model-b's by well over the default 2.0
    # ratio, so it should win outright rather than reporting DISAGREEMENT.
    fusion = _fusion([(a, 0.9), (b, 0.1)], agreement_threshold=0.1)

    result = fusion.detect(np.zeros((400, 400, 3), dtype=np.uint8))

    assert result.fusion_status == STATUS_DISAGREEMENT_RESOLVED
    assert result.center == pytest.approx((100.0, 100.0))
    assert result.source_detectors == ("model-a",)


def test_a_single_available_candidate_is_used_alone():
    a = _StubDetector("model-a", _detection(50.0, 60.0, 0.9, True, "model-a"))
    b = _StubDetector("model-b", _detection(0.0, 0.0, 0.0, False, "model-b"), supports_result=False)
    fusion = FusionEyeDetector(
        sub_detectors=[a, b],
        config=FusionConfig(model_weights=(ModelWeight("model-a", 1.0), ModelWeight("model-b", 0.0))),
    )

    result = fusion.detect(CROP)

    assert result.fusion_status == STATUS_SINGLE_MODEL
    assert result.center == pytest.approx((50.0, 60.0))
    assert result.source_detectors == ("model-a",)


def test_every_model_at_zero_confidence_yields_low_confidence_not_a_fabricated_point():
    a = _StubDetector("model-a", _detection(50.0, 60.0, 0.0, False, "model-a"))
    b = _StubDetector("model-b", _detection(55.0, 65.0, 0.0, False, "model-b"))
    fusion = _fusion([(a, 0.6), (b, 0.4)])

    result = fusion.detect(CROP)

    assert result.fusion_status == STATUS_LOW_CONFIDENCE
    assert result.accepted is False
    assert result.center is None


def test_weak_rejected_predictions_still_do_not_get_accepted():
    """A rejected prediction is not discarded outright - it can still act as
    a weak corroborating signal if it agrees with another weak one (see the
    module docstring) - but the fused result must still fail
    `min_fused_confidence` and never be reported as accepted."""
    a = _StubDetector("model-a", _detection(50.0, 60.0, 0.2, False, "model-a"))
    b = _StubDetector("model-b", _detection(55.0, 65.0, 0.1, False, "model-b"))
    fusion = _fusion([(a, 0.6), (b, 0.4)])

    result = fusion.detect(CROP)

    assert result.accepted is False


def test_an_empty_crop_is_declined_rather_than_crashing():
    a = _StubDetector("model-a", _detection(1.0, 1.0, 1.0, True, "model-a"))
    fusion = _fusion([(a, 1.0)])

    result = fusion.detect(np.zeros((0, 0, 3), dtype=np.uint8))

    assert result.fusion_status == STATUS_NO_DETECTION
    assert result.accepted is False


def test_supports_is_true_if_any_sub_detector_supports_the_label():
    a = _StubDetector("model-a", _detection(1.0, 1.0, 1.0, True), supports_result=False)
    b = _StubDetector("model-b", _detection(1.0, 1.0, 1.0, True), supports_result=True)
    fusion = FusionEyeDetector(sub_detectors=[a, b], config=FusionConfig())

    assert fusion.supports(16) is True


def test_model_weight_is_configurable_and_changes_the_fused_point():
    """Same two disagreeing-but-not-wildly-so predictions, reweighted -
    the fused point should move toward whichever model now has more
    weight, proving the weight is read from config rather than fixed."""
    a = _StubDetector("model-a", _detection(100.0, 100.0, 0.9, True, "model-a"))
    b = _StubDetector("model-b", _detection(110.0, 100.0, 0.9, True, "model-b"))

    fusion_favoring_a = _fusion([(a, 0.9), (b, 0.1)], agreement_threshold=0.5)
    fusion_favoring_b = _fusion([(a, 0.1), (b, 0.9)], agreement_threshold=0.5)

    result_a = fusion_favoring_a.detect(CROP)
    result_b = fusion_favoring_b.detect(CROP)

    assert result_a.center[0] < result_b.center[0]


def test_min_fused_confidence_gates_acceptance():
    a = _StubDetector("model-a", _detection(100.0, 100.0, 0.55, True, "model-a"))
    fusion = FusionEyeDetector(
        sub_detectors=[a],
        config=FusionConfig(model_weights=(ModelWeight("model-a", 1.0),), min_fused_confidence=0.9),
    )

    result = fusion.detect(CROP)

    assert result.accepted is False  # confidence 0.55 does not clear 0.9
    assert result.fusion_status == STATUS_SINGLE_MODEL
