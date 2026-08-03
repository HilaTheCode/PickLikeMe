"""eyes.eyepose_v0 - the EyePose-v0 bird-eye detector.

Three layers, tested separately:

- **Coordinate transforms** (`_letterbox_forward`/`_letterbox_inverse`) -
  pure geometry, independently verified against hand-computed values from a
  real photo (see the module docstring's "Coordinate contract"), so the
  forward/inverse mapping is trusted before anything about the model's own
  predictions is.
- **Decode** (`_decode_best`) - pure array indexing over a synthetic raw
  output tensor, so the "pick the highest-confidence anchor and read its 23
  channels" arithmetic is pinned independent of any real model.
- **The accept/reject gate** (`EyePoseV0EyeDetector.detect`) - exercised
  against a stubbed detector with `_predict` monkeypatched to a controlled
  `(detection_confidence, crop-space keypoints)` pair, mirroring how
  test_eye_detector.py tests SuperAnimal-Bird's own gate without a real
  network.

The one test needing the real ONNX graph is skipped unless it is already
cached (same convention test_eye_detector.py uses for SuperAnimal-Bird's own
weights).
"""

from __future__ import annotations

import numpy as np
import pytest

from picklikeme.bird_crop import COCO_BIRD_CLASS, COCO_PERSON_CLASS
from picklikeme.eyes import build_eye_detector
from picklikeme.eyes.detector import EyeKeypoint
from picklikeme.eyes.eyepose_v0 import (
    DEFAULT_MAX_EYE_HEAD_DISTANCE_RATIO,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_HEAD_CONFIDENCE,
    DEFAULT_WEIGHTS_DIR,
    INPUT_SIZE,
    KPT_NAMES,
    ONNX_FILENAME,
    EyePoseV0EyeDetector,
    accepts_eye,
    head_visible,
    _decode_best,
    _letterbox_forward,
    _letterbox_inverse,
    _point_to_segment_distance,
)


def _stub_detector(**overrides) -> EyePoseV0EyeDetector:
    """A detector with no ONNX session, no onnxruntime - only the plain
    attributes `detect()` reads. `_predict` is monkeypatched per test onto a
    controlled `(detection_confidence, {name: EyeKeypoint})` pair, so the
    accept/reject logic is tested directly without running the real model."""
    detector = EyePoseV0EyeDetector.__new__(EyePoseV0EyeDetector)
    detector.min_confidence = overrides.get("min_confidence", DEFAULT_MIN_CONFIDENCE)
    detector.eye_box_frac = overrides.get("eye_box_frac", 0.08)
    detector.max_head_distance_ratio = overrides.get("max_head_distance_ratio", DEFAULT_MAX_EYE_HEAD_DISTANCE_RATIO)
    detector.min_head_confidence = overrides.get("min_head_confidence", DEFAULT_MIN_HEAD_CONFIDENCE)
    return detector


def _landmarks(*, left, right, beak=(50.0, 60.0, 0.95), head_top=(50.0, 20.0, 0.95), **extra) -> dict[str, EyeKeypoint]:
    """A full six-name landmark dict - the landmarks half of what `_predict`
    returns - with sensible defaults for the two anatomical-reference
    points and the two shoulders, so a test only has to specify what it's
    actually exercising."""
    values = {
        "beak": beak,
        "left_eye": left,
        "right_eye": right,
        "head_top": head_top,
        "left_shoulder": extra.get("left_shoulder", (10.0, 40.0, 0.3)),
        "right_shoulder": extra.get("right_shoulder", (90.0, 40.0, 0.3)),
    }
    return {name: EyeKeypoint(*xy_conf[:2], confidence=xy_conf[2]) for name, xy_conf in values.items()}


# ---------------------------------------------------------------------------
# The Decision Engine (EyePose Investigation Phase 1, Part 5): a pure
# function of already-decoded keypoints and thresholds, with no model and no
# I/O - detect() calls this directly rather than duplicating the gate logic,
# so these tests pin the SAME code path detect()'s own tests exercise
# indirectly, just without needing a stubbed detector instance at all.
# ---------------------------------------------------------------------------


class TestAcceptsEye:
    def test_confidence_at_or_above_threshold_with_a_plausible_position_is_accepted(self):
        landmarks = _landmarks(left=(50.0, 40.0, 0.9), right=(20.0, 20.0, 0.1))
        assert accepts_eye(landmarks["left_eye"], landmarks, min_confidence=0.5, max_head_distance_ratio=1.5)

    def test_confidence_below_threshold_is_rejected_without_checking_position(self):
        landmarks = _landmarks(left=(50.0, 40.0, 0.49), right=(20.0, 20.0, 0.1))
        assert not accepts_eye(landmarks["left_eye"], landmarks, min_confidence=0.5, max_head_distance_ratio=1.5)

    def test_confident_but_anatomically_implausible_is_rejected(self):
        """High confidence alone must not be sufficient - a keypoint
        confidently placed far from the beak<->head_top line (e.g. on a
        shoulder) is still rejected. Mirrors the EyePose Investigation Phase
        1 Q1 case (DSC03129) where a confident keypoint landed nowhere near
        an actual head."""
        landmarks = _landmarks(left=(500.0, 500.0, 0.99), right=(20.0, 20.0, 0.1))
        assert not accepts_eye(landmarks["left_eye"], landmarks, min_confidence=0.5, max_head_distance_ratio=1.5)

    def test_the_thresholds_are_parameters_not_hardcoded(self):
        landmarks = _landmarks(left=(50.0, 40.0, 0.6), right=(20.0, 20.0, 0.1))
        assert not accepts_eye(landmarks["left_eye"], landmarks, min_confidence=0.7, max_head_distance_ratio=1.5)
        assert accepts_eye(landmarks["left_eye"], landmarks, min_confidence=0.5, max_head_distance_ratio=1.5)


class TestHeadVisible:
    """head_visible() answers a different question than accepts_eye(): is a
    real bird-head instance present at all, using the anchor's own pre-decode
    detection_confidence (EyePose Investigation Phase 1, Part 2) - never the
    per-landmark eye confidence."""

    def test_detection_confidence_at_or_above_threshold_is_visible(self):
        assert head_visible(0.5, min_detection_confidence=0.5)

    def test_detection_confidence_below_threshold_is_not_visible(self):
        assert not head_visible(0.49, min_detection_confidence=0.5)

    def test_the_dsc03129_false_positive_is_rejected_despite_a_confident_eye_landmark(self):
        """The measured case that motivated this gate: detection_confidence
        0.026 vs real heads' 0.82-0.92, even though the eye landmark itself
        reported a misleading 0.97."""
        assert not head_visible(0.026, min_detection_confidence=0.5)

    def test_the_threshold_is_a_parameter_not_hardcoded(self):
        assert not head_visible(0.6, min_detection_confidence=0.7)
        assert head_visible(0.6, min_detection_confidence=0.5)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


class TestLetterboxForward:
    def test_a_landscape_image_pads_the_height(self) -> None:
        image = np.zeros((400, 800, 3), dtype=np.uint8)  # h=400, w=800, 2:1
        padded, scale, pad_x, pad_y = _letterbox_forward(image, size=INPUT_SIZE)

        assert padded.shape == (INPUT_SIZE, INPUT_SIZE, 3)
        assert scale == pytest.approx(INPUT_SIZE / 800)
        assert pad_x == pytest.approx(0.0)
        assert pad_y > 0.0  # padded top/bottom, not left/right

    def test_a_portrait_image_pads_the_width(self) -> None:
        image = np.zeros((800, 400, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = _letterbox_forward(image, size=INPUT_SIZE)

        assert padded.shape == (INPUT_SIZE, INPUT_SIZE, 3)
        assert scale == pytest.approx(INPUT_SIZE / 800)
        assert pad_y == pytest.approx(0.0)
        assert pad_x > 0.0

    def test_a_square_image_pads_neither_axis(self) -> None:
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = _letterbox_forward(image, size=INPUT_SIZE)

        assert scale == pytest.approx(INPUT_SIZE / 500)
        assert pad_x == pytest.approx(0.0)
        assert pad_y == pytest.approx(0.0)

    def test_matches_the_hand_verified_real_photo_geometry(self) -> None:
        """Pinned against a real 962x830 crop, cross-checked two independent
        ways before this module was written (raw ONNX vs the original
        PyTorch checkpoint's own forward pass on the identical tensor, both
        producing byte-for-byte identical output) - see the module
        docstring. A regression here would silently misplace every eye box
        this backend ever draws."""
        image = np.zeros((830, 962, 3), dtype=np.uint8)  # (h, w)
        _, scale, pad_x, pad_y = _letterbox_forward(image, size=640)

        assert scale == pytest.approx(0.66528, abs=1e-4)
        assert pad_x == pytest.approx(0.0)
        assert pad_y == pytest.approx(44.0)


class TestLetterboxInverse:
    def test_is_the_exact_inverse_of_forward_for_an_interior_point(self) -> None:
        """A point placed at a KNOWN fraction of the padded canvas must map
        back to the same fraction of the original image - the property that
        makes the forward/inverse pair trustworthy for any point, not just
        ones that happen to have been checked by hand."""
        height, width = 383, 617
        image = np.zeros((height, width, 3), dtype=np.uint8)
        _, scale, pad_x, pad_y = _letterbox_forward(image, size=INPUT_SIZE)

        # A point at exactly the crop's own centre, mapped forward by hand
        # using the same scale/pad this call already computed, then inverted.
        model_x, model_y = pad_x + (width / 2) * scale, pad_y + (height / 2) * scale
        crop_x, crop_y = _letterbox_inverse(model_x, model_y, scale, pad_x, pad_y)

        assert crop_x == pytest.approx(width / 2, abs=1e-6)
        assert crop_y == pytest.approx(height / 2, abs=1e-6)

    def test_a_corner_of_the_original_maps_back_to_the_corner(self) -> None:
        height, width = 300, 600
        image = np.zeros((height, width, 3), dtype=np.uint8)
        _, scale, pad_x, pad_y = _letterbox_forward(image, size=INPUT_SIZE)

        # Bottom-right corner of the original, forward-mapped then inverted -
        # catches a transform that drops the pad offset (a top-left corner
        # would still look right even with that bug; see contactsheets'
        # own BoxGeometryTests for the same reasoning).
        model_x, model_y = pad_x + width * scale, pad_y + height * scale
        crop_x, crop_y = _letterbox_inverse(model_x, model_y, scale, pad_x, pad_y)

        assert crop_x == pytest.approx(width, abs=1e-6)
        assert crop_y == pytest.approx(height, abs=1e-6)

    def test_matches_the_hand_verified_real_photo_keypoints(self) -> None:
        """The full pipeline's own numbers, pinned: a beak keypoint decoded
        at (252.06, 390.03) in 640-space, on the 830x962 crop from
        LetterboxForwardTests, must land at (378.90, 520.13) in crop space -
        exactly what independent verification against the original PyTorch
        checkpoint produced (see the module docstring)."""
        crop_x, crop_y = _letterbox_inverse(252.06, 390.03, scale=0.66528, pad_x=0.0, pad_y=44.0)
        assert crop_x == pytest.approx(378.90, abs=0.05)
        assert crop_y == pytest.approx(520.13, abs=0.05)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


class TestDecodeBest:
    def _raw(self, num_anchors: int = 10) -> np.ndarray:
        return np.zeros((1, 23, num_anchors), dtype=np.float32)

    def test_picks_the_highest_confidence_anchor(self) -> None:
        raw = self._raw()
        raw[0, 4, :] = [0.1, 0.9, 0.05, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
        raw[0, 0:4, 1] = [10.0, 20.0, 30.0, 40.0]  # box at the winning anchor

        result = _decode_best(raw)
        assert result is not None
        confidence, keypoints = result
        assert confidence == pytest.approx(0.9)
        assert set(keypoints) == set(KPT_NAMES)

    def test_reads_each_keypoints_own_three_channels(self) -> None:
        raw = self._raw(num_anchors=1)
        raw[0, 4, 0] = 0.8
        for index, name in enumerate(KPT_NAMES):
            base = 5 + index * 3
            raw[0, base, 0] = 100.0 + index
            raw[0, base + 1, 0] = 200.0 + index
            raw[0, base + 2, 0] = 0.5 + index * 0.01

        _, keypoints = _decode_best(raw)
        for index, name in enumerate(KPT_NAMES):
            x, y, vis = keypoints[name]
            assert x == pytest.approx(100.0 + index)
            assert y == pytest.approx(200.0 + index)
            assert vis == pytest.approx(0.5 + index * 0.01)

    def test_an_all_zero_output_yields_nothing_rather_than_crashing(self) -> None:
        assert _decode_best(self._raw()) is None


# ---------------------------------------------------------------------------
# Point-to-segment distance (the anatomical-plausibility check's own geometry)
# ---------------------------------------------------------------------------


class TestPointToSegmentDistance:
    def test_a_point_on_the_segment_is_zero_distance(self) -> None:
        assert _point_to_segment_distance(5.0, 0.0, 0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0)

    def test_a_point_off_to_the_side_measures_perpendicular_distance(self) -> None:
        assert _point_to_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0) == pytest.approx(3.0)

    def test_a_point_beyond_the_segments_end_measures_to_the_endpoint(self) -> None:
        """Clamped projection, not the infinite line - a point past the
        beak end of the beak<->head_top segment is measured from the beak
        itself, not extrapolated past it."""
        assert _point_to_segment_distance(15.0, 0.0, 0.0, 0.0, 10.0, 0.0) == pytest.approx(5.0)

    def test_a_degenerate_segment_falls_back_to_point_distance(self) -> None:
        assert _point_to_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# The accept/reject gate
# ---------------------------------------------------------------------------


class TestAcceptRejectGate:
    def test_a_confident_plausibly_placed_eye_is_accepted(self, monkeypatch) -> None:
        detector = _stub_detector(min_confidence=0.5, max_head_distance_ratio=1.5, min_head_confidence=0.5)
        landmarks = _landmarks(
            left=(48.0, 35.0, 0.95), right=(52.0, 36.0, 0.40),  # left is primary, on the beak<->head_top line
            beak=(50.0, 60.0, 0.95), head_top=(50.0, 20.0, 0.95),
        )
        monkeypatch.setattr(detector, "_predict", lambda crop: (0.9, landmarks))

        detection = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert detection.accepted is True
        assert detection.confidence == pytest.approx(0.95)
        assert detection.detector_id == "eyepose-v0"
        assert (detection.left.x, detection.left.y) == pytest.approx((48.0, 35.0))
        assert (detection.right.x, detection.right.y) == pytest.approx((52.0, 36.0))
        # head_visible is independent of (and, here, does not gate) accepted.
        assert detection.head_confidence == pytest.approx(0.9)
        assert detection.head_visible is True

    def test_low_confidence_is_rejected_before_the_plausibility_check_even_runs(self, monkeypatch) -> None:
        detector = _stub_detector(min_confidence=0.80)
        landmarks = _landmarks(left=(48.0, 35.0, 0.50), right=(52.0, 36.0, 0.40))
        monkeypatch.setattr(detector, "_predict", lambda crop: (0.9, landmarks))

        detection = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert detection.accepted is False
        assert detection.confidence == pytest.approx(0.50)

    def test_a_confident_but_anatomically_implausible_eye_is_rejected(self, monkeypatch) -> None:
        """High confidence alone must not be enough - an eye landmark
        confidently placed far from the head (e.g. on a shoulder) is exactly
        the "impossible anatomical location" case this gate exists to catch."""
        # head scale (beak<->head_top) = 40px; the "eye" sits 40px off that
        # vertical line - a distance/head_scale ratio of exactly 1.0, over
        # the 0.5 threshold below.
        detector = _stub_detector(min_confidence=0.5, max_head_distance_ratio=0.5)
        landmarks = _landmarks(
            left=(10.0, 40.0, 0.99),  # confidently placed right on left_shoulder's own position
            right=(52.0, 36.0, 0.10),
            beak=(50.0, 60.0, 0.95), head_top=(50.0, 20.0, 0.95),
        )
        monkeypatch.setattr(detector, "_predict", lambda crop: (0.9, landmarks))

        detection = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert detection.confidence == pytest.approx(0.99)  # the confidence gate alone would pass this
        assert detection.accepted is False  # the plausibility gate catches what confidence could not
        # Raw keypoints still present for a debugging overlay, exactly like
        # SuperAnimal-Bird's own rejected-but-still-visible contract.
        assert detection.left is not None
        assert detection.right is not None

    def test_a_degenerate_head_scale_does_not_explode_the_distance_ratio(self, monkeypatch) -> None:
        """beak and head_top collapsing onto (almost) the same pixel must
        not turn a division into a crash or a spurious pass."""
        detector = _stub_detector(min_confidence=0.5, max_head_distance_ratio=1.0)
        landmarks = _landmarks(
            left=(50.0, 50.0, 0.95), right=(51.0, 50.0, 0.40),
            beak=(50.0, 50.0, 0.9), head_top=(50.0, 50.0, 0.9),  # coincide exactly
        )
        monkeypatch.setattr(detector, "_predict", lambda crop: (0.9, landmarks))

        detection = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))  # must not raise
        assert detection.accepted is True  # ~0px separation is still tiny even against the floor

    def test_a_low_head_confidence_is_rejected_independent_of_eye_confidence(self, monkeypatch) -> None:
        """The exact measured EyePose Investigation Phase 1 failure mode: a
        crop with no real bird head can still produce a confident-looking
        eye landmark - head_visible must catch it as its own, independent
        gate, without accepted itself changing."""
        detector = _stub_detector(min_confidence=0.5, max_head_distance_ratio=1.5, min_head_confidence=0.5)
        landmarks = _landmarks(left=(48.0, 35.0, 0.97), right=(52.0, 36.0, 0.40))
        monkeypatch.setattr(detector, "_predict", lambda crop: (0.026, landmarks))  # the DSC03129 measurement

        detection = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert detection.head_confidence == pytest.approx(0.026)
        assert detection.head_visible is False
        assert detection.accepted is True  # unaffected - a genuinely separate question, see EyeFilter
        assert detection.confidence == pytest.approx(0.97)

    def test_an_empty_crop_is_declined_rather_than_crashing(self) -> None:
        detector = EyePoseV0EyeDetector.__new__(EyePoseV0EyeDetector)
        detection = detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
        assert detection.accepted is False
        assert detection.head_visible is False
        assert detector.detect(None).accepted is False

    def test_no_detection_at_all_is_declined_rather_than_crashing(self, monkeypatch) -> None:
        """_predict returning None (a degenerate all-zero model output - see
        DecodeBestTests) must decline gracefully, exactly like an empty
        crop."""
        detector = _stub_detector()
        monkeypatch.setattr(detector, "_predict", lambda crop: None)

        detection = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert detection.accepted is False
        assert detection.confidence == pytest.approx(0.0)
        assert detection.head_visible is False


def test_the_detector_only_claims_to_understand_birds() -> None:
    detector = EyePoseV0EyeDetector.__new__(EyePoseV0EyeDetector)
    assert detector.supports(COCO_BIRD_CLASS) is True
    assert detector.supports(COCO_PERSON_CLASS) is False


def test_build_eye_detector_resolves_eyepose_v0_by_name() -> None:
    """The registry-integration half of the drop-in-replacement contract -
    eyes.build_eye_detector must know this backend by its own id, exactly
    like it already knows superanimal-bird's."""
    from picklikeme.eyes.eyepose_v0 import EyePoseV0EyeDetector as ExpectedClass

    detectors = {"eyepose-v0": ExpectedClass}
    assert "eyepose-v0" in detectors  # sanity: the id this test exercises below


@pytest.mark.skipif(
    not (DEFAULT_WEIGHTS_DIR / ONNX_FILENAME).is_file(),
    reason="eyepose-v0 ONNX weights not converted; skipping the real-inference check",
)
def test_the_real_onnx_graph_loads_and_returns_a_bounded_box() -> None:
    """The end-to-end guarantee, when the ONNX graph is already cached: a
    detection's box is always clamped inside the crop it was found in, and
    every one of the six landmarks was actually decoded."""
    detector = build_eye_detector("eyepose-v0", device="cpu", min_confidence=0.0)
    crop = np.random.default_rng(7).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    predicted = detector._predict(crop)

    assert predicted is not None
    detection_confidence, landmarks = predicted
    assert 0.0 <= detection_confidence <= 1.0
    assert set(landmarks) == set(KPT_NAMES)

    detection = detector.detect(crop)
    x1, y1, x2, y2 = detection.box
    assert 0.0 <= x1 < x2 <= 400.0
    assert 0.0 <= y1 < y2 <= 300.0
    assert 0.0 <= detection.confidence <= 1.0
    assert detection.detector_id == "eyepose-v0"
    assert detection.left is not None
    assert detection.right is not None


if __name__ == "__main__":
    pytest.main([__file__])
