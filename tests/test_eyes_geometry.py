from __future__ import annotations

import pytest

from picklikeme.eyes.geometry import (
    build_head_frame,
    eye_pair_disagreement,
    normalized_distance,
    point_in_bounds,
    point_to_segment_distance,
)


def test_head_frame_projects_the_axis_point_to_one_zero():
    frame = build_head_frame((0.0, 0.0), (10.0, 0.0))
    assert frame is not None
    assert frame.project(10.0, 0.0) == pytest.approx((1.0, 0.0))
    assert frame.project(0.0, 0.0) == pytest.approx((0.0, 0.0))


def test_head_frame_rotates_with_the_head():
    """A head turned 90 degrees between frames still measures the same
    relative eye position - the whole point of a landmark-axis frame over a
    plain image-plane bounding-box normalisation (see the module
    docstring)."""
    frame_upright = build_head_frame((0.0, 0.0), (0.0, -10.0))  # axis points "up"
    frame_turned = build_head_frame((0.0, 0.0), (10.0, 0.0))  # axis points "right"

    # A point one head-width to the axis's own right, half a head-width along it.
    u1, v1 = frame_upright.project(5.0, -5.0)
    u2, v2 = frame_turned.project(5.0, 5.0)
    assert u1 == pytest.approx(u2, abs=1e-6)
    assert v1 == pytest.approx(v2, abs=1e-6)


def test_build_head_frame_returns_none_for_coincident_landmarks():
    assert build_head_frame((5.0, 5.0), (5.0, 5.0)) is None


def test_build_head_frame_returns_none_for_missing_landmark():
    assert build_head_frame(None, (1.0, 1.0)) is None
    assert build_head_frame((1.0, 1.0), None) is None


def test_point_to_segment_distance_on_segment_is_zero():
    assert point_to_segment_distance(5.0, 0.0, 0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0)


def test_point_to_segment_distance_clamps_past_the_endpoints():
    """Beyond `b`, distance is measured to `b` itself, not extrapolated
    along the infinite line."""
    assert point_to_segment_distance(20.0, 0.0, 0.0, 0.0, 10.0, 0.0) == pytest.approx(10.0)


def test_point_to_segment_distance_handles_a_degenerate_segment():
    assert point_to_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(5.0)


def test_normalized_distance_scales_by_head_size():
    assert normalized_distance((0.0, 0.0), (10.0, 0.0), head_scale=10.0) == pytest.approx(1.0)
    assert normalized_distance((0.0, 0.0), (10.0, 0.0), head_scale=100.0) == pytest.approx(0.1)


def test_normalized_distance_floors_a_degenerate_head_scale():
    # Must not explode into a huge or infinite value for a near-zero scale.
    result = normalized_distance((0.0, 0.0), (1.0, 0.0), head_scale=0.0001)
    assert result < 1.0


def test_point_in_bounds_accepts_a_point_inside():
    assert point_in_bounds((50.0, 50.0), width=100, height=100) is True


def test_point_in_bounds_allows_a_small_overshoot():
    assert point_in_bounds((-2.0, 50.0), width=100, height=100) is True


def test_point_in_bounds_rejects_a_point_far_outside():
    assert point_in_bounds((-500.0, 50.0), width=100, height=100) is False


def test_point_in_bounds_rejects_a_degenerate_crop():
    assert point_in_bounds((5.0, 5.0), width=0, height=100) is False


def test_eye_pair_disagreement_is_none_without_both_channels():
    assert eye_pair_disagreement(None, (1.0, 1.0), head_scale=10.0) is None
    assert eye_pair_disagreement((1.0, 1.0), None, head_scale=10.0) is None


def test_eye_pair_disagreement_measures_normalized_separation():
    assert eye_pair_disagreement((0.0, 0.0), (5.0, 0.0), head_scale=10.0) == pytest.approx(0.5)
