from __future__ import annotations

import pytest

from picklikeme.eyes.burst_consistency import (
    BurstEyeObservation,
    evaluate_burst_consistency,
    head_relative_observation,
)
from picklikeme.eyes.geometry import build_head_frame


def test_a_head_relative_position_that_stays_similar_is_not_an_outlier():
    """The project brief's own worked example: the head moves substantially
    between frames (absolute coordinates would look wildly different), but
    the eye stays at roughly the same position RELATIVE to the head - 42%
    of head width/31% of head height, then 43%/30% - and that must be
    treated as consistent."""
    observations = [
        BurstEyeObservation(image_path="frame1.jpg", burst_id="b1", u=0.42, v=0.31),
        BurstEyeObservation(image_path="frame2.jpg", burst_id="b1", u=0.43, v=0.30),
        BurstEyeObservation(image_path="frame3.jpg", burst_id="b1", u=0.42, v=0.315),
    ]
    results = evaluate_burst_consistency(observations)
    assert results["frame1.jpg"].is_outlier is False
    assert results["frame2.jpg"].is_outlier is False
    assert results["frame3.jpg"].is_outlier is False


def test_a_head_relative_position_that_jumps_is_an_outlier():
    """Same worked example, continued: a third frame's eye suddenly appears
    at 85%/90% of head size - suspicious even though the camera moved and
    absolute pixel coordinates differ too."""
    observations = [
        BurstEyeObservation(image_path="frame1.jpg", burst_id="b1", u=0.42, v=0.31),
        BurstEyeObservation(image_path="frame2.jpg", burst_id="b1", u=0.43, v=0.30),
        BurstEyeObservation(image_path="frame3.jpg", burst_id="b1", u=0.85, v=0.90),
    ]
    results = evaluate_burst_consistency(observations)
    assert results["frame1.jpg"].is_outlier is False
    assert results["frame2.jpg"].is_outlier is False
    assert results["frame3.jpg"].is_outlier is True


def test_a_singleton_burst_cannot_be_flagged_as_an_outlier():
    """No other member to compare against - additional evidence, never
    invented from nothing (see the module docstring)."""
    observations = [BurstEyeObservation(image_path="only.jpg", burst_id="solo", u=0.99, v=0.99)]
    results = evaluate_burst_consistency(observations)
    result = results["only.jpg"]
    assert result.comparison_count == 0
    assert result.deviation is None
    assert result.is_outlier is False


def test_different_bursts_are_never_compared_against_each_other():
    observations = [
        BurstEyeObservation(image_path="a1.jpg", burst_id="burst-a", u=0.4, v=0.3),
        BurstEyeObservation(image_path="a2.jpg", burst_id="burst-a", u=0.41, v=0.31),
        BurstEyeObservation(image_path="b1.jpg", burst_id="burst-b", u=0.9, v=0.9),
    ]
    results = evaluate_burst_consistency(observations)
    # burst-b has no other member of ITS OWN burst - unevaluable, not an
    # outlier just because it differs from burst-a's members.
    assert results["b1.jpg"].comparison_count == 0
    assert results["b1.jpg"].is_outlier is False


def test_head_relative_observation_uses_the_frames_own_head_frame():
    frame = build_head_frame((0.0, 0.0), (10.0, 0.0))
    observation = head_relative_observation("img.jpg", "burst-1", (5.0, 2.0), frame)
    assert observation is not None
    assert observation.u == pytest.approx(0.5)
    assert observation.v == pytest.approx(0.2)


def test_head_relative_observation_is_none_without_a_head_frame():
    assert head_relative_observation("img.jpg", "burst-1", (5.0, 2.0), None) is None
