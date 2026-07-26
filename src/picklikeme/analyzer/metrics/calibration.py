"""Capability 6 (metric half) - is the model's confidence trustworthy?

A model can rank perfectly and still be badly calibrated: if everything it
keeps scores 0.95, "0.95" carries no information about how likely that image is
actually a keeper. Calibration is what makes a probability usable as a
threshold, so it is measured rather than assumed.

The curve data itself lives here too, so the chart and the metric can never
disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..model import MatchedImage
from .base import Metric, safe_divide

CATEGORY = "calibration"
DEFAULT_BINS = 10


@dataclass(frozen=True)
class CalibrationBin:
    """One bucket of the reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_rate: float

    @property
    def gap(self) -> float:
        """Signed miscalibration: positive means overconfident."""
        return self.mean_probability - self.observed_rate


@dataclass(frozen=True)
class CalibrationCurve:
    bins: list[CalibrationBin]
    expected_calibration_error: float | None
    maximum_calibration_error: float | None
    brier_score: float | None

    @property
    def populated(self) -> list[CalibrationBin]:
        return [b for b in self.bins if b.count > 0]


def with_probabilities(images: Sequence[MatchedImage]) -> list[MatchedImage]:
    return [image for image in images if image.probability is not None]


def calibration_curve(images: Sequence[MatchedImage], bins: int = DEFAULT_BINS) -> CalibrationCurve:
    """Reliability diagram data plus the standard calibration error summaries.

    Bins are equal-width over [0, 1] (the usual ECE definition). Empty bins are
    kept in the list so the chart shows the gaps rather than silently
    compressing the x axis.
    """
    usable = with_probabilities(images)
    edges = [i / bins for i in range(bins + 1)]
    result: list[CalibrationBin] = []
    total = len(usable)
    weighted_gap = 0.0
    worst_gap = 0.0

    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        # The final bin is closed so probability == 1.0 is not dropped.
        members = [
            image
            for image in usable
            if (lower <= image.probability < upper) or (index == bins - 1 and image.probability == upper)
        ]
        if members:
            mean_probability = sum(image.probability for image in members) / len(members)
            observed = sum(1 for image in members if image.truth == 1) / len(members)
            weighted_gap += len(members) / total * abs(mean_probability - observed)
            worst_gap = max(worst_gap, abs(mean_probability - observed))
        else:
            mean_probability = (lower + upper) / 2
            observed = 0.0
        result.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(members),
                mean_probability=mean_probability,
                observed_rate=observed,
            )
        )

    brier = (
        sum((image.probability - (1.0 if image.truth == 1 else 0.0)) ** 2 for image in usable) / total
        if total
        else None
    )
    return CalibrationCurve(
        bins=result,
        expected_calibration_error=weighted_gap if total else None,
        maximum_calibration_error=worst_gap if total else None,
        brier_score=brier,
    )


class _CalibrationMetric(Metric):
    category = CATEGORY
    higher_is_better = False

    def applies_to(self, images):
        if not images:
            return False, "no matched images"
        if not with_probabilities(images):
            return False, "ranking file carries no probabilities"
        return True, ""


class ExpectedCalibrationError(_CalibrationMetric):
    name = "expected_calibration_error"
    description = "Average gap between stated confidence and observed accuracy"
    sort_key = 10

    def compute(self, images):
        return calibration_curve(images).expected_calibration_error


class MaximumCalibrationError(_CalibrationMetric):
    name = "maximum_calibration_error"
    description = "Worst single-bin gap between confidence and accuracy"
    sort_key = 11

    def compute(self, images):
        return calibration_curve(images).maximum_calibration_error


class BrierScore(_CalibrationMetric):
    name = "brier_score"
    description = "Mean squared error of the predicted probabilities"
    sort_key = 12

    def compute(self, images):
        return calibration_curve(images).brier_score


class MeanConfidence(_CalibrationMetric):
    name = "mean_confidence"
    description = "Average commitment of the model, in [0.5, 1]"
    higher_is_better = True
    sort_key = 20

    def compute(self, images):
        confidences = [image.confidence for image in images if image.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else None


class ConfidenceAccuracyGap(_CalibrationMetric):
    name = "overconfidence"
    description = "Mean confidence minus accuracy; positive means overconfident"
    sort_key = 21

    def compute(self, images):
        confidences = [image.confidence for image in images if image.confidence is not None]
        if not confidences:
            return None
        correct = sum(1 for image in images if not image.is_error)
        accuracy = safe_divide(correct, len(images))
        if accuracy is None:
            return None
        return sum(confidences) / len(confidences) - accuracy
