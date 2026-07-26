"""Capability 2 - classification metrics at the configured threshold.

Every metric here is derived from the same confusion counts, so they cannot
disagree with one another or with the rendered confusion matrix. ROC AUC reuses
the pipeline's existing tie-aware implementation rather than adding a second
one.
"""

from __future__ import annotations

import math
from typing import Sequence

from ...evaluate import roc_auc as pipeline_roc_auc
from ..model import MatchedImage
from .base import Metric, both_classes_present, counts_of, labels_and_scores, safe_divide

CATEGORY = "classification"


class _CountMetric(Metric):
    """Base for the raw confusion counts (rendered as integers)."""

    category = CATEGORY
    fmt = "{:,.0f}"
    higher_is_better = True


class TruePositives(_CountMetric):
    name = "true_positives"
    description = "Kept by the model and kept by you"
    sort_key = 1

    def compute(self, images: Sequence[MatchedImage]) -> float:
        return float(counts_of(images).tp)


class FalsePositives(_CountMetric):
    name = "false_positives"
    description = "Kept by the model but rejected by you"
    higher_is_better = False
    sort_key = 2

    def compute(self, images: Sequence[MatchedImage]) -> float:
        return float(counts_of(images).fp)


class TrueNegatives(_CountMetric):
    name = "true_negatives"
    description = "Rejected by the model and rejected by you"
    sort_key = 3

    def compute(self, images: Sequence[MatchedImage]) -> float:
        return float(counts_of(images).tn)


class FalseNegatives(_CountMetric):
    name = "false_negatives"
    description = "Rejected by the model but kept by you"
    higher_is_better = False
    sort_key = 4

    def compute(self, images: Sequence[MatchedImage]) -> float:
        return float(counts_of(images).fn)


class Accuracy(Metric):
    name = "accuracy"
    description = "Fraction of all images classified correctly"
    category = CATEGORY
    sort_key = 10

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        return safe_divide(counts.tp + counts.tn, counts.total)


class Precision(Metric):
    name = "precision"
    description = "Of the images the model kept, how many you also kept"
    category = CATEGORY
    sort_key = 11

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        return safe_divide(counts.tp, counts.predicted_positive)


class Recall(Metric):
    name = "recall"
    description = "Of the images you kept, how many the model also kept"
    category = CATEGORY
    sort_key = 12

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        return safe_divide(counts.tp, counts.actual_positive)


class Specificity(Metric):
    name = "specificity"
    description = "Of the images you rejected, how many the model also rejected"
    category = CATEGORY
    sort_key = 13

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        return safe_divide(counts.tn, counts.actual_negative)


class BalancedAccuracy(Metric):
    name = "balanced_accuracy"
    description = "Mean of recall and specificity - unaffected by class imbalance"
    category = CATEGORY
    sort_key = 14

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        recall = safe_divide(counts.tp, counts.actual_positive)
        specificity = safe_divide(counts.tn, counts.actual_negative)
        if recall is None or specificity is None:
            return None
        return (recall + specificity) / 2


class F1Score(Metric):
    name = "f1"
    description = "Harmonic mean of precision and recall"
    category = CATEGORY
    sort_key = 15

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        precision = safe_divide(counts.tp, counts.predicted_positive)
        recall = safe_divide(counts.tp, counts.actual_positive)
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)


class FalsePositiveRate(Metric):
    name = "false_positive_rate"
    description = "Share of your rejects the model wrongly kept"
    category = CATEGORY
    higher_is_better = False
    sort_key = 16

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        return safe_divide(counts.fp, counts.actual_negative)


class FalseNegativeRate(Metric):
    name = "false_negative_rate"
    description = "Share of your keepers the model wrongly rejected"
    category = CATEGORY
    higher_is_better = False
    sort_key = 17

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        return safe_divide(counts.fn, counts.actual_positive)


class NegativePredictiveValue(Metric):
    name = "negative_predictive_value"
    description = "Of the images the model rejected, how many you also rejected"
    category = CATEGORY
    sort_key = 18

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        return safe_divide(counts.tn, counts.predicted_negative)


class MatthewsCorrelation(Metric):
    name = "mcc"
    description = "Matthews correlation coefficient in [-1, 1]; 0 is chance"
    category = CATEGORY
    sort_key = 19

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        numerator = counts.tp * counts.tn - counts.fp * counts.fn
        denominator = math.sqrt(
            float(counts.predicted_positive)
            * float(counts.actual_positive)
            * float(counts.actual_negative)
            * float(counts.predicted_negative)
        )
        return safe_divide(numerator, denominator)


class YoudenJ(Metric):
    name = "youden_j"
    description = "Recall + specificity - 1; 0 is chance, 1 is perfect"
    category = CATEGORY
    sort_key = 20

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        recall = safe_divide(counts.tp, counts.actual_positive)
        specificity = safe_divide(counts.tn, counts.actual_negative)
        if recall is None or specificity is None:
            return None
        return recall + specificity - 1


class RocAuc(Metric):
    name = "roc_auc"
    description = "Probability a random keeper outranks a random reject"
    category = CATEGORY
    sort_key = 30

    def applies_to(self, images: Sequence[MatchedImage]) -> tuple[bool, str]:
        if not images:
            return False, "no matched images"
        if not both_classes_present(images):
            return False, "needs both selected and rejected images"
        return True, ""

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        labels, scores = labels_and_scores(images)
        return pipeline_roc_auc(labels, scores)


class PrAuc(Metric):
    name = "pr_auc"
    description = "Area under the precision-recall curve (average precision)"
    category = CATEGORY
    sort_key = 31

    def applies_to(self, images: Sequence[MatchedImage]) -> tuple[bool, str]:
        if not images:
            return False, "no matched images"
        if not any(image.truth == 1 for image in images):
            return False, "needs at least one selected image"
        return True, ""

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        return average_precision(images)


def average_precision(images: Sequence[MatchedImage]) -> float | None:
    """Average precision: precision at each positive, averaged.

    This is the step-wise definition (the one `sklearn.average_precision_score`
    uses), not trapezoidal interpolation of the PR curve, which is optimistic.
    Ties are broken by grouping equal scores so ordering within a tie cannot
    change the result.
    """
    ordered = sorted(images, key=lambda image: image.score, reverse=True)
    total_positive = sum(1 for image in ordered if image.truth == 1)
    if total_positive == 0:
        return None

    running_tp = 0
    seen = 0
    score_sum = 0.0
    index = 0
    while index < len(ordered):
        # Consume all images sharing this score together.
        group_end = index
        while group_end + 1 < len(ordered) and ordered[group_end + 1].score == ordered[index].score:
            group_end += 1
        group = ordered[index : group_end + 1]
        positives_here = sum(1 for image in group if image.truth == 1)
        running_tp += positives_here
        seen += len(group)
        if positives_here:
            score_sum += positives_here * (running_tp / seen)
        index = group_end + 1

    return score_sum / total_positive
