"""Capabilities 4 and 5 - threshold sweep, recommendation, confusion matrix.

The threshold is the one knob a photographer can turn *after* training, and the
default 0.5 is almost never the right one on an imbalanced cull. This sweeps
every threshold and recommends one against a stated objective - "minimise
missed keepers" and "minimise wasted review time" are different objectives and
must produce different answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .config import OPTIMIZATION_TARGETS
from .matching import classify, predict
from .model import MatchedImage, Outcome, RankedImage


@dataclass(frozen=True)
class ThresholdPoint:
    """Every headline metric at one threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float | None
    recall: float | None
    specificity: float | None
    accuracy: float | None
    balanced_accuracy: float | None
    f1: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    mcc: float | None
    youden_j: float | None

    def get(self, target: str) -> float | None:
        return getattr(self, target, None)

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "mcc": self.mcc,
            "youden_j": self.youden_j,
        }


def _divide(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate_threshold(images: Sequence[MatchedImage], threshold: float) -> ThresholdPoint:
    """Re-score the matched dataset at a different threshold.

    Recomputed from each image's probability rather than from its stored
    outcome, since the stored outcome belongs to the *configured* threshold.
    """
    tp = fp = tn = fn = 0
    for image in images:
        value = image.probability if image.probability is not None else image.score
        predicted = 1 if value >= threshold else 0
        if predicted == 1:
            if image.truth == 1:
                tp += 1
            else:
                fp += 1
        else:
            if image.truth == 1:
                fn += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    mcc_denominator = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return ThresholdPoint(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        specificity=specificity,
        accuracy=_divide(tp + tn, total),
        balanced_accuracy=balanced,
        f1=f1,
        false_positive_rate=_divide(fp, fp + tn),
        false_negative_rate=_divide(fn, fn + tp),
        mcc=_divide(tp * tn - fp * fn, mcc_denominator),
        youden_j=(recall + specificity - 1) if recall is not None and specificity is not None else None,
    )


@dataclass
class ThresholdSweep:
    """A full sweep plus the recommendation drawn from it."""

    points: list[ThresholdPoint]
    current: ThresholdPoint
    recommended: ThresholdPoint
    optimize_for: str

    @property
    def improvement(self) -> float | None:
        """How much the objective gains by moving to the recommendation."""
        current = self.current.get(self.optimize_for)
        best = self.recommended.get(self.optimize_for)
        if current is None or best is None:
            return None
        return best - current

    @property
    def is_worth_changing(self) -> bool:
        """Only worth reporting as a recommendation if it actually moves the
        objective - a 0.001 gain is noise, not advice."""
        improvement = self.improvement
        return improvement is not None and improvement > 0.01

    def best_for(self, target: str) -> ThresholdPoint | None:
        candidates = [p for p in self.points if p.get(target) is not None]
        return max(candidates, key=lambda p: p.get(target)) if candidates else None

    def as_dict(self) -> dict:
        return {
            "optimize_for": self.optimize_for,
            "current": self.current.as_dict(),
            "recommended": self.recommended.as_dict(),
            "improvement": self.improvement,
            "is_worth_changing": self.is_worth_changing,
            "points": [point.as_dict() for point in self.points],
            "best_per_target": {
                target: (self.best_for(target).as_dict() if self.best_for(target) else None)
                for target in OPTIMIZATION_TARGETS
            },
        }


def sweep_thresholds(
    images: Sequence[MatchedImage],
    current_threshold: float = 0.5,
    steps: int = 101,
    optimize_for: str = "f1",
) -> ThresholdSweep:
    """Evaluate `steps` thresholds across [0, 1] and pick the best one.

    Ties are broken toward the threshold closest to the current one, so the
    tool never advises a gratuitous move that buys nothing.
    """
    if optimize_for not in OPTIMIZATION_TARGETS:
        raise ValueError(f"Unknown optimisation target {optimize_for!r}")

    points = [evaluate_threshold(images, index / (steps - 1)) for index in range(steps)]
    current = evaluate_threshold(images, current_threshold)

    scored = [(point.get(optimize_for), point) for point in points]
    usable = [(value, point) for value, point in scored if value is not None]
    if usable:
        best_value = max(value for value, _ in usable)
        tied = [point for value, point in usable if value == best_value]
        recommended = min(tied, key=lambda point: abs(point.threshold - current_threshold))
    else:
        recommended = current

    return ThresholdSweep(
        points=points, current=current, recommended=recommended, optimize_for=optimize_for
    )


@dataclass(frozen=True)
class ConfusionMatrix:
    """Capability 5 - counts, percentages, and a renderable table."""

    tp: int
    fp: int
    tn: int
    fn: int
    threshold: float

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def percent(self, count: int) -> float:
        return (count / self.total * 100.0) if self.total else 0.0

    @property
    def cells(self) -> list[list[int]]:
        """Row = truth (kept, rejected), column = prediction (kept, rejected)."""
        return [[self.tp, self.fn], [self.fp, self.tn]]

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "counts": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
            "percentages": {
                "tp": self.percent(self.tp),
                "fp": self.percent(self.fp),
                "tn": self.percent(self.tn),
                "fn": self.percent(self.fn),
            },
            "total": self.total,
        }

    def render(self) -> str:
        """Text table with counts and row-normalised percentages."""
        kept_total = self.tp + self.fn
        rejected_total = self.fp + self.tn

        def cell(count: int, row_total: int) -> str:
            share = f"{count / row_total * 100:5.1f}%" if row_total else "    - "
            return f"{count:>7,} ({share})"

        return "\n".join(
            [
                "Confusion matrix",
                "================",
                f"  threshold: {self.threshold:.3f}   images: {self.total:,}",
                "",
                f"  {'':<18}{'model: KEEP':>17}{'model: REJECT':>17}",
                f"  {'you: KEPT':<18}{cell(self.tp, kept_total):>17}{cell(self.fn, kept_total):>17}",
                f"  {'you: REJECTED':<18}{cell(self.fp, rejected_total):>17}{cell(self.tn, rejected_total):>17}",
                "",
                "  (percentages are row-normalised: share of your kept / rejected images)",
            ]
        )


def confusion_matrix(images: Sequence[MatchedImage], threshold: float) -> ConfusionMatrix:
    point = evaluate_threshold(images, threshold)
    return ConfusionMatrix(tp=point.tp, fp=point.fp, tn=point.tn, fn=point.fn, threshold=threshold)


def rematch_at_threshold(images: Sequence[MatchedImage], threshold: float) -> list[MatchedImage]:
    """Re-derive outcomes at a new threshold, keeping ground truth as matched.

    Used by comparison mode and by "what if I moved the threshold" views; the
    expensive folder join is never repeated.
    """
    rescored: list[MatchedImage] = []
    for image in images:
        predicted = predict(image.ranked, threshold)
        rescored.append(
            MatchedImage(
                ranked=image.ranked,
                truth=image.truth,
                predicted=predicted,
                outcome=classify(image.truth, predicted),
            )
        )
    return rescored
