"""Capabilities 7, 8 and 9 - where the model is wrong, and which mistakes matter.

A confusion count says *how many* mistakes; this says *which ones*, ordered so
the most informative appear first. That ordering is the point: on a 55k-image
cull nobody reviews every false positive, so the ones surfaced first must be
the ones worth a human's attention.

Three different notions of "worst" are kept separate, because they select
genuinely different images:

- **confident mistakes** - the model was sure and wrong. These usually point at
  a systematic blind spot.
- **rank disagreements** - the image sits far from where it belongs in the
  ordering, regardless of the threshold.
- **borderline** - the model had no opinion. Cheapest to label, highest
  expected information gain for the next training round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .metrics.ranking import by_model_rank, ideal_order
from .model import MatchedImage, Outcome


@dataclass(frozen=True)
class ErrorRecord:
    """One mistake, with everything a report row needs.

    `severity` is the sort key. It is deliberately *not* just the score: a
    false positive at 0.99 and a false negative at 0.01 are equally severe, and
    both are worse than a mistake made at 0.51.
    """

    image: MatchedImage
    severity: float
    rank_displacement: int
    reason: str

    @property
    def filename(self) -> str:
        return self.image.filename

    @property
    def image_path(self) -> str:
        return self.image.image_path

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "image_path": self.image_path,
            "outcome": self.image.outcome.value,
            "score": self.image.score,
            "probability": self.image.probability,
            "confidence": self.image.confidence,
            "rank": self.image.rank,
            "rank_displacement": self.rank_displacement,
            "predicted": self.image.predicted,
            "truth": self.image.truth,
            "severity": self.severity,
            "reason": self.reason,
        }


@dataclass
class ErrorAnalysis:
    """Capability 7/8/9 output: every mistake list a report needs."""

    false_positives: list[ErrorRecord] = field(default_factory=list)
    false_negatives: list[ErrorRecord] = field(default_factory=list)
    confident_false_positives: list[ErrorRecord] = field(default_factory=list)
    confident_false_negatives: list[ErrorRecord] = field(default_factory=list)
    largest_rank_disagreements: list[ErrorRecord] = field(default_factory=list)
    most_surprising: list[ErrorRecord] = field(default_factory=list)
    borderline: list[ErrorRecord] = field(default_factory=list)
    top_ranked: list[MatchedImage] = field(default_factory=list)
    lowest_ranked: list[MatchedImage] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "false_positives": [record.as_dict() for record in self.false_positives],
            "false_negatives": [record.as_dict() for record in self.false_negatives],
            "confident_false_positives": [r.as_dict() for r in self.confident_false_positives],
            "confident_false_negatives": [r.as_dict() for r in self.confident_false_negatives],
            "largest_rank_disagreements": [r.as_dict() for r in self.largest_rank_disagreements],
            "most_surprising": [r.as_dict() for r in self.most_surprising],
            "borderline": [record.as_dict() for record in self.borderline],
        }


def severity_of(image: MatchedImage) -> float:
    """How badly wrong the model was, in [0, 1].

    Distance of the predicted probability from the truth, so a false positive
    at 0.98 scores 0.98 and a false negative at 0.02 also scores 0.98. Falls
    back to the raw score when no probability exists.
    """
    value = image.probability if image.probability is not None else image.score
    target = 1.0 if image.truth == 1 else 0.0
    return abs(value - target)


def _displacements(images: Sequence[MatchedImage]) -> dict[int, int]:
    """Signed distance from each image's model position to its ideal one."""
    model_positions = {id(image): position for position, image in enumerate(by_model_rank(images))}
    ideal_positions = {id(image): position for position, image in enumerate(ideal_order(images))}
    return {id(image): model_positions[id(image)] - ideal_positions[id(image)] for image in images}


def analyse_errors(
    images: Sequence[MatchedImage],
    *,
    borderline_low: float = 0.45,
    borderline_high: float = 0.55,
    limit: int = 60,
) -> ErrorAnalysis:
    """Build every mistake list from one matched dataset."""
    evaluable = [image for image in images if image.outcome is not Outcome.UNKNOWN]
    displacement = _displacements(evaluable) if evaluable else {}

    def record(image: MatchedImage, reason: str) -> ErrorRecord:
        return ErrorRecord(
            image=image,
            severity=severity_of(image),
            rank_displacement=abs(displacement.get(id(image), 0)),
            reason=reason,
        )

    false_positives = [
        record(image, "kept by the model, rejected by you")
        for image in evaluable
        if image.outcome is Outcome.FALSE_POSITIVE
    ]
    false_negatives = [
        record(image, "rejected by the model, kept by you")
        for image in evaluable
        if image.outcome is Outcome.FALSE_NEGATIVE
    ]

    # Highest severity first: the mistakes the model was most certain about.
    false_positives.sort(key=lambda r: r.severity, reverse=True)
    false_negatives.sort(key=lambda r: r.severity, reverse=True)

    errors = false_positives + false_negatives
    by_displacement = sorted(errors, key=lambda r: r.rank_displacement, reverse=True)

    # "Surprising" weights being wrong by how far the image also moved in the
    # ranking - a confident mistake that is *also* badly misplaced is the most
    # informative single image in the set.
    surprising = sorted(
        errors,
        key=lambda r: r.severity * (1.0 + r.rank_displacement / max(1, len(evaluable))),
        reverse=True,
    )

    borderline = [
        record(image, f"probability {image.probability:.3f} inside the uncertainty band")
        for image in evaluable
        if image.probability is not None and borderline_low <= image.probability <= borderline_high
    ]
    # Closest to dead-centre first: maximum uncertainty, maximum information.
    midpoint = (borderline_low + borderline_high) / 2
    borderline.sort(key=lambda r: abs((r.image.probability or 0.0) - midpoint))

    ordered = by_model_rank(evaluable)
    return ErrorAnalysis(
        false_positives=false_positives[:limit],
        false_negatives=false_negatives[:limit],
        confident_false_positives=false_positives[: min(limit, 20)],
        confident_false_negatives=false_negatives[: min(limit, 20)],
        largest_rank_disagreements=by_displacement[:limit],
        most_surprising=surprising[: min(limit, 20)],
        borderline=borderline[:limit],
        top_ranked=ordered[:limit],
        lowest_ranked=ordered[-limit:][::-1] if ordered else [],
    )


@dataclass(frozen=True)
class ScoreDistribution:
    """Capability 6 (data half) - histograms the charts and report share."""

    edges: list[float]
    all_counts: list[int]
    positive_counts: list[int]
    negative_counts: list[int]
    confidence_edges: list[float]
    confidence_counts: list[int]

    def as_dict(self) -> dict:
        return {
            "edges": self.edges,
            "all": self.all_counts,
            "positive": self.positive_counts,
            "negative": self.negative_counts,
            "confidence_edges": self.confidence_edges,
            "confidence_counts": self.confidence_counts,
        }


def score_distribution(images: Sequence[MatchedImage], bins: int = 20) -> ScoreDistribution:
    """Score histograms split by ground truth.

    The overlap between the positive and negative histograms is the single most
    diagnostic picture in the whole report: if they separate cleanly a
    threshold exists that works, and if they overlap heavily no threshold can
    save the model.
    """
    scores = [image.score for image in images]
    low = min(scores, default=0.0)
    high = max(scores, default=1.0)
    if high == low:
        high = low + 1.0
    width = (high - low) / bins
    edges = [low + index * width for index in range(bins + 1)]

    def histogram(values: Sequence[float], lo: float, hi: float, count: int) -> list[int]:
        step = (hi - lo) / count
        buckets = [0] * count
        for value in values:
            index = int((value - lo) / step) if step else 0
            buckets[min(max(index, 0), count - 1)] += 1
        return buckets

    positives = [image.score for image in images if image.truth == 1]
    negatives = [image.score for image in images if image.truth == 0]
    confidences = [image.confidence for image in images if image.confidence is not None]

    confidence_edges = [0.5 + index * 0.5 / bins for index in range(bins + 1)]
    return ScoreDistribution(
        edges=edges,
        all_counts=histogram(scores, low, high, bins),
        positive_counts=histogram(positives, low, high, bins),
        negative_counts=histogram(negatives, low, high, bins),
        confidence_edges=confidence_edges,
        confidence_counts=histogram(confidences, 0.5, 1.0, bins) if confidences else [0] * bins,
    )
