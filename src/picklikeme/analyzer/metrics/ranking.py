"""Capability 3 - ranking quality.

For culling, the ordering matters more than the threshold: what a photographer
actually does is review the top slice of a shoot. These metrics ask how much of
that slice is worth reviewing, and how closely the model's ordering tracks the
photographer's own.

The model's ordering is compared against an *ideal* ordering (every keeper
above every reject). Ground truth is binary, so ties are pervasive and every
correlation here is computed tie-aware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..model import MatchedImage
from .base import Metric, safe_divide

CATEGORY = "ranking"

# Percentile cut-offs exposed as individual metrics. Kept in one place so the
# metric classes below are generated from it rather than copy-pasted.
TOP_PERCENTS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)


def by_model_rank(images: Sequence[MatchedImage]) -> list[MatchedImage]:
    """Images in the model's order: best score first, ties broken by rank then
    filename so the ordering is fully deterministic across runs."""
    return sorted(images, key=lambda image: (-image.score, image.rank, image.filename))


def top_k_count(total: int, percent: float) -> int:
    """How many images make up the top `percent` of `total`.

    Always at least one: "the top 1% of 40 images" is a real question and the
    honest answer is the single best image, not an empty slice.
    """
    return max(1, math.ceil(total * percent / 100.0))


def precision_at_k(images: Sequence[MatchedImage], k: int) -> float | None:
    ordered = by_model_rank(images)[:k]
    if not ordered:
        return None
    return sum(1 for image in ordered if image.truth == 1) / len(ordered)


def recall_at_k(images: Sequence[MatchedImage], k: int) -> float | None:
    total_positive = sum(1 for image in images if image.truth == 1)
    if total_positive == 0:
        return None
    ordered = by_model_rank(images)[:k]
    return sum(1 for image in ordered if image.truth == 1) / total_positive


def dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(position + 2) for position, rel in enumerate(relevances))


def ndcg(images: Sequence[MatchedImage], k: int | None = None) -> float | None:
    """Normalised discounted cumulative gain.

    Binary relevance (kept = 1), so the ideal ordering is simply every keeper
    first; nDCG is then the model's DCG over that ideal.
    """
    ordered = by_model_rank(images)
    if k is not None:
        ordered = ordered[:k]
    relevances = [1 if image.truth == 1 else 0 for image in ordered]
    ideal = sorted([1 if image.truth == 1 else 0 for image in images], reverse=True)[: len(relevances)]
    ideal_dcg = dcg(ideal)
    return None if ideal_dcg == 0 else dcg(relevances) / ideal_dcg


def _ranks_with_ties(values: Sequence[float]) -> list[float]:
    """Average ranks, so tied values share a rank. Required for both
    correlations: binary truth means huge tie groups, and naive ranking would
    invent an ordering inside them."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def spearman(model_scores: Sequence[float], truth: Sequence[float]) -> float | None:
    """Pearson correlation of the tie-corrected ranks."""
    if len(model_scores) < 2:
        return None
    x = _ranks_with_ties(model_scores)
    y = _ranks_with_ties(truth)
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    var_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    return safe_divide(cov, var_x * var_y)


def kendall_tau_b(model_scores: Sequence[float], truth: Sequence[float]) -> float | None:
    """Kendall's tau-b - the tie-adjusted variant.

    O(n^2). Ranking analyses run on a culled shoot rather than the full
    archive, and correctness matters more here than speed; the caller
    subsamples if it ever needs to.
    """
    n = len(model_scores)
    if n < 2:
        return None
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = model_scores[i] - model_scores[j]
            dy = truth[i] - truth[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if dx == 0:
                    tied_x += 1
                if dy == 0:
                    tied_y += 1
    total_pairs = n * (n - 1) / 2
    denominator = math.sqrt((total_pairs - tied_x) * (total_pairs - tied_y))
    return safe_divide(concordant - discordant, denominator)


@dataclass(frozen=True)
class RankDisplacement:
    """How far images move between the model's order and the ideal order."""

    average: float | None
    median: float | None
    maximum: int | None
    worst_image: MatchedImage | None


def ideal_order(images: Sequence[MatchedImage]) -> list[MatchedImage]:
    """The photographer's ordering: keepers first, then rejects. Within a class
    the model's own order is kept, so displacement measures only the errors the
    model makes about the *class*, not arbitrary intra-class churn."""
    ordered = by_model_rank(images)
    keepers = [image for image in ordered if image.truth == 1]
    rejects = [image for image in ordered if image.truth != 1]
    return keepers + rejects


def rank_displacement(images: Sequence[MatchedImage]) -> RankDisplacement:
    if not images:
        return RankDisplacement(None, None, None, None)
    model_positions = {id(image): position for position, image in enumerate(by_model_rank(images))}
    ideal_positions = {id(image): position for position, image in enumerate(ideal_order(images))}

    displacements = [(abs(model_positions[id(i)] - ideal_positions[id(i)]), i) for i in images]
    magnitudes = sorted(d for d, _ in displacements)
    middle = len(magnitudes) // 2
    median = (
        magnitudes[middle]
        if len(magnitudes) % 2
        else (magnitudes[middle - 1] + magnitudes[middle]) / 2
    )
    worst = max(displacements, key=lambda pair: pair[0])
    return RankDisplacement(
        average=sum(magnitudes) / len(magnitudes),
        median=float(median),
        maximum=worst[0],
        worst_image=worst[1],
    )


def rank_overlap(images: Sequence[MatchedImage], k: int) -> float | None:
    """Share of the model's top-k that is also in the ideal top-k."""
    if k <= 0 or not images:
        return None
    model_top = {id(image) for image in by_model_rank(images)[:k]}
    ideal_top = {id(image) for image in ideal_order(images)[:k]}
    return len(model_top & ideal_top) / k


# ---------------------------------------------------------------------------
# Metric plugins
# ---------------------------------------------------------------------------

class _RankingMetric(Metric):
    category = CATEGORY

    def applies_to(self, images):
        if not images:
            return False, "no matched images"
        if not any(image.truth == 1 for image in images):
            return False, "needs at least one selected image"
        return True, ""


def _make_top_percent_metric(percent: float) -> type[_RankingMetric]:
    """Build the Precision@top-N% metric class for one cut-off."""

    class TopPercentPrecision(_RankingMetric):
        name = f"precision_at_top_{percent:g}pct"
        description = f"Share of the model's top {percent:g}% that you kept"
        sort_key = int(percent * 10)

        def compute(self, images):
            return precision_at_k(images, top_k_count(len(images), percent))

    TopPercentPrecision.__name__ = f"PrecisionAtTop{str(percent).replace('.', '_')}Pct"
    return TopPercentPrecision


def _make_top_percent_recall(percent: float) -> type[_RankingMetric]:
    class TopPercentRecall(_RankingMetric):
        name = f"recall_at_top_{percent:g}pct"
        description = f"Share of your keepers found in the model's top {percent:g}%"
        sort_key = 300 + int(percent * 10)

        def compute(self, images):
            return recall_at_k(images, top_k_count(len(images), percent))

    TopPercentRecall.__name__ = f"RecallAtTop{str(percent).replace('.', '_')}Pct"
    return TopPercentRecall


# Instantiating these classes registers them; one entry per cut-off, no
# hand-written duplication.
TOP_PERCENT_METRICS = [_make_top_percent_metric(p) for p in TOP_PERCENTS]
TOP_PERCENT_RECALL_METRICS = [_make_top_percent_recall(p) for p in TOP_PERCENTS]


class AveragePrecisionMetric(_RankingMetric):
    name = "average_precision"
    description = "Precision averaged over every position that holds a keeper"
    sort_key = 600

    def compute(self, images):
        from .classification import average_precision

        return average_precision(images)


class MeanAveragePrecision(_RankingMetric):
    name = "mean_average_precision"
    description = "Average precision averaged over shoots (folders)"
    sort_key = 601

    def compute(self, images):
        from pathlib import Path

        from .classification import average_precision

        # One "query" per source folder: a shoot is the natural retrieval unit,
        # and averaging over shoots stops one huge folder dominating.
        groups: dict[str, list[MatchedImage]] = {}
        for image in images:
            groups.setdefault(str(Path(image.image_path).parent), []).append(image)
        scores = [ap for ap in (average_precision(group) for group in groups.values()) if ap is not None]
        return sum(scores) / len(scores) if scores else None


class NdcgMetric(_RankingMetric):
    name = "ndcg"
    description = "Normalised discounted cumulative gain over the full ranking"
    sort_key = 610

    def compute(self, images):
        return ndcg(images)


class NdcgAtTop10(_RankingMetric):
    name = "ndcg_at_top_10pct"
    description = "nDCG restricted to the model's top 10%"
    sort_key = 611

    def compute(self, images):
        return ndcg(images, top_k_count(len(images), 10.0))


class SpearmanCorrelation(_RankingMetric):
    name = "spearman"
    description = "Rank correlation between model score and your decision"
    sort_key = 620

    def compute(self, images):
        return spearman([i.score for i in images], [float(i.truth) for i in images])


class KendallTau(_RankingMetric):
    name = "kendall_tau"
    description = "Kendall tau-b between model score and your decision"
    sort_key = 621
    # O(n^2): above this many images the pair loop costs more than the metric
    # is worth, and a uniform subsample answers the same question.
    max_exact = 4000

    def compute(self, images):
        sample = list(images)
        if len(sample) > self.max_exact:
            step = len(sample) / self.max_exact
            sample = [sample[int(i * step)] for i in range(self.max_exact)]
        return kendall_tau_b([i.score for i in sample], [float(i.truth) for i in sample])


class RankOverlapAtTop10(_RankingMetric):
    name = "rank_overlap_at_top_10pct"
    description = "Overlap between the model's top 10% and the ideal top 10%"
    sort_key = 630

    def compute(self, images):
        return rank_overlap(images, top_k_count(len(images), 10.0))


class AverageRankDisplacement(_RankingMetric):
    name = "average_rank_displacement"
    description = "Mean positions an image sits from where it should"
    higher_is_better = False
    fmt = "{:,.1f}"
    sort_key = 640

    def compute(self, images):
        return rank_displacement(images).average


class MedianRankDisplacement(_RankingMetric):
    name = "median_rank_displacement"
    description = "Median positions an image sits from where it should"
    higher_is_better = False
    fmt = "{:,.1f}"
    sort_key = 641

    def compute(self, images):
        return rank_displacement(images).median


class MaxRankDisagreement(_RankingMetric):
    name = "max_rank_disagreement"
    description = "Largest gap between an image's model rank and its ideal rank"
    higher_is_better = False
    fmt = "{:,.0f}"
    sort_key = 642

    def compute(self, images):
        maximum = rank_displacement(images).maximum
        return None if maximum is None else float(maximum)


class RankingAgreement(_RankingMetric):
    name = "ranking_agreement"
    description = "Share of keeper/reject pairs the model orders correctly"
    sort_key = 650

    def compute(self, images):
        # Equivalent to ROC AUC, but stated in ranking terms because that is
        # how it is read here: "how often does the model put a keeper above a
        # reject". Computed from the shared tie-aware implementation.
        from ...evaluate import roc_auc

        return roc_auc([int(i.truth) for i in images], [i.score for i in images])
