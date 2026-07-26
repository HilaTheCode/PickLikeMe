"""Capability 14 - comparing two model runs.

Built for regression testing: point it at the previous model's ranking and the
new one's, and it answers the only question that matters at release time - did
this get better, and what specifically broke?

Per-image comparison is the valuable half. An aggregate "F1 +0.02" hides that
the model fixed 40 images and broke 30 different ones; the newly-corrected and
newly-broken lists show exactly which.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .io import load_ranking
from .matching import match_dataset
from .metrics.base import MetricValue
from .model import MatchedImage, Outcome
from . import metrics as metrics_package

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .analysis import AnalysisResult
    from .config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricDelta:
    """One metric, before and after, with the sign interpreted."""

    name: str
    baseline: float | None
    candidate: float | None
    higher_is_better: bool

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline

    @property
    def improved(self) -> bool | None:
        delta = self.delta
        if delta is None or delta == 0:
            return None
        return delta > 0 if self.higher_is_better else delta < 0

    @property
    def relative(self) -> float | None:
        if self.baseline in (None, 0) or self.candidate is None:
            return None
        return (self.candidate - self.baseline) / abs(self.baseline)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "relative": self.relative,
            "improved": self.improved,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(frozen=True)
class ImageChange:
    """One image whose prediction or rank moved between runs."""

    filename: str
    image_path: str
    baseline_outcome: str
    candidate_outcome: str
    baseline_score: float
    candidate_score: float
    baseline_rank: int
    candidate_rank: int

    @property
    def score_delta(self) -> float:
        return self.candidate_score - self.baseline_score

    @property
    def rank_delta(self) -> int:
        # Negative means it moved up the ranking (toward position 1).
        return self.candidate_rank - self.baseline_rank

    @property
    def fixed(self) -> bool:
        return self.baseline_outcome in ("false_positive", "false_negative") and self.candidate_outcome in (
            "true_positive",
            "true_negative",
        )

    @property
    def broken(self) -> bool:
        return self.baseline_outcome in ("true_positive", "true_negative") and self.candidate_outcome in (
            "false_positive",
            "false_negative",
        )

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "image_path": self.image_path,
            "baseline_outcome": self.baseline_outcome,
            "candidate_outcome": self.candidate_outcome,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "score_delta": self.score_delta,
            "baseline_rank": self.baseline_rank,
            "candidate_rank": self.candidate_rank,
            "rank_delta": self.rank_delta,
            "fixed": self.fixed,
            "broken": self.broken,
        }


@dataclass
class ComparisonResult:
    baseline_label: str
    candidate_label: str
    deltas: list[MetricDelta] = field(default_factory=list)
    fixed: list[ImageChange] = field(default_factory=list)
    broken: list[ImageChange] = field(default_factory=list)
    rank_movers: list[ImageChange] = field(default_factory=list)
    common_images: int = 0
    baseline_only: int = 0
    candidate_only: int = 0

    @property
    def improvements(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.improved is True]

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.improved is False]

    @property
    def verdict(self) -> str:
        """A one-line answer for a release decision."""
        if not self.deltas:
            return "no comparable metrics"
        wins, losses = len(self.improvements), len(self.regressions)
        net = len(self.fixed) - len(self.broken)
        if wins > losses and net >= 0:
            return f"IMPROVED - {wins} metrics up, {losses} down, {net:+d} images net corrected"
        if losses > wins and net <= 0:
            return f"REGRESSED - {losses} metrics down, {wins} up, {net:+d} images net corrected"
        return f"MIXED - {wins} metrics up, {losses} down, {net:+d} images net corrected"

    def as_dict(self) -> dict:
        return {
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "verdict": self.verdict,
            "common_images": self.common_images,
            "baseline_only": self.baseline_only,
            "candidate_only": self.candidate_only,
            "deltas": [d.as_dict() for d in self.deltas],
            "improvements": [d.name for d in self.improvements],
            "regressions": [d.name for d in self.regressions],
            "fixed": [c.as_dict() for c in self.fixed],
            "broken": [c.as_dict() for c in self.broken],
            "rank_movers": [c.as_dict() for c in self.rank_movers],
        }

    def render(self) -> str:
        lines = [
            "Model comparison",
            "================",
            f"  {self.baseline_label} -> {self.candidate_label}",
            f"  verdict: {self.verdict}",
            f"  images in both runs: {self.common_images:,}"
            + (f" (+{self.candidate_only:,} new, -{self.baseline_only:,} dropped)" if self.candidate_only or self.baseline_only else ""),
            "",
            f"  {'metric':<32}{self.baseline_label:>12}{self.candidate_label:>12}{'delta':>12}",
        ]
        for delta in self.deltas:
            if delta.delta is None:
                continue
            marker = "+" if delta.improved else ("-" if delta.improved is False else " ")
            base = "n/a" if delta.baseline is None else f"{delta.baseline:.4f}"
            cand = "n/a" if delta.candidate is None else f"{delta.candidate:.4f}"
            lines.append(f"  {marker} {delta.name:<30}{base:>12}{cand:>12}{delta.delta:>+12.4f}")
        lines += [
            "",
            f"  newly corrected: {len(self.fixed):,}",
            f"  newly broken:    {len(self.broken):,}",
        ]
        for change in self.broken[:10]:
            lines.append(
                f"    BROKE {change.filename}: {change.baseline_outcome} -> {change.candidate_outcome} "
                f"(score {change.baseline_score:.3f} -> {change.candidate_score:.3f})"
            )
        return "\n".join(lines)


# Metrics worth comparing. Raw counts are excluded: they move with dataset size
# rather than model quality and would swamp the table.
COMPARED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "ndcg",
    "spearman",
    "ranking_agreement",
    "expected_calibration_error",
    "brier_score",
    "precision_at_top_1pct",
    "precision_at_top_5pct",
    "precision_at_top_10pct",
    "average_rank_displacement",
)


def compare_runs(config: "AnalysisConfig", baseline: "AnalysisResult") -> ComparisonResult:
    """Analyse the comparison ranking and diff it against the baseline result.

    The candidate is matched against the same ground truth at the same
    threshold, so any difference is the model's, not the harness's.
    """
    logger.info("Comparison: loading %s", config.compare_ranking_path)
    candidate_ranking = load_ranking(config.compare_ranking_path)
    candidate_match = match_dataset(
        candidate_ranking.images,
        config.selected_root,
        config.rejected_root,
        threshold=config.threshold,
    )
    candidate_metrics = metrics_package.compute(candidate_match.evaluable)

    deltas: list[MetricDelta] = []
    for name in COMPARED_METRICS:
        base_value: MetricValue | None = baseline.metrics.by_name(name)
        cand_value: MetricValue | None = candidate_metrics.by_name(name)
        if base_value is None and cand_value is None:
            continue
        reference = base_value or cand_value
        deltas.append(
            MetricDelta(
                name=name,
                baseline=base_value.value if base_value else None,
                candidate=cand_value.value if cand_value else None,
                higher_is_better=reference.higher_is_better,
            )
        )

    def index(images: list[MatchedImage]) -> dict[str, MatchedImage]:
        return {image.image_path: image for image in images}

    base_index = index(baseline.evaluable)
    cand_index = index(candidate_match.evaluable)
    shared = set(base_index) & set(cand_index)

    changes: list[ImageChange] = []
    for path in shared:
        before, after = base_index[path], cand_index[path]
        if before.outcome is after.outcome and before.rank == after.rank:
            continue
        changes.append(
            ImageChange(
                filename=before.filename,
                image_path=path,
                baseline_outcome=before.outcome.value,
                candidate_outcome=after.outcome.value,
                baseline_score=before.score,
                candidate_score=after.score,
                baseline_rank=before.rank,
                candidate_rank=after.rank,
            )
        )

    fixed = sorted((c for c in changes if c.fixed), key=lambda c: -abs(c.score_delta))
    broken = sorted((c for c in changes if c.broken), key=lambda c: -abs(c.score_delta))
    movers = sorted(changes, key=lambda c: -abs(c.rank_delta))[: config.max_examples]

    return ComparisonResult(
        baseline_label=config.baseline_label,
        candidate_label=config.compare_label,
        deltas=deltas,
        fixed=fixed[: config.max_examples],
        broken=broken[: config.max_examples],
        rank_movers=movers,
        common_images=len(shared),
        baseline_only=len(set(base_index) - shared),
        candidate_only=len(set(cand_index) - shared),
    )
