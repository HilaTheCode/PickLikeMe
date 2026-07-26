"""The orchestrator: config in, AnalysisResult out.

Everything the reports need is computed here, once, and handed over as data.
Renderers (text, JSON, HTML, charts, contact sheets) consume AnalysisResult and
never recompute anything - so the number in the HTML summary card is by
construction the same number in the JSON and the text report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import format_duration
from . import metrics as metrics_package
from .config import AnalysisConfig
from .errors import ErrorAnalysis, ScoreDistribution, analyse_errors, score_distribution
from .io import RankingFile, load_ranking
from .matching import MatchResult, match_dataset
from .metrics.base import MetricSet
from .metrics.calibration import CalibrationCurve, calibration_curve
from .model import Outcome
from .thresholds import ConfusionMatrix, ThresholdSweep, confusion_matrix, sweep_thresholds

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Everything one analysis produced. Renderer-agnostic."""

    config: AnalysisConfig
    ranking: RankingFile
    match: MatchResult
    metrics: MetricSet
    confusion: ConfusionMatrix
    sweep: ThresholdSweep
    calibration: CalibrationCurve
    distribution: ScoreDistribution
    errors: ErrorAnalysis
    suggestions: list = field(default_factory=list)
    comparison: object | None = None
    elapsed_seconds: float = 0.0
    generated_at: str = ""

    @property
    def evaluable(self):
        return self.match.evaluable

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.match.evaluable)

    def headline(self) -> dict[str, float | None]:
        """The handful of numbers that go on summary cards."""
        return {
            "accuracy": self.metrics.get("accuracy"),
            "precision": self.metrics.get("precision"),
            "recall": self.metrics.get("recall"),
            "f1": self.metrics.get("f1"),
            "roc_auc": self.metrics.get("roc_auc"),
            "pr_auc": self.metrics.get("pr_auc"),
            "balanced_accuracy": self.metrics.get("balanced_accuracy"),
            "mcc": self.metrics.get("mcc"),
        }

    def as_dict(self) -> dict:
        """Full JSON form - the machine-readable record of the run."""
        return {
            "generated_at": self.generated_at,
            "elapsed_seconds": self.elapsed_seconds,
            "config": self.config.to_dict(),
            "ranking": {
                "path": str(self.ranking.path),
                "chunks": [str(p) for p in self.ranking.chunk_paths],
                "images": len(self.ranking.images),
                "detected_columns": self.ranking.detected_columns,
                "preamble": self.ranking.preamble,
                "has_probabilities": self.ranking.has_probabilities,
            },
            "matching": {
                "counts": self.match.counts,
                "num_selected": self.match.num_selected,
                "num_rejected": self.match.num_rejected,
                "unmatched": len(self.match.unmatched),
                "unranked_selected": len(self.match.unranked_selected),
                "unranked_rejected": len(self.match.unranked_rejected),
                "strategies": dict(self.match.strategy_counts),
                "warnings": self.match.warnings,
            },
            "metrics": self.metrics.as_dict(),
            "confusion_matrix": self.confusion.as_dict(),
            "thresholds": self.sweep.as_dict(),
            "calibration": {
                "expected_calibration_error": self.calibration.expected_calibration_error,
                "maximum_calibration_error": self.calibration.maximum_calibration_error,
                "brier_score": self.calibration.brier_score,
                "bins": [
                    {
                        "lower": b.lower,
                        "upper": b.upper,
                        "count": b.count,
                        "mean_probability": b.mean_probability,
                        "observed_rate": b.observed_rate,
                        "gap": b.gap,
                    }
                    for b in self.calibration.bins
                ],
            },
            "distribution": self.distribution.as_dict(),
            "errors": self.errors.as_dict(),
            "suggestions": [s.as_dict() for s in self.suggestions],
            "comparison": self.comparison.as_dict() if self.comparison is not None else None,
        }


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Load, match, measure. Pure computation - writes nothing."""
    started = time.perf_counter()
    logger.info("Loading ranking from %s", config.ranking_path)
    ranking = load_ranking(config.ranking_path)
    for warning in ranking.warnings[:10]:
        logger.warning("%s", warning)

    match = match_dataset(
        ranking.images,
        config.selected_root,
        config.rejected_root,
        threshold=config.threshold,
    )
    for warning in match.warnings[:10]:
        logger.warning("%s", warning)

    evaluable = match.evaluable
    if not evaluable:
        logger.warning(
            "No ranked image could be matched to ground truth; metrics will be empty. "
            "Check --selected/--rejected, or use a ranking file that carries labels."
        )

    metric_set = metrics_package.compute(evaluable)
    confusion = confusion_matrix(evaluable, config.threshold)
    sweep = sweep_thresholds(
        evaluable,
        current_threshold=config.threshold,
        steps=config.threshold_steps,
        optimize_for=config.optimize_for,
    )
    calibration = calibration_curve(evaluable)
    distribution = score_distribution(evaluable)
    errors = analyse_errors(
        evaluable,
        borderline_low=config.borderline_low,
        borderline_high=config.borderline_high,
        limit=config.max_examples,
    )

    result = AnalysisResult(
        config=config,
        ranking=ranking,
        match=match,
        metrics=metric_set,
        confusion=confusion,
        sweep=sweep,
        calibration=calibration,
        distribution=distribution,
        errors=errors,
        elapsed_seconds=time.perf_counter() - started,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Suggestions read the finished result, so every recommendation is backed
    # by a number that also appears elsewhere in the report.
    from .suggestions import generate_suggestions

    result.suggestions = generate_suggestions(result)

    if config.comparison_mode:
        from .comparison import compare_runs

        result.comparison = compare_runs(config, result)

    logger.info("Analysis complete in %s", format_duration(result.elapsed_seconds))
    return result
