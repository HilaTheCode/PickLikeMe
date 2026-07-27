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
    # Annotations: display data only. Loaded after every metric has been
    # computed, and never read by a metric, a threshold sweep or a suggestion
    # rule - human knowledge must not move the numbers it explains.
    #
    # `annotations` is keyed by path and holds both categories together (a
    # given image is never both, so there is no collision); `annotation_summary`
    # and `fp_annotation_summary` are the per-category breakdowns, computed
    # identically so the two are directly comparable.
    annotations: dict = field(default_factory=dict)
    annotation_summary: object | None = None
    fp_annotation_summary: object | None = None
    # Detector boxes for the false negatives only, for the diagnostic
    # overlay. Display data: no metric reads them.
    detections: dict = field(default_factory=dict)
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
            "false_negative_detections": {
                path: record.as_dict() for path, record in self.detections.items()
            },
            "false_negative_annotations": {
                "summary": self.annotation_summary.as_dict() if self.annotation_summary else None,
                "by_image": {
                    path: a.as_dict()
                    for path, a in self.annotations.items()
                    if path in {r.image_path for r in self.errors.false_negatives}
                },
            },
            "false_positive_annotations": {
                "summary": self.fp_annotation_summary.as_dict() if self.fp_annotation_summary else None,
                "by_image": {
                    path: a.as_dict()
                    for path, a in self.annotations.items()
                    if path in {r.image_path for r in self.errors.false_positives}
                },
            },
        }


def _attach_annotations(result: AnalysisResult) -> None:
    """Load annotations onto a finished result, for both mistake categories.

    False negatives and false positives are both annotatable, with the same
    fields and vocabulary, so a photographer can compare the two directly.
    `summarise()` is called once per category against the same store, giving
    two independently-computed breakdowns; the per-image annotations are then
    merged into one dict since an image is never in both categories at once.

    A missing or unreadable database is not an error - the knowledge base is
    optional, and an analysis must never fail because of it.
    """
    from .annotations import AnnotationStore, summarise

    fn_paths = [record.image_path for record in result.errors.false_negatives]
    fp_paths = [record.image_path for record in result.errors.false_positives]
    try:
        with AnnotationStore(result.config.annotations_db_path) as store:
            fn_annotations, result.annotation_summary = summarise(store, fn_paths)
            fp_annotations, result.fp_annotation_summary = summarise(store, fp_paths)
    except Exception as exc:  # noqa: BLE001 - optional feature, never fatal
        logger.warning("Could not read the annotation database: %s", exc)
        return

    result.annotations = {**fn_annotations, **fp_annotations}
    logger.info(
        "Annotations: %d of %d false negatives, %d of %d false positives (%d total in database)",
        result.annotation_summary.annotated,
        result.annotation_summary.total_images,
        result.fp_annotation_summary.annotated,
        result.fp_annotation_summary.total_images,
        result.annotation_summary.total_in_database,
    )


def _attach_detections(result: AnalysisResult) -> None:
    """Resolve detector boxes for every image this run will show a thumbnail
    for - every contact sheet and every HTML thumbnail table, not just false
    negatives (the overlay used to be scoped to false negatives only; it now
    covers the whole report).

    `sheet_specs()` already enumerates exactly that universe (it is also what
    decides which images get a thumbnail generated at all), so it is reused
    here rather than re-deriving the same set a second way.

    Prefers what preprocessing recorded, then the analyzer's own cache, and only
    then runs the detector - never for images outside this report. A failure
    leaves the overlay off rather than losing the report. Reporting a much
    larger image set than the old false-negatives-only scope can mean detecting
    on a proportionally larger un-recorded backlog the first time this runs
    against an older crop cache; every result is cached by content identity, so
    later runs over the same images are free regardless.
    """
    from ..config import DEFAULT_CROP_CACHE_DIR
    from .contactsheets import sheet_specs
    from .detections import DetectionCache

    paths = sorted({image.image_path for spec in sheet_specs(result) for image in spec.images})
    if not paths:
        return
    config = result.config
    crop_cache = config.crop_cache_dir or DEFAULT_CROP_CACHE_DIR
    try:
        with DetectionCache(config.detections_db_path, crop_cache) as cache:
            result.detections = cache.get_many(
                paths,
                conf_threshold=config.detection_conf_threshold,
                device=config.detection_device,
                allow_detect=config.detect_missing_boxes,
            )
    except Exception as exc:  # noqa: BLE001 - the overlay is a nicety
        logger.warning("Could not resolve detector boxes: %s", exc)


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

    # Annotations are loaded LAST, deliberately: by this point every metric,
    # threshold and suggestion is already fixed, so human notes cannot
    # influence any of them. They are attached purely for the report to show.
    if config.annotations_enabled:
        _attach_annotations(result)

    # Detector boxes for the false-negative overlay. Loaded after the metrics
    # for the same reason as annotations: this is diagnosis, not measurement.
    if config.annotate_detections:
        _attach_detections(result)

    if config.comparison_mode:
        from .comparison import compare_runs

        result.comparison = compare_runs(config, result)

    logger.info("Analysis complete in %s", format_duration(result.elapsed_seconds))
    return result
