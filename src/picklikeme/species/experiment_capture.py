"""Runs `arrange_by_species` exactly as before, plus records a full
analytics experiment for it - Part 4 of the BioCLIP multi-backend
infrastructure work ("Extend the Analytics backend... Just collect all
required information").

Deliberately a thin wrapper, not a change to `arrange_by_species` itself:
the real classify-and-file pass and the analytics observation of it are two
separate concerns (the same separation `ranking.classic.rank_folder`
already keeps between scoring and `analytics.capture.record_run`). This
module owns none of the classification logic - it only listens to
`arrange_by_species`'s `on_result` hook and `species.experiment.build_
experiment_metadata`'s introspection, then hands what it collected to the
same generic `analytics.capture.record_run` ranking runs already use.

No benchmark comparison logic lives here (agreement rate, precision/recall,
confusion matrices) - per the explicit instruction, this phase collects,
it does not yet report. See docs/Analytics_Dashboard_Plan.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..analytics import DEFAULT_ANALYTICS_DB, record_run
from .arrange import SpeciesArrangeResult, arrange_by_species
from .cache import SpeciesCache
from .classifier import SpeciesClassifier, SpeciesPrediction, UNKNOWN_SPECIES
from .experiment import ExperimentMetadata, build_experiment_metadata


def run_with_analytics(
    input_folder: str | Path,
    classifier: SpeciesClassifier,
    backend: str,
    cache: SpeciesCache,
    *,
    dry_run: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    folder_name_fn: Callable[[str], str] | None = None,
    species_list_path: str | None = None,
    analytics_db: str | Path = DEFAULT_ANALYTICS_DB,
) -> tuple[SpeciesArrangeResult, str | None, ExperimentMetadata]:
    """`arrange_by_species`'s own result, unchanged, plus the recorded
    analytics `run_id` (`None` if analytics could not be written - see
    `record_run`'s "never fatal" contract, unchanged here: an analytics
    failure never raises out of this function, it just means `run_id` is
    `None`) and the `ExperimentMetadata` that was recorded alongside it.

    Collects, per the current phase's scope, exactly what Part 4 asked for
    and no more: per-image top-1..top-5 confidence and inference time
    (`run_image_metrics`), the predicted-species distribution including
    "Unknown" (`run_reject_counts`/`category_counts`), and run-level
    runtime/throughput/error-count scalars (`run_summary_metrics`). Not
    collected: raw error message text (kept on `SpeciesArrangeResult.
    failures`, which this function still returns unchanged - a log/debug
    concern, not a numeric-tracking one) and any cross-run comparison
    (explicitly deferred - see this module's own docstring).
    """
    metadata = build_experiment_metadata(classifier, backend, species_list_path=species_list_path)

    image_metrics: dict[str, dict[str, float]] = {}
    species_counts: dict[str, int] = {}

    def _observe(image_path: str, prediction: SpeciesPrediction, elapsed_seconds: float) -> None:
        metrics: dict[str, float] = {"inference_seconds": elapsed_seconds}
        if prediction.confidence is not None:
            metrics["top1_confidence"] = prediction.confidence
        if prediction.top_predictions:
            for rank, (_species, confidence) in enumerate(prediction.top_predictions, start=1):
                metrics[f"top{rank}_confidence"] = confidence
        image_metrics[image_path] = metrics
        species_counts[prediction.species] = species_counts.get(prediction.species, 0) + 1

    started = time.monotonic()
    result = arrange_by_species(
        input_folder, classifier, cache,
        dry_run=dry_run, on_progress=on_progress, folder_name_fn=folder_name_fn, on_result=_observe,
    )
    elapsed_total = time.monotonic() - started

    images_observed = len(image_metrics)
    unknown_count = species_counts.get(UNKNOWN_SPECIES, 0)
    inference_times = [m["inference_seconds"] for m in image_metrics.values()]

    summary_metrics: dict[str, float] = {
        "runtime_seconds": elapsed_total,
        "images_per_second": (images_observed / elapsed_total) if elapsed_total > 0 else 0.0,
        "errors": float(result.errors),
    }
    if inference_times:
        summary_metrics["average_inference_seconds"] = sum(inference_times) / len(inference_times)
    if images_observed:
        summary_metrics["unknown_rate"] = unknown_count / images_observed

    run_id = record_run(
        input_folder,
        backend,
        considered=result.total,
        accepted=max(0, images_observed - unknown_count),
        reject_counts=species_counts,
        image_metrics=image_metrics,
        summary_metrics=summary_metrics,
        params=metadata.to_dict(),
        device=getattr(classifier, "device", None),
        db_path=analytics_db,
    )

    return result, run_id, metadata
