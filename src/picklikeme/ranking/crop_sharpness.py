"""Crop Sharpness - ranks by how sharp the existing subject crop is, plus a
smaller bonus for how much of the frame the subject fills.

Deliberately the simplest possible ranking signal in this project: no eye
detection, no head detection, no domain classification - just the crop the
shared crop/localization pipeline already produced (see `bird_crop.py`,
unmodified by this module) and two pure measurements of it,
`ranking.metrics.subject_focus_measure`/`normalized_subject_size`, which
already existed for exactly this purpose (Classic Vision's own subject-size
and subject-sharpness terms). This strategy exists to test whether
whole-crop sharpness alone is a useful wildlife-photo ranking signal,
independent of whether an eye can be located at all - see the crop cannot be
in focus everywhere a photographer might frame a subject, so this is a
different, complementary question from "is the eye sharp."

Only one filter gates an image here: does a valid subject crop exist at all
(`SubjectFilter`, the same first-stage gate every other strategy in this
project uses). There is no second filter, because there is no eye to be
visible or not - an image with a valid crop always produces a sharpness
value, however low.

**The score is absolute, and it has two forms.** An image whose crop came
from a REAL subject detection scores `0.80 * sharpness + 0.20 * relative
subject size`. An image that fell back to the whole frame - nothing was
located in it - scores `1.00 * sharpness`, with no subject-size term at all,
because there is no subject whose size could be measured. Both terms are
fixed maps onto [0, 1] (`metrics.absolute_sharpness_score` and
`normalized_subject_size`), never a percentile within the current run, so a
crop's score does not change with the size or composition of the folder it
was ranked in. See `combine`.

Same two-phase shape as `ranking.classic` (filter, then measure/score for
survivors), and the same sidecar conventions (`sidecar.discover_metric_reports`
picks up this module's own `crop-sharpness_metrics.json` exactly like it
already does for Classic Vision's, since the payload shape - `version`/
`strategy`/`metrics`) is what that discovery function actually reads, not any
hardcoded strategy list).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..analytics import DEFAULT_ANALYTICS_DB, record_run
from ..analytics.environment import resolve_environment_info
from ..auto_crop import resolve_device
from ..bird_crop import CropParams
from ..config import DEFAULT_CROP_CACHE_DIR, DEFAULT_MAX_CSV_ROWS
from ..dataset import UnlabeledImageDataset
from ..preprocess import build_cache
from ..sidecar import SIDECAR_DIRNAME, ensure_sidecar_dir, strategy_ranking_path, write_run_metadata
from .base import GROUP_WEIGHTS, ParamSpec, StrategyInfo, WeightedParams, use_subject_filter_spec
from .classic import analysis_targets, read_filter_report, write_filter_report
from .classic import ClassicVisionStrategy as _ClassicVisionStrategy
from .filters import FilterChain, SubjectFilter
from .metrics import absolute_sharpness_score, normalized_subject_size, subject_focus_measure

logger = logging.getLogger(__name__)

STRATEGY_ID = "crop-sharpness"

# Bumped whenever this module's own scoring logic changes in a way that
# could change a result - same "which exact axis changed" discipline
# ranking.classic.ALGORITHM_VERSION already applies.
#
# v2: scores became ABSOLUTE. Sharpness now goes through
# `metrics.absolute_sharpness_score` (a fixed curve) instead of the
# folder-relative `robust_normalize`, subject size is used as the raw area
# fraction it already is, and a full-frame-fallback image is scored on
# sharpness alone with no subject-size term at all. A v1 score and a v2
# score for the same image are not comparable.
ALGORITHM_VERSION = "2"

METRICS_REPORT_FILENAME = "crop-sharpness_metrics.json"

METRIC_LABELS: dict[str, str] = {
    "crop_sharpness": "Crop Sharpness",
    "relative_subject_size": "Relative Subject Size",
}


def _scoring_weight_specs() -> tuple[ParamSpec, ...]:
    return (
        ParamSpec(
            name="crop_sharpness_weight",
            label="Crop sharpness",
            default=80.0,
            minimum=0.0,
            maximum=1000.0,
            group=GROUP_WEIGHTS,
            help="How sharp the whole subject crop is - the primary signal.",
        ),
        ParamSpec(
            name="relative_subject_size_weight",
            label="Relative subject size",
            default=20.0,
            minimum=0.0,
            maximum=1000.0,
            group=GROUP_WEIGHTS,
            help="How much of the frame the subject fills - a secondary bonus.",
        ),
    )


@dataclass(frozen=True)
class CropSharpnessParams(WeightedParams):
    """The two weights this strategy combines - 80/20 by default, but any
    two non-negative numbers mean the same thing (see
    `WeightedParams.normalized_weights`): a photographer typing 4/1 gets an
    identical ranking to the 80/20 default."""

    crop_sharpness_weight: float = 80.0
    relative_subject_size_weight: float = 20.0

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        return (*_scoring_weight_specs(), use_subject_filter_spec())


@dataclass
class ImageMetrics:
    """The raw measurements for one surviving image.

    `relative_subject_size` is None - not 0.0 - for an image that reached
    scoring through the full-frame fallback. There is no subject box to
    measure there, so there is no such measurement to record, and a 0.0
    would be a fabricated one that every reader (the metrics report, the
    Loupe's diagnostics line, the analytics store) would then display as a
    real "this subject fills 0% of the frame". Absent and zero are different
    facts; see `measure`.
    """

    image_path: str
    crop_sharpness: float
    relative_subject_size: float | None
    # Whether a REAL subject detection produced this image's crop, as opposed
    # to the whole-frame fallback - `FilterCandidate.has_selected_detection`,
    # carried forward because `combine` scores the two cases differently.
    has_subject_detection: bool = True


def measure(candidate) -> ImageMetrics:
    """The metrics for one image that passed the filter chain - a pure
    function of the candidate, exactly like `ranking.classic.measure`.

    Whole-crop sharpness is always measured: whatever the crop turned out to
    be, its sharpness is a real property of real pixels.

    Relative subject size is measured ONLY when a real subject detection
    exists. On the full-frame fallback the "subject box" IS the frame, so
    `normalized_subject_size` would return 1.0 by construction - the maximum
    possible score for a subject that was never located at all, handing every
    undetected image the full size bonus. It is not measured, not stored, and
    not scored; see `combine`.
    """
    assert candidate.subject_crop is not None  # noqa: S101 - guaranteed by FilterChain
    has_detection = candidate.has_selected_detection
    return ImageMetrics(
        image_path=candidate.image_path,
        crop_sharpness=subject_focus_measure(candidate.subject_crop),
        relative_subject_size=(
            normalized_subject_size(candidate.subject_box, candidate.source_size or (0, 0))
            if has_detection and candidate.subject_box is not None
            else None
        ),
        has_subject_detection=has_detection,
    )


def combine(metrics: list[ImageMetrics], weights: dict[str, float]) -> list[float]:
    """One score per image, on an ABSOLUTE 0-1 scale.

    Two cases, and the difference between them is the point:

    **A real subject crop** - the pipeline located a subject - scores
    `0.80 * sharpness + 0.20 * relative subject size` (the configured
    weights; 80/20 by default).

    **The full-frame fallback** - no subject was located - scores
    `1.00 * sharpness`. The subject-size term is not down-weighted or
    defaulted to zero, it is ABSENT: the sharpness weight is renormalised to
    the whole score, so a fallback image is judged purely on how sharp it is
    and is neither rewarded nor punished for a subject nobody found. Scoring
    it as `0.8 * sharpness + 0.2 * <something>` would need a value for
    `<something>`, and every available choice is a lie - 1.0 says the subject
    fills the frame, 0.0 says it is invisibly small, and both are claims
    about a subject that was never detected.

    Both terms are absolute. Sharpness goes through
    `metrics.absolute_sharpness_score` (a fixed curve - see its docstring),
    and `normalized_subject_size` is already an absolute area fraction in
    [0, 1]. Neither consults the other images in the run, so this function is
    a pure per-image map: the same crop scores the same number whether it is
    ranked alone or alongside 6,000 others. `robust_normalize`, which this
    used on both terms before, made the result a position within the current
    folder's distribution instead - two runs over different subsets of the
    same shoot could not be compared, and the clipping flattened the top and
    bottom of every run onto identical 1.000/0.000 plateaus.
    """
    sharpness_weight = weights["crop_sharpness_weight"]
    size_weight = weights["relative_subject_size_weight"]
    total = sharpness_weight + size_weight
    scores: list[float] = []
    for metric in metrics:
        sharpness = absolute_sharpness_score(metric.crop_sharpness)
        if not metric.has_subject_detection or metric.relative_subject_size is None:
            scores.append(sharpness)
            continue
        scores.append(
            (sharpness_weight * sharpness + size_weight * metric.relative_subject_size) / total
            if total > 0
            else 0.0
        )
    return scores


def write_metrics_report(input_folder: str | Path, metrics: list[ImageMetrics]) -> Path:
    """Record every surviving image's raw crop-sharpness/relative-size
    measurements, in the same self-describing shape every other strategy's
    metrics report uses (see `sidecar.discover_metric_reports`) - so the
    Loupe's diagnostics line picks this up with no code change there."""
    ensure_sidecar_dir(input_folder)
    target = Path(input_folder) / SIDECAR_DIRNAME / METRICS_REPORT_FILENAME
    payload = {
        "version": 1,
        "strategy": STRATEGY_ID,
        "metrics": {
            m.image_path: {
                "crop_sharpness": m.crop_sharpness,
                # Absent (null) for a full-frame-fallback image - see
                # ImageMetrics. A reader that shows this must show "not
                # measured", never 0.0.
                "relative_subject_size": m.relative_subject_size,
                "has_subject_detection": m.has_subject_detection,
            }
            for m in metrics
        },
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_metrics_report(input_folder: str | Path) -> dict:
    """The last run's raw per-image metrics for this folder, or `{}`."""
    target = Path(input_folder) / SIDECAR_DIRNAME / METRICS_REPORT_FILENAME
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", target, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


class CropSharpnessStrategy:
    """Ranks a folder by whole-crop sharpness plus a relative-size bonus.
    Implements `ranking.base.RankingStrategy`.

    Reuses the shared crop cache (`preprocess.build_cache`, unmodified) for
    its input exactly like every other strategy, and `ranking.filters.
    SubjectFilter` as its only eligibility gate - there is deliberately no
    second filter and no eye detector of any kind, per this strategy's own
    purpose (see the module docstring).
    """

    info = StrategyInfo(
        strategy_id=STRATEGY_ID,
        display_name="Crop Sharpness",
        description=(
            "Ranks by the sharpness of the existing subject crop, with a smaller bonus for how "
            "much of the frame the subject fills. No eye or head detection - tests whether "
            "whole-crop sharpness alone is a useful wildlife-photo ranking signal."
        ),
        score_label="Crop Sharpness",
    )
    params_class = CropSharpnessParams
    param_specs = CropSharpnessParams.specs()
    metric_labels = METRIC_LABELS

    def rank_folder(
        self,
        input_folder: str | Path,
        *,
        params: CropSharpnessParams | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        crop_cache_dir: str | Path = DEFAULT_CROP_CACHE_DIR,
        device: str | None = None,
        max_rows: int = DEFAULT_MAX_CSV_ROWS,
        force_preprocess: bool = False,
        analytics_db: str | Path = DEFAULT_ANALYTICS_DB,
    ) -> dict:
        """See the class docstring. `force_preprocess`/`crop_cache_dir`/
        `analytics_db` mean exactly what they mean for every other strategy
        (see `ranking.classic.ClassicVisionStrategy.rank_folder`) - this
        strategy shares the same crop cache and analytics store, never a
        strategy-specific one."""
        from ..train import write_results_csv

        start_time = time.perf_counter()
        params = params or self.params_class()
        input_folder = Path(input_folder)
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

        image_paths = analysis_targets(input_folder)
        if not image_paths:
            raise ValueError(f"No images found under {input_folder.resolve()}")
        dataset = UnlabeledImageDataset(image_paths)

        resolved_device = resolve_device(device)

        # Step 1: the shared crop cache - unmodified defaults, exactly as
        # every other strategy builds it. This strategy never changes crop
        # geometry, thresholds, or preprocessing.
        if on_stage is not None:
            on_stage("Building subject-crop cache")
        crop_params = CropParams()
        build_cache(image_paths, crop_cache_dir, crop_params, device=resolved_device, force=force_preprocess)

        if on_stage is not None:
            on_stage("Measuring crop sharpness")
        chain = FilterChain([SubjectFilter()])
        measurements: list[ImageMetrics] = []
        rejected: dict[str, str] = {}
        counts: dict[str, int] = {}
        for index, image_path in enumerate(image_paths, start=1):
            candidate = _ClassicVisionStrategy._load_candidate(
                image_path, crop_cache_dir, require_selected_detection=params.use_subject_filter
            )
            reason = chain.reject_reason(candidate)
            if reason is not None:
                rejected[image_path] = reason
                counts[reason] = counts.get(reason, 0) + 1
            else:
                measurements.append(measure(candidate))
            if on_progress is not None:
                on_progress(index, len(image_paths))

        if on_stage is not None:
            on_stage("Scoring and writing the ranking")
        scores = combine(measurements, params.normalized_weights())
        ranked = [
            (Path(m.image_path).name, score, 0, str(m.image_path))
            for m, score in zip(measurements, scores)
        ]
        score_by_path = {m.image_path: score for m, score in zip(measurements, scores)}
        ranked.sort(key=lambda entry: entry[1], reverse=True)

        output_paths = write_results_csv(
            strategy_ranking_path(input_folder, STRATEGY_ID),
            dataset,
            ranked,
            select_root=str(input_folder),
            reject_root="(crop sharpness - no labels)",
            max_rows=max_rows,
        )
        write_filter_report(input_folder, rejected, counts, strategy_id=STRATEGY_ID)
        write_metrics_report(input_folder, measurements)
        write_run_metadata(
            input_folder,
            strategy=STRATEGY_ID,
            image_count=len(ranked),
            considered=len(image_paths),
            filtered=counts,
            weights=params.normalized_weights(),
        )
        runtime_seconds = time.perf_counter() - start_time
        record_run(
            input_folder,
            STRATEGY_ID,
            considered=len(image_paths),
            accepted=len(ranked),
            reject_counts=counts,
            image_metrics={
                m.image_path: {
                    "score": score_by_path[m.image_path],
                    "crop_sharpness": m.crop_sharpness,
                    **(
                        {"relative_subject_size": m.relative_subject_size}
                        if m.relative_subject_size is not None
                        else {}
                    ),
                }
                for m in measurements
            },
            summary_metrics={
                "runtime_seconds": runtime_seconds,
                "images_per_second": len(image_paths) / runtime_seconds if runtime_seconds > 0 else 0.0,
            },
            params={
                "algorithm_version": ALGORITHM_VERSION,
                "weights": params.normalized_weights(),
                **resolve_environment_info(),
            },
            device=resolved_device,
            db_path=analytics_db,
        )

        return {
            "strategy": STRATEGY_ID,
            "output_csv": output_paths[0],
            "extra_csv_files": output_paths[1:],
            "image_count": len(ranked),
            "considered": len(image_paths),
            "filtered": counts,
            "device": resolved_device,
            "top": [(name, score) for name, score, _label, _path in ranked[:10]],
        }
