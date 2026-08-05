"""Score Explanation (Phase 6): for one image in one Classic Vision ranking
run, exactly how its final score was produced.

`AnalyticsStore` only ever persisted the three RAW per-image metrics
(`ranking.classic.ImageMetrics`) and the final combined score - never the
intermediate normalized values, weights, or per-metric contributions that
produced it (see `ranking.classic.combine`, which computes and discards
them in the same call). Nothing here is fabricated: every number is
recomputed using the exact same `ranking.metrics.robust_normalize` +
weighted-sum arithmetic `combine()` already uses, re-run against the run's
own recorded values rather than invented.

Deliberately excludes `eye_confidence`/`head_confidence`: both are recorded
per-image (see `ImageMetrics`), but `combine()` never weights either of them
into the final score - showing a weight/contribution for a metric that
never actually influenced the score would misrepresent how the number was
produced, the one thing this module exists to explain truthfully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ranking.classic import METRIC_LABELS
from ..ranking.metrics import NORMALIZE_HIGH_PERCENTILE, NORMALIZE_LOW_PERCENTILE, robust_normalize

# The only metrics ranking.classic.combine() actually weights into the final
# score, in the same order combine() adds them - "Running Total" only means
# something for a fixed, consistent order.
WEIGHTED_METRICS: tuple[str, ...] = ("eye_sharpness", "subject_sharpness", "subject_size")


@dataclass(frozen=True)
class ScoreExplanationRow:
    metric: str
    label: str
    raw_value: float
    normalized_value: float
    weight: float
    contribution: float
    running_total: float


@dataclass(frozen=True)
class ScoreExplanation:
    rows: tuple[ScoreExplanationRow, ...]
    final_score: float | None
    # Sum of every row's contribution - should equal final_score up to float
    # rounding when every weighted metric was recorded; a caller can use a
    # mismatch as a signal that this run recorded a metric set combine()
    # itself never saw (should not happen, but is not this module's job to
    # silently paper over).
    recomputed_score: float | None


def _normalize_one(all_values: list[float], value: float) -> float:
    """The exact percentile-clip arithmetic `robust_normalize` applies to a
    whole list, applied to a single already-known value against that same
    list's own low/high bounds - avoids `robust_normalize(all_values).index
    (value)`, which would silently pick the wrong entry whenever two images
    in the same run happen to share an identical raw metric value.
    """
    if not all_values:
        return 0.5
    array = np.asarray(all_values, dtype=np.float64)
    low = float(np.percentile(array, NORMALIZE_LOW_PERCENTILE))
    high = float(np.percentile(array, NORMALIZE_HIGH_PERCENTILE))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-12:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def explain_score(store, run_id: str, image_path: str) -> ScoreExplanation | None:
    """None when this image has no recorded metrics at all for this run (an
    AI-model/species run - see `ranking.ai_model`, which records only a bare
    `score` - or an image this run never scored). A ranking run missing just
    one or two of the three weighted metrics still yields whichever rows ARE
    present, rather than nothing.
    """
    image_metrics = store.image_metrics(run_id, image_path)
    if not image_metrics:
        return None
    present_metrics = [name for name in WEIGHTED_METRICS if name in image_metrics]
    if not present_metrics:
        return None

    run = store.get_run(run_id) or {}
    weights = (run.get("params") or {}).get("weights") or {}

    rows: list[ScoreExplanationRow] = []
    running_total = 0.0
    for name in present_metrics:
        raw = float(image_metrics[name])
        all_values = store.metric_values(run_id, name)
        normalized = _normalize_one(all_values, raw)
        weight = float(weights.get(f"{name}_weight", 0.0))
        contribution = weight * normalized
        running_total += contribution
        rows.append(ScoreExplanationRow(
            metric=name, label=METRIC_LABELS.get(name, name), raw_value=raw,
            normalized_value=normalized, weight=weight, contribution=contribution,
            running_total=running_total,
        ))

    return ScoreExplanation(
        rows=tuple(rows), final_score=image_metrics.get("score"), recomputed_score=running_total,
    )
