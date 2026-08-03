"""Turns one run's own already-computed results into a durable analytics
record - originally built for ranking runs, now shared by species-
classification runs too (see `species.experiment_capture`), unchanged,
because nothing about its shape was ever ranking-specific.

Deliberately generic: `record_run` takes only primitive shapes (counts, a
`{label: count}` dict, a `{image_path: {metric_name: value}}` dict, a plain
`params` dict, an optional `{metric_name: value}` run-level summary dict) -
never a `ClassicVisionParams`, an `ImageMetrics`, an `ExperimentMetadata`,
or any other module-specific type; callers pass `.to_dict()`/equivalent
plain data. Every current and future ranking or classification strategy
already computes exactly this shape of data on its way to writing its own
results (see `ranking.classic.rank_folder`'s, `rank.rank_folder`'s, and
`species.experiment_capture.run_with_analytics`'s own endings) - this
module asks for nothing a caller does not already have in hand, and imports
nothing from `ranking` or `species`.

Failure to record is never fatal - a photographer's run must complete and
produce its real output (a ranking CSV, filed species folders) regardless
of whether analytics history could be written, exactly like `eyes.cache.
save_eye_detection`'s own "failure to write is not fatal" contract.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

from .store import DEFAULT_ANALYTICS_DB, AnalyticsStore

logger = logging.getLogger(__name__)


def record_run(
    folder: str | Path,
    strategy_id: str,
    *,
    considered: int,
    accepted: int,
    reject_counts: dict[str, int] | None = None,
    image_metrics: dict[str, dict[str, float]] | None = None,
    summary_metrics: dict[str, float] | None = None,
    params: dict | None = None,
    device: str | None = None,
    db_path: str | Path = DEFAULT_ANALYTICS_DB,
) -> str | None:
    """Record one completed run (ranking or species classification).
    Returns the generated `run_id`, or `None` if the record could not be
    written (logged, never raised) - a run's own real output is the thing
    that must never be blocked by this.

    `folder`/`strategy_id`/`considered`/`accepted` are the only genuinely
    required facts about a run - "an image count and where it happened."
    Everything else is optional because not every strategy has it (the AI
    ranking model has no reject reasons; a strategy with no configurable
    parameters has nothing for `params`). `reject_counts` doubles as any
    categorical outcome breakdown - a species-classification run passes its
    predicted-species distribution here, not just ranking reject reasons -
    see `store.py`'s own docstring. `summary_metrics` is for run-level
    scalars (total runtime, images/second) as opposed to `image_metrics`'s
    per-image numbers.
    """
    try:
        run_id = str(uuid.uuid4())
        with AnalyticsStore(db_path) as store:
            store.insert_run(
                run_id,
                folder=str(folder),
                strategy_id=strategy_id,
                started_at=datetime.now().isoformat(timespec="seconds"),
                considered=considered,
                accepted=accepted,
                device=device,
                params=params or {},
                reject_counts=reject_counts or {},
                image_metrics=image_metrics or {},
                summary_metrics=summary_metrics or {},
            )
        return run_id
    except Exception as exc:  # noqa: BLE001 - analytics must never break a ranking run
        logger.warning("Could not record analytics for %s run on %s: %s", strategy_id, folder, exc)
        return None
