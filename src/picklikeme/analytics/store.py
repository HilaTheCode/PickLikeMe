"""Persisted ranking-run history, in its own SQLite database.

Shaped like `species.cache.SpeciesCache` and `analyzer.annotations.
AnnotationStore`: one dedicated DB (`cache/analytics.db`), `CREATE TABLE IF
NOT EXISTS`, never a table bolted onto an unrelated store - analytics is a
distinct, droppable/rebuildable concern (unlike AnnotationStore's review
decisions, losing this database loses history, not photographer intent).

**Algorithm-agnostic by construction, not by convention.** The tables below
have no column for "eye confidence" or "detection confidence" or anything
else a specific strategy computes - see `run_reject_counts` (any string
reason, not an enum of known ones) and `run_image_metrics` (an
entity-attribute-value table: one row per (image, metric name), not one
column per metric). A future ranking algorithm that invents a brand new
metric or reject reason needs zero schema migration here; it just writes
rows with a new string. This is the literal mechanism behind "the analytics
layer should never assume which algorithm produced the ranking" - it is not
a design intention that has to be remembered and honoured elsewhere, it is
structurally impossible for this schema to assume that.

`runs.params_json` holds whatever the strategy's own params were - free-form
JSON, not columns - for the same reason: `ClassicVisionEyePoseParams` and a
future model's params share nothing structurally except "some parameters
were used," which is exactly what a JSON blob captures without forcing a
shared shape neither strategy actually has. Species-classification runs
store their `ExperimentMetadata` here (see `species.experiment`) for
exactly the same reason - it is just another strategy's "whatever it
needed to remember," not a schema the ranking-specific fields anticipated.

**`run_reject_counts` is not ranking-specific despite the name.** It is a
generic "count of outcomes by category label" table - a ranking run's
reject reasons and a species-classification run's predicted-species
distribution (including "Unknown" as one of the labels) are the exact same
shape: `{run_id, label, count}`. The column is still called `reason` (not
renamed to something more generic) to avoid an unnecessary migration of
existing ranking history for a purely cosmetic rename - see `reports.py`'s
`category_counts` for the generic accessor name used going forward.

**`run_summary_metrics`** holds run-*level* scalar numbers - total
runtime, images/second, average per-image inference time - as opposed to
`run_image_metrics`'s per-image numbers. Kept as its own small table rather
than folded into `run_image_metrics` under a sentinel image_path, because
"a fact about this run as a whole" and "a fact about one image in it" are
different enough concepts to deserve not being conflated, even though both
are technically `{name: value}` bags.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..config import PROJECT_ROOT

DEFAULT_ANALYTICS_DB = PROJECT_ROOT / "cache" / "analytics.db"

# Bumped whenever the table shape changes - see bird_crop.CROP_CACHE_VERSION
# and eyes.cache.EYE_CACHE_VERSION for the same discipline. Unlike those
# caches, a version bump here should prefer an ALTER/migration over a wipe
# where practical, since this table is *history*, not a rebuildable cache -
# but no reader may assume a row shape newer code didn't actually write.
ANALYTICS_SCHEMA_VERSION = 1


class AnalyticsStore:
    """Ranking-run history, keyed by a generated `run_id` - never by path or
    content identity, because a run is an event (something that happened at
    a point in time), not a fact about an image the way a detection or a
    species prediction is."""

    def __init__(self, db_path: str | Path = DEFAULT_ANALYTICS_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_info(key, value) VALUES ('version', ?)",
                (str(ANALYTICS_SCHEMA_VERSION),),
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       TEXT PRIMARY KEY,
                    folder       TEXT NOT NULL,
                    strategy_id  TEXT NOT NULL,
                    started_at   TEXT NOT NULL,
                    considered   INTEGER NOT NULL,
                    accepted     INTEGER NOT NULL,
                    device       TEXT,
                    params_json  TEXT NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_folder ON runs(folder)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy_id)")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_reject_counts (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    reason TEXT NOT NULL,
                    count  INTEGER NOT NULL,
                    PRIMARY KEY (run_id, reason)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_image_metrics (
                    run_id      TEXT NOT NULL REFERENCES runs(run_id),
                    image_path  TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value       REAL NOT NULL,
                    PRIMARY KEY (run_id, image_path, metric_name)
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_image_metrics(run_id)")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_summary_metrics (
                    run_id      TEXT NOT NULL REFERENCES runs(run_id),
                    metric_name TEXT NOT NULL,
                    value       REAL NOT NULL,
                    PRIMARY KEY (run_id, metric_name)
                )
                """
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AnalyticsStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def insert_run(
        self,
        run_id: str,
        *,
        folder: str,
        strategy_id: str,
        started_at: str,
        considered: int,
        accepted: int,
        device: str | None,
        params: dict,
        reject_counts: dict[str, int],
        image_metrics: dict[str, dict[str, float]],
        summary_metrics: dict[str, float] | None = None,
    ) -> None:
        """One run, atomically - a run with reject counts recorded but no
        row in `runs` (or vice versa) would make every aggregate query
        wrong, so this is a single transaction, not a sequence of calls a
        caller could partially complete."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, folder, strategy_id, started_at, considered, accepted, device, params_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, folder, strategy_id, started_at, considered, accepted, device, json.dumps(params)),
            )
            self._conn.execute("DELETE FROM run_reject_counts WHERE run_id = ?", (run_id,))
            self._conn.executemany(
                "INSERT INTO run_reject_counts(run_id, reason, count) VALUES (?, ?, ?)",
                [(run_id, reason, count) for reason, count in reject_counts.items()],
            )
            self._conn.execute("DELETE FROM run_image_metrics WHERE run_id = ?", (run_id,))
            self._conn.executemany(
                "INSERT INTO run_image_metrics(run_id, image_path, metric_name, value) VALUES (?, ?, ?, ?)",
                [
                    (run_id, image_path, metric_name, value)
                    for image_path, metrics in image_metrics.items()
                    for metric_name, value in metrics.items()
                    if value is not None
                ],
            )
            self._conn.execute("DELETE FROM run_summary_metrics WHERE run_id = ?", (run_id,))
            self._conn.executemany(
                "INSERT INTO run_summary_metrics(run_id, metric_name, value) VALUES (?, ?, ?)",
                [
                    (run_id, name, value)
                    for name, value in (summary_metrics or {}).items()
                    if value is not None
                ],
            )

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["params"] = json.loads(result.pop("params_json"))
        return result

    def reject_counts(self, run_id: str) -> dict[str, int]:
        """Ranking's own name for `category_counts` - kept as a separate,
        identically-implemented method so existing ranking call sites read
        naturally; see this module's own docstring for why the underlying
        table is generic."""
        return self.category_counts(run_id)

    def category_counts(self, run_id: str) -> dict[str, int]:
        """Any categorical outcome breakdown for a run - a ranking run's
        reject reasons, or a species-classification run's predicted-species
        distribution (see this module's own docstring)."""
        rows = self._conn.execute(
            "SELECT reason, count FROM run_reject_counts WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {row["reason"]: row["count"] for row in rows}

    def metric_names(self, run_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT metric_name FROM run_image_metrics WHERE run_id = ? ORDER BY metric_name", (run_id,)
        ).fetchall()
        return [row["metric_name"] for row in rows]

    def metric_values(self, run_id: str, metric_name: str) -> list[float]:
        rows = self._conn.execute(
            "SELECT value FROM run_image_metrics WHERE run_id = ? AND metric_name = ?",
            (run_id, metric_name),
        ).fetchall()
        return [row["value"] for row in rows]

    def image_paths(self, run_id: str) -> list[str]:
        """Every image this run recorded at least one metric for - the
        image list an Image Inspector browses."""
        rows = self._conn.execute(
            "SELECT DISTINCT image_path FROM run_image_metrics WHERE run_id = ? ORDER BY image_path", (run_id,)
        ).fetchall()
        return [row["image_path"] for row in rows]

    def image_metrics(self, run_id: str, image_path: str) -> dict[str, float]:
        """Every metric recorded for one specific image in this run - what
        an Image Inspector shows for a single selected image."""
        rows = self._conn.execute(
            "SELECT metric_name, value FROM run_image_metrics WHERE run_id = ? AND image_path = ?",
            (run_id, image_path),
        ).fetchall()
        return {row["metric_name"]: row["value"] for row in rows}

    def summary_metrics(self, run_id: str) -> dict[str, float]:
        """Run-level scalar metrics (runtime, images/sec, ...) - see this
        module's own docstring on why these live in their own table."""
        rows = self._conn.execute(
            "SELECT metric_name, value FROM run_summary_metrics WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {row["metric_name"]: row["value"] for row in rows}

    def list_runs(self, *, folder: str | None = None, strategy_id: str | None = None) -> list[dict]:
        """Every recorded run, most recent first - optionally narrowed to
        one folder and/or one strategy. The raw material for "is the
        algorithm improving over time" (compare runs across `started_at`)."""
        query = "SELECT run_id, folder, strategy_id, started_at, considered, accepted FROM runs WHERE 1=1"
        params: list[str] = []
        if folder is not None:
            query += " AND folder = ?"
            params.append(folder)
        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        query += " ORDER BY started_at DESC"
        return [dict(row) for row in self._conn.execute(query, params).fetchall()]
