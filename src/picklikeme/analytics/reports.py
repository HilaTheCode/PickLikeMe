"""Phase 1 reports: Run Statistics and Rejection Analysis, over one recorded
run - the two report types asked for first (see docs/Analytics_Dashboard_
Plan.md), built purely from `AnalyticsStore`'s generic tables.

Deliberately not here yet: user-vs-algorithm agreement, feature attribution,
threshold-change simulation, cross-session/long-term trends. Those need
data this phase does not capture (user override events, per-component score
attribution) - see the plan doc for why they are later phases, not missing
by oversight.

Every function takes a `run_id`, never a strategy name or an assumption
about which metrics/reasons exist - see `store.py`'s own docstring for why.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .store import AnalyticsStore


def run_statistics(store: AnalyticsStore, run_id: str) -> dict:
    """One run's summary: counts plus the mean of every metric that was
    actually recorded for it - whatever those turn out to be for this
    strategy, not a fixed list. `None` if `run_id` is not on record.
    """
    run = store.get_run(run_id)
    if run is None:
        return {}

    reject_counts = store.reject_counts(run_id)
    metric_means = {
        name: round(statistics.fmean(values), 4)
        for name in store.metric_names(run_id)
        if (values := store.metric_values(run_id, name))
    }

    return {
        "run_id": run_id,
        "folder": run["folder"],
        "strategy_id": run["strategy_id"],
        "started_at": run["started_at"],
        "device": run["device"],
        "considered": run["considered"],
        "accepted": run["accepted"],
        "rejected": run["considered"] - run["accepted"],
        "reject_counts": reject_counts,
        "metric_means": metric_means,
        "params": run["params"],
    }


def rejection_analysis(store: AnalyticsStore, run_id: str) -> list[dict]:
    """Every reject reason for this run, with both a count and a percentage
    of `considered` - the two numbers the spec asks for side by side, so a
    reader is never left computing the percentage themselves."""
    run = store.get_run(run_id)
    if run is None or run["considered"] == 0:
        return []
    considered = run["considered"]
    counts = store.reject_counts(run_id)
    rows = [
        {"reason": reason, "count": count, "percent_of_considered": round(100.0 * count / considered, 2)}
        for reason, count in counts.items()
    ]
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


def confidence_distribution(store: AnalyticsStore, run_id: str, metric_name: str) -> list[float]:
    """The raw values for one metric across every image in this run - the
    material for a histogram (or, today, a CSV column) rather than only a
    mean, so a threshold can be tuned from the real spread, not a single
    number. Empty if this run never recorded `metric_name` - not every
    strategy computes the same metrics (see store.py's own docstring)."""
    return store.metric_values(run_id, metric_name)


def export_run_statistics_csv(store: AnalyticsStore, run_id: str, path: str | Path) -> Path:
    stats = run_statistics(store, run_id)
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "value"])
        for key in ("run_id", "folder", "strategy_id", "started_at", "device", "considered", "accepted", "rejected"):
            writer.writerow([key, stats.get(key, "")])
        for reason, count in stats.get("reject_counts", {}).items():
            writer.writerow([f"reject_count:{reason}", count])
        for name, mean in stats.get("metric_means", {}).items():
            writer.writerow([f"metric_mean:{name}", mean])
    return path


def export_rejection_analysis_csv(store: AnalyticsStore, run_id: str, path: str | Path) -> Path:
    rows = rejection_analysis(store, run_id)
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["reason", "count", "percent_of_considered"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_confidence_distribution_csv(store: AnalyticsStore, run_id: str, metric_name: str, path: str | Path) -> Path:
    values = confidence_distribution(store, run_id, metric_name)
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([metric_name])
        for value in values:
            writer.writerow([value])
    return path
