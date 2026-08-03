"""Diagnostics & Analytics - the foundation phase.

Answers "is the algorithm improving, and where is it failing" from data
already produced by every ranking run, without assuming which algorithm
produced it. See this package's own module docstrings (`store.py` for the
schema, `capture.py` for how a run gets recorded, `reports.py` for what can
be asked of it) for the detail; the short version:

    ranking run finishes
        -> capture.record_run(...)              generic, algorithm-agnostic
            -> store.AnalyticsStore              three tables, no per-algorithm columns
    photographer/developer wants a report
        -> reports.run_statistics/rejection_analysis/export_*_csv

This is Phase 1 of a larger plan (run statistics + rejection analysis,
CSV export) - user-vs-algorithm agreement, feature attribution, threshold
simulation, and cross-session learning are later phases, deliberately not
built yet, but the schema here (an EAV-shaped metrics table, a reason-count
table, free-form JSON params) is designed so those phases extend it rather
than replace it.
"""

from .capture import record_run
from .store import AnalyticsStore, DEFAULT_ANALYTICS_DB

__all__ = ["record_run", "AnalyticsStore", "DEFAULT_ANALYTICS_DB"]
