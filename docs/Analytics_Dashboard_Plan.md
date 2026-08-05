# Diagnostics & Analytics Dashboard - Plan

**Status:** Phase 1 (Foundation + Run Statistics + Rejection Analysis) implemented 2026-08-03. Phases 2-6 below are scoped but not started.

## Why phased, not built all at once

The full ask spans ten areas: run statistics, rejection analysis, confidence
distributions, user-vs-algorithm agreement, learning statistics, feature
attribution, threshold-change simulation, long-term/cross-session learning,
algorithm comparison, and data export. Several of the later areas (user
overrides, feature attribution, threshold simulation, cross-session trends)
all need the same thing to exist first: a durable, algorithm-agnostic record
of what every ranking run actually did. Building that schema is the one
decision here that is expensive to redo - every later phase either reads
from it or writes more into it - so it comes first, on its own, rather than
being sketched implicitly by whichever phase happened to get built first.

## Phase 1 (done): Foundation, Run Statistics, Rejection Analysis

- `analytics/store.py` - `AnalyticsStore`, a SQLite database
  (`cache/analytics.db`) with three tables: `runs` (one row per ranking run,
  free-form JSON `params`), `run_reject_counts` (`{run_id, reason, count}` -
  any reason string, not an enum), `run_image_metrics` (`{run_id,
  image_path, metric_name, value}` - an entity-attribute-value table, not
  one column per metric). No column anywhere assumes which algorithm, which
  reject reasons, or which metrics exist - see the module's own docstring
  for why that is structural, not just a convention to remember.
- `analytics/capture.py` - `record_run(...)`, a generic function taking
  only primitive shapes (counts, dicts), never a ranking-module type.
  Imports nothing from `ranking`. Never raises - a failure to record
  analytics history must never break a photographer's ranking run.
- `analytics/reports.py` - `run_statistics`, `rejection_analysis`,
  `confidence_distribution`, and CSV export for each - the two report types
  asked for first, plus the raw per-metric values a histogram (or, for now,
  a CSV column) needs.
- Wired into `ranking.classic.ClassicVisionStrategy.rank_folder` (both
  Classic Vision backends share this method) and `rank.rank_folder` (the AI
  model) - every ranking run, from any current entry point, is now
  recorded automatically. A future ranking strategy gets this for free by
  calling `record_run` at the same point its own `rank_folder` already
  writes its CSV/filter report/metrics report.

**What Phase 1 does NOT capture yet, and why:** average processing time and
GPU utilization (asked for in the original spec's Run Statistics list) are
not recorded because no current ranking strategy measures them - adding
them is a small, later change (one more `image_metrics`/`params` field, no
schema migration, per the EAV design), not attempted here to avoid
recording numbers nobody asked a strategy to actually compute.

## Phase 2 (not started): User vs Algorithm

Needs a second event source Phase 1 does not touch: `ReviewSession.
set_review_status` (and any future promote/demote/reorder action) recording
an override event, tied back to **the run that most recently produced a
score for that image** (via `runs.run_id`, looked up by folder + most
recent `started_at` at the time of the override - never by assuming a
strategy). Agreement rate, override rate, and precision@K are aggregate
queries over "override events joined against the run they overrode."

## Phase 3 (not started): Feature Attribution

Needs the per-component score breakdown (`ClassicVisionStrategy.combine`'s
weighted eye/subject/size inputs, or the AI model's equivalent) recorded
per accepted image - already partially available via `run_image_metrics`
for Classic Vision (eye_sharpness/subject_sharpness/subject_size are
already captured there); the AI model would need a decomposition it does
not currently produce (a single opaque score today).

## Phase 4 (not started): Threshold-change simulation

A read-only query over Phase 1's already-recorded `run_image_metrics`:
"how many additional images would `run_reject_counts`'s LOW_VISIBLE_EYE
count include if `eye_confidence_threshold` moved from X to Y" is directly
answerable once enough real runs are on record, by counting metric values
that fall in the changed range - no new capture needed, only a new report
function.

## Phase 5 (not started): Long-term / cross-session learning

Slicing agreement/accuracy by species, focal length, camera body, lens,
month needs EXIF metadata joined against `runs`/override events - a new
capture step (reading EXIF once per image, not currently done anywhere in
this codebase) feeding a new, likely EAV-shaped `image_context` table
alongside the three from Phase 1.

## Phase 6 (not started): Algorithm comparison, interactive dashboard

`AnalyticsStore.list_runs(folder=...)` (Phase 1) already gives "every run
ever recorded for this folder, most recent first" - the raw material for
"how would ranking this folder again today differ" is already there once
enough history accumulates; a proper diff report is a Phase 6 report
function, not a new capture mechanism. The interactive dashboard itself
(vs. today's CSV export) was explicitly deferred by the original ask
("initially CSV... later this can become an interactive dashboard").
