# Algorithm Runs and Result Ownership

Concise developer reference for a set of rules that are easy to violate by
accident and hard to notice when broken - a UI panel silently showing a
different algorithm's result than the one currently selected does not throw
an exception, it just quietly lies. This document exists so that does not
happen again.

## 1. Algorithm Run

A **run** is one ranking strategy's complete pass over a folder. Every run
is identified by its **strategy ID** (`ranking.base.StrategyInfo.strategy_id`,
e.g. `"classic-vision-fusion-mammals"`) - the one identifier that is unique
per registered strategy and stable across every artifact that run produces.

What one run owns, all keyed by the same strategy ID:

| Artifact | Where | Written by |
|---|---|---|
| Ranking CSV | `sidecar.strategy_ranking_path(folder, strategy_id)` | `write_results_csv` |
| Filter report | `<strategy_id>_filters.json` | `ranking.classic.write_filter_report` |
| Metrics report | `<strategy_id>_metrics.json` | `ranking.classic.write_metrics_report` |
| Eye-detector result (per image) | `eyes.cache.eye_cache_path(cache_dir, image, strategy_id)` | `eyes.cache.save_eye_detection` |
| Provenance (single, most-recent-of-any-strategy) | `sidecar.run_metadata_path` (`run.json`) | `sidecar.write_run_metadata` |

Every one of these (except `run.json`, see below) is a **separate file per
strategy** - two strategies never share a slot, and running one never
touches another's files.

## 2. Latest Run / "Algorithm Ran Last"

`run.json` is the **one** shared file: every strategy's `rank_folder`
overwrites it with its own `strategy` id and a timestamp when it completes.
It is provenance, not a growing history - reading it answers "which
strategy most recently completed a run on this folder", nothing more.

`ReviewSession.latest_run_strategy()` is the single, centralized
implementation of "Algorithm Ran Last": it reads `run.json`'s `strategy`
field and validates it against `sidecar.discover_strategy_rankings` (in
case the CSV was since deleted), falling back to the AI model only for a
folder that has genuinely never been ranked.

This is used in exactly two places, both calling the same method - there is
no second implementation anywhere:

- `ReviewSession.open_folder` seeds `burst_strategy` with it.
- The desktop Color Source combo's `ALGORITHM_RAN_LAST` sentinel
  (`main_window.py`) resolves through it **on every use**
  (`MainWindow._resolve_color_source`), not once at selection time - so it
  keeps tracking the true latest run even as new rankings complete, and a
  photographer can return to "whichever is latest" after manually picking a
  specific strategy without needing to know its name.

Picking a strategy **by name** from the same combo pins to it; only the
sentinel re-resolves dynamically. Both cases end up in
`ReviewSession.burst_strategy`, which is what Grid coloring, Apply Cutoff,
Filtering, and Burst ranking all read - see section 4.

## 3. Cache retention policy

PeakPic keeps **exactly one (the latest) completed run per strategy**, never
a history. This falls directly out of every artifact in the table above
being a fixed path computed from `(image, strategy_id)` (or just
`strategy_id` for the CSV/reports): re-running a strategy overwrites its own
file in place. There is nothing to prune - there is only ever one file to
begin with.

Per-file writes are atomic (temp file + `os.replace`) throughout the crop
cache, detection cache, and eye cache - a process killed mid-write leaves
the previous valid file untouched, never a truncated one masquerading as
current. This is enforced for the review preview/thumbnail caches too (see
section 6) after a real bug: non-atomic writes there left permanently
corrupt cached JPEGs that `QPixmap` failed to decode silently.

**Known limitation:** this atomicity is per-file, not per-run. A crash
partway through a multi-hundred-image ranking run can leave some images'
eye-cache sidecars updated to the new run while others (later in the
iteration order) still hold the previous run's data, and the ranking CSV
itself (written once, at the end) still reflects the old run until the
whole pass finishes. A fully transactional "stage the whole run, then swap
it in atomically" design was considered and deliberately not built - the
per-image atomicity above already prevents corruption, and the added
complexity was not judged worth it for a failure mode (a crash mid-run)
that a re-run fully repairs.

## 4. Grid / Loupe / Filter / Cutoff / Color source of truth

`ReviewSession.burst_strategy` is the **one** place "which run is currently
selected" lives. Everything downstream reads it, directly or through
`ImageItem.algorithm_suggestion` (computed by `ReviewSession.suggestions_for
(burst_strategy)`):

- Grid coloring (`ThumbnailCardDelegate`, via `MainWindow._resolve_color_source`)
- Apply Cutoff (`MainWindow._apply_cutoff`)
- Filtering (`algorithm_suggestion`-based filters)
- Burst ranking (`ReviewSession.burst_info`)

None of these have their own, independent notion of "which strategy" -
there used to be a second, UI-local `MainWindow._color_source` that could
drift out of sync with `burst_strategy` (the combo showing one thing while
the session used another); `_sync_color_source_from_session` /
`_resolve_color_source` is what keeps them locked together now, called on
every folder open and every completed ranking run.

## 5. Loupe: Elements / Boxes / Preview independence

The Loupe's image display is **independent of algorithm success**. It
reads pixels (`review.thumbnails.review_preview`) and navigates a
caller-provided ordered list; it has no dependency on any ranking result
existing at all. A missing/failed/low-confidence detection never prevents
an image from opening - only the optional Boxes/Elements overlays have
nothing to draw.

Boxes and Elements are independently toggleable (`LoupeDialog.
_refresh_detection_overlay`) - both, either, or neither can be active; one
is never forced off by the other.

Both read `ReviewService.eye_keypoints(image_path, strategy_id=...)`,
defaulting to the current `burst_strategy` - **never** "whichever detector
happened to run last." This is enforced structurally, not by a runtime
check: `eyes.cache` is keyed by `(image, strategy_id)` (section 1), so
asking for a strategy that never ran on an image simply returns `None`;
there is no shared slot for a wrong answer to hide in.

## 6. Preview/thumbnail cache corruption (P0 fix, keep this)

`review.thumbnails.review_preview` / `analyzer.contactsheets.build_thumbnail`
write atomically and treat a 0-byte cached file as a miss, not a hit.
`LoupeDialog._load_current` checks `QPixmap.isNull()` after loading a cached
preview and, on failure, deletes the stale file and regenerates once before
giving up (with a visible warning, never a silent blank frame). Do not
remove either half of this - the write-side fix prevents new corruption,
the read-side fix self-heals anything corrupted before it existed.

## 7. Domain-aware pipeline (Birds / Mammals / Birds+Mammals)

See `eyes/domains.py` for the authoritative module docstring. Summary: a
`DomainProfile` names which detector IDs and default fusion weights belong
to a Ranking Mode; `eyes.fusion.FusionEyeDetector` itself is domain-agnostic
and never hard-codes a species. `ranking.combined` classifies a Burst's
domain once (not per image) via a lightweight CLIP zero-shot check and
routes to the matching profile.

## 8. Crop cache ownership - deliberately NOT per-strategy (known limitation)

Unlike the eye-detection cache (section 1), the **subject/bird crop cache**
(`bird_crop.crop_cache_path`/`detections_cache_path`, written by
`preprocess.build_cache`) is still a single, global cache keyed only by the
resolved image path - every registered strategy shares it.

This is safe **today** because every registered strategy (Birds, Mammals,
Fusion, Combined alike) uses the identical object detector
(`fasterrcnn_resnet50_fpn_v2`) and, by default, identical `CropParams` -
genuinely the same crop, legitimately shared as an optimization (see
`eyes.cache`'s own module docstring for the "physical sharing is fine when
the config is truly identical" principle this follows).

If a strategy is ever configured with different detection/crop thresholds
than what is cached, this is **not silent**: `preprocess.build_cache`
already raises `CropCacheVersionMismatch` and refuses to proceed without an
explicit `--force`/rebuild - the photographer is told, not silently served
a wrong crop. What it does NOT yet do is let two *different, incompatible*
configurations coexist side by side the way the eye cache now lets two
strategies' eye results coexist.

**Not fixed in this pass** - deliberately, given no active bug was found
here (every current strategy really does share one config) and the
existing mismatch guard already prevents silent corruption. Restructuring
`bird_crop.py`'s cache-path scheme to be per-(strategy-config) would mean
threading a config identity through `preprocess.build_cache` and every one
of its several callers - a larger change than this pass's scope justified
without a demonstrated bug to fix. If a future strategy genuinely needs a
different object detector or crop parameters from the others, this is the
first place that will need to change, following the same pattern section 1
already established for the eye cache.

## What to check before changing anything in this area

- Does this artifact need a `strategy_id` in its cache key? (If two
  different strategies could ever produce a different answer for the same
  image, yes.)
- Does this UI element read `ReviewSession.burst_strategy` (or
  `MainWindow._resolve_color_source()`), or does it have its own notion of
  "current strategy"? It should not have its own.
- Does this write happen atomically (temp file + replace)? Every cache
  writer in this codebase should.
