# BioCLIP Multi-Backend Infrastructure - Deliverables

**Date:** 2026-08-03. **Scope:** Reproducibility/isolation/analytics/scalability infrastructure requested after the architecture review (`docs/BioCLIP_Backend_Architecture_Review.md`) - no benchmarking, no model replacement, no BioCLIP 2.5 integration, per explicit instruction. All 8 parts implemented, tested (44 new tests, full suite 1161 passed), and validated end-to-end against real photos through the real Desktop code path (not only mocks).

---

## 1. Architecture summary

Three new layers, each addressing one review finding, wired together without touching how `arrange_by_species`'s core classify-and-file loop works:

```
Desktop "Organize by Species" (main_window.py)
    -> ReviewService.organize_by_species (services.py)
        -> species.classifier.build_classifier(backend, ...)   [unchanged registry]
        -> species.experiment.build_experiment_metadata(...)   [NEW - Part 2]
        -> species.experiment_capture.run_with_analytics(...)  [NEW - Part 4]
            -> species.arrange.arrange_by_species(..., on_result=...)  [+1 optional hook]
                -> species.cache.SpeciesCache (v2 schema)       [FIXED - Part 1]
            -> analytics.capture.record_run(...)                [reused, extended - Part 4]
                -> analytics.store.AnalyticsStore                [extended - Part 4]

desktop.dialogs.analytics_dashboard.AnalyticsDashboard          [NEW - Part 5/6]
    reads only AnalyticsStore's generic accessors
```

**The one deliberate design decision worth calling out:** `arrange_by_species` itself was extended by exactly one optional parameter (`on_result`), not rewritten. The real classify-and-file work and its analytics observation stay two separate concerns - the same separation `ranking.classic.rank_folder` already keeps from `analytics.capture.record_run`. This is also why Top-5 predictions (Part 4) cost nothing extra: `BioClipSpeciesClassifier.classify()` already computes the full per-species similarity vector before collapsing it to the single winning answer - `top_predictions` just keeps a slice of what already existed in memory, no second forward pass.

**The Analytics Dashboard reuses the ranking analytics schema built in a prior session, unmodified in shape.** `AnalyticsStore.category_counts`/`reject_counts` are the same method under two names over the same table - a species run's predicted-species distribution and a ranking run's reject-reason breakdown are structurally identical (`{run_id, label, count}`). Verified, not assumed: a real ranking-shaped run was inserted and rendered correctly in the same dashboard used for species runs (screenshot in §5).

## 2. Files added

| File | Purpose |
|---|---|
| `src/picklikeme/species/experiment.py` | `ExperimentMetadata` + `build_experiment_metadata()` - Part 2 |
| `src/picklikeme/species/experiment_capture.py` | `run_with_analytics()` - Part 4's wrapper around `arrange_by_species` |
| `src/picklikeme/desktop/dialogs/analytics_dashboard.py` | `AnalyticsDashboard` + its three tabs - Part 5/6 |
| `tests/test_species_cache.py` | Part 1: migration + coexistence tests (6) |
| `tests/test_species_classifier.py` *(pre-existing, extended earlier)* | not new this round |
| `tests/test_species_experiment.py` | Part 2 tests (12) |
| `tests/test_bioclip_device_resolution.py` | Part 3 tests (6) |
| `tests/test_bioclip_classify_top_n.py` | Part 4a tests (4) |
| `tests/test_species_arrange.py` | Part 4's `on_result` hook tests (4) |
| `tests/test_species_experiment_capture.py` | Part 4 integration tests (2) |
| `tests/test_analytics_dashboard.py` | Part 5/6 tests (5) |
| `docs/BioCLIP_Infrastructure_Deliverables.md` | this document |

`tests/test_analytics.py` was extended (not added) with 5 new tests for `summary_metrics`/`category_counts`/per-image lookups.

## 3. Files modified

| File | Change |
|---|---|
| `species/cache.py` | Composite `(image_hash, classifier_id)` primary key + automatic v1->v2 migration - Part 1 |
| `species/classifier.py` | `SpeciesPrediction.top_predictions` (new optional field); stale docstring example fixed |
| `species/bioclip_classifier.py` | `device` now defaults to `None` (auto-resolve) at the actual source, not just its callers; session-start logging; `classify()` gained `top_n`; `CLASSIFIER_VERSION` constant |
| `species/cli.py` | `--device` no longer hardcodes `cpu` |
| `species/arrange.py` | `arrange_by_species` gained one optional `on_result` hook (no other behavior change) |
| `desktop/services.py` | `organize_by_species`: `device` default fixed, now calls `run_with_analytics`, gained `analytics_db` (test isolation, mirrors `rank_folder`) |
| `desktop/dialogs/workflow_dialogs.py` | *(from the prior backend-registration round, not this round)* |
| `desktop/main_window.py` | New "Analytics Dashboard…" menu action + handler |
| `analytics/store.py` | New `run_summary_metrics` table; `category_counts` (generic name for `reject_counts`); `image_paths`/`image_metrics` accessors; docstring generalized beyond ranking |
| `analytics/capture.py` | `record_run` gained `summary_metrics` passthrough; docstring generalized |
| `tests/test_desktop_workflow.py` | Updated the one test that monkeypatched the now-superseded call path |

## 4. Migration notes

**`SpeciesCache` v1 -> v2, automatic, on first open of an existing database:**

- **Why invalidation is required, exactly:** SQLite has no `ALTER TABLE ... ADD PRIMARY KEY` - changing a primary key requires recreating the table. This is *not* the same kind of invalidation `CROP_CACHE_VERSION`/`EYE_CACHE_VERSION` bumps use elsewhere in this project (which discard and rebuild); here every existing row is copied forward, none are discarded.
- **What happens automatically:** `species_cache` is renamed to `species_cache_v1`, a new `species_cache` table is created with `PRIMARY KEY (image_hash, classifier_id)`, every row is copied across unchanged, then the old table is dropped - one transaction. Verified against a real copy of the project's own 65-row `cache/species.db`: all 65 rows present afterward, byte-identical field values.
- **Idempotent:** a second open of an already-migrated database is a no-op (checked via a `schema_info` version marker, with a structural fallback check in case an even older, unversioned database is encountered).
- **Separately, not part of this migration:** every one of those 65 existing rows carries a `classifier_id` in the *pre-classifier_id-fix* format (`"bioclip2:bioclip-2:<digest>"`, from before the earlier architecture-review round's fix). They are harmless, inert history - `get()` requires an exact `classifier_id` match, so these rows are simply never served again, the same safe-miss behaviour a version mismatch has always had here. Not touched by this migration because touching them wasn't necessary to fix the actual bug (the primary key), and rewriting historical rows to a new format they were never computed under would be revisionist, not a migration.

## 5. Validation results

**Species caches never overwrite each other - proven, not just tested in isolation.** A real end-to-end run through the actual `ReviewService.organize_by_species` (not a mock) classified 2 real photos with BioCLIP 2, then the same 2 photos with the original BioCLIP - both through the real Desktop code path. Reading the resulting `species_cache` table directly:

```
image_hash                  | classifier_id            | species       | confidence
p1:3f636b51ac788765c...     | bioclip-2:aeb5a3073ad9   | Kingfisher    | 0.9383
p1:3f636b51ac788765c...     | bioclip:aeb5a3073ad9     | Unknown       | 0.3721
p1:6975bbeb660fb3c4f...     | bioclip-2:aeb5a3073ad9   | Common Tern   | 0.9959
p1:6975bbeb660fb3c4f...     | bioclip:aeb5a3073ad9     | Snowy Owl     | 0.7656
```

4 rows for 2 images x 2 backends - exactly as many as expected, neither backend's row missing or overwritten. This is the literal bug the architecture review found, now reproduced fixed on real data.

**CPU/GPU selection works correctly - proven.** The same real run logged `Species classifier ready: ... requested device=None -> execution device=cuda ... GPU=NVIDIA GeForce RTX 5070` with zero device argument passed anywhere in the call chain - confirming the fix holds at the actual entry point Desktop uses, not only in a direct unit test. `inspect.signature` regression tests additionally pin the constructor, `ReviewService.organize_by_species`, and the CLI parser's defaults to `None`, so this specific regression cannot silently return.

**Experiment metadata is persisted correctly - proven.** The same real run's recorded `ExperimentMetadata` included the exact Hugging Face Hub commit SHA (`2957b322090f9cb17ae72c71981c7218a28d81e0` for BioCLIP 2) matching what the architecture review found by manually reading the HF cache days earlier - an independent cross-check, not a value invented for this run.

**Dashboard displays correct information - proven visually**, not just by row counts. Screenshots captured from the real, running dialog (via `QWidget.grab()`, not a mockup) are shown inline in this conversation:
1. Run Summary tab, real data - every `ExperimentMetadata` field, runtime metrics, and mean Top-1..Top-5 confidence, all populated.
2. Species Analysis tab, real data - the actual species distribution and Unknown rate from the real run.
3. Image Inspector tab, real data - per-image Top-1..Top-5 confidences and inference time; **found and fixed a real bug in the process** (see §6).
4. The same dashboard rendering a ranking-shaped (not species) experiment correctly, with reject reasons instead of species names in the identical UI - proof the dashboard is genuinely backend-agnostic, not just species-shaped code with a generic-sounding name.

**Existing functionality remains unchanged - proven.** Full suite: **1161 passed**, up from 1117 before this round (44 new tests, 0 regressions, 0 skipped beyond the two pre-existing model-weight-dependent skips already present before this work).

## 6. Remaining technical debt

- **Recorded image paths can go stale within the same run.** `on_result` fires with an image's path *before* `arrange_by_species` moves that file into its species subfolder. Inspecting the Image Inspector afterward for a `dry_run=False` run may find the file already relocated. Found during validation (§5's Image Inspector screenshot), not left silent - a text fallback ("Original not available... may have moved") now degrades gracefully instead of showing a blank widget, but the underlying path is not rewritten. `sidecar.rewrite_ranking_paths` already solves the equivalent problem for ranking CSVs after Organize; the same pattern would apply here if this becomes a real workflow friction point.
- **The "Species Analysis" tab is generically-shaped but not generically-named.** It correctly renders a ranking run's reject-reason breakdown (proven in §5), but its tab label still says "Species Analysis" regardless of what kind of run is selected - a minor, cosmetic mismatch, not a functional one.
- **Errors are counted, not detailed, in analytics.** `run_summary_metrics.errors` records how many images failed; the actual error text stays on `SpeciesArrangeResult.failures` (returned to the caller, shown in the existing UI success message) rather than being duplicated into the analytics database, which is optimized for numeric/categorical tracking, not a log store. Revisit if a future need specifically wants historical error text queryable per experiment.
- **No cross-run comparison logic exists yet, deliberately** - agreement rate, precision/recall, confusion matrices are the benchmark framework's job, explicitly out of scope for this round.
- **`min_confidence`/`prompt_template` are the only thresholds captured in `ExperimentMetadata.thresholds` today** - correct for the current single-model-family setup; a structurally different future backend (see the architecture review's §3/§10 findings on closed-set models) will need its own threshold shape, which `build_experiment_metadata`'s `getattr(..., None)` pattern already tolerates without crashing, but has not been exercised against a real non-BioCLIP backend.
