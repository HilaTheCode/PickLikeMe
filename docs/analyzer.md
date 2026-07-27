# Evaluation & Analysis module

`picklikeme analyze` measures how well a trained model reproduces your own
keep/reject decisions, explains where it fails, and says what to do next.

```bash
picklikeme analyze \
    --ranking rankings_20260726-091500.csv \
    --selected "D:\shoot\keep" \
    --rejected "D:\shoot\drop" \
    --output analysis/
```

Open `analysis/report.html` when it finishes.

---

## Contents

- [Design rules](#design-rules)
- [Architecture](#architecture)
- [Inputs](#inputs)
- [What it produces](#what-it-produces)
- [CLI reference](#cli-reference)
  - [Entry points](#entry-points)
- [Configuration files](#configuration-files)
- [Metrics reference](#metrics-reference)
- [Annotation knowledge base](#annotation-knowledge-base)
- [Detector-box thumbnail overlay](#detector-box-thumbnail-overlay)
- [Extension guide](#extension-guide)
- [Usage examples](#usage-examples)

---

## Design rules

**Read-only.** The analyzer never writes to checkpoints, the crop cache, source
images, training data or ranking output. The only files it creates live under
`--output`. A test asserts this by comparing mtimes of every input before and
after a full run.

**One-way dependency.** The analyzer imports from the pipeline
(`evaluate.roc_auc`, `bird_crop.crop_cache_path`, `config`); nothing in
training, preprocessing or ranking imports the analyzer. Deleting the whole
`analyzer/` package would leave the pipeline working.

**Compute once, render many.** `run_analysis()` produces an `AnalysisResult`.
Every renderer — text, JSON, CSV, charts, contact sheets, HTML — consumes that
object and computes nothing of its own, so the accuracy on an HTML summary card
is by construction the accuracy in the JSON.

**No unevidenced advice.** Every suggestion carries the statistic that produced
it and the threshold that statistic had to cross.

---

## Architecture

```
src/picklikeme/analyzer/
    model.py          RankedImage, MatchedImage, Outcome     (no I/O, no deps)
    config.py         AnalysisConfig - every tunable
    io.py             ranking loading w/ field auto-detection, folder scanning
    matching.py       C1  ranking <-> ground truth join
    metrics/
        base.py       C16 Metric ABC, registry, MetricSet
        classification.py  C2
        ranking.py         C3
        calibration.py     C6
    thresholds.py     C4  sweep + recommendation, C5 confusion matrix
    errors.py         C7/C8/C9 mistakes, hard cases, borderline; C6 histograms
    suggestions.py    C13 evidence-backed rules
    comparison.py     C14 two-run diff
    visualization.py  C12 matplotlib charts
    contactsheets.py  C10 sheets, C15 thumbnail cache + parallel generation
    analysis.py       orchestrator -> AnalysisResult
    reports/
        text.py       console / report.txt
        html.py       C11 offline HTML
        __init__.py   JSON + per-category CSVs
    cli.py            C17
```

Note on layout: the brief suggested `src/analyzer/`. It lives at
`src/picklikeme/analyzer/` instead so it is importable as part of the installed
package — a sibling top-level `analyzer` package would not ship with
`pick-likeme` and could collide with any other `analyzer` on the path.

### Data flow

```
ranking CSV ──> io.load_ranking ──> [RankedImage]
                                        │
selected/ rejected/ ──> matching.match_dataset ──> MatchResult ([MatchedImage])
                                        │
                    ┌───────────────────┼────────────────────┐
                 metrics            thresholds            errors
                    └───────────────────┼────────────────────┘
                                 AnalysisResult
                                        │
        text · JSON · CSV · charts · contact sheets · HTML
```

---

## Inputs

### Ranking file

Any CSV with a path-like column and a score-like column. Columns are
**auto-detected** from these aliases (first match wins):

| Field | Accepted column names |
| --- | --- |
| `image_path` | `image_path`, `path`, `filepath`, `file_path`, `full_path`, `filename`, `file`, `name` |
| `score` | `score`, `prediction`, `predicted_score`, `pred`, `value` |
| `rank` | `rank`, `position`, `ranking`, `index` |
| `label` | `label`, `ground_truth`, `truth`, `actual`, `y_true` |
| `probability` | `probability`, `prob`, `predicted_probability`, `p`, `confidence_score` |
| `predicted_class` | `predicted_class`, `prediction_class`, `y_pred`, `predicted_label` |
| `confidence` | `confidence`, `certainty` |

Only path and score are required. Behaviour when fields are absent:

- **no `rank`** — assigned by descending score.
- **no `probability`** — the score is used if it lies in `[0, 1]` (PickLikeMe
  regresses toward 1.0/0.0, so it already is one). Outside that range no
  probability is invented and calibration metrics report *not applicable*
  rather than a fabricated number.
- **no `label`** — ground truth must come from the folders.

PickLikeMe's own metrics preamble is skipped automatically, and chunked outputs
(`rankings.csv`, `rankings_1.csv`, …) are picked up by *computing* successive
names — never by globbing, so an unrelated `rankings_final.csv` is not swept in.

### Ground-truth folders

`--selected` and `--rejected` are scanned recursively for RAW and common image
formats. Both are optional: with neither, labels from the ranking file are used,
which makes the analyzer work directly on a training run's `training_results_*.csv`.

### Matching

Tried in widening stages, and the report states how many images each stage
resolved:

1. `resolved_path` — exact, after resolution.
2. `case_insensitive_path` — Windows re-casing.
3. `unique_path_suffix` — the folder moved drive or parent (`D:\old\shoot\a.nef`
   → `E:\archive\shoot\a.nef`), accepted only when all candidates agree.
4. `unique_filename` — last resort, only when the filename is unique across both
   folders.

Anything unresolved becomes `Outcome.UNKNOWN`, is excluded from every metric,
and is warned about. It is never counted as a negative — that would inflate
specificity with images you never judged.

---

## What it produces

Every CLI run (`picklikeme analyze` / `python -m picklikeme.analyzer`) writes to
its own timestamped folder, so consecutive full analysis reports never
overwrite each other: `analysis` becomes `analysis_20260727-093015` (the run's
*start* time, stamped once, before any file is written). A meaningful `--output`
name is preserved as a prefix: `--output analysis/nightly` becomes
`analysis/nightly_20260727-093015`. On the rare chance two runs would compute the
identical stamp (both finish parsing args within the same second), a numbered
suffix (`_1`, `_2`, ...) is added rather than colliding. The resolved path is
printed as the first line of output, and the `picklikeme annotate --output ...`
command printed afterward already has it filled in — nothing to note down.

This stamping happens only in the CLI. A library caller that builds
`AnalysisConfig` directly (tests, notebooks, a script calling `run_analysis()`)
gets exactly the `output_dir` it specified, unstamped — see
[Configuration files](#configuration-files) below.

```
analysis_20260727-093015/
    report.html                 interactive, offline, light + dark
    report.txt                  full text report
    analysis.json               machine-readable, for CI diffing
    tables/*.csv                per-category mistake lists
    charts/*.png                9 charts
    contact_sheets/*.png        labelled grids, paginated
    thumbnails/                 cache (safe to delete)
```

---

## CLI reference

### Entry points

Three equivalent ways to run any subcommand (`analyze`, `annotate`,
`build-manifest`, ...):

| Form | Requires | Notes |
| --- | --- | --- |
| `picklikeme <command> ...` | `pip install -e .` **and** that environment's `Scripts`/`bin` directory on `PATH` | The installed console script (`[project.scripts]` in `pyproject.toml`) |
| `python -m picklikeme <command> ...` | The package importable by whichever `python` you run | Works regardless of `PATH`; the form every command printed by the tool itself uses |
| `python -m picklikeme.analyzer ...` | Same | Older alias for `analyze` specifically (predates the unified `picklikeme` dispatcher); does not cover `annotate` |

`picklikeme <command>` is the shortest to type once set up, but it silently
stops working the moment the console script isn't on `PATH` - easy to hit in
this project, which keeps two virtualenvs (`.venv` CUDA, `.venv-1` CPU-only),
each with its own `picklikeme.exe`, only one of which can be active at a time.
`python -m picklikeme` sidesteps that: it only needs the interpreter you
invoke to have the package installed, never `PATH`.

For that reason, every command this tool prints for you to run next (the
`annotate` follow-up after `analyze`, error messages that suggest a fix) is
generated with `sys.executable -m picklikeme ...` - the exact interpreter
already running, named explicitly - rather than the bare `picklikeme` form.
Copy it as printed; it is guaranteed to be the right one for the environment
that produced it, even if that is `.venv-1` and your shell's `PATH` currently
points at `.venv`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ranking` | required | Ranking CSV (chunks auto-discovered) |
| `--selected` / `--rejected` | none | Ground-truth folders |
| `--output` | `<repo>/analysis` | Base directory for this run's reports; the run's start date/time is appended (see [What it produces](#what-it-produces)) |
| `--config` | none | JSON config; explicit flags override it |
| `--title` | "PickLikeMe model analysis" | Report title |
| `--threshold` | `0.5` | Decision threshold |
| `--optimize-for` | `f1` | `f1`, `balanced_accuracy`, `accuracy`, `precision`, `recall`, `mcc`, `youden_j` |
| `--threshold-steps` | `101` | Points in the sweep |
| `--borderline-low` / `--borderline-high` | `0.45` / `0.55` | Uncertainty band |
| `--max-examples` | `60` | Rows per table / sheet |
| `--thumbnail-size` | `400` | Preview edge (px); previews show the whole frame, so this is also the contact-sheet tile size |
| `--thumbnail-workers` | `8` | Parallel thumbnail threads |
| `--compare-ranking` | none | Second ranking → comparison mode |
| `--baseline-label` / `--compare-label` | `baseline` / `candidate` | Names in comparison output |
| `--no-html`, `--no-charts`, `--no-contact-sheets` | off | Skip an output |
| `--quiet` | off | Print only the summary |
| `-v/--verbose` | off | Debug logging |

Contact sheets dominate runtime. `--no-contact-sheets` makes a run near-instant.

---

## Configuration files

```json
{
  "ranking_path": "rankings.csv",
  "selected_root": "D:/shoot/keep",
  "rejected_root": "D:/shoot/drop",
  "output_dir": "analysis/nightly",
  "report_title": "Nightly regression",
  "threshold": 0.55,
  "optimize_for": "recall",
  "borderline_low": 0.4,
  "borderline_high": 0.6,
  "max_examples": 100,
  "contact_sheets": false
}
```

```bash
picklikeme analyze --config nightly.json --threshold 0.6   # flag wins
```

Unknown keys are rejected rather than ignored, so a typo fails loudly.

`output_dir` here is still just a base name when reached through the CLI — the
run timestamp is appended on top of whatever this file or `--output` says, the
same as any other CLI invocation. It is only taken literally, unstamped, when
`AnalysisConfig` is built directly by library code (`run_analysis(config)`
called from Python without going through `analyzer.cli.run()`).

---

## Metrics reference

**Classification** (at the threshold) — counts, accuracy, precision, recall,
specificity, balanced accuracy, F1, FPR, FNR, NPV, MCC, Youden's J, ROC AUC,
PR AUC.

**Ranking** — precision@ and recall@ top 1/2/5/10/20/30%, average precision,
mAP (averaged over folders, i.e. per shoot), nDCG, nDCG@10%, Spearman,
Kendall tau-b, rank overlap@10%, average/median rank displacement, max
disagreement, ranking agreement.

**Calibration** — ECE, MCE, Brier score, mean confidence, overconfidence.

Two notes worth knowing when reading the numbers:

- **Correlations are tie-aware.** Ground truth is binary, so ties are
  pervasive. Spearman between distinct scores and binary truth is capped below
  1.0 — with two keepers and two rejects the perfect value is 0.894, not 1.0. An
  implementation that reports 1.0 there is wrong.
- **Undefined is `None`, not `0.0`.** Precision with no positive predictions is
  undefined; reporting 0.0 would claim the model was wrong rather than silent.

---

## Annotation knowledge base

A false negative is an image you deliberately kept and the model rejected; a
false positive is the reverse — the model kept an image you rejected. Both are
the model disagreeing with you, and both are the most valuable failures to
understand, so the analyzer lets you record **why** for either, using the same
three fields, the same fixed vocabulary and the same editor. That symmetry is
deliberate: it is what makes a false-negative diagnosis and a false-positive
diagnosis directly comparable, not two different measurements.

**The annotations are yours.** Nothing in the codebase infers, suggests or
pre-fills a value. There is no model, heuristic or default in the path: the
report renders what the database holds and stores exactly what you selected.
The schema itself does not record which category an annotation came from - a
record is a diagnosis of the image, not of the run that flagged it, so the
false-negative/false-positive split shown in a report is computed by asking the
store about two path lists, one per category, not by a stored label. (An image
cannot be both at once: false negative and false positive are mutually
exclusive by definition against the ground truth, so there is never an
ambiguous record to split.) The separate detector-box thumbnail overlay (below)
applies more broadly still — report-wide, not just to annotatable images.

**They never touch the metrics.** Annotations load after every metric, threshold
sweep and suggestion is already computed, and `test_annotations_never_change_any_metric`
asserts that every metric, the confusion matrix, the recommended threshold and
the suggestion list are identical with and without a populated database.

### Recording a diagnosis

Saving needs a local endpoint — a `file://` page cannot write to SQLite:

```bash
picklikeme analyze --ranking rankings.csv --selected keep/ --rejected drop/
picklikeme annotate --output "analysis_20260727-093015/"   # serves on 127.0.0.1:8756
```

The `analyze` command prints the exact command to run, with the real
(timestamped) directory already filled in — copy that line rather than
retyping it, since it also uses whichever invocation is guaranteed to work in
*your* environment (see [Entry points](#entry-points) below: it is not always
literally `picklikeme annotate ...`). Or skip the copy-paste in one step:
`picklikeme analyze ... --serve`.

Each false negative and each false positive gets a panel with its thumbnail
(with detector boxes drawn on it, if any were resolved — see
[Detector-box thumbnail overlay](#detector-box-thumbnail-overlay) below),
score, confidence, rank, displacement and which way you and the model
disagreed (`you kept it` / `you rejected it`), an **Edit** button, three
dropdowns and **Save**. Opened straight from disk the report still *shows*
existing annotations, with a banner explaining that editing needs `annotate`.

The two categories get their own report sections - "False negatives -
annotate why they were missed" and "False positives - annotate why they were
kept" - each with its own filter bar and its own annotated-count. Filtering one
category never affects the other's panels.

Example of a completed annotation:

```
Crop Quality:              [ Too Small          ▾ ]
Image Quality:             [ Good               ▾ ]
Agree with Model Decision: [ No                 ▾ ]
```

### Three independent fields

Every field is a **closed vocabulary with no free-text option**, because the
point of the knowledge base is data that can be counted. A growable tag list
makes frequencies incomparable over time: "Backlit" and "backlighting" become
two rows in a breakdown but one phenomenon.

The three axes are independent — a technically fine crop of an out-of-focus
bird and a badly placed crop of a perfectly sharp one are different diagnoses.

| Field | Values | Question it answers |
| --- | --- | --- |
| **Crop Quality** | `Good`, `Too Small`, `Wrong Location`, `Too Large` | How good is the crop the detector chose, regardless of the image quality |
| **Image Quality** | `Good`, `Missing Eye`, `Out of Focus`, `No Relevant Subject`, `Group Scene` | How good is the photograph itself, regardless of the crop |
| **Agree with Model Decision** | `Yes`, `No` | Having looked at it, was the model right about this image |

`No Relevant Subject` means there is nothing meaningful to evaluate — the bird
is too small, heavily occluded, or effectively absent. `Group Scene` is for
several subjects together, judged as a scene rather than a single bird's pose
or sharpness.

Every field is optional and independent: any one of them alone is enough to keep
the record, which is deleted only when all three are cleared. A value outside a
vocabulary is **refused** (`InvalidAnnotationValue`, HTTP 400), never stored, so
the counts stay trustworthy.

### Records written before this redesign

Databases annotated under the earlier scheme (a growable category checklist, a
primary-failure-cause radio and a free-text notes box) keep everything they held.
That content is shown read-only beside the image and reported as a count in the
summary, but it is **not** auto-mapped onto the three fields and never enters a
breakdown: guessing that an old `Subject too small` tag meant Crop Quality
`Too Small` would invent an answer you never gave and then count it. Re-annotate
those images to bring them into the statistics.

### False Negative Summary / False Positive Summary

One report section per category, identical in layout, each showing one
frequency breakdown per field, the most common whole-record combinations
(`crop / image / agree`, with `(unset)` for unanswered fields), a card counting
how often you disagreed with the model, recently annotated images, and
everything not yet annotated. Each panel list filters by an exact value on any
field and by annotated / not annotated - scoped to its own category, so a
false-negative filter never hides a false-positive panel or vice versa.

Because the two summaries are computed identically (same `summarise()` call,
once per category), the numbers are directly comparable: if false positives
show far more `Crop Quality: Too Large` than false negatives, that is a real
signal about the detector's behaviour, not an artefact of different counting.

### Storage

| | |
| --- | --- |
| Location | `<project>/annotations/false_negatives.db` (`--annotations-db` to move) |
| Why there | Outside every output directory — those are per-run and get replaced; a knowledge base must outlive them |
| Tables | `annotations_v2`, `annotation_categories_v2`, `categories`, `identity_cache`, `unmigrated_v1`, `schema_info` |
| Per record | `image_hash` (identity), filename, original path, capture datetime if readable, timestamps, the three fields, plus any pre-redesign categories / cause / notes |
| Journal | WAL, so generating a report never blocks the UI writing |
| Written | Only this database. Never a ranking, checkpoint, dataset, image or report |

### Identity

Annotations are keyed on **content identity** — `identity.image_identity()`,
which returns `"p1:<sha1>"` computed from the file's size plus its first and last
512 KB. Filename, original path and capture time are stored as display metadata
and are **never** matched on.

That is what makes a diagnosis follow the image through:

| | Path digest | Content identity |
| --- | --- | --- |
| folder renamed | ✗ lost | ✓ follows |
| archive reorganised | ✗ lost | ✓ follows |
| moved to another drive | ✗ lost | ✓ follows |
| dataset relocated *and* file renamed | ✗ lost | ✓ follows |
| same filename, different image | ✗ could mismatch | ✓ correctly distinct |

There is **no fallback**. If identity cannot be established (the file is
missing, unreadable or empty) the analyzer reports the condition, the report
lists those images as un-annotatable, and the API answers `409` with
`identity_unavailable`. Nothing is stored against a guess: attaching a diagnosis
to the wrong image is worse than losing it.

Identity is memoised in the database against `(path, size, mtime)`, so a repeat
run over an unchanged archive does no I/O to resolve it.

#### Why a partial digest — design review

Measured on this project's own archive (52 MB mean NEF, images on the SATA HDD):

| | |
| --- | --- |
| Hash throughput in RAM | sha1 2280 MB/s · **sha256 4184 MB/s** · blake2b 1011 MB/s |
| Current scheme | 17.6 ms/image → **0.27 h** for 55k images |
| Hypothetical full-file sha1 | 262.6 ms/image → **4.01 h** for 55k images |
| Split of the full-file cost | **6.2% hashing, 93.8% reading** |

**Identity is I/O-bound, not CPU-bound**, and that settles the algorithm
question. A faster primitive optimises the 6% and leaves the 94% alone. The 15x
saving comes from reading 1 MB instead of 52 MB — the amount read, not the hash.

On BLAKE3 specifically: it is not installed, so adopting it means a new
dependency; its advantage is raw throughput, which is the part that does not
matter here; and it would require re-keying every stored annotation. Note also
that sha256 is already *faster* than sha1 on this CPU (SHA extensions), so even
"use something stronger" is not a performance argument.

Collision resistance is not the binding constraint either. There is no adversary
crafting a RAW to collide, and 160 bits over 55k items is astronomically safe.
The residual risk belongs to the *partial read*, not the primitive: two files
would have to agree on size **and** the first 512 KB **and** the last 512 KB, and
a camera RAW's head carries a unique EXIF timestamp, body serial and frame
counter. Swapping in sha256 would not reduce that risk by any amount.

**The one real weakness**, stated plainly: a tool that rewrites metadata *inside*
the RAW rather than into a sidecar `.xmp` changes the head, changes the identity,
and orphans that image's annotation. Uncommon — Lightroom writes sidecars for
NEF/ARW — and the alternatives are worse: hashing only the tail weakens identity,
and extracting just the image payload needs format-specific parsing per camera.
The `p1:` prefix plus the store's migration machinery means this is revisitable
without data loss.

**Conclusion: left unchanged**, on the measurements above rather than on inertia.

#### Two kinds of key, one module

`picklikeme/identity.py` holds both, so the distinction cannot be missed:

- `cache_key(path)` — digest of the **resolved path**. Correct for *derived
  artifacts*: a crop belongs to the file as found, and if the file moves the crop
  is simply rebuilt. This is what the crop cache has always used, unchanged.
- `image_identity(path)` — digest of the **content**. The canonical answer to
  "which image is this", for anything that must outlive a filesystem layout.

Nothing else in the codebase should introduce a third scheme.

### Migration from path-keyed annotations

Databases written before this change keyed annotations on a path digest. Opening
one with the current code migrates it **automatically**:

- every v1 record whose file can be found is re-keyed to content identity, with
  categories, notes and `created_at` preserved (as legacy content — see
  [Records written before this redesign](#records-written-before-this-redesign));
- two v1 records that turn out to be the same image are **merged** (categories
  unioned, both notes kept) rather than duplicated;
- records whose file cannot be found are **parked**, not dropped — they are kept
  verbatim in `unmigrated_v1` with the reason, reported in every run, so nothing
  is lost;
- it is idempotent: migrated rows leave the v1 table, so reopening is a no-op.

A second, unrelated kind of upgrade also runs automatically on open: an existing
v2 database from before the three fields existed gains each missing column via a
guarded `ALTER TABLE` (checked with `PRAGMA table_info` first, so it is silent
and idempotent) — every existing row simply reads back with all three unset until
you answer them.

### Server

`picklikeme annotate` binds **127.0.0.1 only** — never `0.0.0.0`. It has no
authentication, so it must not be reachable from the network. It serves only
files inside the analysis directory (paths are resolved then checked, so `..`
cannot escape), and the only write endpoint is `POST /api/annotations`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Probe; the page uses it to decide whether Save is available |
| `GET /api/fields` | The three fixed vocabularies and their display labels |
| `GET /api/annotations` | All records, to refresh the page |
| `POST /api/annotations` | `{image_path, crop_quality, image_quality, agree_with_model_decision}`; an out-of-vocabulary value is `400`, not stored |
| `GET /source?path=...` | Streams an original image; used so a *served* report can still open source files (browsers block navigating from a served page straight to a `file://` link). Confined to the dataset roots recorded in this report's own `analysis.json` — nothing outside it is reachable |

Built on `http.server` and `sqlite3` from the standard library — no new
dependency for a tool that runs for a few minutes at a time.

### Intended use

The knowledge base is a long-term record of *why* valuable images are missed, to
guide the detector, crop generation, the ranking model and the training set. If
"Subject too small" dominates, the crop strategy is the problem; if "Detector
mistake" dominates, detection is; if "Action shot" dominates, the training set
lacks dynamic poses. That is a judgement for you to make from the evidence — the
analyzer counts, it does not conclude.

---

## Detector-box thumbnail overlay

Every thumbnail in the report — contact sheets and HTML tables alike, whatever
category the image falls in (false negative, false positive, true positive,
top-ranked, ...) — is drawn with the detector's boxes when a detection record
was resolved for it: **solid green** is the box that became the crop the model
actually scored, **dashed amber** are other detections the model passed over,
and **red** means nothing was detected at all (the model saw the whole frame).
A legend explaining this appears once near the top of the report, only when at
least one overlay actually exists. An image with no resolved detection record
keeps its plain thumbnail — there is no fallback that invents a box.

This is diagnostic display only, like the annotations: `--no-detect-missing-boxes`
and comparing metrics with and without it enabled are both covered by tests
that assert every metric stays bit-identical either way.

### Previews are always the whole frame

Every thumbnail shows the **entire original image**, resized to fit and
letterboxed into a square — never cropped, and never the cached bird crop.

Two reasons. Boxes are recorded in full-frame coordinates, so a crop-based
preview draws every box in the wrong place. And a crop cannot show what a
photographer most needs to see, which is whether the detector chose the wrong
region to begin with.

Cost is kept down without cropping: RAWs use their embedded full-frame JPEG
preview (milliseconds, versus about a second for a demosaic), falling back to a
demosaic only for the rare RAW that has none. Thumbnails are then cached under
`thumbnails/`, keyed by source path, size **and a cache version** — bumping that
version is how a change like this retires every stale entry without deleting
anything.

### Box geometry

Boxes are recorded in full-frame pixel coordinates against the `source_size`
stored with them, and the preview is that same frame scaled by
`min(size/width, size/height)` and centred in a square. So a box is drawn at
`offset + coordinate × scale` and nothing else — there is no second crop, resize
or normalisation anywhere in the chain to get out of step.

Measured on the 58 false negatives of a real run: every edge of every selected
box lands within 3 px of that formula on a 400 px preview, and the residual is
the stroke width itself. `tests/test_fn_overlay.py::BoxGeometryTests` pins the
transform for landscape, portrait and corner cases.

Because the subject is now a small fraction of the frame, the drawing adapts to
the box rather than the canvas: the outline is capped at a quarter of the box's
shorter side, dash length scales with the box, and a label wider than its box is
dropped (on a small box the text sprawls over neighbouring detections and reads
as labelling those). `--thumbnail-size` defaults to **400** for the same reason —
at the old 200 a distant bird landed in ~10 px and its box was a solid blob.

### Where the boxes come from

1. **The record `picklikeme preprocess` wrote** beside the cached crop, if the
   detector already ran during preprocessing — free, no extra inference.
2. **The analyzer's own cache** (`analyzer.detections.DetectionCache`, a
   separate SQLite database from the annotation knowledge base — so
   `--no-annotations` never touches it and vice versa), keyed by content
   identity so it survives a moved file.
3. **One detection pass**, only for images in neither of the above, and only for
   the images actually shown in this report (bounded by `--max-examples` and the
   number of contact-sheet categories) — never for the whole dataset. Every
   result is cached, so it costs once per image, ever.

`--no-detect-missing-boxes` restricts the analyzer to steps 1 and 2, so it never
runs the detector itself.

### Cost

Extending the overlay from false negatives only to every thumbnail means step 3
can now run for a proportionally larger set of images the first time this is
used against a crop cache built before preprocessing recorded detections (an
older cache). Re-runs over the same images are free regardless — the cache in
step 2 is permanent. `--no-detect-missing-boxes` is the escape hatch if the
first-run cost matters more than the boxes for images preprocessing never
recorded.

---

## Extension guide

### Add a metric

Create one file in `analyzer/metrics/`. Nothing else — no registry edit, no
report edit. Discovery imports every module in the package, and subclassing
registers the class.

```python
# analyzer/metrics/my_metric.py
from typing import Sequence
from ..model import MatchedImage
from .base import Metric, counts_of, safe_divide


class KeepRate(Metric):
    name = "keep_rate"
    description = "Share of images the model would keep"
    category = "classification"
    higher_is_better = True
    fmt = "{:.3f}"
    sort_key = 50

    def applies_to(self, images: Sequence[MatchedImage]) -> tuple[bool, str]:
        if not images:
            return False, "no matched images"
        return True, ""

    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        counts = counts_of(images)
        return safe_divide(counts.predicted_positive, counts.total)
```

It now appears in the text report, the JSON, and the HTML metrics table.

Use `applies_to` to opt out cleanly — the report then explains *why* a value is
missing instead of printing a bare `n/a`. An exception inside `compute` is
caught, logged, and reported as a failed metric; one broken plugin never costs
you the report.

### Add a suggestion rule

```python
# in analyzer/suggestions.py
@rule
def my_rule(result) -> list[Suggestion]:
    value = result.metrics.get("keep_rate")
    if value is None or value < 0.9:
        return []
    return [Suggestion(
        title=f"The model keeps almost everything ({value:.1%})",
        detail="A cull that keeps 90%+ of the shoot is not saving you work.",
        evidence=f"keep_rate = {value:.4f} over {len(result.evaluable):,} images (>0.90 flags).",
        severity=WARNING,
        category="threshold",
        action="Raise the threshold.",
    )]
```

The `evidence` field is mandatory in spirit: a rule without a measured trigger
does not belong here.

### Add a chart

Write `def my_chart(result, path) -> Path` in `visualization.py` and add it to
the `CHARTS` dict. Raise to opt out; failures are caught per-chart.

---

## Usage examples

Analyse a ranking against folders:

```bash
picklikeme analyze --ranking rankings_20260726-091500.csv \
    --selected "D:\shoot\keep" --rejected "D:\shoot\drop"
```

Analyse a training run with no folders (labels travel in the file):

```bash
picklikeme analyze --ranking training_results_20260726-091500.csv
```

Tune for recall — missing a keeper costs more than an extra review:

```bash
picklikeme analyze --ranking rankings.csv --selected keep/ --rejected drop/ \
    --optimize-for recall --threshold 0.4
```

Regression-test a new model against the previous one:

```bash
picklikeme analyze --ranking new_model.csv --compare-ranking old_model.csv \
    --selected keep/ --rejected drop/ --output analysis/v3_vs_v2
```

Fast metrics-only pass for CI:

```bash
picklikeme analyze --ranking rankings.csv --selected keep/ --rejected drop/ \
    --no-contact-sheets --no-charts --no-html --quiet
```

As a library:

```python
from picklikeme.analyzer import AnalysisConfig
from picklikeme.analyzer.analysis import run_analysis

result = run_analysis(AnalysisConfig(
    ranking_path=Path("rankings.csv"),
    selected_root=Path("keep"),
    rejected_root=Path("drop"),
))
print(result.metrics.get("roc_auc"))
for suggestion in result.suggestions:
    print(suggestion.severity, suggestion.title)
```
