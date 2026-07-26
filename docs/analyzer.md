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
- [Configuration files](#configuration-files)
- [Metrics reference](#metrics-reference)
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

```
analysis/
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

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ranking` | required | Ranking CSV (chunks auto-discovered) |
| `--selected` / `--rejected` | none | Ground-truth folders |
| `--output` | `<repo>/analysis` | Where reports go |
| `--config` | none | JSON config; explicit flags override it |
| `--title` | "PickLikeMe model analysis" | Report title |
| `--threshold` | `0.5` | Decision threshold |
| `--optimize-for` | `f1` | `f1`, `balanced_accuracy`, `accuracy`, `precision`, `recall`, `mcc`, `youden_j` |
| `--threshold-steps` | `101` | Points in the sweep |
| `--borderline-low` / `--borderline-high` | `0.45` / `0.55` | Uncertainty band |
| `--max-examples` | `60` | Rows per table / sheet |
| `--thumbnail-size` | `200` | Thumbnail edge (px) |
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
