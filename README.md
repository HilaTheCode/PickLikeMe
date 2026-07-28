# Pick Like Me

Pick Like Me is a personal machine-learning project for learning an individual wildlife photo selection process from historical keep/reject decisions.

## What it does

- Scans a photo archive into labeled training data
- Builds a manifest of images and metadata
- Supports training a simple preference model from selected/rejected images
- Can rank images after training based on how likely they are to be selected

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pulls whatever CPU/CUDA build of torch pip resolves by
default. To actually use an NVIDIA GPU, install a CUDA build matching your
driver from the [PyTorch wheel index](https://download.pytorch.org/whl/),
e.g. for a driver that supports CUDA 13.0:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
```

Verify with `python -c "import torch; print(torch.cuda.is_available())"`.
Training defaults to `cuda` (see `ProjectConfig.device`) and falls back to
CPU automatically, with a warning, if CUDA isn't available.

`pip install -e .` is what makes the package importable and registers the
`picklikeme` console script (from `[project.scripts]` in `pyproject.toml`).
Without it, `picklikeme ...` (and every `python -m picklikeme...` command
below) fails with "command not found" / "No module named picklikeme".

Every command in this README also works as `python -m picklikeme <command>
...` instead of the bare `picklikeme <command> ...` shown - useful if the
`picklikeme` console script isn't on `PATH` (for example if you keep more
than one virtualenv and only one is active), since `-m` only needs the
*interpreter you run it with* to have the package installed, not `PATH`. Both
forms run the identical code.

### 2. Build a manifest from a Select/Reject folder pair

```bash
python -m picklikeme.ingest.cli build-manifest --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject" --manifest-path data/manifest.parquet
```

`data/manifest.parquet` (image_path, label, burst_id, capture metadata, ...) is
the single source of truth for labels and burst membership; there is no
separate labels CSV to keep in sync with it.

### 3. Create the frozen evaluation split (once)

```bash
python -m picklikeme.split --manifest data/manifest.parquet --output data/split.csv
```

The split is assigned per burst (never per image) and is frozen: every model
version trains and evaluates against the same split so results are comparable.
The command refuses to overwrite an existing split unless `--force` is given.

### 4. Train the model

```bash
python -m picklikeme.train --epochs 20 --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject" --manifest data/manifest.parquet --split data/split.csv
```

`--epochs` is **required** and is the number of epochs to run in *this*
invocation, not a cumulative total. A fresh start (`--fresh-start`) trains
epochs 1..N; `--resume` trains N *more* epochs, continuing the numbering from
the checkpoint — resuming a checkpoint left at epoch 20 with `--epochs 10`
trains epochs 21-30. (If `--epochs` is omitted the command exits with an error
saying it is required.)

With `--split`, training uses only train-split images and afterwards reports the
protocol metrics (Top-1/Top-3 burst accuracy, ROC AUC, precision/recall) on the
held-out test split, writing them to a timestamped
`evaluation_metrics_<date>-<time>.json` (see below).

`--resize-mode stretch` reproduces the V1 baseline preprocessing; the default
`letterbox` is the V2 aspect-ratio-preserving behavior.

### Reading the training log

Every run opens with a one-screen header recording what it is about to do, so a
log from a multi-day run is self-explanatory later:

```
-------------------------------------------------
Training run
  mode:            resuming from checkpoint at epoch 20 (best loss so far 0.1509)
  checkpoint:      C:\Code Projects\PickLikeMe\checkpoints\model_checkpoint.pt
  best checkpoint: C:\Code Projects\PickLikeMe\checkpoints\model_checkpoint_best.pt
  epochs:          21-30 (10 this run)
  labeled images:  54,000 = 21,000 selected / 33,000 rejected
  training set:    54,000 images, 3,375 batches of 16
  validation set:  none (pass --split to hold out a test set)
  backbone:        vit_huge_plus_patch16_dinov3 (1,281 trainable / 632,145,000 params)
  learning rate:   1.00e-04
  device:          NVIDIA GeForce RTX 5070 (7.2 GB free / 12.0 GB total)
  torch:           2.13.0+cu130 (CUDA 13.0)
-------------------------------------------------
```

Then, per epoch, a two-line summary — loss, learning rate, how long the epoch
took, when the run should finish, and where the checkpoint went (`(+best)` marks
an epoch that also improved the best-loss checkpoint):

```
[14:32:10] Completed epoch 21/30 | train_loss 0.1483 | best 0.1483 | lr 1.00e-04 | epoch 12m04s | run eta 1h48m32s
           images 54,000 train / 0 val | checkpoint C:\...\model_checkpoint.pt (+best)
```

There is no per-epoch validation pass, so no validation loss is reported per
epoch: with `--split`, the held-out set is scored **once** after the final epoch.
Within an epoch a progress line prints every `--log-interval-batches` batches
(default 50, plus always the last batch of the epoch) with the batch loss and an
in-epoch ETA. Pass `--log-interval-batches 1` for a line per batch, or `0` for
epoch summaries only.

Preprocessing reports progress on a 30-second timer — position, rate, detections
so far, and ETA — then a summary with a per-class breakdown of what was found:

```
[09:14:02] 12,800/54,000 (23.7%) | 3.4 img/s | detected 12,140 | fallback 630 | skipped 30 | errors 0 | elapsed 1h02m35s | eta 3h21m48s
```

### Every run's output files are timestamped

All per-run result files are named after the run's **start** time, so no run ever
overwrites an earlier one's results:

| Command | Output |
| --- | --- |
| `train` / `run` | `training_results_20260725-143000.csv` (overflow chunks: `…-143000_1.csv`) |
| `train` / `run` with `--split` | `evaluation_metrics_20260725-143000.json` |
| `rank` | `rankings_20260725-143000.csv` |

`--output-csv` / `--metrics-json` change the base name; the timestamp is always
appended to the stem. All outputs of a single run share one stamp (taken at
startup, so a multi-day run is named for when it began), and the resolved paths
are printed before the long work starts. The `%Y%m%d-%H%M%S` format sorts
chronologically, so a plain `ls` lists runs in order.

**Not** timestamped, on purpose: the rolling checkpoint (`checkpoints/`) and
`training_status.json`. Those live at stable paths because `--resume` and
progress polling need to find them.

### Animal-cropped input (default)

Training feeds a tight crop around the detected animal, so the model sees a
subject-centered image (background discarded). This is **on by default**; pass
`--no-crop-birds` to train on full frames instead (e.g. to A/B-compare).

The detector accepts every COCO animal class — the wildlife targets `bird`,
`elephant`, `bear`, `zebra`, `giraffe`, plus `cat`, `dog`, `horse`, `sheep`,
`cow`. Among the surviving detections, **the largest bounding box wins,
whatever its class** — for this archive, the intended subject is almost
always the biggest animal in frame, and a small but confidently-detected
animal must not be preferred over it. Confidence only breaks a tie between
detections whose areas are already close (`--area-tie-frac`, default 10%).
At `--group-scene-threshold` or more surviving detections (default 10), the
image is treated as a **group scene** — a flock, herd or colony — and no
single detection is picked at all: the crop instead tightly encloses the
whole group, never falling back to the full frame. See
`bird_crop.select_best_detection()` for the exact policy. Images with no
supported animal fall back to the full frame. The flags and cache keep their
historical `bird` names.

Build the crop cache once **before training** (detects the animal per image,
crops with a small safety margin, aspect ratio preserved, and caches the
result):

```bash
python -m picklikeme.preprocess --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject"
```

Then train (cropping is already the default):

```bash
python -m picklikeme.train --epochs 20 --select-root "..." --reject-root "..."
```

If `--crop-birds` is on but the cache is empty, training prints a warning and
falls back to full frames — so build the cache first, or pass
`--no-crop-birds` intentionally.

#### How preprocessing spends its time

RAW demosaic dominates: measured at **~80% of wall clock**, with the GPU busy
only ~5% of it. So preprocessing runs as a three-stage pipeline that overlaps
decode with detection and writing:

```
decode pool (8 threads) -> bounded window (12) -> main thread (GPU) -> writer thread
read + rawpy.postprocess                          detect + crop        save PNG
```

Threads rather than processes because `rawpy` releases the GIL during
`postprocess` (measured x2.75 on 8 threads, matching a process pool), so decoded
frames never cross a process boundary. The window bounds decode-side RAM at
roughly 700 MB.

**Detection is unchanged by this**: images reach the detector one at a time, in
source order, with no batching — batched Faster R-CNN is *not* output-identical
(measured box deltas up to 0.077 px, score deltas up to 0.0067, enough to flip a
detection at the 0.30 threshold), so it is deliberately not used.

`--decode-workers` tunes the pool (default `min(8, cpu_count)`). More is not
better: on a 20-core machine with the RAWs on a SATA HDD, 8 workers measured
3.85 img/s while 12 measured 2.42 — libraw is already ~5.8-core parallel per
frame, and concurrent readers reduce HDD throughput. Lower it if you need the
machine for anything else while preprocessing runs; it saturates all cores.

Set `PICKLIKEME_PROFILE=1` for a per-stage timing breakdown (every 500 images
plus a final report). It is inert otherwise — no added CUDA synchronization.

### One command: preprocess → train → rank

`picklikeme.run` chains the two steps above in a single process — it builds the
crop cache, then trains (or resumes) and ranks — so you never have to remember
to preprocess first:

```bash
python -m picklikeme.run --epochs 20 --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject"
```

It accepts every `picklikeme.train` flag (the required `--epochs`, `--split`,
`--backbone`, `--resume`/`--fresh-start`, …) plus the preprocessing knobs
`--margin-frac`, `--conf-threshold`, `--max-side`, and `--force-preprocess`.
Pass `--no-crop-birds` to train on full frames (the preprocess step is then
skipped), or `--skip-preprocess` to reuse an already-built cache. The result is
the same timestamped `training_results_<date>-<time>.csv` the `train` command
writes.

### Evaluate a model against your own decisions

`picklikeme analyze` measures how well a ranking reproduces your keep/reject
choices, shows where it fails, and recommends what to fix:

```bash
picklikeme analyze --ranking rankings_20260726-091500.csv \
    --selected "D:\shoot\keep" --rejected "D:\shoot\drop"
```

Every run writes to its own timestamped folder (`analysis_<date>-<time>/` by
default, or `<your --output>_<date>-<time>/`) so consecutive reports never
overwrite each other; the exact path is printed as the first line of output. It
writes an offline interactive HTML report (light and dark) with detector-box
overlays on every thumbnail that has one, 9 charts, labelled contact sheets of
every mistake category, per-category CSVs, and a JSON record for CI.
`--compare-ranking old.csv` turns it into a regression test between two model
versions.

The analyzer is strictly read-only: it never touches checkpoints, the crop
cache, source images or training data. Full documentation, including the
metrics reference and how to add a metric in one file, is in
[docs/analyzer.md](docs/analyzer.md).

`picklikeme analyze` prints the exact follow-up command for recording *why* a
mistake happened - `picklikeme annotate --output "<the timestamped dir>"` -
which serves the report on `127.0.0.1` so the HTML page's Save button has
something to write to (a report opened straight from disk can display
annotations but can't save new ones - there's no browser API for writing to
SQLite). Or skip the copy-paste with `picklikeme analyze ... --serve`.

### Rank a new, unseen folder with a trained model

Once a model is trained, `picklikeme.rank` scores a directory the model has
**never** seen — no labels, no training. It builds the bird crops for that
folder, loads the checkpoint, and writes a ranked CSV (highest predicted "keep"
score first):

```bash
python -m picklikeme.rank --input "D:\\NewShoot" --checkpoint checkpoints/model_checkpoint.pt
```

`--checkpoint` defaults to the project's rolling checkpoint. The `--backbone`
must match the one the checkpoint was trained with (it defaults to the same
DINOv3-Huge+ default as training). Output goes to a timestamped
`rankings_<date>-<time>.csv` by default (`--output-csv` to change the base
name). Pass `--no-crop-birds` only if the model was trained on full frames.

Detection uses torchvision's COCO-pretrained Faster R-CNN v2 (the animal
classes, see above) and runs **once** in this single-process pass — never per
epoch and never in DataLoader workers. Crops are cached as PNGs under
`cache/crops/` (keyed by source path, reusable across model input sizes). Images
with no detected animal fall back to the full frame. The crop is a true
sub-rectangle then letterbox-padded to the input size, so the animal is never
stretched.

**Cache layout.** Entries are sharded into 256 subdirectories by the first two
hex characters of the path digest, so no directory holds 55k files:

```
cache/crops/
    3f/3fa81234....png
    a7/a7d4bc12....png
    crop_params.json
```

The path is always *computed* from the digest by the single helper
`crop_cache_path()` — the cache is never scanned, globbed or walked, so lookup
cost does not grow with cache size. A cache built before sharding (flat) is not
found by the current code; move each `<digest>.png` into `<digest[:2]>/` to
migrate it.

`cache/crops/crop_params.json` records the parameters and a cache version (`v1`
= bird-only detection, `v2` = all animal classes). Preprocessing refuses to add
to a cache built with different parameters — pass `--force` to rebuild, so a
training set can never mix crops from two different detector configurations.

To visually validate detection + cropping, `picklikeme.inspect_crops` has two
read-only modes (neither changes the cache, detector, preprocessing, or
training).

**Acceptance-folder mode** — run the live pipeline on a folder of
representative images (e.g. ~30 covering small/large/flying/perched/occluded
birds and hard backgrounds):

```bash
python -m picklikeme.inspect_crops --input-folder "C:\\path\\to\\acceptance_set"
```

Every run writes a self-contained, timestamped folder under `inspection/`
(`run_YYYYmmdd_HHMMSS/`) containing:
- `comparison_sheet_*.png` — rows of **original → final crop** (the crop shown
  exactly as the model receives it: detected, cropped, aspect-preserving
  resize, padded), filename beneath each.
- `bbox_overlay_sheet_*.png` — the originals with the detected box (green) and
  the expanded crop region (yellow) drawn on top, labelled with the detected
  COCO class and confidence (`zebra 0.91`), to check the right subject was found
  and nothing is clipped.
- `images/<name>_compare.png` — the per-image original→crop pair.
- `report.txt` + `report.csv` — images processed, successful detections,
  failures, success rate, and per image: confidence, box coordinates, original
  size, crop size, with fallbacks clearly flagged.

**Cache-sampling mode** — sample an already-built cache instead:

```bash
python -m picklikeme.inspect_crops --select-root "..." --reject-root "..."
```

writing `crops_sheet_*.png`, `fallback_sheet_*.png`, and `report.txt` into
`inspection/`. Detected-vs-fallback is re-derived by running the detector
read-only on the sampled crops, since preprocessing doesn't record per-image
outcomes.

### Auto-crop for photo editors (Lightroom)

Separately from training, `picklikeme.auto_crop` writes editor crop metadata
from the same bird detector, to run **before** importing RAWs into Lightroom
(`RAW → PickLikeMe Auto Crop → Lightroom`):

```bash
python -m picklikeme.auto_crop --input "D:\\Photos" --margin 12
```

For every supported RAW (NEF/ARW/CR3/DNG, recursive, case-insensitive) it
detects the bird and computes a **composition** crop that is deliberately
different from the training crop: it expands the bird box by `--margin` percent
and grows it to the **original image aspect ratio** (never square, never
letterboxed, never stretched), then writes it as normalized `[0,1]`
Lightroom `crs:` crop fields only (no develop settings):

- proprietary RAW → a `.xmp` sidecar next to the file (`DSC1234.NEF → DSC1234.xmp`)
- DNG → embedded into the DNG's own XMP via `exiftool` (Lightroom ignores a
  sidecar next to a DNG), so exiftool must be on PATH when DNGs are present

Existing sidecars / embedded crops are left untouched unless `--overwrite-xmp`
is given. The training/preprocessing crop (`build_crop`) is unchanged — this
reuses `BirdDetector.detect_best_bird` and a shared `compute_composition_crop`,
and new editors can be added as exporters in `exporters.py` without touching
detection or crop logic.

The default backbone is a pretrained **DINOv3-Huge+** ViT (V3, ~840M params),
used as a frozen feature extractor (linear probe) behind the same
`PreferenceHead` — the largest DINOv3 variant that comfortably fits a 12GB
GPU (~4.4GB peak at batch size 16). Pass `--backbone cnn` to reproduce the
V1/V2 custom-CNN backbone for comparison, `--backbone vit_small_patch16_dinov3`
(or another `timm` DINOv3 name) for a smaller/faster variant, or
`--unfreeze-backbone` to fine-tune instead of linear-probing. Downloading
pretrained weights requires internet access on first run (cached under the
Hugging Face hub cache afterwards).

### Checkpointing and resuming

- A checkpoint is written to `--checkpoint-path` at the end of every epoch,
  plus periodically during a long epoch (`--checkpoint-interval-minutes`,
  default 15). Writes are atomic (temp file + rename), so a checkpoint file is
  never left half-written.
- The default checkpoint location is `<project-root>/checkpoints/model_checkpoint.pt`,
  resolved from the package's own file location — so it is the **same directory
  no matter which working directory you launch training from**. Pass
  `--checkpoint-path` to override it.
- The lowest-average-training-loss epoch is also saved separately to
  `--best-checkpoint-path` (default `<checkpoint-path>_best.pt`).
- Pressing Ctrl+C saves a checkpoint before the process exits, so at most the
  epoch currently in progress is lost.
- Every save and load prints a transparent diagnostics block (path, reason,
  epoch, best loss, SUCCESS/FAILED), so it is always visible in the console
  exactly what was written where. A failed save is reported but does not abort
  training.
- Training resumes automatically from `--checkpoint-path` if it exists (pass
  `--fresh-start` to ignore it, or `--resume` to force resuming). Resuming
  continues from the last **fully completed** epoch and runs `--epochs` *more*
  epochs (that flag is per-run, not a cumulative target), keeping the epoch
  numbering continuous — it does not restart the count or re-run
  already-completed epochs.
- This only protects processes started with this checkpointing logic —
  editing the code has no effect on a training run already in progress.

## Project structure

- docs/architecture.md: architecture rationale and design decisions
- docs/roadmap.md: the V1-V10 version roadmap and evaluation protocol
- src/picklikeme/: Python package containing the training and ingestion pipeline
- tests/: unit tests

## Notes

This project is still in an early prototype stage and is designed for personal experimentation with a custom photo-selection workflow.
