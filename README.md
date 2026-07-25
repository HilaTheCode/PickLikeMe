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
held-out test split, writing them to `evaluation_metrics.json`.

`--resize-mode stretch` reproduces the V1 baseline preprocessing; the default
`letterbox` is the V2 aspect-ratio-preserving behavior.

### Bird-cropped input (default)

Training feeds a tight crop around the detected bird, so the model sees a
bird-centered image (background discarded). This is **on by default**; pass
`--no-crop-birds` to train on full frames instead (e.g. to A/B-compare).

Build the crop cache once **before training** (detects the bird per image,
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
the same `training_results.csv` the `train` command writes.

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
DINOv3-Huge+ default as training). Output goes to `rankings.csv` by default
(`--output-csv` to change). Pass `--no-crop-birds` only if the model was
trained on full frames.

Detection uses torchvision's COCO-pretrained Faster R-CNN v2 (the "bird" class)
and runs **once** in this single-process pass — never per epoch and never in
DataLoader workers. Crops are cached as PNGs under `cache/crops/` (keyed by
source path, reusable across model input sizes). Images with no detected bird
fall back to the full frame. The crop is a true sub-rectangle then
letterbox-padded to the input size, so the bird is never stretched.

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
  the expanded crop region (yellow) drawn on top, to check the right bird was
  found and beak/wings/tail aren't clipped.
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
