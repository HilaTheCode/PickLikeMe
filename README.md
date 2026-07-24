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
python -m picklikeme.train --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject" --manifest data/manifest.parquet --split data/split.csv
```

With `--split`, training uses only train-split images and afterwards reports the
protocol metrics (Top-1/Top-3 burst accuracy, ROC AUC, precision/recall) on the
held-out test split, writing them to `evaluation_metrics.json`.

`--resize-mode stretch` reproduces the V1 baseline preprocessing; the default
`letterbox` is the V2 aspect-ratio-preserving behavior.

### Optional: bird-cropped input

Instead of the full frame, training can feed a tight crop around the detected
bird, so the model sees a bird-centered image (background is discarded). This
is opt-in via `--crop-birds` so it can be A/B-compared against the full-frame
model.

Build the crop cache once (detects the bird per image, crops with a small
safety margin, aspect ratio preserved, and caches the result):

```bash
python -m picklikeme.preprocess --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject"
```

Then train against it:

```bash
python -m picklikeme.train --select-root "..." --reject-root "..." --crop-birds
```

Detection uses torchvision's COCO-pretrained Faster R-CNN v2 (the "bird" class)
and runs **once** in this single-process pass — never per epoch and never in
DataLoader workers. Crops are cached as PNGs under `cache/crops/` (keyed by
source path, reusable across model input sizes). Images with no detected bird
fall back to the full frame. The crop is a true sub-rectangle then
letterbox-padded to the input size, so the bird is never stretched.

To visually inspect crop quality before training, render contact sheets of
sampled cached crops (exactly as the model receives them, letterbox padding
included) plus a detection-rate report and a separate sheet of full-frame
fallbacks:

```bash
python -m picklikeme.inspect_crops --select-root "..." --reject-root "..."
```

Outputs land in `inspection/` (`crops_sheet_*.png`, `fallback_sheet_*.png`,
`report.txt`). This is read-only — it never changes the cache, detector, or
training. Detected-vs-fallback is re-derived by running the detector read-only
on the sampled crops, since preprocessing doesn't record per-image outcomes.

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
  continues from the last **fully completed** epoch toward the same
  `--epochs` target — it does not restart the epoch count or re-run
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
