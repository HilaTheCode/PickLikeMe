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

### 2. Install ExifTool (required for metadata and DNG crop embedding)

PickLikeMe uses ExifTool for metadata extraction and for embedding Lightroom crop metadata into DNG files.

macOS (Homebrew):

```bash
brew install exiftool
```

If Homebrew installs it outside your shell PATH, make sure one of these locations is available:

- Apple Silicon: `/opt/homebrew/bin/exiftool`
- Intel: `/usr/local/bin/exiftool`

You can verify it with:

```bash
which exiftool
exiftool -ver
```

Windows:

- Install ExifTool and ensure `exiftool.exe` is on your `PATH`, or pass `--exiftool-path` to the relevant command.

### 3. Build a manifest from a Select/Reject folder pair

```bash
python -m picklikeme.ingest.cli build-manifest --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject" --manifest-path data/manifest.parquet
```

`data/manifest.parquet` (image_path, label, burst_id, capture metadata, ...) is
the single source of truth for labels and burst membership; there is no
separate labels CSV to keep in sync with it.

### 4. Create the frozen evaluation split (once)

```bash
python -m picklikeme.split --manifest data/manifest.parquet --output data/split.csv
```

The split is assigned per burst (never per image) and is frozen: every model
version trains and evaluates against the same split so results are comparable.
The command refuses to overwrite an existing split unless `--force` is given.

### 5. Train the model

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

**This crop cache is the Vision Cache** — the shared image source for every
Computer Vision consumer (training, Classic Vision's eye detectors, its
sharpness metrics, and any future vision module), not a training-only
optimization. It stores each crop at its **own original resolution** by
default (`--max-side`, unset = unlimited) as **JPEG** (`--image-format`,
`--jpeg-quality`, default q98 — near-lossless, ~3× smaller than PNG). A
consumer that genuinely needs a smaller input (training's own data loader)
resizes at load time instead of the cache pre-shrinking everything for it —
see `docs/vision_cache.md` for the full investigation, the real disk-size
measurements behind these defaults, and how this differs from the separate,
UI-only Preview Cache. Changing any cache-affecting parameter (resolution,
format, quality) on an existing cache directory is refused
(`SystemExit: ... pass --force to rebuild`) rather than silently mixing
old and new data — pass `--force` (CLI) or `force_preprocess=True`
(`rank.rank_folder`/`ClassicVisionStrategy.rank_folder`) to rebuild.

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

### Analysis modules (AI Model, Classic Vision)

Ranking is pluggable. A **ranking strategy** is an *analysis module*: it reads
a folder's images and writes scores. Everything downstream —
`analyzer.io.load_ranking`, the review gallery, the AI-suggestion cutoff — is
strategy-agnostic, so adding a module touches none of it.

**Analysis and Organize are independent.** An analysis module never moves a
file, never consults whether a folder has been organized, and never refuses to
run because of workflow state. Every module can be run on a folder that has
never been organized, one already arranged into `_Selected`/`_Rejected`, one
already ranked by another module, one already ranked by itself, or one never
ranked at all. In the organized case it analyses the images *inside*
`_Selected` and `_Rejected` too — an image does not stop being an image once
it has been filed, and RAW vs JPEG makes no difference either. Organize is a
separate, optional operation that may *consume* analysis metadata but is never
a prerequisite for producing it.

**Each module owns its own scores file**, so results coexist instead of
overwriting each other:

```
<folder>/.picklikeme/
    ranking.csv                    # AI Model (keeps the original name)
    ranking-classic-vision.csv     # Classic Vision
    classic_vision_filters.json    # why Classic Vision skipped what it skipped
```

The review session discovers whatever files are present rather than asking the
registry, so a folder scored by a module that has since been removed still
displays its numbers. Both the Gallery card and the Loupe show every module's
score independently, and the Sort menu offers one entry per module — all
generated from the registry, so a future module (Burst Analysis, Species
Analysis) appears in the UI without any UI code changing.

Three ship today, selectable from the Desktop app's **Rank** menu (the
toolbar button itself still runs the default):

| Strategy | What it is |
| --- | --- |
| **AI Model** (default) | The trained preference model — `picklikeme.rank`, exactly as before. |
| **Classic Vision Ranking (EyePose-v0, recommended)** | Deterministic. No model checkpoint fine-tuning, no learning. Eye localisation: EyePose-v0. |
| **Classic Vision Ranking (SuperAnimal)** | The same deterministic scoring, eye localisation: SuperAnimal-Bird. |

**Classic Vision is a framework of interchangeable eye-localisation
backends, not a single algorithm.** Filtering, scoring, the crop cache, the
eye cache, the Gallery/Loupe overlay and every diagnostic below are
completely blind to which backend produced an eye location — they only ever
consume `eyes.detector.EyeDetection` (a box, a confidence, left/right
keypoints, an accept/reject flag). Each backend is its **own** ranking
strategy with its own `strategy_id`, so both can be run on the same folder
and their results coexist for direct comparison, exactly like the AI Model
and Classic Vision already coexist:

```
<folder>/.picklikeme/
    ranking-classic-vision-eyepose-v0.csv   # EyePose-v0's own scores
    ranking-classic-vision.csv              # SuperAnimal's own scores (legacy filename, unchanged)
    classic-vision-eyepose-v0_filters.json  # + matching filter/metrics reports per backend
    classic_vision_filters.json
```

Adding a fourth backend later (a future fine-tuned model, a head-pose
pipeline, …) is implementing `eyes.detector.EyeDetector` and registering it
in `eyes.build_eye_detector` plus one subclass in `ranking/classic.py` — no
other code changes, per `ranking.classic`'s own module docstring.

Both backends run the same two phases:

*Phase 1 — filtering.* An image is excluded from scoring, with an explicit
reason, when it has no detected subject (`NO_SUBJECT`), no locatable eye
(`NO_VISIBLE_EYE`), or a subject no eye detector covers
(`UNSUPPORTED_SUBJECT`). Filtered images get **no score at all** rather than a
zero — they appear in the review gallery as Unranked and Neutral, which is the
honest presentation, and the per-image reasons are written to a
backend-specific `*_filters.json`.

*Phase 2 — scoring.* Three independent metrics, identical across backends,
each normalised across the run and combined as a weighted sum (defaults
50 / 30 / 20):

- **Eye sharpness** — focus measured inside the detected eye box alone
- **Subject sharpness** — focus across the whole subject crop
- **Subject size** — subject box area ÷ full frame area

Weights and the accept/reject thresholds are set in an **Algorithm
Parameters** dialog before each run, with *Reset to Defaults* — generated
from each backend's own parameter declarations (they differ: see below), so
adding a parameter later needs no UI code. Any weight values are valid
(50/30/20 and 5/3/2 mean the same thing); they are normalised, not validated.

#### EyePose-v0 (recommended)

[`synthet/eye-pose-v0`](https://huggingface.co/synthet/eye-pose-v0) — a
YOLO11n-pose checkpoint (MIT licence) fine-tuned on CUB-200-2011 for six
bird head/body landmarks: `beak`, `left_eye`, `right_eye`, `head_top`,
`left_shoulder`, `right_shoulder`.

**Running a `.pt` YOLO checkpoint normally means depending on the
`ultralytics` package, which is AGPL-3.0** — the same license this project
already avoided once (see SuperAnimal-Bird below). The resolution:
`picklikeme/eyes/eyepose_v0.py` downloads the published checkpoint and
converts it to **ONNX once**, on first use, caching the result in
`cache/eye_models/`; every run after that uses `onnxruntime` (MIT) only.
`ultralytics` is an **optional, one-time setup dependency**
(`pip install picklikeme[eyepose-export]`, or just `pip install ultralytics`),
never a runtime one — nothing reachable from a normal ranking run imports it.
The ONNX export was verified byte-for-byte identical to the original
checkpoint's own forward pass before shipping — see
`docs/eyepose_v0_validation.md`.

`onnxruntime` itself is a separate choice, not bundled with the base
install (a GPU-only wheel has no business being forced on a CPU-only
machine, or vice versa) — pick exactly one:

```
pip install picklikeme[eyepose-cpu]   # any machine
pip install picklikeme[eyepose-gpu]   # NVIDIA/CUDA machines
```

Either way, `device="cuda"` is requested automatically whenever CUDA is
available (`picklikeme.platform.resolve_torch_device`); with the CPU-only
package installed this request is silently ignored and CPU is used, no
error. `EyePoseV0EyeDetector` prints a short "EyePose runtime" banner at
construction naming the execution provider, device, and ONNX Runtime
version actually in use — see `picklikeme/eyes/runtime_providers.py` for
how CUDA's native library discovery is kept isolated from EyePose itself
(sourced from torch's own already-mandatory CUDA build, not a second,
duplicated one).

Its accept/reject gate mirrors SuperAnimal-Bird's own two-gate shape without
reusing its arithmetic (the two landmark schemas differ too much for that to
mean the same thing): a confidence threshold, then an **anatomical
plausibility check** — the eye must sit close to the `beak`↔`head_top` line,
relative to head size, catching a landmark that confidently lands somewhere
that isn't actually a head. Unlike SuperAnimal-Bird's thresholds (tuned
against a hand-adjudicated sample of this project's own archive), EyePose-v0's
defaults are reasonable starting points pending the same treatment — see
`docs/eyepose_v0_validation.md` for a real, if small, comparison against
SuperAnimal-Bird and what it does and doesn't establish.

#### SuperAnimal-Bird

[SuperAnimal-Bird](https://huggingface.co/DeepLabCut/DeepLabCutModelZoo-SuperAnimal-Bird)
from the DeepLabCut Model Zoo (Ye et al., *Nature Communications* 2024) — the
only free, actively maintained, bird-specific pretrained model with `left_eye`
and `right_eye` keypoints. Its published architecture is a `timm` `resnet50_gn`
backbone plus two transposed-convolution heads, all of which this project
already depends on, so `picklikeme/eyes/superanimal_bird.py` rebuilds it in
~40 lines and loads the published weights **without taking on DeepLabCut as a
dependency**. The ~103 MB checkpoint downloads once into `cache/eye_models/`;
everything after that is fully local, like the COCO detector's weights.

**How accurate is it?** Measured on 30 crops drawn with a fixed seed from this
archive, stratified 15/15 by your own Selected/Rejected verdict, birds only,
with each one adjudicated by eye against a 6× zoom of the predicted eye box:

| | count |
| --- | --- |
| box lands on a real eye | 14 / 30 |
| box on nape, wing or background | 7 / 30 |
| head too small/blurred to judge | 8 / 30 |
| correctly filtered (no eye visible) | 1 / 30 |

The useful result is the **separation**: every correct detection scored ≥ 0.89,
while six of the seven errors scored below 0.80. Hence the default minimum eye
confidence of **0.80**, which lifts precision from 67% to 93% while keeping
every correct detection. On a 500-crop sample it keeps 71% of Selected images
against 53% of Rejected ones — it discards rejects faster than keepers.

Known limitations, all observed in that sample:

- **Birds only.** On a mammal the model returns a confident, wrong answer (a
  tiger's "eyes" on its ear at 0.67/0.90), so non-birds are reported as
  `UNSUPPORTED_SUBJECT` rather than silently mis-scored.
- **Group scenes are outside its contract.** It is a top-down *single-animal*
  pose model, but `bird_crop` crops flocks to the whole group; its output on
  such a crop is arbitrary and its confidence does not reliably drop.
- **Upstream misdetections pass through.** `supports()` trusts the COCO class
  the subject detector recorded — a fruit bat detected as a bird still reaches
  the eye model.
- **One eye is enough** (side-profile is the norm), but taking the more
  confident of the two occasionally picks the worse-localised one.

Both backends reuse the same crop cache the AI path builds, so on an
already-ranked folder each adds only its own eye pass plus cheap OpenCV
arithmetic — neither re-decodes a RAW or re-runs subject detection.

#### Debug mode and coordinate-transform validation

`ClassicVisionStrategy.rank_folder`'s `debug_dir` parameter (off by default,
not exposed in the desktop UI — a development/troubleshooting aid, not a
photographer-facing feature) writes one combined debug image per processed
candidate: the crop, the eye box, both eye keypoints, confidence values, and
the eye box's coordinates in both crop space and projected onto the full
frame. Backend-agnostic — drawn only from `EyeDetection`, so it needs no
per-backend code.

For validating a backend's coordinate math end to end on real photos —
original image → bird crop → the exact tensor the model saw → raw output →
eye overlay on the crop → eye overlay projected back onto the original —
`picklikeme.eyes.inspect_eyepose` is a read-only CLI tool, the same shape as
`picklikeme.inspect_crops`:

```bash
python -m picklikeme.eyes.inspect_eyepose --input-folder "D:\Photos\sample"
```

**Detector Boxes** (Gallery toolbar / View menu, synced into the Loupe) draws
the subject box(es) any module's preprocessing recorded, plus — once Classic
Vision has run — the eye it measured: solid magenta when accepted, dashed
magenta when detected but distrusted (`EyeDetection.accepted`). Both are shown
regardless of the filtering verdict, on purpose — a *rejected* image's raw eye
guess is exactly what you need to see to judge whether `NO_VISIBLE_EYE` was
the right call. This overlay is read-only: it never re-runs a detector, only
draws what an earlier run already recorded, so a folder ranked only by the AI
model shows subject boxes but no eye until Classic Vision has run on it too.
Box outlines (both the subject box and the eye box, Gallery and Loupe alike)
are drawn at ~5× their original thickness — this overlay is PeakPic's primary
tool for judging a detector's output, so it needs to read at a glance; colors
are unchanged.

**Color Source** (Gallery toolbar) picks which strategy's score tints a card's
background — "Review Status" (green/red/neutral by your own Keep/Reject/
Neutral decision, the default) or any registered module's own score as a
low-to-high gradient across whatever is currently visible. Since AI Model and
Classic Vision can rank the same folder in opposite orders, this makes it
explicit, at all times, which one's opinion the colors on screen represent —
generated from the same registry as the Rank/Sort menus, so a future module
is colorable the moment it is runnable, with no UI change.

### Burst Analysis (Collapse Bursts)

Burst Analysis is a processing layer that runs **after** ranking, not a third
ranking strategy — it never looks at a pixel and produces no score of its
own:

```
Image → Ranking Strategy → Score → Burst Analysis → Burst Ranking → Review UI
```

It is handed nothing but each image's path, capture timestamp and score, so
it works identically whichever strategy (AI Model, Classic Vision, or
whatever ships next) produced that score, and never needs to change when a
new one is added. Burst *identification* reuses `picklikeme.burst`'s existing
capture-time-gap clustering as-is — frames within 2 seconds of the previous
one are the same burst; an image with no readable capture time is its own
singleton burst rather than being guessed into one. Burst *ranking* is this
feature's own addition: within one burst, members are ordered by score
(an unscored member sorts last, never mistaken for a score of zero), giving
every image:

- `burst_id` — which burst it belongs to
- `burst_size` — how many images are in that burst
- `burst_rank` — 1-based position within its own burst, best first
- `burst_best` — true for exactly the top-ranked member

Which strategy's score drives `burst_rank` follows the Gallery's own Color
Source selector (falling back to the AI model for "Review Status", which
isn't a ranking strategy) — one selector, one unambiguous meaning across the
whole app.

The Gallery's **Collapse Bursts** toggle (View menu / toolbar, off by
default — the Gallery is otherwise completely unchanged) shows only each
burst's `burst_best` image, with a small "+N" badge for any burst that has
other members. Opening a collapsed card opens the Loupe scoped to that
burst's own members in `burst_rank` order, so you flip through a burst's
frames best-first with the same Keep/Reject/Neutral workflow, rather than the
whole gallery's order.

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
