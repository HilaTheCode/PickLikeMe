# PickLikeMe (PeakPic) — Developer Onboarding Guide for macOS

This is a complete, self-contained guide to setting up a working PickLikeMe
development environment on a brand-new Mac, starting from nothing. It
assumes you know nothing about this project and have not used this
machine for it before. Every command is written out in full — copy and
paste them in order.

If you are reading this weeks or months after your last session, start
at Section 4 and work forward; Sections 1–3 are background you can skim
or skip if your machine is already set up.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements](#2-system-requirements)
3. [Installing Required Software](#3-installing-required-software)
4. [Clone the Repository](#4-clone-the-repository)
5. [Python Environment](#5-python-environment)
6. [AI Models](#6-ai-models)
7. [Running the Application](#7-running-the-application)
8. [Build Verification](#8-build-verification)
9. [Project Structure](#9-project-structure)
10. [Configuration](#10-configuration)
11. [Development Workflow](#11-development-workflow)
12. [Testing](#12-testing)
13. [Troubleshooting](#13-troubleshooting)
14. [Mac-specific Notes](#14-mac-specific-notes)
15. [Backup and Recovery](#15-backup-and-recovery)
16. [Future Development](#16-future-development)

---

## 1. Project Overview

### What PickLikeMe is

PickLikeMe (desktop app name: **PeakPic**) is a personal machine-learning
project that learns an individual wildlife/bird photographer's photo
selection process from historical keep/reject decisions. Given a large
archive of RAW photos already sorted into "Selected" and "Rejected"
folders, it:

- Scans the archive into labeled training data (a manifest of image
  paths, labels, capture timestamps, and burst membership).
- Trains a preference model that predicts "would this photographer keep
  this photo?"
- Ranks brand-new, never-seen folders of photos by predicted keep
  likelihood, so a shoot can be triaged automatically before manual
  review.
- Provides a native desktop review app (PeakPic) to browse, sort, and
  make Keep/Reject/Neutral decisions on ranked photos, with rich
  analytics on how well the algorithm agrees with the photographer.

It is a single-user personal tool, not a multi-tenant product — there is
no server component beyond a local web UI used for two specific
workflows (see below), and no user accounts.

### Overall architecture

Two independent, coexisting ways of producing a "how good is this photo"
score for an image:

1. **The AI Model** — a learned preference model. A frozen, pretrained
   vision transformer backbone (DINOv3-Huge+ by default) feeds a small
   trainable "preference head." Trained on the photographer's own
   historical Select/Reject decisions with a burst-aware split, so
   results are comparable across model versions. This is the subject of
   `docs/architecture.md` and `docs/roadmap.md`'s V1–V10 version
   sequence.
2. **Classic Vision Ranking** — a deterministic, non-learned pipeline
   with no training step at all: detect the subject animal, locate its
   eye (via one of two interchangeable eye-detector backends, EyePose-v0
   or SuperAnimal-Bird), then combine eye sharpness, subject sharpness,
   and subject size into a weighted score. Documented in full in
   `README.md` (search for "Analysis modules (AI Model, Classic
   Vision)"); this is **not** part of the V1–V10 roadmap, which covers
   only the AI Model strategy.

Both strategies plug into the same downstream infrastructure: a shared
"Vision Cache" of cropped subject images (`cache/crops/`), a common
ranking-strategy registry, and the same review/analytics UI. On top of
both sits:

- **The desktop app (PeakPic)** — a native PySide6 (Qt6) GUI: Gallery,
  Loupe (full-screen review), an Analytics Dashboard (Image Explorer,
  Visual Debug overlays, Score Explanation, User vs Algorithm agreement,
  Run Summary, Species Analytics, Burst Analytics), and workflow dialogs
  for ranking, species organization, and ground-truth import. This is
  the primary way you will interact with the project day to day.
- **A local web review UI** (`picklikeme review`) — an older,
  browser-based equivalent of the desktop Gallery/Loupe, still
  maintained and useful for headless/remote scenarios.
- **The analyzer** (`picklikeme analyze` / `picklikeme annotate`) — a
  strictly read-only evaluation tool that measures how well a ranking
  reproduces the photographer's actual decisions and generates a
  detailed offline HTML report.

### Main modules

Every top-level package under `src/picklikeme/`:

| Package | Purpose |
| --- | --- |
| `desktop/` | The PeakPic native desktop GUI (PySide6/Qt6) — Gallery, Loupe, Analytics Dashboard, workflow dialogs. This is what `peakpic-desktop` launches. |
| `analytics/` | Persisted history of ranking runs and the diagnostics built on top of it — "is the algorithm improving, and where does it fail." |
| `analyzer/` | Read-only evaluation of ranking quality against the photographer's own decisions: metrics, mistake surfacing, HTML reports. Never writes to checkpoints, caches, or source images. |
| `eyes/` | The eye-detection framework — `EyeDetector`/`EyeDetection` interface, with two interchangeable backends (EyePose-v0, SuperAnimal-Bird). |
| `ingest/` | Archive ingestion: scanning a Select/Reject folder pair, metadata extraction, burst reconstruction, manifest building. Home of the `picklikeme` console script's `main()`. |
| `ranking/` | The ranking-strategy registry — the one place that knows every ranking strategy that exists (AI Model, Classic Vision × 2 backends). |
| `review/` | The browser-based review web app (`picklikeme review`) — an HTTP server plus a static frontend. |
| `species/` | "Arrange by Species" — an independent, optional post-review step that files Keep-folder images into per-species subfolders using a zero-shot classifier (BioCLIP / BioCLIP-2). |

Plus many top-level modules directly under `src/picklikeme/` that aren't
packages — the most relevant to know by name: `config.py` (every
default path constant), `train.py`/`rank.py`/`preprocess.py`/`run.py`
(the training/ranking CLI pipeline), `bird_crop.py` (the subject
detector and crop cache), `burst.py`/`burst_analysis.py` (capture-time
clustering and burst ranking), `ground_truth.py` (bulk-importing
existing Select/Reject folder decisions as review ground truth),
`auto_crop.py` (Lightroom crop-metadata export), `model.py` (the
preference model architecture).

### Current implementation status

The roadmap (`docs/roadmap.md`) defines versions **V1 through V10**, one
architectural change at a time, so each change's effect can be measured
in isolation. Based on what `README.md` documents as the project's
*current default behavior*:

| Version | What it is | Status |
| --- | --- | --- |
| V1 | Baseline: stretch-resize, custom CNN, MSE loss | Superseded as default; reproducible via `--resize-mode stretch --backbone cnn` for comparison |
| V2 | Aspect-ratio-preserving letterbox preprocessing | **Default** |
| V3 | Pretrained frozen vision backbone (DINOv3-Huge+) | **Default** (`vit_huge_plus_patch16_dinov3`, ~840M params, linear-probe) |
| V4 | Pairwise/margin ranking loss (replacing MSE) | Roadmap item; not confirmed as the shipped default |
| V5 | Burst-aware weighted pair sampling | Roadmap item; not confirmed as the shipped default |
| V6 | Input resolution study (384/512/640) | Roadmap item; no confirmed resolved default beyond the existing 384px references |
| V7 | Data augmentation | Roadmap item; not confirmed as implemented |
| V8 | Hard negative mining | Roadmap item; not confirmed as implemented |
| V9 | Evaluation and reporting | **Substantially delivered** — the `analyzer/` package (Top-1/Top-3 burst accuracy, ROC AUC, precision/recall, rich HTML reports) matches this deliverable |
| V10 | Explainability heads (sharpness/exposure/blur/eye-visibility confidence) | Roadmap item; no evidence of implementation |

**Beyond the V1–V10 roadmap** (which scopes itself to the AI Model
strategy only), substantial functionality has been built that isn't
part of that versioned list at all: Classic Vision Ranking with two eye
detector backends, Burst Analysis, Auto-crop for Lightroom, and the
entire PeakPic desktop application (Gallery/Loupe/Analytics
Dashboard/Image Explorer/Visual Debug/Score Explanation — a large,
continuously-developed body of work documented across `README.md` and
the phase-specific docs under `docs/`, e.g.
`docs/Analytics_Dashboard_Plan.md`, `docs/Desktop_UX_Redesign_Plan.md`).

**Remaining roadmap**: V4, V5, V6, V7, V8, and V10 as described above.
See `docs/roadmap.md` for each version's full design and acceptance
criteria before starting one — the roadmap is explicit that only one
change should be made per version so its effect is measurable.
`docs/roadmap.md` also references a `docs/results/vN_results.md`
results ledger per version; no `docs/results/` directory exists in this
repository yet, so that ledger has not been started — creating it (with
the frozen-split protocol metrics for whichever version you validate
next) would be a reasonable first contribution.

---

## 2. System Requirements

| Requirement | Minimum | Notes |
| --- | --- | --- |
| macOS | 13 (Ventura) recommended | Not documented anywhere in this repository as a hard requirement — this is a practical recommendation based on the minimum macOS versions Qt 6.5+ (which PySide6 6.11, used by this project, is built on) officially supports. An older macOS (12 Monterey) will likely still work but is untested by this project. |
| Python | 3.10 or newer (`requires-python = ">=3.10"` in `pyproject.toml`) | 3.11 or 3.12 is recommended in practice for the widest availability of prebuilt PyTorch/timm wheels at the time of writing. Do not use Python 3.9 or earlier — installation will fail outright. |
| Git | 2.30 or newer | Any Git shipped with recent Xcode Command Line Tools is sufficient. |
| VS Code | Latest stable | No specific minimum is documented; the project has no version-pinned `.vscode/extensions.json` (see Section 3). |
| Homebrew | Latest | Required to install ExifTool (and optionally Git/Python/VS Code — see Section 3). |
| Disk space | 30 GB free, minimum | The desktop app, its caches, and one trained checkpoint alone can approach ~10 GB (see Section 6's model size table); the crop cache and analysis outputs grow with every folder you rank. 60+ GB free is a more comfortable working margin if you plan to train models. |
| Memory (RAM) | 16 GB | 32 GB strongly recommended if you intend to train the AI model locally — the default DINOv3-Huge+ backbone is large even as a frozen feature extractor. Running only the desktop review app and Classic Vision ranking is comfortable on 16 GB. |
| GPU | Not required | **No NVIDIA CUDA exists on macOS**, and — importantly — **this codebase has no Apple Silicon (MPS) acceleration path implemented**. `ProjectConfig.device` defaults to `"cuda"` and falls back to CPU automatically with a warning if CUDA is unavailable, which is always the case on a Mac. This means: the desktop app, Classic Vision ranking, and inference/ranking with an already-trained checkpoint all run acceptably on CPU. **Training the AI Model from scratch on a Mac will be dramatically slower than on the NVIDIA GPU this project was originally developed against** (the README's own example log shows an RTX 5070 doing 10 epochs over 54,000 images in under 2 hours) — plan accordingly, or treat Mac-based training as a small-scale/smoke-test-only activity. |

**Apple Silicon vs Intel**: both are supported by every tool involved
(Homebrew, Python, PySide6, PyTorch all ship Apple Silicon builds), but
Apple Silicon is strongly preferred for performance. If you're on Intel,
expect everything — especially any local training — to be slower still.

---

## 3. Installing Required Software

Open **Terminal** (Applications → Utilities → Terminal, or press
<kbd>Cmd</kbd>+<kbd>Space</kbd>, type "Terminal", press Return) and work
through these steps in order.

### 3.1 Xcode Command Line Tools

Required for Git, compilers used by some Python packages (e.g. `rawpy`),
and general Unix tooling. Install with:

```bash
xcode-select --install
```

A dialog will pop up — click **Install**, accept the license, and wait
for it to finish (several minutes). If you get "command line tools are
already installed", that's fine — skip to the next step.

### 3.2 Homebrew

Homebrew is the standard macOS package manager and is used below to
install Git, Python, VS Code, and ExifTool. Install it with the official
command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions at the end of the installer — on
Apple Silicon Macs it will print one or two `eval "$(/opt/homebrew/bin/brew
shellenv)"` lines you must run (or add to your shell profile) before
`brew` is available in new terminal windows. On Intel Macs Homebrew
installs to `/usr/local` and is usually already on `PATH`.

Verify:

```bash
brew --version
```

You should see output like `Homebrew 4.x.y`.

### 3.3 Git

If Xcode Command Line Tools already provided a usable Git, you can skip
this — check first:

```bash
git --version
```

If that fails, or you want the latest version, install via Homebrew:

```bash
brew install git
```

Verify again with `git --version` (expect `2.4x` or newer).

Configure your identity (required before your first commit on this
machine, if you haven't already):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 3.4 Python

Install Python via Homebrew (this project requires 3.10+; 3.11 or 3.12
is recommended):

```bash
brew install python@3.12
```

Verify:

```bash
python3.12 --version
```

Expect `Python 3.12.x`. (Alternatively, download an official installer
from [python.org/downloads](https://www.python.org/downloads/macos/) if
you prefer not to manage Python via Homebrew — either works equally
well for this project, since Section 5 creates an isolated virtual
environment regardless.)

> **Do not rely on the `python3` that ships with macOS itself** (under
> `/usr/bin/python3`) — it is Apple's own minimal build, intended for
> system scripts, and is not a good base for a virtual environment with
> heavy scientific dependencies like `torch`/`opencv-python`.

### 3.5 Visual Studio Code

Download and install from
[code.visualstudio.com/download](https://code.visualstudio.com/download)
(choose the macOS .zip, drag `Visual Studio Code.app` into
`/Applications`), or via Homebrew:

```bash
brew install --cask visual-studio-code
```

Launch it once from `/Applications` or Spotlight to confirm it opens.

**Install the "code" shell command** (lets you open the project with
`code .` from Terminal): in VS Code, press <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>,
type "Shell Command", select **Shell Command: Install 'code' command in
PATH**, press Return.

#### Required/recommended VS Code extensions

This repository does not ship a `.vscode/extensions.json` (its
`.vscode/` directory is git-ignored entirely — see Section 9), so there
is no automated extension-recommendation prompt. Install these manually,
via the Extensions panel (<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>X</kbd>)
or from Terminal once `code` is on PATH:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
```

- **Python** (`ms-python.python`) — required: interpreter selection,
  debugging, test discovery.
- **Pylance** (`ms-python.vscode-pylance`) — bundled with the Python
  extension in recent VS Code versions, but install explicitly to be
  sure; gives type checking and IntelliSense.

No formatter/linter (Black, Ruff, etc.) is configured anywhere in this
project (no `pyproject.toml` tool section, no `ruff.toml`) — none is
required.

### 3.6 ExifTool

Required for metadata extraction and for embedding Lightroom crop
metadata into DNG files (used by `picklikeme.auto_crop`).

```bash
brew install exiftool
```

If Homebrew installs it somewhere not automatically on your shell's
PATH, confirm one of these exists:

- Apple Silicon: `/opt/homebrew/bin/exiftool`
- Intel: `/usr/local/bin/exiftool`

Verify:

```bash
which exiftool
exiftool -ver
```

### 3.7 Claude Code

This project is developed with the help of Claude Code (Anthropic's
agentic CLI). Install it via npm (requires Node.js 18 or newer — install
Node first if you don't have it: `brew install node`):

```bash
brew install node
npm install -g @anthropic-ai/claude-code
```

Verify:

```bash
claude --version
```

Start it from inside the project directory (after Section 4) with:

```bash
claude
```

The first run will prompt you to authenticate. If any of these commands
have changed since this document was written, check
[docs.claude.com](https://docs.claude.com) for the current official
install instructions.

---

## 4. Clone the Repository

The repository lives on GitHub at:

```
https://github.com/HilaTheCode/PickLikeMe
```

### 4.1 Choose a location and clone

Pick a folder to keep your projects in (this example uses
`~/Code`, adjust as you like):

```bash
mkdir -p ~/Code
cd ~/Code
git clone https://github.com/HilaTheCode/PickLikeMe.git
cd PickLikeMe
```

This creates `~/Code/PickLikeMe` containing the full repository. All
commands in the rest of this document assume your terminal's current
directory is the repository root (`~/Code/PickLikeMe` in this example)
unless stated otherwise.

### 4.2 Open in VS Code

```bash
code .
```

(or, from VS Code itself: **File → Open Folder…** and select the
`PickLikeMe` folder.)

### 4.3 Verify the correct branch

The default/main branch is `main`. Confirm you're on it:

```bash
git branch --show-current
```

Expected output:

```
main
```

If it prints something else, switch:

```bash
git checkout main
```

### 4.4 Verify the current commit

```bash
git log -1 --format="%H%n%cI%n%s"
```

This prints the current commit's full hash, its commit date (ISO 8601),
and its subject line — the same three facts (minus the subject line)
the desktop app's own **Help → About** dialog reports (see Section 8).
Compare the hash against what you see in GitHub's web UI for the `main`
branch if you want to confirm you have the very latest commit:

```bash
git fetch origin
git log origin/main -1 --format="%H"
```

If the two hashes differ, you're behind — run `git pull` to update (see
Section 11).

---

## 5. Python Environment

### 5.1 Create a virtual environment

From the repository root:

```bash
python3.12 -m venv .venv
```

(Substitute whichever Python 3.10+ you installed in Section 3.4, e.g.
`python3.11`, if you didn't install 3.12.)

This creates a `.venv/` folder inside the repository. It is git-ignored
(`.gitignore` excludes `.venv/`) and is entirely local to this machine —
never commit it, and never try to copy it to another computer (see
Section 15 for how to reconstruct it elsewhere instead).

### 5.2 Activate it

```bash
source .venv/bin/activate
```

Your terminal prompt should now be prefixed with `(.venv)`. **You must
run this `source` command in every new terminal window/tab** before
running any `python`, `pip`, `pytest`, or `picklikeme` command for this
project — activation does not persist across terminal sessions.

To deactivate at any time (e.g. to switch to a different project):

```bash
deactivate
```

### 5.3 Install dependencies

With the virtual environment active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

- `requirements.txt` installs every core dependency, including
  `PySide6` (the desktop GUI toolkit) and `open_clip_torch` (BioCLIP
  species classification) — both were added to `requirements.txt` and
  `pyproject.toml` alongside this onboarding document, since neither was
  previously declared anywhere despite being required at runtime. If
  you're working from a much older checkout, `pip install PySide6
  open_clip_torch` explicitly if the app fails with `ModuleNotFoundError`
  for either.
- `pip install -e .` installs the project itself in "editable" mode —
  this is what makes `import picklikeme` work, and registers the two
  console scripts (`picklikeme`, `peakpic-desktop`). **Do not skip this
  step** — without it, every command in this document that starts with
  `picklikeme` or `peakpic-desktop` fails with "command not found", and
  `python -m picklikeme...` fails with "No module named picklikeme".

This will take several minutes — `torch`, `torchvision`, and
`opencv-python` are large downloads (roughly 1–2 GB combined). On a Mac,
`pip` resolves the CPU-only build of PyTorch automatically (there is no
CUDA build for macOS) — this is expected and correct; see Section 2 for
what that means for training speed.

**EyePose-v0 support (Classic Vision Ranking, recommended eye
detector)** needs one more package, `onnxruntime` — pick the CPU
package (there is no GPU/CUDA build relevant on a Mac):

```bash
pip install onnxruntime
```

(Apple Silicon owners: `onnxruntime` publishes native `arm64` wheels, so
this installs and runs natively — no Rosetta needed.)

### 5.4 Verify installation

Run all of these; each should succeed with no errors:

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
python -c "import PySide6; print('PySide6', PySide6.__version__)"
python -c "import open_clip; print('open_clip OK')"
python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
python -c "import picklikeme; print('picklikeme package importable')"
picklikeme --help
peakpic-desktop --help 2>&1 | head -1 || echo "(peakpic-desktop has no --help output; that alone is fine — see Section 7)"
```

`torch.cuda.is_available()` will correctly print `False` on a Mac — that
is expected, not an error (see Section 2).

If every command above ran without a traceback, your environment is
correctly set up.

### 5.5 Update dependencies

When `requirements.txt` or `pyproject.toml` change (e.g. after pulling
new commits — see Section 11), re-run:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -r requirements.txt` is safe to re-run any time — it only
installs/upgrades what's missing or out of date, and does nothing
otherwise. If you ever suspect the environment itself is corrupted
rather than just outdated, see Section 13's "Broken virtual environment"
entry.

---

## 6. AI Models

PickLikeMe uses several pretrained/fine-tuned models, each with its own
download and cache mechanism. **None of these model files are stored in
Git** (`.gitignore` excludes `*.pt`, `*.pth`, `*.onnx`, and the whole
`/cache/` and `/checkpoints/` directories) — every one of them is
fetched (or trained) fresh the first time it's needed.

### 6.1 Summary table

| Model | Used for | Downloaded from | Auto-downloads? | Cached at | Approx. size |
| --- | --- | --- | --- | --- | --- |
| Faster R-CNN v2 (COCO) | Subject/animal detection & cropping (the shared "Vision Cache" input) | torchvision's built-in weights (`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`) | Yes, on first use | Torch Hub's own cache, `~/.cache/torch/hub/checkpoints/` (not overridden by this project) | Not documented; a few hundred MB is typical for this architecture |
| BioCLIP‑2 | Zero-shot species classification (default) | Hugging Face Hub, `hf-hub:imageomics/bioclip-2` | Yes, on first use | `~/.cache/huggingface/hub/models--imageomics--bioclip-2` | ≈ 1.6 GB |
| BioCLIP (v1) | Zero-shot species classification (alternative, selectable) | Hugging Face Hub, `hf-hub:imageomics/bioclip` | Yes, on first use | `~/.cache/huggingface/hub/models--imageomics--bioclip` | ≈ 571 MB |
| DINOv3‑Huge+ ViT | The AI Model's frozen backbone (training/inference) | Hugging Face Hub, via `timm` (`vit_huge_plus_patch16_dinov3`) | Yes, on first use | `~/.cache/huggingface/hub/` | ~840M parameters; file size not documented, expect several GB |
| EyePose‑v0 | Eye/landmark detection, Classic Vision Ranking (recommended backend) | Hugging Face Hub, direct download, `synthet/eye-pose-v0` | Yes, on first use (converts to ONNX once, needs `ultralytics` for that one conversion — see 6.3) | `cache/eye_models/eye_pose_v0.pt` and `.onnx`, inside the project | ≈ 5.4 MB |
| SuperAnimal‑Bird | Eye/landmark detection, Classic Vision Ranking (alternative backend) | Hugging Face Hub, direct download, DeepLabCut Model Zoo | Yes, on first use | `cache/eye_models/superanimal_bird_resnet_50.pt`, inside the project | ≈ 103 MB |
| Your trained checkpoint (`model_checkpoint.pt`) | The AI Model's actual preference predictions | **Not downloaded — you train it yourself**, or copy one from another machine | No | `checkpoints/model_checkpoint.pt` (+ `..._best.pt`), inside the project | Several GB (this project's own reference checkpoint is ≈ 3.36 GB) |

### 6.2 Where things are cached, and why two different locations

Most models use the **standard Hugging Face Hub cache** at
`~/.cache/huggingface/hub/` — this is a machine-wide cache shared with
any other project that also uses Hugging Face models, not specific to
PickLikeMe. It respects the `HF_HOME` environment variable if you've set
one, falling back to `~/.cache/huggingface` otherwise.

The two eye-detector models (EyePose-v0, SuperAnimal-Bird) are
different: they're downloaded directly from a Hugging Face-hosted URL
(not through the `huggingface_hub` library) and cached **inside the
project itself**, under `cache/eye_models/`. This is deliberate —
Classic Vision's eye models are small enough that keeping them local to
the project (rather than in a machine-wide cache) is simple and makes
"does this project have everything it needs" a single directory to
check.

### 6.3 EyePose-v0's one-time setup step

Running EyePose-v0 the very first time needs to convert its published
`.pt` checkpoint to ONNX (see Section 6.4 for why). This conversion step
needs the `ultralytics` package, which is **not** installed by
`requirements.txt`/`pip install -e .` on purpose (it's AGPL-3.0
licensed, and this project avoids depending on it at runtime):

```bash
pip install ultralytics
```

Install this once before the first time you run Classic Vision Ranking
with the EyePose-v0 backend. After the one-time conversion succeeds
(you'll see it print progress and then cache `eye_pose_v0.onnx`), you
can `pip uninstall ultralytics` if you want — nothing reachable from a
normal ranking run imports it again.

### 6.4 Automatic vs manual downloads

Every model in the table above downloads **automatically** the first
time it's actually used:

- Opening the desktop app and running **Rank → Classic Vision Ranking
  (EyePose-v0)** on a folder triggers EyePose-v0's download+conversion
  (and needs `ultralytics` installed first, per 6.3) and/or
  SuperAnimal-Bird's download, depending which backend you pick.
- Running **Organize by Species** (or `picklikeme arrange-species`)
  triggers BioCLIP/BioCLIP-2's download.
- Running `picklikeme.preprocess`, `picklikeme.rank`, or opening the
  desktop app's Rank workflow triggers the Faster R-CNN subject
  detector's download.
- Training the AI Model (`python -m picklikeme.train ...`) triggers the
  DINOv3 backbone's download.

**All of these require an active internet connection the first time**,
after which everything works fully offline. If you're setting up on a
Mac with limited or metered internet, expect the very first run of each
feature to pause for a download — this is normal, not a hang (watch the
terminal for progress output).

### 6.5 How to verify successful installation

After running each feature once, confirm the expected file exists:

```bash
# Faster R-CNN + BioCLIP + DINOv3 (Hugging Face / Torch Hub caches)
ls -la ~/.cache/huggingface/hub/ 2>/dev/null
ls -la ~/.cache/torch/hub/checkpoints/ 2>/dev/null

# EyePose-v0 and SuperAnimal-Bird (project-local cache)
ls -la cache/eye_models/
```

You should see `eye_pose_v0.pt`, `eye_pose_v0.onnx`, and/or
`superanimal_bird_resnet_50.pt` in `cache/eye_models/` once you've run
each backend at least once, and one or more `models--imageomics--...`
directories under the Hugging Face hub cache once BioCLIP has run.

The desktop app also prints a short diagnostic banner to the terminal
when EyePose-v0 loads, naming the execution provider (CPU on a Mac),
device, and ONNX Runtime version actually in use — check your terminal
output (launch the app from Terminal, not by double-clicking, to see
this — see Section 7) if you want to confirm which backend is actually
running.

### 6.6 How to redownload a model

Every one of these caches is just files on disk — there is no
project-specific "redownload" command. Delete the relevant file(s) and
re-run the feature that needs them; the download logic detects the
missing file and fetches it again automatically:

```bash
# Force EyePose-v0 to redownload/re-convert
rm cache/eye_models/eye_pose_v0.pt cache/eye_models/eye_pose_v0.onnx

# Force SuperAnimal-Bird to redownload
rm cache/eye_models/superanimal_bird_resnet_50.pt

# Force BioCLIP-2 to redownload
rm -rf ~/.cache/huggingface/hub/models--imageomics--bioclip-2
```

Both eye-model downloads write to a temporary `.part` file and rename it
only once the download completes, so an interrupted/failed download
never leaves a corrupt file behind that would be mistaken for a
successful cache — if a download fails partway, just re-run.

### 6.7 How to clear the cache

To reclaim disk space or start completely fresh:

```bash
# Everything: eye models, crop cache, thumbnail/preview caches, analytics/species DBs
rm -rf cache/

# Just the Hugging Face-hosted models (shared with any other project on this Mac)
rm -rf ~/.cache/huggingface/hub/models--imageomics--*

# Just the torchvision subject detector
rm -rf ~/.cache/torch/hub/checkpoints/
```

`cache/` will be recreated automatically (empty) the next time the app
or any `picklikeme` command runs — nothing needs to be manually
recreated.

**Do not delete `checkpoints/`** unless you specifically want to discard
your trained AI Model and start over — it is not re-downloadable, only
re-trainable (see Section 6.1's last row).

---

## 7. Running the Application

### 7.1 Launch the desktop app

With the virtual environment active (`source .venv/bin/activate`, from
the repository root):

```bash
python -m picklikeme.desktop
```

or, equivalently (once `pip install -e .` has run):

```bash
peakpic-desktop
```

Both start the identical application.

### 7.2 Using the launcher script

This repository includes `Start PeakPic.command` — a double-clickable
macOS launcher (the Mac equivalent of the pre-existing Windows `Start
PeakPic.bat`). Before it can be double-clicked from Finder, make it
executable once:

```bash
chmod +x "Start PeakPic.command"
```

After that, you can either double-click it in Finder (macOS will open a
Terminal window and run it — the first time, macOS Gatekeeper may show a
security prompt; see Section 14), or run it directly from a terminal:

```bash
./"Start PeakPic.command"
```

Unlike the Windows `.bat` (which hardcodes a Windows path), this script
resolves the project directory from its own location, so it keeps
working regardless of where you cloned the repository — you don't need
to edit it.

### 7.3 Launching from VS Code

No `.vscode/launch.json` exists in this repository (see Section 9), so
there is no built-in "Run and Debug" configuration yet. The simplest way
to run (or debug) from VS Code:

1. Open the integrated terminal (<kbd>Ctrl</kbd>+<kbd>`</kbd>).
2. Confirm it activated the right interpreter — VS Code's Python
   extension usually activates `.venv` automatically once you've
   selected it as the interpreter (<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>
   → **Python: Select Interpreter** → choose the one under
   `.venv/bin/python`). If the terminal prompt doesn't already show
   `(.venv)`, run `source .venv/bin/activate` manually.
3. Run `python -m picklikeme.desktop`.

To **debug** with breakpoints: open the Run and Debug panel
(<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>D</kbd>), click "create a
launch.json file", choose **Python Debugger → Module**, and enter
`picklikeme.desktop` as the module name when prompted. This creates a
`.vscode/launch.json` locally (it stays untracked — see Section 9) that
you can reuse afterward via the Run and Debug panel's green ▷ button.

### 7.4 Expected startup messages

Launched from a terminal, you should see the desktop app's window
appear within a few seconds, with **no error tracebacks printed to the
terminal**. The window title bar reads "PeakPic Desktop". A brief log
line similar to:

```
INFO:picklikeme.desktop:PeakPic desktop shell initialized
```

confirms the shell finished initializing. Opening a folder for the
first time, or running a ranking strategy for the first time, may print
model-loading/download progress (see Section 6.4) — that's expected on
a first run, not an error.

If the app instead exits immediately with a Python traceback ending in
`ModuleNotFoundError`, see Section 13.

### 7.5 How to verify the application started correctly

- The main window is visible, with a menu bar (File, Review, View,
  Tools, Help) and an empty gallery with a prompt to open a folder.
- **Help → About** opens without error and shows real values for every
  field (never blank or "unknown" for Application Version, Python
  Version, or Source Path — see Section 8 for exactly what to check).
- **File → Open Folder…** opens a native macOS folder picker.
- No traceback appeared in the terminal you launched it from.

If all four are true, the application started correctly.

---

## 8. Build Verification

Use this to confirm exactly which build you're running — essential
after a `git pull`, or when reporting a bug.

Open the running desktop app and go to **Help → About**. It reports
five facts, freshly computed every time you open it (never cached from
when the app started):

| Field | What it tells you | How it's computed |
| --- | --- | --- |
| **Application version** | The installed package's declared version (`0.1.0` as of this writing, from `pyproject.toml`) | `importlib.metadata.version("pick-likeme")` |
| **Git commit** | The exact commit your working copy's `picklikeme` package was imported from | `git rev-parse HEAD`, run fresh |
| **Build timestamp (commit date)** | When that commit was authored — the honest proxy for "build time" this project has, since there is no separate packaging/build step | `git log -1 --format=%cI` on that same commit |
| **Python version** | The interpreter actually running the app | `platform.python_version()` |
| **Running from** | The exact source directory the running code was imported from | Resolved from `__file__` at runtime |

**To confirm you're running the latest build**, compare the About
dialog's "Git commit" value against:

```bash
git log -1 --format="%H"
```

run from your terminal in the repository root. They must match exactly.
If they don't match, you have uncommitted local changes to tracked
files that don't affect the commit hash (normal), or you're running the
app from a different checkout/environment than the terminal you're
comparing against (check "Running from" against `pwd`).

If **Git commit** shows `unknown (not a git checkout, or git
unavailable)`, either `git` isn't on `PATH` for the environment running
the app, or the source directory reported under "Running from" isn't
actually inside a `.git` repository (e.g. you copied the `src/`
directory somewhere else instead of running from a clone).

---

## 9. Project Structure

Every top-level file/directory at the repository root:

| Path | Purpose |
| --- | --- |
| `README.md` | The primary reference for the CLI/training/ranking pipeline — most of what Section 1 of this document summarizes lives here in full detail. |
| `docs/Developer_Onboarding_Mac.md` | This document. |
| `pyproject.toml` | Package metadata, dependencies, and the two console scripts (`picklikeme`, `peakpic-desktop`). Authoritative dependency list. |
| `requirements.txt` | A plain pip requirements file mirroring `pyproject.toml`'s core dependencies, for `pip install -r requirements.txt` (Section 5). |
| `Start PeakPic.bat` | Windows launcher script. |
| `Start PeakPic.command` | macOS/Linux launcher script (added alongside this document — see Section 7). |
| `training_status.json` | Live, non-timestamped training-progress state, read by the desktop app and `--resume`. Not meaningful outside an active/completed training run on this machine. |
| `src/` | All Python source, in `src/picklikeme/` (a "src-layout" package — see Section 5). |
| `tests/` | The full pytest test suite — over 1,300 tests as of this writing (Section 12). |
| `docs/` | Design docs, investigation write-ups, and phase-delivery reports: `architecture.md` and `roadmap.md` (the V1–V10 plan), `analyzer.md` (the analyzer's own reference doc), and a number of dated investigation/plan documents (`Analytics_Dashboard_Plan.md`, `BioCLIP_Backend_Architecture_Review.md`, `BioCLIP_Infrastructure_Deliverables.md`, `Desktop_UX_Redesign_Plan.md`, `EyePose_Investigation_Phase_1.md`, `Species_Classification_Investigation.md`, `eyepose_v0_validation.md`, `vision_cache.md`, plus an `eye_detector_eval/` subdirectory). |
| `assets/` | Static assets bundled with the app: `peakpic.ico` (the window/app icon) and `species/All_Birds.txt` (the default species list used by "Organize by Species"). |
| `config/` | `annotations.yaml` — the schema for annotation fields used by the analyzer's review/annotate UI. Data, not code. |
| `tools/` | Standalone utility/debug scripts, not part of the installed package: `debug_eye_pipeline.py`, `debug_species_pipeline.py`, and two Windows PowerShell scripts (`My_selection_to_csv*.ps1`) that will not run natively on macOS — if you need their functionality on a Mac, either install PowerShell Core (`brew install powershell`, then run with `pwsh`) or port the logic to a plain Python/shell script. |
| `checkpoints/` | Trained AI Model checkpoints (`model_checkpoint.pt`, `model_checkpoint_best.pt`). Git-ignored; empty on a fresh clone (Section 6). |
| `cache/` | Every runtime cache: crop cache, eye-model weights, analytics/species SQLite databases, review thumbnail/preview caches (Section 10). Git-ignored; empty on a fresh clone. |
| `annotations/` | The analyzer's own annotation databases (`review.db`, `false_negatives.db`). Git-ignored; empty on a fresh clone. |
| `analysis_results/` | Timestamped output folders from `picklikeme analyze` runs. Git-ignored. |
| `inspection/` | Timestamped output folders from `picklikeme.inspect_crops` / `picklikeme.eyes.inspect_eyepose`. Git-ignored. |
| `logs/` | Training log files (only present if you've run training with logging redirected there). Git-ignored. |
| `.venv/` | Your local Python virtual environment (Section 5). Git-ignored; you create this yourself, it never comes from git. |
| `.vscode/` | VS Code workspace settings. **Entirely git-ignored** — none of this transfers between machines; see Section 3/7 for what to (re)create. |
| `.claude/` | Claude Code's own local project state. Not git-tracked. |

There is currently **no `scripts/` directory and no `sample_data/`
directory** in this repository — if you were expecting either (they're
common conventions in other projects), they simply don't exist here;
don't spend time looking for them.

### Inside `src/picklikeme/`

| Path | Purpose |
| --- | --- |
| `__main__.py` | Makes `python -m picklikeme <command>` work without any console script needing to be on `PATH`. |
| `config.py` | Every default path constant (checkpoint dir, crop cache dir, results dir, inspection dir) and small shared utilities (duration formatting, tee-to-log, the `cli_prefix()` helper used in printed instructions). See Section 10. |
| `desktop/` | The GUI application — see Section 1. |
| `analytics/`, `analyzer/`, `eyes/`, `ingest/`, `ranking/`, `review/`, `species/` | See Section 1's module table. |
| `bird_crop.py` | The subject/animal detector and the shared crop cache ("Vision Cache") both training and Classic Vision consume. |
| `burst.py` / `burst_analysis.py` | Capture-time-gap burst clustering, and post-ranking burst ranking/collapsing. |
| `train.py` / `rank.py` / `preprocess.py` / `run.py` / `split.py` | The training pipeline's CLI entry modules (`python -m picklikeme.<name>`). |
| `ground_truth.py` | Bulk-importing an existing Select/Reject folder structure as review ground truth. |
| `auto_crop.py` / `exporters.py` | Lightroom crop-metadata export, independent of training. |
| `model.py` | The preference-model architecture (frozen backbone + trainable head). |
| `identity.py` / `sidecar.py` | Content-hash-based image identity, and the `.picklikeme/` per-folder sidecar file format used to store ranking scores. |

---

## 10. Configuration

PickLikeMe has no single central config file for runtime settings —
configuration is split across a few well-defined mechanisms:

### 10.1 QSettings (desktop app preferences)

The desktop app persists UI preferences via Qt's `QSettings`, always
constructed as `QSettings("PeakPic", "PeakPicDesktop")`
(organization `"PeakPic"`, application `"PeakPicDesktop"` — one word).
**On macOS, this is stored as a property list file**, typically at:

```
~/Library/Preferences/com.PeakPic.PeakPicDesktop.plist
```

Inspect it from Terminal:

```bash
defaults read com.PeakPic.PeakPicDesktop
```

Reset **all** desktop app preferences to defaults (useful when
troubleshooting a corrupted/confusing UI state — see Section 13):

```bash
defaults delete com.PeakPic.PeakPicDesktop
```

The keys actually stored there:

| Key | What it remembers |
| --- | --- |
| `recent_folders` | The Gallery's recently-opened-folder list |
| `last_opened_folder` | Where the "Open Folder…" dialog starts next time |
| `theme` | Light/dark/system theme choice |
| `window/geometry`, `window/state` | Main window size/position/dock layout |
| `review/species_language` | Species-ID language (default `en`) |
| `review/species_backend` | Species-ID backend (default `bioclip2`) |
| `review/species_list_path` | Path to a custom species list file, if set |
| `review/burst_sort_mode` | Burst-group sort order preference |
| `analytics/dashboard_geometry` | Analytics Dashboard window geometry |
| `dialogs/rank_geometry`, `dialogs/species_language_geometry`, `dialogs/preferences_geometry`, `dialogs/auto_crop_geometry`, `dialogs/set_user_decisions_by_subfolders_geometry`, `dialogs/algorithm_parameters_<StrategyName>_geometry` | Per-dialog remembered window geometry |

None of this transfers between machines automatically — a fresh Mac
starts with all of the above at their defaults.

### 10.2 Cache locations

All project-local caches live under `cache/` (repository root, created
automatically on first use):

| Path | Contents |
| --- | --- |
| `cache/crops/` | The shared "Vision Cache" — cropped subject images, sharded into 256 subdirectories by content hash. See `docs/vision_cache.md`. |
| `cache/eye_models/` | Downloaded EyePose-v0/SuperAnimal-Bird weights (Section 6). |
| `cache/analytics.db` | Persisted history of every ranking run (Analytics Dashboard's data source). Safe to delete — rebuilds as you re-rank. |
| `cache/species.db` | Memoized species-classification predictions, keyed by image content hash + classifier. |
| `cache/analyzer_detections.db` | The analyzer's own cache of detector boxes. |
| `cache/review_thumbs/`, `cache/review_previews/` | Desktop Gallery thumbnail and Lightbox preview caches. |
| `cache/desktop/` | The desktop shell's own in-memory-cache-manager disk backing (path resolved relative to the current working directory, not the project root — always launch the app from the repository root, as every command in this document does, so this lands in the expected place). |

Outside `cache/`:

- `checkpoints/model_checkpoint.pt` (+ `_best.pt`) — your trained AI
  Model (Section 6).
- `annotations/review.db` — the analyzer's ground-truth annotation
  database (a **sibling** of `cache/`, not inside it).
- `~/.cache/huggingface/hub/` and `~/.cache/torch/hub/checkpoints/` —
  machine-wide model caches, not specific to this project (Section 6).

### 10.3 Analytics database

`cache/analytics.db` — a SQLite database recording every ranking run's
parameters, per-image metrics, and summary statistics. Read by the
desktop app's Analytics Dashboard (Run Summary, Species Analytics,
Burst Analytics, User vs Algorithm, Image Explorer). Rebuilds itself
automatically the next time you rank a folder if deleted; deleting it
only loses historical run comparisons, never anything about the images
themselves.

### 10.4 Species cache

`cache/species.db` — per-image species predictions, keyed by content
hash and which classifier produced them (so BioCLIP and BioCLIP-2
predictions for the same image coexist rather than overwriting each
other). Safe to delete; species classification simply re-runs and
re-populates it.

### 10.5 Ground Truth storage

"Ground truth" (the photographer's actual Keep/Reject/Neutral decision
for a specific image, as opposed to the algorithm's suggestion) is
stored in `annotations/review.db`, keyed by image content hash (not
path — so a moved or renamed file is still recognized). This is the
single most important piece of local state in the whole project: it
represents your own accumulated review decisions and **is not
recoverable if lost** unless you have a backup (see Section 15).
`picklikeme.ground_truth` / the desktop app's "Set User Decisions by
Subfolders" workflow can bulk-populate it from an existing
Select/Reject folder structure.

### 10.6 User preferences, recent folders, window layouts

All covered by QSettings — see 10.1 above. There is no separate
preferences file.

---

## 11. Development Workflow

Recommended day-to-day loop:

```bash
# 1. Start from an up-to-date main
cd ~/Code/PickLikeMe
git checkout main
git pull

# 2. (Optional) create a branch for a specific piece of work
git checkout -b your-feature-name

# 3. Activate the environment
source .venv/bin/activate

# 4. Implement the change

# 5. Run the relevant tests as you go, and the full suite before committing
python -m pytest tests/ -q

# 6. Manually validate anything UI-visible by actually running the app
python -m picklikeme.desktop

# 7. Review what you're about to commit
git status
git diff

# 8. Commit
git add <specific files>
git commit -m "Describe what changed and why"

# 9. Push
git push -u origin your-feature-name   # first push of a new branch
git push                                # subsequent pushes
```

Working directly on `main` (skipping step 2) is fine for small,
low-risk personal changes, matching how this repository has actually
been developed so far (a single-developer personal project) — use a
branch when you want to keep `main` always in a known-good state while
you experiment, or if you're picking up someone else's in-progress work.

### Developing with Claude Code

This project has been developed extensively with Claude Code. A
productive way to work with it:

1. From the repository root, with your virtual environment active, run
   `claude` to start a session.
2. Describe the task, referencing specific files/behavior where you
   already know them — Claude Code can explore the codebase itself, but
   pointing it at the right module (e.g. "the Image Explorer in
   `desktop/dialogs/analytics_dashboard.py`") saves time.
3. Let it read relevant code, make changes, and run tests itself —
   it has access to the same `pytest`/shell tools described in this
   document.
4. Review its diffs before accepting them, the same as you would review
   a human contributor's changes — `git diff` or VS Code's built-in
   diff view both work well for this.
5. Ask it to run the full test suite (Section 12) before you consider a
   change complete, and to add regression tests for any bug fix or new
   behavior — this codebase's existing tests consistently follow that
   pattern (one test file per module, docstrings explaining *why* a
   test exists, not just what it checks), and new work should match it.
6. For anything destructive (force-push, `git reset --hard`, deleting
   real data under `cache/`/`annotations/`), Claude Code will ask for
   confirmation before proceeding — don't disable that.

---

## 12. Testing

### 12.1 Run the complete test suite

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

`-q` (quiet) prints a compact dot-per-test progress indicator instead of
one line per test — recommended for the full run, since there are over
1,300 tests. Expect this to take a few minutes. A successful run ends
with a summary line like:

```
1341 passed, 13 subtests passed in 171.23s
```

No test configuration file (`pytest.ini`, `pyproject.toml`'s
`[tool.pytest.ini_options]`, `conftest.py`) exists at the repository
root — `pytest` uses its own defaults, discovering every `test_*.py`
file under `tests/`.

### 12.2 Run individual tests

By file:

```bash
python -m pytest tests/test_bird_crop.py -q
```

By a single test function/method (append `::name`, and `::ClassName::method_name`
for a test inside a `unittest.TestCase` class):

```bash
python -m pytest tests/test_bird_crop.py::test_the_largest_detection_wins -q
python -m pytest tests/test_annotations.py::CaptureTimestampTests::test_reads_the_exif_datetimeoriginal_tag -q
```

By keyword match across the whole suite (matches test names, not file
content):

```bash
python -m pytest tests/ -k "burst and not analytics" -q
```

Drop `-q` (or use `-v`) for verbose, one-line-per-test output — useful
when a subset is failing and you want to see exactly which ones:

```bash
python -m pytest tests/test_bird_crop.py -v
```

### 12.3 Run the desktop smoke tests

`tests/test_desktop_smoke.py` specifically verifies the desktop app's
major windows/dialogs construct without crashing (offscreen, no visible
window needed):

```bash
python -m pytest tests/test_desktop_smoke.py -v
```

Most desktop-related tests (there are many beyond this one file — search
`tests/` for `test_desktop_*.py` and `test_analytics_dashboard.py`) run
with Qt's `offscreen` platform plugin automatically (set via
`QT_QPA_PLATFORM=os.environ.setdefault(...)` inside the test files
themselves) — you do not need a real display or to have the app
actually visible for these to pass, which also makes them safe to run
over SSH or in a headless CI environment.

### 12.4 Perform manual QA

Automated tests verify *code* correctness; they cannot verify that a UI
*feels* right or that a visual overlay actually looks correct. For any
change touching the desktop app's UI:

1. Launch the real app (`python -m picklikeme.desktop`) — never rely on
   test-passing alone as evidence a UI change works.
2. Exercise the actual feature you changed, including edge cases (an
   empty folder, a folder with zero decisions, a very large folder if
   relevant, switching themes, resizing the window).
3. Check for regressions in adjacent, unrelated features you didn't
   mean to touch — it's easy for a shared-widget change to have
   side effects elsewhere.
4. If the change is visual, take a screenshot and compare it against
   what you expected, not just "no exception was thrown."

---

## 13. Troubleshooting

| Problem | Likely cause | Fix |
| --- | --- | --- |
| `zsh: command not found: python3.12` | That specific Python version isn't installed | Run `brew install python@3.12`, or substitute whichever 3.10+ version you actually installed (`python3.11`, `python3`) everywhere in this document. |
| `pip install -e .` fails with a compiler error (often mentioning `rawpy` or a C extension) | Xcode Command Line Tools missing/incomplete | Run `xcode-select --install` (Section 3.1), then retry. |
| `ModuleNotFoundError: No module named 'PySide6'` when launching the app | `PySide6` wasn't installed — either you're on an older checkout from before it was added to `requirements.txt`, or step 5.3 was skipped | `pip install PySide6` (with your `.venv` active), or re-run `pip install -r requirements.txt`. |
| `ModuleNotFoundError: No module named 'picklikeme'` | `pip install -e .` was never run, or you're not in the virtual environment it was installed into | `source .venv/bin/activate` then `pip install -e .`. |
| `picklikeme: command not found` (but `python -m picklikeme` works) | The console script isn't on `PATH` — usually because a different environment is active than the one you installed into, or you're using `sudo`/a different shell | Always prefer `python -m picklikeme ...` (works regardless of `PATH`) over the bare `picklikeme` command — see `config.py`'s own `cli_prefix()` docstring, which exists specifically because of this failure mode. |
| `git clone` fails with a permission/authentication error | You're being asked for GitHub credentials for a public repo, or SSH keys aren't set up | For a public repo, use the `https://` URL exactly as shown in Section 4 (no authentication needed for cloning/pulling); only pushing requires credentials — set those up via `gh auth login` (if you have the GitHub CLI) or a personal access token when you first `git push`. |
| `git pull` fails with "local changes would be overwritten" | You have uncommitted edits to a file that also changed upstream | `git status` to see what's modified; either commit your changes first, or `git stash` them, `git pull`, then `git stash pop`. Never `git checkout -- <file>` or `git reset --hard` without being sure you want to discard those changes permanently. |
| A model download fails/hangs partway (EyePose-v0, SuperAnimal-Bird, BioCLIP) | Network interruption | Downloads write to a temp file and rename atomically on success (Section 6.6), so a failed download never leaves a corrupt cached file — simply re-run the feature; it will retry the download automatically. If it keeps failing, check your internet connection and firewall/VPN settings (Hugging Face's CDN must be reachable). |
| `RuntimeError` mentioning `ultralytics` when running EyePose-v0 for the first time | The one-time `.pt`→ONNX conversion needs `ultralytics`, which isn't a runtime dependency (Section 6.3) | `pip install ultralytics`, retry, then optionally `pip uninstall ultralytics` afterward. |
| Everything in the venv seems broken / bizarre import errors that don't match what's installed | A corrupted or half-upgraded virtual environment | Delete and recreate it: `deactivate` (if active), `rm -rf .venv`, then redo Section 5 from 5.1. This never affects your source code, models, caches, or ground-truth data — only the Python packages. |
| `cache/` or `annotations/` directories seem to be missing entirely | Normal on a fresh clone — neither is stored in git (Section 6/10) | They are created automatically the first time you run the app or any `picklikeme` command that needs them. If you expected data that existed before (a previous checkout, another machine), see Section 15 — you need to restore it from a backup, it doesn't regenerate itself. |
| The desktop app's window is visually broken, stuck showing a stale layout, or a dialog won't open | Corrupted/stale QSettings state | `defaults delete com.PeakPic.PeakPicDesktop` (Section 10.1) resets every remembered preference/window geometry to defaults, then relaunch. |
| The app launches but immediately exits with no window and no visible error | An exception during startup that isn't being shown, or you're accidentally running under `QT_QPA_PLATFORM=offscreen` (used by the test suite) | Launch from a real Terminal window (not from a script that redirects output) and read the full traceback; check `echo $QT_QPA_PLATFORM` is empty/unset before launching manually. |
| `qt.qpa.plugin: Could not load the Qt platform plugin "cocoa"` | A broken/partial PySide6 install, or `QT_QPA_PLATFORM` set incorrectly | `pip uninstall PySide6 && pip install PySide6` inside the active `.venv`; confirm `echo $QT_QPA_PLATFORM` prints nothing (unset it with `unset QT_QPA_PLATFORM` if it's set to something like `offscreen` from a previous test run in the same shell session). |
| Tests pass individually but fail when run as the full suite (or vice versa) | Most likely genuine test-isolation debt in this codebase around `QSettings` (see Section 16 — this is a documented, known limitation, not something new you broke) | Re-run just the failing test in isolation to confirm; if it only fails as part of the full suite, this is a pre-existing issue worth flagging rather than something to chase down yourself. |
| A filename that worked on Windows/another Mac causes an error or "file not found" here | Case-sensitivity or Unicode-normalization difference (Section 14) | Check the exact filename with `ls -la` in the containing folder and compare byte-for-byte against what the code/config expects. |

---

## 14. Mac-specific Notes

Differences worth knowing if you (or documentation you're reading, or
Claude Code's own output) came from a Windows development background —
this project was originally developed on Windows, so some existing
paths/scripts/comments reference Windows conventions directly.

### Paths

- Windows paths in existing docs/examples use backslashes and drive
  letters (`C:\Code Projects\PickLikeMe`, `D:\NewShoot`) — on macOS,
  use forward slashes and no drive letter (`~/Code/PickLikeMe`,
  `/Users/you/Photos/NewShoot`). Every command in *this* document is
  already written in macOS/Unix form.
- `~` expands to your home directory in the shell (`/Users/yourname`)
  but is **not** expanded automatically by Python's `open()` or most
  file dialogs — when a command in other project documentation shows a
  `~`-based path as a `--flag` value, that's fine (the shell expands it
  before Python ever sees it), but don't type a literal `~` inside a
  GUI text field expecting it to resolve.
- The project's own code resolves its important paths (checkpoints,
  caches) relative to the repository root via `Path(__file__).resolve()`
  logic (`config.py`), not relative to your current directory — so most
  commands work correctly regardless of which directory you're in *when
  you launch them*, with the one documented exception of the desktop
  app's `cache/desktop/` subdirectory (Section 10.2), which is
  cwd-relative — always launch from the repository root to be safe.

### Permissions

- macOS Gatekeeper may block a freshly-downloaded or freshly-created
  executable script the first time you try to run it ("cannot be opened
  because the developer cannot be verified" or similar). For
  `Start PeakPic.command`, this is resolved by `chmod +x` (Section 7.2)
  plus, if Gatekeeper still complains on first double-click, right-click
  the file in Finder → **Open** → confirm in the dialog that appears
  (this only needs to happen once per file).
- If macOS asks for permission for Terminal (or VS Code) to access
  Photos, Removable Volumes, or Files and Folders when you try to open a
  folder of RAW photos from an external drive or a protected location —
  grant it via **System Settings → Privacy & Security**. This is a macOS
  sandboxing prompt, unrelated to anything in the PickLikeMe codebase
  itself.

### Finder

- Hidden files/folders (anything starting with `.`, including `.venv`,
  `.git`, `.vscode`) are hidden from Finder by default. Toggle visibility
  with <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>.</kbd> (period) while Finder
  is focused.
- `.command` files (like `Start PeakPic.command`) show up in Finder as
  a generic script icon and, when double-clicked, open in Terminal and
  run — this is standard macOS behavior, not something specific to this
  project.

### Terminal

- Default shell on modern macOS is `zsh`, not `bash`. Every command in
  this document works identically in both. If you deliberately use
  `bash` (e.g. via `bash` explicitly, or an older Mac still defaulting
  to it), that's fine too.
- `Cmd` is used where Windows documentation would say `Ctrl` for most
  system-level shortcuts (copy/paste, new tab, etc.) — but **not**
  inside a running terminal program itself, where `Ctrl+C` (interrupt)
  and `Ctrl+D` (EOF) still mean what they mean on any Unix system,
  matching what `README.md` documents for interrupting a training run.

### Keyboard shortcuts (inside the desktop app and VS Code)

| Action | Windows | macOS |
| --- | --- | --- |
| Copy / Paste | Ctrl+C / Ctrl+V | Cmd+C / Cmd+V |
| Save | Ctrl+S | Cmd+S |
| Find | Ctrl+F | Cmd+F |
| Command Palette (VS Code) | Ctrl+Shift+P | Cmd+Shift+P |
| Quit application | Alt+F4 | Cmd+Q |
| Force quit an unresponsive app | (Task Manager) | Cmd+Option+Esc |

The PySide6 desktop app itself uses Qt's standard key sequences, which
automatically map to the right modifier key per platform — you should
not need any app-specific translation beyond the table above.

### Case-sensitive filesystem issues

**Most Macs ship with a case-insensitive (but case-preserving) file
system by default** (APFS, case-insensitive variant) — same practical
behavior as Windows in this respect, so `Checkpoints/` and
`checkpoints/` would refer to the same folder. However:

- Some Macs (or external/secondary volumes) **are** formatted
  case-sensitive — if you ever create or use one, a mismatch between
  the case used in code (e.g. `cache/eye_models/`, always lowercase in
  this codebase) and what's actually on disk would cause file-not-found
  errors that would silently succeed on a case-insensitive volume.
- **Git itself is always case-sensitive**, regardless of your
  filesystem — two files differing only by case can exist in the
  repository's history even though your Mac's filesystem would treat
  them as one file locally, occasionally causing confusing checkout
  behavior. This hasn't been an issue in this codebase (all paths are
  consistently lowercase), but keep it in mind if you ever rename a
  file changing only its case (`git mv` handles this correctly; a plain
  Finder rename sometimes doesn't register as a change to git at all on
  a case-insensitive volume).

### Qt differences

- PySide6/Qt6 renders natively on macOS (via Cocoa), not through any
  compatibility layer — visual appearance (window chrome, menu bar
  placement at the top of the screen rather than inside the app window,
  native file dialogs) will look different from the Windows screenshots
  you may see in `docs/*.md` phase-delivery write-ups, but functionality
  is identical.
- The application's menu bar appears at the **top of the screen**, not
  inside the app window — this is normal macOS behavior for every
  native app, not specific to PeakPic.
- Dark/light theme switching (the app's own **theme** QSetting, Section
  10.1) is independent of macOS's own system-wide Appearance setting —
  changing one does not automatically change the other unless the app's
  theme preference is explicitly set to follow the system.

### Anything else worth knowing

- The `.venv`/`.venv-1` dual-virtualenv pattern you may see referenced
  in code comments (`config.py`'s `cli_prefix()` docstring) reflects
  this project's original Windows development machine keeping a CUDA
  environment and a CPU-only environment side by side. **This doesn't
  apply on a Mac** — there's no CUDA build to choose between, so a
  single `.venv` (Section 5) is all you need.
- `tools/*.ps1` PowerShell scripts (Section 9) are Windows-authored and
  won't run without PowerShell Core (`brew install powershell`, run
  with `pwsh script.ps1`) — if you need what they do, check what they
  contain and consider a native Python/shell port instead of installing
  PowerShell just for them.

---

## 15. Backup and Recovery

If this Mac is lost, wiped, or you're simply setting up a second
machine, here is the complete recovery procedure. **Read this section
now, before you need it** — the "irreplaceable" flag below is the
whole point.

### What is irreplaceable (back this up separately, regularly)

- `annotations/review.db` (+ any `.db-shm`/`.db-wal` files present) —
  your accumulated ground-truth review decisions and false-negative
  annotations. **Not recoverable from anywhere else.** Copy this file
  off-machine (cloud drive, external disk, Time Machine) on whatever
  schedule matches how much re-review work you're willing to lose.
- `checkpoints/model_checkpoint.pt` (+ `_best.pt`) — your trained AI
  Model, if you've trained one you care about. Re-trainable in
  principle (given the same manifest/split and enough time), but
  expensive to reproduce — back it up if training is slow on your
  hardware.
- Any RAW photo archive itself, and its `data/manifest.parquet` /
  `data/split.csv` (if you've built them) — these are **not** part of
  this git repository at all (see `.gitignore`'s `/data/` exclusion)
  and live wherever you pointed `--select-root`/`--reject-root` at.
  Back these up according to your own photo-archiving practice —
  outside the scope of this document, but do not assume the project
  repository backs them up for you.

### What is fully recoverable (do not bother backing up)

Everything under `.venv/`, `cache/`, and the Hugging Face/Torch Hub
machine-wide caches — all of it is either installed by `pip` or
downloaded automatically (Sections 5–6). The source code itself is
recoverable from GitHub at any time.

### Full recovery procedure on a brand-new Mac

1. **Install prerequisite software** — Section 3 in full (Xcode CLT,
   Homebrew, Git, Python, VS Code, ExifTool, Claude Code).
2. **Clone the repository** — Section 4:
   ```bash
   git clone https://github.com/HilaTheCode/PickLikeMe.git
   cd PickLikeMe
   ```
3. **Restore your irreplaceable data**, before running anything, into
   the exact same relative locations:
   - Copy your backed-up `review.db` (and `-shm`/`-wal` if present) into
     `annotations/` (create the folder if it doesn't exist:
     `mkdir -p annotations`).
   - Copy your backed-up `model_checkpoint.pt`/`_best.pt` into
     `checkpoints/` (`mkdir -p checkpoints`).
   - Restore your RAW photo archive to wherever you intend to point
     `--select-root`/`--reject-root`/the desktop app's "Open Folder" at
     (this can be any location — it doesn't need to be inside the repo).
4. **Set up the Python environment** — Section 5 in full (create
   `.venv`, activate, `pip install -r requirements.txt`, `pip install -e .`,
   `pip install onnxruntime`).
5. **Verify installation** — Section 5.4's verification commands, all
   must succeed.
6. **Run the test suite** — Section 12.1:
   ```bash
   python -m pytest tests/ -q
   ```
   A clean pass confirms the environment itself is sound before you
   trust it with real work.
7. **Launch the application** — Section 7:
   ```bash
   python -m picklikeme.desktop
   ```
   Open your restored photo folder and confirm your restored ground
   truth/checkpoint are recognized (previously-reviewed images should
   show their correct Keep/Reject/Neutral status immediately, since
   that's read from the restored `annotations/review.db`).
8. **Let models re-download as needed** — Section 6.4; this happens
   automatically the first time you use each feature and requires
   internet access, same as it did on your original machine.

At that point you are fully back to where you were.

---

## 16. Future Development

### Completed phases

Based on this document's Section 1 status table and the delivery
write-ups under `docs/`:

- The core CLI pipeline: ingestion, preprocessing (Vision Cache),
  training with checkpointing/resume, ranking, auto-crop for Lightroom.
- Preprocessing improvements V2 (letterbox) and backbone upgrade V3
  (DINOv3-Huge+), both now the shipped defaults.
- The evaluation/analyzer package (`picklikeme analyze`/`annotate`) —
  substantially matching roadmap V9's own description.
- Classic Vision Ranking, as a complete, independent, non-learned
  ranking strategy with two interchangeable eye-detector backends
  (EyePose-v0, SuperAnimal-Bird).
- Burst Analysis (capture-time clustering + in-burst ranking) and the
  Gallery's Collapse Bursts feature.
- The full PeakPic desktop application: Gallery, Loupe, Detector Boxes
  overlay, Color Source selection, and a substantial Analytics
  Dashboard (User vs Algorithm agreement, Run Summary, Species
  Analytics, Burst Analytics, an Image Explorer with combinable
  filters, a Visual Debug overlay system, and Score Explanation).
- Species classification and organization ("Arrange by Species") via
  BioCLIP/BioCLIP-2.

### Remaining technical debt

Documented explicitly elsewhere in this codebase, worth reading before
picking a next task:

- **QSettings test isolation** (`docs/Desktop_UX_Redesign_Plan.md`):
  every construction of `MainWindow` — real app, pytest, or an ad-hoc
  script — reads/writes the same real per-machine `QSettings` store
  (Section 10.1). This has caused test runs to leave artifacts in real
  user preferences before, and is a known, not-yet-fixed gap. A proper
  fix would inject an isolated `QSettings::IniFormat` instance pointed
  at a temp path for tests, rather than the real native-format store.
- **This onboarding pass's own findings** (fixed alongside this
  document, but worth knowing the *shape* of the gap for next time a
  similar one appears): `PySide6` and `open_clip_torch` were both hard
  runtime dependencies of the desktop app that were undeclared in
  `pyproject.toml`/`requirements.txt` — a reminder that dependency
  declarations can silently drift from what the code actually imports.
  Consider periodically diffing `pip freeze` inside a truly fresh venv
  built only from `requirements.txt` against what a full `pytest`
  run + `python -m picklikeme.desktop` actually needs.
- **No results ledger**: `docs/roadmap.md` references a
  `docs/results/vN_results.md` file per validated version; no
  `docs/results/` directory currently exists.
- **No macOS/Linux launcher previously existed** (`Start PeakPic.command`
  was added alongside this document) and there was no documented Apple
  Silicon/MPS guidance anywhere in the project (Section 2) — MPS
  acceleration for training remains entirely unimplemented in this
  codebase, not merely undocumented.

### Future roadmap

`docs/roadmap.md`'s remaining, unimplemented versions, in the order the
roadmap itself presents them (each is meant to be validated in
isolation before starting the next):

- **V4** — pairwise/margin ranking loss, replacing the current MSE loss.
- **V5** — burst-aware weighted pair sampling during training.
- **V6** — input resolution study (384 vs 512 vs 640px).
- **V7** — data augmentation.
- **V8** — hard negative mining.
- **V10** — auxiliary explainability heads (sharpness/exposure/motion-blur/
  eye-visibility/size confidence signals surfaced alongside the main
  score).

Beyond the versioned roadmap, the desktop app's Analytics Dashboard
(Image Explorer, Visual Debug, Score Explanation, Burst Analytics, User
vs Algorithm) is itself a continuously-developed area — check `docs/`
for the most recent phase-delivery write-up before assuming a piece of
functionality doesn't exist yet.

### Suggested development priorities

In rough order of leverage-to-effort, based on what's documented as
outstanding above:

1. Fix the QSettings test-isolation gap — it's a known, explicitly
   documented source of test flakiness and real-preferences pollution,
   and is a contained, well-scoped piece of work.
2. Start the `docs/results/` ledger the roadmap already expects, even
   retroactively for whichever version is currently the shipped default
   (V3) — this gives every future roadmap version a template and a
   comparison baseline.
3. Pick one roadmap version (V4 is the natural next step, since it's
   the first not yet confirmed shipped) and implement it in isolation,
   following the roadmap's own "one change per version" discipline.
4. If Mac-based training turns out to matter in practice (not just
   smoke-testing), investigate an MPS (Apple Silicon GPU) backend path
   — currently entirely unimplemented, and would need its own design
   discussion given `ProjectConfig.device`'s current `"cuda"`-or-CPU-only
   assumption.
