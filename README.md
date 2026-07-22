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

### 2. Build a manifest from a Select/Reject folder pair

```bash
python -m picklikeme.ingest.cli build-manifest --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject" --manifest-path data/manifest.parquet --labels-csv data/labels.csv
```

### 3. Create the frozen evaluation split (once)

```bash
python -m picklikeme.split --labels data/labels.csv --output data/split.csv
```

The split is assigned per burst (never per image) and is frozen: every model
version trains and evaluates against the same split so results are comparable.
The command refuses to overwrite an existing split unless `--force` is given.

### 4. Train the model

```bash
python -m picklikeme.train --select-root "C:\\path\\to\\select" --reject-root "C:\\path\\to\\reject" --labels data/labels.csv --split data/split.csv
```

With `--split`, training uses only train-split images and afterwards reports the
protocol metrics (Top-1/Top-3 burst accuracy, ROC AUC, precision/recall) on the
held-out test split, writing them to `evaluation_metrics.json`.

`--resize-mode stretch` reproduces the V1 baseline preprocessing; the default
`letterbox` is the V2 aspect-ratio-preserving behavior.

## Project structure

- docs/architecture.md: architecture rationale and design decisions
- docs/roadmap.md: the V1-V10 version roadmap and evaluation protocol
- src/picklikeme/: Python package containing the training and ingestion pipeline
- tests/: unit tests

## Notes

This project is still in an early prototype stage and is designed for personal experimentation with a custom photo-selection workflow.
