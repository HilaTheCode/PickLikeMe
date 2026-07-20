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

### 3. Train the model

```bash
python -m picklikeme.train --raw-root "C:\\path\\to\\raw" --labels data/labels.csv
```

## Project structure

- docs/architecture.md: architecture rationale and design decisions
- src/picklikeme/: Python package containing the training and ingestion pipeline
- tests/: unit tests

## Notes

This project is still in an early prototype stage and is designed for personal experimentation with a custom photo-selection workflow.
