"""Frozen burst-level train/test split.

The evaluation protocol in docs/roadmap.md requires one held-out split,
created once and reused unchanged by every model version, so that metric
deltas between versions are attributable to the version's single change.

The split is assigned per burst, never per image: frames from one burst are
near-duplicates, so an image-level split would leak test bursts into
training and inflate every metric. Images without a burst_id are treated as
singleton bursts. Assignment is deterministic for a given (manifest, seed,
fraction), and the CLI refuses to overwrite an existing split file unless
--force is given, because regenerating the split invalidates all previously
recorded version results.

The input is the manifest produced by picklikeme.ingest.cli build-manifest
(image_path, label, burst_id, ...); there is no separate labels CSV to keep
in sync with it.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .dataset import load_table

DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SEED = 42

TRAIN = "train"
TEST = "test"


def assign_burst_splits(
    manifest: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if "image_path" not in manifest.columns:
        raise ValueError("manifest frame must contain an image_path column")

    frame = manifest.copy()
    has_burst = "burst_id" in frame.columns
    group_keys: list[str] = []
    for row in frame.itertuples(index=False):
        burst = getattr(row, "burst_id", None) if has_burst else None
        if burst is None or (isinstance(burst, float) and pd.isna(burst)) or str(burst) == "":
            group_keys.append(f"__solo__::{getattr(row, 'image_path')}")
        else:
            group_keys.append(str(burst))
    frame["_group"] = group_keys

    group_sizes = frame.groupby("_group").size()
    groups = sorted(group_sizes.index)
    rng = random.Random(seed)
    rng.shuffle(groups)

    test_target = test_fraction * len(frame)
    test_groups: set[str] = set()
    assigned = 0
    for group in groups:
        if assigned >= test_target:
            break
        test_groups.add(group)
        assigned += int(group_sizes[group])

    frame["split"] = [TEST if g in test_groups else TRAIN for g in frame["_group"]]
    return frame.drop(columns=["_group"])


def create_split(
    manifest_path: str | Path,
    output_path: str | Path,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> pd.DataFrame:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Split file already exists at {output_path}. The split is frozen by design; "
            "pass force=True (--force) only if you intend to invalidate all recorded results."
        )
    manifest = load_table(manifest_path)
    split = assign_burst_splits(manifest, test_fraction=test_fraction, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(output_path, index=False)
    return split


def load_split(split_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(split_path)
    missing = {"image_path", "split"} - set(frame.columns)
    if missing:
        raise ValueError(f"Split file {split_path} is missing columns: {sorted(missing)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the frozen burst-level train/test split")
    parser.add_argument("--manifest", default="data/manifest.parquet", help="Manifest parquet with image_path,label,burst_id (from build-manifest)")
    parser.add_argument("--output", default="data/split.csv", help="Where to write the split file")
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing split (invalidates recorded results)")
    args = parser.parse_args()

    split = create_split(
        args.manifest,
        args.output,
        test_fraction=args.test_fraction,
        seed=args.seed,
        force=args.force,
    )
    counts = split["split"].value_counts()
    print(f"Wrote split to {args.output}")
    print(f"  train images: {counts.get(TRAIN, 0)}")
    print(f"  test images:  {counts.get(TEST, 0)}")


if __name__ == "__main__":
    main()
