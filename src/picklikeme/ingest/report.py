"""Human-readable summary of a manifest build: class balance, burst stats,
metadata problems, and folder-scan anomalies that need manual attention.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .scan import ScanIssues


def summarize(manifest: pd.DataFrame, issues: ScanIssues) -> str:
    lines: list[str] = []
    total = len(manifest)
    lines.append(f"Total images: {total}")
    if total:
        keeps = int((manifest["label"] == 1).sum())
        rejects = int((manifest["label"] == 0).sum())
        lines.append(f"Keep: {keeps} ({keeps / total:.1%})  Reject: {rejects} ({rejects / total:.1%})")
        lines.append(f"Shoots: {manifest['shoot_id'].nunique()}")

    if total and manifest["burst_id"].notna().any():
        burst_sizes = manifest.groupby("burst_id").size()
        lines.append(
            f"Bursts: {burst_sizes.shape[0]} "
            f"(median size {burst_sizes.median():.0f}, max {int(burst_sizes.max())}, "
            f"singletons {int((burst_sizes == 1).sum())})"
        )

    if total and "metadata_status" in manifest:
        for status, count in manifest["metadata_status"].value_counts().items():
            if status != "ok":
                lines.append(f"Metadata issue '{status}': {count} files")

    if issues.unmatched_subfolders:
        lines.append(f"Unmatched subfolders (not recognized as Keep/Reject): {len(issues.unmatched_subfolders)}")
        for p in issues.unmatched_subfolders[:20]:
            lines.append(f"  - {p}")
    if issues.incomplete_shoots:
        lines.append(f"Shoots missing a Keep or Reject side: {len(issues.incomplete_shoots)}")
        for s in issues.incomplete_shoots[:20]:
            lines.append(f"  - {s}")
    if issues.duplicate_filenames:
        lines.append(f"Same filename present in both Keep and Reject: {len(issues.duplicate_filenames)}")
        for shoot_id, a, b in issues.duplicate_filenames[:20]:
            lines.append(f"  - {shoot_id}: {a}  <->  {b}")

    return "\n".join(lines)


def gap_histogram(manifest: pd.DataFrame, out_path: Path) -> bool:
    """Write a histogram of inter-frame gaps to help calibrate the burst gap threshold.

    Returns False (and writes nothing) if there isn't enough timestamped
    data yet to make the histogram meaningful.
    """
    import matplotlib.pyplot as plt

    df = manifest.dropna(subset=["capture_timestamp"]).copy()
    if df.empty:
        return False

    df["capture_dt"] = pd.to_datetime(
        df["capture_timestamp"], format="%Y-%m-%dT%H:%M:%S", errors="coerce"
    ) + pd.to_timedelta(df["subsecond"].fillna(0), unit="ms")

    gaps: list[float] = []
    for _, group in df.groupby(["shoot_id", "camera_model"]):
        deltas = group["capture_dt"].sort_values().diff().dt.total_seconds().dropna()
        gaps.extend(deltas[(deltas > 0) & (deltas < 30)].tolist())

    if not gaps:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(gaps, bins=100)
    plt.xlabel("Gap between consecutive frames within a shoot (seconds)")
    plt.ylabel("Count")
    plt.title("Inter-frame gap distribution (use to calibrate --gap-seconds)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True
