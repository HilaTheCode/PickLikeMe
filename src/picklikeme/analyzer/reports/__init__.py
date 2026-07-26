"""Report renderers. Each consumes an AnalysisResult and writes files; none
computes a metric, so every format shows the same numbers by construction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..analysis import AnalysisResult
from .text import render_full, render_summary, write_text_report

logger = logging.getLogger(__name__)


def write_json_report(result: AnalysisResult, output_path: Path) -> Path:
    """Machine-readable record of the whole analysis.

    This is what a CI job diffs between model versions, so it contains
    everything - including the config that produced it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    return output_path


def write_csv_reports(result: AnalysisResult, output_dir: Path) -> list[Path]:
    """Per-image CSVs for the mistake categories, for spreadsheet triage."""
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    categories = {
        "false_positives": result.errors.false_positives,
        "false_negatives": result.errors.false_negatives,
        "borderline": result.errors.borderline,
        "rank_disagreements": result.errors.largest_rank_disagreements,
        "most_surprising": result.errors.most_surprising,
    }
    for name, records in categories.items():
        if not records:
            continue
        path = output_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].as_dict().keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(record.as_dict())
        written.append(path)
    return written


__all__ = [
    "render_full",
    "render_summary",
    "write_csv_reports",
    "write_json_report",
    "write_text_report",
]
