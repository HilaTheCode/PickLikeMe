"""Analyzer configuration.

One frozen dataclass carries every tunable, so the CLI, a config file and a
library caller all configure the analyzer the same way and the business logic
never reads argv or the environment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..config import PROJECT_ROOT

DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# Percentile cut-offs for top-K ranking quality. These are the fractions a
# photographer actually culls at: "show me the best 1%..30% of the shoot".
DEFAULT_TOP_PERCENTS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)


@dataclass(frozen=True)
class AnalysisConfig:
    """Everything the analyzer needs to know, in one place.

    Paths stay `Path`; percentages are 0-100 (not fractions) because that is
    how they are written on the command line and printed in reports.
    """

    ranking_path: Path
    selected_root: Path | None = None
    rejected_root: Path | None = None
    output_dir: Path = DEFAULT_ANALYSIS_DIR
    report_title: str = "PickLikeMe model analysis"

    # Decision threshold applied to the probability (or score) to turn a
    # ranking into a keep/reject prediction.
    threshold: float = 0.5

    top_percents: tuple[float, ...] = DEFAULT_TOP_PERCENTS

    # An image whose probability falls inside this band is "borderline": the
    # model has no real opinion, which makes it the most informative thing to
    # label next.
    borderline_low: float = 0.45
    borderline_high: float = 0.55

    # How many rows each per-category report and contact sheet shows.
    max_examples: int = 60
    thumbnail_size: int = 200
    contact_sheet_columns: int = 6

    html_report: bool = True
    contact_sheets: bool = True
    charts: bool = True

    # Optional second ranking file; when set, the analyzer runs in comparison
    # mode and reports what changed between the two models.
    compare_ranking_path: Path | None = None
    compare_label: str = "candidate"
    baseline_label: str = "baseline"

    threshold_steps: int = 101
    optimize_for: str = "f1"
    verbose: bool = False
    thumbnail_workers: int = 8

    # False-negative knowledge base. The database deliberately defaults outside
    # output_dir: output directories are per-run, and annotations must outlive
    # every one of them. None means "use the project default".
    annotations_db: Path | None = None
    annotations_enabled: bool = True

    # False-negative diagnostic overlay: draw the detector's boxes on the FN
    # thumbnails. Applies to no other report section.
    annotate_detections: bool = True
    # Where preprocessing recorded its detections. None means the project default.
    crop_cache_dir: Path | None = None
    # Detect only for false negatives that have no recorded boxes (an existing
    # cache predates the recording). Set False to keep the analyzer strictly
    # inference-free.
    detect_missing_boxes: bool = True
    detection_conf_threshold: float = 0.30
    detection_device: str = "cpu"
    # Cache of backfilled boxes; separate from the annotation database so
    # --no-annotations really does leave that file alone.
    detections_db: Path | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.borderline_low >= self.borderline_high:
            raise ValueError(
                f"borderline_low ({self.borderline_low}) must be below "
                f"borderline_high ({self.borderline_high})"
            )
        if self.threshold_steps < 2:
            raise ValueError(f"threshold_steps must be at least 2, got {self.threshold_steps}")
        if self.optimize_for not in OPTIMIZATION_TARGETS:
            raise ValueError(
                f"optimize_for must be one of {sorted(OPTIMIZATION_TARGETS)}, got {self.optimize_for!r}"
            )

    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"

    @property
    def sheets_dir(self) -> Path:
        return self.output_dir / "contact_sheets"

    @property
    def thumbnails_dir(self) -> Path:
        return self.output_dir / "thumbnails"

    @property
    def comparison_mode(self) -> bool:
        return self.compare_ranking_path is not None

    @property
    def detections_db_path(self) -> Path:
        from .detections import DEFAULT_DETECTIONS_DB

        return self.detections_db or DEFAULT_DETECTIONS_DB

    @property
    def annotations_db_path(self) -> Path:
        from .annotations import DEFAULT_ANNOTATIONS_DB

        return self.annotations_db or DEFAULT_ANNOTATIONS_DB

    def to_dict(self) -> dict:
        """JSON-ready form, embedded in reports so a result can always be
        traced back to the settings that produced it."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                out[f.name] = str(value)
            elif isinstance(value, tuple):
                out[f.name] = list(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_file(cls, config_path: str | Path, **overrides) -> "AnalysisConfig":
        """Load a JSON config file. Explicit `overrides` (from the CLI) win, so
        a shared file can be kept under version control and tweaked per run."""
        config_path = Path(config_path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")

        path_fields = {
            "ranking_path",
            "selected_root",
            "rejected_root",
            "output_dir",
            "compare_ranking_path",
            "annotations_db",
            "crop_cache_dir",
            "detections_db",
        }
        kwargs = {}
        for key, value in data.items():
            if key in path_fields and value is not None:
                kwargs[key] = Path(value)
            elif key == "top_percents" and value is not None:
                kwargs[key] = tuple(float(v) for v in value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


# Targets a threshold sweep can be optimised for. Each maps to a key produced
# by thresholds.evaluate_threshold, so adding a target is a one-line change.
OPTIMIZATION_TARGETS: dict[str, str] = {
    "f1": "Maximise F1 (balance precision and recall)",
    "balanced_accuracy": "Maximise balanced accuracy (equal weight to both classes)",
    "accuracy": "Maximise raw accuracy (dominated by the majority class)",
    "precision": "Maximise precision (minimise wasted keeps)",
    "recall": "Maximise recall (minimise missed keepers)",
    "mcc": "Maximise Matthews correlation (robust to class imbalance)",
    "youden_j": "Maximise Youden's J (recall + specificity - 1)",
}
