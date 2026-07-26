"""Evaluation & analysis for PickLikeMe.

Measures model quality against the photographer's own keep/reject decisions:
matches ranking output to ground truth, computes classification and ranking
metrics, analyses thresholds and calibration, surfaces the model's worst
mistakes, and renders reports.

**Strictly read-only.** Nothing here writes to checkpoints, the crop cache,
source images, training data or ranking output; the only files it creates are
the reports it is asked to produce, under the output directory. The dependency
arrow points one way: the analyzer imports from the pipeline, never the reverse.
"""

from .config import AnalysisConfig
from .matching import MatchedImage, MatchResult, Outcome, match_dataset
from .model import RankedImage

__all__ = [
    "AnalysisConfig",
    "MatchedImage",
    "MatchResult",
    "Outcome",
    "RankedImage",
    "match_dataset",
]
