"""Core data types shared across the analyzer.

Kept free of I/O and of any pipeline import so every other analyzer module can
depend on it without creating cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Outcome(str, Enum):
    """Where one image lands in the confusion matrix.

    UNKNOWN is a first-class outcome, not an error: a ranking file routinely
    contains images that are in neither ground-truth folder (moved, deleted,
    or ranked from a different shoot). Those must be reported and excluded from
    metrics, never allowed to abort the analysis.
    """

    TRUE_POSITIVE = "true_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    UNKNOWN = "unknown"

    @property
    def is_error(self) -> bool:
        return self in (Outcome.FALSE_POSITIVE, Outcome.FALSE_NEGATIVE)

    @property
    def short(self) -> str:
        return {
            Outcome.TRUE_POSITIVE: "TP",
            Outcome.TRUE_NEGATIVE: "TN",
            Outcome.FALSE_POSITIVE: "FP",
            Outcome.FALSE_NEGATIVE: "FN",
            Outcome.UNKNOWN: "??",
        }[self]


@dataclass(frozen=True)
class RankedImage:
    """One row of a ranking file, after field auto-detection.

    Only `image_path`, `score` and `rank` are guaranteed. Everything else is
    optional because different producers emit different columns, and the
    analyzer degrades gracefully rather than demanding a schema: metrics that
    need probabilities simply do not run when probabilities are absent.
    """

    image_path: str
    score: float
    rank: int
    label: int | None = None
    probability: float | None = None
    predicted_class: int | None = None
    confidence: float | None = None

    @property
    def filename(self) -> str:
        return Path(self.image_path).name

    @property
    def path(self) -> Path:
        return Path(self.image_path)


@dataclass(frozen=True)
class MatchedImage:
    """A ranked image joined to its ground-truth label and scored outcome."""

    ranked: RankedImage
    truth: int | None
    predicted: int
    outcome: Outcome

    @property
    def image_path(self) -> str:
        return self.ranked.image_path

    @property
    def filename(self) -> str:
        return self.ranked.filename

    @property
    def score(self) -> float:
        return self.ranked.score

    @property
    def rank(self) -> int:
        return self.ranked.rank

    @property
    def probability(self) -> float | None:
        return self.ranked.probability

    @property
    def confidence(self) -> float | None:
        """How strongly the model committed, in [0.5, 1.0].

        Derived from the probability rather than the raw score so it is
        comparable across images; None when no probability is available.
        """
        probability = self.ranked.probability
        if self.ranked.confidence is not None:
            return self.ranked.confidence
        if probability is None:
            return None
        return max(probability, 1.0 - probability)

    @property
    def is_error(self) -> bool:
        return self.outcome.is_error
