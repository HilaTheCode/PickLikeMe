"""Capability 16 - the metric plugin interface and registry.

A metric is a class with a name, a description, and a `compute` that turns a
matched dataset into a MetricValue. Subclassing registers it; the package's
`__init__` imports every module in the folder, so **adding a metric means
adding one file** and nothing else - no registry edit, no report edit, because
reports render whatever the registry yields.

`applies_to` lets a metric opt out cleanly (ROC AUC without probabilities, or
with only one class present) instead of returning a misleading number.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Iterable, Sequence

from ..model import MatchedImage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricValue:
    """One computed metric, ready to render.

    `value` is None when the metric legitimately does not apply; `detail`
    carries the reason so a report can say *why* a number is missing rather
    than printing a bare "n/a".
    """

    name: str
    value: float | None
    description: str = ""
    detail: str = ""
    category: str = "general"
    higher_is_better: bool = True
    fmt: str = "{:.4f}"

    def rendered(self) -> str:
        if self.value is None:
            return "n/a"
        return self.fmt.format(self.value)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "rendered": self.rendered(),
            "description": self.description,
            "detail": self.detail,
            "category": self.category,
            "higher_is_better": self.higher_is_better,
        }


class Metric(ABC):
    """Base class for every metric. Subclasses are auto-registered."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[str] = "general"
    higher_is_better: ClassVar[bool] = True
    fmt: ClassVar[str] = "{:.4f}"
    # Metrics with a lower sort_key are rendered first within their category.
    sort_key: ClassVar[int] = 100

    _registry: ClassVar[list[type["Metric"]]] = []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract intermediates (no name) are scaffolding, not metrics.
        if getattr(cls, "name", ""):
            Metric._registry.append(cls)

    def applies_to(self, images: Sequence[MatchedImage]) -> tuple[bool, str]:
        """(applicable, reason-if-not). Default: needs at least one image."""
        if not images:
            return False, "no matched images"
        return True, ""

    @abstractmethod
    def compute(self, images: Sequence[MatchedImage]) -> float | None:
        """The metric itself, over ground-truth-matched images only."""

    def render(self, images: Sequence[MatchedImage]) -> MetricValue:
        """Compute and wrap, converting a failure into a reported detail
        rather than an exception - one broken metric must not lose a report."""
        applicable, reason = self.applies_to(images)
        if not applicable:
            return MetricValue(
                name=self.name,
                value=None,
                description=self.description,
                detail=f"not applicable: {reason}",
                category=self.category,
                higher_is_better=self.higher_is_better,
                fmt=self.fmt,
            )
        try:
            value = self.compute(images)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            logger.warning("Metric %s failed: %s", self.name, exc)
            return MetricValue(
                name=self.name,
                value=None,
                description=self.description,
                detail=f"failed: {type(exc).__name__}: {exc}",
                category=self.category,
                higher_is_better=self.higher_is_better,
                fmt=self.fmt,
            )
        return MetricValue(
            name=self.name,
            value=value,
            description=self.description,
            detail="",
            category=self.category,
            higher_is_better=self.higher_is_better,
            fmt=self.fmt,
        )


@dataclass
class MetricSet:
    """The computed metrics for one dataset, grouped for rendering."""

    values: list[MetricValue] = field(default_factory=list)

    def by_name(self, name: str) -> MetricValue | None:
        return next((value for value in self.values if value.name == name), None)

    def get(self, name: str, default: float | None = None) -> float | None:
        value = self.by_name(name)
        return default if value is None or value.value is None else value.value

    @property
    def categories(self) -> list[str]:
        seen: list[str] = []
        for value in self.values:
            if value.category not in seen:
                seen.append(value.category)
        return seen

    def in_category(self, category: str) -> list[MetricValue]:
        return [value for value in self.values if value.category == category]

    def as_dict(self) -> dict:
        return {value.name: value.as_dict() for value in self.values}


def registered_metrics() -> list[Metric]:
    """Every discovered metric, instantiated, in stable render order."""
    instances = [cls() for cls in Metric._registry]
    instances.sort(key=lambda m: (m.category, m.sort_key, m.name))
    return instances


def compute_all(images: Sequence[MatchedImage], metrics: Iterable[Metric] | None = None) -> MetricSet:
    """Run every registered metric over the matched dataset."""
    chosen = list(metrics) if metrics is not None else registered_metrics()
    return MetricSet(values=[metric.render(images) for metric in chosen])


# ---------------------------------------------------------------------------
# Shared helpers - every metric derives its counts from here so the confusion
# matrix is defined exactly once.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def actual_positive(self) -> int:
        return self.tp + self.fn

    @property
    def actual_negative(self) -> int:
        return self.tn + self.fp

    @property
    def predicted_positive(self) -> int:
        return self.tp + self.fp

    @property
    def predicted_negative(self) -> int:
        return self.tn + self.fn


def counts_of(images: Sequence[MatchedImage]) -> Counts:
    from ..model import Outcome

    tally = {outcome: 0 for outcome in Outcome}
    for image in images:
        tally[image.outcome] += 1
    return Counts(
        tp=tally[Outcome.TRUE_POSITIVE],
        fp=tally[Outcome.FALSE_POSITIVE],
        tn=tally[Outcome.TRUE_NEGATIVE],
        fn=tally[Outcome.FALSE_NEGATIVE],
    )


def safe_divide(numerator: float, denominator: float) -> float | None:
    """None rather than 0.0 on an empty denominator: an undefined metric and a
    genuinely zero one mean different things to whoever reads the report."""
    return None if denominator == 0 else numerator / denominator


def labels_and_scores(images: Sequence[MatchedImage]) -> tuple[list[int], list[float]]:
    return [int(image.truth) for image in images], [image.score for image in images]


def both_classes_present(images: Sequence[MatchedImage]) -> bool:
    labels = {image.truth for image in images}
    return {0, 1} <= labels
