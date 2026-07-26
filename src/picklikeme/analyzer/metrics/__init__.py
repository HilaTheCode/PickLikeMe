"""Metric plugins, discovered automatically.

Every module in this package is imported on first use, which triggers
`Metric.__init_subclass__` and registers whatever metrics it defines. Dropping a
new `my_metric.py` here is therefore the entire process for adding a metric:
nothing imports it by name, and no report lists metrics explicitly.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

from .base import (
    Counts,
    Metric,
    MetricSet,
    MetricValue,
    compute_all,
    counts_of,
    registered_metrics,
    safe_divide,
)

logger = logging.getLogger(__name__)

_discovered = False


def discover(force: bool = False) -> list[type[Metric]]:
    """Import every metric module in this package exactly once."""
    global _discovered
    if _discovered and not force:
        return list(Metric._registry)

    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_") or module_info.name == "base":
            continue
        try:
            importlib.import_module(f"{__name__}.{module_info.name}")
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not kill the run
            logger.warning("Could not load metric module %r: %s", module_info.name, exc)

    _discovered = True
    logger.debug("Discovered %d metrics", len(Metric._registry))
    return list(Metric._registry)


def all_metrics() -> list[Metric]:
    """Every registered metric instance, discovering on first call."""
    discover()
    return registered_metrics()


def compute(images, metrics=None) -> MetricSet:
    """Compute every discovered metric over a matched dataset."""
    discover()
    return compute_all(images, metrics)


__all__ = [
    "Counts",
    "Metric",
    "MetricSet",
    "MetricValue",
    "all_metrics",
    "compute",
    "compute_all",
    "counts_of",
    "discover",
    "registered_metrics",
    "safe_divide",
]
