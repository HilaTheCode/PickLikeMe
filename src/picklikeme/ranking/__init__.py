"""The ranking-strategy registry.

One place that knows which strategies exist. The desktop Rank menu, and
anything else that offers a choice, reads `available_strategies()` and
resolves a selection through `get_strategy()` - so adding a strategy is
adding a module and one line in `_STRATEGIES` below, with no UI change
beyond it appearing in the menu.

`DEFAULT_STRATEGY_ID` is the AI model, and stays that way: strategies were
introduced to give the trained model company, not a competitor promoted over
it.

Constructing a strategy is cheap by design - none of them load a model in
`__init__` - so building the menu never pulls in torch. The heavy imports all
live inside `rank_folder`.
"""

from __future__ import annotations

from .ai_model import AIModelParams, AIModelStrategy
from .base import (
    GROUP_THRESHOLDS,
    GROUP_WEIGHTS,
    ParamSpec,
    RankingStrategy,
    StrategyInfo,
    WeightedParams,
)
from .classic import (
    ClassicVisionBirdFusionParams,
    ClassicVisionBirdFusionStrategy,
    ClassicVisionEyePoseParams,
    ClassicVisionEyePoseStrategy,
    ClassicVisionMammalFusionParams,
    ClassicVisionMammalFusionStrategy,
    ClassicVisionParams,
    ClassicVisionStrategy,
)
from .combined import ClassicVisionCombinedParams, ClassicVisionCombinedStrategy

DEFAULT_STRATEGY_ID = AIModelStrategy.info.strategy_id

# Ordered: this is the order the Rank menu offers them in, default first.
# The Classic Vision entries are separate, independently selectable
# strategies (see ranking.classic's module docstring) - not one strategy
# with a hidden backend switch - so the photographer always knows exactly
# which algorithm a run used, and every backend's results coexist on a
# folder for direct comparison. EyePose-v0 is listed first among the
# single-model Bird backends as the recommended one for new analyses;
# SuperAnimal-Bird stays fully available. The two Fusion strategies (see
# eyes.domains - Ranking Mode = Birds/Mammals) are listed after the
# established single-model Bird backends: Birds-Fusion combines EyePose-v0
# with SuperAnimal-Bird through the shared Fusion layer, Mammals-Fusion is
# the new safari-workflow entry point (SuperAnimal-Quadruped) - see
# eyes.superanimal_quadruped for why EyePose-v0 is deliberately not used
# there.
_STRATEGIES: tuple[type, ...] = (
    AIModelStrategy,
    ClassicVisionEyePoseStrategy,
    ClassicVisionStrategy,
    ClassicVisionBirdFusionStrategy,
    ClassicVisionMammalFusionStrategy,
    ClassicVisionCombinedStrategy,
)


def available_strategies() -> list[StrategyInfo]:
    """Every registered strategy, in menu order. Cheap - reads class
    attributes, constructs nothing."""
    return [strategy.info for strategy in _STRATEGIES]


def score_labels() -> dict[str, str]:
    """strategy_id -> the short label its score is shown under.

    For UI that displays several modules' scores side by side. A caller must
    fall back to the raw id for anything not in here: a folder may carry
    results from a module that has since been removed, and dropping those
    numbers would be worse than showing them under an unfriendly name.
    """
    return {s.info.strategy_id: (s.info.score_label or s.info.display_name) for s in _STRATEGIES}


def metric_labels() -> dict[str, dict[str, str]]:
    """strategy_id -> {metric_name: label}, for a diagnostics UI that shows
    a module's raw per-metric measurements (see
    `sidecar.discover_metric_reports`) rather than just its combined score.

    Read from each strategy class's own optional `metric_labels` attribute
    (see `ClassicVisionStrategy.metric_labels`), so a strategy that writes no
    metrics report simply is not a key here, and a future one that does
    needs no change to this function - only its own attribute. A caller
    falls back to the raw metric name for anything not covered, the same
    convention `score_labels` already establishes.
    """
    labels: dict[str, dict[str, str]] = {}
    for strategy in _STRATEGIES:
        declared = getattr(strategy, "metric_labels", None)
        if declared:
            labels[strategy.info.strategy_id] = declared
    return labels


def get_strategy(strategy_id: str) -> RankingStrategy:
    """Construct a strategy by id."""
    for strategy in _STRATEGIES:
        if strategy.info.strategy_id == strategy_id:
            return strategy()
    known = ", ".join(s.info.strategy_id for s in _STRATEGIES)
    raise ValueError(f"Unknown ranking strategy {strategy_id!r}. Available: {known}")


__all__ = [
    "DEFAULT_STRATEGY_ID",
    "GROUP_THRESHOLDS",
    "GROUP_WEIGHTS",
    "AIModelParams",
    "AIModelStrategy",
    "ClassicVisionBirdFusionParams",
    "ClassicVisionBirdFusionStrategy",
    "ClassicVisionCombinedParams",
    "ClassicVisionCombinedStrategy",
    "ClassicVisionEyePoseParams",
    "ClassicVisionEyePoseStrategy",
    "ClassicVisionMammalFusionParams",
    "ClassicVisionMammalFusionStrategy",
    "ClassicVisionParams",
    "ClassicVisionStrategy",
    "ParamSpec",
    "RankingStrategy",
    "StrategyInfo",
    "WeightedParams",
    "available_strategies",
    "get_strategy",
    "metric_labels",
    "score_labels",
]
