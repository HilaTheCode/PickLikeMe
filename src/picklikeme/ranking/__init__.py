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
    GROUP_OPTIONS,
    GROUP_THRESHOLDS,
    GROUP_WEIGHTS,
    ParamSpec,
    RankingStrategy,
    StrategyInfo,
    WeightedParams,
    use_subject_filter_spec,
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
from .crop_sharpness import CropSharpnessParams, CropSharpnessStrategy

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
# there. Crop Sharpness is listed last: it is a deliberately different kind
# of signal (whole-crop sharpness, no eye/head detection at all - see
# crop_sharpness.py's own module docstring) rather than another member of
# the eye-detection family the entries above build on.
_STRATEGIES: tuple[type, ...] = (
    AIModelStrategy,
    ClassicVisionEyePoseStrategy,
    ClassicVisionStrategy,
    ClassicVisionBirdFusionStrategy,
    ClassicVisionMammalFusionStrategy,
    ClassicVisionCombinedStrategy,
    CropSharpnessStrategy,
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


def eye_detector_names() -> dict[str, str]:
    """strategy_id -> the `eyes.build_eye_detector` name that strategy's own
    ranking run used, for a caller that needs to check whether a cached
    `eyes.cache.EyeRecord` (see that module's own docstring: one slot per
    image, overwritten by whichever eye detector ran on it last) actually
    belongs to the CURRENTLY selected strategy - `desktop.services.
    ReviewService.eye_keypoints`/`.detection_boxes` is exactly that caller:
    the Loupe's Elements/Boxes overlay must never show a different run's
    result as if it were the selected one (see the "Result / Run
    Consistency" investigation this responds to).

    Read from each strategy class's own `_eye_detector_name` attribute (see
    `ranking.classic.ClassicVisionStrategy`), the same "declared on the
    class, discovered generically" convention `metric_labels` already
    establishes - a strategy with no eye detector at all (the AI model) is
    simply not a key here, which a caller reads as "this strategy has no
    eye data, full stop," not "unknown."
    """
    names: dict[str, str] = {}
    for strategy in _STRATEGIES:
        declared = getattr(strategy, "_eye_detector_name", None)
        if declared:
            names[strategy.info.strategy_id] = declared
    return names


def get_strategy(strategy_id: str) -> RankingStrategy:
    """Construct a strategy by id."""
    for strategy in _STRATEGIES:
        if strategy.info.strategy_id == strategy_id:
            return strategy()
    known = ", ".join(s.info.strategy_id for s in _STRATEGIES)
    raise ValueError(f"Unknown ranking strategy {strategy_id!r}. Available: {known}")


__all__ = [
    "DEFAULT_STRATEGY_ID",
    "GROUP_OPTIONS",
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
    "CropSharpnessParams",
    "CropSharpnessStrategy",
    "ParamSpec",
    "RankingStrategy",
    "StrategyInfo",
    "WeightedParams",
    "available_strategies",
    "eye_detector_names",
    "get_strategy",
    "metric_labels",
    "score_labels",
]
