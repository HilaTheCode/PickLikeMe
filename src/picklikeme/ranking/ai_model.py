"""The trained preference model, as one ranking strategy among several.

This is an adapter, not an implementation. `rank.rank_folder` is the
long-standing AI ranking entry point - the CLI (`python -m picklikeme.rank`)
calls it, `ReviewService.rank_folder` called it directly before strategies
existed, and it is unchanged by their introduction. Everything below merely
describes it in the vocabulary the strategy registry speaks (a `StrategyInfo`
for the menu, `ParamSpec`s for the dialog) and forwards the call.

Keeping it an adapter is the point: the AI path must keep behaving exactly as
it always has, so it gets no new code in the ranking hot path - the same
function, with the same arguments, producing the same CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import DEFAULT_CHECKPOINT_PATH
from .base import ParamSpec, StrategyInfo

STRATEGY_ID = "ai-model"


@dataclass(frozen=True)
class AIModelParams:
    """What `RankDialog` has always collected for an AI ranking run.

    Not a `WeightedParams`: the AI model has no weights to balance, and
    pretending otherwise to fit a shared base class would be worse than
    having two shapes of parameter object.
    """

    checkpoint: str = str(DEFAULT_CHECKPOINT_PATH)
    crop_birds: bool = True
    # None means "auto" (CUDA when available), which is what rank_folder has
    # always defaulted to; carried here so the strategy call forwards it
    # exactly as the pre-strategy `ReviewService.rank_folder` did.
    device: str | None = None

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        # Neither parameter is a number on a slider - one is a file path, the
        # other a checkbox - so the AI strategy keeps its own hand-written
        # dialog (RankDialog) rather than a generated one, and declares no
        # numeric specs. The framework supports both; a strategy is not
        # required to be dialog-generatable.
        return ()


class AIModelStrategy:
    """Implements `ranking.base.RankingStrategy` over `rank.rank_folder`."""

    info = StrategyInfo(
        strategy_id=STRATEGY_ID,
        display_name="AI Model",
        description="Score every image with the trained preference model (the default).",
        score_label="AI",
    )
    params_class = AIModelParams
    param_specs: tuple[ParamSpec, ...] = ()

    def rank_folder(
        self,
        input_folder: str | Path,
        *,
        params: AIModelParams | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        force_preprocess: bool = False,
    ) -> dict:
        from ..rank import rank_folder as run_rank_folder

        params = params or AIModelParams()
        result = run_rank_folder(
            input_folder,
            checkpoint=params.checkpoint,
            crop_birds=params.crop_birds,
            device=params.device,
            on_stage=on_stage,
            on_progress=on_progress,
            force_preprocess=force_preprocess,
        )
        # The two keys every strategy reports, added around (never instead of)
        # what rank_folder already returns, so existing consumers of its
        # result dict keep working untouched. The AI model scores every image
        # it is given - it has no filtering phase - so nothing is ever
        # excluded and `filtered` is always empty.
        result.setdefault("strategy", STRATEGY_ID)
        result.setdefault("considered", result.get("image_count", 0))
        result.setdefault("filtered", {})
        return result
