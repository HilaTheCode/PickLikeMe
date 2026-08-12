"""Image item model for the future Qt gallery view."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...sidecar import AI_STRATEGY_ID


@dataclass(slots=True)
class ImageItem:
    path: str
    file_name: str
    # The photographer's own verdict in the legacy three-value spelling -
    # "keep"/"reject"/"neutral". Kept because the card's K/R/N buttons, the
    # status filters and the web page all speak it; `user_decision` below is
    # the same fact in the explicit vocabulary. Both are the USER's decision
    # only: an algorithm cutoff recorded for this image (see
    # ReviewImage.algorithm_decision) never reaches either one.
    review_status: str = "neutral"
    ai_suggestion: str | None = None
    # The same kind of suggestion as ai_suggestion, but for whichever
    # strategy is currently selected (the same selection Burst Analysis and
    # the toolbar's Color Source picker already share - see
    # ReviewSession.burst_strategy's own docstring). Equal to ai_suggestion
    # whenever that strategy IS the AI model. Backs the generalized
    # "Algorithm Keep/Reject" conflict filters (main_window.py's
    # _filter_items) - see ReviewSession.suggestions_for.
    algorithm_suggestion: str | None = None
    selected: bool = False
    # The file's own EXIF capture date/time (ISO-8601), or None if it has
    # none - see ReviewImage.captured_at. ISO-8601 sorts lexicographically
    # in chronological order, so gallery sort-by-capture-time needs no
    # date parsing.
    captured_at: str | None = None
    # Every analysis module's result for this image, keyed by strategy id -
    # {"ai-model": {"score": .., "rank": ..}, "classic-vision": {...}}. The
    # UI-side mirror of ReviewImage.ranking_results: score and rank always
    # belong together as properties of ONE strategy, never a separate global
    # pair. A new module appears here automatically, which is why the gallery
    # card and the Loupe iterate this rather than naming the two that exist
    # today.
    ranking_results: dict[str, dict] = field(default_factory=dict)
    # The UI-side mirror of ReviewImage.filter_reasons - why a strategy did
    # NOT score this image, keyed by strategy id -> reason (e.g.
    # {"classic-vision": "NO_VISIBLE_EYE"}). Empty for an image nothing has
    # filtered, even if nothing has scored it either - those are simply two
    # different kinds of "unranked".
    filter_reasons: dict[str, str] = field(default_factory=dict)
    # The UI-side mirror of ReviewImage.metrics - a strategy's raw
    # per-metric measurements, keyed by strategy id ->
    # {metric_name: value}, for a diagnostics display of what its combined
    # score was actually made of.
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # Burst Analysis's own output (see picklikeme.burst_analysis /
    # ReviewSession.burst_info) - which burst this image belongs to, how
    # many members that burst has, this image's 1-based rank within it (by
    # whichever strategy ReviewSession.burst_strategy currently names), and
    # whether it is that burst's top-ranked member. Always populated - a
    # burst of one is still a burst - so burst_size/rank default to 1 and
    # burst_best to True, matching what an image with no burstmates gets
    # from ReviewImage.as_dict.
    burst_id: str | None = None
    burst_size: int = 1
    burst_rank: int = 1
    burst_best: bool = True
    # A recorded algorithm cutoff for this image ("keep"/"reject"), or None.
    # Informational only - see ReviewImage.algorithm_decision. Deliberately
    # NOT folded into review_status/user_decision: that is the confusion the
    # whole User Decision separation exists to prevent.
    algorithm_decision: str | None = None

    @property
    def user_decision(self) -> str:
        """KEEP / REJECT / UNDECIDED - the photographer's own verdict in the
        explicit three-state vocabulary, which is what the Grid's User
        Decision coloring reads (`design_system.resolve_user_decision`).
        Derived from `review_status`, so the two can never disagree."""
        from ...review.user_decision import normalize

        return normalize(self.review_status)

    @property
    def is_decided(self) -> bool:
        """True only for an explicit user Keep or Reject."""
        from ...review.user_decision import is_decided

        return is_decided(self.review_status)

    @property
    def display_name(self) -> str:
        return Path(self.path).name if self.path else self.file_name

    @property
    def score(self) -> float | None:
        """The AI model's own score - see ReviewImage.score for why this
        stays a fixed name rather than "whichever module ran last": the
        AI cutoff, the AI-suggestion filters and the default sort are all
        defined against this specific strategy's ordering."""
        return self.score_for(AI_STRATEGY_ID)

    @property
    def rank(self) -> int | None:
        """The AI model's own rank - see `score`."""
        return self.rank_for(AI_STRATEGY_ID)

    def score_for(self, strategy_id: str) -> float | None:
        entry = self.ranking_results.get(strategy_id) or {}
        return entry.get("score")

    def rank_for(self, strategy_id: str) -> int | None:
        entry = self.ranking_results.get(strategy_id) or {}
        return entry.get("rank")

    def reason_for(self, strategy_id: str) -> str | None:
        """Why `strategy_id` did not score this image, or None if it was
        never filtered by that strategy (it may simply never have run, or it
        may have scored this image just fine - see `score_for`)."""
        return self.filter_reasons.get(strategy_id)
