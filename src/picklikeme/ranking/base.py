"""The pluggable ranking-strategy boundary.

A ranking strategy answers one question: given a folder of photographs,
produce a score per image and write the ranking CSV this project already
reads everywhere. What it uses to decide - a trained preference model, a
deterministic computer-vision pipeline, a future ensemble - is entirely its
own business.

Everything downstream of a ranking is already strategy-agnostic and stays
that way: `sidecar.ranking_path` says where the CSV goes, `analyzer.io.load_ranking`
reads it, `review.session.ReviewSession` merges it into the gallery, and the
AI-suggestion threshold works off the resulting order. None of them know or
care which strategy produced the numbers, so adding a strategy never touches
any of them.

The three pieces:

- `StrategyInfo` - what the UI needs to *offer* a strategy without importing
  it. Cheap by construction, so listing strategies never loads torch.
- `RankingStrategy` - what the UI needs to *run* one.
- `ParamSpec` - how a strategy declares its tunable parameters, so the
  parameter dialog is generated from the strategy rather than hand-written
  per strategy. Adding a parameter is adding one `ParamSpec` and one
  dataclass field; no UI code changes.

Sorting is deliberately not a strategy's job. `train.write_results_csv`
already writes rows in descending score order and `ReviewSession` re-sorts
by score on load, so a strategy that produced its own ordering would just be
a second, divergent implementation of something that already works.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

# A parameter that is one of several weights to be normalised against each
# other before use (see `WeightedParams.normalized_weights`), rather than a
# standalone threshold. The parameter dialog groups by this too.
GROUP_WEIGHTS = "weights"
GROUP_THRESHOLDS = "thresholds"
# A yes/no pipeline switch (see `use_subject_filter_spec` below) rather than a
# tunable number - kept as its own group so the dialog can give it its own
# clearly-labelled section instead of burying it among numeric thresholds.
GROUP_OPTIONS = "options"


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter of a strategy, described well enough that a
    dialog can be built from it without knowing what it means."""

    name: str            # the attribute name on the params dataclass
    label: str           # what the dialog shows next to the field
    default: float
    minimum: float
    maximum: float
    group: str = GROUP_WEIGHTS
    decimals: int = 0
    suffix: str = ""
    help: str = ""
    # True for a yes/no switch (rendered as a checkbox, not a spin box - see
    # desktop.dialogs.workflow_dialogs.AlgorithmParametersDialog). The
    # underlying value is still stored as 0.0/1.0 on `default`/`minimum`/
    # `maximum` so every other consumer of ParamSpec (reset-to-defaults,
    # range validation) keeps working unchanged for a boolean the same way it
    # already does for a number - only the widget and the value type a dialog
    # reads back differ.
    is_boolean: bool = False


@dataclass(frozen=True)
class StrategyInfo:
    """What the UI needs to list a strategy in a menu.

    Pure data, with no reference to the strategy object, so building the
    menu never imports a model.
    """

    strategy_id: str
    display_name: str
    description: str
    # Short label for the score this strategy produces, as the gallery and
    # Loupe show it next to the number ("AI", "Classic"). Kept short because
    # several appear stacked on one thumbnail card.
    score_label: str = ""


class RankingStrategy(Protocol):
    """Anything that can rank a folder.

    `rank_folder` must:

    - write the folder's ranking CSV via `sidecar.ranking_path` (so
      `ReviewSession` finds it without being told where it is),
    - raise `FileNotFoundError`/`ValueError` rather than exiting, so a UI can
      report the problem,
    - report progress through the two optional callbacks, which have the
      same signatures the existing `rank.rank_folder` already uses.

    It returns a plain dict rather than a dataclass because the desktop
    layer passes the result straight through a background thread alongside
    the reloaded session state, exactly as it already does for the AI path.
    See `ranking.result` for the keys every strategy provides.
    """

    @property
    def info(self) -> StrategyInfo: ...

    @property
    def param_specs(self) -> tuple[ParamSpec, ...]: ...

    # The dataclass `rank_folder` accepts as `params`. Exposed so a UI can
    # generate a parameter dialog from `params_class.specs()` without knowing
    # which strategy it is talking to.
    @property
    def params_class(self) -> type: ...

    def rank_folder(
        self,
        input_folder,
        *,
        params: Any = None,
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        force_preprocess: bool = False,
    ) -> dict: ...


def use_subject_filter_spec() -> ParamSpec:
    """The one shared spec every `WeightedParams` subclass may include in its
    own `specs()` to surface `WeightedParams.use_subject_filter` in the
    generated parameter dialog - see that field's own docstring.

    Off by default: the Eye-Detector Ensemble Evidence Study found the
    upstream subject/object filter (COCO class + confidence gating a
    "selected" detection) rejects images a downstream ranking strategy could
    still usefully score from the same cached crop - see
    `ranking.classic.ClassicVisionStrategy._load_candidate`'s own docstring
    for exactly what changes when this is off. Turning it on restores
    today's stricter behavior unchanged, for a photographer who wants it.
    """
    return ParamSpec(
        name="use_subject_filter",
        label="Use Subject Filter",
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        group=GROUP_OPTIONS,
        decimals=0,
        is_boolean=True,
        help=(
            "When on, only images where the shared crop pipeline confidently identified a subject are "
            "ranked - the original, stricter behavior. When off (default), any image with a cached crop "
            "reaches this strategy, and a missing eye or subject is handled by the strategy's own "
            "existing logic instead of being rejected upstream."
        ),
    )


@dataclass(frozen=True)
class WeightedParams:
    """Mixin for a params dataclass whose GROUP_WEIGHTS fields combine into a
    weighted sum.

    The photographer may type any numbers at all - 50/30/20, 5/3/2, or
    7/7/7 - and they mean the same thing, because they are normalised to sum
    to 1 here rather than being validated into a particular range. All-zero
    weights fall back to equal weighting: it is the only reading of "none of
    these matter" that still produces an ordering instead of scoring every
    image identically.

    `use_subject_filter` lives here, not on any one strategy's own dataclass,
    so every strategy built on this mixin gets the identical switch with the
    identical default - see `use_subject_filter_spec` below for what it
    controls and why it defaults to off. A strategy declares it in `specs()`
    the same way every other shared field (see `_detection_specs` in
    `ranking.classic`) already gets declared - by including
    `use_subject_filter_spec()` in the tuple it returns - rather than this
    class forcing it into every subclass's spec list, since not every
    subclass necessarily wants it surfaced in the dialog the same way.
    """

    # kw_only so this shared field never shifts a subclass's own positional
    # argument order - ClassicVisionParams(100, 0, 0) and every other
    # existing positional construction (see this project's own tests) must
    # keep meaning exactly what it meant before this field existed.
    use_subject_filter: bool = field(default=False, kw_only=True)

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        raise NotImplementedError

    @classmethod
    def defaults(cls):
        return cls()

    @classmethod
    def from_values(cls, values: dict[str, float | bool]):
        """Build from a dialog's raw {name: value} mapping, ignoring anything
        that is not a declared parameter. Values are almost always numeric
        (a spin box), but a boolean spec (see `ParamSpec.is_boolean`) passes
        a real `bool` through unchanged rather than a stringified 0.0/1.0."""
        known = {spec.name for spec in cls.specs()}
        return cls(**{name: value for name, value in values.items() if name in known})

    def replace(self, **changes):
        return replace(self, **changes)

    def normalized_weights(self) -> dict[str, float]:
        """The GROUP_WEIGHTS parameters, scaled to sum to 1.0."""
        weight_specs = [spec for spec in self.specs() if spec.group == GROUP_WEIGHTS]
        raw = {spec.name: max(0.0, float(getattr(self, spec.name))) for spec in weight_specs}
        total = sum(raw.values())
        if total <= 0.0:
            equal = 1.0 / len(raw) if raw else 0.0
            return {name: equal for name in raw}
        return {name: value / total for name, value in raw.items()}
