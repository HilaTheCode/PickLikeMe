"""Burst-level head-relative consistency evidence for eye-detection fusion.

PeakPic groups images into Bursts (`picklikeme.burst`/`burst_analysis`), and
within a Burst the camera frequently moves - the bird's head can shift from
one side of the frame to the other, or the photographer reframes entirely,
between two consecutive shutter presses. **Absolute image coordinates are
therefore never compared here.** What should stay stable across a Burst of
the same bird is the eye's position RELATIVE TO ITS OWN HEAD in each frame -
see `geometry.HeadFrame`, which is what makes that comparison meaningful
even when the head itself rotated between frames.

This module is a post-processing step, not part of `FusionEyeDetector.detect`
itself: consistency is a property of a *group* of frames, so it cannot be
decided from a single crop the way `fusion.py`'s per-image logic can (see
`burst_analysis.py`'s own module docstring for the same reasoning applied to
scoring - "Burst Analysis... runs after ranking, not another ranking
strategy"; this is the equivalent split for eye-detection evidence).
`evaluate_burst_consistency` is meant to be called once ranking has produced
a fused `EyeDetection` for every member of a folder, using whatever burst
grouping `burst_analysis.analyze_bursts`/`burst.reconstruct_bursts` already
produced from capture timestamps.

Burst consistency is ADDITIONAL evidence, never the sole verdict: a burst of
one frame (or one where every other member lacks a usable position to
compare against) cannot be judged an outlier by definition, and this module
never invents a deviation for it - see `evaluate_burst_consistency`'s own
docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import HeadFrame

# How far a frame's own head-relative eye position may sit from its burst's
# median position (in head-scale units - see geometry.HeadFrame) before it
# is flagged as an outlier.
#
# Not empirically tuned against real Burst sequences: this project's only
# available real Burst-grouped, timestamped photographs on this machine are
# a non-bird species (see the accompanying report's benchmark-dataset
# section), so there is no real bird Burst to fit this against yet. Chosen
# instead to sit comfortably above the ~0.3-0.4 normal single-frame
# agreement noise `fusion.FusionConfig.agreement_threshold` was fitted to
# (see that module) - a within-Burst deviation should be smaller than
# cross-model disagreement on the same frame, since it is the same bird's
# same head, not two different models' two different opinions - while
# staying well clear of 0, which would flag ordinary sub-pixel jitter. A
# starting point to refine once real Burst sequences are available, in the
# same spirit as EyePose-v0's own "reasonable starting point, not
# empirically validated" thresholds.
DEFAULT_OUTLIER_THRESHOLD = 0.6

# A burst smaller than this has too few independent members for a median
# position to mean anything; consistency is never evaluated below it.
MIN_BURST_SIZE_FOR_EVIDENCE = 2


@dataclass(frozen=True)
class BurstEyeObservation:
    """One frame's already-fused eye position, expressed head-relative and
    ready for burst comparison. Built by a caller from `HeadFrame.project`
    (see `geometry.py`) - this module never looks at raw pixels or an
    `EyeDetection` directly, so it stays usable regardless of which fusion
    (or single-model) detector produced the position."""

    image_path: str
    burst_id: str
    u: float
    v: float


@dataclass(frozen=True)
class BurstConsistencyResult:
    """What Burst evidence has to say about one observation - additional
    evidence, never a verdict on its own (see the module docstring)."""

    image_path: str
    burst_id: str
    # How many OTHER members of this burst had a usable position to compare
    # against. 0 means this observation could not be evaluated at all -
    # `deviation`/`is_outlier` below carry no information in that case.
    comparison_count: int
    # Distance from this frame's own (u, v) to the MEDIAN (u, v) of every
    # other member of its burst, in head-scale units - `None` when
    # `comparison_count` is 0.
    deviation: float | None
    is_outlier: bool


def evaluate_burst_consistency(
    observations: list[BurstEyeObservation],
    *,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> dict[str, BurstConsistencyResult]:
    """Group `observations` by `burst_id`, then for each one measure how far
    its head-relative position sits from the MEDIAN position of every OTHER
    member of its own burst - image path -> `BurstConsistencyResult`, one
    entry per observation given.

    The median excludes the observation being scored (never compared
    against itself) and is a plain per-axis median of `u` and `v`
    independently, chosen over a mean specifically so that one wildly wrong
    frame in a burst cannot drag the reference point toward itself and mask
    its own outlier status - the exact failure mode a mean-based reference
    would have.

    A burst with fewer than `MIN_BURST_SIZE_FOR_EVIDENCE` members - or an
    observation whose burst-mates all happen to lack a usable position -
    can never be flagged: `comparison_count` is 0, `deviation` is `None`,
    `is_outlier` is `False`. See the module docstring's "additional
    evidence, never the sole verdict" - a singleton burst is not suspicious,
    it is simply unevaluable.
    """
    by_burst: dict[str, list[BurstEyeObservation]] = {}
    for obs in observations:
        by_burst.setdefault(obs.burst_id, []).append(obs)

    results: dict[str, BurstConsistencyResult] = {}
    for burst_id, members in by_burst.items():
        if len(members) < MIN_BURST_SIZE_FOR_EVIDENCE:
            for obs in members:
                results[obs.image_path] = BurstConsistencyResult(
                    image_path=obs.image_path, burst_id=burst_id, comparison_count=0, deviation=None, is_outlier=False
                )
            continue
        for obs in members:
            others = [m for m in members if m is not obs]
            median_u = _median([m.u for m in others])
            median_v = _median([m.v for m in others])
            deviation = ((obs.u - median_u) ** 2 + (obs.v - median_v) ** 2) ** 0.5
            results[obs.image_path] = BurstConsistencyResult(
                image_path=obs.image_path,
                burst_id=burst_id,
                comparison_count=len(others),
                deviation=deviation,
                is_outlier=deviation > outlier_threshold,
            )
    return results


def head_relative_observation(
    image_path: str, burst_id: str, eye_point: tuple[float, float], head_frame: HeadFrame | None
) -> BurstEyeObservation | None:
    """Build one `BurstEyeObservation` from a frame's fused eye point and
    its OWN frame's `HeadFrame` (never another frame's - see the module
    docstring on why absolute/cross-frame coordinates are never compared
    directly). `None` when no head frame could be built for this frame
    (e.g. the reference landmarks were not both available), matching every
    other "cannot evaluate this one" case in this module rather than
    fabricating a position."""
    if head_frame is None:
        return None
    u, v = head_frame.project(eye_point[0], eye_point[1])
    return BurstEyeObservation(image_path=image_path, burst_id=burst_id, u=u, v=v)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
