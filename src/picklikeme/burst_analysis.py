"""Burst Analysis - a processing layer that runs after ranking, not another
ranking strategy.

    Image -> Ranking Strategy -> Score -> Burst Analysis -> Burst Ranking -> Review UI

A ranking strategy (`picklikeme.ranking`) answers "how good is this
photograph". Burst Analysis answers a completely different question: "which
photographs were taken in the same continuous burst, and which one of them is
the best of that burst" - and it answers that second question using only the
first one's *output*, never its internals. It receives a flat list of
`ScoredImage(path, captured_at, score)` triples and knows nothing else about
where the score came from - not whether it was the AI model, Classic Vision,
or a future hybrid, and not any of their parameters, weights, or filters. That
is deliberate: it is what lets a brand new ranking strategy participate in
burst ranking automatically, with zero change to this module.

Burst *identification* - which frames belong together - reuses `burst.py`'s
existing capture-time-gap clustering wholesale (`reconstruct_bursts`) rather
than re-implementing it: the same logic already used to reassemble bursts
after Keep/Reject sorting has physically separated their frames applies just
as well here, where the frames haven't moved but a review session still wants
them grouped. An image with no readable capture time - or an unparseable one -
becomes a singleton burst of its own rather than being guessed into one with
its neighbours, the same policy `ingest.burst` documents for the ingestion
pipeline's own (heavier, multi-camera) burst reconstruction.

Burst *ranking* - `burst_rank`/`burst_best` - is this module's own addition:
within one burst, the members are ordered by `score` descending (highest
first); an image the chosen strategy never scored sorts after every scored
member, in whatever order it was given in. `burst_rank` is 1 at the top of
that order, and `burst_best` is true for exactly the member at rank 1 - the
one a "Collapse Bursts" gallery view shows in place of the whole group.
"""

from __future__ import annotations

from dataclasses import dataclass

from .burst import BurstEntry, reconstruct_bursts

# The same default reconstruct_bursts itself uses - a review session groups
# a single folder's own frames (already one shoot, typically one camera),
# so there is no need for ingest.burst's heavier per-(shoot, camera) grouping
# or its own, separately-tuned gap.
DEFAULT_MAX_GAP_SECONDS = 2.0


@dataclass(frozen=True)
class ScoredImage:
    """What Burst Analysis is given for one image - nothing about how the
    score was produced, on purpose (see the module docstring)."""

    path: str
    captured_at: str | None
    score: float | None


@dataclass(frozen=True)
class BurstInfo:
    """What Burst Analysis hands back for one image."""

    burst_id: str
    burst_size: int
    # 1-based position within this image's own burst, ordered by score
    # descending - never a global rank across every burst in the folder.
    burst_rank: int
    burst_best: bool


def analyze_bursts(
    images: list[ScoredImage], *, max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS
) -> dict[str, BurstInfo]:
    """Group `images` into bursts by capture-time gap, then rank each burst's
    own members by `score` - image path -> BurstInfo, one entry per input
    image (duplicate paths keep only the last, same as a dict comprehension
    would).

    Every image gets an entry, including one with no score at all (it can
    still belong to a burst and be ranked, last, among its members) and one
    with no usable capture time (its own singleton burst - see the module
    docstring). Grouping never raises on a malformed timestamp; unparseable
    values are treated exactly like a missing one.
    """
    scores = {image.path: image.score for image in images}

    timed_entries: list[BurstEntry] = []
    untimed_paths: list[str] = []
    for image in images:
        if image.captured_at and _is_parseable(image.captured_at):
            timed_entries.append(BurstEntry(path=image.path, timestamp=image.captured_at))
        else:
            untimed_paths.append(image.path)

    groups: list[list[str]] = [
        [entry.path for entry in group] for group in reconstruct_bursts(timed_entries, max_gap_seconds)
    ]
    groups.extend([path] for path in untimed_paths)

    result: dict[str, BurstInfo] = {}
    for index, members in enumerate(groups, start=1):
        burst_id = f"burst-{index:04d}"
        ordered = sorted(
            range(len(members)),
            key=lambda i: (scores[members[i]] is None, -(scores[members[i]] or 0.0)),
        )
        for rank, position in enumerate(ordered, start=1):
            path = members[position]
            result[path] = BurstInfo(
                burst_id=burst_id, burst_size=len(members), burst_rank=rank, burst_best=(rank == 1)
            )
    return result


def _is_parseable(value: str) -> bool:
    from datetime import datetime

    candidate = value[:-1] if value.endswith("Z") else value
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True
