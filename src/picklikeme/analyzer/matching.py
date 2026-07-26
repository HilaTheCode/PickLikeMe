"""Capability 1 - joining ranking output to the photographer's decisions.

Matching is the foundation every metric rests on, and it is the step most
likely to go quietly wrong: paths in a ranking file were written on some
earlier run and the folders may since have been renamed, moved to another
drive, or re-cased by Windows. So the join is tried in three widening stages
and always reports which stage produced each hit.

An image that cannot be matched becomes Outcome.UNKNOWN and a warning. It is
never fatal, and never silently counted as a negative - that would inflate
specificity with images the photographer never judged.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .io import enumerate_ground_truth
from .model import MatchedImage, Outcome, RankedImage

logger = logging.getLogger(__name__)

SELECTED = 1
REJECTED = 0


# Match strategies, in the order they are attempted. Plain strings rather than
# an Enum because they are only ever reported, never branched on.
MATCH_RESOLVED = "resolved_path"
MATCH_NORMALISED = "case_insensitive_path"
MATCH_SUFFIX = "unique_path_suffix"
MATCH_FILENAME = "unique_filename"


@dataclass
class MatchResult:
    """The matched dataset plus everything needed to audit the join."""

    images: list[MatchedImage]
    threshold: float
    unmatched: list[RankedImage] = field(default_factory=list)
    unranked_selected: list[Path] = field(default_factory=list)
    unranked_rejected: list[Path] = field(default_factory=list)
    strategy_counts: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    @property
    def evaluable(self) -> list[MatchedImage]:
        """Images with known ground truth - the only ones metrics may use."""
        return [image for image in self.images if image.outcome is not Outcome.UNKNOWN]

    def by_outcome(self, outcome: Outcome) -> list[MatchedImage]:
        return [image for image in self.images if image.outcome is outcome]

    @property
    def counts(self) -> dict[str, int]:
        counter = Counter(image.outcome for image in self.images)
        return {outcome.value: counter.get(outcome, 0) for outcome in Outcome}

    @property
    def num_selected(self) -> int:
        return sum(1 for image in self.evaluable if image.truth == SELECTED)

    @property
    def num_rejected(self) -> int:
        return sum(1 for image in self.evaluable if image.truth == REJECTED)

    def summary(self) -> str:
        counts = self.counts
        total = len(self.images)
        matched = len(self.evaluable)
        lines = [
            "Dataset matching",
            "================",
            f"  ranked images:        {total:,}",
            f"  matched to truth:     {matched:,} ({matched / total * 100:.1f}%)" if total else "  matched to truth: 0",
            f"    selected (positive):{self.num_selected:>8,}",
            f"    rejected (negative):{self.num_rejected:>8,}",
            f"  unmatched (unknown):  {counts['unknown']:,}",
            "",
            f"  true positives:       {counts['true_positive']:,}",
            f"  true negatives:       {counts['true_negative']:,}",
            f"  false positives:      {counts['false_positive']:,}",
            f"  false negatives:      {counts['false_negative']:,}",
        ]
        if self.strategy_counts:
            lines.append("")
            lines.append("  matched by:")
            for strategy, count in self.strategy_counts.most_common():
                lines.append(f"    {strategy:<24}{count:,}")
        if self.unranked_selected or self.unranked_rejected:
            lines.append("")
            lines.append(
                f"  in ground truth but absent from the ranking: "
                f"{len(self.unranked_selected):,} selected, {len(self.unranked_rejected):,} rejected"
            )
        return "\n".join(lines)


class _TruthIndex:
    """Looks up a ground-truth label for a ranked path.

    Holds four progressively looser indexes. Looser lookups are only consulted
    when stricter ones miss, and any lookup whose key is ambiguous (the same
    filename in both the selected and rejected folders) refuses to answer -
    a wrong label is worse than an honest UNKNOWN.
    """

    def __init__(self, selected: list[Path], rejected: list[Path]):
        self._resolved: dict[str, int] = {}
        self._normalised: dict[str, int] = {}
        self._by_suffix: dict[str, list[tuple[tuple[str, ...], int]]] = {}
        self._by_name: dict[str, set[int]] = {}
        self._ambiguous_names: set[str] = set()

        for paths, label in ((selected, SELECTED), (rejected, REJECTED)):
            for path in paths:
                resolved = path.resolve()
                self._resolved[str(resolved)] = label
                self._normalised[str(resolved).lower()] = label
                parts = tuple(part.lower() for part in resolved.parts)
                self._by_suffix.setdefault(parts[-1], []).append((parts, label))
                self._by_name.setdefault(parts[-1], set()).add(label)

        self._ambiguous_names = {name for name, labels in self._by_name.items() if len(labels) > 1}

    def lookup(self, image_path: str) -> tuple[int | None, str | None]:
        """Return (label, strategy) or (None, None)."""
        candidate = Path(image_path)
        try:
            resolved = str(candidate.resolve())
        except OSError:  # pragma: no cover - malformed path on some filesystems
            resolved = str(candidate)

        label = self._resolved.get(resolved)
        if label is not None:
            return label, MATCH_RESOLVED

        label = self._normalised.get(resolved.lower())
        if label is not None:
            return label, MATCH_NORMALISED

        # Suffix match: the ranking may hold D:\old\shoot\a.nef while the folder
        # now lives at E:\archive\shoot\a.nef. Accept only when every candidate
        # sharing the filename agrees on the label.
        parts = tuple(part.lower() for part in candidate.parts)
        if not parts:
            return None, None
        name = parts[-1]
        candidates = self._by_suffix.get(name, [])
        suffix_labels = {
            label for stored_parts, label in candidates if parts[-len(stored_parts) :] == stored_parts
        }
        if len(suffix_labels) == 1:
            return next(iter(suffix_labels)), MATCH_SUFFIX

        # Last resort: a unique filename anywhere in the ground truth.
        if name not in self._ambiguous_names:
            labels = self._by_name.get(name)
            if labels and len(labels) == 1:
                return next(iter(labels)), MATCH_FILENAME
        return None, None


def classify(truth: int | None, predicted: int) -> Outcome:
    """Confusion-matrix cell for one image."""
    if truth is None:
        return Outcome.UNKNOWN
    if predicted == 1:
        return Outcome.TRUE_POSITIVE if truth == SELECTED else Outcome.FALSE_POSITIVE
    return Outcome.FALSE_NEGATIVE if truth == SELECTED else Outcome.TRUE_NEGATIVE


def predict(image: RankedImage, threshold: float) -> int:
    """Turn a ranking row into a keep/reject decision.

    An explicit predicted_class from the producer wins; otherwise the
    probability is thresholded, falling back to the raw score when the file
    carried no probability.
    """
    if image.predicted_class is not None:
        return 1 if image.predicted_class == 1 else 0
    value = image.probability if image.probability is not None else image.score
    return 1 if value >= threshold else 0


def match_dataset(
    images: list[RankedImage],
    selected_root: str | Path | None,
    rejected_root: str | Path | None,
    threshold: float = 0.5,
    *,
    max_warnings: int = 20,
) -> MatchResult:
    """Join ranked images to ground truth and assign each an outcome.

    When no ground-truth folders are given, labels already present in the
    ranking file are used instead - which is what makes the analyzer usable on
    a training run's own results CSV, where the labels travel with the rows.
    """
    warnings: list[str] = []
    strategy_counts: Counter = Counter()
    matched: list[MatchedImage] = []
    unmatched: list[RankedImage] = []

    index: _TruthIndex | None = None
    selected_paths: list[Path] = []
    rejected_paths: list[Path] = []

    if selected_root is not None or rejected_root is not None:
        selected_paths = enumerate_ground_truth(selected_root) if selected_root else []
        rejected_paths = enumerate_ground_truth(rejected_root) if rejected_root else []
        logger.info(
            "Ground truth: %d selected, %d rejected", len(selected_paths), len(rejected_paths)
        )
        index = _TruthIndex(selected_paths, rejected_paths)
        if index._ambiguous_names:
            warnings.append(
                f"{len(index._ambiguous_names)} filename(s) appear in BOTH ground-truth folders; "
                "those are matched only by full path, never by name."
            )

    consumed: set[str] = set()
    for image in images:
        if index is not None:
            truth, strategy = index.lookup(image.image_path)
            if truth is not None:
                strategy_counts[strategy] += 1
                consumed.add(Path(image.image_path).name.lower())
        else:
            truth, strategy = image.label, "ranking_file_label"
            if truth is not None:
                strategy_counts[strategy] += 1

        outcome = classify(truth, predict(image, threshold))
        if outcome is Outcome.UNKNOWN:
            unmatched.append(image)
            if len(warnings) < max_warnings:
                warnings.append(f"No ground truth for {image.image_path}")
        matched.append(
            MatchedImage(ranked=image, truth=truth, predicted=predict(image, threshold), outcome=outcome)
        )

    if unmatched and len(unmatched) > max_warnings:
        warnings.append(f"... and {len(unmatched) - max_warnings} more unmatched images")

    # The reverse direction matters too: ground-truth images the ranking never
    # scored are invisible to every metric, and silently shrink recall's
    # denominator if nobody says so.
    unranked_selected = [p for p in selected_paths if p.name.lower() not in consumed]
    unranked_rejected = [p for p in rejected_paths if p.name.lower() not in consumed]
    if unranked_selected or unranked_rejected:
        warnings.append(
            f"{len(unranked_selected)} selected and {len(unranked_rejected)} rejected images "
            "are absent from the ranking and excluded from all metrics."
        )

    result = MatchResult(
        images=matched,
        threshold=threshold,
        unmatched=unmatched,
        unranked_selected=unranked_selected,
        unranked_rejected=unranked_rejected,
        strategy_counts=strategy_counts,
        warnings=warnings,
    )
    logger.info(
        "Matched %d/%d ranked images (%d unknown)", len(result.evaluable), len(images), len(unmatched)
    )
    return result
