"""Part of the ranking workflow: file the images by the final verdict.

`picklikeme.rank` produces an ordering and `picklikeme review` lets the
photographer correct it; this acts on the result, moving images into
`_Selected/` and `_Rejected/`. That turns a ranking into something a
photographer can actually work with in Lightroom or Explorer.

Two entry points over one move loop:

- `organize_by_decision(selected, rejected, ...)` - explicit sets, which is what
  a reviewed shoot produces once manual Keep/Reject overrides mean the selected
  images are no longer simply the top of the ranking.
- `organize_ranked_images(ranked, ..., percentage)` - the unreviewed case, which
  splits the ranking at a percentage and defers to the above.

Lives in the ranking module, not the analyzer: the analyzer is a read-only
reporting tool and must never move a file. Nothing here imports from
`picklikeme.analyzer`, and nothing there imports this.

Safety, in order of importance - these images are the only copy:

- **Never overwrite.** A colliding destination gets a numbered suffix; an
  existing file is never replaced, and never silently.
- **Recognise work already done.** A file that is already at its destination is
  reported as skipped rather than moved onto itself.
- **Keep going.** One unmovable file (locked, permission denied) is recorded and
  the rest still get filed.
- **Move, not copy.** These are 20-60 MB RAWs; duplicating a shoot is not what
  was asked for, and a same-volume move is a metadata operation.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

SELECTED_DIRNAME = "_Selected"
REJECTED_DIRNAME = "_Rejected"

# Folders this module creates. Enumeration must skip them, or a second ranking
# run would re-rank its own output and shuffle files between the two.
ORGANIZE_DIRNAMES = frozenset({SELECTED_DIRNAME, REJECTED_DIRNAME})

DEFAULT_SELECTION_PERCENTAGE = 25.0


class InvalidSelectionPercentage(ValueError):
    """Raised for a percentage outside [0, 100]."""


def validate_selection_percentage(value: float) -> float:
    """Check the percentage, with a message that says what to do about it."""
    try:
        percentage = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidSelectionPercentage(
            f"--selection-percentage must be a number between 0 and 100, got {value!r}"
        ) from exc
    if math.isnan(percentage) or not 0.0 <= percentage <= 100.0:
        raise InvalidSelectionPercentage(
            f"--selection-percentage must be between 0 and 100, got {percentage:g}. "
            "0 selects nothing, 100 selects everything."
        )
    return percentage


def selection_count(total: int, percentage: float) -> int:
    """How many of `total` ranked images go to `_Selected`.

    Rounded to nearest so 25% of 10 is 2 rather than 3, with the endpoints exact:
    0% selects nothing and 100% selects everything, whatever the rounding would
    otherwise do.
    """
    if total <= 0:
        return 0
    if percentage <= 0.0:
        return 0
    if percentage >= 100.0:
        return total
    return min(total, max(0, round(total * percentage / 100.0)))


def unique_destination(destination: Path) -> Path:
    """A free path at `destination`, suffixing `_1`, `_2`, ... on collision.

    Filenames are preserved wherever possible; only a genuine clash is renamed,
    and the original file is left untouched.
    """
    if not destination.exists():
        return destination
    for index in range(1, 10_000):
        candidate = destination.with_name(f"{destination.stem}_{index}{destination.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"could not find a free filename for {destination}")


@dataclass
class OrganizeResult:
    """What the organize pass did, for the summary and for tests."""

    ranked: int = 0
    selected: int = 0
    rejected: int = 0
    moved: int = 0
    skipped: int = 0
    errors: int = 0
    renamed: int = 0
    selected_dir: Path | None = None
    rejected_dir: Path | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)
    moves: dict[str, Path] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "",
            f"Images ranked:      {self.ranked:,}",
            f"Selected:           {self.selected:,}",
            f"Rejected:           {self.rejected:,}",
            f"Moved successfully: {self.moved:,}",
            f"Skipped:            {self.skipped:,}",
            f"Errors:             {self.errors:,}",
        ]
        if self.renamed:
            lines.append(f"Renamed on collision: {self.renamed:,} (no file was overwritten)")
        if self.selected_dir is not None:
            lines += [
                "",
                f"  {SELECTED_DIRNAME}: {self.selected_dir}",
                f"  {REJECTED_DIRNAME}: {self.rejected_dir}",
            ]
        for path, reason in self.failures[:10]:
            lines.append(f"  ERROR {Path(path).name}: {reason}")
        if len(self.failures) > 10:
            lines.append(f"  ... and {len(self.failures) - 10:,} more errors")
        return "\n".join(lines)


def organize_by_decision(
    selected_paths: Sequence[str],
    rejected_paths: Sequence[str],
    destination_root: str | Path,
    *,
    dry_run: bool = False,
    announce: bool = True,
) -> OrganizeResult:
    """Move images into `_Selected` / `_Rejected` by an explicit verdict.

    The two sets are given outright rather than derived from a cut, because a
    reviewed shoot's selection is not a prefix of the ranking: a manual Keep on
    a low-scoring frame, or a manual Reject on a high-scoring one, breaks that
    assumption entirely.

    Anything the caller left out of both lists is simply not touched. That is
    how an image with no ranking and no manual decision stays where it is
    instead of being swept somewhere it was never judged to belong.

    `dry_run` still fills in every count and `moves`, so a confirmation dialog
    can show exactly what would happen before a single file is touched.
    """
    destination_root = Path(destination_root)
    result = OrganizeResult(
        ranked=len(selected_paths) + len(rejected_paths),
        selected=len(selected_paths),
        rejected=len(rejected_paths),
        selected_dir=destination_root / SELECTED_DIRNAME,
        rejected_dir=destination_root / REJECTED_DIRNAME,
    )
    if result.ranked == 0:
        return result

    if announce:
        print(f"Organizing images into {SELECTED_DIRNAME} and {REJECTED_DIRNAME}...")
        print(f"  {result.selected:,} -> {SELECTED_DIRNAME}, {result.rejected:,} -> {REJECTED_DIRNAME}")

    if not dry_run:
        for directory in (result.selected_dir, result.rejected_dir):
            directory.mkdir(parents=True, exist_ok=True)

    for raw_path, target_dir in [
        *((path, result.selected_dir) for path in selected_paths),
        *((path, result.rejected_dir) for path in rejected_paths),
    ]:
        source = Path(raw_path)
        try:
            if not source.is_file():
                result.skipped += 1
                result.failures.append((raw_path, "source file not found"))
                continue

            destination = target_dir / source.name
            if source.resolve() == destination.resolve():
                # Already filed - a re-run must be a no-op, not a shuffle.
                result.skipped += 1
                continue

            final = unique_destination(destination)
            if final != destination:
                result.renamed += 1
                logger.info("Collision: %s -> %s", source.name, final.name)

            if not dry_run:
                shutil.move(str(source), str(final))
            result.moved += 1
            result.moves[raw_path] = final
        except Exception as exc:  # noqa: BLE001 - record and carry on
            result.errors += 1
            result.failures.append((raw_path, f"{type(exc).__name__}: {exc}"))
            logger.warning("Could not organize %s: %s", source, exc)

    if announce:
        print(result.render())
    return result


def organize_ranked_images(
    ranked_paths: Sequence[str],
    destination_root: str | Path,
    selection_percentage: float = DEFAULT_SELECTION_PERCENTAGE,
    *,
    dry_run: bool = False,
) -> OrganizeResult:
    """Move ranked images into `_Selected` / `_Rejected` by a percentage cut.

    `ranked_paths` must be in ranking order, best first - the caller already has
    that from the ranking, and recomputing it here would risk disagreeing with
    the CSV that was just written.

    The unreviewed path: the model's ordering is taken at face value. Once a
    photographer has reviewed a shoot, `organize_by_decision` is what runs.
    """
    percentage = validate_selection_percentage(selection_percentage)
    cut = selection_count(len(ranked_paths), percentage)
    if not ranked_paths:
        return organize_by_decision([], [], destination_root, dry_run=dry_run, announce=False)

    # Announced here rather than by organize_by_decision, so the header can name
    # the percentage that produced the split.
    print(f"Organizing images into {SELECTED_DIRNAME} and {REJECTED_DIRNAME}...")
    print(f"  top {percentage:g}% of {len(ranked_paths):,} ranked images -> {SELECTED_DIRNAME}")

    result = organize_by_decision(
        ranked_paths[:cut], ranked_paths[cut:], destination_root, dry_run=dry_run, announce=False
    )
    print(result.render())
    return result
