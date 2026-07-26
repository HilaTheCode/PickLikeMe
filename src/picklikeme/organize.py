"""Part of the ranking workflow: file the ranked images by the model's verdict.

`picklikeme.rank` produces an ordering; this acts on it, moving the top
`--selection-percentage` of images into `selected_by_ai/` and the rest into
`rejected_by_ai/`. That turns a ranking into something a photographer can
actually work with in Lightroom or Explorer.

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

SELECTED_DIRNAME = "selected_by_ai"
REJECTED_DIRNAME = "rejected_by_ai"

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
    """How many of `total` ranked images go to selected_by_ai.

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


def organize_ranked_images(
    ranked_paths: Sequence[str],
    destination_root: str | Path,
    selection_percentage: float = DEFAULT_SELECTION_PERCENTAGE,
    *,
    dry_run: bool = False,
) -> OrganizeResult:
    """Move ranked images into selected_by_ai / rejected_by_ai.

    `ranked_paths` must be in ranking order, best first - the caller already has
    that from the ranking, and recomputing it here would risk disagreeing with
    the CSV that was just written.
    """
    percentage = validate_selection_percentage(selection_percentage)
    destination_root = Path(destination_root)
    total = len(ranked_paths)
    cut = selection_count(total, percentage)

    result = OrganizeResult(
        ranked=total,
        selected=cut,
        rejected=total - cut,
        selected_dir=destination_root / SELECTED_DIRNAME,
        rejected_dir=destination_root / REJECTED_DIRNAME,
    )
    if total == 0:
        return result

    print(f"Organizing images into {SELECTED_DIRNAME} and {REJECTED_DIRNAME}...")
    print(f"  top {percentage:g}% of {total:,} ranked images -> {SELECTED_DIRNAME}")

    if not dry_run:
        for directory in (result.selected_dir, result.rejected_dir):
            directory.mkdir(parents=True, exist_ok=True)

    for position, raw_path in enumerate(ranked_paths):
        source = Path(raw_path)
        target_dir = result.selected_dir if position < cut else result.rejected_dir
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

    print(result.render())
    return result
