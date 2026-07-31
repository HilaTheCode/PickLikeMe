"""Arrange by Species: classify every image already in a folder (typically
the Keep folder Review/Arrange produced) and file it into a per-species
subfolder.

    Keep Folder -> Open Folder -> Arrange by Species -> Species
    Classification Engine (classifier.py) -> species folders -> files moved

Runs entirely after, and independently of, Review/Arrange - it operates on
whatever is already in a folder (via analyzer.io.enumerate_ground_truth, the
same general "every image here" helper the analysis report and Review both
already use), moves files with the exact same never-overwrite,
rename-on-collision, keep-going-on-error safety organize.py's own
organize_by_decision already established (unique_destination is reused
directly, not reimplemented), and never imports anything from bird_crop,
rank, train, or review.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..organize import unique_destination
from .cache import SpeciesCache
from .classifier import UNKNOWN_SPECIES, SpeciesClassifier

logger = logging.getLogger(__name__)

# Characters unsafe (or reserved) in a Windows or POSIX folder name. A
# classifier's own species list is expected to already be clean common
# names, but a photographer-supplied --species-list file is not guaranteed
# to be, so this is defensive rather than assumed unnecessary.
_UNSAFE_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_species_folder_name(species: str) -> str:
    """A species name, made safe as a single folder path component.

    Falls back to UNKNOWN_SPECIES if sanitizing leaves nothing usable (an
    empty or all-punctuation species string), so a bad species-list entry
    degrades to "Unknown" rather than raising deep inside a long-running
    arrange pass.
    """
    cleaned = _UNSAFE_FOLDER_CHARS.sub("_", species).strip().strip(".")
    return cleaned or UNKNOWN_SPECIES


@dataclass
class SpeciesArrangeResult:
    """What the pass did, for the CLI summary and for tests."""

    total: int = 0
    classified: int = 0
    moved: int = 0
    skipped: int = 0
    errors: int = 0
    renamed: int = 0
    species_counts: dict[str, int] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)
    moves: dict[str, Path] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "",
            f"Images found:       {self.total:,}",
            f"Classified:         {self.classified:,}",
            f"Moved successfully: {self.moved:,}",
            f"Skipped:            {self.skipped:,}",
            f"Errors:             {self.errors:,}",
        ]
        if self.renamed:
            lines.append(f"Renamed on collision: {self.renamed:,} (no file was overwritten)")
        if self.species_counts:
            lines.append("")
            lines.append("By species:")
            for name, count in sorted(self.species_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"  {name}: {count:,}")
        for path, reason in self.failures[:10]:
            lines.append(f"  ERROR {Path(path).name}: {reason}")
        if len(self.failures) > 10:
            lines.append(f"  ... and {len(self.failures) - 10:,} more errors")
        return "\n".join(lines)


def arrange_by_species(
    input_folder: str | Path,
    classifier: SpeciesClassifier,
    cache: SpeciesCache,
    *,
    dry_run: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    folder_name_fn: Callable[[str], str] | None = None,
) -> SpeciesArrangeResult:
    """Classify and file every image in `input_folder`.

    Every image gets exactly one classification attempt; an image that
    cannot even be decoded is recorded as an error and left where it is -
    never moved on a guess. `dry_run` still classifies (see cache.
    get_or_classify - the answer is cached either way, so a dry run is not
    wasted work) and fills in the whole result, including `moves`, without
    touching the filesystem beyond that.

    `on_progress(done, total)`, if given, is called after every image -
    the CLI's own progress line (see cli.py); this function has no opinion
    on how progress is reported.

    `folder_name_fn`, if given, maps the classifier's own species string to
    the folder name actually used (e.g. translating an English common name
    to Hebrew) - defaults to `sanitize_species_folder_name`, applied
    directly to the classifier's answer.
    """
    from ..analyzer.io import enumerate_ground_truth

    name_fn = folder_name_fn or sanitize_species_folder_name
    input_folder = Path(input_folder)
    images = enumerate_ground_truth(input_folder)
    result = SpeciesArrangeResult(total=len(images))

    for index, source in enumerate(images, start=1):
        try:
            prediction = cache.get_or_classify(str(source), classifier)
            result.classified += 1
            folder_name = name_fn(prediction.species)
            result.species_counts[folder_name] = result.species_counts.get(folder_name, 0) + 1

            destination_dir = input_folder / folder_name
            destination = destination_dir / source.name
            if source.resolve() == destination.resolve():
                # Already filed - a re-run must be a no-op, not a shuffle.
                result.skipped += 1
                continue

            final = unique_destination(destination)
            if final != destination:
                result.renamed += 1
                logger.info("Collision: %s -> %s", source.name, final.name)

            if not dry_run:
                destination_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(final))
            result.moved += 1
            result.moves[str(source)] = final
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the pass
            result.errors += 1
            result.failures.append((str(source), f"{type(exc).__name__}: {exc}"))
            logger.warning("Could not classify/move %s: %s", source, exc)

        if on_progress:
            on_progress(index, result.total)

    return result
