"""The per-shoot state directory: a ranking stored with the images it describes.

`picklikeme review --input <folder>` has to find that folder's ranking without
the photographer ever naming a file. The way to make that deterministic is not
a better search - it is to stop putting the ranking somewhere unrelated in the
first place. `rank` writes here, `review` reads exactly this path, and nothing
scans, guesses, or consults a global index.

    <folder>/
        .picklikeme/
            ranking.csv        # canonical; chunks are ranking_1.csv, ranking_2.csv, ...
            run.json           # provenance: what produced the ranking, and when
        IMG_0001.NEF
        ...

Consequences of storing it here rather than in a project-level directory:

- **Deterministic.** One path, computed from the folder. No timestamps in a
  filename, no ambiguity about which of several runs is "the" ranking.
- **Portable.** Move the shoot to another drive or machine and its ranking
  travels with it; the folder always answers "have I been ranked?" on its own.
- **Not a new mechanism.** Same CSV, same `analyzer.io.load_ranking` reader,
  same chunking. Only the location changed.

Deliberately top-level rather than inside `review/` or `analyzer/`: `rank`
writes it and `organize` rewrites it after moving files, and neither of those
may import either of those packages (see `organize.py`'s module docstring).
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SIDECAR_DIRNAME = ".picklikeme"
RANKING_FILENAME = "ranking.csv"
RUN_FILENAME = "run.json"


def sidecar_dir(folder: str | Path) -> Path:
    """The shoot's state directory. Not created by this call."""
    return Path(folder) / SIDECAR_DIRNAME


def ranking_path(folder: str | Path) -> Path:
    """Where this folder's ranking lives. The one path `review` looks at."""
    return sidecar_dir(folder) / RANKING_FILENAME


def run_metadata_path(folder: str | Path) -> Path:
    return sidecar_dir(folder) / RUN_FILENAME


def has_ranking(folder: str | Path) -> bool:
    return ranking_path(folder).is_file()


def ensure_sidecar_dir(folder: str | Path) -> Path:
    target = sidecar_dir(folder)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_run_metadata(folder: str | Path, **fields) -> Path:
    """Record what produced this ranking, so the review UI can say so.

    Provenance only - nothing reads this to make a decision, so a missing or
    unreadable run.json never stops a review.
    """
    payload = {"written_at": datetime.now().isoformat(timespec="seconds"), **fields}
    ensure_sidecar_dir(folder)
    target = run_metadata_path(folder)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_run_metadata(folder: str | Path) -> dict:
    """Provenance for display, or `{}` when absent or unreadable."""
    target = run_metadata_path(folder)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", target, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def rewrite_ranking_paths(folder: str | Path, moves: dict[str, Path]) -> int:
    """Point the ranking at where the images actually are now.

    Arranging moves every file into `_Selected`/`_Rejected`, which would leave
    the ranking describing paths that no longer exist - and the folder could
    never be reviewed a second time. `OrganizeResult.moves` is the exact
    old -> new map, so this is a rewrite rather than a re-derivation.

    Every chunk is rewritten in place, preamble intact. Rows whose path is not
    in `moves` (skipped, failed, or never moved) are left exactly as they were.
    Returns the number of rows repointed.

    Never raises: a shoot whose files were filed successfully must not report
    failure because its bookkeeping could not be updated.
    """
    if not moves:
        return 0

    from .analyzer.io import discover_chunks

    target = ranking_path(folder)
    if not target.is_file():
        return 0

    # Both the raw string the CSV holds and its resolved form, since the ranking
    # may have been written with a different spelling of the same path.
    lookup: dict[str, str] = {}
    for old, new in moves.items():
        lookup[old] = str(new)
        try:
            lookup[str(Path(old).resolve())] = str(new)
        except OSError:  # pragma: no cover - unresolvable path
            continue

    rewritten = 0
    for chunk in discover_chunks(target):
        try:
            with chunk.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.reader(handle))
        except OSError as exc:
            logger.warning("Could not read %s to repoint it: %s", chunk, exc)
            continue

        changed = False
        for row in rows:
            for index, cell in enumerate(row):
                replacement = lookup.get(cell.strip())
                if replacement is not None and replacement != cell:
                    row[index] = replacement
                    changed = True
                    rewritten += 1
        if not changed:
            continue
        try:
            with chunk.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
        except OSError as exc:
            logger.warning("Could not repoint %s: %s", chunk, exc)

    if rewritten:
        logger.info("Repointed %d ranking row(s) in %s after arranging", rewritten, target)
    return rewritten
