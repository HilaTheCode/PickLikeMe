"""The per-shoot state directory: a ranking stored with the images it describes.

`picklikeme review --input <folder>` has to find that folder's ranking without
the photographer ever naming a file. The way to make that deterministic is not
a better search - it is to stop putting the ranking somewhere unrelated in the
first place. `rank` writes here, `review` reads exactly this path, and nothing
scans, guesses, or consults a global index.

    <folder>/
        .picklikeme/
            ranking.csv                    # the AI model's; chunks ranking_1.csv, ...
            ranking-classic-vision.csv     # one file per additional strategy
            run.json                       # provenance: what produced it, and when
        IMG_0001.NEF
        ...

**One file per analysis module, never a shared one.** Ranking strategies are
independent analysis modules (see `picklikeme.ranking`) that produce
independent metadata: an image can carry an AI score and a Classic Vision
score at the same time, and running one must never destroy the other's
results. So each strategy owns its own CSV, and a folder's full set of scores
is whatever files are present - discovered, not enumerated from a hard-coded
list, so a future module needs no change here.

The AI model keeps the original unsuffixed `ranking.csv` name. That is
deliberate backwards compatibility, not an exception to the rule: every shoot
ranked before strategies existed already has that file, and `review` has
always read exactly that path.

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
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SIDECAR_DIRNAME = ".picklikeme"
RANKING_FILENAME = "ranking.csv"
RUN_FILENAME = "run.json"

# The strategy whose scores live in the unsuffixed `ranking.csv` - see the
# module docstring. Named here rather than imported from `picklikeme.ranking`
# so this module stays free of the registry (organize.py imports it, and must
# not pull in a ranking strategy to move files).
AI_STRATEGY_ID = "ai-model"

# Every other strategy's file is `ranking-<strategy_id>.csv`.
STRATEGY_RANKING_PREFIX = "ranking-"


def strategy_ranking_path(folder: str | Path, strategy_id: str) -> Path:
    """Where one analysis module's scores for this folder live."""
    if strategy_id == AI_STRATEGY_ID:
        return ranking_path(folder)
    return sidecar_dir(folder) / f"{STRATEGY_RANKING_PREFIX}{strategy_id}.csv"


def discover_strategy_rankings(folder: str | Path) -> dict[str, Path]:
    """Every strategy that has scored this folder -> its ranking CSV.

    Discovered from what is on disk rather than by asking the registry, so a
    folder scored by a module that no longer exists still displays its
    results, and a module added later needs no change here or in the review
    session that consumes this.

    Continuation chunks (`ranking_1.csv`, `ranking-x_2.csv` - see
    write_results_csv) are excluded: they belong to the file they continue,
    and `analyzer.io.discover_chunks` finds them from it.
    """
    directory = sidecar_dir(folder)
    if not directory.is_dir():
        return {}

    found: dict[str, Path] = {}
    canonical = directory / RANKING_FILENAME
    if canonical.is_file():
        found[AI_STRATEGY_ID] = canonical

    for candidate in sorted(directory.glob(f"{STRATEGY_RANKING_PREFIX}*.csv")):
        strategy_id = candidate.stem[len(STRATEGY_RANKING_PREFIX):]
        if _CHUNK_SUFFIX.search(strategy_id):
            continue
        if strategy_id:
            found[strategy_id] = candidate
    return found


_CHUNK_SUFFIX = re.compile(r"_\d+$")

# Any analysis module's self-describing per-image sidecar - a filter report
# ("why was this image excluded from scoring") or a metrics report ("the raw
# measurements behind the combined score") - ends in one of these. Discovered
# by glob + the payload's own "strategy" field (see discover_filter_reports/
# discover_metric_reports below), never by naming a strategy here, so a
# future module's diagnostics appear in the review UI the moment it writes
# one in this shape, with no change to this file.
FILTER_REPORT_SUFFIX = "_filters.json"
METRICS_REPORT_SUFFIX = "_metrics.json"


def _discover_self_described_reports(folder: str | Path, suffix: str, payload_key: str) -> dict[str, dict]:
    """Every `*{suffix}` file under `.picklikeme/` whose JSON payload names
    its own strategy - shared by discover_filter_reports and
    discover_metric_reports, which differ only in which suffix and which key
    inside the payload holds the per-image data."""
    directory = sidecar_dir(folder)
    if not directory.is_dir():
        return {}

    found: dict[str, dict] = {}
    for candidate in sorted(directory.glob(f"*{suffix}")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s: %s", candidate, exc)
            continue
        if not isinstance(payload, dict):
            continue
        strategy_id = payload.get("strategy")
        data = payload.get(payload_key)
        if strategy_id and isinstance(data, dict):
            found[strategy_id] = data
    return found


def discover_filter_reports(folder: str | Path) -> dict[str, dict[str, str]]:
    """Every analysis module's filter verdicts for this folder, keyed by
    strategy id -> {image_path: reason}.

    A module that filters images (Classic Vision today) writes one of these
    beside its ranking CSV (see `ranking.classic.write_filter_report`); an
    image absent from a module's ranking but present here has an explicit,
    honest reason instead of an unexplained gap.
    """
    return _discover_self_described_reports(folder, FILTER_REPORT_SUFFIX, "images")


def discover_metric_reports(folder: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Every analysis module's raw, per-metric measurements for this folder,
    keyed by strategy id -> {image_path: {metric_name: value}}.

    The breakdown behind a module's single combined score (see
    `ranking.classic.write_metrics_report`) - what the Loupe's diagnostics
    line reads to show, e.g., why a weak-eyed image still ranked
    respectably.
    """
    return _discover_self_described_reports(folder, METRICS_REPORT_SUFFIX, "metrics")


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
    """Point every analysis module's scores at where the images actually are now.

    Arranging moves files into `_Selected`/`_Rejected`, which would leave the
    stored scores describing paths that no longer exist - and the folder could
    never be reviewed a second time. `OrganizeResult.moves` is the exact
    old -> new map, so this is a rewrite rather than a re-derivation.

    Applies to EVERY strategy's CSV, not only the AI model's: Organize is a
    workflow operation that may consume analysis metadata, but it must never
    invalidate it, and a Classic Vision score has to survive a folder being
    filed exactly as an AI score does.

    Every chunk is rewritten in place, preamble intact. Rows whose path is not
    in `moves` (skipped, failed, or never moved) are left exactly as they were.
    Returns the total number of rows repointed across all of them.

    Never raises: a shoot whose files were filed successfully must not report
    failure because its bookkeeping could not be updated.
    """
    if not moves:
        return 0

    targets = discover_strategy_rankings(folder)
    if not targets:
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

    from .analyzer.io import discover_chunks

    rewritten = 0
    for target in targets.values():
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
        logger.info(
            "Repointed %d score row(s) across %d analysis file(s) in %s after arranging",
            rewritten, len(targets), sidecar_dir(folder),
        )
    return rewritten
