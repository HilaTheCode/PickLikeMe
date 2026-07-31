"""`picklikeme arrange-species` - classify a folder's images by species and
file each into a same-named subfolder.

    picklikeme arrange-species --input "D:\\Shoot\\_Selected"

Meant to run after Review/Arrange, on the Keep folder those produced - a
separate, optional step, never part of `rank` or `review`. See
species/__init__.py for the full picture.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ..config import format_duration

logger = logging.getLogger("picklikeme.species")


def build_arrange_species_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="picklikeme arrange-species",
        description=(
            "Classify every image in a folder by species and file it into a same-named "
            "subfolder (Unknown/ for anything unsupported or low-confidence). Intended to run "
            "on the Keep folder Review/Arrange already produced - a separate, optional step "
            "that never changes the detector, the ranking model, or a review decision."
        ),
        add_help=add_help,
    )
    parser.add_argument("--input", required=True, help="The folder to arrange (e.g. the shoot's _Selected folder).")
    parser.add_argument(
        "--classifier",
        default="bioclip2",
        help="Which registered classifier to use (default: bioclip2). See species.classifier.build_classifier.",
    )
    parser.add_argument(
        "--species-list",
        default=None,
        help="A text file, one species name per line ('#' comments and blank lines ignored), "
        "replacing the classifier's built-in candidate list. Extending species coverage is "
        "editing this file, never retraining or changing code.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Below this confidence (0-1), an image goes to Unknown/ rather than a guessed "
        "species folder (default: 0.5).",
    )
    parser.add_argument(
        "--species-db",
        default=None,
        help="SQLite file caching predictions by content identity (default: cache/species.db). "
        "Re-running an already-classified folder reads from here instead of reclassifying.",
    )
    parser.add_argument("--device", default="cpu", help="torch device for the classifier (default: cpu).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Classify and report what would happen; move nothing."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def run_arrange_species(args: argparse.Namespace) -> int:
    from .arrange import arrange_by_species
    from .cache import DEFAULT_SPECIES_DB, SpeciesCache
    from .classifier import build_classifier

    _configure_logging(getattr(args, "verbose", False))

    folder = Path(args.input)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    print(f"Loading the species classifier ({args.classifier})...")
    classifier = build_classifier(
        args.classifier,
        species_list_path=args.species_list,
        min_confidence=args.min_confidence,
        device=args.device,
    )

    db_path = Path(args.species_db) if args.species_db else DEFAULT_SPECIES_DB
    cache = SpeciesCache(db_path)

    start = time.monotonic()

    def _report_progress(done: int, total: int) -> None:
        if total == 0 or (done != total and done % 10 != 0):
            return
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = format_duration((total - done) / rate) if rate > 0 else "n/a"
        percent = 100 * done / total
        print(f"  {done:,}/{total:,} ({percent:.1f}%) | {rate:.1f} img/s | ETA {eta}")

    try:
        print(f"Arranging {folder} by species{' (dry run)' if args.dry_run else ''}...")
        result = arrange_by_species(
            folder, classifier, cache, dry_run=args.dry_run, on_progress=_report_progress
        )
    finally:
        cache.close()

    print(result.render())
    return 1 if result.errors and result.moved == 0 else 0


def main(argv: list[str] | None = None) -> int:
    return run_arrange_species(build_arrange_species_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
