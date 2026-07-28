"""`picklikeme review` - open a ranked folder for review.

    picklikeme rank   --input "D:\\Shoot"
    picklikeme review --input "D:\\Shoot"

The second command takes the same argument as the first and nothing else. That
is the whole point of the sidecar: `rank` leaves the ranking inside the folder
it ranked, so `review` finds it by computing one path rather than searching
history, and the photographer never types a timestamp or an internal path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config import cli_prefix
from ..organize import DEFAULT_SELECTION_PERCENTAGE
from ..sidecar import RANKING_FILENAME, SIDECAR_DIRNAME, has_ranking, ranking_path

logger = logging.getLogger("picklikeme.review")


def build_review_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="picklikeme review",
        description=(
            "Review a ranked folder and file it. Shows the model's ordering, lets you "
            "override it, and moves the files into _Selected / _Rejected when you say so. "
            "Never runs the model."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="The folder to review. Must already have been ranked by `picklikeme rank`.",
    )
    parser.add_argument(
        "--ranking",
        default=None,
        help=f"Use this ranking CSV instead of the folder's own "
        f"{SIDECAR_DIRNAME}/{RANKING_FILENAME}. Only needed if the ranking was written "
        "elsewhere (`rank --output-csv`, or a `train` run).",
    )
    parser.add_argument(
        "--keep-percent",
        type=float,
        default=DEFAULT_SELECTION_PERCENTAGE,
        help=f"Percentage of the ranking selected on open (default: {DEFAULT_SELECTION_PERCENTAGE:g}). "
        "Changeable in the page; manual decisions always win over it.",
    )
    parser.add_argument("--annotations-db", default=None, help="Knowledge-base SQLite file")
    parser.add_argument(
        "--annotations-config",
        default=None,
        help="YAML file defining the annotation fields and their values",
    )
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 8757)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def run_review(args: argparse.Namespace) -> int:
    from ..analyzer.annotation_config import DEFAULT_ANNOTATIONS_CONFIG, load_annotation_fields
    from ..analyzer.annotations import DEFAULT_ANNOTATIONS_DB, AnnotationStore
    from .server import DEFAULT_REVIEW_PORT, serve_review
    from .session import ReviewSession
    from .thumbnails import close_detections

    _configure_logging(getattr(args, "verbose", False))

    folder = Path(args.input)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    # The one place the two-command flow can break down, so it says exactly how
    # to fix it rather than failing on a missing file deeper in.
    if not args.ranking and not has_ranking(folder):
        raise SystemExit(
            f"No ranking found for {folder}.\n"
            f"  Expected: {ranking_path(folder)}\n"
            f"  Rank it first:  {cli_prefix()} rank --input \"{folder}\"\n"
            f"  Or point at an existing ranking:  --ranking <csv>"
        )

    fields_config = load_annotation_fields(
        Path(args.annotations_config) if args.annotations_config else DEFAULT_ANNOTATIONS_CONFIG
    )
    store = AnnotationStore(
        Path(args.annotations_db) if args.annotations_db else DEFAULT_ANNOTATIONS_DB,
        fields_config=fields_config,
    )
    try:
        session = ReviewSession(
            folder,
            store,
            ranking_file=Path(args.ranking) if args.ranking else None,
            keep_percent=args.keep_percent,
        )
        serve_review(
            session,
            store,
            args.port or DEFAULT_REVIEW_PORT,
            open_browser=not args.no_browser,
        )
    finally:
        close_detections()
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_review(build_review_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
