"""`picklikeme review` - open a ranked folder for review.

    picklikeme rank   --input "D:\\Shoot"
    picklikeme review --input "D:\\Shoot"

The second command takes the same argument as the first and nothing else. That
is the whole point of the sidecar: `rank` leaves the ranking inside the folder
it ranked, so `review` finds it by computing one path rather than searching
history, and the photographer never types a timestamp or an internal path.

`--input` is optional: `picklikeme review` with nothing after it starts with
an empty gallery, and the "Open Folder..." button in the page picks a folder
from there instead - the only way to review a folder that was never ranked
at all, since there is no ranking to compute a path from.
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
            "Review a folder and file it. The AI ranking, if any, is shown as a read-only "
            "suggestion; every image is independently Keep, Reject or Neutral, and only that "
            "verdict ever moves a file, into _Selected / _Rejected, when you say so. Never "
            "runs the model."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "--input",
        default=None,
        help="The folder to review. Omit to start empty and pick one from the page's own "
        "'Open Folder...' button instead.",
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
        help=f"The AI's suggestion threshold on open (default: {DEFAULT_SELECTION_PERCENTAGE:g}) - what "
        "fraction of ranked images it hints should be kept. Purely informational: it never sets "
        "anyone's review status by itself. Changeable in the page.",
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

    if args.ranking and not args.input:
        raise SystemExit("--ranking requires --input (it says which ranking belongs to that folder).")

    folder = Path(args.input) if args.input else None
    if folder is not None and not folder.is_dir():
        # Not fatal - ReviewSession opens it anyway (folder_missing=True) so
        # the page can load and offer to relocate it (moved, renamed, or a
        # changed drive letter): the photographer picks the new location once
        # and every stored path is repointed automatically.
        print(
            f"Folder not found: {folder}\n"
            "  Starting anyway - the page will ask where it went; every stored path "
            "(the ranking, any review decisions) is repointed automatically once you pick it."
        )
    # An unranked folder is not an error - ReviewSession already handles it
    # (every image starts Neutral, sorted for Keep/Reject/Neutral by hand) -
    # just worth a heads-up, since `rank` first is the common case.
    elif folder is not None and not args.ranking and not has_ranking(folder):
        print(
            f"No ranking found for {folder}; there is no AI suggestion for any image in it.\n"
            f"  Expected: {ranking_path(folder)}\n"
            f"  To rank it first:  {cli_prefix()} rank --input \"{folder}\"\n"
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
