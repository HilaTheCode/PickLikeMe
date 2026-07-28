"""Command-line entry point for the ingestion pipeline.

    picklikeme build-manifest --archive-root D:\\Photos\\Wildlife
    picklikeme generate-previews --archive-root D:\\Photos\\Wildlife
    picklikeme report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .manifest import build_manifest, save_manifest
from .metadata import ExifToolNotFoundError
from .preview import generate_previews
from .report import gap_histogram, summarize
from .scan import ScanIssues, scan_select_reject_roots


def _build_manifest_command(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest_path)

    if args.select_root and args.reject_root:
        images, issues = scan_select_reject_roots(args.select_root, args.reject_root)
        manifest = pd.DataFrame(
            [
                {
                    "image_path": str(path),
                    "label": img.label,
                    "shoot_id": img.shoot_id,
                    "burst_id": None,
                    "sequence_in_burst": None,
                    "burst_size": None,
                    "capture_timestamp": None,
                    "subsecond": None,
                    "camera_model": None,
                    "lens_model": None,
                    "raw_format": img.raw_format,
                    "iso": None,
                    "shutter_speed": None,
                    "f_number": None,
                    "focal_length": None,
                    "file_size_bytes": path.stat().st_size,
                    "file_mtime": path.stat().st_mtime,
                    "metadata_status": "manual_scan",
                    "pipeline_version": 1,
                }
                for img in images
                for path in [img.path.resolve()]
            ]
        )
        save_manifest(manifest, manifest_path)
        print(f"Manifest written to {manifest_path}")
        print()
        print(summarize(manifest, issues))
        return

    archive_root = Path(args.archive_root)
    try:
        manifest, issues = build_manifest(
            archive_root=archive_root,
            exiftool_path=args.exiftool_path,
            gap_seconds=args.gap_seconds,
            existing_manifest_path=manifest_path if manifest_path.exists() else None,
        )
    except ExifToolNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    save_manifest(manifest, manifest_path)
    print(f"Manifest written to {manifest_path}")
    print()
    print(summarize(manifest, issues))


def _generate_previews_command(args: argparse.Namespace) -> None:
    archive_root = Path(args.archive_root).resolve()
    manifest = pd.read_parquet(args.manifest_path)
    rows = [(str(archive_root / row.image_path), row.image_path) for row in manifest.itertuples()]

    failures = generate_previews(
        rows, Path(args.preview_root), long_edge=args.long_edge, workers=args.workers
    )
    print(f"Generated previews for {len(rows) - len(failures)}/{len(rows)} images")
    if failures:
        print(f"{len(failures)} failures:")
        for path, err in failures[:20]:
            print(f"  - {path}: {err}")


def _report_command(args: argparse.Namespace) -> None:
    manifest = pd.read_parquet(args.manifest_path)
    print(summarize(manifest, ScanIssues()))
    if gap_histogram(manifest, Path(args.gap_histogram_path)):
        print(f"\nGap histogram written to {args.gap_histogram_path}")


def _analyze_command(args: argparse.Namespace) -> None:
    from ..analyzer.cli import run

    raise SystemExit(run(args))


def _annotate_command(args: argparse.Namespace) -> None:
    from ..analyzer.cli import run_annotate

    raise SystemExit(run_annotate(args))


def _review_command(args: argparse.Namespace) -> None:
    from ..review.cli import run_review

    raise SystemExit(run_review(args))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="picklikeme", description="Pick Like Me ingestion pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser(
        "build-manifest", help="Scan the archive, extract metadata, reconstruct bursts, write the manifest"
    )
    build_parser.add_argument("--archive-root", help="Root folder containing one subfolder per shoot")
    build_parser.add_argument("--select-root", help="Full path to a Select/Keep root folder to scan recursively")
    build_parser.add_argument("--reject-root", help="Full path to a Reject root folder to scan recursively")
    build_parser.add_argument("--manifest-path", default="data/manifest.parquet")
    build_parser.add_argument("--exiftool-path", default="exiftool")
    build_parser.add_argument("--gap-seconds", type=float, default=1.5, help="Max gap between frames in the same burst")
    build_parser.set_defaults(func=_build_manifest_command)

    preview_parser = sub.add_parser(
        "generate-previews", help="Decode RAW files referenced in the manifest to a JPEG preview cache"
    )
    preview_parser.add_argument("--archive-root", required=True)
    preview_parser.add_argument("--manifest-path", default="data/manifest.parquet")
    preview_parser.add_argument("--preview-root", default="data/previews")
    preview_parser.add_argument("--long-edge", type=int, default=512)
    preview_parser.add_argument("--workers", type=int, default=8)
    preview_parser.set_defaults(func=_generate_previews_command)

    report_parser = sub.add_parser("report", help="Print a summary of the current manifest")
    report_parser.add_argument("--manifest-path", default="data/manifest.parquet")
    report_parser.add_argument("--gap-histogram-path", default="data/reports/gap_histogram.png")
    report_parser.set_defaults(func=_report_command)

    # The analyzer owns its own (large) argument set, so it is attached whole
    # rather than restated here; importing it lazily keeps `picklikeme --help`
    # from pulling in matplotlib.
    from ..analyzer.cli import build_parser as build_analyzer_parser

    analyze_parser = sub.add_parser(
        "analyze",
        parents=[build_analyzer_parser(add_help=False)],
        help="Measure model quality against your keep/reject decisions",
        description="Measure model quality against your keep/reject decisions",
    )
    analyze_parser.set_defaults(func=_analyze_command)

    from ..analyzer.cli import build_annotate_parser

    annotate_parser = sub.add_parser(
        "annotate",
        parents=[build_annotate_parser(add_help=False)],
        help="Serve an analysis report so false-negative annotations can be saved",
        description="Serve an analysis report so false-negative annotations can be saved",
    )
    annotate_parser.set_defaults(func=_annotate_command)

    from ..review.cli import build_review_parser

    review_parser = sub.add_parser(
        "review",
        parents=[build_review_parser(add_help=False)],
        help="Review a ranked folder and file it into _Selected / _Rejected",
        description="Review a ranked folder and file it into _Selected / _Rejected",
    )
    review_parser.set_defaults(func=_review_command)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
