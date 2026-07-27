"""Capability 17 - the analyzer command line.

    picklikeme analyze --ranking rankings.csv --selected keep/ --rejected drop/
    python -m picklikeme.analyzer --ranking rankings.csv --output analysis/

The CLI only parses arguments, builds an AnalysisConfig and prints; every
decision lives in the library, so the same analysis is reachable from a
notebook or a test without touching argv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from ..config import cli_prefix, fatal_errors_logged_to_stdout, format_duration
from .config import DEFAULT_ANALYSIS_DIR, OPTIMIZATION_TARGETS, AnalysisConfig, timestamped_output_dir

logger = logging.getLogger("picklikeme.analyzer")


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="picklikeme analyze",
        description="Measure model quality against your own keep/reject decisions",
        add_help=add_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  picklikeme analyze --ranking rankings.csv --selected keep/ --rejected drop/\n"
            "  picklikeme analyze --ranking training_results.csv          # labels come from the file\n"
            "  picklikeme analyze --ranking new.csv --compare-ranking old.csv --selected keep/ --rejected drop/\n"
            "  picklikeme analyze --config analysis.json --threshold 0.6\n"
        ),
    )
    parser.add_argument("--ranking", help="Ranking CSV from picklikeme rank / train (chunks are picked up automatically)")
    parser.add_argument("--selected", default=None, help="Folder of images you kept (ground-truth positives)")
    parser.add_argument("--rejected", default=None, help="Folder of images you rejected (ground-truth negatives)")
    parser.add_argument(
        "--output",
        default=None,
        help=f"Base directory for this run's reports (default: {DEFAULT_ANALYSIS_DIR}). The run's "
        "start date/time is appended to the folder name so consecutive analyses never overwrite "
        "each other (e.g. analysis_20260727-093015/).",
    )
    parser.add_argument("--config", default=None, help="JSON config file; explicit flags override it")
    parser.add_argument("--title", default=None, help="Report title")

    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: 0.5)")
    parser.add_argument(
        "--optimize-for",
        default=None,
        choices=sorted(OPTIMIZATION_TARGETS),
        help="What the recommended threshold should maximise (default: f1)",
    )
    parser.add_argument("--threshold-steps", type=int, default=None, help="Points in the threshold sweep (default: 101)")
    parser.add_argument("--borderline-low", type=float, default=None, help="Lower edge of the uncertainty band (default: 0.45)")
    parser.add_argument("--borderline-high", type=float, default=None, help="Upper edge of the uncertainty band (default: 0.55)")
    parser.add_argument("--max-examples", type=int, default=None, help="Rows per error table / contact sheet (default: 60)")
    parser.add_argument("--thumbnail-size", type=int, default=None, help="Preview thumbnail edge in pixels (default: 400)")
    parser.add_argument("--thumbnail-workers", type=int, default=None, help="Threads generating thumbnails (default: 8)")

    parser.add_argument("--compare-ranking", default=None, help="Second ranking file to compare against")
    parser.add_argument("--baseline-label", default=None, help="Name for the first run in comparison output")
    parser.add_argument("--compare-label", default=None, help="Name for the second run in comparison output")

    parser.add_argument(
        "--annotations-db",
        default=None,
        help="SQLite knowledge base of false-negative diagnoses "
        "(default: <project>/annotations/false_negatives.db, outside any output dir so it survives runs)",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Do not read the annotation database",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="After analysing, serve the report on 127.0.0.1 so annotations can be saved",
    )
    parser.add_argument("--port", type=int, default=None, help="Port for --serve / annotate (default: 8756)")
    parser.add_argument("--no-html", action="store_true", help="Skip the HTML report")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart rendering")
    parser.add_argument("--no-contact-sheets", action="store_true", help="Skip contact sheets (much faster)")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Merge config file and flags. Flags win; the file supplies the rest."""
    overrides: dict = {}
    mapping = {
        "ranking": "ranking_path",
        "selected": "selected_root",
        "rejected": "rejected_root",
        "output": "output_dir",
        "title": "report_title",
        "threshold": "threshold",
        "optimize_for": "optimize_for",
        "threshold_steps": "threshold_steps",
        "borderline_low": "borderline_low",
        "borderline_high": "borderline_high",
        "max_examples": "max_examples",
        "thumbnail_size": "thumbnail_size",
        "thumbnail_workers": "thumbnail_workers",
        "compare_ranking": "compare_ranking_path",
        "baseline_label": "baseline_label",
        "compare_label": "compare_label",
        "annotations_db": "annotations_db",
    }
    for arg_name, config_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[config_name] = value

    if getattr(args, "no_annotations", False):
        overrides["annotations_enabled"] = False
    if args.no_html:
        overrides["html_report"] = False
    if args.no_charts:
        overrides["charts"] = False
    if args.no_contact_sheets:
        overrides["contact_sheets"] = False
    if args.verbose:
        overrides["verbose"] = True

    if args.config:
        return AnalysisConfig.from_file(args.config, **overrides)
    if "ranking_path" not in overrides:
        raise SystemExit("--ranking is required (or supply it via --config).")
    return AnalysisConfig.from_dict(overrides)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,  # keep logs in the same stream as the report
    )


def run(args: argparse.Namespace) -> int:
    """Execute an analysis and write every enabled artefact."""
    from .analysis import run_analysis
    from .reports import render_full, render_summary, write_csv_reports, write_json_report, write_text_report

    config = config_from_args(args)
    _configure_logging(config.verbose)

    # Every CLI run gets its own timestamped folder, so two analyses never
    # overwrite each other's reports - stamped here, not inside AnalysisConfig
    # or run_analysis, so a library caller that builds a config directly (tests,
    # notebooks, a script) keeps exact control over where its output lands.
    #
    # The %Y%m%d-%H%M%S stamp has 1-second resolution, so two runs that both
    # finish parsing args within the same second (a small ranking, no charts,
    # or a tight test/automation loop) would otherwise collide. If the stamped
    # directory already exists, a numbered suffix is added - the same
    # never-overwrite guarantee the ranking/CSV outputs already give, just
    # applied to a directory instead of a file.
    stamped = timestamped_output_dir(config.output_dir, datetime.now())
    output_dir = stamped
    suffix = 1
    while output_dir.exists():
        output_dir = stamped.with_name(f"{stamped.name}_{suffix}")
        suffix += 1
    config = replace(config, output_dir=output_dir)

    with fatal_errors_logged_to_stdout():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Analysis output: {output_dir.resolve()}")

        result = run_analysis(config)
        written: list[Path] = [
            write_text_report(result, output_dir / "report.txt"),
            write_json_report(result, output_dir / "analysis.json"),
        ]
        written += write_csv_reports(result, output_dir / "tables")

        if config.charts:
            from .visualization import render_charts

            written += render_charts(result)

        if config.contact_sheets:
            from .contactsheets import render_contact_sheets

            written += render_contact_sheets(result)

        if config.html_report:
            from .reports.html import write_html_report

            written.append(write_html_report(result))

        print(render_summary(result) if args.quiet else render_full(result))
        print()
        print(f"Wrote {len(written)} file(s) to {output_dir.resolve()}")
        for path in written[:12]:
            print(f"  {path.relative_to(output_dir) if output_dir in path.parents else path}")
        if len(written) > 12:
            print(f"  ... and {len(written) - 12} more")
        if config.html_report:
            print(f"\nOpen: {(output_dir / 'report.html').resolve()}")
        if config.annotations_enabled and (
            result.annotation_summary is not None or result.fp_annotation_summary is not None
        ):
            bits = []
            if result.annotation_summary is not None:
                s = result.annotation_summary
                bits.append(f"false negatives {s.annotated}/{s.total_images}")
            if result.fp_annotation_summary is not None:
                s = result.fp_annotation_summary
                bits.append(f"false positives {s.annotated}/{s.total_images}")
            db = (result.annotation_summary or result.fp_annotation_summary)
            print(f"Annotations: {', '.join(bits)} annotated, {db.total_in_database} in {db.database_path}")
            if config.html_report and not getattr(args, "serve", False):
                print("To record diagnoses, run:")
                print(f'  {cli_prefix()} annotate --output "{output_dir}"')

    if getattr(args, "serve", False) and config.html_report:
        from .server import DEFAULT_PORT, serve

        serve(config.output_dir, config.annotations_db_path, args.port or DEFAULT_PORT)
    return 0


def build_annotate_parser(add_help: bool = True) -> "argparse.ArgumentParser":
    """`picklikeme annotate` - serve an existing report so Save works."""
    parser = argparse.ArgumentParser(
        prog="picklikeme annotate",
        description=(
            "Serve an existing analysis report on 127.0.0.1 so false-negative annotations "
            "can be saved to the knowledge base. Writes only the annotation database."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ANALYSIS_DIR),
        help=f"Analysis directory containing report.html (default: {DEFAULT_ANALYSIS_DIR})",
    )
    parser.add_argument("--annotations-db", default=None, help="Knowledge-base SQLite file")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 8756)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    return parser


def run_annotate(args: argparse.Namespace) -> int:
    from .annotations import DEFAULT_ANNOTATIONS_DB
    from .server import DEFAULT_PORT, serve

    _configure_logging(getattr(args, "verbose", False))
    serve(
        Path(args.output),
        Path(args.annotations_db) if args.annotations_db else DEFAULT_ANNOTATIONS_DB,
        args.port or DEFAULT_PORT,
        open_browser=not args.no_browser,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
