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

from ..config import DEFAULT_RESULTS_DIR, RUN_TIMESTAMP_FORMAT, cli_prefix, fatal_errors_logged_to_stdout, format_duration
from .annotation_config import DEFAULT_ANNOTATIONS_CONFIG
from .config import OPTIMIZATION_TARGETS, AnalysisConfig, timestamped_output_dir

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
        help="Base directory for this run's reports. By default, if --ranking points inside "
        f"an {DEFAULT_RESULTS_DIR.name}/<timestamp>/ranking/ directory (as written by "
        "picklikeme train), the report is written into that same run's report/ subdirectory; "
        f"otherwise a fresh timestamped directory is created under {DEFAULT_RESULTS_DIR}. Pass "
        "--output to override this and force a specific (still timestamped) location instead.",
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
        "--annotations-config",
        default=None,
        help="YAML file defining the annotation fields and their values "
        f"(default: {DEFAULT_ANNOTATIONS_CONFIG})",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Do not read the annotation database",
    )
    parser.add_argument(
        "--serve",
        dest="serve",
        action="store_true",
        default=True,
        help="After analysing, serve the report on 127.0.0.1 and open a browser (default)",
    )
    parser.add_argument(
        "--no-serve",
        dest="serve",
        action="store_false",
        help="Write the report and exit without starting the server or opening a browser "
        "(for scripts and automation)",
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
        "annotations_config": "annotations_config",
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


def resolve_run_dir(ranking_path: Path | None, now: datetime) -> Path:
    """The run directory this analysis' report should live under.

    Reuses the training run's own directory when --ranking points inside one
    (analysis_results/<timestamp>/ranking/...), so the report lands next to
    the CSV that produced it without the user having to pass a --run-dir by
    hand. Anything else (a hand-picked or foreign ranking file) gets a fresh,
    freshly-timestamped run directory of its own.
    """
    if ranking_path is not None:
        resolved = ranking_path.resolve()
        if resolved.parent.name == "ranking" and resolved.parent.parent.parent == DEFAULT_RESULTS_DIR.resolve():
            return resolved.parent.parent

    stamp = now.strftime(RUN_TIMESTAMP_FORMAT)
    candidate = DEFAULT_RESULTS_DIR / stamp
    suffix = 1
    while candidate.exists():
        candidate = DEFAULT_RESULTS_DIR / f"{stamp}_{suffix}"
        suffix += 1
    return candidate


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
    if args.output is not None:
        # Explicit --output: honour it exactly, stamped the same way it always
        # has been. The %Y%m%d-%H%M%S stamp has 1-second resolution, so two
        # runs that both finish parsing args within the same second (a small
        # ranking, no charts, or a tight test/automation loop) would otherwise
        # collide - a numbered suffix is added when that happens.
        stamped = timestamped_output_dir(config.output_dir, datetime.now())
        output_dir = stamped
        suffix = 1
        while output_dir.exists():
            output_dir = stamped.with_name(f"{stamped.name}_{suffix}")
            suffix += 1
    else:
        # No explicit --output: land the report inside the run directory this
        # ranking CSV belongs to (see resolve_run_dir), under report/. Only
        # the report/ leaf gets a collision suffix - the run directory itself
        # is stable and shared with the training run that produced the CSV.
        run_dir = resolve_run_dir(config.ranking_path, datetime.now())
        base_report_dir = run_dir / "report"
        output_dir = base_report_dir
        suffix = 1
        while output_dir.exists():
            output_dir = base_report_dir.with_name(f"{base_report_dir.name}_{suffix}")
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

        serve(
            config.output_dir,
            config.annotations_db_path,
            args.port or DEFAULT_PORT,
            annotations_config_path=config.annotations_config_path,
        )
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
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Analysis directory containing report.html, e.g. "
        f"{DEFAULT_RESULTS_DIR}/<timestamp>/report (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument("--annotations-db", default=None, help="Knowledge-base SQLite file")
    parser.add_argument(
        "--annotations-config",
        default=None,
        help=f"YAML file defining the annotation fields and their values (default: {DEFAULT_ANNOTATIONS_CONFIG})",
    )
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
        annotations_config_path=Path(args.annotations_config) if args.annotations_config else DEFAULT_ANNOTATIONS_CONFIG,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
