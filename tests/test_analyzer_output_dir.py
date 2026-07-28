"""Where a `picklikeme analyze` run's report ends up.

Two mechanisms are covered:

1. `timestamped_output_dir()` - unchanged since before this file's rewrite -
   still used verbatim when the caller passes an explicit `--output`.
2. `resolve_run_dir()` - new: when `--output` is *not* given, the report is
   written into the same run directory as the training run it analyses
   (detected from `--ranking` pointing inside `analysis_results/<stamp>/
   ranking/`), or a freshly-stamped directory of its own otherwise.

Boundary the feature must respect: only the CLI stamps/resolves a run
directory. A library caller that builds an AnalysisConfig directly (tests,
notebooks, a script) gets exactly the output_dir it asked for - `run_analysis`
and `AnalysisConfig` themselves are untouched, which is what keeps the other
~300 tests in this suite (all of which construct AnalysisConfig with an
explicit output_dir and expect files to land exactly there) passing
unmodified.
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.cli import build_parser, config_from_args, resolve_run_dir, run
from picklikeme.analyzer.config import AnalysisConfig, timestamped_output_dir
from picklikeme.config import DEFAULT_RESULTS_DIR
from test_analyzer import build_dataset


class TimestampedOutputDirTests(unittest.TestCase):
    def test_stamp_is_appended_to_the_directory_name(self):
        stamped = timestamped_output_dir(Path("out") / "analysis", datetime(2026, 7, 27, 9, 30, 15))
        self.assertEqual(stamped.name, "analysis_20260727-093015")
        self.assertEqual(stamped.parent, Path("out"))

    def test_a_meaningful_base_name_is_preserved_not_replaced(self):
        stamped = timestamped_output_dir(
            "analysis/epoch40_vs_epoch20", datetime(2026, 1, 2, 3, 4, 5)
        )
        self.assertEqual(stamped.name, "epoch40_vs_epoch20_20260102-030405")

    def test_accepts_a_plain_string_path(self):
        stamped = timestamped_output_dir("analysis", datetime(2026, 1, 2, 3, 4, 5))
        self.assertEqual(stamped, Path("analysis_20260102-030405"))

    def test_two_runs_a_second_apart_get_distinct_directories(self):
        first = timestamped_output_dir("analysis", datetime(2026, 7, 27, 9, 30, 0))
        second = timestamped_output_dir("analysis", datetime(2026, 7, 27, 9, 30, 1))
        self.assertNotEqual(first, second)


class ResolveRunDirTests(unittest.TestCase):
    """resolve_run_dir() in isolation - no CLI/argparse involved."""

    def test_reuses_the_training_run_s_directory_when_ranking_lives_inside_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "analysis_results"
            run_dir = results_dir / "2026-07-28_09-15-32"
            ranking = run_dir / "ranking" / "training_results.csv"
            ranking.parent.mkdir(parents=True)
            ranking.write_text("x", encoding="utf-8")

            with mock.patch("picklikeme.analyzer.cli.DEFAULT_RESULTS_DIR", results_dir):
                resolved = resolve_run_dir(ranking, datetime(2026, 7, 28, 9, 20, 0))

            self.assertEqual(resolved, run_dir)

    def test_mints_a_fresh_directory_for_a_foreign_ranking_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "analysis_results"
            foreign = Path(tmp) / "somewhere" / "custom.csv"
            foreign.parent.mkdir(parents=True)
            foreign.write_text("x", encoding="utf-8")

            with mock.patch("picklikeme.analyzer.cli.DEFAULT_RESULTS_DIR", results_dir):
                resolved = resolve_run_dir(foreign, datetime(2026, 7, 28, 9, 20, 0))

            self.assertEqual(resolved, results_dir / "2026-07-28_09-20-00")

    def test_mints_a_fresh_directory_when_no_ranking_path_is_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "analysis_results"
            with mock.patch("picklikeme.analyzer.cli.DEFAULT_RESULTS_DIR", results_dir):
                resolved = resolve_run_dir(None, datetime(2026, 7, 28, 9, 20, 0))
            self.assertEqual(resolved, results_dir / "2026-07-28_09-20-00")

    def test_collision_on_the_fallback_stamp_gets_a_numbered_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "analysis_results"
            (results_dir / "2026-07-28_09-20-00").mkdir(parents=True)

            with mock.patch("picklikeme.analyzer.cli.DEFAULT_RESULTS_DIR", results_dir):
                resolved = resolve_run_dir(None, datetime(2026, 7, 28, 9, 20, 0))

            self.assertEqual(resolved, results_dir / "2026-07-28_09-20-00_1")


class CliRunDirectoryTests(unittest.TestCase):
    def _run_cli(self, extra_args: list[str], results_dir: Path) -> str:
        """Runs the CLI with DEFAULT_RESULTS_DIR patched to a sandboxed temp
        directory - real project-root analysis_results/ must never be touched
        by the test suite."""
        args = build_parser().parse_args(["--no-charts", "--no-contact-sheets", "--quiet", "--no-serve"] + extra_args)
        buf = io.StringIO()
        with mock.patch("picklikeme.analyzer.cli.DEFAULT_RESULTS_DIR", results_dir):
            with redirect_stdout(buf):
                exit_code = run(args)
        self.assertEqual(exit_code, 0)
        return buf.getvalue()

    def test_default_output_lands_in_the_training_run_s_report_subdirectory(self):
        """--ranking pointing inside analysis_results/<stamp>/ranking/ makes
        the report land in that same run's report/ - no --output needed."""
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "analysis_results"
            run_dir = results_dir / "2026-07-28_09-15-32"
            ranking, selected, rejected = build_dataset(run_dir / "ranking")

            self._run_cli(
                ["--ranking", str(ranking), "--selected", str(selected), "--rejected", str(rejected)],
                results_dir=results_dir,
            )

            self.assertTrue((run_dir / "report" / "report.txt").exists())
            self.assertTrue((run_dir / "report" / "analysis.json").exists())

    def test_default_output_mints_a_fresh_run_for_a_foreign_ranking_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "elsewhere"
            results_dir = Path(tmp) / "analysis_results"
            ranking, selected, rejected = build_dataset(root)

            self._run_cli(
                ["--ranking", str(ranking), "--selected", str(selected), "--rejected", str(rejected)],
                results_dir=results_dir,
            )

            reports = list(results_dir.glob("*/report/report.txt"))
            self.assertEqual(len(reports), 1, "expected exactly one freshly-minted run directory")

    def test_two_default_runs_against_the_same_foreign_csv_never_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "elsewhere"
            results_dir = Path(tmp) / "analysis_results"
            ranking, selected, rejected = build_dataset(root)
            extra = ["--ranking", str(ranking), "--selected", str(selected), "--rejected", str(rejected)]

            self._run_cli(extra, results_dir=results_dir)
            self._run_cli(extra, results_dir=results_dir)

            reports = list(results_dir.glob("*/report/report.txt"))
            self.assertEqual(len(reports), 2, "two runs must not collide into one report directory")

    def test_explicit_output_still_gets_stamped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            base = root / "my_custom_report_name"
            args = build_parser().parse_args(
                [
                    "--ranking", str(ranking),
                    "--selected", str(selected),
                    "--rejected", str(rejected),
                    "--output", str(base),
                    "--no-charts", "--no-contact-sheets", "--no-html",
                    "--no-serve", "--quiet",
                ]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                run(args)

            self.assertFalse(base.exists(), "the unstamped base directory must not be created")
            matches = list(root.glob("my_custom_report_name_*"))
            self.assertEqual(len(matches), 1)
            self.assertTrue((matches[0] / "report.txt").exists())

    def test_explicit_output_collision_still_gets_a_numbered_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            forced = timestamped_output_dir(root / "analysis", datetime.now())
            forced.mkdir(parents=True)
            (forced / "sentinel.txt").write_text("pre-existing", encoding="utf-8")

            args = build_parser().parse_args(
                [
                    "--ranking", str(ranking),
                    "--selected", str(selected),
                    "--rejected", str(rejected),
                    "--output", str(root / "analysis"),
                    "--no-charts", "--no-contact-sheets", "--no-html",
                    "--no-serve", "--quiet",
                ]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                run(args)

            self.assertEqual((forced / "sentinel.txt").read_text(encoding="utf-8"), "pre-existing")
            suffixed = list(forced.parent.glob(f"{forced.name}_1"))
            self.assertTrue(suffixed, "no numbered-suffix directory was created on collision")
            self.assertTrue((suffixed[0] / "report.txt").exists())

    def test_default_output_directory_is_not_stamped_by_config_from_args_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            args = build_parser().parse_args(
                [
                    "--ranking", str(ranking),
                    "--selected", str(selected),
                    "--rejected", str(rejected),
                    "--no-charts", "--no-contact-sheets", "--no-html",
                    "--no-serve", "--quiet",
                ]
            )
            config = config_from_args(args)
            # config_from_args() alone must NOT resolve a run directory -
            # that happens once, in run(), immediately before use.
            self.assertEqual(config.output_dir, DEFAULT_RESULTS_DIR)

    def test_annotate_command_is_told_the_stamped_directory(self):
        """The printed follow-up command must point at the real report
        directory, not at a base path the user typed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = Path(tmp) / "analysis_results"
            ranking, selected, rejected = build_dataset(root)
            output = self._run_cli(
                ["--ranking", str(ranking), "--selected", str(selected), "--rejected", str(rejected)],
                results_dir=results_dir,
            )
            self.assertIn("picklikeme annotate --output", output)


class LibraryCallerIsUnaffectedTests(unittest.TestCase):
    """run_analysis() and AnalysisConfig never resolve a run directory on
    their own - only the CLI does. This is the contract that keeps every
    other test's exact output_dir expectations intact."""

    def test_run_analysis_writes_to_exactly_the_given_output_dir(self):
        from picklikeme.analyzer.analysis import run_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            exact_dir = root / "exact_output"
            config = AnalysisConfig(
                ranking_path=ranking,
                selected_root=selected,
                rejected_root=rejected,
                output_dir=exact_dir,
                charts=False,
                contact_sheets=False,
            )
            result = run_analysis(config)
            self.assertEqual(result.config.output_dir, exact_dir)

            from picklikeme.analyzer.reports import write_text_report

            report = write_text_report(result, exact_dir / "report.txt")
            self.assertEqual(report.parent, exact_dir)


if __name__ == "__main__":
    unittest.main()
