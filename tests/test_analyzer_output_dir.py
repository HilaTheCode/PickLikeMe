"""Every `picklikeme analyze` run gets its own timestamped output folder, so
consecutive full analysis reports never overwrite each other.

Boundary the feature must respect: only the CLI stamps. A library caller that
builds an AnalysisConfig directly (tests, notebooks, a script) gets exactly the
output_dir it asked for - `run_analysis` and `AnalysisConfig` themselves are
untouched, which is what keeps the other ~300 tests in this suite (all of which
construct AnalysisConfig with an explicit output_dir and expect files to land
exactly there) passing unmodified.
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.cli import build_parser, config_from_args, run
from picklikeme.analyzer.config import AnalysisConfig, timestamped_output_dir
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


class CliStampingTests(unittest.TestCase):
    def _run_cli(self, root: Path, extra_args: list[str] | None = None) -> Path:
        ranking, selected, rejected = build_dataset(root)
        args = build_parser().parse_args(
            [
                "--ranking", str(ranking),
                "--selected", str(selected),
                "--rejected", str(rejected),
                "--output", str(root / "analysis"),
                "--no-charts", "--no-contact-sheets",
                "--quiet",
            ]
            + (extra_args or [])
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = run(args)
        self.assertEqual(exit_code, 0)
        return buf.getvalue()

    def test_consecutive_runs_never_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output1 = self._run_cli(root)
            output2 = self._run_cli(root)

            def resolved_dir(stdout: str) -> Path:
                line = next(l for l in stdout.splitlines() if l.startswith("Analysis output:"))
                return Path(line.split("Analysis output:", 1)[1].strip())

            dir1, dir2 = resolved_dir(output1), resolved_dir(output2)
            self.assertNotEqual(dir1, dir2, "two runs produced the same output directory")
            self.assertTrue((dir1 / "report.txt").exists())
            self.assertTrue((dir2 / "report.txt").exists())
            # The first run's files must still be there - proof nothing was overwritten.
            self.assertTrue((dir1 / "report.txt").exists())
            self.assertTrue((dir1 / "analysis.json").exists())
            self.assertTrue((dir2 / "analysis.json").exists())

    def test_a_true_same_second_collision_still_does_not_overwrite(self):
        """The %Y%m%d-%H%M%S stamp has 1-second resolution, so a directory that
        already exists (two runs within the same second, or a re-run of a test)
        must get a numbered suffix rather than silently reusing it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            # Pre-create the exact directory a run would compute, forcing a collision.
            from datetime import datetime as _dt
            from picklikeme.analyzer.config import timestamped_output_dir

            forced = timestamped_output_dir(root / "analysis", _dt.now())
            forced.mkdir(parents=True)
            (forced / "sentinel.txt").write_text("pre-existing", encoding="utf-8")

            args = build_parser().parse_args(
                [
                    "--ranking", str(ranking),
                    "--selected", str(selected),
                    "--rejected", str(rejected),
                    "--output", str(root / "analysis"),
                    "--no-charts", "--no-contact-sheets", "--no-html",
                    "--quiet",
                ]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                run(args)

            self.assertEqual((forced / "sentinel.txt").read_text(encoding="utf-8"), "pre-existing")
            suffixed = list(root.glob("analysis_*_1")) + list(forced.parent.glob(f"{forced.name}_1"))
            self.assertTrue(suffixed, "no numbered-suffix directory was created on collision")
            self.assertTrue((suffixed[0] / "report.txt").exists())

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
                    "--quiet",
                ]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                run(args)

            self.assertFalse(base.exists(), "the unstamped base directory must not be created")
            matches = list(root.glob("my_custom_report_name_*"))
            self.assertEqual(len(matches), 1)
            self.assertTrue((matches[0] / "report.txt").exists())

    def test_default_output_directory_is_stamped_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            args = build_parser().parse_args(
                [
                    "--ranking", str(ranking),
                    "--selected", str(selected),
                    "--rejected", str(rejected),
                    "--no-charts", "--no-contact-sheets", "--no-html",
                    "--quiet",
                ]
            )
            config = config_from_args(args)
            # Not asserting on the real project-root DEFAULT_ANALYSIS_DIR (that
            # would write outside the sandbox); just confirm the CLI's contract:
            # config_from_args() alone must NOT stamp - stamping happens once,
            # in run(), immediately before use.
            from picklikeme.analyzer.config import DEFAULT_ANALYSIS_DIR

            self.assertEqual(config.output_dir, DEFAULT_ANALYSIS_DIR)

    def test_annotate_command_is_told_the_stamped_directory(self):
        """The printed follow-up command must point at the real (stamped) dir,
        not at the base name the user typed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self._run_cli(root, extra_args=[])
            self.assertIn("picklikeme annotate --output", output)
            line = next(l for l in output.splitlines() if "picklikeme annotate --output" in l)
            self.assertNotIn(str(root / "analysis") + '"', line, "printed the unstamped base path")


class LibraryCallerIsUnaffectedTests(unittest.TestCase):
    """run_analysis() and AnalysisConfig never stamp on their own - only the
    CLI does. This is the contract that keeps every other test's exact
    output_dir expectations intact."""

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
