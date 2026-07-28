import csv
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.config import DEFAULT_MAX_CSV_ROWS, DEFAULT_RESULTS_DIR, run_dir_for_timestamp
from picklikeme.dataset import FolderLabelDataset
from picklikeme.train import CSV_PREAMBLE_LINES, write_results_csv


class TrainingCsvOutputTests(unittest.TestCase):
    def test_write_results_csv_creates_expected_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            select_root = Path(tmpdir) / "select"
            reject_root = Path(tmpdir) / "reject"
            select_root.mkdir(parents=True)
            reject_root.mkdir(parents=True)
            (select_root / "img1.arw").write_bytes(b"x")
            (reject_root / "img2.arw").write_bytes(b"x")

            dataset = FolderLabelDataset(select_root=str(select_root), reject_root=str(reject_root), raw_root=str(select_root))
            ranked = [(str(dataset[0].image_path), 0.95, 1), (str(dataset[1].image_path), 0.2, 0)]
            output_path = Path(tmpdir) / "results.csv"

            written_paths = write_results_csv(output_path, dataset, ranked, str(select_root), str(reject_root), max_rows=10)

            self.assertEqual(written_paths, [output_path])
            self.assertTrue(output_path.exists())
            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

            self.assertIn(["relevant_images", str(len(dataset))], rows)
            self.assertIn(["detected_sequences", "2"], rows)
            self.assertIn(["1", str(dataset[0].image_path), "0.950000", "1"], rows)

    def test_write_results_csv_rolls_to_new_file_when_row_limit_exceeded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            select_root = Path(tmpdir) / "select"
            reject_root = Path(tmpdir) / "reject"
            select_root.mkdir(parents=True)
            reject_root.mkdir(parents=True)
            for idx in range(3):
                (select_root / f"img{idx}.arw").write_bytes(b"x")
            (reject_root / "bad.arw").write_bytes(b"x")

            dataset = FolderLabelDataset(select_root=str(select_root), reject_root=str(reject_root), raw_root=str(select_root))
            ranked = [(str(item.image_path), float(idx), 1) for idx, item in enumerate(dataset)]
            output_path = Path(tmpdir) / "results.csv"

            written_paths = write_results_csv(output_path, dataset, ranked, str(select_root), str(reject_root), max_rows=2)

            self.assertEqual(len(written_paths), 2)
            self.assertTrue(output_path.exists())
            self.assertTrue((Path(tmpdir) / "results_1.csv").exists())
            with (Path(tmpdir) / "results_1.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertTrue(any(row and row[0] == "3" and row[1].endswith("img2.arw") for row in rows))


class RowLimitTests(unittest.TestCase):
    """The split limit is configuration, not a literal buried in the writer."""

    def test_default_comes_from_the_project_configuration(self):
        import inspect

        from picklikeme import rank, train

        self.assertEqual(DEFAULT_MAX_CSV_ROWS, 30_000)
        # The writer's own default must be the configured value, not a copy.
        signature = inspect.signature(write_results_csv)
        self.assertEqual(signature.parameters["max_rows"].default, DEFAULT_MAX_CSV_ROWS)

        # Both CLIs must inherit the configured value rather than restate it.
        for parser in (train.build_arg_parser(), rank.build_arg_parser()):
            action = next(a for a in parser._actions if a.dest == "max_rows")
            self.assertEqual(action.default, DEFAULT_MAX_CSV_ROWS)

    def test_a_whole_file_stays_within_the_limit_including_the_preamble(self):
        """max_rows bounds total lines, not just data rows - which is why the
        preamble is subtracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            select_root = Path(tmpdir) / "select"
            reject_root = Path(tmpdir) / "reject"
            select_root.mkdir(parents=True)
            reject_root.mkdir(parents=True)
            for idx in range(30):
                (select_root / f"img{idx}.arw").write_bytes(b"x")

            dataset = FolderLabelDataset(
                select_root=str(select_root), reject_root=str(reject_root), raw_root=str(select_root)
            )
            ranked = [(str(item.image_path), float(idx), 1) for idx, item in enumerate(dataset)]
            limit = 12
            written = write_results_csv(
                Path(tmpdir) / "r.csv", dataset, ranked, str(select_root), str(reject_root), max_rows=limit
            )

            self.assertGreater(len(written), 1, "30 rows at a 12-line limit must split")
            for path in written:
                lines = path.read_text(encoding="utf-8").strip().splitlines()
                self.assertLessEqual(len(lines), limit, f"{path.name} exceeded the limit")
            data_rows = sum(
                len(p.read_text(encoding="utf-8").strip().splitlines()) - CSV_PREAMBLE_LINES
                for p in written
            )
            self.assertEqual(data_rows, 30, "no ranking row may be lost across the split")

    def test_a_large_limit_keeps_everything_in_one_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            select_root = Path(tmpdir) / "select"
            reject_root = Path(tmpdir) / "reject"
            select_root.mkdir(parents=True)
            reject_root.mkdir(parents=True)
            for idx in range(50):
                (select_root / f"img{idx}.arw").write_bytes(b"x")

            dataset = FolderLabelDataset(
                select_root=str(select_root), reject_root=str(reject_root), raw_root=str(select_root)
            )
            ranked = [(str(item.image_path), float(idx), 1) for idx, item in enumerate(dataset)]
            written = write_results_csv(
                Path(tmpdir) / "r.csv", dataset, ranked, str(select_root), str(reject_root)
            )
            self.assertEqual(len(written), 1, "50 rows must not split at the 30,000 default")


class RunDirForTimestampTests(unittest.TestCase):
    """A training run's own directory - analysis_results/<timestamp>/ - which
    now carries the uniqueness that filenames used to (see train_and_rank)."""

    def test_run_dir_is_nested_under_the_project_results_directory(self):
        run_dir = run_dir_for_timestamp(datetime(2026, 7, 25, 14, 30, 0))
        self.assertEqual(run_dir.parent, DEFAULT_RESULTS_DIR)
        self.assertEqual(run_dir.name, "2026-07-25_14-30-00")

    def test_two_runs_a_second_apart_get_distinct_directories(self):
        first = run_dir_for_timestamp(datetime(2026, 7, 25, 14, 30, 0))
        second = run_dir_for_timestamp(datetime(2026, 7, 25, 14, 30, 1))
        self.assertNotEqual(first, second)

    def test_chunked_files_land_together_in_the_run_s_ranking_directory(self):
        """write_results_csv appends _1/_2 for overflow; both files must land
        in the same run directory since the directory - not the filename -
        now carries the uniqueness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            select_root = Path(tmpdir) / "select"
            reject_root = Path(tmpdir) / "reject"
            select_root.mkdir(parents=True)
            reject_root.mkdir(parents=True)
            for idx in range(3):
                (select_root / f"img{idx}.arw").write_bytes(b"x")
            (reject_root / "bad.arw").write_bytes(b"x")

            dataset = FolderLabelDataset(select_root=str(select_root), reject_root=str(reject_root), raw_root=str(select_root))
            ranked = [(str(item.image_path), float(idx), 1) for idx, item in enumerate(dataset)]
            # A run's ranking directory (train_and_rank derives this from
            # run_dir_for_timestamp() / "ranking" / <base filename>).
            output_path = Path(tmpdir) / "2026-07-25_14-30-00" / "ranking" / "results.csv"

            written_paths = write_results_csv(output_path, dataset, ranked, str(select_root), str(reject_root), max_rows=2)

            self.assertEqual([path.name for path in written_paths], ["results.csv", "results_1.csv"])
            self.assertEqual({path.parent for path in written_paths}, {output_path.parent})


if __name__ == "__main__":
    unittest.main()
