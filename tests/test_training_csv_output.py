import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.dataset import FolderLabelDataset
from picklikeme.train import write_results_csv


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


if __name__ == "__main__":
    unittest.main()
