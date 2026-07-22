import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.dataset import LabelDataset


class BurstPreferenceDatasetTests(unittest.TestCase):
    def test_dataset_reads_preference_and_burst_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.csv"
            manifest_path.write_text(
                "image_path,label,burst_id,preference\n"
                "img1.jpg,0,b1,0\n"
                "img2.jpg,1,b1,2\n",
                encoding="utf-8",
            )

            dataset = LabelDataset(str(manifest_path), str(root))
            first = dataset[0]
            second = dataset[1]

            self.assertEqual(first.burst_id, "b1")
            self.assertEqual(first.preference, 0.0)
            self.assertEqual(second.preference, 2.0)


if __name__ == "__main__":
    unittest.main()
