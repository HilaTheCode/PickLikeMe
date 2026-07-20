import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.dataset import FolderLabelDataset


class FolderLabelDatasetTests(unittest.TestCase):
    def test_dataset_uses_folder_location_as_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            select_root = root / "accepted"
            reject_root = root / "rejected"
            (select_root / "nested").mkdir(parents=True)
            (reject_root / "nested").mkdir(parents=True)
            (select_root / "nested" / "keep.arw").write_bytes(b"keep")
            (reject_root / "nested" / "drop.nef").write_bytes(b"drop")

            dataset = FolderLabelDataset(select_root=str(select_root), reject_root=str(reject_root), raw_root=str(root))
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0].label, 1)
            self.assertEqual(dataset[1].label, 0)


if __name__ == "__main__":
    unittest.main()
