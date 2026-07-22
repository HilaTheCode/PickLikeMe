import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.dataset import FolderLabelDataset, PathSuffixIndex


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

    def test_burst_ids_resolved_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            select_root = root / "accepted"
            reject_root = root / "rejected"
            (select_root / "nested").mkdir(parents=True)
            (reject_root / "nested").mkdir(parents=True)
            (select_root / "nested" / "keep.arw").write_bytes(b"keep")
            (reject_root / "nested" / "drop.nef").write_bytes(b"drop")

            manifest_csv = root / "manifest.csv"
            manifest_csv.write_text(
                "image_path,label,burst_id\n"
                "nested/keep.arw,1,burst-01\n"
                "nested/drop.nef,0,burst-01\n",
                encoding="utf-8",
            )

            dataset = FolderLabelDataset(
                select_root=str(select_root),
                reject_root=str(reject_root),
                raw_root=str(root),
                manifest_path=str(manifest_csv),
            )
            self.assertEqual(dataset[0].burst_id, "burst-01")
            self.assertEqual(dataset[1].burst_id, "burst-01")


class PathSuffixIndexTests(unittest.TestCase):
    def test_unique_suffix_match_resolves(self):
        index = PathSuffixIndex()
        index.add("shoot1/Keep/DSC0001.ARW", "b1")
        self.assertEqual(index.get(r"C:\archive\shoot1\Keep\DSC0001.ARW"), "b1")

    def test_repeated_filename_across_shoots_is_disambiguated_by_path(self):
        index = PathSuffixIndex()
        index.add("shoot1/Keep/DSC0001.ARW", "b1")
        index.add("shoot2/Keep/DSC0001.ARW", "b2")
        self.assertEqual(index.get(r"C:\archive\shoot2\Keep\DSC0001.ARW"), "b2")

    def test_ambiguous_match_returns_none(self):
        index = PathSuffixIndex()
        index.add("DSC0001.ARW", "b1")
        index.add("DSC0001.ARW", "b2")
        self.assertIsNone(index.get(r"C:\archive\shoot1\Keep\DSC0001.ARW"))

    def test_unknown_path_returns_none(self):
        index = PathSuffixIndex()
        index.add("shoot1/Keep/DSC0001.ARW", "b1")
        self.assertIsNone(index.get(r"C:\archive\shoot1\Keep\DSC9999.ARW"))


if __name__ == "__main__":
    unittest.main()
