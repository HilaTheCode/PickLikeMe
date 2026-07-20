import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.ingest.scan import scan_select_reject_roots


class SelectRejectScanTests(unittest.TestCase):
    def test_scan_select_reject_roots_recurses_and_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            select_root = root / "select"
            reject_root = root / "reject"

            (select_root / "seqA" / "nested").mkdir(parents=True)
            (reject_root / "seqA" / "nested").mkdir(parents=True)

            (select_root / "seqA" / "nested" / "keep1.nef").write_bytes(b"keep")
            (select_root / "seqB" / "keep2.cr3").parent.mkdir(parents=True, exist_ok=True)
            (select_root / "seqB" / "keep2.cr3").write_bytes(b"keep")

            (reject_root / "seqA" / "nested" / "reject1.nef").write_bytes(b"reject")
            (reject_root / "seqB" / "reject2.arw").parent.mkdir(parents=True, exist_ok=True)
            (reject_root / "seqB" / "reject2.arw").write_bytes(b"reject")

            images, issues = scan_select_reject_roots(select_root, reject_root)

            self.assertEqual(len(images), 4)
            self.assertEqual(issues.unmatched_subfolders, [])
            self.assertEqual(sum(1 for img in images if img.label == 1), 2)
            self.assertEqual(sum(1 for img in images if img.label == 0), 2)
            self.assertEqual({img.shoot_id for img in images if img.label == 1}, {"seqA/nested", "seqB"})
            self.assertTrue(all(str(img.path).startswith(str(root)) for img in images))


if __name__ == "__main__":
    unittest.main()
