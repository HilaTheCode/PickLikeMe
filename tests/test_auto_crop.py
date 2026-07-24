import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.auto_crop import discover_raw_images, resolve_device


class DiscoverRawTests(unittest.TestCase):
    def test_finds_nef_arw_cr3_dng_recursively_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            names = ["a.NEF", "b.arw", "c.CR3", "d.dng", "sub/e.Cr3", "sub/f.DNG"]
            for n in names:
                (root / n).write_bytes(b"raw")
            (root / "note.txt").write_bytes(b"skip")
            (root / "prev.jpg").write_bytes(b"skip")

            found = {Path(p).name for p in discover_raw_images(root)}
            self.assertEqual(found, {"a.NEF", "b.arw", "c.CR3", "d.dng", "e.Cr3", "f.DNG"})


class ResolveDeviceTests(unittest.TestCase):
    def test_auto_uses_cuda_when_available(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device(None), "cuda")

    def test_auto_falls_back_to_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device(None), "cpu")
            self.assertEqual(resolve_device("cuda"), "cpu")

    def test_explicit_cpu(self):
        self.assertEqual(resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
