import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.raw_io import RawImageLoader


class RawLoaderTests(unittest.TestCase):
    def test_load_image_returns_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))

            loader = RawImageLoader(raw_root=".", output_size=(64, 64))
            image = loader.load_image(str(image_path))

            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
