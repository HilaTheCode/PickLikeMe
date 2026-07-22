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

    def test_letterbox_preserves_aspect_ratio_with_padding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "wide.png"
            cv2.imwrite(str(image_path), np.full((32, 64, 3), 255, dtype=np.uint8))

            loader = RawImageLoader(raw_root=".", output_size=(64, 64), resize_mode="letterbox")
            image = loader.load_image(str(image_path))

            self.assertEqual(image.shape, (64, 64, 3))
            # 32x64 scales to 32x64 content centered vertically: 16 pad rows top and bottom
            self.assertTrue(np.all(image[:16] == 0.0))
            self.assertTrue(np.all(image[48:] == 0.0))
            self.assertTrue(np.all(image[16:48] == 1.0))

    def test_stretch_mode_reproduces_v1_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "wide.png"
            cv2.imwrite(str(image_path), np.full((32, 64, 3), 255, dtype=np.uint8))

            loader = RawImageLoader(raw_root=".", output_size=(64, 64), resize_mode="stretch")
            image = loader.load_image(str(image_path))

            self.assertEqual(image.shape, (64, 64, 3))
            self.assertTrue(np.all(image == 1.0))

    def test_invalid_resize_mode_rejected(self):
        with self.assertRaises(ValueError):
            RawImageLoader(raw_root=".", resize_mode="crop")


if __name__ == "__main__":
    unittest.main()
