import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.bird_crop import BirdDetection, NormalizedCrop, compute_composition_crop


def _det(x1, y1, x2, y2, score=0.9):
    return BirdDetection(box=(x1, y1, x2, y2), score=score)


class CompositionCropTests(unittest.TestCase):
    W, H = 400, 300  # 4:3 image

    def test_preserves_image_aspect_ratio(self):
        crop = compute_composition_crop(_det(150, 120, 250, 180), self.W, self.H, margin_frac=0.0)
        norm_w = crop.right - crop.left
        norm_h = crop.bottom - crop.top
        # aspect-preserving crop => equal normalized width/height (Lightroom encoding)
        self.assertAlmostEqual(norm_w, norm_h, places=5)
        # ...which means the crop's PIXEL aspect equals the image aspect (never square photo)
        pixel_aspect = (norm_w * self.W) / (norm_h * self.H)
        self.assertAlmostEqual(pixel_aspect, self.W / self.H, places=5)

    def test_all_coordinates_in_unit_range(self):
        for det in [_det(0, 0, 40, 30), _det(360, 270, 400, 300), _det(150, 120, 250, 180)]:
            crop = compute_composition_crop(det, self.W, self.H, margin_frac=0.2)
            for v in (crop.left, crop.top, crop.right, crop.bottom):
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)
            self.assertLess(crop.left, crop.right)
            self.assertLess(crop.top, crop.bottom)

    def test_centered_bird_stays_centered(self):
        crop = compute_composition_crop(_det(150, 120, 250, 180), self.W, self.H, margin_frac=0.1)
        self.assertAlmostEqual((crop.left + crop.right) / 2, 0.5, places=5)
        self.assertAlmostEqual((crop.top + crop.bottom) / 2, 0.5, places=5)

    def test_bird_near_edge_shifts_inside_frame(self):
        # bird hugging the top-left; crop must stay in-frame (left/top clamp to >= 0)
        crop = compute_composition_crop(_det(0, 0, 60, 45), self.W, self.H, margin_frac=0.1)
        self.assertAlmostEqual(crop.left, 0.0, places=6)
        self.assertAlmostEqual(crop.top, 0.0, places=6)
        self.assertAlmostEqual(crop.right - crop.left, crop.bottom - crop.top, places=5)

    def test_huge_margin_falls_back_to_full_frame(self):
        crop = compute_composition_crop(_det(150, 120, 250, 180), self.W, self.H, margin_frac=5.0)
        self.assertEqual(crop, NormalizedCrop(0.0, 0.0, 1.0, 1.0))

    def test_margin_enlarges_crop(self):
        small = compute_composition_crop(_det(150, 120, 250, 180), self.W, self.H, margin_frac=0.0)
        large = compute_composition_crop(_det(150, 120, 250, 180), self.W, self.H, margin_frac=0.3)
        self.assertGreater(large.right - large.left, small.right - small.left)


if __name__ == "__main__":
    unittest.main()
