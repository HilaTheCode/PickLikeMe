import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.bird_crop import crop_cache_path, save_crop_png
from picklikeme.inspect_crops import (
    build_contact_sheet,
    classify_pairs,
    find_cached_pairs,
    _model_input_thumb,
)
from picklikeme.raw_io import RawImageLoader


class StubDetector:
    """Reports a bird for cache files whose parent-encoded flag says so. Here we
    key off pixel brightness: white cached crop -> bird, black -> fallback."""

    def best_bird_box(self, image_rgb):
        return (0, 0, 1, 1) if image_rgb.mean() > 127 else None


def _make_cache(root: Path):
    select = root / "select"
    reject = root / "reject"
    (select).mkdir()
    (reject).mkdir()
    cache = root / "cache"
    cache.mkdir()

    # Two "detected" (white crop) and one "fallback" (black frame).
    specs = [
        (select / "bird_a.arw", 255),
        (reject / "bird_b.arw", 255),
        (reject / "nobird_c.arw", 0),
    ]
    for source, value in specs:
        source.write_bytes(b"raw")  # enumeration only reads the filename
        crop = np.full((20, 30, 3), value, dtype=np.uint8)
        save_crop_png(crop_cache_path(cache, source), crop)
    return select, reject, cache


class FindPairsTests(unittest.TestCase):
    def test_finds_only_cached_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            select, reject, cache = _make_cache(root)
            # An extra uncached source must be excluded.
            (select / "uncached.arw").write_bytes(b"raw")

            pairs = find_cached_pairs(str(select), str(reject), cache)
            names = sorted(Path(s).name for s, _c in pairs)
            self.assertEqual(names, ["bird_a.arw", "bird_b.arw", "nobird_c.arw"])


class ClassifyTests(unittest.TestCase):
    def test_classify_splits_detected_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            select, reject, cache = _make_cache(root)
            pairs = find_cached_pairs(str(select), str(reject), cache)
            reader = RawImageLoader(raw_root=str(select))

            tagged = classify_pairs(pairs, StubDetector(), reader)
            found = {Path(s).name: f for s, _c, f in tagged}
            self.assertTrue(found["bird_a.arw"])
            self.assertTrue(found["bird_b.arw"])
            self.assertFalse(found["nobird_c.arw"])


class ContactSheetTests(unittest.TestCase):
    def test_sheet_dimensions_and_write(self):
        cells = [(Image.new("RGB", (64, 64), (10, 200, 10)), f"img_{i}.arw") for i in range(5)]
        sheet = build_contact_sheet(cells, cols=3, thumb=64, caption_h=16)
        # 5 cells, 3 cols -> 2 rows
        self.assertEqual(sheet.size, (3 * 64, 2 * (64 + 16)))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sheet.png"
            sheet.save(out)
            self.assertTrue(out.exists() and out.stat().st_size > 0)

    def test_model_input_thumb_is_square_and_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            select, reject, cache = _make_cache(root)
            loader = RawImageLoader(
                raw_root=str(select), output_size=(48, 48), resize_mode="letterbox", crop_cache_dir=str(cache)
            )
            source = str(select / "bird_a.arw")
            thumb = _model_input_thumb(loader, source, thumb=64)
            self.assertEqual(thumb.size, (64, 64))  # square thumbnail
            # White cached crop -> bright thumbnail (proves it used the cache)
            self.assertGreater(np.asarray(thumb).mean(), 120)


if __name__ == "__main__":
    unittest.main()
