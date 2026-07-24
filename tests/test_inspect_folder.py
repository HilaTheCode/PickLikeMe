import sys
import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.bird_crop import BirdDetection, CropParams, build_crop
from picklikeme.inspect_crops import (
    PipelineResult,
    _build_loader,
    build_pair_sheet,
    discover_images,
    draw_bbox_overlay,
    resolve_device,
    run_pipeline,
    write_folder_report,
)
from picklikeme.raw_io import RawImageLoader

FIXED_BOX = (10.0, 10.0, 50.0, 50.0)


class StubDetector:
    """Stands in for BirdDetector via its public detect_best_bird API,
    returning a fixed bird detection (no real model)."""

    def detect_best_bird(self, image_rgb):
        return BirdDetection(box=FIXED_BOX, score=0.9)


class NoBirdDetector:
    def detect_best_bird(self, image_rgb):
        return None


class RunPipelineFaithfulnessTests(unittest.TestCase):
    def test_model_input_matches_build_crop_plus_letterbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "img.png")
            frame = (np.random.default_rng(0).random((60, 80, 3)) * 255).astype(np.uint8)
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            loader = RawImageLoader(raw_root=tmp, output_size=(32, 32), resize_mode="letterbox")
            params = CropParams()

            full = loader._decode_full_frame(path)
            expected_input = loader._letterbox(build_crop(full, StubDetector(), params).crop)

            result = run_pipeline(loader, StubDetector(), path, params)
            self.assertTrue(result.found)
            self.assertAlmostEqual(result.score, 0.9, places=5)
            self.assertEqual(result.model_input_rgb.shape, (32, 32, 3))
            self.assertTrue(np.array_equal(result.model_input_rgb, expected_input))
            self.assertEqual(result.original_size, (80, 60))

    def test_fallback_when_no_bird(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "img.png")
            cv2.imwrite(path, np.full((40, 40, 3), 100, dtype=np.uint8))
            loader = RawImageLoader(raw_root=tmp, output_size=(32, 32), resize_mode="letterbox")
            result = run_pipeline(loader, NoBirdDetector(), path, CropParams())
            self.assertFalse(result.found)
            self.assertIsNone(result.box)
            self.assertEqual(result.model_input_rgb.shape, (32, 32, 3))


class RenderingTests(unittest.TestCase):
    def _result(self, found=True):
        full = np.full((60, 80, 3), 120, dtype=np.uint8)
        return PipelineResult(
            source_path="/x/DSC_0001.arw",
            found=found,
            score=0.85 if found else None,
            box=(10, 10, 50, 50) if found else None,
            expanded_box=(8, 8, 52, 52) if found else None,
            original_size=(80, 60),
            crop_size=(40, 40) if found else (80, 60),
            full_rgb=full,
            model_input_rgb=np.full((32, 32, 3), 200, dtype=np.uint8),
        )

    def test_draw_overlay_returns_image(self):
        self.assertIsInstance(draw_bbox_overlay(self._result(True)), Image.Image)
        self.assertIsInstance(draw_bbox_overlay(self._result(False)), Image.Image)

    def test_pair_sheet_dimensions(self):
        left = Image.new("RGB", (64, 64))
        right = Image.new("RGB", (64, 64))
        sheet = build_pair_sheet([(left, right, "a.arw"), (left, right, "b.arw")], thumb=64, caption_h=18, arrow_w=44)
        self.assertEqual(sheet.size, (64 + 44 + 64, (64 + 18) * 2))

    def test_report_txt_and_csv_flag_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_folder_report(out, [self._result(True), self._result(False)], errors=[])
            txt = (out / "report.txt").read_text(encoding="utf-8")
            csv_text = (out / "report.csv").read_text(encoding="utf-8")
            self.assertIn("Detection success rate:     50.0%", txt)
            self.assertIn("FALLBACK", txt)
            self.assertIn("conf=0.850", txt)
            self.assertIn("True", csv_text)  # fallback column


class DeviceDefaultTests(unittest.TestCase):
    def test_auto_selects_cuda_when_available(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device(None), "cuda")
            self.assertEqual(resolve_device("cuda"), "cuda")

    def test_auto_falls_back_to_cpu_when_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device(None), "cpu")
            self.assertEqual(resolve_device("cuda"), "cpu")

    def test_explicit_cpu_is_respected(self):
        self.assertEqual(resolve_device("cpu"), "cpu")


class DiscoverImagesTests(unittest.TestCase):
    def test_finds_raw_formats_recursively_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            names = ["a.CR3", "b.cr3", "c.Nef", "d.ARW", "sub/e.nef", "sub/f.Arw"]
            for n in names:
                (root / n).write_bytes(b"raw")
            (root / "notes.txt").write_bytes(b"skip")  # unsupported

            found = {Path(p).name for p in discover_images(root)}
            self.assertEqual(found, {"a.CR3", "b.cr3", "c.Nef", "d.ARW", "e.nef", "f.Arw"})

    def test_cr3_nef_arw_are_supported(self):
        for ext in (".cr3", ".nef", ".arw"):
            self.assertIn(ext, RawImageLoader.RAW_EXTENSIONS)


class BuildLoaderDefaultTests(unittest.TestCase):
    def test_omitting_output_size_uses_training_default(self):
        args = SimpleNamespace(output_size=None, resize_mode="letterbox")
        loader = _build_loader(".", args)
        # Matches RawImageLoader's own default -- exactly what training uses.
        self.assertEqual(loader.output_size, RawImageLoader(".").output_size)
        self.assertEqual(loader.resize_mode, "letterbox")

    def test_explicit_output_size_is_applied(self):
        args = SimpleNamespace(output_size=256, resize_mode="letterbox")
        loader = _build_loader(".", args)
        self.assertEqual(loader.output_size, (256, 256))


if __name__ == "__main__":
    unittest.main()
