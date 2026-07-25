import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.bird_crop import (
    COCO_BIRD_CLASS,
    DOMESTIC_ANIMAL_CLASSES,
    SUPPORTED_ANIMAL_CLASSES,
    WILDLIFE_CLASSES,
    BirdDetection,
    BirdDetector,
    CropParams,
    crop_cache_path,
    crop_to_box,
    downscale_long_side,
    expand_and_clamp_box,
    read_crop_params,
    save_crop_png,
    write_crop_params,
)
from picklikeme.raw_io import RawImageLoader


def _detector_with_fake_model(boxes, labels, scores, conf_threshold=0.3, classes=None):
    """A BirdDetector whose torchvision model is replaced by a fixed output, so
    detect_best_bird's real selection logic can be tested without downloading
    or running the actual network."""
    detector = BirdDetector.__new__(BirdDetector)
    detector._torch = torch
    detector.device = "cpu"
    detector.conf_threshold = conf_threshold
    detector.classes = frozenset(SUPPORTED_ANIMAL_CLASSES if classes is None else classes)
    output = {
        "boxes": torch.tensor(boxes, dtype=torch.float),
        "labels": torch.tensor(labels),
        "scores": torch.tensor(scores),
    }
    detector.model = lambda images: [output]
    return detector


class DetectBestBirdTests(unittest.TestCase):
    IMG = np.zeros((40, 40, 3), dtype=np.uint8)

    def test_picks_highest_scoring_bird_above_threshold(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 2, 2]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS, 1],  # two birds + a person
            scores=[0.5, 0.8, 0.99],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertIsInstance(detection, BirdDetection)
        self.assertEqual(detection.box, (5.0, 5.0, 20.0, 20.0))  # bird with score 0.8
        self.assertAlmostEqual(detection.score, 0.8, places=5)
        self.assertEqual(detection.label, COCO_BIRD_CLASS)

    def test_best_bird_box_wrapper_returns_just_the_box(self):
        detector = _detector_with_fake_model(
            boxes=[[5, 5, 20, 20]], labels=[COCO_BIRD_CLASS], scores=[0.8]
        )
        self.assertEqual(detector.best_bird_box(self.IMG), (5.0, 5.0, 20.0, 20.0))

    def test_none_when_only_non_birds_or_below_threshold(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 20, 20]],
            labels=[COCO_BIRD_CLASS, 1],  # bird below threshold, person high
            scores=[0.1, 0.99],
        )
        self.assertIsNone(detector.detect_best_bird(self.IMG))
        self.assertIsNone(detector.best_bird_box(self.IMG))


class SupportedClassTests(unittest.TestCase):
    IMG = np.zeros((40, 40, 3), dtype=np.uint8)

    def test_coco_indices_match_torchvision_metadata(self):
        """Pins our hardcoded indices to torchvision's own category list, so a
        weights-metadata change can never silently point us at a wrong class."""
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

        categories = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"]
        for index, name in SUPPORTED_ANIMAL_CLASSES.items():
            self.assertEqual(categories[index], name)

    def test_required_wildlife_classes_are_supported(self):
        self.assertEqual(
            set(WILDLIFE_CLASSES.values()), {"bird", "elephant", "bear", "zebra", "giraffe"}
        )
        self.assertTrue(set(WILDLIFE_CLASSES) <= set(SUPPORTED_ANIMAL_CLASSES))
        self.assertTrue(set(DOMESTIC_ANIMAL_CLASSES) <= set(SUPPORTED_ANIMAL_CLASSES))

    def test_each_supported_class_is_detected_and_its_label_reported(self):
        for index, name in SUPPORTED_ANIMAL_CLASSES.items():
            with self.subTest(animal=name):
                detector = _detector_with_fake_model(
                    boxes=[[5, 5, 20, 20]], labels=[index], scores=[0.9]
                )
                detection = detector.detect_best_bird(self.IMG)
                self.assertIsNotNone(detection)
                self.assertEqual(detection.label, index)

    def test_highest_confidence_wins_across_different_animal_classes(self):
        # An elephant scoring higher than a bird must win: selection is by
        # confidence only, with no class priority.
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 20, 20]],
            labels=[COCO_BIRD_CLASS, 22],
            scores=[0.6, 0.85],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.box, (5.0, 5.0, 20.0, 20.0))
        self.assertEqual(detection.label, 22)

    def test_non_animal_classes_are_still_rejected(self):
        # person(1), car(3), airplane(5) must never be cropped to, however
        # confident the detector is.
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [1, 1, 5, 5], [2, 2, 8, 8]],
            labels=[1, 3, 5],
            scores=[0.99, 0.98, 0.97],
        )
        self.assertIsNone(detector.detect_best_bird(self.IMG))

    def test_classes_argument_can_restrict_back_to_birds_only(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 20, 20]],
            labels=[COCO_BIRD_CLASS, 24],  # zebra scores higher but is excluded
            scores=[0.6, 0.95],
            classes={COCO_BIRD_CLASS},
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.label, COCO_BIRD_CLASS)
        self.assertEqual(detection.box, (0.0, 0.0, 10.0, 10.0))


class BoxGeometryTests(unittest.TestCase):
    def test_expand_adds_margin_and_clamps_to_image(self):
        # 100x100 box centered in a 1000x1000 image, 10% margin -> +10px each side.
        box = (400.0, 400.0, 500.0, 500.0)
        self.assertEqual(expand_and_clamp_box(box, 0.1, 1000, 1000), (390, 390, 510, 510))

    def test_expand_clamps_at_edges(self):
        box = (0.0, 0.0, 100.0, 100.0)
        # margin would push x1/y1 negative and is clamped to 0.
        x1, y1, x2, y2 = expand_and_clamp_box(box, 0.5, 200, 200)
        self.assertEqual((x1, y1), (0, 0))
        self.assertEqual((x2, y2), (150, 150))

    def test_crop_preserves_subrectangle_aspect(self):
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        crop = crop_to_box(image, (50, 20, 150, 120))
        self.assertEqual(crop.shape, (100, 100, 3))  # true sub-rectangle, no distortion

    def test_downscale_only_shrinks_and_preserves_ratio(self):
        image = np.zeros((500, 1000, 3), dtype=np.uint8)  # 1:2
        out = downscale_long_side(image, 400)
        self.assertEqual(out.shape[:2], (200, 400))  # long side capped, ratio kept
        # already-small image is returned unchanged (never upscaled)
        small = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(downscale_long_side(small, 400).shape[:2], (100, 100))


class CachePathTests(unittest.TestCase):
    def test_same_source_same_path_different_source_different_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = crop_cache_path(tmp, r"C:\photos\a.arw")
            a2 = crop_cache_path(tmp, r"C:\photos\a.arw")
            b = crop_cache_path(tmp, r"C:\photos\b.arw")
            self.assertEqual(a, a2)
            self.assertNotEqual(a, b)
            self.assertEqual(a.suffix, ".png")

    def test_params_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = CropParams(margin_frac=0.07, conf_threshold=0.4, max_side=800)
            write_crop_params(tmp, params)
            self.assertEqual(read_crop_params(tmp), params)


class LoaderCropCacheTests(unittest.TestCase):
    def test_loader_reads_crop_from_cache_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "crops"
            cache.mkdir()
            source = root / "img.png"
            # Full frame is solid black; the "crop" in the cache is solid white,
            # so we can prove the loader used the cache, not the source.
            cv2.imwrite(str(source), np.zeros((40, 40, 3), dtype=np.uint8))
            crop = np.full((20, 20, 3), 255, dtype=np.uint8)
            save_crop_png(crop_cache_path(cache, source), crop)

            loader = RawImageLoader(raw_root=str(root), output_size=(32, 32), crop_cache_dir=str(cache))
            image = loader.load_image(str(source))
            self.assertEqual(image.shape, (32, 32, 3))
            self.assertGreater(image.max(), 0.9)  # white crop, not the black frame

    def test_loader_falls_back_to_full_frame_when_crop_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "crops"
            cache.mkdir()
            source = root / "img.png"
            cv2.imwrite(str(source), np.full((40, 40, 3), 128, dtype=np.uint8))

            loader = RawImageLoader(raw_root=str(root), output_size=(32, 32), crop_cache_dir=str(cache))
            image = loader.load_image(str(source))  # no cache entry -> full frame
            self.assertEqual(image.shape, (32, 32, 3))
            self.assertTrue(np.allclose(image, 128 / 255.0, atol=0.02))

    def test_no_cache_dir_uses_full_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "img.png"
            cv2.imwrite(str(source), np.full((40, 40, 3), 200, dtype=np.uint8))
            loader = RawImageLoader(raw_root=str(root), output_size=(32, 32))
            image = loader.load_image(str(source))
            self.assertEqual(image.shape, (32, 32, 3))


if __name__ == "__main__":
    unittest.main()
