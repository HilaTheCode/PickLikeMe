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
    DEFAULT_AREA_TIE_FRAC,
    DOMESTIC_ANIMAL_CLASSES,
    SUPPORTED_ANIMAL_CLASSES,
    WILDLIFE_CLASSES,
    BirdDetection,
    BirdDetector,
    CropParams,
    box_area,
    build_crop,
    crop_cache_path,
    crop_to_box,
    downscale_long_side,
    expand_and_clamp_box,
    read_crop_params,
    save_crop_png,
    select_best_detection,
    write_crop_params,
)
from picklikeme.raw_io import RawImageLoader


def _detector_with_fake_model(
    boxes, labels, scores, conf_threshold=0.3, classes=None, area_tie_frac=DEFAULT_AREA_TIE_FRAC
):
    """A BirdDetector whose torchvision model is replaced by a fixed output, so
    the real selection logic (select_best_detection, via detect_best_bird /
    detect_with_all) can be tested without downloading or running the actual
    network."""
    detector = BirdDetector.__new__(BirdDetector)
    detector._torch = torch
    detector.device = "cpu"
    detector.conf_threshold = conf_threshold
    detector.classes = frozenset(SUPPORTED_ANIMAL_CLASSES if classes is None else classes)
    detector.area_tie_frac = area_tie_frac
    output = {
        "boxes": torch.tensor(boxes, dtype=torch.float),
        "labels": torch.tensor(labels),
        "scores": torch.tensor(scores),
    }
    detector.model = lambda images: [output]
    return detector


class SelectBestDetectionTests(unittest.TestCase):
    """The selection policy itself, tested directly and independent of the
    detector/model plumbing: area dominates confidence, which only breaks a
    tie between detections whose areas are already close."""

    def test_empty_candidates_returns_none(self):
        self.assertIsNone(select_best_detection([]))

    def test_a_single_candidate_is_returned_regardless_of_its_score(self):
        only = BirdDetection(box=(0, 0, 5, 5), score=0.31, label=COCO_BIRD_CLASS)
        self.assertIs(select_best_detection([only]), only)

    def test_the_largest_area_wins_when_areas_are_clearly_different(self):
        small = BirdDetection(box=(0, 0, 10, 10), score=0.99, label=COCO_BIRD_CLASS)  # area 100
        large = BirdDetection(box=(0, 0, 100, 100), score=0.31, label=COCO_BIRD_CLASS)  # area 10000
        self.assertIs(select_best_detection([small, large]), large)

    def test_confidence_breaks_a_tie_within_the_default_ten_percent_band(self):
        a = BirdDetection(box=(0, 0, 100, 100), score=0.4, label=COCO_BIRD_CLASS)  # area 10000
        b = BirdDetection(box=(0, 0, 95, 100), score=0.9, label=COCO_BIRD_CLASS)  # area 9500 (95%)
        self.assertIs(select_best_detection([a, b]), b)

    def test_a_detection_just_outside_the_tie_band_loses_despite_higher_confidence(self):
        a = BirdDetection(box=(0, 0, 100, 100), score=0.31, label=COCO_BIRD_CLASS)  # area 10000
        b = BirdDetection(box=(0, 0, 89, 100), score=0.99, label=COCO_BIRD_CLASS)  # area 8900 (89%)
        self.assertIs(select_best_detection([a, b]), a)

    def test_the_tie_boundary_is_inclusive_at_exactly_the_configured_fraction(self):
        a = BirdDetection(box=(0, 0, 10, 10), score=0.4, label=COCO_BIRD_CLASS)  # area 100
        b = BirdDetection(box=(0, 0, 9, 10), score=0.9, label=COCO_BIRD_CLASS)  # area 90, exactly 90%
        self.assertIs(select_best_detection([a, b]), b)

    def test_area_tie_frac_is_a_parameter_not_a_hardcoded_constant(self):
        a = BirdDetection(box=(0, 0, 100, 100), score=0.31, label=COCO_BIRD_CLASS)
        b = BirdDetection(box=(0, 0, 70, 100), score=0.99, label=COCO_BIRD_CLASS)  # 70% area
        # Default 10% band: b is well outside it, a wins on size alone.
        self.assertIs(select_best_detection([a, b]), a)
        # A wider 35% band pulls b into contention, where it wins on confidence.
        self.assertIs(select_best_detection([a, b], area_tie_frac=0.35), b)

    def test_the_tie_band_scales_with_size_not_a_fixed_pixel_margin(self):
        """The tolerance is a fraction of the largest area, so it applies the
        same way to a subject filling the frame and a tiny distant one."""
        big_a = BirdDetection(box=(0, 0, 1000, 1000), score=0.4, label=COCO_BIRD_CLASS)
        big_b = BirdDetection(box=(0, 0, 950, 1000), score=0.9, label=COCO_BIRD_CLASS)  # 95%
        self.assertIs(select_best_detection([big_a, big_b]), big_b)

        tiny_a = BirdDetection(box=(0, 0, 10, 10), score=0.4, label=COCO_BIRD_CLASS)
        tiny_b = BirdDetection(box=(0, 0, 9, 10), score=0.9, label=COCO_BIRD_CLASS)  # 90%
        self.assertIs(select_best_detection([tiny_a, tiny_b]), tiny_b)

    def test_degenerate_zero_area_boxes_fall_back_to_confidence(self):
        """A box with zero width or height should never come from a real
        detector, but must not crash or behave arbitrarily if it does."""
        a = BirdDetection(box=(5, 5, 5, 5), score=0.4, label=COCO_BIRD_CLASS)  # zero area
        b = BirdDetection(box=(5, 5, 5, 8), score=0.9, label=COCO_BIRD_CLASS)  # zero area (zero width)
        self.assertIs(select_best_detection([a, b]), b)

    def test_never_returns_a_detection_absent_from_the_input(self):
        candidates = [
            BirdDetection(box=(0, 0, 10, 10), score=s, label=COCO_BIRD_CLASS) for s in (0.31, 0.5, 0.99)
        ]
        self.assertIn(select_best_detection(candidates), candidates)


class BoxAreaTests(unittest.TestCase):
    def test_area_of_a_normal_box(self):
        self.assertEqual(box_area((0.0, 0.0, 10.0, 20.0)), 200.0)

    def test_a_malformed_box_never_yields_negative_area(self):
        self.assertEqual(box_area((10.0, 10.0, 5.0, 5.0)), 0.0)


class DetectBestBirdTests(unittest.TestCase):
    IMG = np.zeros((40, 40, 3), dtype=np.uint8)

    def test_picks_the_largest_animal_above_threshold(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 30, 30], [1, 1, 2, 2]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS, 1],  # two birds + a person
            scores=[0.99, 0.31, 0.99],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertIsInstance(detection, BirdDetection)
        # (5,5,30,30) has area 625 vs (0,0,10,10)'s 100: the larger bird wins
        # even though its own score (0.31) is far below the smaller bird's
        # (0.99), and even though the (excluded) person scores highest of all.
        self.assertEqual(detection.box, (5.0, 5.0, 30.0, 30.0))
        self.assertAlmostEqual(detection.score, 0.31, places=5)
        self.assertEqual(detection.label, COCO_BIRD_CLASS)

    def test_area_dominates_confidence_even_with_a_much_more_confident_smaller_box(self):
        """The design requirement the crop-selection redesign exists for: no
        confidence value may compensate for a much smaller box."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 100, 100], [0, 0, 15, 15]],  # areas 10000 vs 225
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.31, 0.99],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.box, (0.0, 0.0, 100.0, 100.0))

    def test_confidence_only_breaks_ties_between_similarly_sized_detections(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 100, 100], [0, 0, 96, 100]],  # areas 10000 vs 9600 (96%, inside the tie band)
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.4, 0.9],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.box, (0.0, 0.0, 96.0, 100.0))
        self.assertAlmostEqual(detection.score, 0.9, places=5)

    def test_detect_with_all_and_detect_best_bird_always_agree(self):
        """The two entry points must never disagree about the winner - both
        delegate to the same select_best_detection()."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 100, 100], [0, 0, 15, 15]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.31, 0.99],
        )
        winner_a = detector.detect_best_bird(self.IMG)
        winner_b, accepted = detector.detect_with_all(self.IMG)
        self.assertEqual(winner_a, winner_b)
        self.assertEqual(len(accepted), 2)

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

    def test_the_largest_animal_wins_regardless_of_class_or_confidence(self):
        # A bird with a tiny box and near-perfect confidence must lose to a
        # much larger elephant with a modest score: neither class nor
        # confidence breaks a size gap this wide.
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 90, 90]],
            labels=[COCO_BIRD_CLASS, 22],
            scores=[0.99, 0.35],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.box, (5.0, 5.0, 90.0, 90.0))
        self.assertEqual(detection.label, 22)

    def test_class_has_no_priority_when_areas_are_tied(self):
        # With the boxes near-equal in size, only confidence should decide -
        # and it must not favour one class over another.
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 100, 100], [0, 0, 96, 100]],  # 96%, inside the tie band
            labels=[COCO_BIRD_CLASS, 22],
            scores=[0.5, 0.85],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.label, 22)
        self.assertEqual(detection.box, (0.0, 0.0, 96.0, 100.0))

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


class BuildCropUsesAreaDominantSelectionTests(unittest.TestCase):
    """End to end through build_crop with the real BirdDetector (fake model
    output): the pixels actually cached must come from the larger detection,
    not the more confident one - closing the loop from selection policy to
    the crop training will see."""

    def test_the_cached_crop_comes_from_the_larger_detection(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:20, 10:20] = (255, 0, 0)     # small, high-confidence region
        frame[30:90, 30:90] = (0, 255, 0)     # large, lower-confidence region

        detector = _detector_with_fake_model(
            boxes=[[10, 10, 20, 20], [30, 30, 90, 90]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.99, 0.31],
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.0), collect_detections=True)

        self.assertEqual(result.detection.box, (30.0, 30.0, 90.0, 90.0))
        # The cached crop is the large green region, not the small red one.
        mean_pixel = result.crop.reshape(-1, 3).mean(axis=0)
        self.assertGreater(mean_pixel[1], mean_pixel[0], "crop should be the large green box, not the small red one")


if __name__ == "__main__":
    unittest.main()
