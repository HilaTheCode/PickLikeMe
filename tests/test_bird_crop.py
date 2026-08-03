import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.bird_crop import (
    CATALOGUED_CLASSES,
    COCO_BIRD_CLASS,
    COCO_PERSON_CLASS,
    CROP_CACHE_VERSION,
    DEFAULT_GROUP_SCENE_THRESHOLD,
    DEFAULT_MIN_CROP_CONFIDENCE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_JPEG_QUALITY,
    DETECTION_CATEGORIES,
    DETECTION_CATEGORY_BIRD,
    DETECTION_CATEGORY_HUMAN,
    DETECTION_CATEGORY_MAMMAL,
    DOMESTIC_ANIMAL_CLASSES,
    IMAGE_FORMAT_EXTENSIONS,
    SUPPORTED_ANIMAL_CLASSES,
    WILDLIFE_CLASSES,
    BirdDetection,
    BirdDetector,
    CropParams,
    box_area,
    build_crop,
    crop_cache_path,
    crop_to_box,
    detection_category,
    downscale_long_side,
    enclosing_box,
    expand_and_clamp_box,
    read_crop_params,
    save_crop_png,
    select_best_detection,
    write_crop_params,
)
from picklikeme.raw_io import RawImageLoader


def _detector_with_fake_model(
    boxes,
    labels,
    scores,
    conf_threshold=0.3,
    classes=None,
    catalogue_classes=None,
    min_crop_confidence=DEFAULT_MIN_CROP_CONFIDENCE,
    group_scene_threshold=DEFAULT_GROUP_SCENE_THRESHOLD,
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
    detector.catalogue_classes = frozenset(
        (CATALOGUED_CLASSES if classes is None else detector.classes)
        if catalogue_classes is None
        else catalogue_classes
    )
    detector.min_crop_confidence = min_crop_confidence
    detector.group_scene_threshold = group_scene_threshold
    output = {
        "boxes": torch.tensor(boxes, dtype=torch.float),
        "labels": torch.tensor(labels),
        "scores": torch.tensor(scores),
    }
    detector.model = lambda images: [output]
    return detector


class SelectBestDetectionTests(unittest.TestCase):
    """The selection policy itself, tested directly and independent of the
    detector/model plumbing: v7 policy (see bird_crop.py's module docstring)
    - an absolute confidence floor (min_crop_confidence) discards unreliable
    candidates outright; area then decides among whatever survives. Neither
    a pure area-first policy (v3-v5) nor a pure confidence-first one (v6)
    survived contact with real data - see the EyePose Investigation Phase 1
    report's "Detection selection policy" discussion for the algebraic proof
    that a linear weighted score can't satisfy both failure modes either."""

    def test_empty_candidates_returns_none(self):
        self.assertIsNone(select_best_detection([]))

    def test_a_single_candidate_above_the_floor_is_returned_regardless_of_score(self):
        only = BirdDetection(box=(0, 0, 5, 5), score=0.61, label=COCO_BIRD_CLASS)
        self.assertIs(select_best_detection([only]), only)

    def test_a_single_candidate_below_the_floor_is_rejected_not_returned(self):
        """No candidate to fall back to - this is a deliberate "no reliable
        subject" outcome (build_crop's full-frame fallback takes over), not
        a bug - see the module docstring's "v7" entry."""
        only = BirdDetection(box=(0, 0, 5, 5), score=0.31, label=COCO_BIRD_CLASS)
        self.assertIsNone(select_best_detection([only], min_crop_confidence=0.6))

    def test_an_unreliable_detection_never_wins_however_large_its_box(self):
        unreliable = BirdDetection(box=(0, 0, 1000, 1000), score=0.31, label=COCO_BIRD_CLASS)  # huge, low confidence
        reliable = BirdDetection(box=(0, 0, 10, 10), score=0.99, label=COCO_BIRD_CLASS)  # tiny, high confidence
        self.assertIs(select_best_detection([unreliable, reliable], min_crop_confidence=0.6), reliable)

    def test_the_largest_of_several_reliable_candidates_wins(self):
        """Once every candidate has cleared the reliability floor, area (not
        confidence) decides - restoring v3's original intent among
        genuinely comparable candidates."""
        small_but_more_confident = BirdDetection(box=(0, 0, 10, 10), score=0.95, label=COCO_BIRD_CLASS)
        large_and_reliable = BirdDetection(box=(0, 0, 100, 100), score=0.75, label=COCO_BIRD_CLASS)
        self.assertIs(
            select_best_detection([small_but_more_confident, large_and_reliable], min_crop_confidence=0.6),
            large_and_reliable,
        )

    def test_the_floor_is_an_absolute_value_not_relative_to_the_winner(self):
        """The historical bug this policy replaces (v6): a relative tie band
        could never protect a moderately-confident real subject from a very
        confident small distractor, because the distractor's own confidence
        set the band. An absolute floor has no such relationship - both
        clearing 0.6 is what matters, not their ratio to each other."""
        intended_subject = BirdDetection(box=(0, 0, 100, 100), score=0.75, label=COCO_BIRD_CLASS)  # large, real
        small_distractor = BirdDetection(box=(0, 0, 5, 5), score=0.99, label=COCO_BIRD_CLASS)  # tiny, very confident
        self.assertIs(
            select_best_detection([intended_subject, small_distractor], min_crop_confidence=0.6),
            intended_subject,
        )

    def test_the_floor_boundary_is_inclusive(self):
        exactly_at_floor = BirdDetection(box=(0, 0, 100, 100), score=0.6, label=COCO_BIRD_CLASS)
        self.assertIs(select_best_detection([exactly_at_floor], min_crop_confidence=0.6), exactly_at_floor)

    def test_min_crop_confidence_is_a_parameter_not_a_hardcoded_constant(self):
        a = BirdDetection(box=(0, 0, 10, 10), score=0.5, label=COCO_BIRD_CLASS)
        # At the default floor (0.6), 0.5 does not clear it.
        self.assertIsNone(select_best_detection([a]))
        # Lowering the floor admits it.
        self.assertIs(select_best_detection([a], min_crop_confidence=0.4), a)

    def test_never_returns_a_detection_absent_from_the_input(self):
        candidates = [
            BirdDetection(box=(0, 0, 10, 10), score=s, label=COCO_BIRD_CLASS) for s in (0.61, 0.75, 0.99)
        ]
        self.assertIn(select_best_detection(candidates), candidates)

    def test_a_real_false_positive_no_longer_beats_a_confident_true_positive(self):
        """The exact failure mode found during the EyePose investigation
        (docs/EyePose_Investigation_Phase_1.md's Q1, image DSC03129): a
        0.998-confidence bird lost the crop to an unrelated, much larger,
        0.458-confidence false positive (originally mislabelled "cow"). At
        the default 0.6 floor, the false positive never even reaches the
        area comparison."""
        bird = BirdDetection(box=(2158.2, 929.6, 3071.4, 2090.5), score=0.998, label=COCO_BIRD_CLASS)
        false_positive = BirdDetection(box=(18.2, 0.0, 2329.1, 3642.3), score=0.458, label=21)  # cow
        self.assertIs(select_best_detection([bird, false_positive]), bird)

    def test_reconstructed_historical_failure_mode_also_resolves_correctly(self):
        """A plausible reconstruction (not measured data - flagged as such
        in the report) of the failure v3 was originally built to fix, and
        v6 reopened: a legitimately-real, moderately-confident intended
        subject competing against a small, very-confident background bird.
        Both clear the default 0.6 floor, so area (not confidence) decides,
        and the intended subject - the larger of the two - wins."""
        intended_subject = BirdDetection(box=(0, 0, 300, 300), score=0.75, label=COCO_BIRD_CLASS)
        background_bird = BirdDetection(box=(0, 0, 20, 20), score=0.99, label=COCO_BIRD_CLASS)
        self.assertIs(select_best_detection([intended_subject, background_bird]), intended_subject)


def _flock(count: int, start_score: float = 0.5) -> list[BirdDetection]:
    """`count` small, non-overlapping detections spread out in a row, each a
    little more confident than the last - stand-ins for a flock/herd/colony."""
    return [
        BirdDetection(box=(i * 20.0, 0.0, i * 20.0 + 10.0, 10.0), score=start_score + i * 0.01, label=COCO_BIRD_CLASS)
        for i in range(count)
    ]


class GroupSceneSelectionTests(unittest.TestCase):
    """select_best_detection()'s other policy: at or above group_scene_threshold
    surviving detections, no individual is selected - the target becomes the
    box enclosing the whole group. Intentional for wildlife photography, where
    a flock/herd/colony is often the actual subject, not any one animal in it."""

    def test_fewer_than_the_threshold_uses_the_normal_largest_box_policy(self):
        candidates = _flock(9)  # scores 0.50-0.58 - below min_crop_confidence's own default (0.6)
        # min_crop_confidence=0.0: this test is about the group-scene boundary,
        # not the reliability floor - isolate the one behaviour under test.
        winner = select_best_detection(candidates, min_crop_confidence=0.0, group_scene_threshold=10)
        self.assertIn(winner, candidates, "below threshold, the winner must be one real detection")
        self.assertNotEqual(winner.box, enclosing_box([c.box for c in candidates]))

    def test_exactly_the_threshold_is_a_group_scene(self):
        candidates = _flock(10)
        winner = select_best_detection(candidates, group_scene_threshold=10)
        self.assertNotIn(winner, candidates, "a group scene's box must not be any single detection")
        self.assertEqual(winner.box, enclosing_box([c.box for c in candidates]))

    def test_more_than_the_threshold_is_a_group_scene(self):
        candidates = _flock(25)
        winner = select_best_detection(candidates, group_scene_threshold=10)
        self.assertEqual(winner.box, enclosing_box([c.box for c in candidates]))

    def test_the_threshold_is_configurable(self):
        candidates = _flock(4)  # scores 0.50-0.53 - below min_crop_confidence's own default (0.6)
        # Below a threshold of 5, normal per-detection selection applies...
        # min_crop_confidence=0.0 isolates the group-scene boundary under
        # test from the (unrelated) reliability floor - see the sibling test
        # above.
        normal = select_best_detection(candidates, min_crop_confidence=0.0, group_scene_threshold=5)
        self.assertIn(normal, candidates)
        # ...but the same 4 detections are a group scene once the threshold is lowered to 4.
        group = select_best_detection(candidates, group_scene_threshold=4)
        self.assertEqual(group.box, enclosing_box([c.box for c in candidates]))

    def test_the_group_box_encloses_every_valid_detection_with_mixed_sizes(self):
        """Mixed box sizes and positions - not a neat row - so the enclosing
        box genuinely has to take the min/max of all four edges, not just
        assume the members are laid out predictably."""
        candidates = [
            BirdDetection(box=(100.0, 200.0, 140.0, 260.0), score=0.5, label=COCO_BIRD_CLASS),  # far top-left-ish
            BirdDetection(box=(300.0, 50.0, 305.0, 55.0), score=0.6, label=COCO_BIRD_CLASS),  # tiny, high up
            BirdDetection(box=(10.0, 400.0, 500.0, 420.0), score=0.4, label=COCO_BIRD_CLASS),  # wide, low
            BirdDetection(box=(250.0, 150.0, 260.0, 170.0), score=0.9, label=COCO_BIRD_CLASS),  # small, central
            BirdDetection(box=(480.0, 480.0, 520.0, 520.0), score=0.7, label=COCO_BIRD_CLASS),  # far bottom-right
            *(_flock(5, start_score=0.3))  # pad up to the default threshold of 10
        ]
        self.assertGreaterEqual(len(candidates), 10)
        winner = select_best_detection(candidates, group_scene_threshold=10)

        expected = enclosing_box([c.box for c in candidates])
        self.assertEqual(winner.box, expected)
        # Every member's box must fit entirely inside the group box.
        for member in candidates:
            x1, y1, x2, y2 = member.box
            gx1, gy1, gx2, gy2 = winner.box
            self.assertGreaterEqual(x1, gx1)
            self.assertGreaterEqual(y1, gy1)
            self.assertLessEqual(x2, gx2)
            self.assertLessEqual(y2, gy2)

    def test_group_box_score_and_label_are_the_most_confident_members(self):
        """score/label are informational only (crop geometry uses the box),
        but they should still mean something rather than being arbitrary."""
        candidates = _flock(10)
        winner = select_best_detection(candidates, group_scene_threshold=10)
        most_confident = max(candidates, key=lambda d: d.score)
        self.assertEqual(winner.score, most_confident.score)
        self.assertEqual(winner.label, most_confident.label)

    def test_group_scene_ignores_min_crop_confidence(self):
        """The reliability gate is a below-threshold concept only - it must
        not leak into (or change) group-scene selection, even when set so
        strict that no individual candidate would ever pass it."""
        candidates = _flock(10)
        permissive = select_best_detection(candidates, min_crop_confidence=0.0, group_scene_threshold=10)
        strict = select_best_detection(candidates, min_crop_confidence=1.0, group_scene_threshold=10)
        expected = enclosing_box([c.box for c in candidates])
        self.assertEqual(permissive.box, expected)
        self.assertEqual(strict.box, expected)


class EnclosingBoxTests(unittest.TestCase):
    def test_a_single_box_encloses_itself(self):
        box = (1.0, 2.0, 3.0, 4.0)
        self.assertEqual(enclosing_box([box]), box)

    def test_encloses_boxes_scattered_in_every_direction(self):
        boxes = [
            (10.0, 10.0, 20.0, 20.0),
            (0.0, 15.0, 5.0, 18.0),   # extends the left edge
            (18.0, 0.0, 22.0, 5.0),   # extends the top edge and right edge
            (5.0, 25.0, 12.0, 30.0),  # extends the bottom edge
        ]
        self.assertEqual(enclosing_box(boxes), (0.0, 0.0, 22.0, 30.0))

    def test_a_box_fully_inside_another_does_not_shrink_the_result(self):
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (40.0, 40.0, 60.0, 60.0)
        self.assertEqual(enclosing_box([outer, inner]), outer)


class BoxAreaTests(unittest.TestCase):
    def test_area_of_a_normal_box(self):
        self.assertEqual(box_area((0.0, 0.0, 10.0, 20.0)), 200.0)

    def test_a_malformed_box_never_yields_negative_area(self):
        self.assertEqual(box_area((10.0, 10.0, 5.0, 5.0)), 0.0)


class DetectBestBirdTests(unittest.TestCase):
    IMG = np.zeros((40, 40, 3), dtype=np.uint8)

    def test_picks_the_most_confident_animal_above_threshold(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 30, 30], [1, 1, 2, 2]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS, 1],  # two birds + a person
            scores=[0.99, 0.31, 0.995],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertIsInstance(detection, BirdDetection)
        # The smaller (0,0,10,10) bird wins: its own confidence (0.99) clears
        # min_crop_confidence's default floor (0.6), the larger (5,5,30,30)
        # one (0.31) does not, and the person is never crop-eligible at all
        # regardless of its own (higher still) score.
        self.assertEqual(detection.box, (0.0, 0.0, 10.0, 10.0))
        self.assertAlmostEqual(detection.score, 0.99, places=5)
        self.assertEqual(detection.label, COCO_BIRD_CLASS)

    def test_confidence_dominates_area_even_with_a_much_larger_less_confident_box(self):
        """The design requirement the v7 selection policy exists for (see
        docs/EyePose_Investigation_Phase_1.md's Q1 and "Detection selection
        policy" discussion): a detection below the reliability floor may
        never win, however much larger its box is."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 100, 100], [0, 0, 15, 15]],  # areas 10000 vs 225
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.31, 0.99],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.box, (0.0, 0.0, 15.0, 15.0))

    def test_area_decides_once_both_detections_clear_the_reliability_floor(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 100, 100], [0, 0, 96, 100]],  # areas 10000 vs 9600
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.9, 0.82],  # both clear min_crop_confidence's default floor (0.6)
        )
        detection = detector.detect_best_bird(self.IMG)
        # Both are reliable, so the LARGER box (10000) wins, even though its
        # own confidence (0.9) is not the higher of the two.
        self.assertEqual(detection.box, (0.0, 0.0, 100.0, 100.0))
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

    def test_a_real_model_output_with_a_flock_becomes_a_group_scene(self):
        """End to end through the real filtering (class + confidence), not
        just select_best_detection() in isolation: a raw model output with
        ten accepted birds plus an excluded person must still correctly
        count only the ten toward the group threshold."""
        boxes = [[i * 15, 0, i * 15 + 8, 8] for i in range(10)] + [[500, 500, 520, 520]]
        labels = [COCO_BIRD_CLASS] * 10 + [1]  # ten birds + a person (excluded by class)
        scores = [0.5 + i * 0.01 for i in range(10)] + [0.99]
        detector = _detector_with_fake_model(boxes=boxes, labels=labels, scores=scores)

        detection = detector.detect_best_bird(self.IMG)
        expected = enclosing_box([tuple(float(v) for v in b) for b in boxes[:10]])
        self.assertEqual(detection.box, expected)


class CatalogueClassesTests(unittest.TestCase):
    """A person is catalogued (recorded, exposed as metadata) but must never
    be a crop TARGET - the two are different questions BirdDetector answers
    from the same forward pass (see catalogue_classes vs classes)."""

    IMG = np.zeros((40, 40, 3), dtype=np.uint8)

    def test_a_person_is_recorded_in_all_detections_but_never_wins(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [0, 0, 39, 39]],  # small bird, huge person
            labels=[COCO_BIRD_CLASS, COCO_PERSON_CLASS],
            scores=[0.7, 0.99],  # bird above min_crop_confidence's own default (0.6)
        )
        winner, catalogued = detector.detect_with_all(self.IMG)

        self.assertEqual(winner.label, COCO_BIRD_CLASS, "the person must never win the crop, however large/confident")
        self.assertEqual(len(catalogued), 2, "but both are still catalogued")
        self.assertIn(COCO_PERSON_CLASS, [d.label for d in catalogued])

    def test_catalogue_classes_defaults_to_a_superset_of_classes(self):
        detector = _detector_with_fake_model(boxes=[], labels=[], scores=[])
        self.assertTrue(detector.classes <= detector.catalogue_classes)
        self.assertIn(COCO_PERSON_CLASS, detector.catalogue_classes)
        self.assertNotIn(COCO_PERSON_CLASS, detector.classes, "a person is never a crop target by default")

    def test_restricting_classes_also_narrows_the_default_catalogue(self):
        """Passing a restricted `classes` (e.g. WILDLIFE_CLASSES) without an
        explicit catalogue_classes must not silently catalogue the full
        default set - the caller asked for a narrower detector."""
        detector = _detector_with_fake_model(
            boxes=[], labels=[], scores=[], classes=WILDLIFE_CLASSES
        )
        self.assertEqual(detector.catalogue_classes, frozenset(WILDLIFE_CLASSES))

    def test_an_explicit_catalogue_classes_is_honoured_even_when_classes_is_restricted(self):
        detector = _detector_with_fake_model(
            boxes=[], labels=[], scores=[], classes={COCO_BIRD_CLASS}, catalogue_classes=CATALOGUED_CLASSES
        )
        self.assertEqual(detector.catalogue_classes, frozenset(CATALOGUED_CLASSES))

    def test_only_a_person_present_yields_no_crop_winner(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 39, 39]], labels=[COCO_PERSON_CLASS], scores=[0.99]
        )
        self.assertIsNone(detector.detect_best_bird(self.IMG))
        _, catalogued = detector.detect_with_all(self.IMG)
        self.assertEqual(len(catalogued), 1, "still catalogued, even with no crop winner at all")


class DetectionCategoryTests(unittest.TestCase):
    """The review app's own taxonomy - broader than any one detector's class
    list, so a future detector can populate categories this one cannot."""

    def test_bird_maps_to_the_bird_category(self):
        self.assertEqual(detection_category(COCO_BIRD_CLASS), DETECTION_CATEGORY_BIRD)

    def test_person_maps_to_the_human_category(self):
        self.assertEqual(detection_category(COCO_PERSON_CLASS), DETECTION_CATEGORY_HUMAN)

    def test_every_supported_animal_class_except_bird_maps_to_mammal(self):
        """COCO simply has no reptile/amphibian/fish/insect/arachnid class -
        every other animal it recognizes (cat..giraffe) is a mammal."""
        for class_id in SUPPORTED_ANIMAL_CLASSES:
            if class_id == COCO_BIRD_CLASS:
                continue
            self.assertEqual(detection_category(class_id), DETECTION_CATEGORY_MAMMAL)

    def test_an_uncatalogued_class_has_no_category(self):
        self.assertIsNone(detection_category(999))

    def test_every_catalogued_class_has_a_valid_category(self):
        for class_id in CATALOGUED_CLASSES:
            self.assertIn(detection_category(class_id), DETECTION_CATEGORIES)

    def test_the_taxonomy_has_a_slot_for_categories_no_current_model_can_reach(self):
        """Reptile/amphibian/fish/insect/arachnid are real, named categories
        in the taxonomy even though COCO cannot populate them today - the
        whole point of designing this as an extensible vocabulary rather than
        a raw COCO class list."""
        reachable_today = {detection_category(class_id) for class_id in CATALOGUED_CLASSES}
        for category in ("reptile", "amphibian", "fish", "insect", "arachnid"):
            self.assertIn(category, DETECTION_CATEGORIES)
            self.assertNotIn(category, reachable_today)


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

    def test_the_most_confident_animal_wins_regardless_of_class_or_area(self):
        # A bird with a tiny box and near-perfect confidence must beat a much
        # larger elephant whose own confidence (0.35) never even clears the
        # reliability floor: neither class nor box size can compensate for that.
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 90, 90]],
            labels=[COCO_BIRD_CLASS, 22],
            scores=[0.99, 0.35],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.box, (0.0, 0.0, 10.0, 10.0))
        self.assertEqual(detection.label, COCO_BIRD_CLASS)

    def test_class_has_no_priority_once_both_detections_clear_the_reliability_floor(self):
        # With both confidences clearing min_crop_confidence's default floor
        # (0.6), area decides - and it must not favour one class over another.
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 100, 100], [0, 0, 96, 100]],  # areas 10000 vs 9600
            labels=[COCO_BIRD_CLASS, 22],
            scores=[0.9, 0.82],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.label, COCO_BIRD_CLASS)
        self.assertEqual(detection.box, (0.0, 0.0, 100.0, 100.0))

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
            self.assertEqual(a.suffix, ".jpg")  # DEFAULT_IMAGE_FORMAT="jpeg" - see bird_crop.CropParams

    def test_params_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = CropParams(
                margin_frac=0.07, conf_threshold=0.4, max_side=800,
                min_crop_confidence=0.2, group_scene_threshold=6,
            )
            write_crop_params(tmp, params)
            reloaded = read_crop_params(tmp)
            self.assertEqual(reloaded, params)
            self.assertEqual(reloaded.group_scene_threshold, 6)

    def test_cache_path_extension_follows_image_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            jpeg_path = crop_cache_path(tmp, r"C:\photos\a.arw", image_format="jpeg")
            png_path = crop_cache_path(tmp, r"C:\photos\a.arw", image_format="png")
            self.assertEqual(jpeg_path.suffix, ".jpg")
            self.assertEqual(png_path.suffix, ".png")
            # Same digest (same stem) regardless of format - only the
            # extension differs, so the two never collide on disk.
            self.assertEqual(jpeg_path.stem, png_path.stem)
            self.assertNotEqual(jpeg_path, png_path)

    def test_an_unknown_format_falls_back_to_the_default_rather_than_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = crop_cache_path(tmp, r"C:\photos\a.arw", image_format="tiff")
            self.assertEqual(path.suffix, IMAGE_FORMAT_EXTENSIONS[DEFAULT_IMAGE_FORMAT])


# ---------------------------------------------------------------------------
# Vision Cache - configurable resolution, format and quality (see bird_crop.
# py's module docstring "Vision Cache infrastructure" section, and the
# audit that motivated it: a fixed max_side=1024 PNG cap was silently
# discarding real detail before it ever reached a Computer Vision model or
# the sharpness metrics that read the cache directly).
# ---------------------------------------------------------------------------


class ConfigurableResolutionTests(unittest.TestCase):
    def test_the_default_is_unlimited_not_1024(self):
        """Pins the actual default, so a future accidental revert back to a
        hardcoded cap is caught immediately."""
        self.assertIsNone(CropParams().max_side)

    def test_max_side_none_never_downscales_however_large(self):
        big = np.zeros((6000, 8000, 3), dtype=np.uint8)
        out = downscale_long_side(big, None)
        self.assertEqual(out.shape[:2], (6000, 8000))
        self.assertIs(out, big, "no copy should be made when nothing changes")

    def test_an_explicit_max_side_still_caps_as_before(self):
        """The capping behaviour itself is unchanged and still available -
        only the DEFAULT changed, for a machine/use case that genuinely
        wants to trade detail for disk space."""
        image = np.zeros((500, 1000, 3), dtype=np.uint8)
        out = downscale_long_side(image, 400)
        self.assertEqual(out.shape[:2], (200, 400))

    def test_build_crop_respects_an_unlimited_max_side_end_to_end(self):
        frame = np.zeros((3000, 4000, 3), dtype=np.uint8)
        detector = _detector_with_fake_model(
            boxes=[[100.0, 100.0, 3900.0, 2900.0]], scores=[0.95], labels=[COCO_BIRD_CLASS],
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.0, max_side=None))
        # The crop is large (a big chunk of a 4000x3000 frame) and must come
        # back completely uncapped - this is exactly the case that used to
        # lose real detail to the old hardcoded 1024px cap.
        self.assertGreater(max(result.crop.shape[:2]), 1024)


class ConfigurableFormatAndQualityTests(unittest.TestCase):
    def _rich_crop(self, size: int = 64) -> np.ndarray:
        """Photo-like structured detail (smooth gradients plus a few sharp
        edges), not a flat color and not pure noise. A flat color makes
        quality differences invisible (compresses to nearly nothing at any
        setting); pure random noise is the opposite failure - real photos
        have local spatial correlation JPEG's DCT exploits well, so testing
        against noise (which has none) would make even a high quality look
        artificially lossy. This lands in between, like real image content.
        """
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:size, 0:size]
        gradient = (xx * (255 / size)).astype(np.int16)
        image = np.stack([gradient, gradient[::-1, :], np.full_like(gradient, 128)], axis=-1)
        image[size // 4 : size // 2, size // 4 : size // 2] = (220, 40, 40)  # a sharp-edged block
        image = image + rng.integers(-3, 4, image.shape, dtype=np.int16)
        return np.clip(image, 0, 255).astype(np.uint8)

    def test_default_format_and_quality(self):
        self.assertEqual(CropParams().image_format, DEFAULT_IMAGE_FORMAT)
        self.assertEqual(CropParams().jpeg_quality, DEFAULT_JPEG_QUALITY)

    def test_a_dot_jpg_path_is_written_as_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jpg"
            save_crop_png(path, self._rich_crop())
            # A real JPEG file, not a PNG saved under a misleading extension.
            self.assertEqual(cv2.imread(str(path)).shape[:2], (64, 64))
            header = path.read_bytes()[:3]
            self.assertEqual(header, b"\xff\xd8\xff", "not a JPEG file signature")

    def test_a_dot_png_path_is_written_as_lossless_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.png"
            crop = self._rich_crop()
            save_crop_png(path, crop)
            header = path.read_bytes()[:8]
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n", "not a PNG file signature")
            # Lossless: reading it back must reproduce the source exactly.
            roundtrip = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            self.assertTrue(np.array_equal(roundtrip, crop))

    def test_jpeg_quality_is_configurable_and_affects_file_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            crop = self._rich_crop()
            low = Path(tmp) / "low.jpg"
            high = Path(tmp) / "high.jpg"
            save_crop_png(low, crop, jpeg_quality=10)
            save_crop_png(high, crop, jpeg_quality=100)
            self.assertLess(
                low.stat().st_size, high.stat().st_size,
                "a lower JPEG quality must produce a smaller file - the parameter is not being applied",
            )

    def test_the_default_quality_98_is_close_to_lossless(self):
        """Not a hard numeric accuracy budget (JPEG's exact error depends on
        content), just the sanity check behind choosing 98 as the default -
        see docs/vision_cache.md: near-lossless in practice, not merely
        'better than a very low quality'."""
        with tempfile.TemporaryDirectory() as tmp:
            crop = self._rich_crop()
            path = Path(tmp) / "a.jpg"
            save_crop_png(path, crop, jpeg_quality=DEFAULT_JPEG_QUALITY)
            roundtrip = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            mean_abs_error = np.abs(roundtrip.astype(int) - crop.astype(int)).mean()
            # A tiny (64px) test crop concentrates a sharp synthetic edge
            # into a large fraction of its own 8x8 DCT blocks - a real,
            # much larger photo has proportionally far less edge area, so
            # this loose bound (out of a possible 255) is deliberately
            # generous, not a tight accuracy budget.
            self.assertLess(mean_abs_error, 4.0, "q98 should be visually near-lossless on real content")


class CacheVersioningParticipationTests(unittest.TestCase):
    """The mechanism `build_cache` already uses to refuse a mismatched cache
    (see test_preprocess_pipeline.py's CacheVersionMismatchTests) is just
    CropParams equality - so image_format/jpeg_quality/max_side automatically
    participate in it the moment they exist as fields, with no separate
    version-tracking code. These tests pin exactly that participation."""

    def test_a_different_jpeg_quality_is_not_equal_to_the_default(self):
        self.assertNotEqual(CropParams(), CropParams(jpeg_quality=50))

    def test_a_different_image_format_is_not_equal_to_the_default(self):
        self.assertNotEqual(CropParams(), CropParams(image_format="png"))

    def test_a_different_max_side_is_not_equal_to_the_default(self):
        self.assertNotEqual(CropParams(), CropParams(max_side=1024))

    def test_current_version_constant_matches_the_dataclass_default(self):
        self.assertEqual(CROP_CACHE_VERSION, CropParams().version)


class OldFormatCacheIsOrphanedNotMisreadTests(unittest.TestCase):
    """Backward compatibility: an old (pre-Vision-Cache) cache entry must
    never be silently treated as a valid new one - see bird_crop.py's module
    docstring. Rather than an explicit staleness check, the file EXTENSION
    itself already guarantees this: the new default format's path simply
    does not exist yet for anything only ever built under the old one."""

    def test_a_legacy_png_entry_is_invisible_to_the_new_default_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = r"C:\photos\legacy.arw"
            legacy_path = crop_cache_path(tmp, source, image_format="png")
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            save_crop_png(legacy_path, np.zeros((10, 10, 3), dtype=np.uint8))
            self.assertTrue(legacy_path.exists())

            # A caller using today's defaults (no image_format override)
            # looks for a *different* path and finds nothing - it will
            # rebuild fresh rather than reading the old, lower-quality file.
            current_path = crop_cache_path(tmp, source)
            self.assertNotEqual(current_path, legacy_path)
            self.assertFalse(current_path.exists())


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


class BuildCropRejectsUnreliableDetectionsTests(unittest.TestCase):
    """End to end through build_crop with the real BirdDetector (fake model
    output): the pixels actually cached must come from the detection that
    clears the reliability floor, not the larger-but-unreliable one - closing
    the loop from selection policy to the crop training/EyePose will see. See
    docs/EyePose_Investigation_Phase_1.md's Q1 for the real failure mode
    (a large, low-confidence false positive winning the crop) this v7 policy
    fixes."""

    def test_the_cached_crop_comes_from_the_reliable_detection(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:20, 10:20] = (255, 0, 0)     # small, reliable region
        frame[30:90, 30:90] = (0, 255, 0)     # large, unreliable region

        detector = _detector_with_fake_model(
            boxes=[[10, 10, 20, 20], [30, 30, 90, 90]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.99, 0.31],
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.0), collect_detections=True)

        self.assertEqual(result.detection.box, (10.0, 10.0, 20.0, 20.0))
        # The cached crop is the small red region, not the large green one.
        mean_pixel = result.crop.reshape(-1, 3).mean(axis=0)
        self.assertGreater(mean_pixel[0], mean_pixel[1], "crop should be the small red box, not the large green one")


class BuildCropHandlesGroupScenesTests(unittest.TestCase):
    """End to end through build_crop for a group scene: the crop must be a
    tight region enclosing the whole group, never the full-frame fallback -
    the "even if the group is small, do not preserve unnecessary background"
    requirement."""

    def test_a_small_flock_in_a_large_frame_gets_a_tight_group_crop_not_the_full_frame(self):
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # Ten small detections clustered in one corner - tiny relative to the frame.
        boxes = [[50 + i * 20, 50 + i * 5, 50 + i * 20 + 15, 50 + i * 5 + 15] for i in range(10)]
        detector = _detector_with_fake_model(
            boxes=boxes,
            labels=[COCO_BIRD_CLASS] * 10,
            scores=[0.5 + i * 0.01 for i in range(10)],
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.05), collect_detections=True)

        self.assertIsNotNone(result.detection, "a group is still a real subject - never the full-frame fallback")
        self.assertIsNotNone(result.expanded_box)

        frame_area = frame.shape[0] * frame.shape[1]
        crop_area = result.crop.shape[0] * result.crop.shape[1]
        self.assertLess(
            crop_area / frame_area, 0.05,
            "the crop must stay tight around the small group, not balloon toward the full frame",
        )

    def test_the_margin_is_applied_to_the_group_box_like_any_other_crop(self):
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        boxes = [[100 + i * 10, 100, 100 + i * 10 + 8, 108] for i in range(10)]
        detector = _detector_with_fake_model(
            boxes=boxes, labels=[COCO_BIRD_CLASS] * 10, scores=[0.5 + i * 0.01 for i in range(10)]
        )
        no_margin = build_crop(frame, detector, CropParams(margin_frac=0.0), collect_detections=True)
        with_margin = build_crop(frame, detector, CropParams(margin_frac=0.2), collect_detections=True)

        no_margin_area = (no_margin.expanded_box[2] - no_margin.expanded_box[0]) * (
            no_margin.expanded_box[3] - no_margin.expanded_box[1]
        )
        with_margin_area = (with_margin.expanded_box[2] - with_margin.expanded_box[0]) * (
            with_margin.expanded_box[3] - with_margin.expanded_box[1]
        )
        self.assertGreater(with_margin_area, no_margin_area)

    def test_every_detection_falls_within_the_final_expanded_crop(self):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        boxes = [[20 + i * 25, 30 + (i % 3) * 40, 20 + i * 25 + 12, 30 + (i % 3) * 40 + 12] for i in range(12)]
        detector = _detector_with_fake_model(
            boxes=boxes, labels=[COCO_BIRD_CLASS] * 12, scores=[0.4 + i * 0.01 for i in range(12)]
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.05), collect_detections=True)
        gx1, gy1, gx2, gy2 = result.expanded_box
        for x1, y1, x2, y2 in boxes:
            self.assertGreaterEqual(x1, gx1)
            self.assertGreaterEqual(y1, gy1)
            self.assertLessEqual(x2, gx2)
            self.assertLessEqual(y2, gy2)


if __name__ == "__main__":
    unittest.main()
