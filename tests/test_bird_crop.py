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
    center_proximity,
    crop_cache_path,
    crop_to_box,
    detection_category,
    detections_cache_path,
    downscale_long_side,
    enclosing_box,
    expand_and_clamp_box,
    read_crop_params,
    relative_box_area,
    relative_size_score,
    save_crop_png,
    select_best_detection,
    selection_score,
    subject_size_score,
    write_crop_params,
)
from picklikeme.raw_io import RawImageLoader


def _detector_with_fake_model(
    boxes,
    labels,
    scores,
    conf_threshold=0.3,
    min_crop_confidence=DEFAULT_MIN_CROP_CONFIDENCE,
    group_scene_threshold=DEFAULT_GROUP_SCENE_THRESHOLD,
):
    """A BirdDetector whose torchvision model is replaced by a fixed output, so
    the real selection logic (select_best_detection, via detect_best_bird /
    detect_with_all) can be tested without downloading or running the actual
    network.

    There is deliberately no `classes`/`catalogue_classes` knob to set up:
    v9 removed both, and this helper mirroring the real constructor is what
    keeps a test from configuring a class gate that production no longer has.
    """
    detector = BirdDetector.__new__(BirdDetector)
    detector._torch = torch
    detector.device = "cpu"
    detector.conf_threshold = conf_threshold
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
    """The individual-selection policy itself, tested directly and
    independent of the detector/model plumbing: v9 (see bird_crop.py's module
    docstring) - every candidate is scored on a fixed

        0.50 * centre proximity + 0.30 * size score + 0.20 * confidence

    and the highest score wins. The size score is `subject_size_score`:
    the area fraction scaled by 10 and capped at 1.0, NOT the raw fraction.
    No candidate is ever rejected - not by confidence, not by class - so a
    non-empty candidate list always yields exactly one winner and the
    full-frame fallback has exactly one cause: nothing was detected at all.
    """

    # A square frame keeps the hand-computed expectations below readable -
    # centre proximity is resolution-independent (see its own test).
    FRAME = (1000, 1000)

    @staticmethod
    def _box_at(cx, cy, size=10.0):
        """A `size`x`size` box centred on (cx, cy)."""
        return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)

    def test_empty_candidates_returns_none(self):
        """The ONLY route to None, and therefore the only cause of
        build_crop's full-frame fallback."""
        self.assertIsNone(select_best_detection([], self.FRAME))

    def test_a_single_candidate_is_always_returned_whatever_its_confidence(self):
        for score in (0.01, 0.31, 0.6, 0.99):
            with self.subTest(score=score):
                only = BirdDetection(box=(0, 0, 5, 5), score=score, label=COCO_BIRD_CLASS)
                self.assertIs(select_best_detection([only], self.FRAME), only)

    # -- 1. centre proximity decides between otherwise-similar candidates ----

    def test_a_centred_candidate_beats_an_identical_off_centre_one(self):
        centred = BirdDetection(box=self._box_at(500, 500, 100), score=0.8, label=COCO_BIRD_CLASS)
        off_centre = BirdDetection(box=self._box_at(900, 900, 100), score=0.8, label=COCO_BIRD_CLASS)

        # Same size, same confidence - position is the only difference.
        self.assertIs(select_best_detection([off_centre, centred], self.FRAME), centred)

    def test_a_centred_candidate_beats_the_best_possible_corner_candidate(self):
        """The exact guarantee the 50% weight buys, stated as the bound it
        actually is: a candidate centred on the frame corner scores 0 on the
        centre term, so its ceiling is 0.30 + 0.20 = 0.50 however large and
        however confident it is. Any candidate at the frame's centre clears
        that on the centre term alone."""
        centred = BirdDetection(box=self._box_at(500, 500, 100), score=0.31, label=COCO_BIRD_CLASS)
        # Centre exactly on the frame corner, maximal size and confidence.
        corner = BirdDetection(box=(900, 900, 1100, 1100), score=1.0, label=COCO_BIRD_CLASS)

        self.assertLessEqual(selection_score(corner, self.FRAME), 0.50)
        self.assertGreater(selection_score(centred, self.FRAME), 0.50)
        self.assertIs(select_best_detection([corner, centred], self.FRAME), centred)

    def test_scaling_the_size_term_made_a_large_off_centre_subject_competitive(self):
        """A real, deliberate behaviour change from v8, pinned so it is a
        decision rather than a surprise.

        These two candidates are the pair v8's own test used to assert that
        centre proximity "outweighs area and confidence together". Under v8
        the corner box's raw area fraction (0.16) contributed 0.048 and the
        centred box won 0.592 to 0.446. Under v9 that same 16% is past the
        size cap and contributes the full 0.30, and the corner box wins
        0.698 to 0.592.

        This is the intended trade: the size term was nearly inert before and
        now does real work. Centre proximity still dominates at equal size
        (see the test above and `_beats_an_identical_off_centre_one`), but it
        no longer overrides a subject that is genuinely much larger.
        """
        centred = BirdDetection(box=self._box_at(500, 500, 100), score=0.31, label=COCO_BIRD_CLASS)
        large_off_centre = BirdDetection(box=(0, 0, 400, 400), score=0.99, label=COCO_BIRD_CLASS)

        self.assertAlmostEqual(selection_score(centred, self.FRAME), 0.592)
        self.assertAlmostEqual(selection_score(large_off_centre, self.FRAME), 0.698)
        self.assertIs(
            select_best_detection([large_off_centre, centred], self.FRAME), large_off_centre
        )

    # -- 2/3/4. the exact composite ------------------------------------------

    def test_the_composite_is_exactly_50_centre_30_size_20_confidence(self):
        detection = BirdDetection(box=(400, 400, 600, 600), score=0.5, label=COCO_BIRD_CLASS)

        score = selection_score(detection, self.FRAME)

        # Centred exactly -> 1.0; 200x200 of 1000x1000 -> 0.04 area fraction
        # -> 10 * 0.04 = 0.40 size score.
        expected = 0.50 * 1.0 + 0.30 * 0.40 + 0.20 * 0.5
        self.assertAlmostEqual(score, expected)
        self.assertAlmostEqual(score, 0.72)

    def test_size_contributes_exactly_thirty_percent_of_the_scaled_score(self):
        """Two candidates identical but for size, both dead-centre: the score
        gap must be exactly 0.30 * the SIZE-SCORE difference - the scaled and
        capped value, not the raw area fraction."""
        # 1% and 4% of the frame -> size scores 0.10 and 0.40, both below
        # the cap so the difference is a real one.
        small = BirdDetection(box=self._box_at(500, 500, 100), score=0.5, label=COCO_BIRD_CLASS)
        large = BirdDetection(box=self._box_at(500, 500, 200), score=0.5, label=COCO_BIRD_CLASS)

        gap = selection_score(large, self.FRAME) - selection_score(small, self.FRAME)

        self.assertAlmostEqual(gap, 0.30 * (0.40 - 0.10))
        self.assertIs(select_best_detection([small, large], self.FRAME), large)

    def test_the_size_term_uses_ten_times_the_area_fraction_capped_at_one(self):
        """The exact curve the spec names: clamp01(10 * area_fraction)."""
        cases = {
            0.01: 0.10,   # 1% of the frame
            0.05: 0.50,
            0.065: 0.65,  # this archive's median real subject
            0.10: 1.00,   # the cap
            0.20: 1.00,
            0.90: 1.00,
        }
        for fraction, expected in cases.items():
            with self.subTest(fraction=fraction):
                self.assertAlmostEqual(subject_size_score(fraction), expected)

    def test_the_size_score_is_reachable_from_a_box_and_matches_the_fraction_form(self):
        box = self._box_at(500, 500, 200)  # 4% of a 1000x1000 frame
        self.assertAlmostEqual(relative_box_area(box, self.FRAME), 0.04)
        self.assertAlmostEqual(relative_size_score(box, self.FRAME), 0.40)
        self.assertAlmostEqual(
            relative_size_score(box, self.FRAME),
            subject_size_score(relative_box_area(box, self.FRAME)),
        )

    def test_two_subjects_past_the_cap_are_separated_by_position_not_size(self):
        """A deliberate consequence of capping: above 10% of the frame the
        size term stops discriminating, and the remaining 70% of the score
        decides. Pinned so the cap's cost is visible rather than surprising."""
        big_off_centre = BirdDetection(box=self._box_at(800, 800, 400), score=0.5, label=COCO_BIRD_CLASS)
        bigger_centred = BirdDetection(box=self._box_at(500, 500, 350), score=0.5, label=COCO_BIRD_CLASS)

        self.assertAlmostEqual(relative_size_score(big_off_centre.box, self.FRAME), 1.0)
        self.assertAlmostEqual(relative_size_score(bigger_centred.box, self.FRAME), 1.0)
        self.assertIs(
            select_best_detection([big_off_centre, bigger_centred], self.FRAME), bigger_centred
        )

    def test_confidence_contributes_exactly_twenty_percent(self):
        """Identical geometry, different confidence - the gap must be exactly
        0.20 * the confidence difference."""
        unsure = BirdDetection(box=self._box_at(500, 500, 100), score=0.20, label=COCO_BIRD_CLASS)
        sure = BirdDetection(box=self._box_at(500, 500, 100), score=0.90, label=COCO_BIRD_CLASS)

        gap = selection_score(sure, self.FRAME) - selection_score(unsure, self.FRAME)

        self.assertAlmostEqual(gap, 0.20 * (0.90 - 0.20))
        self.assertIs(select_best_detection([unsure, sure], self.FRAME), sure)

    # -- 5. no confidence floor ----------------------------------------------

    def test_a_candidate_below_the_old_floor_still_wins_on_composite_score(self):
        """THE behavioural change. Under v7 this candidate (0.31, below the
        0.6 floor) was discarded outright and the image fell through to the
        full-frame fallback despite a real detection existing."""
        below_old_floor = BirdDetection(box=self._box_at(500, 500, 200), score=0.31, label=COCO_BIRD_CLASS)

        self.assertIs(select_best_detection([below_old_floor], self.FRAME), below_old_floor)

    def test_every_candidate_below_the_old_floor_still_yields_a_selection(self):
        candidates = [
            BirdDetection(box=self._box_at(200, 200, 50), score=0.05, label=COCO_BIRD_CLASS),
            BirdDetection(box=self._box_at(500, 500, 50), score=0.10, label=COCO_BIRD_CLASS),
            BirdDetection(box=self._box_at(800, 800, 50), score=0.15, label=COCO_BIRD_CLASS),
        ]

        winner = select_best_detection(candidates, self.FRAME)

        self.assertIsNotNone(winner, "candidates existing must always produce a selection")
        self.assertIs(winner, candidates[1], "the centred one")

    # -- 6. always exactly one winner ----------------------------------------

    def test_many_candidates_always_produce_exactly_one_winner(self):
        for count in range(1, DEFAULT_GROUP_SCENE_THRESHOLD):
            with self.subTest(count=count):
                candidates = [
                    BirdDetection(box=self._box_at(100 * (i + 1), 100 * (i + 1), 40),
                                  score=0.1 * (i + 1), label=COCO_BIRD_CLASS)
                    for i in range(count)
                ]
                winner = select_best_detection(candidates, self.FRAME)
                self.assertIsNotNone(winner)
                self.assertEqual(sum(1 for c in candidates if c is winner), 1)

    # -- 8/9. centre-proximity normalisation ---------------------------------

    def test_centre_proximity_is_one_at_the_centre_and_zero_at_a_corner(self):
        self.assertAlmostEqual(center_proximity(self._box_at(500, 500, 0), self.FRAME), 1.0)
        for corner in ((0, 0), (1000, 0), (0, 1000), (1000, 1000)):
            with self.subTest(corner=corner):
                self.assertAlmostEqual(center_proximity(self._box_at(*corner, 0), self.FRAME), 0.0)

    def test_centre_proximity_is_one_half_halfway_to_a_corner(self):
        """Linear in Euclidean distance, normalised by the centre-to-corner
        distance - so the midpoint of that diagonal is exactly 0.5."""
        self.assertAlmostEqual(center_proximity(self._box_at(750, 750, 0), self.FRAME), 0.5)

    def test_centre_proximity_is_resolution_independent(self):
        """The same RELATIVE position must score identically at any size -
        otherwise a 6000px frame and a 600px one would rank differently."""
        scores = [
            center_proximity(self._box_at(w * 0.75, h * 0.75, 0), (w, h))
            for w, h in ((1000, 1000), (6000, 6000), (600, 400), (4000, 3000), (100, 100))
        ]
        for value in scores:
            self.assertAlmostEqual(value, scores[0])
        self.assertAlmostEqual(scores[0], 0.5)

    def test_centre_proximity_stays_within_zero_and_one_for_a_box_outside_the_frame(self):
        """A detector box can extend past the frame edge; clamping keeps the
        term well-defined instead of going negative."""
        self.assertEqual(center_proximity(self._box_at(-5000, -5000, 0), self.FRAME), 0.0)

    def test_a_degenerate_frame_does_not_crash_the_selection(self):
        candidates = [BirdDetection(box=(0, 0, 10, 10), score=0.5, label=COCO_BIRD_CLASS)]
        self.assertIs(select_best_detection(candidates, (0, 0)), candidates[0])

    def test_relative_area_is_box_area_over_frame_area_clamped(self):
        self.assertAlmostEqual(relative_box_area((0, 0, 500, 500), self.FRAME), 0.25)
        self.assertAlmostEqual(relative_box_area((0, 0, 1000, 1000), self.FRAME), 1.0)
        # A box larger than the frame clamps rather than exceeding 1.0.
        self.assertAlmostEqual(relative_box_area((0, 0, 5000, 5000), self.FRAME), 1.0)

    def test_the_score_is_always_within_zero_and_one(self):
        for box, score in (
            ((0, 0, 1000, 1000), 1.0), ((0, 0, 1, 1), 0.0),
            ((-100, -100, 2000, 2000), 1.5), (self._box_at(500, 500, 10), -0.5),
        ):
            with self.subTest(box=box, score=score):
                value = selection_score(BirdDetection(box=box, score=score, label=COCO_BIRD_CLASS), self.FRAME)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_never_returns_a_detection_absent_from_the_input(self):
        candidates = [
            BirdDetection(box=self._box_at(300 + 100 * i, 500, 40), score=s, label=COCO_BIRD_CLASS)
            for i, s in enumerate((0.61, 0.75, 0.99))
        ]
        self.assertIn(select_best_detection(candidates, self.FRAME), candidates)

    # -- the two historical failure modes still resolve correctly ------------

    def test_the_real_cow_false_positive_still_loses_to_the_confident_bird(self):
        """The exact failure mode found during the EyePose investigation
        (docs/EyePose_Investigation_Phase_1.md's Q1, image DSC03129): a
        0.998-confidence bird lost the crop to an unrelated, much larger,
        0.458-confidence false positive (originally mislabelled "cow").

        v7 resolved it with an absolute confidence floor the false positive
        could not clear. v8 has no floor at all, so this is a real check
        that removing it did not reopen the case: the bird is both nearer
        the frame centre and far more confident, and wins 0.626 to 0.442
        even though the false positive's box is ~8x larger."""
        frame = (6000, 4000)  # the source frame these real boxes came from
        bird = BirdDetection(box=(2158.2, 929.6, 3071.4, 2090.5), score=0.998, label=COCO_BIRD_CLASS)
        false_positive = BirdDetection(box=(18.2, 0.0, 2329.1, 3642.3), score=0.458, label=21)  # cow

        self.assertIs(select_best_detection([bird, false_positive], frame), bird)
        self.assertGreater(selection_score(bird, frame), selection_score(false_positive, frame))

    def test_a_small_very_confident_distractor_still_loses_to_the_intended_subject(self):
        """A plausible reconstruction (not measured data - flagged as such in
        the report) of the failure v3 was built to fix and v6 reopened: a
        legitimately-real, moderately-confident intended subject against a
        small, very-confident background bird. Under v8 the subject wins on
        centre proximity and area together, despite the lower confidence."""
        intended_subject = BirdDetection(box=(0, 0, 300, 300), score=0.75, label=COCO_BIRD_CLASS)
        background_bird = BirdDetection(box=(0, 0, 20, 20), score=0.99, label=COCO_BIRD_CLASS)

        self.assertIs(select_best_detection([intended_subject, background_bird], self.FRAME), intended_subject)


class SelectionIsClassAgnosticTests(unittest.TestCase):
    """v9's central requirement: COCO is a LOCALIZATION tool, and its class
    label must not decide whether a box can be cropped to.

    The detector answers "where might the subject be?", never "is this a
    valid wildlife subject?". These tests are the guard on that - if a class
    gate is ever reintroduced anywhere in the crop path, several of them fail
    immediately.
    """

    FRAME = (1000, 1000)

    @staticmethod
    def _box_at(cx, cy, size=10.0):
        return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)

    # Classes spanning every category the old gate distinguished: an accepted
    # animal, the specifically-excluded person, and three COCO classes this
    # project has never catalogued at all (car, potted plant, teddy bear -
    # the last being a real label COCO gives primates and other unsupported
    # animals).
    CLASS_IDS = (COCO_BIRD_CLASS, COCO_PERSON_CLASS, 3, 64, 88)

    def test_the_winner_is_identical_whatever_class_the_candidates_carry(self):
        """The same geometry must produce the same selection under every
        label, including labels with no category at all."""
        for class_id in self.CLASS_IDS:
            with self.subTest(class_id=class_id):
                centred = BirdDetection(box=self._box_at(500, 500, 200), score=0.5, label=class_id)
                corner = BirdDetection(box=self._box_at(950, 950, 200), score=0.5, label=class_id)
                self.assertIs(select_best_detection([corner, centred], self.FRAME), centred)

    def test_the_score_is_identical_across_classes_for_identical_geometry(self):
        box = self._box_at(400, 600, 150)
        scores = {
            class_id: selection_score(BirdDetection(box=box, score=0.7, label=class_id), self.FRAME)
            for class_id in self.CLASS_IDS
        }
        self.assertEqual(
            len(set(round(v, 12) for v in scores.values())), 1, f"class changed the score: {scores}"
        )

    def test_a_person_wins_when_it_has_the_highest_composite_score(self):
        """Explicitly permitted, and the single most important case: COCO has
        no primate class, so a monkey is routinely labelled "person". A gate
        that excludes people excludes those monkeys."""
        person = BirdDetection(box=self._box_at(500, 500, 300), score=0.95, label=COCO_PERSON_CLASS)
        bird = BirdDetection(box=self._box_at(950, 60, 40), score=0.99, label=COCO_BIRD_CLASS)

        winner = select_best_detection([bird, person], self.FRAME)

        self.assertIs(winner, person)
        self.assertEqual(winner.label, COCO_PERSON_CLASS)

    def test_an_uncatalogued_class_wins_when_it_has_the_highest_composite_score(self):
        """A class the project has no display name for at all is still a
        perfectly good candidate REGION - which is the only thing selection
        is choosing between."""
        unknown = BirdDetection(box=self._box_at(500, 500, 300), score=0.9, label=88)  # "teddy bear"
        bird = BirdDetection(box=self._box_at(950, 60, 40), score=0.99, label=COCO_BIRD_CLASS)

        winner = select_best_detection([bird, unknown], self.FRAME)

        self.assertIs(winner, unknown)
        self.assertIsNone(detection_category(88), "and it genuinely has no category to fall back on")

    def test_a_person_only_frame_still_produces_a_selection(self):
        """THE regression from the real archive. Under v8 this image had
        confident boxes, no selection, and was filed as a full-frame fallback
        - identically to an image the detector found nothing in at all. On
        this project's 5,986-image DCIM tree that was 1,506 images, whose
        best candidate confidence had a median of 0.886."""
        people = [
            BirdDetection(box=(1827, 1057, 2260, 2469), score=0.9994, label=COCO_PERSON_CLASS),
            BirdDetection(box=(1191, 1183, 1544, 1773), score=0.9816, label=COCO_PERSON_CLASS),
        ]

        winner = select_best_detection(people, (5496, 3672))

        self.assertIsNotNone(winner, "person-only boxes must not mean 'nothing was detected'")
        self.assertIs(winner, people[0], "the larger, more central of the two")

    def test_no_class_can_be_eliminated_by_being_the_only_candidate(self):
        for class_id in (*self.CLASS_IDS, 999):  # 999: not a COCO class at all
            with self.subTest(class_id=class_id):
                only = BirdDetection(box=self._box_at(500, 500, 100), score=0.05, label=class_id)
                self.assertIs(select_best_detection([only], self.FRAME), only)

    def test_selection_never_reads_the_label_attribute(self):
        """Structural, not behavioural: a detection whose `label` raises on
        access must still be selectable. This catches a class gate added
        anywhere in the scoring path, including one that only reads the label
        to break a tie."""

        class LabelExplodes:
            """Duck-typed like a BirdDetection, minus a readable class."""

            def __init__(self, box, score):
                self.box = box
                self.score = score

            @property
            def label(self):
                raise AssertionError("selection must never read a detection's class")

        candidate = LabelExplodes(self._box_at(500, 500, 200), 0.8)

        self.assertAlmostEqual(
            selection_score(candidate, self.FRAME),
            selection_score(
                BirdDetection(box=self._box_at(500, 500, 200), score=0.8, label=COCO_BIRD_CLASS),
                self.FRAME,
            ),
        )
        self.assertIs(select_best_detection([candidate], self.FRAME), candidate)


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
    a flock/herd/colony is often the actual subject, not any one animal in it.

    UNCHANGED by v8. This branch is evaluated before any individual scoring,
    so neither the new composite score nor `source_size` participates in it -
    these tests are the guard on that, and they assert exactly what they
    asserted under v7."""

    FRAME = (1000, 1000)

    def test_fewer_than_the_threshold_uses_normal_individual_selection(self):
        candidates = _flock(9)  # below the threshold: individual selection applies
        winner = select_best_detection(candidates, self.FRAME, group_scene_threshold=10)
        self.assertIn(winner, candidates, "below threshold, the winner must be one real detection")
        self.assertNotEqual(winner.box, enclosing_box([c.box for c in candidates]))

    def test_exactly_the_threshold_is_a_group_scene(self):
        candidates = _flock(10)
        winner = select_best_detection(candidates, self.FRAME, group_scene_threshold=10)
        self.assertNotIn(winner, candidates, "a group scene's box must not be any single detection")
        self.assertEqual(winner.box, enclosing_box([c.box for c in candidates]))

    def test_more_than_the_threshold_is_a_group_scene(self):
        candidates = _flock(25)
        winner = select_best_detection(candidates, self.FRAME, group_scene_threshold=10)
        self.assertEqual(winner.box, enclosing_box([c.box for c in candidates]))

    def test_the_threshold_is_configurable(self):
        candidates = _flock(4)
        # Below a threshold of 5, normal per-detection selection applies...
        normal = select_best_detection(candidates, self.FRAME, group_scene_threshold=5)
        self.assertIn(normal, candidates)
        # ...but the same 4 detections are a group scene once the threshold is lowered to 4.
        group = select_best_detection(candidates, self.FRAME, group_scene_threshold=4)
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
        winner = select_best_detection(candidates, self.FRAME, group_scene_threshold=10)

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
        winner = select_best_detection(candidates, self.FRAME, group_scene_threshold=10)
        most_confident = max(candidates, key=lambda d: d.score)
        self.assertEqual(winner.score, most_confident.score)
        self.assertEqual(winner.label, most_confident.label)

    def test_group_scene_is_unaffected_by_the_frame_it_is_measured_against(self):
        """v8 guard: the individual path scores candidates against the frame,
        so a different `source_size` reorders individual selection. A group
        scene must be identical either way - it never scores anything."""
        candidates = _flock(10)
        expected = enclosing_box([c.box for c in candidates])
        for frame in ((1000, 1000), (6000, 4000), (100, 100), (0, 0)):
            with self.subTest(frame=frame):
                winner = select_best_detection(candidates, frame, group_scene_threshold=10)
                self.assertEqual(winner.box, expected)

    def test_group_scene_never_rejects_low_confidence_members(self):
        """_flock()'s scores (0.50-0.58) all sit below v7's removed 0.6 floor;
        the group box must still enclose every one of them."""
        candidates = _flock(10)
        winner = select_best_detection(candidates, self.FRAME, group_scene_threshold=10)
        self.assertEqual(winner.box, enclosing_box([c.box for c in candidates]))


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

    def test_picks_the_best_scoring_animal_and_never_a_person(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 30, 30], [1, 1, 2, 2]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS, 1],  # two birds + a person
            scores=[0.99, 0.31, 0.995],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertIsInstance(detection, BirdDetection)
        # In this 40x40 frame the larger (5,5,30,30) bird is nearly centred
        # and fills 39% of it: 0.617 against the corner bird's 0.342, so it
        # wins despite its much lower confidence (0.31 vs 0.99). Under v7's
        # removed floor the 0.31 candidate was discarded and the corner bird
        # won instead - this is the v8 behaviour change, end to end.
        self.assertEqual(detection.box, (5.0, 5.0, 30.0, 30.0))
        self.assertAlmostEqual(detection.score, 0.31, places=5)
        # The person is never crop-eligible, whatever it scores.
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

    def test_none_only_when_every_box_is_below_the_detector_threshold(self):
        """The sole route to None at the detector level. A high-confidence
        non-animal no longer contributes to it - v9's whole point - so this
        pins the case where the model genuinely produced nothing usable."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 20, 20]],
            labels=[COCO_BIRD_CLASS, COCO_PERSON_CLASS],
            scores=[0.1, 0.2],  # both below conf_threshold 0.3
        )
        self.assertIsNone(detector.detect_best_bird(self.IMG))
        self.assertIsNone(detector.best_bird_box(self.IMG))

    def test_a_high_confidence_non_animal_no_longer_produces_none(self):
        """The same input as the v8 test this replaces (bird below the
        threshold, person well above it), asserting the opposite outcome."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 20, 20]],
            labels=[COCO_BIRD_CLASS, COCO_PERSON_CLASS],
            scores=[0.1, 0.99],
        )
        detection = detector.detect_best_bird(self.IMG)

        self.assertIsNotNone(detection)
        self.assertEqual(detection.label, COCO_PERSON_CLASS)
        self.assertEqual(detector.best_bird_box(self.IMG), (5.0, 5.0, 20.0, 20.0))

    def test_a_real_model_output_with_a_flock_becomes_a_group_scene(self):
        """End to end through the real confidence filtering, not just
        select_best_detection() in isolation: ten birds above the threshold
        are a group scene, and the enclosing box covers all ten."""
        boxes = [[i * 15, 0, i * 15 + 8, 8] for i in range(10)]
        labels = [COCO_BIRD_CLASS] * 10
        scores = [0.5 + i * 0.01 for i in range(10)]
        detector = _detector_with_fake_model(boxes=boxes, labels=labels, scores=scores)

        detection = detector.detect_best_bird(self.IMG)
        expected = enclosing_box([tuple(float(v) for v in b) for b in boxes])
        self.assertEqual(detection.box, expected)

    def test_a_non_animal_box_now_counts_toward_the_group_scene_threshold(self):
        """A flagged, accepted consequence of removing the class gate (see
        bird_crop's v9 note): candidates of ANY class count toward
        `group_scene_threshold`, so nine birds plus one person is a group
        scene where under v8 it was nine birds and an individual selection.
        Pinned here so the interaction is a recorded decision rather than a
        surprise in the field."""
        boxes = [[i * 15, 0, i * 15 + 8, 8] for i in range(9)] + [[500, 500, 520, 520]]
        labels = [COCO_BIRD_CLASS] * 9 + [COCO_PERSON_CLASS]
        scores = [0.5 + i * 0.01 for i in range(9)] + [0.99]
        detector = _detector_with_fake_model(boxes=boxes, labels=labels, scores=scores)

        detection = detector.detect_best_bird(self.IMG)

        expected = enclosing_box([tuple(float(v) for v in b) for b in boxes])
        self.assertEqual(detection.box, expected, "all ten boxes, person included, are the group")


class RecordedDetectionsTests(unittest.TestCase):
    """What `detect_with_all` records, and its relationship to what can win.

    Through v8 these were two different sets: `catalogue_classes` decided
    what was recorded, `classes` decided what could be cropped to, and the
    first was a strict superset of the second. v9 collapses them - every box
    above `conf_threshold` is both recorded and eligible - so an image can
    never again display candidate boxes that were barred from winning.
    """

    IMG = np.zeros((40, 40, 3), dtype=np.uint8)

    def test_the_winner_is_drawn_from_exactly_the_recorded_detections(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [0, 0, 39, 39], [1, 1, 5, 5]],
            labels=[COCO_BIRD_CLASS, COCO_PERSON_CLASS, 88],
            scores=[0.7, 0.99, 0.5],
        )
        winner, recorded = detector.detect_with_all(self.IMG)

        self.assertEqual(len(recorded), 3, "every candidate is recorded, whatever its class")
        self.assertIn(winner, recorded, "and the winner is one of them")

    def test_a_person_is_recorded_and_may_win(self):
        """The inverse of the v8 test this replaces. The large, confident,
        near-centred person wins over the tiny corner bird - on composition,
        which is the only thing selection judges."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [0, 0, 39, 39]],  # small corner bird, huge person
            labels=[COCO_BIRD_CLASS, COCO_PERSON_CLASS],
            scores=[0.7, 0.99],
        )
        winner, recorded = detector.detect_with_all(self.IMG)

        self.assertEqual(winner.label, COCO_PERSON_CLASS)
        self.assertEqual(len(recorded), 2)

    def test_the_detector_exposes_no_class_gate_at_all(self):
        """Structural: the attributes that used to gate crop eligibility are
        gone, not merely defaulted wide. A future reader cannot set them back
        without noticing they no longer exist."""
        detector = _detector_with_fake_model(boxes=[], labels=[], scores=[])
        self.assertFalse(hasattr(detector, "classes"))
        self.assertFalse(hasattr(detector, "catalogue_classes"))

    def test_only_a_person_present_still_yields_a_crop_winner(self):
        """The 1,506-image regression, at the detector level."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 39, 39]], labels=[COCO_PERSON_CLASS], scores=[0.99]
        )
        winner, recorded = detector.detect_with_all(self.IMG)

        self.assertIsNotNone(winner, "a person-only frame is not an empty frame")
        self.assertEqual(winner.label, COCO_PERSON_CLASS)
        self.assertEqual(len(recorded), 1)
        self.assertIsNotNone(detector.best_bird_box(self.IMG))

    def test_an_uncatalogued_class_is_recorded_and_can_win(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 39, 39]], labels=[88], scores=[0.8]  # "teddy bear"
        )
        winner, recorded = detector.detect_with_all(self.IMG)

        self.assertIsNotNone(winner)
        self.assertEqual(winner.label, 88)
        self.assertEqual(len(recorded), 1)

    def test_confidence_still_decides_whether_a_box_exists_at_all(self):
        """`conf_threshold` is NOT a class gate and is deliberately kept: it
        decides whether the model produced a box worth calling a candidate,
        not whether an existing candidate is allowed to win."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [5, 5, 30, 30]],
            labels=[COCO_BIRD_CLASS, COCO_PERSON_CLASS],
            scores=[0.1, 0.99],  # the bird is below the detector's own floor
            conf_threshold=0.3,
        )
        winner, recorded = detector.detect_with_all(self.IMG)

        self.assertEqual(len(recorded), 1, "the sub-threshold box is not a candidate at all")
        self.assertEqual(winner.label, COCO_PERSON_CLASS)


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

    def test_selection_is_decided_purely_by_the_score_not_the_class(self):
        # A tiny, near-perfectly-confident bird in the corner against a
        # larger, dead-centre, far less confident elephant, both fully inside
        # the 40x40 frame. The elephant wins 0.870 to 0.341 - not because it
        # is an elephant, but because position and size together outweigh the
        # confidence gap. Nothing in the selection path reads `label` at all.
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 6, 6], [12, 12, 28, 28]],
            labels=[COCO_BIRD_CLASS, 22],
            scores=[0.99, 0.35],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.box, (12.0, 12.0, 28.0, 28.0))
        self.assertEqual(detection.label, 22)

    def test_two_animal_classes_are_ranked_only_by_composition(self):
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 100, 100], [0, 0, 96, 100]],
            labels=[COCO_BIRD_CLASS, 22],
            scores=[0.9, 0.82],
        )
        detection = detector.detect_best_bird(self.IMG)
        self.assertEqual(detection.label, COCO_BIRD_CLASS)
        self.assertEqual(detection.box, (0.0, 0.0, 100.0, 100.0))

    def test_non_animal_classes_are_now_valid_crop_candidates(self):
        """The inverse of the v8 test this replaces. person(1), car(3) and
        airplane(5) are all candidate REGIONS; the biggest, most central of
        them wins, which is the only question selection asks."""
        detector = _detector_with_fake_model(
            boxes=[[0, 0, 10, 10], [1, 1, 5, 5], [2, 2, 8, 8]],
            labels=[COCO_PERSON_CLASS, 3, 5],
            scores=[0.99, 0.98, 0.97],
        )
        detection = detector.detect_best_bird(self.IMG)

        self.assertIsNotNone(detection)
        self.assertEqual(detection.label, COCO_PERSON_CLASS)
        self.assertEqual(detection.box, (0.0, 0.0, 10.0, 10.0))

    def test_no_argument_exists_to_restrict_selection_back_to_birds_only(self):
        """v9 removed `classes` deliberately rather than defaulting it wide,
        so a caller cannot quietly reinstate the gate this change exists to
        remove."""
        with self.assertRaises(TypeError):
            BirdDetector(classes={COCO_BIRD_CLASS})


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


class BuildCropSelectsTheBestScoringDetectionTests(unittest.TestCase):
    """End to end through build_crop with the real BirdDetector (fake model
    output): the pixels actually cached must come from the highest-scoring
    detection - closing the loop from selection policy to the crop that
    training and EyePose will actually see."""

    def test_the_cached_crop_comes_from_the_best_scoring_detection(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:20, 10:20] = (255, 0, 0)     # small, off-centre, very confident
        frame[30:90, 30:90] = (0, 255, 0)     # large, central, barely confident

        detector = _detector_with_fake_model(
            boxes=[[10, 10, 20, 20], [30, 30, 90, 90]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.99, 0.31],
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.0), collect_detections=True)

        # Green scores 0.570 (centred, 36% of the frame) against red's 0.351
        # (a corner box at 1%), so the low-confidence central subject wins -
        # v7's removed floor would have discarded it and cropped the red box.
        self.assertEqual(result.detection.box, (30.0, 30.0, 90.0, 90.0))
        mean_pixel = result.crop.reshape(-1, 3).mean(axis=0)
        self.assertGreater(mean_pixel[1], mean_pixel[0], "crop should be the central green box")

    def test_candidates_existing_never_produces_the_full_frame_fallback(self):
        """The v8 invariant end to end: however unconfident every candidate
        is, a detection was found, so a real crop - not the full frame - is
        what gets cached."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = _detector_with_fake_model(
            boxes=[[40, 40, 60, 60], [0, 0, 8, 8]],
            labels=[COCO_BIRD_CLASS, COCO_BIRD_CLASS],
            scores=[0.31, 0.35],  # both far below v7's removed 0.6 floor
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.0), collect_detections=True)

        self.assertIsNotNone(result.detection, "candidates existed - there must be a selection")
        self.assertIsNotNone(result.expanded_box)
        self.assertEqual(result.detection.box, (40.0, 40.0, 60.0, 60.0), "the centred one")
        self.assertNotEqual(result.crop.shape[:2], frame.shape[:2], "not the full-frame fallback")

    def test_a_person_only_frame_produces_a_real_crop_not_the_full_frame(self):
        """The v9 invariant end to end, and THE regression this change is
        for: boxes of a class the old gate excluded still produce a real
        crop rather than the full-frame fallback."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = _detector_with_fake_model(
            boxes=[[40, 40, 60, 60], [0, 0, 8, 8]],
            labels=[COCO_PERSON_CLASS, COCO_PERSON_CLASS],
            scores=[0.9, 0.95],
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.0), collect_detections=True)

        self.assertIsNotNone(result.detection, "person boxes are candidates like any other")
        self.assertEqual(result.detection.label, COCO_PERSON_CLASS)
        self.assertEqual(result.detection.box, (40.0, 40.0, 60.0, 60.0), "the centred one")
        self.assertNotEqual(result.crop.shape[:2], frame.shape[:2], "not the full-frame fallback")
        self.assertEqual(len(result.all_detections), 2, "and both are recorded for the overlay")

    def test_zero_detections_is_the_only_route_to_the_full_frame_fallback(self):
        """The other half of the invariant: nothing detected at all IS still
        a full-frame fallback, unchanged, and the crop really is the frame."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = _detector_with_fake_model(boxes=[], labels=[], scores=[])

        result = build_crop(frame, detector, CropParams(margin_frac=0.0), collect_detections=True)

        self.assertIsNone(result.detection)
        self.assertIsNone(result.expanded_box)
        self.assertEqual(result.all_detections, [])
        self.assertEqual(result.crop.shape[:2], frame.shape[:2], "the whole frame, unchanged")


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


class SelectedDetectionIsPersistedAndDisplayedTests(unittest.TestCase):
    """The winner of `select_best_detection` is what reaches the cache as
    `selected`, and what the overlay draws GREEN - every other candidate is
    persisted alongside it and drawn yellow. This closes the loop from the
    v8 scoring rule to what the photographer actually sees on screen."""

    def test_the_best_scoring_candidate_is_the_one_saved_as_selected(self):
        import json

        from picklikeme.bird_crop import save_detections

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = _detector_with_fake_model(
            boxes=[[40, 40, 60, 60], [0, 0, 10, 10], [85, 85, 95, 95]],
            labels=[COCO_BIRD_CLASS] * 3,
            scores=[0.31, 0.99, 0.95],
        )
        result = build_crop(frame, detector, CropParams(margin_frac=0.0), collect_detections=True)

        with tempfile.TemporaryDirectory() as tmp:
            image_path = str(Path(tmp) / "shot.arw")
            save_detections(tmp, image_path, result)
            # Sharded path - always computed, never guessed (see crop_cache_path).
            payload = json.loads(detections_cache_path(tmp, image_path).read_text(encoding="utf-8"))

        # The centred, least-confident candidate wins and is persisted as
        # `selected`; all three are persisted in `detections`.
        self.assertEqual(payload["selected"]["box"], [40.0, 40.0, 60.0, 60.0])
        self.assertEqual(len(payload["detections"]), 3)

    def test_the_selected_box_renders_green_and_the_rest_yellow(self):
        """`analyzer.detections` marks exactly the box whose coordinates match
        `selected`; the Loupe draws that one with SELECTED_BOX (green) and
        every other with OTHER_BOX (yellow) - see loupe_dialog's overlay."""
        from picklikeme.analyzer.contactsheets import OTHER_BOX, SELECTED_BOX
        from picklikeme.analyzer.detections import _from_payload

        payload = {
            "source_size": [100, 100],
            "selected": {"box": [40.0, 40.0, 60.0, 60.0], "score": 0.31, "label": COCO_BIRD_CLASS},
            "detections": [
                {"box": [40.0, 40.0, 60.0, 60.0], "score": 0.31, "label": COCO_BIRD_CLASS},
                {"box": [0.0, 0.0, 10.0, 10.0], "score": 0.99, "label": COCO_BIRD_CLASS},
                {"box": [85.0, 85.0, 95.0, 95.0], "score": 0.95, "label": COCO_BIRD_CLASS},
            ],
            "expanded_box": [40.0, 40.0, 60.0, 60.0],
        }

        record = _from_payload(payload, origin="cache")

        self.assertIsNotNone(record.selected)
        self.assertEqual((record.selected.x1, record.selected.y1), (40.0, 40.0))
        self.assertEqual(len(record.others), 2, "the two runners-up")
        self.assertNotIn((40.0, 40.0), [(b.x1, b.y1) for b in record.others])
        # Green vs yellow, so a change to either constant fails here.
        self.assertEqual(SELECTED_BOX, (16, 185, 129))
        self.assertEqual(OTHER_BOX, (250, 204, 21))
