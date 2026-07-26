"""Part 2: annotated false-negative thumbnails.

The point of the feature is that one thumbnail answers "did the detector
contribute to this miss?", so the tests check that the boxes are actually drawn,
that they land in the right place, and - just as importantly - that no other
report section is touched.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.contactsheets import (
    annotate_thumbnail,
    annotated_thumbnail_path,
    build_false_negative_overlays,
    render_contact_sheets,
    _thumbnail_cache_path,
)
from picklikeme.analyzer.detections import Box, DetectionCache, DetectionRecord, _from_payload
from picklikeme.bird_crop import (
    BirdDetection,
    CropParams,
    CropResult,
    build_crop,
    detections_cache_path,
    read_detections,
    save_detections,
)


class TwoBoxDetector:
    """A winner plus a runner-up, from one call."""

    def detect_with_all(self, image_rgb):
        winner = BirdDetection(box=(10.0, 20.0, 60.0, 70.0), score=0.92, label=16)
        other = BirdDetection(box=(120.0, 30.0, 150.0, 55.0), score=0.41, label=24)
        return winner, [winner, other]

    def detect_best_bird(self, image_rgb):
        return self.detect_with_all(image_rgb)[0]


class NothingDetector:
    def detect_with_all(self, image_rgb):
        return None, []

    def detect_best_bird(self, image_rgb):
        return None


def make_thumb(path: Path, size: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), 40, dtype=np.uint8)).save(path, "JPEG")
    return path


def record_of(boxes, size=(200, 100)) -> DetectionRecord:
    return DetectionRecord(boxes=boxes, source_size=size, origin="test")


class DetectorOutputReuseTests(unittest.TestCase):
    """The boxes must come from the pass the detector already made."""

    def test_one_forward_pass_yields_the_winner_and_the_runners_up(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = build_crop(frame, TwoBoxDetector(), CropParams(), collect_detections=True)
        self.assertEqual(len(result.all_detections), 2)
        self.assertEqual(result.detection.score, 0.92)
        self.assertEqual(result.source_size, (200, 100))

    def test_collecting_detections_does_not_change_the_crop(self):
        """The overlay must not be able to alter what the model was shown."""
        frame = (np.random.default_rng(0).random((120, 220, 3)) * 255).astype(np.uint8)
        plain = build_crop(frame, TwoBoxDetector(), CropParams())
        rich = build_crop(frame, TwoBoxDetector(), CropParams(), collect_detections=True)

        self.assertTrue(np.array_equal(plain.crop, rich.crop))
        self.assertEqual(plain.expanded_box, rich.expanded_box)
        self.assertEqual(plain.detection.box, rich.detection.box)
        self.assertEqual(plain.all_detections, [], "default must stay lean")

    def test_records_round_trip_through_the_crop_cache_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.NEF"
            source.write_bytes(b"x")
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            result = build_crop(frame, TwoBoxDetector(), CropParams(), collect_detections=True)

            written = save_detections(root / "cache", source, result)
            self.assertIsNotNone(written)
            self.assertEqual(written, detections_cache_path(root / "cache", source))

            payload = read_detections(root / "cache", source)
            record = _from_payload(payload, "preprocess")
            self.assertEqual(len(record.boxes), 2)
            self.assertIsNotNone(record.selected)
            self.assertEqual(record.selected.score, 0.92)
            self.assertEqual(len(record.others), 1)
            self.assertEqual(record.source_size, (200, 100))

    def test_a_fallback_image_records_that_nothing_was_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.NEF"
            source.write_bytes(b"x")
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            result = build_crop(frame, NothingDetector(), CropParams(), collect_detections=True)
            save_detections(root / "cache", source, result)

            record = _from_payload(read_detections(root / "cache", source), "preprocess")
            self.assertEqual(record.boxes, [])
            self.assertIsNone(record.selected)

    def test_recorded_boxes_are_preferred_over_detecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.NEF"
            source.write_bytes(b"pixels")
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            save_detections(
                root / "cache",
                source,
                build_crop(frame, TwoBoxDetector(), CropParams(), collect_detections=True),
            )

            with DetectionCache(root / "det.db", root / "cache") as cache:
                # allow_detect=False proves nothing was inferred.
                record = cache.get(source, allow_detect=False)
            self.assertEqual(record.origin, "preprocess")
            self.assertEqual(len(record.boxes), 2)

    def test_without_a_record_and_without_permission_nothing_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.NEF"
            source.write_bytes(b"pixels")
            with DetectionCache(root / "det.db", root / "cache") as cache:
                record = cache.get(source, allow_detect=False)
            self.assertEqual(record.boxes, [])
            self.assertEqual(record.origin, "unavailable")

    def test_backfilled_boxes_are_cached_by_content_identity(self):
        """A second run must not re-detect, and a moved file must still hit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.NEF"
            source.write_bytes(b"unique pixels")
            payload = {
                "version": 1,
                "source_size": [200, 100],
                "selected": {"box": [1, 2, 3, 4], "score": 0.8, "label": 16},
                "detections": [{"box": [1, 2, 3, 4], "score": 0.8, "label": 16}],
            }
            with DetectionCache(root / "det.db", root / "no_cache") as cache:
                cache._store(cache._identity(source), payload)
                self.assertEqual(cache.get(source, allow_detect=False).origin, "cache")

                # Same content at a new path: identity matches, so does the cache.
                moved = root / "elsewhere" / "renamed.NEF"
                moved.parent.mkdir()
                moved.write_bytes(b"unique pixels")
                self.assertEqual(cache.get(moved, allow_detect=False).origin, "cache")


class OverlayDrawingTests(unittest.TestCase):
    def test_boxes_are_drawn_onto_the_thumbnail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 100)
            before = np.asarray(Image.open(plain)).copy()

            out = annotate_thumbnail(
                plain,
                record_of([
                    Box(10, 20, 60, 70, 0.9, 16, selected=True),
                    Box(120, 30, 150, 55, 0.4, 24, selected=False),
                ]),
                root / "t_boxes.jpg",
                100,
            )
            self.assertIsNotNone(out)
            after = np.asarray(Image.open(out))
            self.assertFalse(np.array_equal(before, after), "nothing was drawn")

    def test_size_and_aspect_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 120)
            out = annotate_thumbnail(
                plain, record_of([Box(5, 5, 50, 50, 0.9, 16, selected=True)]),
                root / "t_boxes.jpg", 120,
            )
            self.assertEqual(Image.open(out).size, Image.open(plain).size)

    def test_the_selected_box_is_visually_distinct_from_the_others(self):
        """Green for the box that was cropped, amber for the rest - the whole
        diagnostic value is in telling them apart."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 200)
            out = annotate_thumbnail(
                plain,
                record_of(
                    [
                        Box(10, 10, 80, 80, 0.9, 16, selected=True),
                        Box(300, 10, 380, 80, 0.4, 24, selected=False),
                    ],
                    size=(400, 200),
                ),
                root / "t_boxes.jpg",
                200,
            )
            pixels = np.asarray(Image.open(out)).reshape(-1, 3)
            # Green channel dominant somewhere (selected) and a yellow-ish pixel
            # somewhere (runner-up: red and green both high, blue low).
            greenish = ((pixels[:, 1] > 120) & (pixels[:, 0] < 90) & (pixels[:, 2] < 140)).sum()
            yellowish = ((pixels[:, 0] > 150) & (pixels[:, 1] > 150) & (pixels[:, 2] < 110)).sum()
            self.assertGreater(greenish, 10, "no selected-box colour found")
            self.assertGreater(yellowish, 10, "no runner-up colour found")

    def test_no_detection_is_marked_rather_than_left_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 100)
            # A record with boxes but no selected one cannot happen; the
            # not-detected case is signalled by an unselected-only record.
            out = annotate_thumbnail(
                plain, record_of([Box(1, 1, 9, 9, 0.35, 24, selected=False)]),
                root / "t_boxes.jpg", 100,
            )
            pixels = np.asarray(Image.open(out)).reshape(-1, 3)
            reddish = ((pixels[:, 0] > 150) & (pixels[:, 1] < 110) & (pixels[:, 2] < 110)).sum()
            self.assertGreater(reddish, 5, "the no-detection marker is missing")

    def test_an_empty_record_produces_no_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 100)
            self.assertIsNone(annotate_thumbnail(plain, None, root / "o.jpg", 100))
            self.assertIsNone(
                annotate_thumbnail(plain, record_of([], size=None), root / "o.jpg", 100)
            )

    def test_the_overlay_is_a_sibling_file_leaving_the_plain_one_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = _thumbnail_cache_path(root, "/img/a.NEF", 100)
            make_thumb(plain, 100)
            original = plain.read_bytes()
            annotated = annotated_thumbnail_path(root, "/img/a.NEF", 100)
            self.assertNotEqual(annotated, plain)
            self.assertIn("_boxes", annotated.name)

            annotate_thumbnail(plain, record_of([Box(1, 1, 40, 40, 0.9, 16, True)]), annotated, 100)
            self.assertEqual(plain.read_bytes(), original, "the plain thumbnail was modified")


class ScopeTests(unittest.TestCase):
    """Only false negatives get the overlay."""

    def _result(self, root: Path):
        from test_report_links import analyse

        result, config = analyse(root, root / "out")
        # Give every image a detection record so scope, not availability, decides.
        frame_boxes = [
            Box(5, 5, 30, 30, 0.9, 16, selected=True),
            Box(40, 5, 55, 20, 0.4, 24, selected=False),
        ]
        result.detections = {
            record.image_path: record_of(frame_boxes, size=(60, 40))
            for record in result.errors.false_negatives
        }
        return result, config

    def test_only_false_negative_thumbnails_are_annotated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config = self._result(root)
            from picklikeme.analyzer.contactsheets import generate_thumbnails

            everything = [
                image
                for spec_images in (
                    [r.image for r in result.errors.false_negatives],
                    [r.image for r in result.errors.false_positives],
                    result.errors.top_ranked,
                )
                for image in spec_images
            ]
            thumbs = generate_thumbnails(
                everything, config.thumbnail_size, config.thumbnails_dir, root / "no_crops", workers=2
            )
            overlays = build_false_negative_overlays(result, thumbs, result.detections)

            fn_paths = {r.image_path for r in result.errors.false_negatives}
            self.assertTrue(overlays)
            self.assertTrue(set(overlays) <= fn_paths, "a non-false-negative was annotated")

            fp_paths = {r.image_path for r in result.errors.false_positives}
            for path in fp_paths:
                self.assertNotIn(path, overlays)
                self.assertFalse(
                    annotated_thumbnail_path(
                        config.thumbnails_dir, path, config.thumbnail_size
                    ).exists(),
                    "a false positive got an overlay file",
                )

    def test_the_report_uses_the_annotated_thumbnail_for_false_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config = self._result(root)
            render_contact_sheets(result, crop_cache_dir=root / "no_crops")

            from picklikeme.analyzer.reports.html import build_html

            html = build_html(result)
            self.assertIn("_boxes.jpg", html, "the panel is not showing the annotated thumbnail")
            self.assertIn("solid green", html, "the legend explaining the overlay is missing")
            # The overlay is described in the panel metadata too.
            self.assertIn("detector:", html)

    def test_other_sections_still_use_the_plain_thumbnail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config = self._result(root)
            render_contact_sheets(result, crop_cache_dir=root / "no_crops")

            from picklikeme.analyzer.reports.html import _error_table, write_html_report

            write_html_report(result)
            thumbs = {
                r.image_path: _thumbnail_cache_path(
                    config.thumbnails_dir, r.image_path, config.thumbnail_size
                )
                for r in result.errors.false_positives
            }
            table = _error_table(result.errors.false_positives, config.output_dir, thumbs)
            self.assertNotIn("_boxes", table)

    def test_disabling_the_overlay_leaves_the_report_working(self):
        from picklikeme.analyzer.analysis import run_analysis
        from picklikeme.analyzer.config import AnalysisConfig
        from picklikeme.analyzer.reports.html import build_html
        from test_report_links import build_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            result = run_analysis(
                AnalysisConfig(
                    ranking_path=ranking,
                    selected_root=selected,
                    rejected_root=rejected,
                    output_dir=root / "out",
                    charts=False,
                    annotate_detections=False,
                    annotations_db=root / "kb.db",
                )
            )
            self.assertEqual(result.detections, {})
            self.assertIn("<!doctype html>", build_html(result))


class MetricIsolationTests(unittest.TestCase):
    def test_detector_boxes_never_change_a_metric(self):
        """Like annotations, the overlay is diagnosis and must not move numbers."""
        from picklikeme.analyzer.analysis import run_analysis
        from picklikeme.analyzer.config import AnalysisConfig
        from test_report_links import build_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)

            def run(enabled):
                return run_analysis(
                    AnalysisConfig(
                        ranking_path=ranking,
                        selected_root=selected,
                        rejected_root=rejected,
                        output_dir=root / "out",
                        charts=False,
                        contact_sheets=False,
                        annotate_detections=enabled,
                        annotations_db=root / "kb.db",
                        detections_db=root / "det.db",
                    )
                )

            without = run(False)
            with_boxes = run(True)
            self.assertEqual(
                {v.name: v.value for v in without.metrics.values},
                {v.name: v.value for v in with_boxes.metrics.values},
            )
            self.assertEqual(without.confusion.as_dict(), with_boxes.confusion.as_dict())


if __name__ == "__main__":
    unittest.main()
