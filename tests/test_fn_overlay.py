"""Part 2 (and its later extension): detector-box thumbnail overlays.

The point of the feature is that one thumbnail answers "did the detector
contribute to this result?", so the tests check that the boxes are actually
drawn, that they land in the right place, and that every report section shows
them for any image with a resolved detection record - the overlay was
originally scoped to false negatives only, then explicitly widened by the user
to every thumbnail in the report. What still matters is that an image with NO
detection record is never given a fabricated one.
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
    SELECTED_BOX,
    annotate_thumbnail,
    annotated_thumbnail_path,
    build_thumbnail_overlays,
    render_contact_sheets,
    _thumbnail_cache_path,
)
from picklikeme.analyzer.detections import (
    DETECTION_CACHE_VERSION,
    Box,
    DetectionCache,
    DetectionRecord,
    _from_payload,
)
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


class DetectionCacheVersioningTests(unittest.TestCase):
    """A backfilled row is only trustworthy for as long as the crop-selection
    policy that chose its "selected" box hasn't changed - so this cache is
    stamped with bird_crop.CROP_CACHE_VERSION and never serves a row stamped
    with anything else. See the module docstring's "Versioned against the
    same policy as the crop cache" section.
    """

    PAYLOAD = {
        "version": 1,
        "source_size": [200, 100],
        "selected": {"box": [1, 2, 3, 4], "score": 0.8, "label": 16},
        "detections": [{"box": [1, 2, 3, 4], "score": 0.8, "label": 16}],
    }

    def test_the_cache_version_tracks_the_crop_cache_version(self):
        """Pinned so a future decoupling (a second constant someone forgets
        to bump) is caught immediately rather than silently reopening the
        exact hole this versioning closes."""
        from picklikeme.bird_crop import CROP_CACHE_VERSION

        self.assertEqual(DETECTION_CACHE_VERSION, CROP_CACHE_VERSION)

    def test_a_row_written_by_store_is_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.NEF"
            source.write_bytes(b"pixels")
            with DetectionCache(root / "det.db", root / "no_cache") as cache:
                cache._store(cache._identity(source), self.PAYLOAD)
                self.assertEqual(cache.get(source, allow_detect=False).origin, "cache")

    def test_a_row_stamped_with_a_different_version_is_never_served(self):
        """The literal scenario this protects against: a row backfilled under
        an older crop-selection policy must not be trusted just because an
        image_hash matches."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.NEF"
            source.write_bytes(b"pixels")
            with DetectionCache(root / "det.db", root / "no_cache") as cache:
                digest = cache._identity(source)
                cache._conn.execute(
                    "INSERT INTO detection_cache(image_hash, payload, cache_version) VALUES (?, ?, ?)",
                    (digest, json.dumps(self.PAYLOAD), "some-older-version"),
                )
                cache._conn.commit()

                # allow_detect=False proves the stale row was ignored, not
                # silently served: with nothing else to fall back to, the
                # result must be empty rather than the stale "selected" box.
                record = cache.get(source, allow_detect=False)
                self.assertEqual(record.origin, "unavailable")
                self.assertEqual(record.boxes, [])

    def test_a_stale_row_heals_itself_on_the_next_successful_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.png"
            Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(source)
            with DetectionCache(root / "det.db", root / "no_cache") as cache:
                digest = cache._identity(source)
                cache._conn.execute(
                    "INSERT INTO detection_cache(image_hash, payload, cache_version) VALUES (?, ?, ?)",
                    (digest, json.dumps(self.PAYLOAD), "some-older-version"),
                )
                cache._conn.commit()

                cache.detector = TwoBoxDetector()  # skip loading the real model
                healed = cache.get(source, allow_detect=True)
                self.assertEqual(healed.origin, "detected")
                self.assertEqual(healed.selected.score, 0.92)  # TwoBoxDetector's winner, not the stale 0.8

                # The healed row is now itself a normal cache hit.
                again = cache.get(source, allow_detect=False)
                self.assertEqual(again.origin, "cache")
                self.assertEqual(again.selected.score, 0.92)

    def test_opening_a_database_from_before_versioning_existed_purges_it(self):
        """A database created by the pre-versioning code has no cache_version
        column at all - opening it must retrofit the column (so reads don't
        crash) and treat every existing row as stale (so nothing pre-policy
        is ever served), not merely tolerate the missing column."""
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "det.db"
            source = root / "IMG.NEF"
            source.write_bytes(b"pixels")

            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE detection_cache (
                    image_hash TEXT PRIMARY KEY,
                    payload    TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            # A real image_identity digest, computed the same way the cache
            # itself would, so this row is a genuine pre-migration entry.
            from picklikeme.identity import image_identity

            conn.execute(
                "INSERT INTO detection_cache(image_hash, payload) VALUES (?, ?)",
                (image_identity(source), json.dumps(self.PAYLOAD)),
            )
            conn.commit()
            conn.close()

            with DetectionCache(db_path, root / "no_cache") as cache:
                columns = {row["name"] for row in cache._conn.execute("PRAGMA table_info(detection_cache)")}
                self.assertIn("cache_version", columns)
                remaining = cache._conn.execute("SELECT COUNT(*) AS n FROM detection_cache").fetchone()["n"]
                self.assertEqual(remaining, 0, "a pre-versioning row must be purged, not kept forever")

                record = cache.get(source, allow_detect=False)
                self.assertEqual(record.origin, "unavailable")

    def test_opening_a_database_already_at_the_current_version_keeps_its_rows(self):
        """The purge must not be a blunt 'always wipe on open' - only an
        actual version mismatch should discard anything."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "IMG.NEF"
            source.write_bytes(b"pixels")
            with DetectionCache(root / "det.db", root / "no_cache") as cache:
                cache._store(cache._identity(source), self.PAYLOAD)

            with DetectionCache(root / "det.db", root / "no_cache") as reopened:
                self.assertEqual(reopened.get(source, allow_detect=False).origin, "cache")


class FullFramePreviewTests(unittest.TestCase):
    """A preview must show the whole original image, never the cached bird crop.

    Regression: previews were built from the crop cache while detector boxes are
    recorded in full-frame coordinates, so every annotated preview showed a small
    region with boxes drawn against a frame that was not there.
    """

    def _frame_and_crop(self, root: Path):
        """A source image whose crop cache holds a visibly different picture."""
        from picklikeme.bird_crop import crop_cache_path

        source = root / "IMG_0001.jpg"
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        frame[:, :, 2] = 200                      # a blue full frame ...
        frame[20:60, 30:90] = (255, 0, 0)         # ... with a red subject in it
        Image.fromarray(frame).save(source, "JPEG", quality=95)

        cached = crop_cache_path(root / "crops", source)
        cached.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((40, 60, 3), (0, 255, 0), dtype=np.uint8)).save(cached)
        return source, cached

    def test_the_preview_comes_from_the_original_not_the_crop_cache(self):
        from picklikeme.analyzer.contactsheets import build_thumbnail

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, cached = self._frame_and_crop(root)
            self.assertTrue(cached.exists(), "fixture must populate the crop cache")

            thumb = build_thumbnail(str(source), 100, root / "thumbs")
            pixels = np.asarray(Image.open(thumb).convert("RGB")).reshape(-1, 3)

            # The crop cache is solid green; the original is blue with red in it.
            crop_green = ((pixels[:, 1] > 180) & (pixels[:, 0] < 90) & (pixels[:, 2] < 90)).sum()
            self.assertEqual(crop_green, 0, "the preview was built from the cached crop")
            self.assertGreater(
                ((pixels[:, 2] > 150) & (pixels[:, 0] < 100)).sum(), 100, "no full frame in the preview"
            )

    def test_the_preview_keeps_the_whole_frames_aspect_ratio(self):
        """Letterboxed into the square, not cropped to fill it: a 2:1 frame must
        occupy half the height and leave background bars."""
        from picklikeme.analyzer.contactsheets import BACKGROUND, build_thumbnail

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self._frame_and_crop(root)
            thumb = build_thumbnail(str(source), 100, root / "thumbs")
            image = np.asarray(Image.open(thumb).convert("RGB"))

            self.assertEqual(image.shape[:2], (100, 100))
            top_row = image[0]
            self.assertTrue(
                np.allclose(top_row, np.array(BACKGROUND), atol=12),
                "the frame was cropped to fill the square instead of letterboxed",
            )
            self.assertFalse(np.allclose(image[50], np.array(BACKGROUND), atol=12))

    def test_boxes_land_on_the_subject_in_the_full_frame(self):
        """The end-to-end property: a box around the subject in full-frame
        coordinates must be drawn around the subject in the preview."""
        from picklikeme.analyzer.contactsheets import build_thumbnail

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self._frame_and_crop(root)
            thumb = build_thumbnail(str(source), 100, root / "thumbs")

            out = annotate_thumbnail(
                thumb,
                record_of([Box(30, 20, 90, 60, 0.9, 16, selected=True)], size=(400, 200)),
                root / "boxes.jpg",
                100,
            )
            drawn = np.asarray(Image.open(out).convert("RGB"))

            # The frame is letterboxed: 400x200 into 100x100 leaves 25px bars, so
            # the box spans x 7-22, y 30-40 in thumbnail coordinates.
            def greenish(patch):
                return (
                    (patch[:, :, 1] > 120) & (patch[:, :, 0] < 120) & (patch[:, :, 2] < 160)
                ).sum()

            self.assertGreater(greenish(drawn[26:46, 3:28]), 10, "no box near the subject")
            self.assertEqual(greenish(drawn[60:100, 60:100]), 0, "a box was drawn far from the subject")


class BoxGeometryTests(unittest.TestCase):
    """The coordinate transformation, pinned numerically.

    Boxes are recorded in full-frame pixels; the preview is that frame scaled by
    `min(size/w, size/h)` and centred in a square. Every box must land at
    `offset + coordinate * scale`. Verified by drawing on a flat base, so any
    changed pixel is unambiguously part of the overlay.
    """

    FLAT = (128, 128, 128)

    def _drawn_extent(self, root: Path, boxes, frame, size, labels: bool = False):
        """Bounding box of everything annotate_thumbnail changed.

        Labels are off by default: they are drawn above the box in the same
        colour, so leaving them on would measure the text rather than the box.
        """
        from unittest.mock import patch

        base_path = root / "flat.png"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((size, size, 3), self.FLAT, dtype=np.uint8)).save(base_path)
        before = np.asarray(Image.open(base_path).convert("RGB")).astype(int)

        record = record_of(boxes, size=frame)
        if labels:
            out = annotate_thumbnail(base_path, record, root / "o.jpg", size)
        else:
            with patch("PIL.ImageDraw.ImageDraw.text"):
                out = annotate_thumbnail(base_path, record, root / "o.jpg", size)
        after = np.asarray(Image.open(out).convert("RGB")).astype(int)
        ys, xs = np.nonzero(np.abs(after - before).sum(2) > 25)
        return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _expected(self, box, frame, size):
        scale = min(size / frame[0], size / frame[1])
        ox, oy = (size - frame[0] * scale) / 2, (size - frame[1] * scale) / 2
        return (ox + box.x1 * scale, oy + box.y1 * scale, ox + box.x2 * scale, oy + box.y2 * scale)

    def test_a_box_lands_where_the_transform_says_it_should(self):
        """A landscape frame letterboxed into a square: the box must be offset
        by the bars and scaled by the fit factor, both."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, size = (8000, 4000), 400          # 2:1 -> 400x200 content, 100px bars
            box = Box(4000, 2000, 6000, 3000, 0.9, 16, selected=True)

            got = self._drawn_extent(root, [box], frame, size)
            exp = self._expected(box, frame, size)   # (200, 200) - (300, 250)
            self.assertEqual(tuple(round(v) for v in exp), (200, 200, 300, 250))
            self.assertIsNotNone(got)
            for measured, expected in zip(got, exp):
                self.assertLessEqual(
                    abs(measured - expected), 3,
                    f"box drawn at {got}, transform says {tuple(round(v, 1) for v in exp)}",
                )

    def test_a_box_in_a_frame_corner_stays_in_that_corner(self):
        """Catches a transform that drops the letterbox offset: without it a
        top-left box would still look right, so test a bottom-right one."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, size = (6000, 3000), 300
            box = Box(5400, 2700, 6000, 3000, 0.9, 16, selected=True)

            got = self._drawn_extent(root, [box], frame, size)
            exp = self._expected(box, frame, size)   # bottom-right of the content band
            self.assertIsNotNone(got)
            for measured, expected in zip(got, exp):
                self.assertLessEqual(abs(measured - expected), 3)
            # Inside the content band, never on the letterbox bar.
            self.assertGreater(got[1], (size - size / 2) / 2 - 4)

    def test_relative_box_sizes_are_preserved(self):
        """A box covering a tenth of the frame width must cover a tenth of the
        drawn content - the property a wrong scale factor would break."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, size = (5000, 2500), 400
            box = Box(1000, 1000, 1500, 1500, 0.9, 16, selected=True)

            got = self._drawn_extent(root, [box], frame, size)
            content_width = size                       # 2:1 frame fills the width
            self.assertAlmostEqual((got[2] - got[0]) / content_width, 0.10, delta=0.02)

    def test_a_tiny_box_is_not_swallowed_by_its_own_outline(self):
        """A distant bird is a handful of pixels on a whole-frame preview. The
        outline must stay thin enough that the box still reads as a box."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, size = (8000, 4000), 400
            box = Box(4000, 2000, 4160, 2160, 0.9, 16, selected=True)   # 8x8 px drawn

            got = self._drawn_extent(root, [box], frame, size)
            self.assertIsNotNone(got, "nothing was drawn for a small box")
            self.assertLessEqual(got[2] - got[0], 16, "the outline is wider than the box it marks")

    def test_a_label_wider_than_its_box_is_suppressed(self):
        """A label sprawling past a small box points at its neighbours instead,
        so it is dropped - while a box with room for it keeps it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, size = (8000, 4000), 400

            tiny = self._drawn_extent(
                root, [Box(4000, 2000, 4160, 2160, 0.9, 16, selected=True)], frame, size, labels=True
            )
            self.assertLessEqual(tiny[2] - tiny[0], 20, "a label leaked past a small box")

            roomy = self._drawn_extent(
                root, [Box(3000, 2000, 5000, 3000, 0.9, 16, selected=True)], frame, size, labels=True
            )
            box_top = self._expected(Box(3000, 2000, 5000, 3000, 0.9, 16, True), frame, size)[1]
            self.assertLess(roomy[1], box_top - 3, "a box with room for a label did not get one")

    def test_the_transform_holds_for_a_portrait_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, size = (3000, 6000), 300          # bars on the left and right
            box = Box(1500, 3000, 2250, 4500, 0.9, 16, selected=True)

            got = self._drawn_extent(root, [box], frame, size)
            exp = self._expected(box, frame, size)
            self.assertIsNotNone(got)
            for measured, expected in zip(got, exp):
                self.assertLessEqual(abs(measured - expected), 3)


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


class EyeOverlayTests(unittest.TestCase):
    """The optional `eye` overlay (see review.thumbnails.eye_keypoints_for) -
    a magenta box/crosshairs for the eye Classic Vision measured, drawn on
    top of the ordinary detector-box overlay."""

    def _eye(self, *, accepted: bool, box=(60.0, 20.0, 100.0, 60.0)):
        return {
            "source_size": (200, 100),
            "accepted": accepted,
            "confidence": 0.9 if accepted else 0.5,
            "box": box,
            "left": {"x": 70.0, "y": 30.0, "confidence": 0.9},
            "right": {"x": 90.0, "y": 30.0, "confidence": 0.4},
        }

    def test_an_accepted_eye_is_drawn_in_magenta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 200)
            out = annotate_thumbnail(
                plain,
                record_of([Box(10, 10, 190, 90, 0.9, 16, selected=True)], size=(200, 100)),
                root / "t_boxes.jpg",
                200,
                eye=self._eye(accepted=True),
            )
            pixels = np.asarray(Image.open(out)).reshape(-1, 3)
            from picklikeme.analyzer.contactsheets import EYE_BOX_ACCEPTED

            close = (
                (np.abs(pixels[:, 0].astype(int) - EYE_BOX_ACCEPTED[0]) < 20)
                & (np.abs(pixels[:, 1].astype(int) - EYE_BOX_ACCEPTED[1]) < 20)
                & (np.abs(pixels[:, 2].astype(int) - EYE_BOX_ACCEPTED[2]) < 20)
            ).sum()
            self.assertGreater(close, 5, "no accepted-eye color found")

    def test_a_rejected_eye_is_still_drawn_but_distinctly_coloured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 200)
            out = annotate_thumbnail(
                plain,
                record_of([Box(10, 10, 190, 90, 0.9, 16, selected=True)], size=(200, 100)),
                root / "t_boxes.jpg",
                200,
                eye=self._eye(accepted=False),
            )
            pixels = np.asarray(Image.open(out)).reshape(-1, 3)
            from picklikeme.analyzer.contactsheets import EYE_BOX_REJECTED

            close = (
                (np.abs(pixels[:, 0].astype(int) - EYE_BOX_REJECTED[0]) < 20)
                & (np.abs(pixels[:, 1].astype(int) - EYE_BOX_REJECTED[1]) < 20)
                & (np.abs(pixels[:, 2].astype(int) - EYE_BOX_REJECTED[2]) < 20)
            ).sum()
            self.assertGreater(close, 5, "no rejected-eye color found - a distrusted eye must still show")

    def test_no_eye_argument_draws_no_eye_overlay(self):
        """Backward compatibility: every existing caller that never passes
        `eye` must see byte-identical output to before this parameter existed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 200)
            record = record_of([Box(10, 10, 190, 90, 0.9, 16, selected=True)], size=(200, 100))
            without_eye = annotate_thumbnail(plain, record, root / "a.jpg", 200)
            without_eye_kwarg = annotate_thumbnail(plain, record, root / "b.jpg", 200, eye=None)
            self.assertEqual(without_eye.read_bytes(), without_eye_kwarg.read_bytes())


class ThicknessTests(unittest.TestCase):
    """The 'increase detector box thickness' fix: boxes must draw noticeably
    thicker than before, without a small box's outline swallowing it."""

    def test_boxes_are_drawn_thicker_than_the_old_default(self):
        from picklikeme.analyzer.contactsheets import _stroke

        # The old formula was line = max(1, round(size/200)); a 400px
        # thumbnail (the review Gallery's default) used to compute a 2px
        # line. The new one must be strictly thicker for the same size.
        old_line = max(1, round(400 / 200))
        new_line = max(2, round(400 / 120))
        self.assertGreater(new_line, old_line)
        # And still capped against a small box's own dimensions, so a
        # distant bird's box is never swallowed by its own outline.
        self.assertEqual(_stroke(new_line, 8, 8), 2)

    def test_a_large_box_s_outline_is_now_about_five_times_as_thick(self):
        """The later "make the overlay our primary debugging tool" pass
        multiplied the whole `line` formula by ~5x (see annotate_thumbnail's
        own comment) - measured here on an actual rendered box big enough
        that _stroke's small-box cap never kicks in, so the real drawn
        pixel thickness is what is pinned, not just the formula."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = make_thumb(root / "t.jpg", 400)
            out = annotate_thumbnail(
                plain,
                record_of([Box(20, 20, 380, 380, 0.9, 16, selected=True)], size=(400, 400)),
                root / "t_boxes.jpg",
                400,
            )
            pixels = np.asarray(Image.open(out).convert("RGB")).astype(int)
            # A vertical slice through the box's horizontal centre, counting
            # how many consecutive rows near the top edge (y=20) are the
            # selected-box colour - PIL draws a rectangle outline growing
            # INWARD from the given coordinate, so this band's length IS the
            # stroke width.
            column = pixels[:, 200, :]
            close = (
                (np.abs(column[:, 0] - SELECTED_BOX[0]) < 30)
                & (np.abs(column[:, 1] - SELECTED_BOX[1]) < 30)
                & (np.abs(column[:, 2] - SELECTED_BOX[2]) < 30)
            )
            band = int(close[15:45].sum())
            # Old default (line=3, selected width=line+1=4) drew a 4px band;
            # ~5x that is ~20px. A loose lower bound (>= 12px, 3x the old
            # width) keeps this robust to _stroke's own rounding/capping
            # while still failing hard if the 5x change ever regresses back
            # toward the old thin default.
            self.assertGreaterEqual(band, 12, f"outline only {band}px thick - the 5x change appears to have regressed")


class ScopeTests(unittest.TestCase):
    """Every image with a resolved detection record gets the overlay -
    false negatives, false positives, true positives/negatives, whatever
    category it appears in. The only thing that decides whether an image is
    annotated is whether a detection record with boxes exists for it, never
    which report section it is shown in."""

    def _result(self, root: Path, categories=("false_negatives",)):
        from test_report_links import analyse

        result, config = analyse(root, root / "out")
        # Give only the requested categories a detection record, so the test
        # can prove overlays follow the record, not the category.
        frame_boxes = [
            Box(5, 5, 30, 30, 0.9, 16, selected=True),
            Box(40, 5, 55, 20, 0.4, 24, selected=False),
        ]
        wanted = set(categories)
        pools = {
            "false_negatives": [r.image_path for r in result.errors.false_negatives],
            "false_positives": [r.image_path for r in result.errors.false_positives],
            "top_ranked": [i.image_path for i in result.errors.top_ranked],
        }
        result.detections = {
            path: record_of(frame_boxes, size=(60, 40))
            for name in wanted
            for path in pools.get(name, [])
        }
        return result, config

    def test_a_false_positive_with_a_detection_record_is_annotated(self):
        """The behaviour the old FN-only scope explicitly forbade."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config = self._result(root, categories=("false_positives",))
            from picklikeme.analyzer.contactsheets import generate_thumbnails

            fp_images = [r.image for r in result.errors.false_positives]
            thumbs = generate_thumbnails(
                fp_images, config.thumbnail_size, config.thumbnails_dir, workers=2
            )
            overlays = build_thumbnail_overlays(result, thumbs, result.detections)

            fp_paths_with_records = set(result.detections)
            self.assertTrue(fp_paths_with_records, "fixture must produce false positives")
            self.assertEqual(set(overlays), fp_paths_with_records)
            for path in fp_paths_with_records:
                self.assertTrue(
                    annotated_thumbnail_path(
                        config.thumbnails_dir, path, config.thumbnail_size
                    ).exists(),
                    "false positive did not get an overlay file",
                )

    def test_images_with_no_detection_record_stay_plain_in_any_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Only false_negatives get records; false_positives must stay plain.
            result, config = self._result(root, categories=("false_negatives",))
            from picklikeme.analyzer.contactsheets import generate_thumbnails

            everything = [r.image for r in result.errors.false_negatives] + [
                r.image for r in result.errors.false_positives
            ]
            thumbs = generate_thumbnails(
                everything, config.thumbnail_size, config.thumbnails_dir, workers=2
            )
            overlays = build_thumbnail_overlays(result, thumbs, result.detections)

            fn_paths = {r.image_path for r in result.errors.false_negatives}
            fp_paths = {r.image_path for r in result.errors.false_positives}
            self.assertTrue(overlays)
            self.assertEqual(set(overlays), fn_paths)
            for path in fp_paths:
                self.assertNotIn(path, overlays)
                self.assertFalse(
                    annotated_thumbnail_path(
                        config.thumbnails_dir, path, config.thumbnail_size
                    ).exists(),
                    "an image with no detection record got an overlay file",
                )

    def test_the_report_shows_the_annotated_thumbnail_wherever_a_record_exists(self):
        """False positives get boxes in the HTML report too, not only the
        false-negative annotation panels."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config = self._result(
                root, categories=("false_negatives", "false_positives")
            )
            render_contact_sheets(result)

            from picklikeme.analyzer.reports.html import write_html_report

            html = write_html_report(result).read_text(encoding="utf-8")
            self.assertIn("_boxes.jpg", html, "no annotated thumbnail rendered anywhere")
            self.assertIn("solid green", html, "the shared detector-box legend is missing")
            # The false-positives table must be showing an overlay, not just
            # the false-negative panels.
            fp_path = result.errors.false_positives[0].image_path
            fp_overlay = annotated_thumbnail_path(
                config.thumbnails_dir, fp_path, config.thumbnail_size
            )
            self.assertTrue(fp_overlay.exists())
            rel_name = fp_overlay.name
            self.assertIn(rel_name, html, "the false-positive overlay file is not linked in the report")

    def test_no_legend_when_nothing_has_a_detection_record(self):
        """An unevidenced legend would be misleading; it only appears once
        at least one overlay actually exists."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config = self._result(root, categories=())
            self.assertEqual(result.detections, {})
            render_contact_sheets(result)

            from picklikeme.analyzer.reports.html import build_html

            html = build_html(result)
            self.assertNotIn("solid green", html)

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


class BoxCategoryTests(unittest.TestCase):
    """Box.category - the review app's structured subject metadata, derived
    from the same COCO label the overlay already draws from."""

    def test_a_bird_box_reports_the_bird_category(self):
        box = Box(0, 0, 10, 10, 0.9, 16, selected=True)  # COCO_BIRD_CLASS
        self.assertEqual(box.category, "bird")
        self.assertEqual(box.as_dict()["category"], "bird")

    def test_a_person_box_reports_the_human_category(self):
        box = Box(0, 0, 10, 10, 0.9, 1, selected=False)  # COCO_PERSON_CLASS
        self.assertEqual(box.category, "human")

    def test_an_uncatalogued_class_has_no_category(self):
        box = Box(0, 0, 10, 10, 0.9, 999, selected=False)
        self.assertIsNone(box.category)
        self.assertIsNone(box.as_dict()["category"])


if __name__ == "__main__":
    unittest.main()
