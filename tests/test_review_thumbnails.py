"""review/thumbnails.py: the review app's own caches (thumbnail overlay,
preview, category lookup) built on top of the analyzer's shared detection
cache. Box/detection_category themselves are tested directly in
test_bird_crop.py and test_fn_overlay.py; this covers the thin wiring that
turns a detection record into detected_category_for()'s answer, plus the
preview cache's own size-budget/LRU eviction.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.review import thumbnails as thumbnails_module
from picklikeme.review.thumbnails import (
    _cache_entries,
    _enforce_cache_budget,
    detected_category_for,
    detection_boxes_for,
    review_preview,
)


class DetectedCategoryForTests(unittest.TestCase):
    def test_reads_the_selected_detection_s_category(self):
        record = mock.Mock(selected=mock.Mock(category="bird"))
        with mock.patch("picklikeme.review.thumbnails._detections") as detections:
            detections.return_value.get.return_value = record
            self.assertEqual(detected_category_for("some/path.jpg"), "bird")
            detections.return_value.get.assert_called_once_with("some/path.jpg", allow_detect=False)

    def test_no_selected_detection_means_no_category(self):
        """A recorded box that lost the crop-selection (a runner-up) must
        never be mistaken for the subject - only .selected counts."""
        record = mock.Mock(selected=None)
        with mock.patch("picklikeme.review.thumbnails._detections") as detections:
            detections.return_value.get.return_value = record
            self.assertIsNone(detected_category_for("some/path.jpg"))

    def test_an_uncatalogued_selected_detection_has_no_category(self):
        record = mock.Mock(selected=mock.Mock(category=None))
        with mock.patch("picklikeme.review.thumbnails._detections") as detections:
            detections.return_value.get.return_value = record
            self.assertIsNone(detected_category_for("some/path.jpg"))

    def test_an_unreadable_cache_is_not_fatal(self):
        """A bad detection cache must not break loading the gallery, exactly
        like review_thumbnail's own overlay lookup."""
        with mock.patch("picklikeme.review.thumbnails._detections", side_effect=RuntimeError("boom")):
            self.assertIsNone(detected_category_for("some/path.jpg"))

    def test_never_runs_the_detector_itself(self):
        """Review must only ever read what preprocessing (or an earlier
        backfill) already computed - allow_detect=False is not optional."""
        with mock.patch("picklikeme.review.thumbnails._detections") as detections:
            detections.return_value.get.return_value = mock.Mock(selected=None)
            detected_category_for("some/path.jpg")
            _, kwargs = detections.return_value.get.call_args
            self.assertFalse(kwargs.get("allow_detect", True))


class DetectionBoxesForTests(unittest.TestCase):
    """detection_boxes_for: the Image Inspector's "Show Boxes" overlay reads
    this for its Detection Box / Expanded Crop rectangles (Eye ROI comes
    from eye_keypoints_for instead - see EyeKeypointsForTests)."""

    def test_includes_the_expanded_crop_box_alongside_the_tight_detection(self):
        """Regression: the Image Inspector's overlay needs BOTH the tight
        detection box and the margin-grown crop rectangle actually cached
        (see DetectionRecord.expanded_box) to draw them as two distinct
        rectangles, not merged into one - this used to be missing entirely,
        only `selected`/`others` were exposed."""
        selected = mock.Mock()
        selected.as_dict.return_value = {"box": (10.0, 10.0, 20.0, 20.0)}
        record = mock.Mock(
            source_size=(100, 100), selected=selected, others=[], expanded_box=(5.0, 5.0, 25.0, 25.0),
        )
        with mock.patch("picklikeme.review.thumbnails._detections") as detections:
            detections.return_value.get.return_value = record
            result = detection_boxes_for("some/path.jpg")

        self.assertEqual(result["expanded_box"], [5.0, 5.0, 25.0, 25.0])
        self.assertEqual(result["selected"], {"box": (10.0, 10.0, 20.0, 20.0)})

    def test_no_expanded_box_on_a_full_frame_fallback_is_none_not_a_crash(self):
        record = mock.Mock(source_size=(100, 100), selected=None, others=[], expanded_box=None)
        with mock.patch("picklikeme.review.thumbnails._detections") as detections:
            detections.return_value.get.return_value = record
            result = detection_boxes_for("some/path.jpg")

        self.assertIsNone(result["expanded_box"])


class EyeKeypointsForTests(unittest.TestCase):
    """review_thumbnails.eye_keypoints_for: rescaling a Classic Vision run's
    cached eye result from the subject crop's own pixel space onto full-frame
    coordinates - the same read-only wiring detected_category_for uses."""

    def test_no_cached_eye_record_means_no_overlay(self):
        """No eye record at all - Classic Vision has never run on this image."""
        with mock.patch("picklikeme.eyes.cache.read_eye_detection", return_value=None):
            self.assertIsNone(thumbnails_module.eye_keypoints_for("some/path.jpg"))

    def test_no_recorded_subject_means_no_overlay(self):
        """An eye was cached, but there is no subject box to map it against -
        a stale or partial cache must not raise."""
        from picklikeme.eyes.cache import EyeRecord

        record = EyeRecord(
            detector_id="fake", subject_crop_size=(50, 50), accepted=True,
            box=(10.0, 10.0, 20.0, 20.0), confidence=0.9, left=None, right=None,
        )
        with mock.patch("picklikeme.eyes.cache.read_eye_detection", return_value=record):
            with mock.patch("picklikeme.review.thumbnails._detections") as detections:
                detections.return_value.get.return_value = mock.Mock(
                    selected=None, source_size=(800, 600), expanded_box=None,
                )
                self.assertIsNone(thumbnails_module.eye_keypoints_for("some/path.jpg"))

    def test_no_expanded_box_means_no_overlay(self):
        """A record with a selected subject but no recorded expanded_box (a
        pre-v6 cache entry) must not fall back to the tight `selected` box -
        that would silently reintroduce the Q1 projection bug (see
        docs/EyePose_Investigation_Phase_1.md)."""
        from picklikeme.eyes.cache import EyeRecord

        record = EyeRecord(
            detector_id="fake", subject_crop_size=(50, 50), accepted=True,
            box=(10.0, 10.0, 20.0, 20.0), confidence=0.9, left=None, right=None,
        )
        with mock.patch("picklikeme.eyes.cache.read_eye_detection", return_value=record):
            with mock.patch("picklikeme.review.thumbnails._detections") as detections:
                detections.return_value.get.return_value = mock.Mock(
                    selected=mock.Mock(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
                    source_size=(800, 600),
                    expanded_box=None,
                )
                self.assertIsNone(thumbnails_module.eye_keypoints_for("some/path.jpg"))

    def test_rescales_the_box_and_keypoints_from_crop_space_to_frame_space(self):
        """A 100x100 subject crop mapped onto a 200x200 full-frame box at
        (400, 300) - a plain 2x scale plus an offset, the same trick
        contactsheets.annotate_thumbnail uses for the subject box itself.
        `expanded_box` (not `selected`) is what actually drives the
        scale/offset - see docs/EyePose_Investigation_Phase_1.md's Q1."""
        from picklikeme.eyes.cache import EyeRecord
        from picklikeme.eyes.detector import EyeKeypoint

        record = EyeRecord(
            detector_id="fake", subject_crop_size=(100, 100), accepted=True,
            box=(10.0, 20.0, 30.0, 40.0), confidence=0.95,
            left=EyeKeypoint(x=15.0, y=25.0, confidence=0.9),
            right=EyeKeypoint(x=18.0, y=22.0, confidence=0.4),
            beak=EyeKeypoint(x=20.0, y=30.0, confidence=0.8),
            head_top=EyeKeypoint(x=20.0, y=10.0, confidence=0.7),
        )
        # Deliberately different from expanded_box, so a test that still used
        # `selected` for the projection would fail this - proving the fix.
        subject = mock.Mock(x1=411.0, y1=322.0, x2=633.0, y2=544.0)
        with mock.patch("picklikeme.eyes.cache.read_eye_detection", return_value=record):
            with mock.patch("picklikeme.review.thumbnails._detections") as detections:
                detections.return_value.get.return_value = mock.Mock(
                    selected=subject, source_size=(1920, 1080), expanded_box=(400.0, 300.0, 600.0, 500.0),
                )
                result = thumbnails_module.eye_keypoints_for("some/path.jpg")

        self.assertEqual(result["source_size"], (1920, 1080))
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["confidence"], 0.95)
        self.assertEqual(result["box"], (420.0, 340.0, 460.0, 380.0))
        self.assertAlmostEqual(result["left"]["x"], 430.0)
        self.assertAlmostEqual(result["left"]["y"], 350.0)
        self.assertAlmostEqual(result["right"]["x"], 436.0)
        self.assertAlmostEqual(result["right"]["y"], 344.0)
        # The other four landmarks go through the exact same to_frame()
        # transform as left/right - beak/head_top were supplied here, the
        # two shoulders were not, and must come back as None rather than a
        # fabricated (0, 0) point.
        self.assertAlmostEqual(result["beak"]["x"], 440.0)
        self.assertAlmostEqual(result["beak"]["y"], 360.0)
        self.assertAlmostEqual(result["head_top"]["x"], 440.0)
        self.assertAlmostEqual(result["head_top"]["y"], 320.0)
        self.assertIsNone(result["left_shoulder"])
        self.assertIsNone(result["right_shoulder"])

    def test_never_runs_the_eye_detector_or_subject_detector_itself(self):
        """Review only ever reads what an earlier Classic Vision run already
        computed - allow_detect=False is not optional, the same rule
        detected_category_for follows."""
        from picklikeme.eyes.cache import EyeRecord

        record = EyeRecord(
            detector_id="fake", subject_crop_size=(10, 10), accepted=True,
            box=(0.0, 0.0, 5.0, 5.0), confidence=0.9, left=None, right=None,
        )
        with mock.patch("picklikeme.eyes.cache.read_eye_detection", return_value=record):
            with mock.patch("picklikeme.review.thumbnails._detections") as detections:
                detections.return_value.get.return_value = mock.Mock(
                    selected=mock.Mock(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
                    source_size=(100, 100),
                    expanded_box=(0.0, 0.0, 10.0, 10.0),
                )
                thumbnails_module.eye_keypoints_for("some/path.jpg")
                _, kwargs = detections.return_value.get.call_args
                self.assertFalse(kwargs.get("allow_detect", True))


def _write_fake_cache_file(cache_dir: Path, name: str, size_bytes: int, age_seconds: float) -> Path:
    """A file inside the cache's own sharded layout (cache_dir/xx/name.jpg),
    with a controlled size and mtime - the two things LRU eviction reads,
    without needing a real JPEG or a real detector for these tests."""
    shard = cache_dir / name[:2]
    shard.mkdir(parents=True, exist_ok=True)
    target = shard / f"{name}.jpg"
    target.write_bytes(b"0" * size_bytes)
    when = time.time() - age_seconds
    os.utime(target, (when, when))
    return target


class CacheBudgetTests(unittest.TestCase):
    """The preview cache's own size accounting and least-recently-used
    eviction - see _enforce_cache_budget. Independent of review_preview
    itself, which just calls this at the right moments."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cache_dir = Path(self._tmp.name) / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def test_cache_entries_finds_every_sharded_file_with_size_and_mtime(self):
        _write_fake_cache_file(self.cache_dir, "aaa1", size_bytes=100, age_seconds=10)
        _write_fake_cache_file(self.cache_dir, "bbb2", size_bytes=200, age_seconds=5)

        entries = _cache_entries(self.cache_dir)

        self.assertEqual(len(entries), 2)
        self.assertEqual(sorted(size for _, size, _ in entries), [100, 200])

    def test_a_nonexistent_cache_dir_is_simply_empty(self):
        self.assertEqual(_cache_entries(self.cache_dir), [])

    def test_does_nothing_when_already_under_budget(self):
        path = _write_fake_cache_file(self.cache_dir, "aaa1", size_bytes=100, age_seconds=1)

        removed = _enforce_cache_budget(self.cache_dir, max_bytes=1_000_000)

        self.assertEqual(removed, 0)
        self.assertTrue(path.exists())

    def test_evicts_the_least_recently_used_files_first(self):
        oldest = _write_fake_cache_file(self.cache_dir, "aaa1", size_bytes=100, age_seconds=100)
        middle = _write_fake_cache_file(self.cache_dir, "bbb2", size_bytes=100, age_seconds=50)
        newest = _write_fake_cache_file(self.cache_dir, "ccc3", size_bytes=100, age_seconds=1)

        # Budget for only one file's worth - two of the three must go.
        removed = _enforce_cache_budget(self.cache_dir, max_bytes=100)

        self.assertEqual(removed, 2)
        self.assertFalse(oldest.exists(), "least recently used must go first")
        self.assertFalse(middle.exists())
        self.assertTrue(newest.exists(), "most recently used survives")

    def test_stops_as_soon_as_it_is_back_under_budget(self):
        _write_fake_cache_file(self.cache_dir, "aaa1", size_bytes=100, age_seconds=100)
        newer = _write_fake_cache_file(self.cache_dir, "bbb2", size_bytes=100, age_seconds=1)

        removed = _enforce_cache_budget(self.cache_dir, max_bytes=100)

        self.assertEqual(removed, 1, "only the one oldest file needed removing")
        self.assertTrue(newer.exists())


class ReviewPreviewCacheTests(unittest.TestCase):
    """review_preview()'s own use of the budget/LRU machinery: a hit
    refreshes "recently used" standing, and a miss occasionally checks the
    budget rather than on every single write."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cache_dir = Path(self._tmp.name) / "cache"
        self.source = Path(self._tmp.name) / "photo.jpg"
        Image.new("RGB", (4, 4), color="blue").save(self.source, format="JPEG")
        thumbnails_module._writes_since_sweep.clear()

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_cache_hit_refreshes_the_file_s_mtime(self):
        first = review_preview(str(self.source), cache_dir=self.cache_dir)
        old_mtime = first.stat().st_mtime
        os.utime(first, (old_mtime - 1000, old_mtime - 1000))  # simulate it aging

        with mock.patch("picklikeme.analyzer.contactsheets.load_source_image") as load:
            second = review_preview(str(self.source), cache_dir=self.cache_dir)

        load.assert_not_called()
        self.assertGreater(second.stat().st_mtime, old_mtime - 1000)

    def test_a_miss_checks_the_budget_only_every_n_writes_not_every_time(self):
        with mock.patch("picklikeme.review.thumbnails.PREVIEW_CACHE_SWEEP_INTERVAL_WRITES", 3):
            with mock.patch("picklikeme.review.thumbnails._enforce_cache_budget") as enforce:
                for index in range(5):
                    source = Path(self._tmp.name) / f"photo{index}.jpg"
                    Image.new("RGB", (4, 4), color="red").save(source, format="JPEG")
                    review_preview(str(source), cache_dir=self.cache_dir, max_bytes=10_000)

                # 5 writes at an interval of 3: one check after the 3rd write,
                # not after the 1st, 2nd, 4th or 5th.
                self.assertEqual(enforce.call_count, 1)

    def test_the_cache_is_actually_kept_under_budget_over_many_writes(self):
        """End to end, no mocking of the budget enforcement itself: write
        enough distinct images that eviction must actually run, then check
        the real on-disk total."""
        with mock.patch("picklikeme.review.thumbnails.PREVIEW_CACHE_SWEEP_INTERVAL_WRITES", 2):
            for index in range(10):
                source = Path(self._tmp.name) / f"photo{index}.jpg"
                Image.new("RGB", (4, 4), color="green").save(source, format="JPEG")
                review_preview(str(source), cache_dir=self.cache_dir, max_bytes=2000)

        total = sum(size for _, size, _ in _cache_entries(self.cache_dir))
        self.assertLessEqual(total, 2000)


class PreviewCacheIsIndependentOfTheVisionCacheTests(unittest.TestCase):
    """The Preview Cache (this module) and the Vision Cache (bird_crop.py's
    crop cache) are deliberately separate concerns - a UI optimization vs
    the source of truth for Computer Vision - and the Vision Cache's
    resolution/format/quality becoming configurable must not have touched
    this one at all. See bird_crop.py's module docstring and
    docs/vision_cache.md."""

    def test_review_preview_builds_from_the_original_image_never_the_crop_cache(self):
        """review_preview must never import or depend on bird_crop's crop
        cache - it reads the ORIGINAL image directly (see
        contactsheets.load_source_image), which is exactly why a change to
        how bird crops are cached cannot affect it."""
        import inspect

        source = inspect.getsource(thumbnails_module.review_preview)
        self.assertNotIn("crop_cache", source)
        self.assertNotIn("bird_crop", source)

    def test_preview_cache_constants_are_unchanged_by_the_vision_cache_work(self):
        """Pinned so an accidental edit made while touching the Vision Cache
        elsewhere is caught immediately."""
        self.assertEqual(thumbnails_module.REVIEW_PREVIEW_CACHE_VERSION, 1)
        self.assertEqual(thumbnails_module.DEFAULT_PREVIEW_CACHE_MAX_BYTES, 20 * 1024**3)


if __name__ == "__main__":
    unittest.main()
