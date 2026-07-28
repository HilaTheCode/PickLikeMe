"""Cache layout (sharded) and the parallel preprocessing pipeline.

The pipeline overlaps decode with GPU work, so these tests pin the properties
that overlap must not break: order, sequential detection, bounded memory,
exact accounting, and resume-after-interruption.
"""

import io
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import picklikeme.preprocess as preprocess_module
from picklikeme.bird_crop import (
    CACHE_SHARD_CHARS,
    CROP_CACHE_VERSION,
    CROP_PARAMS_FILENAME,
    BirdDetection,
    CropParams,
    crop_cache_path,
    read_crop_params,
    save_crop_png,
    write_crop_params,
)
from picklikeme.preprocess import DECODE_WINDOW, build_cache, default_decode_workers
from picklikeme.raw_io import RawImageLoader

BOX = (10.0, 10.0, 30.0, 30.0)


class FakeDecoder:
    """Stands in for RawImageLoader. Returns a per-path constant image so a
    written crop can be traced back to its source, and records the threads that
    did the decoding."""

    def __init__(self, fail_paths=(), interrupt_paths=()):
        self.fail_paths = set(fail_paths)
        self.interrupt_paths = set(interrupt_paths)
        self.lock = threading.Lock()
        self.decoded = []
        self.threads = set()
        self.started = 0

    def _decode_full_frame(self, image_path):
        with self.lock:
            self.decoded.append(image_path)
            self.threads.add(threading.current_thread().name)
            self.started += 1
        if image_path in self.fail_paths:
            raise ValueError("simulated unreadable file")
        if image_path in self.interrupt_paths:
            raise KeyboardInterrupt
        value = (abs(hash(image_path)) % 200) + 20
        return np.full((40, 60, 3), value, dtype=np.uint8)


class FakeDetector:
    """Sequential-use detector. Fails the test if two threads are ever inside it
    at once, and records call order plus how far decoding has run ahead."""

    def __init__(self, decoder=None, no_detect_paths=(), interrupt_on=None):
        self.decoder = decoder
        self.no_detect_paths = set(no_detect_paths)
        self.interrupt_on = interrupt_on
        self.calls = 0
        self.max_decode_lead = 0
        self.concurrent = False
        self._busy = threading.Lock()
        self.seen_values = []

    def detect_with_all(self, image_rgb):
        best = self.detect_best_bird(image_rgb)
        return best, ([best] if best is not None else [])

    def detect_best_bird(self, image_rgb):
        if not self._busy.acquire(blocking=False):
            self.concurrent = True
            raise AssertionError("detector called concurrently")
        try:
            self.calls += 1
            self.seen_values.append(int(image_rgb[0, 0, 0]))
            if self.decoder is not None:
                with self.decoder.lock:
                    lead = self.decoder.started - self.calls
                self.max_decode_lead = max(self.max_decode_lead, lead)
            if self.interrupt_on is not None and self.calls == self.interrupt_on:
                raise KeyboardInterrupt
            # Emulate GPU latency so decode really does run ahead.
            import time

            time.sleep(0.005)
            return BirdDetection(box=BOX, score=0.9, label=16)
        finally:
            self._busy.release()


def _run_build(paths, cache_dir, decoder, detector, force=False, decode_workers=4, params=None):
    """Run the real build_cache with fakes swapped in for the decoder/detector."""
    original_loader = preprocess_module.RawImageLoader
    original_detector = preprocess_module.BirdDetector
    preprocess_module.RawImageLoader = lambda *a, **k: decoder
    preprocess_module.BirdDetector = lambda *a, **k: detector
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            stats = build_cache(
                paths,
                cache_dir,
                params if params is not None else CropParams(),
                device="cpu",
                force=force,
                decode_workers=decode_workers,
            )
    finally:
        preprocess_module.RawImageLoader = original_loader
        preprocess_module.BirdDetector = original_detector
    return stats, buf.getvalue()


def _fake_paths(tmpdir, count, start=0):
    return [str(Path(tmpdir) / f"img{i:03d}.arw") for i in range(start, start + count)]


class ShardedLayoutTests(unittest.TestCase):
    def test_path_is_sharded_by_first_two_digest_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = crop_cache_path(tmp, r"C:\photos\a.arw")
            self.assertEqual(path.suffix, ".png")
            self.assertEqual(len(path.parent.name), CACHE_SHARD_CHARS)
            # The shard is a prefix of the filename, so the path is derivable
            # from the digest alone - never by searching.
            self.assertTrue(path.stem.startswith(path.parent.name))
            self.assertEqual(path.parent.parent, Path(tmp))

    def test_same_source_same_path_different_source_different_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = crop_cache_path(tmp, r"C:\photos\a.arw")
            a2 = crop_cache_path(tmp, r"C:\photos\a.arw")
            b = crop_cache_path(tmp, r"C:\photos\b.arw")
            self.assertEqual(a, a2)
            self.assertNotEqual(a, b)

    def test_build_writes_only_into_shard_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 6)
            decoder = FakeDecoder()
            stats, _ = _run_build(paths, cache, decoder, FakeDetector(decoder))

            self.assertEqual(stats["cached"], 6)
            # No crop at the root; every crop one level down in its own shard.
            self.assertEqual(list(cache.glob("*.png")), [])
            for path in paths:
                entry = crop_cache_path(cache, path)
                self.assertTrue(entry.exists(), f"missing {entry}")
                self.assertEqual(entry.parent.parent, cache)
            # Only shard dirs plus the params file live at the root.
            roots = {p.name for p in cache.iterdir()}
            self.assertIn(CROP_PARAMS_FILENAME, roots)
            for name in roots - {CROP_PARAMS_FILENAME}:
                self.assertEqual(len(name), CACHE_SHARD_CHARS)

    def test_shard_directories_are_created_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "does" / "not" / "exist"
            target = crop_cache_path(cache, str(Path(tmp) / "x.arw"))
            self.assertFalse(target.parent.exists())
            save_crop_png(target, np.full((8, 8, 3), 7, dtype=np.uint8))
            self.assertTrue(target.exists())


class CacheReadWriteTests(unittest.TestCase):
    def test_written_crop_is_read_back_by_the_loader(self):
        """Write through save_crop_png, read through RawImageLoader: the two
        sides must agree on the sharded path."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "crops"
            source = root / "img.png"
            cv2.imwrite(str(source), np.zeros((40, 40, 3), dtype=np.uint8))  # black full frame
            save_crop_png(crop_cache_path(cache, source), np.full((20, 20, 3), 255, dtype=np.uint8))

            loader = RawImageLoader(raw_root=str(root), output_size=(32, 32), crop_cache_dir=str(cache))
            image = loader.load_image(str(source))
            self.assertGreater(image.max(), 0.9)  # the white cached crop, not the black frame

    def test_cache_miss_falls_back_to_the_full_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "crops"
            cache.mkdir()
            source = root / "img.png"
            cv2.imwrite(str(source), np.full((40, 40, 3), 128, dtype=np.uint8))

            loader = RawImageLoader(raw_root=str(root), output_size=(32, 32), crop_cache_dir=str(cache))
            image = loader.load_image(str(source))
            self.assertTrue(np.allclose(image, 128 / 255.0, atol=0.02))

    def test_cache_hit_skips_decode_entirely(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 3)
            decoder = FakeDecoder()
            _run_build(paths, cache, decoder, FakeDetector(decoder))
            self.assertEqual(len(decoder.decoded), 3)

            # Second pass: everything present, so nothing may be decoded again.
            decoder2 = FakeDecoder()
            stats, _ = _run_build(paths, cache, decoder2, FakeDetector(decoder2))
            self.assertEqual(stats["skipped"], 3)
            self.assertEqual(stats["cached"], 0)
            self.assertEqual(decoder2.decoded, [])

    def test_force_rebuilds_even_when_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 3)
            decoder = FakeDecoder()
            _run_build(paths, cache, decoder, FakeDetector(decoder))

            decoder2 = FakeDecoder()
            stats, _ = _run_build(paths, cache, decoder2, FakeDetector(decoder2), force=True)
            self.assertEqual(stats["cached"], 3)
            self.assertEqual(stats["skipped"], 0)
            self.assertEqual(len(decoder2.decoded), 3)


class CacheVersionMismatchTests(unittest.TestCase):
    """A cache built under a different crop-selection algorithm must never be
    silently reused. This is the exact regression every crop-selection change
    depends on (highest-confidence -> area-dominant -> group-scene handling):
    nothing about those changes altered what a CropParams *value* looks like
    on its own, only what the detector does with the boxes it selects among -
    so without a version bump, an old cache and new code would compare equal
    and the stale crops would be reused forever without any error.
    """

    def _seed_stale_cache(self, cache_dir: Path, version: str) -> None:
        """A cache directory holding only crop_params.json - like a real one
        that build_cache has already populated, minus the (irrelevant here)
        actual PNGs."""
        write_crop_params(cache_dir, CropParams(version=version))

    def test_current_code_writes_the_bumped_version_by_default(self):
        """Pins the version string itself, so a future accidental revert of
        the bump is caught immediately rather than silently reopening this
        exact hole."""
        self.assertEqual(CROP_CACHE_VERSION, "v4")
        self.assertEqual(CropParams().version, "v4")

    def test_a_cache_from_a_previous_selection_algorithm_is_refused(self):
        """The literal scenario this protects against: an existing cache built
        by an earlier algorithm (area-dominant v3, or highest-confidence v2)
        must stop the current run cold, rather than letting training silently
        proceed on stale crops."""
        for stale_version in ("v2", "v3"):
            with self.subTest(stale_version=stale_version):
                with tempfile.TemporaryDirectory() as tmp:
                    cache = Path(tmp) / "crops"
                    self._seed_stale_cache(cache, version=stale_version)

                    with self.assertRaises(SystemExit) as ctx:
                        build_cache(_fake_paths(tmp, 3), cache, CropParams(), device="cpu", force=False)
                    message = str(ctx.exception)
                    self.assertIn("different parameters", message)
                    self.assertIn("--force", message)

                    # Refused before doing anything: the stale params file is
                    # left exactly as it was, not silently overwritten.
                    self.assertEqual(read_crop_params(cache).version, stale_version)

    def test_force_rebuilds_over_a_stale_version_and_records_the_new_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            self._seed_stale_cache(cache, version="v2")
            paths = _fake_paths(tmp, 3)

            decoder = FakeDecoder()
            stats, _ = _run_build(paths, cache, decoder, FakeDetector(decoder), force=True)

            self.assertEqual(stats["cached"], 3, "a stale cache must be rebuilt in full, not skipped")
            self.assertEqual(len(decoder.decoded), 3)
            self.assertEqual(read_crop_params(cache).version, "v4")

    def test_a_cache_already_at_the_current_version_is_reused_normally(self):
        """The mismatch guard must not become a reason to always rebuild -
        only an actual parameter difference should trigger it."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 3)
            decoder = FakeDecoder()
            _run_build(paths, cache, decoder, FakeDetector(decoder))  # writes v3 params

            decoder2 = FakeDecoder()
            stats, _ = _run_build(paths, cache, decoder2, FakeDetector(decoder2), force=False)
            self.assertEqual(stats["skipped"], 3)
            self.assertEqual(decoder2.decoded, [])


class ResumeTests(unittest.TestCase):
    def test_resume_builds_only_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            first = _fake_paths(tmp, 3)
            everything = first + _fake_paths(tmp, 3, start=3)

            decoder = FakeDecoder()
            _run_build(first, cache, decoder, FakeDetector(decoder))

            decoder2 = FakeDecoder()
            stats, _ = _run_build(everything, cache, decoder2, FakeDetector(decoder2))
            self.assertEqual(stats["skipped"], 3)
            self.assertEqual(stats["cached"], 3)
            self.assertEqual(sorted(decoder2.decoded), sorted(everything[3:]))
            for path in everything:
                self.assertTrue(crop_cache_path(cache, path).exists())

    def test_interruption_keeps_finished_crops_and_the_next_run_completes(self):
        """A KeyboardInterrupt mid-pass must still flush crops already handed to
        the writer (no lost images), and the following run must pick up exactly
        the remainder."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 8)

            decoder = FakeDecoder()
            detector = FakeDetector(decoder, interrupt_on=4)  # dies on the 4th image
            with self.assertRaises(KeyboardInterrupt):
                _run_build(paths, cache, decoder, detector, decode_workers=3)

            done = [p for p in paths if crop_cache_path(cache, p).exists()]
            self.assertEqual(done, paths[:3], "the first three crops must be on disk, in order")

            decoder2 = FakeDecoder()
            stats, _ = _run_build(paths, cache, decoder2, FakeDetector(decoder2), decode_workers=3)
            self.assertEqual(stats["skipped"], 3)
            self.assertEqual(stats["cached"], 5)
            self.assertEqual(stats["errors"], 0)
            for path in paths:
                self.assertTrue(crop_cache_path(cache, path).exists())

    def test_no_temp_files_are_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 5)
            decoder = FakeDecoder()
            _run_build(paths, cache, decoder, FakeDetector(decoder))
            leftovers = [p.name for shard in cache.iterdir() if shard.is_dir() for p in shard.iterdir() if ".tmp" in p.name]
            self.assertEqual(leftovers, [])


class PipelineContractTests(unittest.TestCase):
    def test_detection_is_sequential_and_in_source_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 24)
            decoder = FakeDecoder()
            detector = FakeDetector(decoder)
            stats, _ = _run_build(paths, cache, decoder, detector, decode_workers=6)

            self.assertFalse(detector.concurrent, "detector must never run concurrently")
            self.assertEqual(detector.calls, 24)
            self.assertEqual(stats["cached"], 24)
            # Each fake frame carries a value derived from its path, so the
            # sequence of values proves images reached the GPU in input order.
            expected = [(abs(hash(p)) % 200) + 20 for p in paths]
            self.assertEqual(detector.seen_values, expected)

    def test_decode_runs_ahead_of_the_gpu_but_stays_inside_the_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 40)
            decoder = FakeDecoder()
            detector = FakeDetector(decoder)
            _run_build(paths, cache, decoder, detector, decode_workers=6)

            # Overlap actually happened...
            self.assertGreater(detector.max_decode_lead, 1, "decode never ran ahead of the GPU")
            # ...but never beyond the bounded window, which is what caps RAM.
            self.assertLessEqual(detector.max_decode_lead, DECODE_WINDOW)

    def test_decode_workers_bounds_the_decoder_thread_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 30)
            decoder = FakeDecoder()
            _run_build(paths, cache, decoder, FakeDetector(decoder), decode_workers=3)
            self.assertLessEqual(len(decoder.threads), 3)

            decoder2 = FakeDecoder()
            _run_build(paths, cache, decoder2, FakeDetector(decoder2), decode_workers=1, force=True)
            self.assertEqual(len(decoder2.threads), 1)

    def test_default_worker_count_is_capped_at_eight(self):
        self.assertLessEqual(default_decode_workers(), 8)
        self.assertGreaterEqual(default_decode_workers(), 1)

    def test_decode_failure_is_counted_and_the_pass_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 6)
            decoder = FakeDecoder(fail_paths={paths[2]})
            stats, output = _run_build(paths, cache, decoder, FakeDetector(decoder))

            self.assertEqual(stats["errors"], 1)
            self.assertEqual(stats["cached"], 5)
            self.assertIn("simulated unreadable file", output)
            self.assertFalse(crop_cache_path(cache, paths[2]).exists())
            for path in paths[:2] + paths[3:]:
                self.assertTrue(crop_cache_path(cache, path).exists())

    def test_write_failure_is_counted_and_its_stats_are_undone(self):
        """A crop is counted when handed to the writer; if the write then fails,
        the totals must end up exactly as a serial run's would."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 4)
            doomed = crop_cache_path(cache, paths[1])
            real_save = preprocess_module.save_crop_png

            def flaky_save(target, crop):
                if Path(target) == doomed:
                    raise OSError("simulated disk full")
                return real_save(target, crop)

            preprocess_module.save_crop_png = flaky_save
            try:
                decoder = FakeDecoder()
                stats, output = _run_build(paths, cache, decoder, FakeDetector(decoder))
            finally:
                preprocess_module.save_crop_png = real_save

            self.assertEqual(stats["errors"], 1)
            self.assertEqual(stats["cached"], 3)
            self.assertEqual(stats["birds"], 3)
            self.assertEqual(sum(stats["class_counts"].values()), 3)
            self.assertIn("simulated disk full", output)
            self.assertFalse(doomed.exists())

    def test_fallback_images_are_still_cached_as_full_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 4)
            decoder = FakeDecoder()

            class SometimesNothing(FakeDetector):
                def detect_best_bird(self, image_rgb):
                    self.calls += 1
                    return None if self.calls == 2 else BirdDetection(box=BOX, score=0.9, label=16)

                def detect_with_all(self, image_rgb):
                    best = self.detect_best_bird(image_rgb)
                    return best, ([best] if best is not None else [])

            stats, _ = _run_build(paths, cache, decoder, SometimesNothing(decoder))
            self.assertEqual(stats["fallbacks"], 1)
            self.assertEqual(stats["birds"], 3)
            self.assertEqual(stats["cached"], 4)
            for path in paths:
                self.assertTrue(crop_cache_path(cache, path).exists())

    def test_all_images_are_accounted_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 17)
            decoder = FakeDecoder(fail_paths={paths[5]})
            stats, _ = _run_build(paths, cache, decoder, FakeDetector(decoder), decode_workers=5)
            self.assertEqual(
                stats["cached"] + stats["skipped"] + stats["errors"], stats["total"], stats
            )

    def test_no_pipeline_threads_survive_the_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "crops"
            paths = _fake_paths(tmp, 12)
            decoder = FakeDecoder()
            _run_build(paths, cache, decoder, FakeDetector(decoder), decode_workers=4)
            alive = [t.name for t in threading.enumerate() if "picklikeme-" in t.name]
            self.assertEqual(alive, [], f"leaked threads: {alive}")


if __name__ == "__main__":
    unittest.main()
