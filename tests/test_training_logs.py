"""Logging-only behavior: the run header, epoch summaries, and preprocessing
progress. These assert what a long run must be able to read back from its log;
they never assert on training behavior."""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.config import ProjectConfig, format_duration
from picklikeme.dataset import ImageLabel
from picklikeme.model import ModelConfig
from picklikeme.train import (
    DEFAULT_LOG_INTERVAL_BATCHES,
    RunCounts,
    count_label_split,
    describe_device,
    train,
)


class TinyDataset:
    def __init__(self, selected=6, rejected=4):
        self.items = [
            ImageLabel(image_path=f"keep{i}.arw", label=1, preference=1.0) for i in range(selected)
        ] + [
            ImageLabel(image_path=f"drop{i}.arw", label=0, preference=0.0) for i in range(rejected)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def count_sequences(self):
        return 1


class DummyLoader:
    def load_image(self, path):
        return np.zeros((16, 16, 3), dtype=np.float32)


def _train_and_capture(tmpdir, *, epochs=1, resume=False, counts=None, log_interval=DEFAULT_LOG_INTERVAL_BATCHES):
    config = ProjectConfig(batch_size=2, learning_rate=1e-3, device="cpu", num_workers=0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        train(
            config,
            DummyLoader(),
            dataset=TinyDataset(),
            checkpoint_path=Path(tmpdir) / "ckpt.pt",
            resume=resume,
            model_config=ModelConfig(backbone="cnn"),
            epochs_this_run=epochs,
            counts=counts,
            log_interval_batches=log_interval,
        )
    return buf.getvalue()


class RunHeaderTests(unittest.TestCase):
    def test_header_reports_scratch_start_and_all_requested_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = _train_and_capture(
                tmpdir, counts=RunCounts(selected=6, rejected=4, validation=3)
            )

            self.assertIn("starting from scratch", output)
            self.assertIn(str((Path(tmpdir) / "ckpt.pt").resolve()), output)
            self.assertIn("epochs:          1-1 (1 this run)", output)
            self.assertIn("10 = 6 selected / 4 rejected", output)
            self.assertIn("10 images, 5 batches of 2", output)
            self.assertIn("3 held-out images", output)
            self.assertIn("backbone:", output)
            self.assertIn("learning rate:   1.00e-03", output)
            self.assertIn("device:", output)
            self.assertIn("torch:", output)

    def test_header_reports_resume_point_and_best_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _train_and_capture(tmpdir, epochs=1)
            output = _train_and_capture(tmpdir, epochs=2, resume=True)

            self.assertIn("resuming from checkpoint at epoch 1", output)
            self.assertIn("best loss so far", output)
            self.assertIn("epochs:          2-3 (2 this run)", output)

    def test_header_states_when_there_is_no_validation_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = _train_and_capture(tmpdir, counts=RunCounts(selected=6, rejected=4))
            self.assertIn("validation set:  none", output)

    def test_counts_are_derived_from_the_dataset_when_not_supplied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = _train_and_capture(tmpdir, counts=None)
            self.assertIn("10 = 6 selected / 4 rejected", output)


class EpochSummaryTests(unittest.TestCase):
    def test_epoch_line_carries_loss_lr_timing_and_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = _train_and_capture(
                tmpdir, epochs=1, counts=RunCounts(selected=6, rejected=4, validation=3)
            )
            summary = next(line for line in output.splitlines() if "Completed epoch 1/1" in line)

            self.assertIn("train_loss", summary)
            self.assertIn("lr 1.00e-03", summary)
            self.assertIn("epoch ", summary)
            self.assertIn("run eta done", summary)  # last epoch of the run
            self.assertIn("images 10 train / 3 val", output)
            self.assertIn(f"checkpoint {Path(tmpdir) / 'ckpt.pt'}", output)

    def test_run_eta_is_reported_while_epochs_remain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = _train_and_capture(tmpdir, epochs=2)
            first = next(line for line in output.splitlines() if "Completed epoch 1/2" in line)
            self.assertIn("run eta", first)
            self.assertNotIn("run eta done", first)

    def test_best_checkpoint_is_flagged_on_the_epoch_that_saved_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = _train_and_capture(tmpdir, epochs=1)
            self.assertIn("(+best)", output)  # first epoch always sets a new best


class BatchLogVolumeTests(unittest.TestCase):
    def test_default_interval_is_sparse_enough_for_a_large_run(self):
        # 54k images at batch 16 is ~3,375 batches per epoch; the default must
        # not produce a line per batch.
        self.assertGreater(DEFAULT_LOG_INTERVAL_BATCHES, 1)

    def test_final_batch_of_an_epoch_is_always_logged(self):
        # 5 batches with interval 4: batch 4 hits the interval, batch 5 is the
        # last one and must still appear.
        with tempfile.TemporaryDirectory() as tmpdir:
            output = _train_and_capture(tmpdir, epochs=1, log_interval=4)
            self.assertIn("batch 5/5", output)

    def test_zero_interval_prints_no_batch_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = _train_and_capture(tmpdir, epochs=1, log_interval=0)
            self.assertNotIn("batch 5/5", output)
            self.assertIn("Completed epoch 1/1", output)  # epoch summary still printed


class HelperTests(unittest.TestCase):
    def test_count_label_split_counts_selected_and_rejected(self):
        self.assertEqual(count_label_split(TinyDataset(selected=7, rejected=2)), (7, 2))

    def test_count_label_split_falls_back_to_indexing(self):
        class NoItems:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                return ImageLabel(image_path="x", label=index)  # label 0 then 1

        self.assertEqual(count_label_split(NoItems()), (1, 1))

    def test_describe_device_reports_cpu_without_touching_cuda(self):
        self.assertEqual(describe_device("cpu"), "CPU")

    def test_format_duration_scales_from_seconds_to_hours(self):
        self.assertEqual(format_duration(45), "45s")
        self.assertEqual(format_duration(125), "2m05s")
        self.assertEqual(format_duration(3725), "1h02m05s")
        self.assertEqual(format_duration(-5), "0s")


class PreprocessProgressTests(unittest.TestCase):
    def test_progress_line_reports_position_rate_findings_and_eta(self):
        from picklikeme.preprocess import _print_progress

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_progress(
                250,
                1000,
                {"birds": 200, "fallbacks": 30, "skipped": 20, "errors": 1},
                elapsed=50.0,
            )
        line = buf.getvalue()

        self.assertIn("250/1,000 (25.0%)", line)
        self.assertIn("5.0 img/s", line)
        self.assertIn("detected 200", line)
        self.assertIn("fallback 30", line)
        self.assertIn("skipped 20", line)
        self.assertIn("errors 1", line)
        self.assertIn("eta 2m30s", line)  # 750 remaining at 5/s

    def test_progress_is_timer_based_so_a_fast_pass_does_not_flood(self):
        """A pass that only skips already-cached files completes in well under
        the interval, so it must log once (the final image) rather than per
        image."""
        import picklikeme.preprocess as preprocess_module
        from picklikeme.bird_crop import CropParams, crop_cache_path

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "crops"
            cache.mkdir(parents=True)
            paths = [str(Path(tmpdir) / f"img{i}.arw") for i in range(40)]
            for path in paths:
                entry = crop_cache_path(cache, path)
                entry.parent.mkdir(parents=True, exist_ok=True)  # sharded layout
                entry.write_bytes(b"cached")

            captured = io.StringIO()
            with redirect_stdout(captured):
                # Detector/decoder are never constructed on the all-skipped path
                # only if we avoid instantiating them; patch both out so the test
                # neither downloads weights nor decodes RAW.
                original_detector = preprocess_module.BirdDetector
                original_loader = preprocess_module.RawImageLoader
                preprocess_module.BirdDetector = lambda *a, **k: None
                preprocess_module.RawImageLoader = lambda *a, **k: None
                try:
                    stats = preprocess_module.build_cache(paths, cache, CropParams(), device="cpu")
                finally:
                    preprocess_module.BirdDetector = original_detector
                    preprocess_module.RawImageLoader = original_loader

            progress_lines = [line for line in captured.getvalue().splitlines() if "img/s" in line]
            self.assertEqual(stats["skipped"], 40)
            self.assertEqual(len(progress_lines), 1)
            self.assertIn("40/40 (100.0%)", progress_lines[0])


if __name__ == "__main__":
    unittest.main()
