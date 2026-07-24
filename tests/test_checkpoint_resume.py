import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.config import ProjectConfig
from picklikeme.dataset import ImageLabel
from picklikeme.model import ModelConfig
from picklikeme.train import save_checkpoint, train


class TinyDataset:
    def __init__(self):
        self.items = [
            ImageLabel(image_path="img1.arw", label=1, burst_id=None, preference=1.0),
            ImageLabel(image_path="img2.arw", label=0, burst_id=None, preference=0.0),
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


class FlakyLoader:
    """Raises KeyboardInterrupt on the Nth image load, simulating Ctrl+C mid-batch."""

    def __init__(self, interrupt_after: int):
        self.interrupt_after = interrupt_after
        self.calls = 0

    def load_image(self, path):
        self.calls += 1
        if self.calls > self.interrupt_after:
            raise KeyboardInterrupt
        return np.zeros((16, 16, 3), dtype=np.float32)


def _run(tmpdir, epochs, resume, checkpoint_path=None, best_checkpoint_path=None):
    checkpoint_path = checkpoint_path or Path(tmpdir) / "checkpoint.pt"
    config = ProjectConfig(batch_size=2, learning_rate=1e-3, epochs=epochs, device="cpu")
    model_config = ModelConfig(backbone="cnn")
    buf = io.StringIO()
    with redirect_stdout(buf):
        model = train(
            config,
            DummyLoader(),
            dataset=TinyDataset(),
            checkpoint_path=checkpoint_path,
            resume=resume,
            model_config=model_config,
            best_checkpoint_path=best_checkpoint_path,
        )
    return model, checkpoint_path, buf.getvalue()


class EpochBookkeepingTests(unittest.TestCase):
    def test_checkpoint_records_last_completed_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, checkpoint_path, _ = _run(tmpdir, epochs=2, resume=False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 2)

    def test_resume_at_same_target_epochs_does_no_extra_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            _run(tmpdir, epochs=2, resume=False, checkpoint_path=checkpoint_path)
            _, _, output = _run(tmpdir, epochs=2, resume=True, checkpoint_path=checkpoint_path)
            self.assertNotIn("Starting epoch", output)

    def test_resume_continues_from_last_completed_epoch_not_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            _run(tmpdir, epochs=2, resume=False, checkpoint_path=checkpoint_path)
            _, checkpoint_path, output = _run(tmpdir, epochs=4, resume=True, checkpoint_path=checkpoint_path)

            self.assertIn("Starting epoch 3/4", output)
            self.assertIn("Starting epoch 4/4", output)
            self.assertNotIn("Starting epoch 1/4", output)
            self.assertNotIn("Starting epoch 2/4", output)

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 4)

    def test_legacy_checkpoint_without_epoch_field_resumes_from_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            from picklikeme.model import PreferenceHead

            legacy_model = PreferenceHead(ModelConfig(backbone="cnn"))
            optimizer = torch.optim.Adam(legacy_model.parameters())
            torch.save(
                {"model_state_dict": legacy_model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                checkpoint_path,
            )

            _, _, output = _run(tmpdir, epochs=2, resume=True, checkpoint_path=checkpoint_path)
            self.assertIn("Starting epoch 1/2", output)
            self.assertIn("Starting epoch 2/2", output)


class BestCheckpointTests(unittest.TestCase):
    def test_best_checkpoint_created_alongside_regular_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, checkpoint_path, _ = _run(tmpdir, epochs=1, resume=False)
            best_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")
            self.assertTrue(best_path.exists())
            best_checkpoint = torch.load(best_path, map_location="cpu")
            self.assertIn("best_loss", best_checkpoint)

    def test_explicit_best_checkpoint_path_is_respected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            best_path = Path(tmpdir) / "custom_best.pt"
            _run(tmpdir, epochs=1, resume=False, best_checkpoint_path=best_path)
            self.assertTrue(best_path.exists())

    def test_regular_checkpoint_reflects_updated_best_loss_same_epoch(self):
        # Regression: the regular checkpoint used to be written before best_loss
        # was updated for the epoch, so resuming always read back best_loss=inf.
        with tempfile.TemporaryDirectory() as tmpdir:
            _, checkpoint_path, _ = _run(tmpdir, epochs=1, resume=False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertLess(checkpoint["best_loss"], math.inf)

    def test_best_loss_survives_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            _run(tmpdir, epochs=1, resume=False, checkpoint_path=checkpoint_path)
            first_best = torch.load(checkpoint_path, map_location="cpu")["best_loss"]

            _run(tmpdir, epochs=2, resume=True, checkpoint_path=checkpoint_path)
            second_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # best_loss can only stay the same or improve, never reset to inf.
            self.assertLessEqual(second_checkpoint["best_loss"], first_best)


class InterruptTests(unittest.TestCase):
    def test_keyboard_interrupt_saves_checkpoint_before_propagating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            config = ProjectConfig(batch_size=1, learning_rate=1e-3, epochs=5, device="cpu")
            model_config = ModelConfig(backbone="cnn")

            with self.assertRaises(KeyboardInterrupt):
                train(
                    config,
                    FlakyLoader(interrupt_after=1),
                    dataset=TinyDataset(),
                    checkpoint_path=checkpoint_path,
                    resume=False,
                    model_config=model_config,
                )

            self.assertTrue(checkpoint_path.exists())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # Interrupted mid-first-epoch: no epoch fully completed yet.
            self.assertEqual(checkpoint["epoch"], 0)
            self.assertFalse((checkpoint_path.parent / (checkpoint_path.name + ".tmp")).exists())


class AtomicWriteTests(unittest.TestCase):
    def test_no_leftover_tmp_file_after_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters())
            save_checkpoint(model, optimizer, checkpoint_path, epoch=1, best_loss=0.5)

            self.assertTrue(checkpoint_path.exists())
            self.assertFalse((checkpoint_path.parent / (checkpoint_path.name + ".tmp")).exists())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["best_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
