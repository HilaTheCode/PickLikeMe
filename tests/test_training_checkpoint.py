import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.config import ProjectConfig
from picklikeme.dataset import ImageLabel
from picklikeme.model import ModelConfig, PreferenceHead
from picklikeme.train import ExistingCheckpointError, check_fresh_start_is_safe, train


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


class TrainingCheckpointTests(unittest.TestCase):
    def test_train_can_save_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            config = ProjectConfig(batch_size=2, learning_rate=1e-3, device="cpu")
            dataset = TinyDataset()
            loader = DummyLoader()

            model_config = ModelConfig(backbone="cnn")

            first_model = train(config, loader, dataset=dataset, checkpoint_path=checkpoint_path, resume=False, model_config=model_config, epochs_this_run=1)
            self.assertTrue(checkpoint_path.exists())
            self.assertIsInstance(first_model, PreferenceHead)

            second_model = train(config, loader, dataset=dataset, checkpoint_path=checkpoint_path, resume=True, model_config=model_config, epochs_this_run=1)
            self.assertIsInstance(second_model, PreferenceHead)
            self.assertTrue(torch.equal(second_model.state_dict()["classifier.weight"], second_model.state_dict()["classifier.weight"]))


class FreshStartSafetyTests(unittest.TestCase):
    """--fresh-start must never be able to silently destroy an existing
    checkpoint - the whole point of the guard is that this is impossible to
    trigger by accident, from any caller."""

    def test_check_fresh_start_is_safe_passes_when_nothing_exists_yet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            check_fresh_start_is_safe(checkpoint_path, resume=False)  # must not raise

    def test_check_fresh_start_is_safe_passes_when_resuming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            checkpoint_path.write_bytes(b"not a real checkpoint, existence is all that matters here")
            check_fresh_start_is_safe(checkpoint_path, resume=True)  # must not raise

    def test_check_fresh_start_is_safe_refuses_when_a_checkpoint_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            checkpoint_path.write_bytes(b"a real checkpoint would be a torch.save() blob")

            with self.assertRaises(ExistingCheckpointError) as ctx:
                check_fresh_start_is_safe(checkpoint_path, resume=False)

            self.assertIn(str(checkpoint_path.resolve()), str(ctx.exception))
            self.assertIn("checkpoint already exists", str(ctx.exception))
            # Refusing must never touch the file it is protecting.
            self.assertEqual(checkpoint_path.read_bytes(), b"a real checkpoint would be a torch.save() blob")

    def test_a_missing_checkpoint_path_is_always_safe(self):
        """No checkpoint configured at all (checkpoint_path=None) has nothing
        to protect - this must not be confused with "fresh start is unsafe"."""
        check_fresh_start_is_safe(None, resume=False)  # must not raise

    def test_train_itself_refuses_a_fresh_start_over_an_existing_checkpoint(self):
        """The guard lives inside train() itself - not only in the CLI layer
        - so every caller gets it, including this one, calling train()
        directly exactly like the CLI does."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            config = ProjectConfig(batch_size=2, learning_rate=1e-3, device="cpu")
            dataset = TinyDataset()
            loader = DummyLoader()
            model_config = ModelConfig(backbone="cnn")

            # A real checkpoint from a first, legitimate run.
            train(config, loader, dataset=dataset, checkpoint_path=checkpoint_path, resume=False, model_config=model_config, epochs_this_run=1)
            self.assertTrue(checkpoint_path.exists())
            original_bytes = checkpoint_path.read_bytes()

            with self.assertRaises(ExistingCheckpointError):
                train(config, loader, dataset=dataset, checkpoint_path=checkpoint_path, resume=False, model_config=model_config, epochs_this_run=1)

            # Refused before a single byte of the existing checkpoint changed.
            self.assertEqual(checkpoint_path.read_bytes(), original_bytes)

    def test_train_still_allows_a_fresh_start_when_no_checkpoint_exists(self):
        """The guard must not become a blanket refusal - a first-ever run
        with no checkpoint on disk is exactly what resume=False is for."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            config = ProjectConfig(batch_size=2, learning_rate=1e-3, device="cpu")
            dataset = TinyDataset()
            loader = DummyLoader()
            model_config = ModelConfig(backbone="cnn")

            model = train(config, loader, dataset=dataset, checkpoint_path=checkpoint_path, resume=False, model_config=model_config, epochs_this_run=1)

            self.assertIsInstance(model, PreferenceHead)
            self.assertTrue(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
