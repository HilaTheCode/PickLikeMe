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
from picklikeme.train import train


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
            config = ProjectConfig(batch_size=2, learning_rate=1e-3, epochs=1, device="cpu")
            dataset = TinyDataset()
            loader = DummyLoader()

            model_config = ModelConfig(backbone="cnn")

            first_model = train(config, loader, dataset=dataset, checkpoint_path=checkpoint_path, resume=False, model_config=model_config)
            self.assertTrue(checkpoint_path.exists())
            self.assertIsInstance(first_model, PreferenceHead)

            second_model = train(config, loader, dataset=dataset, checkpoint_path=checkpoint_path, resume=True, model_config=model_config)
            self.assertIsInstance(second_model, PreferenceHead)
            self.assertTrue(torch.equal(second_model.state_dict()["classifier.weight"], second_model.state_dict()["classifier.weight"]))


if __name__ == "__main__":
    unittest.main()
