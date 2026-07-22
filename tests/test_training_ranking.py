import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.dataset import LabelDataset
from picklikeme.train import rank_dataset


class DummyModel(nn.Module):
    def forward(self, x):
        return torch.tensor([[0.2], [0.9], [0.1]], dtype=torch.float32)


class DummyLoader:
    def load_image(self, path):
        return np.zeros((8, 8, 3), dtype=np.float32)


class RankingTests(unittest.TestCase):
    def test_rank_dataset_returns_sorted_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.csv"
            manifest_path.write_text(
                "image_path,label,burst_id,preference\n"
                "img1.jpg,1,b1,2\n"
                "img2.jpg,0,b1,1\n"
                "img3.jpg,0,b2,0\n",
                encoding="utf-8",
            )

            dataset = LabelDataset(str(manifest_path), str(tmpdir))
            ranked = rank_dataset(DummyModel(), dataset, DummyLoader(), device="cpu")

            self.assertEqual(ranked[0][0], "img1.jpg")
            self.assertGreaterEqual(ranked[0][1], ranked[1][1])
            self.assertGreaterEqual(ranked[0][1], ranked[-1][1])


if __name__ == "__main__":
    unittest.main()
