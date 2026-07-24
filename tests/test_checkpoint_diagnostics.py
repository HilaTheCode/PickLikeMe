import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_CHECKPOINT_PATH, PROJECT_ROOT
from picklikeme.model import ModelConfig, PreferenceHead
from picklikeme.train import load_checkpoint, save_checkpoint


def _model_and_opt():
    model = PreferenceHead(ModelConfig(backbone="cnn"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


class SaveDiagnosticsTests(unittest.TestCase):
    def test_success_block_printed_and_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            model, optimizer = _model_and_opt()
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = save_checkpoint(model, optimizer, path, epoch=3, best_loss=0.25, reason="End of epoch 3")
            output = buf.getvalue()

            self.assertTrue(ok)
            self.assertIn("Saving checkpoint", output)
            self.assertIn("Status: SUCCESS", output)
            self.assertIn("Reason: End of epoch 3", output)
            self.assertIn(str(path.resolve()), output)

    def test_failure_block_printed_and_returns_false_without_raising(self):
        # Point the checkpoint at a path whose parent is a *file*, so mkdir fails.
        with tempfile.TemporaryDirectory() as tmpdir:
            blocker = Path(tmpdir) / "not_a_dir"
            blocker.write_text("x", encoding="utf-8")
            path = blocker / "ckpt.pt"
            model, optimizer = _model_and_opt()
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = save_checkpoint(model, optimizer, path, epoch=1, reason="End of epoch 1")
            output = buf.getvalue()

            self.assertFalse(ok)
            self.assertIn("Checkpoint save FAILED", output)
            self.assertIn("Exception:", output)


class LoadDiagnosticsTests(unittest.TestCase):
    def test_load_block_printed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            model, optimizer = _model_and_opt()
            save_checkpoint(model, optimizer, path, epoch=2, best_loss=0.5, reason="setup")

            buf = io.StringIO()
            with redirect_stdout(buf):
                checkpoint = load_checkpoint(path, map_location="cpu")
            output = buf.getvalue()

            self.assertEqual(checkpoint["epoch"], 2)
            self.assertIn("Loading checkpoint", output)
            self.assertIn("Status: SUCCESS", output)
            self.assertIn("Epoch: 2", output)

    def test_load_failure_raises_and_prints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "does_not_exist.pt"
            buf = io.StringIO()
            with redirect_stdout(buf):
                with self.assertRaises(Exception):
                    load_checkpoint(missing, map_location="cpu")
            self.assertIn("Checkpoint load FAILED", buf.getvalue())


class DefaultPathTests(unittest.TestCase):
    def test_default_checkpoint_path_is_project_relative_and_absolute(self):
        self.assertTrue(DEFAULT_CHECKPOINT_PATH.is_absolute())
        self.assertEqual(DEFAULT_CHECKPOINT_DIR, PROJECT_ROOT / "checkpoints")
        self.assertEqual(DEFAULT_CHECKPOINT_PATH.name, "model_checkpoint.pt")
        # Deterministic regardless of process CWD: derived from the package
        # file location, so it must sit under the repo that contains src/.
        self.assertTrue((PROJECT_ROOT / "src" / "picklikeme").exists())


if __name__ == "__main__":
    unittest.main()
