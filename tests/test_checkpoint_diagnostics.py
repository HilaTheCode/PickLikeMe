import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import picklikeme.train as train_module
from picklikeme.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_CHECKPOINT_PATH, PROJECT_ROOT
from picklikeme.model import ModelConfig, PreferenceHead
from picklikeme.train import load_checkpoint, save_checkpoint


def _model_and_opt():
    model = PreferenceHead(ModelConfig(backbone="cnn"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


class SaveDiagnosticsTests(unittest.TestCase):
    def test_success_is_one_line_with_epoch_loss_reason_and_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            model, optimizer = _model_and_opt()
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = save_checkpoint(model, optimizer, path, epoch=3, best_loss=0.25, reason="End of epoch 3")
            lines = buf.getvalue().splitlines()

            self.assertTrue(ok)
            # A run saves every few minutes for days: one line, not a block.
            self.assertEqual(len(lines), 1)
            self.assertIn("Checkpoint saved", lines[0])
            self.assertIn("epoch 3", lines[0])
            self.assertIn("best loss 0.2500", lines[0])
            self.assertIn("End of epoch 3", lines[0])
            self.assertIn(str(path.resolve()), lines[0])

    def test_success_line_handles_missing_epoch_and_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            model, optimizer = _model_and_opt()
            buf = io.StringIO()
            with redirect_stdout(buf):
                save_checkpoint(model, optimizer, path, reason="mid-epoch, no best yet")
            line = buf.getvalue().strip()

            self.assertIn("epoch n/a", line)
            self.assertIn("best loss n/a", line)  # None and inf both read as n/a

    def test_persistent_failure_retries_once_then_raises(self):
        # Point the checkpoint at a path whose parent is a *file*, so every
        # write attempt fails. Under the reliability policy this must retry
        # once and then raise RuntimeError (retry_delay_seconds=0 keeps it fast).
        with tempfile.TemporaryDirectory() as tmpdir:
            blocker = Path(tmpdir) / "not_a_dir"
            blocker.write_text("x", encoding="utf-8")
            path = blocker / "ckpt.pt"
            model, optimizer = _model_and_opt()
            buf = io.StringIO()
            with redirect_stdout(buf):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(model, optimizer, path, epoch=1, reason="End of epoch 1", retry_delay_seconds=0)
            output = buf.getvalue()

            # Failure output stays verbose on purpose: it is rare, and it is
            # what a broken save has to be diagnosed from.
            self.assertIn("Checkpoint save FAILED", output)
            self.assertIn("Full traceback:", output)
            self.assertIn("Waiting", output)
            self.assertIn("CHECKPOINTING IS NO LONGER RELIABLE", output)

    def test_transient_failure_is_retried_and_succeeds(self):
        # First write attempt raises, second succeeds: save_checkpoint must
        # recover, print a retry-succeeded block, and return True (no raise).
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            model, optimizer = _model_and_opt()
            real_write = train_module._write_checkpoint_atomic
            calls = {"n": 0}

            def flaky_write(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("simulated transient disk error")
                return real_write(*args, **kwargs)

            buf = io.StringIO()
            with mock.patch.object(train_module, "_write_checkpoint_atomic", side_effect=flaky_write):
                with redirect_stdout(buf):
                    ok = save_checkpoint(model, optimizer, path, epoch=2, best_loss=0.3, reason="End of epoch 2", retry_delay_seconds=0)
            output = buf.getvalue()

            self.assertTrue(ok)
            self.assertEqual(calls["n"], 2)
            # The failed first attempt keeps its detailed block; the recovery is
            # reported on the single success line.
            self.assertIn("Checkpoint save FAILED", output)
            self.assertIn("Checkpoint saved (retry SUCCEEDED)", output)
            # The file really exists after the successful retry.
            self.assertTrue(path.exists() and path.stat().st_size > 0)
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 2)


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
