import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import picklikeme.run as run_module


def _invoke(argv):
    calls = []
    with mock.patch.object(run_module, "preprocess_folders", side_effect=lambda *a, **k: calls.append("preprocess")), \
         mock.patch.object(run_module, "train_and_rank", side_effect=lambda args: calls.append("train")):
        old = sys.argv
        sys.argv = ["run"] + argv
        try:
            run_module.main()
        finally:
            sys.argv = old
    return calls


class RunPipelineTests(unittest.TestCase):
    def test_default_runs_preprocess_then_train(self):
        calls = _invoke(["--select-root", "S", "--reject-root", "R", "--epochs", "1"])
        self.assertEqual(calls, ["preprocess", "train"])

    def test_no_crop_birds_skips_preprocess(self):
        calls = _invoke(["--select-root", "S", "--reject-root", "R", "--epochs", "1", "--no-crop-birds"])
        self.assertEqual(calls, ["train"])

    def test_skip_preprocess_reuses_cache(self):
        calls = _invoke(["--select-root", "S", "--reject-root", "R", "--epochs", "1", "--skip-preprocess"])
        self.assertEqual(calls, ["train"])

    def test_requires_both_roots(self):
        with self.assertRaises(SystemExit):
            _invoke(["--select-root", "S", "--epochs", "1"])

    def test_requires_epochs(self):
        with self.assertRaises(SystemExit):
            _invoke(["--select-root", "S", "--reject-root", "R"])


if __name__ == "__main__":
    unittest.main()
