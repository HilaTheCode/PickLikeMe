import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.train import resolve_device


class ResolveDeviceTests(unittest.TestCase):
    def test_cpu_request_passes_through(self):
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_cuda_request_passes_through_when_available(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("cuda"), "cuda")

    def test_cuda_request_falls_back_to_cpu_when_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("cuda"), "cpu")


if __name__ == "__main__":
    unittest.main()
