"""eyes.runtime_providers - the isolated onnxruntime execution-provider
discovery layer EyePose-v0 goes through instead of knowing about torch (or
any future GPU runtime) itself. See the module's own docstring for the
historical bug this replaces: CUDA execution used to work only by the
accident of torch having already been imported elsewhere first.
"""

from __future__ import annotations

import os
import sys

import pytest

from picklikeme.eyes import runtime_providers as rp


@pytest.fixture(autouse=True)
def _preserve_path(monkeypatch):
    """`_prepend_to_path` mutates os.environ["PATH"] directly - hand PATH
    to monkeypatch up front so its normal teardown restores the pre-test
    value regardless of what a test does to it afterward."""
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))


class TestEnsureProviderDiscoverable:
    def test_unknown_provider_is_a_silent_no_op(self):
        before = os.environ["PATH"]
        rp.ensure_provider_discoverable("TensorrtExecutionProvider")  # no strategy registered yet
        assert os.environ["PATH"] == before

    def test_cpu_provider_needs_no_discovery_and_is_a_no_op(self):
        before = os.environ["PATH"]
        rp.ensure_provider_discoverable("CPUExecutionProvider")
        assert os.environ["PATH"] == before

    def test_cuda_prepends_torchs_own_bundled_lib_directory(self, tmp_path, monkeypatch):
        import torch

        fake_lib = tmp_path / "torch_fake" / "lib"
        fake_lib.mkdir(parents=True)
        monkeypatch.setattr(torch, "__file__", str(tmp_path / "torch_fake" / "__init__.py"))
        monkeypatch.setattr(sys, "platform", "win32")

        rp.ensure_provider_discoverable("CUDAExecutionProvider")

        assert str(fake_lib) in os.environ["PATH"].split(os.pathsep)

    def test_cuda_discovery_is_idempotent(self, tmp_path, monkeypatch):
        import torch

        fake_lib = tmp_path / "torch_fake" / "lib"
        fake_lib.mkdir(parents=True)
        monkeypatch.setattr(torch, "__file__", str(tmp_path / "torch_fake" / "__init__.py"))
        monkeypatch.setattr(sys, "platform", "win32")

        rp.ensure_provider_discoverable("CUDAExecutionProvider")
        rp.ensure_provider_discoverable("CUDAExecutionProvider")

        assert os.environ["PATH"].split(os.pathsep).count(str(fake_lib)) == 1

    def test_cuda_discovery_is_a_no_op_off_windows(self, tmp_path, monkeypatch):
        import torch

        fake_lib = tmp_path / "torch_fake" / "lib"
        fake_lib.mkdir(parents=True)
        monkeypatch.setattr(torch, "__file__", str(tmp_path / "torch_fake" / "__init__.py"))
        monkeypatch.setattr(sys, "platform", "linux")

        rp.ensure_provider_discoverable("CUDAExecutionProvider")

        assert str(fake_lib) not in os.environ["PATH"].split(os.pathsep)

    def test_cuda_discovery_is_a_no_op_when_torchs_lib_dir_is_missing(self, tmp_path, monkeypatch):
        """A CPU-only torch build (or a torch install with no bundled
        `lib` directory) must never raise - onnxruntime's own fallback to
        CPU handles "CUDA unavailable" safely from here."""
        import torch

        before = os.environ["PATH"]
        monkeypatch.setattr(torch, "__file__", str(tmp_path / "no_such_torch" / "__init__.py"))
        monkeypatch.setattr(sys, "platform", "win32")

        rp.ensure_provider_discoverable("CUDAExecutionProvider")

        assert os.environ["PATH"] == before

    def test_cuda_discovery_is_a_no_op_when_torch_is_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # forces `import torch` to raise ImportError
        monkeypatch.setattr(sys, "platform", "win32")
        before = os.environ["PATH"]

        rp.ensure_provider_discoverable("CUDAExecutionProvider")  # must not raise

        assert os.environ["PATH"] == before


class TestDescribeDevice:
    def test_cpu_provider_describes_as_cpu(self):
        assert rp.describe_device("CPUExecutionProvider") == "CPU"

    def test_cuda_provider_describes_using_torch(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "NVIDIA GeForce RTX 4090")

        assert rp.describe_device("CUDAExecutionProvider") == "NVIDIA GeForce RTX 4090"

    def test_a_provider_with_no_describer_falls_back_to_its_own_name(self):
        assert rp.describe_device("TensorrtExecutionProvider") == "TensorrtExecutionProvider"

    def test_a_failing_describer_falls_back_to_the_provider_name_rather_than_raising(self, monkeypatch):
        import torch

        def boom(index=0):
            raise RuntimeError("no CUDA device")

        monkeypatch.setattr(torch.cuda, "get_device_name", boom)

        assert rp.describe_device("CUDAExecutionProvider") == "CUDAExecutionProvider"


class TestRecommendedOnnxruntimeExtra:
    def test_recommends_gpu_when_cuda_is_available(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert rp.recommended_onnxruntime_extra() == "eyepose-gpu"

    def test_recommends_cpu_when_cuda_is_not_available(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert rp.recommended_onnxruntime_extra() == "eyepose-cpu"

    def test_recommends_cpu_when_torch_is_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        assert rp.recommended_onnxruntime_extra() == "eyepose-cpu"
