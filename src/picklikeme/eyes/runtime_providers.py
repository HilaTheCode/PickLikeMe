"""Execution-provider native-library discovery for onnxruntime.

onnxruntime's execution providers (`CUDAExecutionProvider`,
`TensorrtExecutionProvider`, `DmlExecutionProvider`,
`OpenVINOExecutionProvider`, ...) are native DLLs/shared objects that
onnxruntime tries to load at `InferenceSession` construction time. Each has
its own runtime dependency (CUDA/cuDNN for CUDA and TensorRT, DirectX for
DirectML, the OpenVINO runtime for OpenVINO...) that has to already be
discoverable on the OS's shared-library search path *before* onnxruntime
tries to load it - if it isn't, the provider fails to load and onnxruntime
silently falls back to the next one in the list (see
`eyepose_v0._select_providers`), which is safe but means "requested CUDA,
silently got CPU" with no error anywhere.

This module is the one place that knows how to make a given provider's
native libraries discoverable, keyed by the provider's own onnxruntime name,
so any onnxruntime-backed detector (EyePose-v0 today, whatever comes after
it) can ask for a provider by name without knowing anything about where its
libraries actually come from - in particular, without importing torch
itself. Today only `CUDAExecutionProvider` has a real strategy
(`_discover_cuda`, sourced from torch's own already-mandatory CUDA build -
see its docstring for why that's deliberate and not just convenient).
Adding TensorRT/DirectML/OpenVINO support later is one more function plus
one more `_DISCOVERY_STRATEGIES` entry here - no caller of
`ensure_provider_discoverable` needs to change.

Why this used to be a real bug: `EyePoseV0EyeDetector.__init__` used to
construct its onnxruntime session directly, with nothing making CUDA's
libraries discoverable itself. On Windows, CUDA execution only ever worked
by accident, because `ranking.classic.rank_folder` happens to call
`build_cache` (which imports torch to run the bird detector) *before*
constructing the eye detector - torch's own import happens to prepend its
bundled `lib` directory onto the process's DLL search path as a side
effect. Anything that constructed `EyePoseV0EyeDetector(device="cuda")`
without torch already having been imported first - a standalone script, a
test, a future caller - would silently get CPU with no error at all. See
`ensure_provider_discoverable`'s call site in `eyepose_v0.py`'s `__init__`
for the fix: discovery is now explicit and self-contained, not a side
effect of import order.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def ensure_provider_discoverable(provider: str) -> None:
    """Best-effort: make `provider`'s native runtime libraries discoverable
    on this OS's shared-library search path, if this module knows how to.

    Silently does nothing for a provider this module has no strategy for
    (including plain `CPUExecutionProvider`, which needs none), or when a
    strategy's own dependency isn't available (e.g. a CPU-only torch
    build) - onnxruntime's own "requested but couldn't load, fall back to
    the next provider" behaviour already handles that safely. This
    function only ever improves the odds that a requested provider
    actually loads; it is never itself a correctness requirement.
    """
    strategy = _DISCOVERY_STRATEGIES.get(provider)
    if strategy is not None:
        strategy()


def describe_device(provider: str) -> str:
    """A short, human-readable name for the hardware behind an *active*
    execution provider - startup-diagnostic use only (see
    `eyepose_v0.EyePoseV0EyeDetector.__init__`), never used for any
    decision-making. Falls back to the provider's own onnxruntime name if
    this module has no better description, or if getting one fails for any
    reason - a cosmetic lookup must never turn into a crash.
    """
    describer = _DEVICE_DESCRIBERS.get(provider)
    if describer is not None:
        try:
            return describer()
        except Exception:  # noqa: BLE001 - cosmetic only, never fatal
            pass
    return provider


def recommended_onnxruntime_extra() -> str:
    """"eyepose-gpu" if this machine has a CUDA-capable torch build,
    else "eyepose-cpu" - the choice a future installer should make
    automatically (see pyproject.toml's `eyepose-cpu`/`eyepose-gpu`
    extras). Not called anywhere yet - this is scaffolding for that future
    installer, not a runtime dependency of EyePose-v0 itself.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "eyepose-gpu"
    except ImportError:
        pass
    return "eyepose-cpu"


def _discover_cuda() -> None:
    """CUDAExecutionProvider's cuBLAS/cuDNN dependency, sourced from
    torch's own bundled CUDA build.

    torch is already a mandatory dependency of this whole project (see
    pyproject.toml's base `dependencies`), and a CUDA-enabled torch wheel
    already bundles its own copies of the exact cuBLAS/cuDNN DLLs
    `onnxruntime-gpu` needs - shipping a second ~1GB copy via
    `onnxruntime-gpu[cuda,cudnn]`'s own extras just for this would be pure
    duplication. If torch isn't installed, or was built without CUDA
    (`torch/lib` has no matching DLLs), this is a silent no-op -
    onnxruntime's own fallback to CPU takes over from there.
    """
    if sys.platform != "win32":
        return  # Linux/macOS CUDA wheels resolve their deps via rpath - no PATH trick needed
    try:
        import torch
    except ImportError:
        return
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.is_dir():
        _prepend_to_path(torch_lib)


def _describe_cuda_device() -> str:
    import torch

    return torch.cuda.get_device_name(0)


def _prepend_to_path(directory: Path) -> None:
    """Idempotent - never adds the same directory twice, even across
    repeated detector construction within one process."""
    current = os.environ.get("PATH", "")
    if str(directory) in current.split(os.pathsep):
        return
    os.environ["PATH"] = str(directory) + os.pathsep + current
    logger.debug("Added %s to PATH for onnxruntime's native provider libraries", directory)


_DISCOVERY_STRATEGIES: dict[str, Callable[[], None]] = {
    "CUDAExecutionProvider": _discover_cuda,
    # TensorrtExecutionProvider, DmlExecutionProvider, OpenVINOExecutionProvider,
    # etc. each add one entry here, matching this same one-function-per-provider
    # shape - nothing above (or in any caller) needs to change to add one.
}

_DEVICE_DESCRIBERS: dict[str, Callable[[], str]] = {
    "CUDAExecutionProvider": _describe_cuda_device,
    "CPUExecutionProvider": lambda: "CPU",
}
