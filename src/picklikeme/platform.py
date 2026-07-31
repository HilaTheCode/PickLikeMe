"""Small platform adapters for cross-platform behaviour."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any


def resolve_torch_device(requested: str | None) -> str:
    """Resolve a torch device with a CUDA -> MPS -> CPU fallback chain.

    ``requested`` may be ``None`` for automatic selection, an explicit device
    like ``"cuda"`` or ``"cpu"``, or a fully-qualified device string.
    """
    if requested is None:
        requested = "auto"

    if requested in {"", "auto"}:
        return _auto_device()

    if requested in {"cpu", "mps"}:
        return requested

    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested if requested != "cuda" else "cuda"
        except Exception:  # noqa: BLE001 - best effort device selection
            pass

        try:
            import torch

            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
                return "mps"
        except Exception:  # noqa: BLE001 - best effort device selection
            pass

        if requested is not None:
            print(f"Requested device '{requested}' but CUDA is not available; using CPU")
        return "cpu"

    return requested


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 - best effort device selection
        pass

    try:
        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
            return "mps"
    except Exception:  # noqa: BLE001 - best effort device selection
        pass

    return "cpu"


def open_in_file_manager(path: Path) -> None:
    """Open ``path`` in the OS file manager."""
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 - loopback-only server, path is already dataset-confined
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
    else:
        subprocess.run(["xdg-open", str(path)], check=True)


def launch_browser(url: str) -> None:
    """Open ``url`` in the default browser, with a best-effort fallback."""
    try:
        webbrowser.open(url, new=1, autoraise=True)
    except webbrowser.Error:
        pass


def collect_environment_status() -> list[dict[str, Any]]:
    """Return lightweight startup diagnostics for optional external components."""
    status: list[dict[str, Any]] = []

    python_version = sys.version.split()[0]
    status.append({"name": "Python", "ok": True, "detail": python_version})

    try:
        import torch

        torch_status = f"{torch.__version__}"
        if torch.cuda.is_available():
            torch_status += " (CUDA available)"
        elif getattr(torch.backends, "mps", None) is not None and getattr(torch.backends.mps, "is_available", lambda: False)():
            torch_status += " (MPS available)"
        else:
            torch_status += " (CPU only)"
        status.append({"name": "PyTorch", "ok": True, "detail": torch_status})
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        status.append({"name": "PyTorch", "ok": False, "detail": str(exc)})

    try:
        import rawpy

        status.append({"name": "RawPy", "ok": True, "detail": getattr(rawpy, "__version__", "unknown")})
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        status.append({"name": "RawPy", "ok": False, "detail": str(exc)})

    try:
        import rawpy

        status.append({"name": "LibRaw", "ok": True, "detail": "available via rawpy"})
    except Exception:  # noqa: BLE001 - diagnostics only
        status.append({"name": "LibRaw", "ok": False, "detail": "not available"})

    exiftool_path = None
    try:
        from .ingest.metadata import _resolve_exiftool_path

        exiftool_path = _resolve_exiftool_path("exiftool")
    except Exception:  # noqa: BLE001 - diagnostics only
        exiftool_path = None

    if exiftool_path:
        status.append({"name": "ExifTool", "ok": True, "detail": exiftool_path})
    else:
        status.append({
            "name": "ExifTool",
            "ok": False,
            "detail": "not found; install via Homebrew on macOS or add exiftool.exe to PATH on Windows",
        })

    try:
        import importlib.util

        has_dng = importlib.util.find_spec("rawpy") is not None
        status.append({"name": "DNG support", "ok": has_dng, "detail": "rawpy available" if has_dng else "rawpy missing"})
    except Exception:  # noqa: BLE001 - diagnostics only
        status.append({"name": "DNG support", "ok": False, "detail": "unknown"})

    return status


def print_environment_status() -> None:
    """Print a concise startup diagnostics summary for external components."""
    print("Environment check:")
    for item in collect_environment_status():
        icon = "✓" if item["ok"] else "!"
        print(f"  {icon} {item['name']}: {item['detail']}")
