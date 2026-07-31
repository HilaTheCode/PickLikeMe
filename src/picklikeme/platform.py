"""Small platform adapters for cross-platform behaviour."""

from __future__ import annotations

import os
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
