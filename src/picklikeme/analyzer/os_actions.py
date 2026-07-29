"""Platform-specific "open this in the OS file manager" for the local server.

A served report cannot reach the OS from the browser - that is what makes the
server necessary in the first place (see server.py's module docstring). This
is the one place that difference is bridged, so every future action that needs
to touch the OS (open an image, reveal a RAW, ...) has one small, mockable
function to build on rather than reimplementing the platform check.

Windows is first-class (`os.startfile`, what "Open Folder" is built for).
macOS/Linux are supported the obvious way for portability, but untested here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def open_in_file_manager(path: Path) -> None:
    """Open `path` (a file or directory) in the OS's file manager."""
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 - loopback-only server, path is already dataset-confined
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
    else:
        subprocess.run(["xdg-open", str(path)], check=True)


def choose_folder(initial_dir: Path | None = None) -> Path | None:
    """Show the OS's native "choose a folder" dialog and return what the
    photographer picked, or None if they cancelled.

    A browser has no API for picking an arbitrary local folder by path (only
    an upload picker, which never yields a real filesystem path) - so, like
    `open_in_file_manager`, this bridges the served page to the OS on its
    behalf. tkinter ships with the standard library; a throwaway hidden root
    window is the documented way to use its dialogs without a full Tk app.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            initialdir=str(initial_dir) if initial_dir and initial_dir.is_dir() else None,
            title="Choose a folder to review",
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None
