"""Shared helper for running a long backend operation with a progress dialog."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from ..core.jobs import run_in_background


def run_with_progress(
    parent: QWidget,
    title: str,
    func: Callable[..., Any],
    *,
    on_success: Callable[[Any], None],
    on_error: Callable[[str], None] | None = None,
):
    """Run `func(on_progress=..., on_stage=...)` on a background thread while
    showing a non-cancellable modal progress dialog (the underlying backend
    operations do not support cooperative cancellation).

    Returns the QThread; the caller must keep a reference to it alive for
    the duration of the run (e.g. store it on `self`).
    """
    dialog = QProgressDialog(title, "", 0, 0, parent)
    # QProgressDialog's first constructor argument is the label text shown
    # inside the dialog, not its window title - without an explicit
    # setWindowTitle() the title bar falls back to the executable name
    # ("python"), not the app name.
    dialog.setWindowTitle(f"PeakPic - {title}")
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setLabelText(title)
    dialog.show()

    def _on_progress(done: int, total: int) -> None:
        if total > 0:
            dialog.setMaximum(total)
            dialog.setValue(done)
        else:
            dialog.setMaximum(0)

    def _on_stage(message: str) -> None:
        dialog.setLabelText(message)

    def _on_finished(result: Any) -> None:
        dialog.close()
        on_success(result)

    def _on_failed(message: str) -> None:
        dialog.close()
        if on_error is not None:
            on_error(message)
        else:
            QMessageBox.warning(parent, f"PeakPic - {title}", f"{title} failed:\n{message}")

    return run_in_background(
        parent,
        func,
        on_progress=_on_progress,
        on_stage=_on_stage,
        on_finished=_on_finished,
        on_failed=_on_failed,
    )
