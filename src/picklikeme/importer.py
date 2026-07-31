from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from .analyzer.annotations import AnnotationStore, REVIEW_KEEP, REVIEW_REJECT
from .organize import SELECTED_DIRNAME


def import_selected_images(
    *,
    source_folder: str | Path,
    destination_root: str | Path,
    store: AnnotationStore | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Copy the current folder's selected images into a destination and update stored review decisions.

    `source_folder` is whatever folder is currently open in review - a memory
    card, a local disk folder, or anywhere else; nothing here assumes a
    particular kind of source, only that a `_Selected` subfolder exists
    under it.
    """
    source_folder = Path(source_folder)
    destination_root = Path(destination_root)
    source_dir = source_folder / SELECTED_DIRNAME
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Selected folder not found: {source_dir}")

    destination_root.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(source_dir.iterdir()) if p.is_file()]
    copied = 0
    for index, src_path in enumerate(files, start=1):
        dest_path = destination_root / src_path.name
        shutil.copy2(src_path, dest_path)
        copied += 1

        if store is not None:
            try:
                store.repoint_review_decisions({str(src_path): dest_path})
            except Exception:
                pass

        if on_progress is not None:
            on_progress(index, len(files))

    return {"copied": copied, "source_folder": str(source_dir), "destination_root": str(destination_root)}
