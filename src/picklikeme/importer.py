from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .analyzer.annotations import AnnotationStore, REVIEW_KEEP, REVIEW_REJECT
from .organize import SELECTED_DIRNAME


def import_selected_images(
    *,
    source_folder: str | Path,
    destination_root: str | Path,
    store: AnnotationStore | None = None,
) -> dict[str, Any]:
    """Copy the current folder's selected images into a destination and update stored review decisions."""
    source_folder = Path(source_folder)
    destination_root = Path(destination_root)
    source_dir = source_folder / SELECTED_DIRNAME
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Selected folder not found: {source_dir}")

    destination_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_path in sorted(source_dir.iterdir()):
        if not src_path.is_file():
            continue
        dest_path = destination_root / src_path.name
        shutil.copy2(src_path, dest_path)
        copied += 1

        if store is not None:
            try:
                store.repoint_review_decisions({str(src_path): dest_path})
            except Exception:
                pass

    return {"copied": copied, "source_folder": str(source_dir), "destination_root": str(destination_root)}
