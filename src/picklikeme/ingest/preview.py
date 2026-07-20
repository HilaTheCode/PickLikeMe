"""Decode RAW files to a cached JPEG preview for fast repeated training reads.

Decoding a RAW file every epoch (1-2s/image for a full-resolution demosaic)
doesn't scale to tens of thousands of images. A raw tensor cache would be
faster to load but locks in one resolution/dtype ahead of augmentation
decisions and isn't human-inspectable, which matters when spot-checking
burst clustering against a messy real archive. A JPEG cache from a
half-size rawpy decode is small, fast to regenerate, and easy to open by
hand, at a resolution well above the model's ~384px input so downstream
resizing/augmentation choices stay flexible.

Preview paths mirror the archive's relative structure under preview_root
so they stay unique and stable across runs, which is what makes the
skip-if-exists check a correct incremental cache.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import rawpy
from tqdm import tqdm


def _decode_preview(raw_path: str, out_path: str, long_edge: int) -> None:
    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8, no_auto_bright=False)
    height, width = rgb.shape[:2]
    scale = long_edge / max(height, width)
    if scale < 1.0:
        rgb = cv2.resize(
            rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _worker(args: tuple[str, str, int]) -> tuple[str, bool, str | None]:
    raw_path, out_path, long_edge = args
    if Path(out_path).exists():
        return raw_path, True, None
    try:
        _decode_preview(raw_path, out_path, long_edge)
        return raw_path, True, None
    except Exception as exc:  # noqa: BLE001 - report every decode failure instead of aborting the batch
        return raw_path, False, str(exc)


def generate_previews(
    manifest_rows: list[tuple[str, str]],
    preview_root: Path,
    long_edge: int = 512,
    workers: int = 8,
) -> list[tuple[str, str]]:
    """manifest_rows: list of (absolute raw path, image_path relative to archive_root)."""
    tasks = [
        (raw_path, str(preview_root / Path(rel_path).with_suffix(".jpg")), long_edge)
        for raw_path, rel_path in manifest_rows
    ]

    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for raw_path, ok, err in tqdm(pool.map(_worker, tasks), total=len(tasks), desc="Generating previews"):
            if not ok:
                failures.append((raw_path, err or "unknown error"))
    return failures
