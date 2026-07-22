"""Assemble the canonical training manifest, incrementally.

The archive scan always walks the full tree (cheap, seconds even at
hundreds of thousands of files). What's expensive is exiftool metadata
extraction and, later, RAW preview decoding, so those are skipped for any
file whose (relative path, size, mtime, label, pipeline version) matches
the previous manifest unchanged.

Burst membership can't be diffed the same way: adding one new frame to a
shoot can change the burst boundaries of its unchanged siblings. So burst
clustering is always fully recomputed for any shoot that contains at least
one new or changed file, using that shoot's complete frame set (reused
metadata for unchanged frames + freshly extracted metadata for changed
ones), while shoots with no changes at all keep their previous burst
assignment untouched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .burst import TimedImage, assign_bursts
from .metadata import ensure_exiftool_available, extract_metadata
from .scan import ScanIssues, ScannedImage, scan_archive

PIPELINE_VERSION = 1

MANIFEST_COLUMNS = [
    "image_path", "label", "shoot_id", "burst_id", "sequence_in_burst", "burst_size",
    "capture_timestamp", "subsecond", "camera_model", "lens_model", "raw_format",
    "iso", "shutter_speed", "f_number", "focal_length",
    "file_size_bytes", "file_mtime", "metadata_status", "pipeline_version",
]


def _fingerprint(path: Path) -> tuple[int, float]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime


def build_manifest(
    archive_root: Path,
    exiftool_path: str = "exiftool",
    gap_seconds: float = 1.5,
    existing_manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, ScanIssues]:
    ensure_exiftool_available(exiftool_path)
    archive_root = archive_root.resolve()

    scanned, issues = scan_archive(archive_root)

    existing = pd.DataFrame(columns=MANIFEST_COLUMNS)
    if existing_manifest_path and existing_manifest_path.exists():
        existing = pd.read_parquet(existing_manifest_path)
    existing_by_path = existing.set_index("image_path") if not existing.empty else None

    changed: list[tuple[ScannedImage, str, int, float]] = []
    reused: dict[str, dict] = {}
    dirty_shoots: set[str] = set()

    for img in scanned:
        rel_path = str(img.path.resolve().relative_to(archive_root))
        size, mtime = _fingerprint(img.path)
        unchanged = (
            existing_by_path is not None
            and rel_path in existing_by_path.index
            and existing_by_path.loc[rel_path, "file_size_bytes"] == size
            and existing_by_path.loc[rel_path, "file_mtime"] == mtime
            and existing_by_path.loc[rel_path, "pipeline_version"] == PIPELINE_VERSION
            and existing_by_path.loc[rel_path, "label"] == img.label
        )
        if unchanged:
            reused[rel_path] = existing_by_path.loc[rel_path].to_dict()
        else:
            changed.append((img, rel_path, size, mtime))
            dirty_shoots.add(img.shoot_id)

    fresh_meta = extract_metadata([c[0].path for c in changed], exiftool_path=exiftool_path) if changed else {}

    all_rows: dict[str, dict] = {}
    for img in scanned:
        rel_path = str(img.path.resolve().relative_to(archive_root))
        if rel_path in reused:
            row = dict(reused[rel_path])
            row["image_path"] = rel_path
        else:
            meta = fresh_meta.get(str(img.path.resolve()))
            size, mtime = _fingerprint(img.path)
            row = {
                "image_path": rel_path,
                "label": img.label,
                "shoot_id": img.shoot_id,
                "burst_id": None,
                "sequence_in_burst": None,
                "burst_size": None,
                "capture_timestamp": meta.capture_timestamp if meta else None,
                "subsecond": meta.subsecond if meta else None,
                "camera_model": meta.camera_model if meta else None,
                "lens_model": meta.lens_model if meta else None,
                "raw_format": img.raw_format,
                "iso": meta.iso if meta else None,
                "shutter_speed": meta.shutter_speed if meta else None,
                "f_number": meta.f_number if meta else None,
                "focal_length": meta.focal_length if meta else None,
                "file_size_bytes": size,
                "file_mtime": mtime,
                "metadata_status": meta.status if meta else "exif_failed",
                "pipeline_version": PIPELINE_VERSION,
            }
        all_rows[rel_path] = row

    dirty_images = [
        TimedImage(
            image_path=rel_path,
            shoot_id=row["shoot_id"],
            camera_model=row["camera_model"],
            capture_timestamp=row["capture_timestamp"],
            subsecond=row["subsecond"],
        )
        for rel_path, row in all_rows.items()
        if row["shoot_id"] in dirty_shoots
    ]
    for rel_path, assignment in assign_bursts(dirty_images, gap_seconds=gap_seconds).items():
        all_rows[rel_path]["burst_id"] = assignment.burst_id
        all_rows[rel_path]["sequence_in_burst"] = assignment.sequence_in_burst
        all_rows[rel_path]["burst_size"] = assignment.burst_size

    manifest = pd.DataFrame(list(all_rows.values()), columns=MANIFEST_COLUMNS)
    return manifest, issues


def save_manifest(manifest: pd.DataFrame, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
