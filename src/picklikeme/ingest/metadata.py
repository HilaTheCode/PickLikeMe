"""EXIF metadata extraction via exiftool.

rawpy/libraw decode pixels reliably across CR3/NEF/ARW but expose weak,
inconsistent EXIF metadata, especially for Canon CR3's MP4-style container.
exiftool is the actively maintained cross-vendor tool that gets subsecond
capture time (needed to order frames within a burst shot at 10-20fps) and
lens/exposure fields uniformly across vendors.

Paths are passed to exiftool via an argfile (-@) rather than on the command
line so batches aren't limited by Windows command-line length, which lets a
single process invocation cover thousands of files and keep process-spawn
overhead negligible across a large archive.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

# A global -n would disable print conversion, which also disables -d date
# formatting; per-tag '#' suffixes get numeric values without that side effect.
EXIFTOOL_ARGS = [
    "-json",
    "-d", "%Y-%m-%dT%H:%M:%S",
    "-DateTimeOriginal",
    "-SubSecTimeOriginal",
    "-Model",
    "-LensModel",
    "-ISO#",
    "-ExposureTime#",
    "-FNumber#",
    "-FocalLength#",
]


@dataclass
class ImageMetadata:
    camera_model: str | None
    lens_model: str | None
    capture_timestamp: str | None
    subsecond: int | None
    iso: float | None
    shutter_speed: float | None
    f_number: float | None
    focal_length: float | None
    status: str  # ok / missing_timestamp / missing_subsecond / exif_failed


class ExifToolNotFoundError(RuntimeError):
    pass


def ensure_exiftool_available(exiftool_path: str) -> None:
    try:
        subprocess.run([exiftool_path, "-ver"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExifToolNotFoundError(
            f"exiftool not found or not runnable at '{exiftool_path}'. "
            "Install it from https://exiftool.org (rename exiftool(-k).exe to "
            "exiftool.exe and put it on PATH), or pass --exiftool-path."
        ) from exc


def _chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_entry(entry: dict) -> ImageMetadata:
    subsec_raw = entry.get("SubSecTimeOriginal")
    subsecond = None
    if subsec_raw is not None:
        try:
            subsecond = int(str(subsec_raw)[:3].ljust(3, "0"))
        except ValueError:
            subsecond = None

    timestamp = entry.get("DateTimeOriginal")
    if not timestamp:
        status = "missing_timestamp"
    elif subsecond is None:
        status = "missing_subsecond"
    else:
        status = "ok"

    return ImageMetadata(
        camera_model=entry.get("Model"),
        lens_model=entry.get("LensModel"),
        capture_timestamp=timestamp,
        subsecond=subsecond,
        iso=entry.get("ISO"),
        shutter_speed=entry.get("ExposureTime"),
        f_number=entry.get("FNumber"),
        focal_length=entry.get("FocalLength"),
        status=status,
    )


def _failed_metadata() -> ImageMetadata:
    return ImageMetadata(
        camera_model=None, lens_model=None, capture_timestamp=None, subsecond=None,
        iso=None, shutter_speed=None, f_number=None, focal_length=None, status="exif_failed",
    )


def extract_metadata(
    paths: list[Path], exiftool_path: str = "exiftool", chunk_size: int = 2000
) -> dict[str, ImageMetadata]:
    results: dict[str, ImageMetadata] = {}
    resolved_paths = [p.resolve() for p in paths]

    for chunk in tqdm(_chunked(resolved_paths, chunk_size), desc="Extracting metadata", unit="batch"):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as arg_file:
            for path in chunk:
                arg_file.write(f"{path}\n")
            arg_file_path = Path(arg_file.name)

        try:
            cmd = [exiftool_path, *EXIFTOOL_ARGS, "-@", str(arg_file_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            try:
                entries = json.loads(proc.stdout) if proc.stdout else []
            except json.JSONDecodeError:
                entries = []

            seen: set[str] = set()
            for entry in entries:
                source = entry.get("SourceFile")
                if not source:
                    continue
                key = str(Path(source).resolve())
                results[key] = _parse_entry(entry)
                seen.add(key)

            for path in chunk:
                key = str(path)
                if key not in seen:
                    results[key] = _failed_metadata()
        finally:
            arg_file_path.unlink(missing_ok=True)

    return results
