"""Scan a physical Keep/Reject archive tree into labeled RAW file records.

The archive is organized as one folder per shoot, each containing a Keep/
subfolder and a Reject/ subfolder of RAW files. Naming has drifted over
years of manual sorting, so subfolder names are matched against an alias
list instead of an exact string, and anything that doesn't match is
reported rather than silently skipped or crashed on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

RAW_EXTENSIONS = {".cr3", ".nef", ".arw"}


def _iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS:
            yield path

KEEP_ALIASES = {"keep", "keeps", "keeper", "keepers", "select", "selects"}
REJECT_ALIASES = {"reject", "rejects", "rejected", "discard", "discards", "trash"}


@dataclass(frozen=True)
class ScannedImage:
    path: Path
    shoot_id: str
    label: int  # 1 = keep, 0 = reject
    raw_format: str


@dataclass
class ScanIssues:
    unmatched_subfolders: list[Path] = field(default_factory=list)
    incomplete_shoots: list[str] = field(default_factory=list)
    duplicate_filenames: list[tuple[str, str, str]] = field(default_factory=list)


def _classify(dirname: str) -> int | None:
    lowered = dirname.strip().lower()
    if lowered in KEEP_ALIASES:
        return 1
    if lowered in REJECT_ALIASES:
        return 0
    return None


def scan_archive(archive_root: Path) -> tuple[list[ScannedImage], ScanIssues]:
    images: list[ScannedImage] = []
    issues = ScanIssues()

    for shoot_dir in sorted(p for p in archive_root.iterdir() if p.is_dir()):
        shoot_id = shoot_dir.name
        label_dirs: dict[int, Path] = {}
        for sub in shoot_dir.iterdir():
            if not sub.is_dir():
                continue
            label = _classify(sub.name)
            if label is None:
                issues.unmatched_subfolders.append(sub)
                continue
            label_dirs[label] = sub

        if len(label_dirs) < 2:
            issues.incomplete_shoots.append(shoot_id)

        seen_names: dict[str, int] = {}
        for label, label_dir in label_dirs.items():
            for file in sorted(label_dir.iterdir()):
                if not file.is_file() or file.suffix.lower() not in RAW_EXTENSIONS:
                    continue
                images.append(
                    ScannedImage(
                        path=file,
                        shoot_id=shoot_id,
                        label=label,
                        raw_format=file.suffix.lower().lstrip("."),
                    )
                )
                other_label = seen_names.get(file.name)
                if other_label is not None and other_label != label:
                    other_dir = label_dirs.get(other_label)
                    issues.duplicate_filenames.append(
                        (shoot_id, str(file), str(other_dir / file.name) if other_dir else "")
                    )
                seen_names[file.name] = label

    return images, issues


def _extract_sequence_token(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def _sequence_base_path(path: Path, root: Path) -> str:
    rel_path = path.relative_to(root)
    parent_parts = rel_path.parts[:-1]
    return "/".join(parent_parts) if parent_parts else root.name


def scan_select_reject_roots(select_root: Path | str, reject_root: Path | str) -> tuple[list[ScannedImage], ScanIssues]:
    select_root = Path(select_root).resolve()
    reject_root = Path(reject_root).resolve()

    if not select_root.exists() or not reject_root.exists():
        raise FileNotFoundError(f"Select or reject root does not exist: {select_root} / {reject_root}")

    images: list[ScannedImage] = []
    issues = ScanIssues()

    all_entries = [
        (path, _sequence_base_path(path, select_root), select_root)
        for path in _iter_image_files(select_root)
    ] + [
        (path, _sequence_base_path(path, reject_root), reject_root)
        for path in _iter_image_files(reject_root)
    ]

    def _collect(root: Path, label: int) -> None:
        if not root.exists():
            return
        for path in _iter_image_files(root):
            base_sequence = _sequence_base_path(path, root)
            numeric_token = _extract_sequence_token(path)
            mtime = path.stat().st_mtime

            for other_path, other_base_path, _other_root in all_entries:
                if other_path == path:
                    continue
                if other_base_path != base_sequence:
                    continue
                other_numeric = _extract_sequence_token(other_path)
                other_mtime = other_path.stat().st_mtime
                if (
                    (numeric_token is not None and other_numeric is not None and abs(numeric_token - other_numeric) <= 3)
                    or abs(mtime - other_mtime) < 1.0
                ):
                    base_sequence = other_base_path
                    break

            images.append(
                ScannedImage(
                    path=path,
                    shoot_id=base_sequence,
                    label=label,
                    raw_format=path.suffix.lower().lstrip("."),
                )
            )

    _collect(select_root, 1)
    _collect(reject_root, 0)

    return images, issues
