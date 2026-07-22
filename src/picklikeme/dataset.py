from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ImageLabel:
    image_path: str
    label: int
    burst_id: Optional[str] = None
    preference: float = 0.0


class PathSuffixIndex:
    """Resolve per-image values (burst_id, split, ...) for absolute paths by
    matching against the relative paths stored in a manifest-derived CSV.

    Filename-only joins are unsafe here: camera file counters reset across
    card reformats, so the same basename recurs across shoots in a multi-year
    archive. A lookup succeeds only when the CSV rows whose relative path is
    a suffix of the queried path all agree on a single value; anything
    ambiguous resolves to None rather than a guess.
    """

    def __init__(self):
        self._by_name: dict[str, list[tuple[tuple[str, ...], str]]] = {}

    @classmethod
    def from_csv(cls, csv_path: str | Path, value_column: str) -> "PathSuffixIndex":
        index = cls()
        frame = pd.read_csv(csv_path)
        for row in frame.itertuples(index=False):
            value = getattr(row, value_column, None)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            index.add(str(getattr(row, "image_path")), str(value))
        return index

    def add(self, relative_path: str, value: str) -> None:
        parts = tuple(part.lower() for part in Path(relative_path).parts)
        if not parts:
            return
        self._by_name.setdefault(parts[-1], []).append((parts, value))

    def get(self, image_path: str | Path) -> Optional[str]:
        parts = tuple(part.lower() for part in Path(image_path).parts)
        if not parts:
            return None
        candidates = self._by_name.get(parts[-1], [])
        matches = {value for cand, value in candidates if parts[-len(cand):] == cand}
        if len(matches) == 1:
            return next(iter(matches))
        return None


class LabelDataset:
    def __init__(self, labels_path: str, raw_root: str):
        self.labels_path = Path(labels_path)
        self.raw_root = Path(raw_root)
        self.frame = pd.read_csv(self.labels_path)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> ImageLabel:
        row = self.frame.iloc[index]
        return ImageLabel(
            image_path=str(self.raw_root / row["image_path"]),
            label=int(row["label"]),
            burst_id=str(row.get("burst_id", "")) or None,
            preference=float(row.get("preference", 0.0)),
        )


class FolderLabelDataset:
    def __init__(
        self,
        select_root: str,
        reject_root: str,
        raw_root: str,
        burst_labels_path: Optional[str] = None,
    ):
        self.select_root = Path(select_root).resolve()
        self.reject_root = Path(reject_root).resolve()
        self.raw_root = Path(raw_root).resolve()
        self.items: list[ImageLabel] = []

        burst_index = (
            PathSuffixIndex.from_csv(burst_labels_path, "burst_id")
            if burst_labels_path is not None
            else None
        )

        allowed_extensions = {".arw", ".nef", ".cr3"}

        def _burst_id(path: Path) -> Optional[str]:
            return burst_index.get(path) if burst_index is not None else None

        for path in sorted(self.select_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                self.items.append(ImageLabel(image_path=str(path), label=1, burst_id=_burst_id(path), preference=1.0))

        for path in sorted(self.reject_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                self.items.append(ImageLabel(image_path=str(path), label=0, burst_id=_burst_id(path), preference=0.0))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ImageLabel:
        return self.items[index]

    def count_sequences(self) -> int:
        seen: set[str] = set()
        for item in self.items:
            path = Path(item.image_path)
            parts = path.parts
            if len(parts) >= 2:
                seen.add("/".join(parts[-2:-1]))
            else:
                seen.add(path.parent.name)
        return len(seen)
