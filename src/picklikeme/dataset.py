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


def load_table(table_path: str | Path) -> pd.DataFrame:
    """Load a manifest-shaped table; the parquet manifest is the canonical
    source, CSV is accepted for small fixtures and ad-hoc inputs."""
    table_path = Path(table_path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


class PathSuffixIndex:
    """Resolve per-image values (burst_id, split, ...) for absolute paths by
    matching against the relative paths stored in the manifest or split table.

    Filename-only joins are unsafe here: camera file counters reset across
    card reformats, so the same basename recurs across shoots in a multi-year
    archive. A lookup succeeds only when the CSV rows whose relative path is
    a suffix of the queried path all agree on a single value; anything
    ambiguous resolves to None rather than a guess.
    """

    def __init__(self):
        self._by_name: dict[str, list[tuple[tuple[str, ...], str]]] = {}

    @classmethod
    def from_table(cls, table_path: str | Path, value_column: str) -> "PathSuffixIndex":
        index = cls()
        frame = load_table(table_path)
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
    def __init__(self, manifest_path: str, raw_root: str):
        self.manifest_path = Path(manifest_path)
        self.raw_root = Path(raw_root)
        self.frame = load_table(self.manifest_path)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> ImageLabel:
        row = self.frame.iloc[index]
        burst = row.get("burst_id", None)
        preference = row.get("preference", 0.0)
        return ImageLabel(
            image_path=str(self.raw_root / row["image_path"]),
            label=int(row["label"]),
            burst_id=None if burst is None or pd.isna(burst) or str(burst) == "" else str(burst),
            preference=0.0 if preference is None or pd.isna(preference) else float(preference),
        )


def _count_parent_sequences(items: list[ImageLabel]) -> int:
    """Count distinct immediate-parent folders across items (a rough proxy for
    the number of shooting sequences / bursts when no manifest burst IDs exist)."""
    seen: set[str] = set()
    for item in items:
        path = Path(item.image_path)
        parts = path.parts
        if len(parts) >= 2:
            seen.add("/".join(parts[-2:-1]))
        else:
            seen.add(path.parent.name)
    return len(seen)


class FolderLabelDataset:
    def __init__(
        self,
        select_root: str,
        reject_root: str,
        raw_root: str,
        manifest_path: Optional[str] = None,
    ):
        self.select_root = Path(select_root).resolve()
        self.reject_root = Path(reject_root).resolve()
        self.raw_root = Path(raw_root).resolve()
        self.items: list[ImageLabel] = []

        burst_index = (
            PathSuffixIndex.from_table(manifest_path, "burst_id")
            if manifest_path is not None
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
        return _count_parent_sequences(self.items)


ALLOWED_RAW_EXTENSIONS = {".arw", ".nef", ".cr3"}


class UnlabeledImageDataset:
    """A flat folder of RAW images to score with an already-trained model.

    Unlike FolderLabelDataset there are no keep/reject labels — every item is
    label 0 purely as a placeholder (it is never used for training here, only
    ranking). Used by picklikeme.rank to score a directory the model has never
    seen. Exposes the same len/__getitem__/count_sequences surface that
    rank_dataset and write_results_csv rely on.
    """

    def __init__(self, image_paths: list[str]):
        self.items: list[ImageLabel] = [ImageLabel(image_path=str(p), label=0) for p in image_paths]

    @classmethod
    def from_folder(cls, input_folder: str | Path) -> "UnlabeledImageDataset":
        root = Path(input_folder)
        paths = sorted(
            str(p)
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_RAW_EXTENSIONS
        )
        return cls(paths)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ImageLabel:
        return self.items[index]

    def count_sequences(self) -> int:
        return _count_parent_sequences(self.items)
