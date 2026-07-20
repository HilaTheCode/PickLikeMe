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
    def __init__(self, select_root: str, reject_root: str, raw_root: str):
        self.select_root = Path(select_root).resolve()
        self.reject_root = Path(reject_root).resolve()
        self.raw_root = Path(raw_root).resolve()
        self.items: list[ImageLabel] = []

        allowed_extensions = {".arw", ".nef", ".cr3"}

        for path in sorted(self.select_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                self.items.append(ImageLabel(image_path=str(path), label=1, burst_id=None, preference=1.0))

        for path in sorted(self.reject_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                self.items.append(ImageLabel(image_path=str(path), label=0, burst_id=None, preference=0.0))

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
