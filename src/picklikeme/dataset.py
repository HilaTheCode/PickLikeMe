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
