"""Image item model for the future Qt gallery view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ImageItem:
    path: str
    file_name: str
    rank: int | None = None
    score: float | None = None
    review_status: str = "neutral"
    ai_suggestion: str | None = None
    selected: bool = False

    @property
    def display_name(self) -> str:
        return Path(self.path).name if self.path else self.file_name
