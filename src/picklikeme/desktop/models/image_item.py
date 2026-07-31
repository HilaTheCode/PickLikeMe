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
    # The file's own EXIF capture date/time (ISO-8601), or None if it has
    # none - see ReviewImage.captured_at. ISO-8601 sorts lexicographically
    # in chronological order, so gallery sort-by-capture-time needs no
    # date parsing.
    captured_at: str | None = None

    @property
    def display_name(self) -> str:
        return Path(self.path).name if self.path else self.file_name
