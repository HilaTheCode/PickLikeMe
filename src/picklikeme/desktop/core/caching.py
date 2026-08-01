"""Cache management infrastructure for desktop assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0


class CacheManager:
    """Owns desktop caches without tying them to widgets."""

    def __init__(self, *, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path("cache") / "desktop"
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self._thumbs: dict[str, Any] = {}
        self._previews: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._crops: dict[str, Any] = {}

    def put_thumbnail(self, key: str, value: Any) -> None:
        self._thumbs[key] = value

    def get_thumbnail(self, key: str) -> Any | None:
        if key in self._thumbs:
            self.stats.hits += 1
            return self._thumbs[key]
        self.stats.misses += 1
        return None

    def clear_thumbnails(self) -> None:
        """Drop every cached thumbnail pixmap - both plain and detector-box
        overlaid.

        Needed after a ranking run: `review_thumbnail(with_boxes=True)`
        picks a different on-disk file depending on what is currently
        recorded for an image (see `annotated_thumbnail_path`'s `has_eye`),
        so an overlaid pixmap already sitting in this in-memory cache from
        before a run - e.g. the Gallery's Detector Boxes toggle was on
        while only the AI model had ranked the folder - would otherwise be
        served back forever even after Classic Vision adds eye data for
        the same (path, with_boxes) key. See MainWindow._rank_with_strategy's
        `_on_success`, the only call site.
        """
        self._thumbs.clear()

    def put_preview(self, key: str, value: Any) -> None:
        self._previews[key] = value

    def get_preview(self, key: str) -> Any | None:
        if key in self._previews:
            self.stats.hits += 1
            return self._previews[key]
        self.stats.misses += 1
        return None

    def put_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Any | None:
        if key in self._metadata:
            self.stats.hits += 1
            return self._metadata[key]
        self.stats.misses += 1
        return None

    def put_crop(self, key: str, value: Any) -> None:
        self._crops[key] = value

    def get_crop(self, key: str) -> Any | None:
        if key in self._crops:
            self.stats.hits += 1
            return self._crops[key]
        self.stats.misses += 1
        return None

    def evict(self) -> None:
        self.stats.evictions += 1
