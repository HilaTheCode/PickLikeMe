"""Detector boxes for the false-negative overlay.

Where the boxes come from, in order:

1. **The record preprocessing wrote** beside the cached crop. Free - the
   detector had already run, so nothing is recomputed.
2. **The analyzer's own cache**, keyed by content identity, for images that were
   preprocessed before that record existed.
3. **One detection pass**, for images in neither - and only for the handful of
   false negatives a report shows (60 by default), never for the dataset.

Step 3 exists because it has to: preprocessing did not record detections until
now, so on an existing 55k-image cache there is nothing to reuse. It is not
inference "solely for visualization" - it backfills the record that should have
been there, and the result is cached so it happens once per image, ever.

The cache lives in the analyzer's own database. The analyzer stays read-only
with respect to the crop cache: it never writes a sidecar there.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..bird_crop import coco_class_name, read_detections
from ..config import PROJECT_ROOT
from ..identity import IdentityUnavailable

# Its own database, deliberately separate from the annotation knowledge base:
# these boxes are derived data that can always be recomputed, while an
# annotation is irreplaceable. Lives under cache/ for the same reason.
DEFAULT_DETECTIONS_DB = PROJECT_ROOT / "cache" / "analyzer_detections.db"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Box:
    """One detector box in full-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: int
    selected: bool = False

    @property
    def class_name(self) -> str:
        return coco_class_name(self.label)

    def as_dict(self) -> dict:
        return {
            "box": [self.x1, self.y1, self.x2, self.y2],
            "score": self.score,
            "label": self.label,
            "class_name": self.class_name,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class DetectionRecord:
    """Every box for one image, plus the frame they are measured against."""

    boxes: list[Box]
    source_size: tuple[int, int] | None  # (width, height)
    origin: str  # "preprocess" | "cache" | "detected" | "unavailable"

    @property
    def selected(self) -> Box | None:
        return next((box for box in self.boxes if box.selected), None)

    @property
    def others(self) -> list[Box]:
        return [box for box in self.boxes if not box.selected]

    def as_dict(self) -> dict:
        return {
            "origin": self.origin,
            "source_size": list(self.source_size) if self.source_size else None,
            "boxes": [box.as_dict() for box in self.boxes],
        }


EMPTY = DetectionRecord(boxes=[], source_size=None, origin="unavailable")


def _from_payload(payload: dict, origin: str) -> DetectionRecord:
    """Turn a stored record into boxes, marking which one was cropped."""
    selected = payload.get("selected") or {}
    selected_box = tuple(selected.get("box") or ()) if selected else ()
    boxes: list[Box] = []
    for entry in payload.get("detections") or []:
        coords = entry.get("box") or []
        if len(coords) != 4:
            continue
        is_selected = bool(selected_box) and tuple(coords) == selected_box
        boxes.append(
            Box(*[float(v) for v in coords], score=float(entry.get("score", 0.0)),
                label=int(entry.get("label", 0)), selected=is_selected)
        )
    # A record written before all-detections existed holds only the winner.
    if selected and not any(box.selected for box in boxes) and len(selected_box) == 4:
        boxes.append(
            Box(*[float(v) for v in selected_box], score=float(selected.get("score", 0.0)),
                label=int(selected.get("label", 0)), selected=True)
        )
    size = payload.get("source_size")
    return DetectionRecord(
        boxes=boxes,
        source_size=(int(size[0]), int(size[1])) if size and len(size) == 2 else None,
        origin=origin,
    )


class DetectionCache:
    """Boxes per image, memoised in the analyzer's own SQLite database."""

    def __init__(self, db_path: str | Path, crop_cache_dir: str | Path):
        self.db_path = Path(db_path)
        self.crop_cache_dir = Path(crop_cache_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_cache (
                    image_hash TEXT PRIMARY KEY,
                    payload    TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
        self.detector = None  # created lazily, only if something must be detected
        self.detected_count = 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DetectionCache":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _identity(self, image_path: str | Path) -> str | None:
        from ..identity import image_identity

        try:
            return image_identity(image_path)
        except IdentityUnavailable:
            return None

    def _cached(self, digest: str) -> DetectionRecord | None:
        row = self._conn.execute(
            "SELECT payload FROM detection_cache WHERE image_hash = ?", (digest,)
        ).fetchone()
        if row is None:
            return None
        try:
            return _from_payload(json.loads(row["payload"]), "cache")
        except json.JSONDecodeError:
            return None

    def _store(self, digest: str, payload: dict) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO detection_cache(image_hash, payload) VALUES (?, ?)",
                (digest, json.dumps(payload)),
            )

    def _detect(self, image_path: str | Path, conf_threshold: float, device: str) -> dict | None:
        """Run the detector once for an image nothing recorded."""
        from ..bird_crop import BirdDetector
        from ..raw_io import RawImageLoader

        if self.detector is None:
            logger.info("Loading the detector to backfill boxes for un-recorded images")
            self.detector = BirdDetector(device=device, conf_threshold=conf_threshold)
        try:
            frame = RawImageLoader(raw_root=".", resize_mode="letterbox")._decode_full_frame(
                str(image_path)
            )
            best, accepted = self.detector.detect_with_all(frame)
        except Exception as exc:  # noqa: BLE001 - one image must not lose the report
            logger.debug("Could not detect on %s: %s", image_path, exc)
            return None

        self.detected_count += 1
        height, width = frame.shape[:2]
        return {
            "version": 1,
            "source_size": [width, height],
            "selected": None
            if best is None
            else {"box": list(best.box), "score": best.score, "label": best.label},
            "detections": [
                {"box": list(d.box), "score": d.score, "label": d.label} for d in accepted
            ],
        }

    def get(
        self,
        image_path: str | Path,
        *,
        conf_threshold: float = 0.30,
        device: str = "cpu",
        allow_detect: bool = True,
    ) -> DetectionRecord:
        """Boxes for one image, from the cheapest available source."""
        recorded = read_detections(self.crop_cache_dir, image_path)
        if recorded is not None:
            return _from_payload(recorded, "preprocess")

        digest = self._identity(image_path)
        if digest is None:
            return EMPTY

        cached = self._cached(digest)
        if cached is not None:
            return cached

        if not allow_detect:
            return EMPTY

        payload = self._detect(image_path, conf_threshold, device)
        if payload is None:
            return EMPTY
        self._store(digest, payload)
        return _from_payload(payload, "detected")

    def get_many(self, image_paths: Sequence[str], **kwargs) -> dict[str, DetectionRecord]:
        records: dict[str, DetectionRecord] = {}
        for path in image_paths:
            record = self.get(path, **kwargs)
            if record.boxes:
                records[str(path)] = record
        if self.detected_count:
            logger.info(
                "Detected boxes for %d image(s) with no recorded detections; cached for next time",
                self.detected_count,
            )
        return records
