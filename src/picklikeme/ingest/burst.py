"""Burst reconstruction from capture timestamps.

Keep/Reject sorting physically separates frames that were originally one
continuous burst, so bursts must be rebuilt from EXIF capture time rather
than folder membership. Frames are grouped per (shoot, camera model) since
two bodies can be in use on the same outing, sorted by capture time with
subsecond precision (filename as a tie-breaker for identical timestamps),
and split into a new burst wherever the gap to the previous frame exceeds
a threshold.

Camera file-sequence counters are not used for clustering: they reset
across card reformats and multi-day archives and aren't comparable across
vendors, so they're a weaker signal than the actual capture-time gap.

Frames with no usable timestamp are made singleton bursts rather than
grouped with anything else that would be a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


@dataclass
class TimedImage:
    image_path: str
    shoot_id: str
    camera_model: str | None
    capture_timestamp: str | None
    subsecond: int | None


@dataclass
class BurstAssignment:
    burst_id: str
    sequence_in_burst: int
    burst_size: int


def parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def assign_bursts(images: list[TimedImage], gap_seconds: float = 1.5) -> dict[str, BurstAssignment]:
    groups: dict[tuple[str, str | None], list[TimedImage]] = {}
    for img in images:
        groups.setdefault((img.shoot_id, img.camera_model), []).append(img)

    burst_members: dict[str, list[str]] = {}

    for (shoot_id, _camera_model), group in groups.items():
        parsed = [(img, parse_timestamp(img.capture_timestamp)) for img in group]
        parsed.sort(key=lambda pair: (pair[1] is None, pair[1] or datetime.min, pair[0].subsecond or 0, pair[0].image_path))

        burst_index = 0
        current_burst_id = ""
        prev_combined: float | None = None

        for img, dt in parsed:
            if dt is None:
                burst_index += 1
                current_burst_id = f"{shoot_id}::b{burst_index:04d}"
                prev_combined = None
            else:
                combined = dt.timestamp() + (img.subsecond or 0) / 1000.0
                if prev_combined is None or combined - prev_combined > gap_seconds:
                    burst_index += 1
                    current_burst_id = f"{shoot_id}::b{burst_index:04d}"
                prev_combined = combined
            burst_members.setdefault(current_burst_id, []).append(img.image_path)

    result: dict[str, BurstAssignment] = {}
    for burst_id, members in burst_members.items():
        for idx, image_path in enumerate(members, start=1):
            result[image_path] = BurstAssignment(
                burst_id=burst_id, sequence_in_burst=idx, burst_size=len(members)
            )
    return result
