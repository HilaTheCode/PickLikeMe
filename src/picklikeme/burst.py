from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class BurstEntry:
    path: str
    timestamp: str
    burst_id: Optional[str] = None


def _parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def reconstruct_bursts(entries: List[BurstEntry], max_gap_seconds: float = 2.0) -> List[List[BurstEntry]]:
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda entry: _parse_timestamp(entry.timestamp))
    bursts: List[List[BurstEntry]] = []
    current_burst: List[BurstEntry] = []

    for entry in sorted_entries:
        current_time = _parse_timestamp(entry.timestamp)
        if not current_burst:
            current_burst = [entry]
            continue

        last_time = _parse_timestamp(current_burst[-1].timestamp)
        if (current_time - last_time).total_seconds() <= max_gap_seconds:
            current_burst.append(entry)
        else:
            bursts.append(current_burst)
            current_burst = [entry]

    if current_burst:
        bursts.append(current_burst)

    return bursts
