"""Lightweight application event bus for desktop components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Event:
    name: str
    payload: Any | None = None


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}

    def subscribe(self, event_name: str, callback: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_name, []).append(callback)

    def publish(self, event_name: str, payload: Any | None = None) -> None:
        event = Event(event_name, payload)
        for callback in list(self._subscribers.get(event_name, [])):
            callback(event)
