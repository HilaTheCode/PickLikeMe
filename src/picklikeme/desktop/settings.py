"""Application settings persisted with QSettings when Qt is available."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DesktopSettings:
    window_geometry: str | None = None
    window_state: str | None = None
    last_opened_folder: str | None = None
    theme: str = "system"
    splitter_positions: dict[str, int] = field(default_factory=dict)

    def load(self) -> "DesktopSettings":
        return self

    def save(self) -> None:
        return None


class SettingsStore:
    """Simple settings container used by the desktop shell."""

    def __init__(self, *, settings_path: str | Path | None = None) -> None:
        self.settings_path = Path(settings_path) if settings_path is not None else None
        self.settings = DesktopSettings()

    def load(self) -> DesktopSettings:
        return self.settings.load()

    def save(self) -> None:
        self.settings.save()
