from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class WorkspaceManager:
    current_workspace: Path | None = None
    _close_handlers: list[Callable[[], None]] = field(default_factory=list)

    def open_workspace(self, folder: str | Path, *, on_close: Optional[Callable[[], None]] = None) -> Path:
        folder = Path(folder).expanduser()

        if self.current_workspace is not None and self.current_workspace != folder:
            for handler in self._close_handlers:
                try:
                    handler()
                except Exception:
                    pass
            self._close_handlers.clear()
        self.current_workspace = folder
        if on_close is not None:
            self._close_handlers.append(on_close)
        return self.current_workspace
