"""Reusable command objects for desktop actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..services import ReviewService


class Command:
    def execute(self) -> Any:
        raise NotImplementedError


@dataclass
class OpenFolderCommand(Command):
    service: ReviewService
    folder: str

    def execute(self) -> dict[str, Any]:
        return self.service.open_folder(self.folder)


@dataclass
class KeepImageCommand(Command):
    service: ReviewService
    image_path: str

    def execute(self) -> str:
        return self.service.set_review_status(self.image_path, "keep")


@dataclass
class RejectImageCommand(Command):
    service: ReviewService
    image_path: str

    def execute(self) -> str:
        return self.service.set_review_status(self.image_path, "reject")


@dataclass
class ArrangeCommand(Command):
    service: ReviewService

    def execute(self) -> dict[str, Any]:
        return self.service.arrange(dry_run=True)
