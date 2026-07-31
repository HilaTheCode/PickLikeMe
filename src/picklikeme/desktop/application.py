"""Application state and coordination for the desktop shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApplicationState:
    """Single source of truth for the desktop application state."""

    current_folder: str | None = None
    current_review_session: Any | None = None
    current_image: str | None = None
    current_selection: list[str] = field(default_factory=list)
    active_filters: dict[str, Any] = field(default_factory=dict)
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    image_count: int = 0
    status_message: str = "Ready"
    gpu_status: str = "CPU"


class WorkerManager:
    """Central infrastructure for background work in the desktop app."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    def register_job(self, name: str, *, description: str | None = None) -> str:
        job_id = f"{name}-{len(self.jobs) + 1}"
        self.jobs[job_id] = {"name": name, "description": description or name, "running": True}
        return job_id

    def complete_job(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["running"] = False

    def clear_completed(self) -> None:
        self.jobs = {job_id: job for job_id, job in self.jobs.items() if job.get("running")}
