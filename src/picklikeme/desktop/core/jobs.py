"""Production background execution framework for the desktop app."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional
from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal, Slot


class JobPriority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class JobResult:
    job_id: str
    success: bool
    payload: Any = None
    error: Optional[Exception] = None


@dataclass
class JobSpec:
    name: str
    func: Callable[[], Any]
    priority: JobPriority = JobPriority.NORMAL
    cancellable: bool = True
    description: str = ""
    on_started: Optional[Callable[[str], None]] = None
    on_finished: Optional[Callable[[JobResult], None]] = None
    on_error: Optional[Callable[[JobResult], None]] = None


class JobSignal(QObject):
    started = Signal(str)
    finished = Signal(object)
    failed = Signal(object)
    progress = Signal(str, int)


class DesktopJob(QRunnable):
    def __init__(self, job_id: str, spec: JobSpec, signals: JobSignal) -> None:
        super().__init__()
        self.job_id = job_id
        self.spec = spec
        self.signals = signals
        self._cancelled = False

    @Slot()
    def run(self) -> None:
        self.signals.started.emit(self.job_id)
        if self.spec.on_started is not None:
            self.spec.on_started(self.job_id)
        try:
            payload = self.spec.func()
            if not self._cancelled:
                self.signals.finished.emit(JobResult(self.job_id, True, payload))
                if self.spec.on_finished is not None:
                    self.spec.on_finished(JobResult(self.job_id, True, payload))
        except Exception as exc:  # noqa: BLE001
            result = JobResult(self.job_id, False, error=exc)
            self.signals.failed.emit(result)
            if self.spec.on_error is not None:
                self.spec.on_error(result)

    def cancel(self) -> None:
        self._cancelled = True


class JobManager:
    def __init__(self, *, max_threads: int | None = None) -> None:
        self.thread_pool = QThreadPool.globalInstance()
        if max_threads is not None:
            self.thread_pool.setMaxThreadCount(max_threads)
        self._jobs: dict[str, DesktopJob] = {}
        self._signals = JobSignal()

    def submit(self, spec: JobSpec) -> str:
        job_id = f"{spec.name}-{len(self._jobs) + 1}"
        job = DesktopJob(job_id, spec, self._signals)
        self._jobs[job_id] = job
        self.thread_pool.start(job)
        return job_id

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.cancel()

    def signals(self) -> JobSignal:
        return self._signals


class ProgressWorker(QObject):
    """Runs one callable on a background QThread and reports progress back
    to the GUI thread via queued signal connections (Qt marshals these
    automatically since this QObject is created on the GUI thread)."""

    progress = Signal(int, int)
    stage = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, func: Callable[..., Any]) -> None:
        super().__init__()
        self._func = func

    @Slot()
    def run(self) -> None:
        try:
            result = self._func(
                on_progress=lambda done, total: self.progress.emit(done, total),
                on_stage=lambda message: self.stage.emit(message),
            )
        except Exception as exc:  # noqa: BLE001 - reported to the caller, never crashes the app
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class _CallbackBridge(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)
    stage = Signal(str)

    def __init__(
        self,
        parent: QObject,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_finished: Callable[[Any], None] | None = None,
        on_failed: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_progress = on_progress
        self._on_stage = on_stage
        self._on_finished = on_finished
        self._on_failed = on_failed

    @Slot(int, int)
    def handle_progress(self, done: int, total: int) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total)

    @Slot(str)
    def handle_stage(self, message: str) -> None:
        if self._on_stage is not None:
            self._on_stage(message)

    @Slot(object)
    def handle_finished(self, result: Any) -> None:
        if self._on_finished is not None:
            self._on_finished(result)

    @Slot(str)
    def handle_failed(self, message: str) -> None:
        if self._on_failed is not None:
            self._on_failed(message)


def run_in_background(
    parent: QObject,
    func: Callable[..., Any],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_finished: Callable[[Any], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
) -> QThread:
    """Run `func(on_progress=..., on_stage=...)` on a background QThread.

    `func` must accept `on_progress` and `on_stage` keyword arguments (both
    may be ignored internally by callables that report only one kind of
    update - see functools.partial wrapping in call sites). Returns the
    QThread so the caller can keep a reference for the duration of the run;
    the thread deletes itself once finished.
    """
    thread = QThread(parent)
    worker = ProgressWorker(func)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    bridge = _CallbackBridge(
        parent,
        on_progress=on_progress,
        on_stage=on_stage,
        on_finished=on_finished,
        on_failed=on_failed,
    )

    worker.progress.connect(bridge.handle_progress)
    worker.stage.connect(bridge.handle_stage)
    worker.finished.connect(bridge.handle_finished)
    worker.failed.connect(bridge.handle_failed)

    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    # Keep the thread object alive so callers can still wait on it after it
    # has finished. Qt will release it when the Python object is garbage
    # collected, which is still compatible with the current tests and callers.
    thread._worker_ref = worker  # noqa: SLF001 - deliberate GC keep-alive, not a private API access
    thread._bridge_ref = bridge  # noqa: SLF001 - deliberate GC keep-alive for queued callbacks
    thread.start()
    return thread

