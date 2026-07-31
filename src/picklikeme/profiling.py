"""Reusable opt-in stage timing for image-processing pipelines.

Enabled by setting the environment variable PICKLIKEME_PROFILE=1. When it is
off, every hook is a no-op that returns a shared null context: production runs
pay nothing, add no CUDA synchronization, and behave exactly as before. That
matters because accurate GPU attribution *requires* synchronizing (a forward
pass is asynchronous, so without a sync its cost silently lands on whichever
later call happens to touch the result), and we do not want that sync in a
normal run.

Timing only. Nothing here changes what is computed or written.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

ENV_VAR = "PICKLIKEME_PROFILE"

# Stage order used in reports: roughly the order work happens per image.
STAGE_ORDER = [
    "file read",
    "raw decode",
    "image decode",
    "detector preprocess",
    "gpu inference",
    "detector postprocess",
    "crop generation",
    "png encode + write",
    "metadata write",
    "cache lookup",
]


@dataclass
class StageStats:
    """A compact summary for one stage, suitable for CLI reporting."""

    name: str
    total_seconds: float
    count: int

    @property
    def mean_ms(self) -> float:
        return (self.total_seconds / self.count * 1000.0) if self.count else 0.0


class _NullContext:
    """Shared zero-cost stand-in used when profiling is disabled."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


_NULL = _NullContext()


class _Stage:
    """Context manager that accumulates elapsed time into a StageTimer."""

    __slots__ = ("timer", "name", "start")

    def __init__(self, timer: "StageTimer", name: str):
        self.timer = timer
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        elapsed = time.perf_counter() - self.start
        self.timer.seconds[self.name] += elapsed
        self.timer.counts[self.name] += 1
        return False


class StageTimer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.wall_start = time.perf_counter()
        self.images = 0

    def stage_stats(self) -> list[StageStats]:
        return [StageStats(name=name, total_seconds=self.seconds[name], count=self.counts[name]) for name in sorted(self.seconds)]

    def reset(self) -> None:
        self.seconds.clear()
        self.counts.clear()
        self.wall_start = time.perf_counter()
        self.images = 0

    def stage(self, name: str):
        return _Stage(self, name) if self.enabled else _NULL

    def cuda_sync(self, torch_module, device: str) -> None:
        """Force pending GPU work to finish so it is charged to the stage that
        issued it. Only ever called while profiling is enabled."""
        if self.enabled and str(device).startswith("cuda"):
            torch_module.cuda.synchronize()

    def image_done(self) -> None:
        self.images += 1

    @property
    def wall_seconds(self) -> float:
        return time.perf_counter() - self.wall_start

    def mean_ms(self, name: str) -> float:
        count = self.counts.get(name, 0)
        return (self.seconds[name] / count * 1000.0) if count else 0.0

    def progress_line(self) -> str:
        """Compact one-liner for the periodic summary."""
        wall = self.wall_seconds
        rate = self.images / wall if wall > 0 else 0.0
        decode = self.seconds["raw decode"] + self.seconds["image decode"] + self.seconds["file read"]
        return (
            f"  timing: {rate:.2f} img/s | decode {self.mean_ms('raw decode') + self.mean_ms('image decode'):.0f}ms "
            f"| infer {self.mean_ms('gpu inference'):.0f}ms | save {self.mean_ms('png encode + write'):.0f}ms "
            f"| decode {decode / wall * 100:.0f}% of wall | gpu busy {self.seconds['gpu inference'] / wall * 100:.1f}%"
        )

    def report(self, title: str = "Preprocessing timing breakdown", *, images: int | None = None) -> str:
        """Full breakdown: share of wall-clock per stage, plus the residual."""
        wall = self.wall_seconds
        effective_images = self.images if images is None else images
        if wall <= 0 or not effective_images:
            return "No profiling data collected."

        measured = sum(self.seconds.values())
        lines = [
            "",
            title,
            "=" * len(title),
            f"  images:              {effective_images:,}",
            f"  wall clock:          {wall:.1f}s",
            f"  throughput:          {effective_images / wall:.2f} img/s ({wall / effective_images * 1000:.0f} ms/image)",
            f"  GPU busy:            {self.seconds['gpu inference']:.1f}s "
            f"({self.seconds['gpu inference'] / wall * 100:.1f}% of wall)",
            "",
            f"  {'stage':<22}{'share':>8}{'total':>11}{'mean/img':>11}{'calls':>9}",
        ]
        named = [name for name in STAGE_ORDER if self.counts.get(name)]
        named += sorted(name for name in self.seconds if name not in STAGE_ORDER)
        for name in named:
            seconds = self.seconds[name]
            lines.append(
                f"  {name:<22}{seconds / wall * 100:>7.1f}%{seconds:>10.1f}s"
                f"{self.mean_ms(name):>10.1f}ms{self.counts[name]:>9,}"
            )
        other = wall - measured
        lines.append(
            f"  {'other / overhead':<22}{other / wall * 100:>7.1f}%{other:>10.1f}s"
            f"{other / effective_images * 1000:>10.1f}ms{'':>9}"
        )
        if measured > wall:
            # Decode runs in a thread pool, so its accumulated time is the sum
            # over workers and legitimately exceeds wall clock. That is the
            # overlap working; shares are no longer a partition of the timeline.
            lines.append(
                f"\n  Note: stages sum to {measured:.1f}s > {wall:.1f}s wall clock because decode"
                f"\n  runs concurrently in the decoder pool. 'mean/img' stays exact;"
                f"\n  'share' is time-in-stage, not a share of a serial timeline."
            )
        return "\n".join(lines)


class PipelineProfiler:
    """Reusable timing helper for end-to-end ranking or other CLI pipelines."""

    def __init__(self, *, images: int, decode_workers: int, device: str) -> None:
        self.images = images
        self.decode_workers = decode_workers
        self.device = device
        self.started_at = time.perf_counter()
        self.stage_times: dict[str, float] = defaultdict(float)
        self.stage_counts: dict[str, int] = defaultdict(int)
        self.cuda_available = False
        self.cuda_peak_bytes = 0
        self.torch_version = "unknown"

    def record_stage(self, name: str, seconds: float, count: int = 1) -> None:
        self.stage_times[name] += seconds
        self.stage_counts[name] += count

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_stage(name, time.perf_counter() - start)

    def capture_environment(self, *, torch_module=None) -> None:
        self.torch_version = getattr(torch_module, "__version__", "unknown") if torch_module is not None else "unknown"
        try:
            import torch

            self.cuda_available = bool(torch.cuda.is_available())
            if self.cuda_available:
                torch.cuda.reset_peak_memory_stats()
                self.cuda_peak_bytes = 0
        except Exception:  # noqa: BLE001 - diagnostics only
            self.cuda_available = False
            self.cuda_peak_bytes = 0

    def capture_peak_memory(self, *, torch_module=None) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                self.cuda_peak_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:  # noqa: BLE001 - diagnostics only
            self.cuda_peak_bytes = 0

    def runtime_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def build_report(self, *, total_runtime: float | None = None) -> str:
        total_runtime = self.runtime_seconds() if total_runtime is None else total_runtime
        if self.images <= 0:
            total_runtime = max(total_runtime, 0.0)

        def fmt_seconds(value: float) -> str:
            return f"{value:.3f}s" if value >= 1.0 else f"{value * 1000:.1f}ms"

        def fmt_avg(value: float) -> str:
            return f"{value:.3f}s" if value >= 1.0 else f"{value * 1000:.1f}ms"

        lines = [
            "",
            "==================================================",
            "RANKING PERFORMANCE REPORT",
            "==================================================",
            "",
            f"Images processed: {self.images}",
            f"Total runtime: {total_runtime:.1f}s",
            f"Average/image: {total_runtime / self.images * 1000:.1f}ms" if self.images else "Average/image: n/a",
            f"Throughput: {self.images / total_runtime:.2f} images/sec" if total_runtime > 0 else "Throughput: n/a",
            "",
            f"{'Stage':<30}{'Total':>12}{'Avg/Image':>12}",
            "-" * 56,
        ]
        for name in [
            "File enumeration",
            "RAW loading",
            "RAW decoding",
            "Preprocessing",
            "Bird detection",
            "Crop generation",
            "Crop cache writing",
            "Ranking inference",
            "CSV generation",
        ]:
            total = self.stage_times.get(name, 0.0)
            avg = total / self.images if self.images else 0.0
            lines.append(f"{name:<30}{fmt_seconds(total):>12}{fmt_avg(avg):>12}")
        lines.extend([
            "",
            "GPU:",
            f"  Device: {self.device}",
            f"  Peak memory: {self.cuda_peak_bytes / (1024 ** 2):.1f} MiB" if self.cuda_peak_bytes else "  Peak memory: n/a",
            f"  CUDA: {'available' if self.cuda_available else 'unavailable'}",
            "",
            "CPU:",
            f"  Decode workers: {self.decode_workers}",
            "",
            f"Torch: {self.torch_version}",
            "",
        ])
        return "\n".join(lines)


PROFILE = StageTimer(enabled=os.environ.get(ENV_VAR, "").strip() not in ("", "0", "false", "False"))
