"""Opt-in stage timing for the preprocessing pipeline.

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

    def report(self) -> str:
        """Full breakdown: share of wall-clock per stage, plus the residual."""
        wall = self.wall_seconds
        if wall <= 0 or not self.images:
            return "No profiling data collected."

        measured = sum(self.seconds.values())
        lines = [
            "",
            "Preprocessing timing breakdown",
            "==============================",
            f"  images:              {self.images:,}",
            f"  wall clock:          {wall:.1f}s",
            f"  throughput:          {self.images / wall:.2f} img/s ({wall / self.images * 1000:.0f} ms/image)",
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
            f"{other / self.images * 1000:>10.1f}ms{'':>9}"
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


PROFILE = StageTimer(enabled=os.environ.get(ENV_VAR, "").strip() not in ("", "0", "false", "False"))
