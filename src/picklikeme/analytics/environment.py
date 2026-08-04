"""Environment facts every recorded run wants, regardless of which
algorithm produced it - the application build, the exact commit, and the
actual hardware used. Moved here (out of `species.experiment`, which had it
first but has nothing species-specific about it) so ranking runs
(`rank.py`, `ranking.classic`) can report "exactly how was this run
produced" just as precisely as a species-classification run already does,
instead of only recording a bare device string.

Every field that cannot be resolved (no git repo, packaged install with no
version metadata, no GPU) is `None`, never guessed - matching this
project's own "explicit unknown, never fabricated" standard applied
throughout `species.experiment`.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def resolve_application_version() -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version("pick-likeme")
    except Exception:  # noqa: BLE001 - version reporting must never break a run
        return None


def resolve_git_commit() -> str | None:
    """Best-effort - a packaged/non-git install must not fail this."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def resolve_gpu_name(torch_module) -> str | None:
    try:
        if torch_module.cuda.is_available():
            return torch_module.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - a GPU query must never break a run
        pass
    return None


def resolve_environment_info() -> dict[str, object]:
    """`application_version`/`git_commit`/`gpu_name`/`cuda_available` - the
    four facts every `Experiment Metadata` display wants regardless of
    which strategy ran. Imports `torch` itself (rather than requiring a
    caller to already have it in hand) so callers that have not otherwise
    imported torch yet (e.g. a ranking strategy before its own model load)
    can still call this directly; safe because every strategy in this
    project already depends on torch."""
    import torch

    return {
        "application_version": resolve_application_version(),
        "git_commit": resolve_git_commit(),
        "gpu_name": resolve_gpu_name(torch),
        "cuda_available": bool(torch.cuda.is_available()),
    }
