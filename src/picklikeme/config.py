import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Project root resolved from this file's location (src/picklikeme/config.py ->
# parents[2] == project root), so paths derived from it are deterministic and
# independent of the directory Python happens to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "model_checkpoint.pt"
DEFAULT_CROP_CACHE_DIR = PROJECT_ROOT / "cache" / "crops"
DEFAULT_INSPECTION_DIR = PROJECT_ROOT / "inspection"


@contextmanager
def fatal_errors_logged_to_stdout():
    """Print an unhandled exception's traceback to **stdout** before re-raising.

    Long runs are piped to a log, and the usual pipe captures stdout only - so a
    crash previously left a log that simply stopped mid-progress with no
    diagnosis, while the traceback went to a console that may be long gone. This
    puts the traceback in the log regardless of how stderr is redirected.

    The exception still propagates, so the exit code and any outer handling are
    unchanged. Ctrl+C is reported as one line rather than a traceback, since it
    is a deliberate stop and the training loop already checkpoints on it.
    """
    try:
        yield
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
        raise
    except BaseException:
        print("\n" + "=" * 64)
        print("FATAL: run stopped with an unhandled exception")
        print(traceback.format_exc().rstrip())
        print("=" * 64)
        raise


def format_duration(seconds: float) -> str:
    """Compact h/m/s duration for progress and ETA logging. Lives here (a
    dependency-free module) so training and preprocessing format elapsed times
    and ETAs identically."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


@dataclass(frozen=True)
class ProjectConfig:
    data_root: str = "data"
    raw_root: str = "data/raw"
    manifest_path: str = "data/manifest.parquet"
    batch_size: int = 16
    image_size: int = 384
    num_workers: int = 4
    learning_rate: float = 1e-4
    epochs: int = 20
    device: str = "cuda"
