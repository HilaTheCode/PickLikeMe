import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Project root resolved from this file's location (src/picklikeme/config.py ->
# parents[2] == project root), so paths derived from it are deterministic and
# independent of the directory Python happens to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "model_checkpoint.pt"
DEFAULT_CROP_CACHE_DIR = PROJECT_ROOT / "cache" / "crops"
DEFAULT_INSPECTION_DIR = PROJECT_ROOT / "inspection"

# Every training run (and the analyzer report generated from it) shares one
# timestamped directory here, so a run's ranking CSV, metrics, log and report
# are always found together instead of scattered across the project root.
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "analysis_results"
RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def run_dir_for_timestamp(run_started: datetime) -> Path:
    """The one directory a run's outputs (ranking CSV, metrics, log, and -
    once `picklikeme analyze` is pointed at that CSV - the report) all share."""
    return DEFAULT_RESULTS_DIR / run_started.strftime(RUN_TIMESTAMP_FORMAT)

# Maximum lines per results/ranking CSV before it rolls over to a numbered
# continuation file (`name.csv`, `name_1.csv`, ...). Counts the metrics
# preamble, not just data rows, so one file never exceeds this many lines.
#
# 30,000 keeps a 55k-image run to two files instead of fifty-odd, while staying
# inside the row limits of older spreadsheet tools that were the reason for
# splitting at all. Override per run with --max-rows.
DEFAULT_MAX_CSV_ROWS = 30000


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


class _Tee:
    """Writes to two streams at once. Forwards unknown attributes (`isatty`,
    `encoding`, ...) to `primary` so code that introspects `sys.stdout` -
    tqdm-style progress bars in particular - keeps working unmodified."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._secondary.write(data)
        return self._primary.write(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def __getattr__(self, name):
        return getattr(self._primary, name)


@contextmanager
def tee_stdout_to_file(log_path: str | Path):
    """Duplicate everything printed during the block into `log_path` as well
    as the console, without touching any of the existing print() call sites.

    A run's own log file, saved next to its other outputs, rather than relying
    on the caller to remember to redirect the shell's output.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout


def cli_prefix() -> str:
    """The always-correct way to invoke this CLI, for instructions printed at
    runtime (e.g. "run this next" hints).

    `picklikeme <command>` only works if the console script installed by
    `pip install -e .` happens to be on PATH - not guaranteed even when the
    package *is* installed, since this project keeps two virtualenvs
    (`.venv` CUDA, `.venv-1` CPU-only) and only one's Scripts/bin directory can
    be active at a time. `sys.executable -m picklikeme` instead names the exact
    interpreter already running this process, which by construction has the
    package importable, so a copy-pasted instruction can never point at an
    inactive environment or a script that was never put on PATH.
    """
    return f'"{sys.executable}" -m picklikeme'


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
