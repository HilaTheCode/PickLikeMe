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
