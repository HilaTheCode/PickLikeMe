from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig
from .dataset import LabelDataset


def build_dataset(config: ProjectConfig) -> LabelDataset:
    return LabelDataset(manifest_path=config.manifest_path, raw_root=config.raw_root)


def ensure_data_dirs(config: ProjectConfig) -> None:
    Path(config.raw_root).mkdir(parents=True, exist_ok=True)
    Path(config.data_root).mkdir(parents=True, exist_ok=True)
