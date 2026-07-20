from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    data_root: str = "data"
    raw_root: str = "data/raw"
    labels_path: str = "data/labels.csv"
    batch_size: int = 16
    image_size: int = 384
    num_workers: int = 4
    learning_rate: float = 1e-4
    epochs: int = 20
    device: str = "cuda"
