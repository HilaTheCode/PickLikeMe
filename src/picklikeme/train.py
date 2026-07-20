from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ProjectConfig
from .dataset import LabelDataset
from .model import PreferenceHead


class ImageTensorDataset(Dataset):
    def __init__(self, dataset: LabelDataset, loader):
        self.dataset = dataset
        self.loader = loader

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        image = self.loader.load_image(item.image_path)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        target = torch.tensor(item.preference if item.preference != 0.0 else item.label, dtype=torch.float32)
        return image_tensor, target


def train(config: ProjectConfig, loader) -> PreferenceHead:
    dataset = LabelDataset(config.labels_path, config.raw_root)
    data_loader = DataLoader(ImageTensorDataset(dataset, loader), batch_size=config.batch_size, shuffle=True)

    model = PreferenceHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(config.epochs):
        for images, labels in data_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()

    return model


def rank_dataset(model: nn.Module, dataset: LabelDataset, loader, device: str = "cpu") -> list[tuple[str, float]]:
    model.eval()
    scored: list[tuple[str, float]] = []
    with torch.no_grad():
        for item in dataset:
            image = loader.load_image(item.image_path)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float()
            logits = model(image_tensor.to(device))
            tensor = logits.squeeze(-1).squeeze(0)
            score = float(tensor.mean().cpu().item()) if tensor.ndim > 0 else float(tensor.cpu().item())
            scored.append((Path(item.image_path).name, score))

    scored.sort(key=lambda entry: entry[1], reverse=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a personal preference model")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--labels", default="data/labels.csv")
    args = parser.parse_args()

    config = ProjectConfig(raw_root=args.raw_root, labels_path=args.labels)
    from .raw_io import RawImageLoader

    loader = RawImageLoader(config.raw_root)
    model = train(config, loader)
    dataset = LabelDataset(config.labels_path, config.raw_root)
    ranked = rank_dataset(model, dataset, loader, device="cpu")
    print("Top-ranked images:")
    for image_name, score in ranked[:10]:
        print(f"{image_name}: {score:.4f}")


if __name__ == "__main__":
    main()
