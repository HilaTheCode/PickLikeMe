from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ProjectConfig
from .dataset import FolderLabelDataset, LabelDataset
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


def train(config: ProjectConfig, loader, dataset=None) -> PreferenceHead:
    if dataset is None:
        dataset = LabelDataset(config.labels_path, config.raw_root)
    data_loader = DataLoader(ImageTensorDataset(dataset, loader), batch_size=config.batch_size, shuffle=True)

    print(f"Starting training with {len(dataset)} relevant images")
    print(f"Loaded {len(dataset)} images from the accepted/rejected roots")
    model = PreferenceHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch + 1}/{config.epochs}")
        processed_batches = 0
        for batch_idx, (images, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            processed_batches += 1
            print(f"  processed batch {processed_batches} (loss={loss.item():.4f})")

    return model


def rank_dataset(model: nn.Module, dataset, loader, device: str = "cpu") -> list[tuple[str, float, int, str]]:
    model.eval()
    scored: list[tuple[str, float, int, str]] = []
    with torch.no_grad():
        processed_images = 0
        for idx, item in enumerate(dataset):
            image = loader.load_image(item.image_path)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float()
            logits = model(image_tensor.to(device))
            tensor = logits.squeeze(-1).squeeze(0)
            score = float(tensor.mean().cpu().item()) if tensor.ndim > 0 else float(tensor.cpu().item())
            scored.append((Path(item.image_path).name, score, int(item.label), str(item.image_path)))
            processed_images += 1
            print(f"  ranked image {processed_images}/{len(dataset)}: {Path(item.image_path).name}")

    scored.sort(key=lambda entry: entry[1], reverse=True)
    return scored


def write_results_csv(
    output_path: str | Path,
    dataset,
    ranked: list[tuple[str, float, int, str]],
    select_root: str,
    reject_root: str,
    max_rows: int = 1000,
) -> list[Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    detected_sequences = dataset.count_sequences()

    files_written: list[Path] = []
    rows_per_file = max_rows - 7
    if rows_per_file <= 0:
        rows_per_file = max_rows

    row_count = 0
    for file_index, chunk in enumerate(
        [ranked[i : i + rows_per_file] for i in range(0, len(ranked), rows_per_file)],
        start=0,
    ):
        target_path = output if file_index == 0 else output.with_name(f"{output.stem}_{file_index}{output.suffix}")
        with target_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerow(["select_root", select_root])
            writer.writerow(["reject_root", reject_root])
            writer.writerow(["relevant_images", len(dataset)])
            writer.writerow(["detected_sequences", detected_sequences])
            writer.writerow([])
            writer.writerow(["rank", "image_path", "score", "label"])
            for rank_offset, entry in enumerate(chunk, start=1):
                image_path = str(entry[3]) if len(entry) >= 4 else str(entry[0])
                score = entry[1]
                label = entry[2]
                writer.writerow([row_count + rank_offset, image_path, f"{score:.6f}", label])
        files_written.append(target_path)
        row_count += len(chunk)

    print(f"Wrote {len(files_written)} ranking result file(s) to {output.parent}")
    return files_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a personal preference model")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--select-root", default=None)
    parser.add_argument("--reject-root", default=None)
    parser.add_argument("--output-csv", default="training_results.csv")
    parser.add_argument("--max-rows", type=int, default=1000)
    args = parser.parse_args()

    if not args.select_root or not args.reject_root:
        raise SystemExit("Both --select-root and --reject-root are required.")

    raw_root = args.raw_root or args.select_root
    config = ProjectConfig(raw_root=raw_root, labels_path=args.labels)
    from .raw_io import RawImageLoader

    loader = RawImageLoader(config.raw_root)
    dataset = FolderLabelDataset(select_root=args.select_root, reject_root=args.reject_root, raw_root=raw_root)
    model = train(config, loader, dataset=dataset)
    ranked = rank_dataset(model, dataset, loader, device="cpu")
    print(f"Detected sequences: {dataset.count_sequences()}")
    output_paths = write_results_csv(args.output_csv, dataset, ranked, args.select_root, args.reject_root, max_rows=args.max_rows)
    print("Top-ranked images:")
    for rank, entry in enumerate(ranked[:10], start=1):
        image_name = entry[0]
        score = entry[1]
        print(f"{rank}. {image_name}: {score:.4f}")
    print(f"CSV written to {output_paths[0]}")
    if len(output_paths) > 1:
        print(f"Additional CSV files: {', '.join(str(path) for path in output_paths[1:])}")


if __name__ == "__main__":
    main()
