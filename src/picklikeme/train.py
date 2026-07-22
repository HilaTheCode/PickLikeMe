from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ProjectConfig
from .dataset import FolderLabelDataset, LabelDataset, PathSuffixIndex
from .evaluate import compute_metrics, format_metrics, score_items, write_metrics_json
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


def save_checkpoint(model: nn.Module, optimizer, checkpoint_path: str | Path, epoch: int | None = None) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(payload, checkpoint_path)


def write_progress_state(status_path: str | Path, epoch: int, total_epochs: int, batch: int, total_batches: int | None, loss: float | None, message: str) -> None:
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "loss": loss,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(
    config: ProjectConfig,
    loader,
    dataset=None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    status_path: str | Path | None = None,
    status_interval: timedelta | None = None,
    checkpoint_interval: timedelta | None = None,
) -> PreferenceHead:
    if dataset is None:
        dataset = LabelDataset(config.labels_path, config.raw_root)
    data_loader = DataLoader(ImageTensorDataset(dataset, loader), batch_size=config.batch_size, shuffle=True)

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Starting training with {len(dataset)} relevant images")
    print(f"[{timestamp}] Loaded {len(dataset)} images from the accepted/rejected roots")

    model = PreferenceHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if resume and checkpoint_path.exists():
            print(f"Resuming from checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif not resume and checkpoint_path.exists():
            print(f"Overwriting existing checkpoint at {checkpoint_path}")

    total_batches = len(data_loader)
    last_status_write = datetime.now()
    last_checkpoint_write = datetime.now()
    model.train()
    for epoch in range(config.epochs):
        current_epoch = epoch + 1
        print(f"Starting epoch {current_epoch}/{config.epochs} (progress: {current_epoch}/{config.epochs})")
        processed_batches = 0
        for batch_idx, (images, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            processed_batches += 1
            now = datetime.now()
            if status_path is not None and status_interval is not None and now - last_status_write >= status_interval:
                write_progress_state(
                    status_path,
                    current_epoch,
                    config.epochs,
                    processed_batches,
                    total_batches,
                    float(loss.item()),
                    f"processed batch {processed_batches}",
                )
                last_status_write = now
            if checkpoint_path is not None and checkpoint_interval is not None and now - last_checkpoint_write >= checkpoint_interval:
                save_checkpoint(model, optimizer, checkpoint_path, epoch=current_epoch)
                print(f"[{now.strftime('%H:%M:%S')}] Saved checkpoint to {checkpoint_path}")
                last_checkpoint_write = now
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   processed batch {processed_batches} (loss={loss.item():.4f})")
        if status_path is not None:
            write_progress_state(
                status_path,
                current_epoch,
                config.epochs,
                processed_batches,
                total_batches,
                None,
                f"completed epoch {current_epoch}/{config.epochs}",
            )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed epoch {current_epoch}/{config.epochs}; {config.epochs - current_epoch} epochs remaining")

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

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
    parser.add_argument("--checkpoint-path", default="model_checkpoint.pt")
    parser.add_argument("--status-path", default="training_status.json")
    parser.add_argument("--status-interval-minutes", type=int, default=10)
    parser.add_argument("--checkpoint-interval-minutes", type=int, default=15)
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint if present")
    parser.add_argument("--fresh-start", action="store_true", help="Start training from scratch instead of resuming")
    parser.add_argument("--split", default=None, help="Frozen split CSV (see picklikeme.split); trains on train rows, evaluates on test rows")
    parser.add_argument("--metrics-json", default="evaluation_metrics.json", help="Where to write test-set metrics when --split is given")
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"], help="letterbox = V2 aspect-preserving; stretch = V1 baseline behavior")
    args = parser.parse_args()

    if not args.select_root or not args.reject_root:
        raise SystemExit("Both --select-root and --reject-root are required.")

    raw_root = args.raw_root or args.select_root
    config = ProjectConfig(raw_root=raw_root, labels_path=args.labels)
    from .raw_io import RawImageLoader

    loader = RawImageLoader(config.raw_root, resize_mode=args.resize_mode)
    burst_labels_path = args.labels if Path(args.labels).exists() else None
    if burst_labels_path is None:
        print(f"Labels CSV not found at {args.labels}; burst IDs will be unavailable (burst metrics need them)")
    dataset = FolderLabelDataset(
        select_root=args.select_root,
        reject_root=args.reject_root,
        raw_root=raw_root,
        burst_labels_path=burst_labels_path,
    )

    test_items = []
    if args.split:
        split_index = PathSuffixIndex.from_csv(args.split, "split")
        all_items = list(dataset.items)
        test_items = [item for item in all_items if split_index.get(item.image_path) == "test"]
        dataset.items = [item for item in all_items if split_index.get(item.image_path) != "test"]
        print(f"Split {args.split}: {len(dataset.items)} train images, {len(test_items)} held-out test images")

    should_resume = args.resume or (not args.fresh_start and Path(args.checkpoint_path).exists())
    model = train(
        config,
        loader,
        dataset=dataset,
        checkpoint_path=args.checkpoint_path,
        resume=should_resume,
        status_path=args.status_path,
        status_interval=timedelta(minutes=args.status_interval_minutes),
        checkpoint_interval=timedelta(minutes=args.checkpoint_interval_minutes),
    )
    if test_items:
        print(f"Evaluating on {len(test_items)} held-out test images")
        metrics = compute_metrics(score_items(model, test_items, loader, device="cpu"))
        print(format_metrics(metrics))
        metrics_path = write_metrics_json(metrics, args.metrics_json)
        print(f"Metrics written to {metrics_path}")

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
