from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import DEFAULT_CHECKPOINT_PATH, ProjectConfig
from .dataset import FolderLabelDataset, LabelDataset, PathSuffixIndex
from .evaluate import compute_metrics, format_metrics, score_items, write_metrics_json
from .model import DINOV3_BACKBONE, ModelConfig, PreferenceHead


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


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested}' but CUDA is not available; falling back to CPU")
        return "cpu"
    return requested


def print_startup_diagnostics(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            index = int(device.split(":", 1)[1]) if ":" in device else 0
        except ValueError:
            index = 0
        print(f"Using device: {torch.cuda.get_device_name(index)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("Using device: CPU")
        print("CUDA: not available")
    print(f"PyTorch: {torch.__version__}")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


_BANNER = "-" * 49
_ALERT_BANNER = "!" * 49

# A save is retried once after this pause, to ride out a transient filesystem
# hiccup (a briefly-locked file, a momentary network-drive drop) without
# aborting a multi-day run.
CHECKPOINT_RETRY_DELAY_SECONDS = 3.0


def _write_checkpoint_atomic(model, optimizer, checkpoint_path: Path, epoch, best_loss) -> None:
    """Perform the actual atomic write: temp file + rename over the target, so
    a crash mid-write can never leave a truncated, unloadable checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if best_loss is not None:
        payload["best_loss"] = best_loss
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(checkpoint_path)


def _print_save_success(checkpoint_path: Path, reason: str, epoch, best_loss, on_retry: bool) -> None:
    print(_BANNER)
    print("Saving checkpoint" + (" (retry SUCCEEDED)" if on_retry else ""))
    print(f"Path: {checkpoint_path.resolve()}")
    print(f"Reason: {reason or 'unspecified'}")
    print(f"Epoch: {epoch}")
    print(f"Best loss: {best_loss}")
    print("Status: SUCCESS")
    print(_BANNER)


def _print_save_failure(checkpoint_path: Path, reason: str, epoch, exc: Exception) -> None:
    print(_BANNER)
    print("Checkpoint save FAILED")
    print(f"Path: {checkpoint_path.resolve()}")
    print(f"Reason: {reason or 'unspecified'}")
    print(f"Epoch: {epoch}")
    print(f"Exception: {type(exc).__name__}: {exc}")
    print("Full traceback:")
    print(traceback.format_exc().rstrip())
    print(_BANNER)


def save_checkpoint(
    model: nn.Module,
    optimizer,
    checkpoint_path: str | Path,
    epoch: int | None = None,
    best_loss: float | None = None,
    reason: str = "",
    retry_delay_seconds: float = CHECKPOINT_RETRY_DELAY_SECONDS,
) -> bool:
    """Save a checkpoint atomically, with a retry-once-then-abort policy.

    `epoch` records the last *fully completed* epoch, not the epoch in
    progress — that's what let resume() know which epoch to restart from
    without silently skipping unfinished work.

    Reliability policy for long-running jobs: if the first save fails, the
    full exception (with traceback), path, and reason are logged, then after a
    short pause the save is retried exactly once. A transient filesystem issue
    should not abort training — but if the retry also fails, a highly visible
    error is printed and a RuntimeError is raised to stop training, because
    continuing a multi-day run with no ability to checkpoint is too risky.

    Returns True on success (first attempt or retry); raises RuntimeError if
    both attempts fail.
    """
    checkpoint_path = Path(checkpoint_path)
    try:
        _write_checkpoint_atomic(model, optimizer, checkpoint_path, epoch, best_loss)
    except Exception as exc:  # noqa: BLE001 - reported, then retried before any abort
        _print_save_failure(checkpoint_path, reason, epoch, exc)
        print(f"Waiting {retry_delay_seconds:.0f}s, then retrying the checkpoint save once...")
        time.sleep(retry_delay_seconds)
        try:
            _write_checkpoint_atomic(model, optimizer, checkpoint_path, epoch, best_loss)
        except Exception as retry_exc:  # noqa: BLE001 - second failure is fatal on purpose
            print(_ALERT_BANNER)
            print("!!! CHECKPOINTING IS NO LONGER RELIABLE - STOPPING TRAINING !!!")
            print("The checkpoint save failed twice (initial attempt + one retry).")
            print(f"Path: {checkpoint_path.resolve()}")
            print(f"Reason: {reason or 'unspecified'}")
            print(f"Retry exception: {type(retry_exc).__name__}: {retry_exc}")
            print("Refusing to continue: a multi-day run that cannot save")
            print("checkpoints could lose all progress on the next failure.")
            print(_ALERT_BANNER)
            raise RuntimeError(
                f"Checkpoint save failed twice for {checkpoint_path.resolve()} "
                f"(reason: {reason or 'unspecified'}); aborting training because "
                "checkpointing is no longer reliable."
            ) from retry_exc
        _print_save_success(checkpoint_path, reason, epoch, best_loss, on_retry=True)
        return True

    _print_save_success(checkpoint_path, reason, epoch, best_loss, on_retry=False)
    return True


def load_checkpoint(checkpoint_path: str | Path, map_location) -> dict:
    """Load a checkpoint and print a transparent diagnostics block."""
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        print(_BANNER)
        print("Loading checkpoint")
        print(f"Path: {checkpoint_path.resolve()}")
        print(f"Epoch: {checkpoint.get('epoch')}")
        print(f"Best loss: {checkpoint.get('best_loss')}")
        print("Status: SUCCESS")
        print(_BANNER)
        return checkpoint
    except Exception as exc:  # noqa: BLE001
        print(_BANNER)
        print("Checkpoint load FAILED")
        print(f"Path: {checkpoint_path.resolve()}")
        print(f"Exception: {type(exc).__name__}: {exc}")
        print(_BANNER)
        raise


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
    model_config: ModelConfig | None = None,
    best_checkpoint_path: str | Path | None = None,
    log_interval_batches: int = 1,
) -> PreferenceHead:
    if dataset is None:
        dataset = LabelDataset(config.manifest_path, config.raw_root)

    device = resolve_device(config.device)
    print_startup_diagnostics(device)

    use_cuda = device.startswith("cuda")
    data_loader = DataLoader(
        ImageTensorDataset(dataset, loader),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        persistent_workers=config.num_workers > 0,
    )

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Starting training with {len(dataset)} relevant images on device={device}")
    print(f"[{timestamp}] Loaded {len(dataset)} images from the accepted/rejected roots")

    model = PreferenceHead(model_config).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {model.config.backbone} (trainable params: {trainable:,} / {total:,})")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_loss = math.inf
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if resume and checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch") or 0
            best_loss = checkpoint.get("best_loss", math.inf)
            print(f"Resumed after epoch {start_epoch}/{config.epochs}; best_loss so far={best_loss}")
        elif not resume and checkpoint_path.exists():
            print(f"Overwriting existing checkpoint at {checkpoint_path}")

    if best_checkpoint_path is not None:
        best_checkpoint_path = Path(best_checkpoint_path)
    elif checkpoint_path is not None:
        best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")

    total_batches = len(data_loader)
    last_status_write = datetime.now()
    last_checkpoint_write = datetime.now()
    last_completed_epoch = start_epoch
    model.train()
    try:
        for epoch in range(start_epoch, config.epochs):
            current_epoch = epoch + 1
            print(f"Starting epoch {current_epoch}/{config.epochs} (progress: {current_epoch}/{config.epochs})")
            processed_batches = 0
            epoch_loss_sum = 0.0
            epoch_start_time = datetime.now()
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(device, non_blocking=use_cuda)
                labels = labels.to(device, non_blocking=use_cuda)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits.squeeze(-1), labels)
                loss.backward()
                optimizer.step()
                # .item() forces a host<->device sync; call it once per batch
                # and reuse the value everywhere instead of syncing twice.
                loss_value = loss.item()
                processed_batches += 1
                epoch_loss_sum += loss_value
                now = datetime.now()
                if status_path is not None and status_interval is not None and now - last_status_write >= status_interval:
                    write_progress_state(
                        status_path,
                        current_epoch,
                        config.epochs,
                        processed_batches,
                        total_batches,
                        loss_value,
                        f"processed batch {processed_batches}",
                    )
                    last_status_write = now
                if checkpoint_path is not None and checkpoint_interval is not None and now - last_checkpoint_write >= checkpoint_interval:
                    # epoch in progress isn't done yet: record last_completed_epoch,
                    # not current_epoch, so a resume redoes this epoch instead of
                    # wrongly treating it as finished.
                    save_checkpoint(
                        model, optimizer, checkpoint_path,
                        epoch=last_completed_epoch, best_loss=best_loss,
                        reason=f"Periodic save (mid-epoch {current_epoch}, last completed epoch {last_completed_epoch})",
                    )
                    last_checkpoint_write = now
                if log_interval_batches > 0 and (processed_batches % log_interval_batches == 0 or processed_batches == total_batches):
                    elapsed = (now - epoch_start_time).total_seconds()
                    rate = processed_batches / elapsed if elapsed > 0 else 0.0
                    remaining = total_batches - processed_batches
                    eta = _format_duration(remaining / rate) if rate > 0 else "n/a"
                    print(
                        f"[{now.strftime('%H:%M:%S')}] epoch {current_epoch}/{config.epochs} "
                        f"batch {processed_batches}/{total_batches} loss={loss_value:.4f} "
                        f"elapsed={_format_duration(elapsed)} eta={eta}"
                    )

            last_completed_epoch = current_epoch
            epoch_avg_loss = epoch_loss_sum / processed_batches if processed_batches else None

            if status_path is not None:
                write_progress_state(
                    status_path,
                    current_epoch,
                    config.epochs,
                    processed_batches,
                    total_batches,
                    epoch_avg_loss,
                    f"completed epoch {current_epoch}/{config.epochs}",
                )

            if epoch_avg_loss is not None and epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                if best_checkpoint_path is not None:
                    save_checkpoint(
                        model, optimizer, best_checkpoint_path,
                        epoch=last_completed_epoch, best_loss=best_loss,
                        reason=f"New best loss {best_loss:.4f} (epoch {current_epoch})",
                    )

            if checkpoint_path is not None:
                # Save after the best_loss update above so a later resume reads
                # back the correct running-best, not a stale pre-epoch value.
                save_checkpoint(
                    model, optimizer, checkpoint_path,
                    epoch=last_completed_epoch, best_loss=best_loss,
                    reason=f"End of epoch {current_epoch}",
                )

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed epoch {current_epoch}/{config.epochs}; {config.epochs - current_epoch} epochs remaining")
    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] KeyboardInterrupt received; saving checkpoint before exiting")
        if checkpoint_path is not None:
            save_checkpoint(
                model, optimizer, checkpoint_path,
                epoch=last_completed_epoch, best_loss=best_loss,
                reason=f"Ctrl+C / KeyboardInterrupt (last completed epoch {last_completed_epoch})",
            )
        raise

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
    parser.add_argument("--manifest", default="data/manifest.parquet", help="Manifest parquet (from picklikeme.ingest.cli build-manifest); supplies burst IDs")
    parser.add_argument("--select-root", default=None)
    parser.add_argument("--reject-root", default=None)
    parser.add_argument("--output-csv", default="training_results.csv")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Where to save the rolling checkpoint (default: <project-root>/checkpoints/model_checkpoint.pt, independent of the current working directory)")
    parser.add_argument("--best-checkpoint-path", default=None, help="Where to save the lowest-training-loss checkpoint (default: <checkpoint-path>_best.pt)")
    parser.add_argument("--status-path", default="training_status.json")
    parser.add_argument("--status-interval-minutes", type=int, default=10)
    parser.add_argument("--checkpoint-interval-minutes", type=int, default=15)
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint if present")
    parser.add_argument("--fresh-start", action="store_true", help="Start training from scratch instead of resuming")
    parser.add_argument("--split", default=None, help="Frozen split CSV (see picklikeme.split); trains on train rows, evaluates on test rows")
    parser.add_argument("--metrics-json", default="evaluation_metrics.json", help="Where to write test-set metrics when --split is given")
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"], help="letterbox = V2 aspect-preserving; stretch = V1 baseline behavior")
    parser.add_argument("--backbone", default=DINOV3_BACKBONE, help="V3 pretrained backbone (any timm model name), or 'cnn' to reproduce the V1/V2 baseline backbone")
    parser.add_argument("--no-pretrained", action="store_true", help="Randomly initialize the backbone instead of loading pretrained weights (debugging only)")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Fine-tune the pretrained backbone instead of the default linear-probe (frozen backbone)")
    parser.add_argument("--device", default=None, help="Override the device (default: cuda if available, else cpu)")
    parser.add_argument("--num-workers", type=int, default=None, help=f"DataLoader worker processes (default: {ProjectConfig.num_workers})")
    parser.add_argument("--log-interval-batches", type=int, default=1, help="Print training progress every N batches (1 = every batch, the default; higher = less noise; 0 = only at epoch end)")
    args = parser.parse_args()

    if not args.select_root or not args.reject_root:
        raise SystemExit("Both --select-root and --reject-root are required.")

    raw_root = args.raw_root or args.select_root
    config = ProjectConfig(
        raw_root=raw_root,
        manifest_path=args.manifest,
        device=args.device or ProjectConfig.device,
        num_workers=args.num_workers if args.num_workers is not None else ProjectConfig.num_workers,
    )
    from .raw_io import RawImageLoader

    loader = RawImageLoader(config.raw_root, resize_mode=args.resize_mode)
    manifest_path = args.manifest if Path(args.manifest).exists() else None
    if manifest_path is None:
        print(f"Manifest not found at {args.manifest}; burst IDs will be unavailable (burst metrics need them)")
    dataset = FolderLabelDataset(
        select_root=args.select_root,
        reject_root=args.reject_root,
        raw_root=raw_root,
        manifest_path=manifest_path,
    )

    test_items = []
    if args.split:
        split_index = PathSuffixIndex.from_table(args.split, "split")
        all_items = list(dataset.items)
        test_items = [item for item in all_items if split_index.get(item.image_path) == "test"]
        dataset.items = [item for item in all_items if split_index.get(item.image_path) != "test"]
        print(f"Split {args.split}: {len(dataset.items)} train images, {len(test_items)} held-out test images")

    model_config = ModelConfig(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
    )
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
        model_config=model_config,
        best_checkpoint_path=args.best_checkpoint_path,
        log_interval_batches=args.log_interval_batches,
    )
    device = resolve_device(config.device)
    if test_items:
        print(f"Evaluating on {len(test_items)} held-out test images")
        metrics = compute_metrics(score_items(model, test_items, loader, device=device))
        print(format_metrics(metrics))
        metrics_path = write_metrics_json(metrics, args.metrics_json)
        print(f"Metrics written to {metrics_path}")

    ranked = rank_dataset(model, dataset, loader, device=device)
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
