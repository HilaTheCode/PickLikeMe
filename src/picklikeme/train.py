from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .bird_crop import read_crop_params
from .config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CROP_CACHE_DIR,
    DEFAULT_MAX_CSV_ROWS,
    ProjectConfig,
    fatal_errors_logged_to_stdout,
    format_duration,
    run_dir_for_timestamp,
    tee_stdout_to_file,
)
from .dataset import FolderLabelDataset, LabelDataset, PathSuffixIndex
from .evaluate import compute_metrics, format_metrics, score_items, write_metrics_json
from .model import DINOV3_BACKBONE, ModelConfig, PreferenceHead
from .platform import resolve_torch_device


class ExistingCheckpointError(RuntimeError):
    """A fresh start was requested but a checkpoint already exists at the
    target path - see check_fresh_start_is_safe."""


def check_fresh_start_is_safe(checkpoint_path: str | Path | None, resume: bool) -> None:
    """Refuse to proceed if starting fresh (`resume=False`) would silently
    overwrite a checkpoint that already exists there.

    Called as the very first thing `train()` does - before the dataset,
    backbone or optimizer are even constructed - so a mistaken `--fresh-start`
    against a real checkpoint fails in milliseconds, not after minutes of GPU
    work, and so every caller of `train()` gets this protection for free
    regardless of how they got there (the CLI's `--fresh-start`, a script, a
    notebook). A checkpoint often represents days, weeks or months of
    training; there is no code path in this module that overwrites one
    without the caller passing `resume=True` first.
    """
    if checkpoint_path is None or resume:
        return
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise ExistingCheckpointError(
            f"Refusing to start fresh: a checkpoint already exists at {checkpoint_path.resolve()}.\n"
            "This likely represents real, possibly irreplaceable training progress. Move, rename, "
            "or delete it first if you really do want to start over, then run again - or pass "
            "--resume (or drop --fresh-start) to continue training from it instead."
        )


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
    return resolve_torch_device(requested)


def _gigabytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.1f} GB"


def describe_device(device: str) -> str:
    """One-line device description for the run header: GPU name plus free and
    total memory, or "CPU". Diagnostics only — never affects training."""
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return "CPU"
    try:
        index = int(device.split(":", 1)[1]) if ":" in device else 0
    except ValueError:
        index = 0
    name = torch.cuda.get_device_name(index)
    try:
        free, total = torch.cuda.mem_get_info(index)
    except Exception:  # noqa: BLE001 - a memory query failure must not stop a run
        return name
    return f"{name} ({_gigabytes(free)} free / {_gigabytes(total)} total)"


@dataclass(frozen=True)
class RunCounts:
    """Dataset sizes reported in the run header and epoch summaries.

    Logging only. Passed in from train_and_rank because the *totals* it reports
    (selected/rejected across the whole labeled set, and the held-out count)
    are known before --split removes the test rows from the training dataset,
    so train() cannot recover them on its own.
    """

    selected: int = 0
    rejected: int = 0
    validation: int = 0


def count_label_split(dataset) -> tuple[int, int]:
    """(selected, rejected) counts for a labeled dataset, for logging only."""
    items = getattr(dataset, "items", None)
    if items is not None:
        labels = [int(item.label) for item in items]
    else:
        frame = getattr(dataset, "frame", None)
        if frame is not None:
            labels = [int(value) for value in frame["label"]]
        else:
            labels = [int(dataset[index].label) for index in range(len(dataset))]
    selected = sum(1 for label in labels if label == 1)
    return selected, len(labels) - selected


def print_run_header(
    *,
    device: str,
    resumed: bool,
    checkpoint_path: Path | None,
    best_checkpoint_path: Path | None,
    checkpoint_epoch: int,
    best_loss: float,
    start_epoch: int,
    target_epoch: int,
    epochs_to_run: int,
    counts: RunCounts,
    train_images: int,
    total_batches: int,
    batch_size: int,
    backbone: str,
    trainable_params: int,
    total_params: int,
    learning_rate: float,
) -> None:
    """Print the once-per-run summary: where the run resumes from, what it will
    train on, and what hardware it is on. One screen, no repetition — everything
    a long run needs recorded at its start."""
    labeled_total = counts.selected + counts.rejected
    if resumed:
        best = "unknown" if best_loss == math.inf else f"{best_loss:.4f}"
        mode = f"resuming from checkpoint at epoch {checkpoint_epoch} (best loss so far {best})"
    else:
        mode = "starting from scratch (no checkpoint loaded)"

    print(_BANNER)
    print("Training run")
    print(f"  mode:            {mode}")
    print(f"  checkpoint:      {checkpoint_path.resolve() if checkpoint_path else 'not saving'}")
    if best_checkpoint_path is not None:
        print(f"  best checkpoint: {best_checkpoint_path.resolve()}")
    print(f"  epochs:          {start_epoch + 1}-{target_epoch} ({epochs_to_run} this run)")
    print(f"  labeled images:  {labeled_total:,} = {counts.selected:,} selected / {counts.rejected:,} rejected")
    print(f"  training set:    {train_images:,} images, {total_batches:,} batches of {batch_size}")
    if counts.validation:
        # No per-epoch validation pass exists, so no per-epoch val loss is
        # reported; the held-out set is scored once after the final epoch.
        print(f"  validation set:  {counts.validation:,} held-out images (scored after the final epoch)")
    else:
        print("  validation set:  none (pass --split to hold out a test set)")
    print(f"  backbone:        {backbone} ({trainable_params:,} trainable / {total_params:,} params)")
    print(f"  learning rate:   {learning_rate:.2e}")
    print(f"  device:          {describe_device(device)}")
    cuda_version = torch.version.cuda if device.startswith("cuda") and torch.cuda.is_available() else "not in use"
    print(f"  torch:           {torch.__version__} (CUDA {cuda_version})")
    print(_BANNER)


_BANNER = "-" * 49
_ALERT_BANNER = "!" * 49

# Batch-level progress cadence. Frequent enough to show a stalled run within
# seconds, sparse enough that a 54k-image epoch (~3,400 batches) produces ~70
# lines instead of ~3,400 — a multi-day log stays readable. Epoch summaries and
# the final batch of each epoch print regardless.
DEFAULT_LOG_INTERVAL_BATCHES = 50

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
    """One line per successful save. A multi-day run saves every few minutes, so
    the multi-line block this replaced dominated the log. Failures keep their
    full diagnostic block (see _print_save_failure) — that output is rare and
    is exactly what's needed to diagnose a broken save."""
    epoch_text = "n/a" if epoch is None else str(epoch)
    best_text = "n/a" if best_loss is None or best_loss == math.inf else f"{best_loss:.4f}"
    retry = " (retry SUCCEEDED)" if on_retry else ""
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Checkpoint saved{retry}"
        f" | epoch {epoch_text} | best loss {best_text}"
        f" | {reason or 'unspecified'} | {checkpoint_path.resolve()}"
    )


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
    log_interval_batches: int = DEFAULT_LOG_INTERVAL_BATCHES,
    epochs_this_run: int | None = None,
    counts: RunCounts | None = None,
) -> PreferenceHead:
    check_fresh_start_is_safe(checkpoint_path, resume)

    if dataset is None:
        dataset = LabelDataset(config.manifest_path, config.raw_root)

    device = resolve_device(config.device)

    # Number of epochs to run in THIS invocation (per-run, not a cumulative
    # target). Falls back to config.epochs only for internal callers that don't
    # pass it; the CLI always supplies --epochs.
    epochs_to_run = epochs_this_run if epochs_this_run is not None else config.epochs

    use_cuda = device.startswith("cuda")
    data_loader = DataLoader(
        ImageTensorDataset(dataset, loader),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        persistent_workers=config.num_workers > 0,
    )

    if counts is None:
        # Direct/internal callers don't supply counts; derive what we can so the
        # header is still accurate for them (no validation set in that case).
        selected, rejected = count_label_split(dataset)
        counts = RunCounts(selected=selected, rejected=rejected)

    backbone_name = (model_config or ModelConfig()).backbone
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading backbone {backbone_name} on {device} ...")
    model = PreferenceHead(model_config).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_loss = math.inf
    resumed = False
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if resume and checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch") or 0
            best_loss = checkpoint.get("best_loss", math.inf)
            resumed = True
        # A fresh start (resume=False) with an existing checkpoint already
        # raised in check_fresh_start_is_safe above - there is no third case
        # to handle here.

    # Per-run target: fresh start (start_epoch == 0) runs epochs 1..N; resuming
    # from a checkpoint at epoch K runs K+1..K+N, keeping the epoch numbering
    # continuous instead of restarting it.
    target_epoch = start_epoch + epochs_to_run

    if best_checkpoint_path is not None:
        best_checkpoint_path = Path(best_checkpoint_path)
    elif checkpoint_path is not None:
        best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")

    total_batches = len(data_loader)
    print_run_header(
        device=device,
        resumed=resumed,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        checkpoint_epoch=start_epoch,
        best_loss=best_loss,
        start_epoch=start_epoch,
        target_epoch=target_epoch,
        epochs_to_run=epochs_to_run,
        counts=counts,
        train_images=len(dataset),
        total_batches=total_batches,
        batch_size=config.batch_size,
        backbone=model.config.backbone,
        trainable_params=trainable,
        total_params=total,
        learning_rate=config.learning_rate,
    )

    last_status_write = datetime.now()
    last_checkpoint_write = datetime.now()
    last_completed_epoch = start_epoch
    run_start_time = datetime.now()
    epochs_completed_this_run = 0
    model.train()
    try:
        for epoch in range(start_epoch, target_epoch):
            current_epoch = epoch + 1
            print(f"Starting epoch {current_epoch}/{target_epoch}")
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
                        target_epoch,
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
                    eta = format_duration(remaining / rate) if rate > 0 else "n/a"
                    print(
                        f"[{now.strftime('%H:%M:%S')}] epoch {current_epoch}/{target_epoch} "
                        f"batch {processed_batches}/{total_batches} loss={loss_value:.4f} "
                        f"elapsed={format_duration(elapsed)} eta={eta}"
                    )

            last_completed_epoch = current_epoch
            epoch_avg_loss = epoch_loss_sum / processed_batches if processed_batches else None

            if status_path is not None:
                write_progress_state(
                    status_path,
                    current_epoch,
                    target_epoch,
                    processed_batches,
                    total_batches,
                    epoch_avg_loss,
                    f"completed epoch {current_epoch}/{target_epoch}",
                )

            saved_best = False
            if epoch_avg_loss is not None and epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                if best_checkpoint_path is not None:
                    save_checkpoint(
                        model, optimizer, best_checkpoint_path,
                        epoch=last_completed_epoch, best_loss=best_loss,
                        reason=f"New best loss {best_loss:.4f} (epoch {current_epoch})",
                    )
                    saved_best = True

            if checkpoint_path is not None:
                # Save after the best_loss update above so a later resume reads
                # back the correct running-best, not a stale pre-epoch value.
                save_checkpoint(
                    model, optimizer, checkpoint_path,
                    epoch=last_completed_epoch, best_loss=best_loss,
                    reason=f"End of epoch {current_epoch}",
                )

            # Epoch summary: two lines carrying everything a long run needs per
            # epoch. The run ETA extrapolates from the epochs completed in THIS
            # invocation, so it stays right when resuming a slower/faster run.
            epochs_completed_this_run += 1
            finished_at = datetime.now()
            epoch_seconds = (finished_at - epoch_start_time).total_seconds()
            epochs_left = target_epoch - current_epoch
            mean_epoch_seconds = (finished_at - run_start_time).total_seconds() / epochs_completed_this_run
            run_eta = format_duration(mean_epoch_seconds * epochs_left) if epochs_left else "done"
            loss_text = f"{epoch_avg_loss:.4f}" if epoch_avg_loss is not None else "n/a"
            best_text = "n/a" if best_loss == math.inf else f"{best_loss:.4f}"
            print(
                f"[{finished_at.strftime('%H:%M:%S')}] Completed epoch {current_epoch}/{target_epoch}"
                f" | train_loss {loss_text} | best {best_text}"
                f" | lr {optimizer.param_groups[0]['lr']:.2e}"
                f" | epoch {format_duration(epoch_seconds)} | run eta {run_eta}"
            )
            saved_to = f"{checkpoint_path}{' (+best)' if saved_best else ''}" if checkpoint_path is not None else "not saved"
            print(
                f"{' ' * 11}images {len(dataset):,} train / {counts.validation:,} val"
                f" | checkpoint {saved_to}"
            )
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


def rank_dataset(
    model: nn.Module,
    dataset,
    loader,
    device: str = "cpu",
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[str, float, int, str]]:
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
            if processed_images % 100 == 0 or processed_images == len(dataset):
                print(f"  ranked image {processed_images}/{len(dataset)}: {Path(item.image_path).name}")
            if on_progress is not None:
                on_progress(processed_images, len(dataset))

    scored.sort(key=lambda entry: entry[1], reverse=True)
    return scored


def timestamped_output_path(output_path: str | Path, run_started: datetime) -> Path:
    """Insert a run timestamp into the filename stem, so consecutive runs write
    distinct result files instead of silently overwriting the previous ones.

    Used by `picklikeme.rank` (ranking a new, unlabeled folder - not part of a
    training run, so it has no analysis_results/<timestamp>/ run directory of
    its own to land in). `train_and_rank` no longer uses this: its outputs
    get their uniqueness from run_dir_for_timestamp() instead.

    The stamp is the run's *start* time, not the write time. Applied to the
    stem so the chunked `_1`/`_2` suffixes from write_results_csv still sort
    next to the first file: `rankings_20260725-143000_1.csv`. The
    `%Y%m%d-%H%M%S` format sorts chronologically under a plain lexicographic
    `ls`.
    """
    output_path = Path(output_path)
    stamp = run_started.strftime("%Y%m%d-%H%M%S")
    return output_path.with_name(f"{output_path.stem}_{stamp}{output_path.suffix}")


# Lines each results CSV spends on its metrics preamble before the data rows:
# the "metric,value" header, four key/value rows, a blank separator, and the
# column header. Subtracted from max_rows so the whole file honours the limit.
CSV_PREAMBLE_LINES = 7


def write_results_csv(
    output_path: str | Path,
    dataset,
    ranked: list[tuple[str, float, int, str]],
    select_root: str,
    reject_root: str,
    max_rows: int = DEFAULT_MAX_CSV_ROWS,
) -> list[Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    detected_sequences = dataset.count_sequences()

    files_written: list[Path] = []
    # The preamble occupies part of the budget, so a file never exceeds
    # max_rows lines in total. A max_rows smaller than the preamble itself
    # would leave no room for data, so fall back to using it as a data budget.
    rows_per_file = max_rows - CSV_PREAMBLE_LINES
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


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a personal preference model", add_help=add_help)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--manifest", default="data/manifest.parquet", help="Manifest parquet (from picklikeme.ingest.cli build-manifest); supplies burst IDs")
    parser.add_argument("--select-root", default=None)
    parser.add_argument("--reject-root", default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs to run in THIS invocation (NOT a cumulative total). "
        "Fresh start trains epochs 1..N; --resume trains N more, continuing the "
        "epoch numbering from the checkpoint (e.g. resume at epoch 20 with "
        "--epochs 10 trains epochs 21-30).",
    )
    parser.add_argument(
        "--output-csv",
        default="training_results.csv",
        help="Base name for the ranking results CSV, written under this run's own "
        "analysis_results/<timestamp>/ranking/ directory so no previous run's "
        "results are ever overwritten.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_CSV_ROWS,
        help=f"Maximum lines per results CSV before it rolls over to name_1.csv, name_2.csv "
        f"(default: {DEFAULT_MAX_CSV_ROWS:,}; the {CSV_PREAMBLE_LINES}-line metrics preamble counts "
        "toward it). Change the default in picklikeme/config.py.",
    )
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Where to save the rolling checkpoint (default: <project-root>/checkpoints/model_checkpoint.pt, independent of the current working directory)")
    parser.add_argument("--best-checkpoint-path", default=None, help="Where to save the lowest-training-loss checkpoint (default: <checkpoint-path>_best.pt)")
    parser.add_argument("--status-path", default="training_status.json")
    parser.add_argument("--status-interval-minutes", type=int, default=10)
    parser.add_argument("--checkpoint-interval-minutes", type=int, default=15)
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint if present")
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Start training from scratch instead of resuming. Refuses (does not start) if a "
        "checkpoint already exists at --checkpoint-path - move, rename, or delete it first.",
    )
    parser.add_argument("--split", default=None, help="Frozen split CSV (see picklikeme.split); trains on train rows, evaluates on test rows")
    parser.add_argument(
        "--metrics-json",
        default="evaluation_metrics.json",
        help="Base name for the test-set metrics written when --split is given, under "
        "this run's own analysis_results/<timestamp>/metrics/ directory.",
    )
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"], help="letterbox = V2 aspect-preserving; stretch = V1 baseline behavior")
    parser.add_argument("--backbone", default=DINOV3_BACKBONE, help="V3 pretrained backbone (any timm model name), or 'cnn' to reproduce the V1/V2 baseline backbone")
    parser.add_argument("--no-pretrained", action="store_true", help="Randomly initialize the backbone instead of loading pretrained weights (debugging only)")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Fine-tune the pretrained backbone instead of the default linear-probe (frozen backbone)")
    parser.add_argument("--device", default=None, help="Override the device (default: cuda if available, else cpu)")
    parser.add_argument("--num-workers", type=int, default=None, help=f"DataLoader worker processes (default: {ProjectConfig.num_workers})")
    parser.add_argument(
        "--log-interval-batches",
        type=int,
        default=DEFAULT_LOG_INTERVAL_BATCHES,
        help=f"Print a progress line every N batches (default: {DEFAULT_LOG_INTERVAL_BATCHES}; "
        "1 = every batch, which floods the log on a large dataset; 0 = epoch summaries only). "
        "The last batch of every epoch is always logged.",
    )
    parser.add_argument("--crop-birds", action=argparse.BooleanOptionalAction, default=True, help="Feed pre-computed bird crops from picklikeme.preprocess (default); pass --no-crop-birds to train on full frames")
    parser.add_argument("--crop-cache-dir", default=str(DEFAULT_CROP_CACHE_DIR), help="Directory of the bird-crop cache used by --crop-birds")
    return parser


def train_and_rank(args) -> None:
    """Train (or resume) on the select/reject folders, then rank every image and
    write the results CSV. Shared by `picklikeme.train` and `picklikeme.run`."""
    if not args.select_root or not args.reject_root:
        raise SystemExit("Both --select-root and --reject-root are required.")

    # Stamped once, up front, and shared by every per-run output: one run
    # directory holds the ranking CSV, metrics, and log, so `picklikeme analyze`
    # can later find them all together by timestamp.
    run_started = datetime.now()
    run_dir = run_dir_for_timestamp(run_started)
    output_csv = run_dir / "ranking" / Path(args.output_csv).name
    metrics_json = run_dir / "metrics" / Path(args.metrics_json).name
    log_path = run_dir / "logs" / "training.log"
    print(f"Run directory: {run_dir.resolve()}")

    with tee_stdout_to_file(log_path):
        with fatal_errors_logged_to_stdout():
            _train_and_rank(args, output_csv, metrics_json)


def _train_and_rank(args, output_csv: Path, metrics_json: Path) -> None:
    raw_root = args.raw_root or args.select_root
    config = ProjectConfig(
        raw_root=raw_root,
        manifest_path=args.manifest,
        device=args.device or ProjectConfig.device,
        num_workers=args.num_workers if args.num_workers is not None else ProjectConfig.num_workers,
    )
    from .raw_io import RawImageLoader

    crop_cache_dir = args.crop_cache_dir if args.crop_birds else None
    if args.crop_birds:
        cache_path = Path(crop_cache_dir)
        # The cache is sharded (cache_dir/<2 hex>/<digest>.png), so a glob for
        # "*.png" at the root would find nothing even for a full cache. Presence
        # is established from crop_params.json, which preprocessing writes for
        # the directory - no scan of a 55k-entry tree.
        if read_crop_params(cache_path) is not None:
            print(f"Bird-crop input enabled (default); reading crops from {cache_path.resolve()}")
        else:
            print("=" * 64)
            print("WARNING: --crop-birds is on (default) but no crop cache was found at")
            print(f"  {cache_path.resolve()}")
            print("Training will FALL BACK TO FULL FRAMES for every image.")
            print("Run `python -m picklikeme.preprocess --select-root ... --reject-root ...`")
            print("first, or pass --no-crop-birds to train on full frames on purpose.")
            print("=" * 64)
    else:
        print("Bird-crop input disabled (--no-crop-birds); using full frames.")
    loader = RawImageLoader(config.raw_root, resize_mode=args.resize_mode, crop_cache_dir=crop_cache_dir)
    manifest_path = args.manifest if Path(args.manifest).exists() else None
    if manifest_path is None:
        print(f"Manifest not found at {args.manifest}; burst IDs will be unavailable (burst metrics need them)")
    dataset = FolderLabelDataset(
        select_root=args.select_root,
        reject_root=args.reject_root,
        raw_root=raw_root,
        manifest_path=manifest_path,
    )

    # Counted before --split removes the held-out rows, so the header reports
    # the totals of the whole labeled set, not just the training portion.
    selected_total, rejected_total = count_label_split(dataset)

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
        epochs_this_run=args.epochs,
        counts=RunCounts(
            selected=selected_total,
            rejected=rejected_total,
            validation=len(test_items),
        ),
    )
    device = resolve_device(config.device)
    if test_items:
        print(f"Evaluating on {len(test_items)} held-out test images")
        metrics = compute_metrics(score_items(model, test_items, loader, device=device))
        print(format_metrics(metrics))
        metrics_path = write_metrics_json(metrics, metrics_json)
        print(f"Metrics written to {metrics_path}")

    ranked = rank_dataset(model, dataset, loader, device=device)
    print(f"Detected sequences: {dataset.count_sequences()}")
    output_paths = write_results_csv(output_csv, dataset, ranked, args.select_root, args.reject_root, max_rows=args.max_rows)
    print("Top-ranked images:")
    for rank, entry in enumerate(ranked[:10], start=1):
        image_name = entry[0]
        score = entry[1]
        print(f"{rank}. {image_name}: {score:.4f}")
    print(f"CSV written to {output_paths[0]}")
    if len(output_paths) > 1:
        print(f"Additional CSV files: {', '.join(str(path) for path in output_paths[1:])}")


def main() -> None:
    args = build_arg_parser().parse_args()
    train_and_rank(args)


if __name__ == "__main__":
    main()
