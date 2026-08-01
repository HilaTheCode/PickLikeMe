"""One-time animal-crop cache builder.

Decodes each RAW once, detects the animal, crops tightly (small margin, aspect
preserved), and writes the crop to the cache that RawImageLoader reads during
training. Run this before training with --crop-birds.

    python -m picklikeme.preprocess --select-root "..." --reject-root "..."

Idempotent: images already cached are skipped, so it can be re-run to finish an
interrupted pass or to pick up newly added images. Detection runs in this
single process (default device cuda) — never inside DataLoader workers.

Pipeline shape (measured: RAW demosaic is ~80% of wall clock, the GPU is busy
~5% of it, so the serial loop left ~19 cores and the GPU idle):

    decode pool (N threads)  ->  bounded window  ->  main thread (GPU)  ->  writer thread
    read + rawpy.postprocess     DECODE_WINDOW      detect + crop          save_crop_png

Threads, not processes: rawpy releases the GIL during postprocess (measured
x2.75 on 8 threads, matching a process pool), so decoded 58 MB frames stay in
one address space with no IPC.

What the pipeline deliberately does NOT change: images reach the detector one
at a time, in the original order, and every per-image computation (decode
parameters, detection, crop math, PNG encoding) is byte-for-byte the code the
serial version ran. Only the *overlap* is new.
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .bird_crop import (
    IMAGE_FORMAT_EXTENSIONS,
    SUPPORTED_ANIMAL_CLASSES,
    BirdDetector,
    CropParams,
    build_crop,
    coco_class_name,
    crop_cache_path,
    detections_cache_path,
    read_crop_params,
    save_crop_png,
    save_detections,
    write_crop_params,
)
from .config import DEFAULT_CROP_CACHE_DIR, fatal_errors_logged_to_stdout, format_duration
from .dataset import FolderLabelDataset
from .profiling import PROFILE
from .raw_io import RawImageLoader

# Preprocessing a 54k-image set takes hours, so progress is reported on a timer
# rather than every N images: a slow pass logs steadily, and a fast pass over an
# already-built cache does not flood the log with thousands of lines.
PROGRESS_INTERVAL_SECONDS = 30.0

# With PICKLIKEME_PROFILE=1, a stage-timing line is added this often (in images).
PROFILE_SUMMARY_INTERVAL_IMAGES = 500

# Decoded full-resolution frames in flight. Each is ~58 MB for a 20 MP RAW, so
# 12 caps decode-side RAM at roughly 700 MB. This is the bounded queue between
# the decode pool and the GPU: the window cannot advance until the GPU consumes,
# so a fast disk can never run ahead and exhaust memory.
DECODE_WINDOW = 12

# Finished crops awaiting PNG encode + write. Crops are capped at max_side so
# each is only a few MB; a short queue is enough to absorb write latency.
WRITE_QUEUE_SIZE = 16

# How long to wait on a full write queue before checking that the writer thread
# is still alive. Bounds the blocking put so a dead writer surfaces as an error
# instead of hanging the run forever.
WRITER_LIVENESS_TIMEOUT_SECONDS = 5.0

# Upper bound on waiting for the writer to drain at shutdown. Long enough that a
# healthy writer (a queue of small PNGs) always finishes, short enough that a
# wedged one cannot hang the process indefinitely.
WRITER_JOIN_TIMEOUT_SECONDS = 300.0


def default_decode_workers() -> int:
    """Decoder threads to use when the caller doesn't specify.

    Capped at 8 because measured decode throughput saturates there: libraw is
    itself multi-threaded (~5.8 cores for a single frame), so more decoder
    threads stop helping and start contending - and past ~4.5 img/s the source
    disk is the limit anyway.
    """
    return min(8, os.cpu_count() or 1)


@dataclass
class _Job:
    """One image moving through the pipeline, in submission order.

    `future` is None for an image already present in the cache: nothing is
    decoded for it, but it still occupies its place in the ordered stream so
    counting and progress stay exactly as they were in the serial version.
    """

    image_path: str
    target: Path
    future: "Future | None"


@dataclass
class _Decoded:
    """Result of one decode attempt. Errors are *returned*, not raised, so a
    single unreadable file cannot tear down the pool - matching the serial
    version's per-image try/except."""

    rgb: np.ndarray | None
    error: BaseException | None


@dataclass
class _WriteJob:
    """A crop handed to the writer thread. Carries what the main thread already
    counted for this image, so a failed write can undo it precisely and the
    final totals match the serial version exactly."""

    target: Path
    crop: np.ndarray
    image_path: str
    class_name: str | None  # None => full-frame fallback (no detection)


def _decode_one(decoder: RawImageLoader, image_path: str) -> _Decoded:
    """Decode a single image in a worker thread.

    RawImageLoader is safe to share: _decode_full_frame keeps no per-call state
    (rawpy and cv2 are given a fresh handle each call), and the crop-cache
    fallback path that owns the one mutable flag is not used here.
    """
    try:
        return _Decoded(rgb=decoder._decode_full_frame(image_path), error=None)
    except BaseException as exc:  # noqa: BLE001 - reported per image, pass continues
        return _Decoded(rgb=None, error=exc)


def _writer_loop(
    write_queue: "queue.Queue", failures: list[tuple[_WriteJob, BaseException]], jpeg_quality: int
) -> None:
    """Drain crops to disk until the sentinel arrives.

    Never exits on error: a failed write is recorded and the loop continues, so
    the main thread can never block forever on a queue whose consumer died.
    `jpeg_quality` is fixed for the whole run (one CropParams, see build_cache)
    and only actually used when job.target's own extension is JPEG - see
    save_crop_png.
    """
    while True:
        job = write_queue.get()
        if job is None:
            return
        try:
            save_crop_png(job.target, job.crop, jpeg_quality=jpeg_quality)
        except BaseException as exc:  # noqa: BLE001 - recorded, reconciled by the main thread
            failures.append((job, exc))


from .platform import resolve_torch_device


def _resolve_device(requested: str) -> str:
    return resolve_torch_device(requested)


def build_cache(
    image_paths: list[str],
    cache_dir: str | Path,
    params: CropParams,
    device: str = "cuda",
    force: bool = False,
    decode_workers: int | None = None,
) -> dict:
    cache_dir = Path(cache_dir)
    device = _resolve_device(device)
    decode_workers = max(1, decode_workers if decode_workers is not None else default_decode_workers())

    existing = read_crop_params(cache_dir)
    if existing is not None and existing != params and not force:
        raise SystemExit(
            f"Existing cache at {cache_dir} was built with different parameters:\n"
            f"  existing: {existing}\n  requested: {params}\n"
            "Pass --force to rebuild, or delete the cache directory."
        )
    write_crop_params(cache_dir, params)

    # RawImageLoader here decodes the full frame (no crop cache) so we can detect.
    decoder = RawImageLoader(raw_root=".", resize_mode="letterbox")
    detector = BirdDetector(
        device=device,
        conf_threshold=params.conf_threshold,
        area_tie_frac=params.area_tie_frac,
        group_scene_threshold=params.group_scene_threshold,
    )

    total = len(image_paths)
    stats = {"total": total, "cached": 0, "skipped": 0, "birds": 0, "fallbacks": 0, "errors": 0}
    class_counts: Counter[str] = Counter()

    print(f"Building crop cache for {total:,} images on device={device}")
    print(f"  {'cache dir:':<20}{Path(cache_dir).resolve()}")
    print(f"  {'accepted classes:':<20}{', '.join(sorted(SUPPORTED_ANIMAL_CLASSES.values()))}")
    print(f"  {'min confidence:':<20}{params.conf_threshold}")
    print(f"  {'selection:':<20}largest area wins; confidence ties within {params.area_tie_frac:.0%} of it")
    print(
        f"  {'group scenes:':<20}{params.group_scene_threshold}+ detections -> crop the whole group, "
        "not one individual"
    )
    print(f"  {'decode workers:':<20}{decode_workers} (window {DECODE_WINDOW}, GPU stays sequential)")
    max_side_desc = "unlimited (original crop resolution)" if params.max_side is None else f"{params.max_side}px"
    format_desc = params.image_format + (f" q{params.jpeg_quality}" if params.image_format == "jpeg" else "")
    print(f"  {'crop resolution:':<20}{max_side_desc}")
    print(f"  {'crop format:':<20}{format_desc}")

    if PROFILE.enabled:
        print(f"  profiling:        ON (stage summary every {PROFILE_SUMMARY_INTERVAL_IMAGES} images)")
        PROFILE.reset()

    source = iter(image_paths)
    pending: deque[_Job] = deque()
    write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
    write_failures: list[tuple[_WriteJob, BaseException]] = []
    writer = threading.Thread(
        target=_writer_loop,
        args=(write_queue, write_failures, params.jpeg_quality),
        name="picklikeme-writer",
        daemon=True,
    )

    def _reconcile_write_failures() -> None:
        """Fold completed-but-failed writes back into the stats.

        The main thread counts an image as cached when it hands the crop over,
        so a write that later fails must undo exactly that bookkeeping - leaving
        the same totals the serial version produced, where the counters were only
        touched after a successful write.
        """
        while write_failures:
            job, exc = write_failures.pop(0)
            stats["errors"] += 1
            stats["cached"] -= 1
            if job.class_name is None:
                stats["fallbacks"] -= 1
            else:
                stats["birds"] -= 1
                class_counts[job.class_name] -= 1
                if class_counts[job.class_name] <= 0:
                    del class_counts[job.class_name]
            print(f"  ERROR writing crop for {job.image_path}: {type(exc).__name__}: {exc}")

    def _enqueue_write(job: _WriteJob) -> None:
        """Hand a crop to the writer, without risking an unbounded block.

        A plain put() on a full queue would hang forever if the writer thread
        were gone; bounded waits re-check liveness so that turns into a raised
        error instead of a stalled multi-hour run.
        """
        while True:
            try:
                write_queue.put(job, timeout=WRITER_LIVENESS_TIMEOUT_SECONDS)
                return
            except queue.Full:
                if not writer.is_alive():
                    raise RuntimeError(
                        "Writer thread is gone but crops are still pending; aborting to avoid a hang."
                    ) from None

    def _refill() -> None:
        """Top the in-flight window back up to DECODE_WINDOW.

        Cached images are recognised here and enter the stream with future=None:
        they cost one stat() and no decode, but keep their position so ordering,
        counting and progress are identical to the serial pass.

        A crop existing is not enough on its own: save_detections() (the
        detector-box sidecar review reads for its overlay) was added after
        this cache format, and get_many()/DetectionCache.get() never runs the
        detector for an image it thinks is already recorded. Without also
        requiring the sidecar, an image whose crop predates that sidecar - or
        whose sidecar was lost for any other reason while the crop survived -
        would skip detection forever, on every future run, with nothing in
        the UI to explain why its boxes never appear. Checking for it here
        lets the normal incremental re-run heal that gap for exactly the
        images missing it, without forcing a full rebuild of an otherwise
        fully cached folder.
        """
        while len(pending) < DECODE_WINDOW:
            image_path = next(source, None)
            if image_path is None:
                return
            with PROFILE.stage("cache lookup"):
                target = crop_cache_path(cache_dir, image_path, image_format=params.image_format)
                has_detections = detections_cache_path(cache_dir, image_path).exists()
                already_cached = target.exists() and has_detections and not force
            future = None if already_cached else pool.submit(_decode_one, decoder, image_path)
            pending.append(_Job(image_path=image_path, target=target, future=future))

    started = time.monotonic()
    last_progress = started
    # Set when the first image that needs building is finished, so the reported
    # rate is not diluted by the near-instant skipping of a cached prefix.
    work_started: float | None = None
    processed = 0
    pool = ThreadPoolExecutor(max_workers=decode_workers, thread_name_prefix="picklikeme-decode")
    writer.start()
    try:
        _refill()
        while pending:
            job = pending.popleft()
            # Refill before the GPU work below, so decoders are busy filling the
            # window while this image is detected, cropped and queued to disk.
            _refill()
            processed += 1

            if job.future is None:
                stats["skipped"] += 1
            else:
                if work_started is None:
                    work_started = time.monotonic()
                decoded = job.future.result()
                if decoded.error is not None:
                    stats["errors"] += 1
                    print(
                        f"  ERROR on {job.image_path}: "
                        f"{type(decoded.error).__name__}: {decoded.error}"
                    )
                else:
                    try:
                        # Identical to the serial version: one image at a time,
                        # same detector call, same crop math, same encoder.
                        # collect_detections: record the runners-up while the
                        # detector has just run, so the analyzer never needs
                        # to re-run inference to draw them.
                        result = build_crop(decoded.rgb, detector, params, collect_detections=True)
                        class_name = (
                            coco_class_name(result.detection.label)
                            if result.detection is not None
                            else None
                        )
                        _enqueue_write(
                            _WriteJob(
                                target=job.target,
                                crop=result.crop,
                                image_path=job.image_path,
                                class_name=class_name,
                            )
                        )
                        # Record what the detector saw while it is free: the
                        # analyzer can then draw the boxes without inference.
                        save_detections(cache_dir, job.image_path, result)
                        stats["cached"] += 1
                        if class_name is None:
                            stats["fallbacks"] += 1
                        else:
                            stats["birds"] += 1
                            class_counts[class_name] += 1
                    except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the pass
                        stats["errors"] += 1
                        print(f"  ERROR on {job.image_path}: {type(exc).__name__}: {exc}")
                PROFILE.image_done()

            _reconcile_write_failures()

            now = time.monotonic()
            if now - last_progress >= PROGRESS_INTERVAL_SECONDS or processed == total:
                _print_progress(
                    processed,
                    total,
                    stats,
                    elapsed=now - started,
                    work_elapsed=0.0 if work_started is None else now - work_started,
                )
                last_progress = now
            if PROFILE.enabled and PROFILE.images and PROFILE.images % PROFILE_SUMMARY_INTERVAL_IMAGES == 0:
                print(PROFILE.progress_line())
    finally:
        # Ordered teardown, and it must hold on the exception/Ctrl+C path too:
        # drop queued decodes, let the writer drain what it already has, then let
        # in-flight decodes finish. The sentinel is FIFO behind every crop already
        # queued, so joining the writer proves nothing was left unwritten.
        for job in pending:
            if job.future is not None:
                job.future.cancel()
        if writer.is_alive():
            try:
                write_queue.put(None, timeout=WRITER_LIVENESS_TIMEOUT_SECONDS)
            except queue.Full:
                pass  # wedged writer; it is a daemon, so it cannot block exit
        writer.join(timeout=WRITER_JOIN_TIMEOUT_SECONDS)
        if writer.is_alive():
            print(
                f"  WARNING: writer thread still busy after {WRITER_JOIN_TIMEOUT_SECONDS:.0f}s; "
                "some crops may not have been written and will be rebuilt on the next run."
            )
        pool.shutdown(wait=True, cancel_futures=True)

    # Writes that failed after the last reconcile inside the loop.
    _reconcile_write_failures()

    stats["class_counts"] = dict(class_counts)
    if PROFILE.enabled:
        print(PROFILE.report())
    return stats


def _print_progress(processed: int, total: int, stats: dict, elapsed: float, work_elapsed: float) -> None:
    """One-line progress report: how far along, how fast, what was found, and
    when it should finish.

    Rate and ETA are measured over images that actually required work, timed
    from the first such image - not over everything processed. On a resumed run
    the already-cached prefix is consumed in seconds, and counting those made
    the rate open at a meaningless several-hundred img/s and the ETA *climb*
    as the average decayed. `work_elapsed` excludes that skip phase, so the
    numbers are usable from the first line.

    The ETA assumes the remaining images all need building, which is what a
    resumed run looks like once past its cached prefix. If a later stretch turns
    out to be cached too, the estimate is simply pessimistic.
    """
    percent = (processed / total * 100.0) if total else 100.0
    worked = stats["cached"] + stats["errors"]
    rate = worked / work_elapsed if work_elapsed > 0 and worked else 0.0
    eta = format_duration((total - processed) / rate) if rate > 0 else "n/a"
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] {processed:,}/{total:,} ({percent:.1f}%) "
        f"| {rate:.1f} img/s | detected {stats['birds']:,} | fallback {stats['fallbacks']:,} "
        f"| skipped {stats['skipped']:,} | errors {stats['errors']:,} "
        f"| elapsed {format_duration(elapsed)} | eta {eta}"
    )


def preprocess_folders(
    select_root: str,
    reject_root: str,
    cache_dir: str | Path,
    params: CropParams,
    device: str = "cuda",
    force: bool = False,
    decode_workers: int | None = None,
) -> dict:
    """Enumerate the select/reject roots and build the bird-crop cache for them.

    Shared by `picklikeme.preprocess` (standalone) and `picklikeme.run` (the
    preprocess -> train -> rank pipeline) so both enumerate images identically.
    """
    dataset = FolderLabelDataset(select_root=select_root, reject_root=reject_root, raw_root=select_root)
    image_paths = [item.image_path for item in dataset.items]
    print(f"Enumerated {len(image_paths)} images from select/reject roots.")
    stats = build_cache(
        image_paths, cache_dir, params, device=device, force=force, decode_workers=decode_workers
    )
    _print_cache_summary(cache_dir, stats)
    return stats


def _print_cache_summary(cache_dir: str | Path, stats: dict) -> None:
    print("\nCrop cache build complete:")
    print(f"  {'cache dir:':<20}{Path(cache_dir).resolve()}")
    print(f"  {'total images:':<20}{stats['total']:,}")
    print(f"  {'newly cached:':<20}{stats['cached']:,}")
    print(f"  {'already cached:':<20}{stats['skipped']:,}")
    print(f"  {'animal detected:':<20}{stats['birds']:,}")
    class_counts = stats.get("class_counts") or {}
    if class_counts:
        breakdown = ", ".join(
            f"{name} {count:,}" for name, count in sorted(class_counts.items(), key=lambda kv: -kv[1])
        )
        print(f"    {'by class:':<18}{breakdown}")
    print(f"  {'no-animal fallback:':<20}{stats['fallbacks']:,} (full frame cached)")
    print(f"  {'errors:':<20}{stats['errors']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the bird-crop cache used by training's --crop-birds")
    parser.add_argument("--select-root", required=True)
    parser.add_argument("--reject-root", required=True)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CROP_CACHE_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin-frac", type=float, default=CropParams.margin_frac)
    parser.add_argument("--conf-threshold", type=float, default=CropParams.conf_threshold)
    parser.add_argument(
        "--max-side",
        type=int,
        default=CropParams.max_side,
        help="Cap the cached crop's long side in pixels (default: unlimited - the crop's own "
        "original resolution, uncapped). Set an explicit value to trade detail for disk usage.",
    )
    parser.add_argument(
        "--image-format",
        choices=sorted(IMAGE_FORMAT_EXTENSIONS),
        default=CropParams.image_format,
        help=f"Vision Cache file format (default: {CropParams.image_format})",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=CropParams.jpeg_quality,
        help=f"JPEG quality when --image-format=jpeg, 0-100 (default: {CropParams.jpeg_quality}); "
        "ignored for --image-format=png, which is always lossless",
    )
    parser.add_argument(
        "--area-tie-frac",
        type=float,
        default=CropParams.area_tie_frac,
        help="Detections within this fraction of the largest survivor's area are tied on "
        "size and broken by confidence; anything smaller loses on size alone "
        f"(default: {CropParams.area_tie_frac})",
    )
    parser.add_argument(
        "--group-scene-threshold",
        type=int,
        default=CropParams.group_scene_threshold,
        help="At or above this many surviving detections, crop the box enclosing the whole "
        "group instead of one individual - flocks, herds, colonies are the subject, not any "
        f"single member of them (default: {CropParams.group_scene_threshold})",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild crops even if already cached")
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=None,
        help=f"Decoder threads feeding the GPU (default: min(8, cpu_count) = {default_decode_workers()}). "
        "Detection stays sequential and one image at a time regardless of this value.",
    )
    args = parser.parse_args()

    params = CropParams(
        margin_frac=args.margin_frac,
        conf_threshold=args.conf_threshold,
        max_side=args.max_side,
        area_tie_frac=args.area_tie_frac,
        group_scene_threshold=args.group_scene_threshold,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
    )
    with fatal_errors_logged_to_stdout():
        preprocess_folders(
            args.select_root,
            args.reject_root,
            args.cache_dir,
            params,
            device=args.device,
            force=args.force,
            decode_workers=args.decode_workers,
        )


if __name__ == "__main__":
    main()
