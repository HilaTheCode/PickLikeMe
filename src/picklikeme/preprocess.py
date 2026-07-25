"""One-time bird-crop cache builder.

Decodes each RAW once, detects the bird, crops tightly (small margin, aspect
preserved), and writes the crop to the cache that RawImageLoader reads during
training. Run this before training with --crop-birds.

    python -m picklikeme.preprocess --select-root "..." --reject-root "..."

Idempotent: images already cached are skipped, so it can be re-run to finish an
interrupted pass or to pick up newly added images. Detection runs in this
single process (default device cuda) — never inside DataLoader workers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .bird_crop import (
    BirdDetector,
    CropParams,
    build_crop,
    crop_cache_path,
    read_crop_params,
    save_crop_png,
    write_crop_params,
)
from .config import DEFAULT_CROP_CACHE_DIR
from .dataset import FolderLabelDataset
from .raw_io import RawImageLoader


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                print(f"Requested device '{requested}' but CUDA is not available; using CPU")
                return "cpu"
        except ImportError:
            return "cpu"
    return requested


def build_cache(
    image_paths: list[str],
    cache_dir: str | Path,
    params: CropParams,
    device: str = "cuda",
    force: bool = False,
) -> dict:
    cache_dir = Path(cache_dir)
    device = _resolve_device(device)

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
    detector = BirdDetector(device=device, conf_threshold=params.conf_threshold)

    stats = {"total": len(image_paths), "cached": 0, "skipped": 0, "birds": 0, "fallbacks": 0, "errors": 0}
    for image_path in tqdm(image_paths, desc="Building bird-crop cache", unit="img"):
        target = crop_cache_path(cache_dir, image_path)
        if target.exists() and not force:
            stats["skipped"] += 1
            continue
        try:
            # Decode to full-resolution RGB uint8 for detection + cropping.
            rgb_uint8 = decoder._decode_full_frame(image_path)
            result = build_crop(rgb_uint8, detector, params)
            save_crop_png(target, result.crop)
            stats["cached"] += 1
            stats["birds" if result.detection is not None else "fallbacks"] += 1
        except Exception as exc:  # noqa: BLE001 - report and continue, one bad file shouldn't stop the pass
            stats["errors"] += 1
            print(f"  ERROR on {image_path}: {type(exc).__name__}: {exc}")

    return stats


def preprocess_folders(
    select_root: str,
    reject_root: str,
    cache_dir: str | Path,
    params: CropParams,
    device: str = "cuda",
    force: bool = False,
) -> dict:
    """Enumerate the select/reject roots and build the bird-crop cache for them.

    Shared by `picklikeme.preprocess` (standalone) and `picklikeme.run` (the
    preprocess -> train -> rank pipeline) so both enumerate images identically.
    """
    dataset = FolderLabelDataset(select_root=select_root, reject_root=reject_root, raw_root=select_root)
    image_paths = [item.image_path for item in dataset.items]
    print(f"Enumerated {len(image_paths)} images from select/reject roots.")
    stats = build_cache(image_paths, cache_dir, params, device=device, force=force)
    _print_cache_summary(cache_dir, stats)
    return stats


def _print_cache_summary(cache_dir: str | Path, stats: dict) -> None:
    print("\nCrop cache build complete:")
    print(f"  cache dir:        {Path(cache_dir).resolve()}")
    print(f"  total images:     {stats['total']}")
    print(f"  newly cached:     {stats['cached']}")
    print(f"  already cached:   {stats['skipped']}")
    print(f"  bird detected:    {stats['birds']}")
    print(f"  no-bird fallback: {stats['fallbacks']} (full frame cached)")
    print(f"  errors:           {stats['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the bird-crop cache used by training's --crop-birds")
    parser.add_argument("--select-root", required=True)
    parser.add_argument("--reject-root", required=True)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CROP_CACHE_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin-frac", type=float, default=CropParams.margin_frac)
    parser.add_argument("--conf-threshold", type=float, default=CropParams.conf_threshold)
    parser.add_argument("--max-side", type=int, default=CropParams.max_side)
    parser.add_argument("--force", action="store_true", help="Rebuild crops even if already cached")
    args = parser.parse_args()

    params = CropParams(
        margin_frac=args.margin_frac,
        conf_threshold=args.conf_threshold,
        max_side=args.max_side,
    )
    preprocess_folders(
        args.select_root,
        args.reject_root,
        args.cache_dir,
        params,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
