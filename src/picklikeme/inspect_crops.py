"""Visual QA for the bird-crop cache (read-only inspection tool).

Renders contact sheets of randomly sampled cached crops exactly as the model
will receive them (letterbox padding included, colors and aspect ratio
untouched), and a small detection-rate report. Detector failures (images that
fell back to the full frame) get their own contact sheet so they can be
eyeballed.

This tool does not modify the detector, preprocessing, caching, or training —
it only reads the cache and, to tell detected crops from full-frame fallbacks
(which preprocessing does not record per image), re-runs the detector
read-only on the cached images.

    python -m picklikeme.inspect_crops --select-root "..." --reject-root "..."
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bird_crop import CropParams, crop_cache_path, read_crop_params
from .config import DEFAULT_CROP_CACHE_DIR, DEFAULT_INSPECTION_DIR
from .dataset import FolderLabelDataset
from .raw_io import RawImageLoader


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                print(f"Requested device '{requested}' but CUDA is not available; using CPU")
                return "cpu"
        except ImportError:
            return "cpu"
    return requested


def find_cached_pairs(select_root: str, reject_root: str, cache_dir: Path) -> list[tuple[str, Path]]:
    """Enumerate source images and pair each with its cache file, keeping only
    those already cached. Enumerating from the source (not globbing the cache)
    is what lets us caption each crop with its original filename — the cache
    filename is only a hash of the source path."""
    dataset = FolderLabelDataset(select_root=select_root, reject_root=reject_root, raw_root=select_root)
    pairs: list[tuple[str, Path]] = []
    for item in dataset.items:
        cache_path = crop_cache_path(cache_dir, item.image_path)
        if cache_path.exists():
            pairs.append((item.image_path, cache_path))
    return pairs


def classify_pairs(pairs: list[tuple[str, Path]], detector, loader: RawImageLoader) -> list[tuple[str, Path, bool]]:
    """Tag each (source, cache) pair with found_bird by running the detector on
    the cached image. Deterministic and read-only; for full-frame fallbacks
    this reproduces exactly the no-bird result preprocessing saw."""
    tagged: list[tuple[str, Path, bool]] = []
    for source_path, cache_path in pairs:
        cached_rgb = loader._read_standard_image(cache_path)
        found = detector.best_bird_box(cached_rgb) is not None
        tagged.append((source_path, cache_path, found))
    return tagged


def _model_input_thumb(loader: RawImageLoader, source_path: str, thumb: int) -> Image.Image:
    """The exact model input (letterboxed, normalized then back to 8-bit),
    downscaled to a square thumbnail. Square->square so nothing is distorted;
    colors are unchanged and letterbox padding is shown as-is."""
    model_input = loader.load_image(source_path)  # HWC float [0,1], letterboxed to output_size
    arr = (model_input * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((thumb, thumb), Image.LANCZOS)


def _truncate(name: str, max_chars: int) -> str:
    return name if len(name) <= max_chars else name[: max_chars - 2] + ".."


def build_contact_sheet(
    cells: list[tuple[Image.Image, str]],
    cols: int,
    thumb: int,
    caption_h: int = 16,
) -> Image.Image:
    """Compose a grid of square thumbnails with a filename caption under each."""
    cell_w = thumb
    cell_h = thumb + caption_h
    rows = max(1, math.ceil(len(cells) / cols))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    max_chars = max(8, cell_w // 6)
    for i, (thumb_img, caption) in enumerate(cells):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h
        sheet.paste(thumb_img, (x, y))
        draw.text((x + 2, y + thumb + 2), _truncate(caption, max_chars), fill=(230, 230, 230), font=font)
    return sheet


def _write_sheets(
    tagged_subset: list[tuple[str, Path, bool]],
    loader: RawImageLoader,
    output_dir: Path,
    prefix: str,
    cols: int,
    thumb: int,
    per_sheet: int,
) -> list[Path]:
    written: list[Path] = []
    for page_start in range(0, len(tagged_subset), per_sheet):
        page = tagged_subset[page_start : page_start + per_sheet]
        cells = [(_model_input_thumb(loader, src, thumb), Path(src).name) for src, _cache, _found in page]
        sheet = build_contact_sheet(cells, cols=cols, thumb=thumb)
        page_no = page_start // per_sheet + 1
        out = output_dir / f"{prefix}_{page_no:02d}.png"
        sheet.save(out)
        written.append(out)
    return written


def _format_report(cache_dir: Path, total_cached: int, tagged: list[tuple[str, Path, bool]], sheets: dict) -> str:
    processed = len(tagged)
    detected = sum(1 for _s, _c, found in tagged if found)
    fallbacks = processed - detected
    rate = (detected / processed * 100.0) if processed else 0.0
    lines = [
        "Bird-crop cache inspection report",
        "=================================",
        f"Cache directory:            {cache_dir.resolve()}",
        f"Total cached images:        {total_cached}",
        f"Images processed (sampled): {processed}",
        f"Successful bird detections: {detected}",
        f"Full-frame fallbacks:       {fallbacks}",
        f"Detection success rate:     {rate:.1f}%",
        "",
        "Note: found/fallback is re-derived by running the detector read-only on the",
        "cached images (preprocessing does not persist per-image outcomes), over the",
        "sampled subset above - not the entire archive.",
        "",
        "Contact sheets:",
    ]
    for label, paths in sheets.items():
        if paths:
            for p in paths:
                lines.append(f"  {label}: {p.resolve()}")
        else:
            lines.append(f"  {label}: (none)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render contact sheets of cached bird crops for visual QA")
    parser.add_argument("--select-root", required=True)
    parser.add_argument("--reject-root", required=True)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CROP_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_INSPECTION_DIR))
    parser.add_argument("--sample-size", type=int, default=100, help="Approx number of cached crops to sample and classify")
    parser.add_argument("--cols", type=int, default=10, help="Thumbnails per row in a contact sheet")
    parser.add_argument("--thumb", type=int, default=256, help="Thumbnail size in pixels")
    parser.add_argument("--per-sheet", type=int, default=100, help="Max thumbnails per contact-sheet image")
    parser.add_argument("--output-size", type=int, default=384, help="Model input size to render (match training)")
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = read_crop_params(cache_dir)
    if params is None:
        print(f"No {cache_dir}/crop_params.json found; using default crop params. Has the cache been built?")
        params = CropParams()

    pairs = find_cached_pairs(args.select_root, args.reject_root, cache_dir)
    if not pairs:
        raise SystemExit(
            f"No cached crops found under {cache_dir.resolve()} for these roots. "
            "Run `python -m picklikeme.preprocess` first."
        )
    print(f"Found {len(pairs)} cached crops for the given roots.")

    rng = random.Random(args.seed)
    sample = pairs if len(pairs) <= args.sample_size else rng.sample(pairs, args.sample_size)
    print(f"Sampling {len(sample)} crops for inspection (seed={args.seed}).")

    # Loader used for display reads the crop cache and letterboxes to the model
    # input size, so thumbnails are exactly what the model receives.
    display_loader = RawImageLoader(
        raw_root=args.select_root,
        output_size=(args.output_size, args.output_size),
        resize_mode=args.resize_mode,
        crop_cache_dir=str(cache_dir),
    )
    # Loader used to read raw cached PNGs at native size for the detector.
    cache_reader = RawImageLoader(raw_root=args.select_root)

    from .bird_crop import BirdDetector

    device = resolve_device(args.device)
    print(f"Loading detector on {device} to classify detected vs fallback (read-only)...")
    detector = BirdDetector(device=device, conf_threshold=params.conf_threshold)

    tagged = classify_pairs(sample, detector, cache_reader)
    detected = [t for t in tagged if t[2]]
    fallbacks = [t for t in tagged if not t[2]]
    print(f"Classified: {len(detected)} detected, {len(fallbacks)} fallback.")

    crop_sheets = _write_sheets(detected, display_loader, output_dir, "crops_sheet", args.cols, args.thumb, args.per_sheet)
    fallback_sheets = _write_sheets(fallbacks, display_loader, output_dir, "fallback_sheet", args.cols, args.thumb, args.per_sheet)

    report = _format_report(
        cache_dir,
        total_cached=len(pairs),
        tagged=tagged,
        sheets={"crops": crop_sheets, "fallbacks": fallback_sheets},
    )
    report_path = output_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nReport written to {report_path.resolve()}")


if __name__ == "__main__":
    main()
