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
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bird_crop import (
    CropParams,
    build_crop,
    crop_cache_path,
    read_crop_params,
)
from .config import DEFAULT_CROP_CACHE_DIR, DEFAULT_INSPECTION_DIR
from .dataset import FolderLabelDataset
from .raw_io import RawImageLoader

# Supported inputs for folder mode: the exact RAW formats the pipeline decodes
# (so the acceptance test mirrors preprocessing/training — includes .nef/.cr3/
# .arw), plus common standard image formats for convenience.
SUPPORTED_INPUT_EXTS = RawImageLoader.RAW_EXTENSIONS | {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}


def resolve_device(requested: str | None) -> str:
    """Auto-select the device: GPU when available, else CPU. `requested` may be
    None (auto) or an explicit override like 'cpu'."""
    want_cuda = requested is None or requested.startswith("cuda")
    if want_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                return requested if (requested and requested != "cuda") else "cuda"
        except ImportError:
            pass
        if requested is not None:
            print(f"Requested device '{requested}' but CUDA is not available; using CPU")
        return "cpu"
    return requested


def discover_images(input_folder: Path) -> list[str]:
    """Recursively find every supported image under a folder, case-insensitively
    (e.g. .CR3, .cr3, .Cr3 all match)."""
    return sorted(
        str(p) for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTS
    )


def _build_loader(raw_root: str, args, crop_cache_dir: str | None = None) -> RawImageLoader:
    """Construct a loader that mirrors training: omit output_size when the user
    didn't override it, so RawImageLoader's own default (what training uses) is
    applied, and default to letterbox padding."""
    kwargs: dict = {"raw_root": raw_root, "resize_mode": args.resize_mode}
    if args.output_size:
        kwargs["output_size"] = (args.output_size, args.output_size)
    if crop_cache_dir is not None:
        kwargs["crop_cache_dir"] = crop_cache_dir
    return RawImageLoader(**kwargs)


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


# ===========================================================================
# Folder mode: run the live pipeline on a user-supplied acceptance folder
# ===========================================================================

@dataclass
class PipelineResult:
    source_path: str
    found: bool
    score: float | None
    box: tuple[int, int, int, int] | None          # raw detected box
    expanded_box: tuple[int, int, int, int] | None  # box actually used for the crop
    original_size: tuple[int, int]                   # (width, height)
    crop_size: tuple[int, int]                       # (width, height) of the tight crop
    full_rgb: np.ndarray                             # decoded original (uint8 RGB)
    model_input_rgb: np.ndarray                      # exactly what the model receives (uint8 RGB)


def run_pipeline(loader: RawImageLoader, detector, source_path: str, params: CropParams) -> PipelineResult:
    """Decode -> build_crop -> resize/pad, then read the box/score/sizes off the
    result for the report and overlay. Reuses bird_crop.build_crop and
    RawImageLoader directly (no re-implemented detection or crop logic), so the
    rendered crop is byte-identical to what training receives."""
    full = loader._decode_full_frame(source_path)  # uint8 RGB, full resolution
    height, width = full.shape[:2]

    result = build_crop(full, detector, params)
    crop = result.crop
    detection = result.detection

    if loader.resize_mode == "letterbox":
        model_input = loader._letterbox(crop)
    else:
        import cv2

        model_input = cv2.resize(crop, loader.output_size, interpolation=cv2.INTER_AREA)

    return PipelineResult(
        source_path=source_path,
        found=detection is not None,
        score=detection.score if detection is not None else None,
        box=tuple(int(round(v)) for v in detection.box) if detection is not None else None,
        expanded_box=result.expanded_box,
        original_size=(width, height),
        crop_size=(crop.shape[1], crop.shape[0]),
        full_rgb=full,
        model_input_rgb=model_input,
    )


def _fit_into_square(image_rgb: np.ndarray, size: int, bg: tuple[int, int, int] = (24, 24, 24)) -> Image.Image:
    """Thumbnail an image into a square canvas, preserving aspect ratio (letterbox
    into the cell) so grids align without distorting the picture."""
    pil = Image.fromarray(image_rgb)
    pil.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(pil, ((size - pil.width) // 2, (size - pil.height) // 2))
    return canvas


def draw_bbox_overlay(result: PipelineResult) -> Image.Image:
    """Original image with the detected box (green) and the expanded crop
    region (yellow) drawn on top; 'NO BIRD' in red for fallbacks."""
    image = Image.fromarray(result.full_rgb).copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_w = max(3, image.width // 250)
    if result.found and result.box is not None:
        draw.rectangle(result.box, outline=(0, 255, 0), width=line_w)
        if result.expanded_box is not None:
            draw.rectangle(result.expanded_box, outline=(255, 230, 0), width=max(2, line_w // 2))
        label = f"bird {result.score:.2f}"
        draw.text((result.box[0] + 2, max(0, result.box[1] - 12)), label, fill=(0, 255, 0), font=font)
    else:
        draw.text((6, 6), "NO BIRD - full-frame fallback", fill=(255, 60, 60), font=font)
    return image


def build_pair_sheet(
    rows: list[tuple[Image.Image, Image.Image, str]],
    thumb: int,
    caption_h: int = 18,
    arrow_w: int = 44,
) -> Image.Image:
    """One row per image: [original] -> [final model input], filename beneath."""
    row_w = thumb + arrow_w + thumb
    row_h = thumb + caption_h
    sheet = Image.new("RGB", (row_w, row_h * max(1, len(rows))), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (left, right, caption) in enumerate(rows):
        y = i * row_h
        sheet.paste(left, (0, y))
        draw.text((thumb + arrow_w // 2 - 6, y + thumb // 2 - 4), "->", fill=(230, 230, 230), font=font)
        sheet.paste(right, (thumb + arrow_w, y))
        draw.text((2, y + thumb + 2), _truncate(caption, max(8, row_w // 6)), fill=(230, 230, 230), font=font)
    return sheet


def _paginate(items: list, per_sheet: int) -> list[list]:
    return [items[i : i + per_sheet] for i in range(0, len(items), per_sheet)] or [[]]


def write_folder_report(output_dir: Path, results: list[PipelineResult], errors: list[tuple[str, str]]) -> Path:
    processed = len(results)
    detected = sum(1 for r in results if r.found)
    failures = processed - detected
    rate = (detected / processed * 100.0) if processed else 0.0

    lines = [
        "Bird detection / crop acceptance report",
        "=======================================",
        f"Images processed:           {processed}",
        f"Successful detections:      {detected}",
        f"Detector failures (fallback): {failures}",
        f"Detection success rate:     {rate:.1f}%",
    ]
    if errors:
        lines.append(f"Unreadable images (skipped): {len(errors)}")
    lines.append("")
    lines.append("Per-image detail:")
    for r in results:
        name = Path(r.source_path).name
        if r.found:
            box = ",".join(str(v) for v in r.box)
            lines.append(
                f"  {name}: DETECTED conf={r.score:.3f} box=[{box}] "
                f"orig={r.original_size[0]}x{r.original_size[1]} crop={r.crop_size[0]}x{r.crop_size[1]}"
            )
        else:
            lines.append(
                f"  {name}: *** FALLBACK (no bird) *** conf=n/a box=n/a "
                f"orig={r.original_size[0]}x{r.original_size[1]} crop={r.crop_size[0]}x{r.crop_size[1]} (full frame)"
            )
    for path, err in errors:
        lines.append(f"  {Path(path).name}: ERROR {err}")

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Machine-readable companion for archiving/comparison.
    with (output_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "detected", "fallback", "confidence", "box_x1", "box_y1", "box_x2", "box_y2", "orig_w", "orig_h", "crop_w", "crop_h"])
        for r in results:
            box = r.box if r.box is not None else ("", "", "", "")
            writer.writerow([
                Path(r.source_path).name, r.found, not r.found,
                f"{r.score:.4f}" if r.score is not None else "",
                *box, r.original_size[0], r.original_size[1], r.crop_size[0], r.crop_size[1],
            ])
    return report_path


def inspect_folder(args) -> None:
    from .bird_crop import BirdDetector

    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    image_paths = discover_images(input_folder)
    if not image_paths:
        raise SystemExit(f"No supported images ({sorted(SUPPORTED_INPUT_EXTS)}) found in {input_folder.resolve()}")
    print(f"Found {len(image_paths)} images in {input_folder.resolve()}")

    # Self-contained, timestamped run folder for archiving.
    run_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    params = CropParams(margin_frac=args.margin_frac, conf_threshold=args.conf_threshold, max_side=args.max_side)
    loader = _build_loader(str(input_folder), args)
    device = resolve_device(args.device)
    print(f"Loading detector on {device} (read-only)...")
    detector = BirdDetector(device=device, conf_threshold=params.conf_threshold)

    results: list[PipelineResult] = []
    errors: list[tuple[str, str]] = []
    for path in image_paths:
        try:
            result = run_pipeline(loader, detector, path, params)
            results.append(result)
            # Per-image side-by-side saved individually for close inspection.
            left = _fit_into_square(result.full_rgb, args.thumb)
            right = Image.fromarray(result.model_input_rgb).resize((args.thumb, args.thumb), Image.LANCZOS)
            build_pair_sheet([(left, right, Path(path).name)], thumb=args.thumb).save(
                images_dir / f"{Path(path).stem}_compare.png"
            )
            status = f"conf={result.score:.2f}" if result.found else "FALLBACK"
            print(f"  {Path(path).name}: {'bird' if result.found else 'no bird'} ({status})")
        except Exception as exc:  # noqa: BLE001 - one unreadable file shouldn't stop the acceptance run
            errors.append((path, f"{type(exc).__name__}: {exc}"))
            print(f"  {Path(path).name}: ERROR {type(exc).__name__}: {exc}")

    # Contact sheet: rows of original -> final crop.
    pair_rows = [
        (
            _fit_into_square(r.full_rgb, args.thumb),
            Image.fromarray(r.model_input_rgb).resize((args.thumb, args.thumb), Image.LANCZOS),
            Path(r.source_path).name,
        )
        for r in results
    ]
    pair_sheets = []
    for page_no, page in enumerate(_paginate(pair_rows, args.per_sheet), start=1):
        if not page:
            continue
        sheet = build_pair_sheet(page, thumb=args.thumb)
        out = run_dir / f"comparison_sheet_{page_no:02d}.png"
        sheet.save(out)
        pair_sheets.append(out)

    # Contact sheet: originals with bounding-box overlay.
    overlay_cells = [(_fit_into_square(np.asarray(draw_bbox_overlay(r)), args.thumb), Path(r.source_path).name) for r in results]
    overlay_sheets = []
    for page_no, page in enumerate(_paginate(overlay_cells, args.per_sheet), start=1):
        if not page:
            continue
        sheet = build_contact_sheet(page, cols=args.cols, thumb=args.thumb)
        out = run_dir / f"bbox_overlay_sheet_{page_no:02d}.png"
        sheet.save(out)
        overlay_sheets.append(out)

    report_path = write_folder_report(run_dir, results, errors)

    detected = sum(1 for r in results if r.found)
    print("\n" + report_path.read_text(encoding="utf-8"))
    print(f"\nAll outputs written to: {run_dir.resolve()}")
    print(f"  comparison sheets: {len(pair_sheets)}")
    print(f"  bbox overlay sheets: {len(overlay_sheets)}")
    print(f"  per-image images:  {images_dir.resolve()}")
    print(f"  report:            {report_path.resolve()} (+ report.csv)")
    print(f"Summary: {detected}/{len(results)} detected, {len(results) - detected} fallback, {len(errors)} errors.")


# ===========================================================================
# Cache mode (existing): sample and inspect an already-built crop cache
# ===========================================================================

def inspect_cache(args) -> None:
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

    display_loader = _build_loader(args.select_root, args, crop_cache_dir=str(cache_dir))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual QA for bird detection and cropping")
    # Two modes: --input-folder (live acceptance run) OR --select-root/--reject-root (sample the built cache).
    parser.add_argument("--input-folder", default=None, help="Run the live pipeline on every image in this folder (acceptance test)")
    parser.add_argument("--select-root", default=None, help="Cache mode: sample crops already built for this select root")
    parser.add_argument("--reject-root", default=None, help="Cache mode: paired reject root")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CROP_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_INSPECTION_DIR))
    parser.add_argument("--sample-size", type=int, default=100, help="Cache mode: approx crops to sample")
    parser.add_argument("--cols", type=int, default=10, help="Thumbnails per row in a grid contact sheet")
    parser.add_argument("--thumb", type=int, default=256, help="Thumbnail size in pixels")
    parser.add_argument("--per-sheet", type=int, default=100, help="Max items per contact-sheet image")
    parser.add_argument("--output-size", type=int, default=None, help="Model input size to render (default: the training pipeline's own default)")
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"], help="Default letterbox = aspect-preserving padding, matching training")
    parser.add_argument("--margin-frac", type=float, default=CropParams.margin_frac, help="Folder mode: crop safety margin (match preprocess)")
    parser.add_argument("--conf-threshold", type=float, default=CropParams.conf_threshold, help="Folder mode: detection threshold (match preprocess)")
    parser.add_argument("--max-side", type=int, default=CropParams.max_side, help="Folder mode: crop long-side cap (match preprocess)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Device (default: auto - CUDA if available, else CPU)")
    args = parser.parse_args()

    if args.input_folder:
        inspect_folder(args)
    elif args.select_root and args.reject_root:
        inspect_cache(args)
    else:
        raise SystemExit("Provide either --input-folder (acceptance run) or --select-root and --reject-root (cache mode).")


if __name__ == "__main__":
    main()
