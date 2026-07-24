"""Auto-crop: write editor crop metadata from the bird detector.

Runs BEFORE importing into a photo editor. For every supported RAW under the
input folder it detects the bird (shared engine), computes an
aspect-ratio-preserving composition crop, and hands it to an exporter that
writes editor-readable crop metadata (Lightroom by default).

    python -m picklikeme.auto_crop --input "D:\\Photos" --margin 12

Reuses BirdDetector + compute_composition_crop (single source of truth);
training and preprocessing are unaffected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .bird_crop import BirdDetector, CropParams, compute_composition_crop
from .exporters import EXPORTERS
from .raw_io import RawImageLoader


def resolve_device(requested: str | None) -> str:
    """Auto-select GPU when available, else CPU. `requested` may be None (auto)
    or an explicit override; an explicit 'cuda' still falls back if unavailable."""
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


def discover_raw_images(input_folder: Path) -> list[str]:
    """Recursively find supported RAW files (NEF/ARW/CR3/DNG/...) case-insensitively."""
    return sorted(
        str(p) for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in RawImageLoader.RAW_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write editor crop sidecars/metadata from bird detection")
    parser.add_argument("--input", required=True, help="Folder of RAW images (processed recursively)")
    parser.add_argument("--margin", type=float, default=0.0, help="Percent to expand the bird box before fitting to the image aspect ratio (e.g. 10)")
    parser.add_argument("--overwrite-xmp", action="store_true", help="Overwrite existing sidecars / re-embed existing DNG crops (default: leave them untouched)")
    parser.add_argument("--exporter", default="lightroom", choices=sorted(EXPORTERS), help="Target editor format")
    parser.add_argument("--conf-threshold", type=float, default=CropParams.conf_threshold, help="Detection confidence threshold")
    parser.add_argument("--device", default=None, help="Device (default: auto - CUDA if available, else CPU)")
    parser.add_argument("--exiftool-path", default="exiftool", help="Path to exiftool (used only for DNG embedding)")
    args = parser.parse_args()

    input_folder = Path(args.input)
    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    images = discover_raw_images(input_folder)
    if not images:
        raise SystemExit(
            f"No supported RAW images ({sorted(RawImageLoader.RAW_EXTENSIONS)}) found under {input_folder.resolve()}"
        )
    print(f"Found {len(images)} RAW images under {input_folder.resolve()}")

    exporter_cls = EXPORTERS[args.exporter]
    exporter = exporter_cls(exiftool_path=args.exiftool_path) if args.exporter == "lightroom" else exporter_cls()

    # Fail early with a clear message if DNGs are present but exiftool is missing.
    if any(Path(p).suffix.lower() == ".dng" for p in images):
        from .ingest.metadata import ensure_exiftool_available

        ensure_exiftool_available(args.exiftool_path)

    device = resolve_device(args.device)
    print(f"Loading detector on {device} (read-only)...")
    detector = BirdDetector(device=device, conf_threshold=args.conf_threshold)
    decoder = RawImageLoader(raw_root=str(input_folder))
    margin_frac = args.margin / 100.0

    stats = {"written": 0, "embedded": 0, "skipped": 0, "no_bird": 0, "errors": 0}
    for path in images:
        name = Path(path).name
        try:
            full = decoder._decode_full_frame(path)
            height, width = full.shape[:2]
            detection = detector.detect_best_bird(full)
            if detection is None:
                stats["no_bird"] += 1
                print(f"  {name}: no bird detected -> skipped (no crop written)")
                continue
            crop = compute_composition_crop(detection, width, height, margin_frac)
            result = exporter.export(Path(path), crop, overwrite=args.overwrite_xmp)
            key = {"written": "written", "embedded": "embedded", "skipped_exists": "skipped"}[result.action]
            stats[key] += 1
            print(f"  {name}: {result.action} -> {result.output_path.name} (conf={detection.score:.2f})")
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the batch
            stats["errors"] += 1
            print(f"  {name}: ERROR {type(exc).__name__}: {exc}")

    print("\nAuto-crop complete:")
    print(f"  sidecars written:   {stats['written']}")
    print(f"  DNG crops embedded: {stats['embedded']}")
    print(f"  skipped (existing): {stats['skipped']}")
    print(f"  no bird detected:   {stats['no_bird']}")
    print(f"  errors:             {stats['errors']}")


if __name__ == "__main__":
    main()
