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
from .platform import resolve_torch_device
from .raw_io import RawImageLoader


def resolve_device(requested: str | None) -> str:
    """Auto-select the best available torch device for this workflow."""
    return resolve_torch_device(requested)


def discover_raw_images(input_folder: Path) -> list[str]:
    """Recursively find supported RAW files (NEF/ARW/CR3/DNG/...) case-insensitively."""
    return sorted(
        str(p) for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in RawImageLoader.RAW_EXTENSIONS
    )


def generate_lightroom_crops(
    input_folder: str | Path,
    *,
    margin_percent: float = 0.0,
    overwrite_xmp: bool = False,
    conf_threshold: float | None = None,
    device: str | None = None,
    exiftool_path: str = "exiftool",
) -> dict:
    """Generate Lightroom-ready crop metadata for every supported RAW under a folder."""
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")

    images = discover_raw_images(input_path)
    if not images:
        return {"processed": 0, "message": "No compatible images were found.", "stats": {}, "details": []}

    exporter_cls = EXPORTERS["lightroom"]
    exporter = exporter_cls(exiftool_path=exiftool_path)

    if any(Path(p).suffix.lower() == ".dng" for p in images):
        from .ingest.metadata import ensure_exiftool_available

        ensure_exiftool_available(exiftool_path)

    selected_device = resolve_device(device)
    detector = BirdDetector(
        device=selected_device,
        conf_threshold=conf_threshold if conf_threshold is not None else CropParams.conf_threshold,
    )
    decoder = RawImageLoader(raw_root=str(input_path))
    margin_frac = margin_percent / 100.0

    stats = {"written": 0, "embedded": 0, "skipped": 0, "no_bird": 0, "errors": 0}
    details: list[dict] = []
    for path in images:
        name = Path(path).name
        try:
            full = decoder._decode_full_frame(path)
            height, width = full.shape[:2]
            detection = detector.detect_best_bird(full)
            if detection is None:
                stats["no_bird"] += 1
                details.append({"path": path, "status": "skipped", "message": f"{name}: no bird detected -> skipped"})
                continue
            crop = compute_composition_crop(detection, width, height, margin_frac)
            result = exporter.export(Path(path), crop, overwrite=overwrite_xmp)
            key = {"written": "written", "embedded": "embedded", "skipped_exists": "skipped"}[result.action]
            stats[key] += 1
            details.append({"path": path, "status": result.action, "message": f"{name}: {result.action}"})
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the batch
            stats["errors"] += 1
            details.append({"path": path, "status": "error", "message": f"{name}: ERROR {type(exc).__name__}: {exc}"})

    return {
        "processed": len(images),
        "message": f"Processed {len(images)} RAW image(s) with auto crop.",
        "stats": stats,
        "details": details,
        "device": selected_device,
    }


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

    result = generate_lightroom_crops(
        input_folder,
        margin_percent=args.margin,
        overwrite_xmp=args.overwrite_xmp,
        conf_threshold=args.conf_threshold,
        device=args.device,
        exiftool_path=args.exiftool_path,
    )
    print(f"Loading detector on {result['device']} (read-only)...")
    for detail in result["details"]:
        print(f"  {detail['message']}")

    print("\nAuto-crop complete:")
    print(f"  sidecars written:   {result['stats']['written']}")
    print(f"  DNG crops embedded: {result['stats']['embedded']}")
    print(f"  skipped (existing): {result['stats']['skipped']}")
    print(f"  no bird detected:   {result['stats']['no_bird']}")
    print(f"  errors:             {result['stats']['errors']}")


if __name__ == "__main__":
    main()
