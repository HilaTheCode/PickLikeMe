"""Rank a NEW folder of images with an already-trained model (no training).

Point this at a directory the model has never seen. It builds the bird-crop
cache for those images (same preprocessing as training), loads the trained
weights from a checkpoint, scores every image by predicted "keep" preference,
and writes a ranked CSV (highest score first).

    python -m picklikeme.rank --input "D:\\NewShoot" --checkpoint checkpoints/model_checkpoint.pt

The backbone must match the one the checkpoint was trained with (default is the
same DINOv3-Huge+ default as training). Pass --no-crop-birds to score full
frames instead of bird crops (only sensible if the model was trained that way).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .auto_crop import resolve_device
from .bird_crop import CropParams
from .config import DEFAULT_CHECKPOINT_PATH, DEFAULT_CROP_CACHE_DIR, DEFAULT_MAX_CSV_ROWS
from .dataset import UnlabeledImageDataset
from .model import DINOV3_BACKBONE, ModelConfig, PreferenceHead
from .organize import (
    DEFAULT_SELECTION_PERCENTAGE,
    ORGANIZE_DIRNAMES,
    InvalidSelectionPercentage,
    organize_ranked_images,
    validate_selection_percentage,
)
from .preprocess import build_cache
from .raw_io import RawImageLoader
from .train import load_checkpoint, rank_dataset, timestamped_output_path, write_results_csv


def _boolean(value: str) -> bool:
    """Accept true/false (and the usual synonyms) for --organize-output.

    argparse's store_true cannot express "--organize-output false", and the
    feature is specified as taking a value.
    """
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if text in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected true or false, got {value!r}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Built separately from main() so the CLI's defaults are inspectable
    without running a ranking pass (mirrors train.build_arg_parser)."""
    parser = argparse.ArgumentParser(description="Rank a new, unlabeled folder with a trained checkpoint (no training)")
    parser.add_argument("--input", required=True, help="Folder of RAW images to rank (recursive, model has not seen these)")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="Trained checkpoint to load (default: the project's rolling checkpoint)")
    parser.add_argument(
        "--output-csv",
        default="rankings.csv",
        help="Base name for the ranked CSV. The run's start date/time is appended "
        "to the stem so every run gets a unique file and no previous rankings are "
        "overwritten (e.g. rankings_20260725-143000.csv).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_CSV_ROWS,
        help=f"Maximum lines per ranked CSV before it rolls over to name_1.csv, name_2.csv "
        f"(default: {DEFAULT_MAX_CSV_ROWS:,}). Change the default in picklikeme/config.py.",
    )
    parser.add_argument("--device", default=None, help="Device (default: auto - CUDA if available, else CPU)")
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"])
    parser.add_argument("--backbone", default=DINOV3_BACKBONE, help="Must match the checkpoint's backbone (default: same as training)")
    parser.add_argument("--crop-birds", action=argparse.BooleanOptionalAction, default=True, help="Score bird crops (default); --no-crop-birds scores full frames")
    parser.add_argument("--crop-cache-dir", default=str(DEFAULT_CROP_CACHE_DIR), help="Where to build/read the bird-crop cache")
    parser.add_argument("--margin-frac", type=float, default=CropParams.margin_frac)
    parser.add_argument("--conf-threshold", type=float, default=CropParams.conf_threshold)
    parser.add_argument("--max-side", type=int, default=CropParams.max_side)
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild crops even if already cached")
    parser.add_argument(
        "--organize-output",
        type=_boolean,
        nargs="?",
        const=True,
        default=True,
        metavar="true/false",
        help="Move the ranked images into selected_by_ai/ and rejected_by_ai/ under the input "
        "folder (default: true). Pass false to leave every file where it is.",
    )
    parser.add_argument(
        "--selection-percentage",
        type=float,
        default=DEFAULT_SELECTION_PERCENTAGE,
        help=f"Percentage of the highest-ranked images to place in selected_by_ai "
        f"(default: {DEFAULT_SELECTION_PERCENTAGE:g}; 0-100).",
    )
    parser.add_argument(
        "--organize-dir",
        default=None,
        help="Where selected_by_ai/ and rejected_by_ai/ are created (default: the --input folder)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_started = datetime.now()

    # Validated before any work: a bad percentage must not surface after an
    # hour of ranking.
    if args.organize_output:
        try:
            validate_selection_percentage(args.selection_percentage)
        except InvalidSelectionPercentage as exc:
            raise SystemExit(str(exc)) from exc

    input_folder = Path(args.input)
    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path.resolve()}\n"
            "Train a model first (python -m picklikeme.run ...), or pass --checkpoint."
        )

    # Skip our own output folders: without this, a second run would rank the
    # images a previous run had already filed and shuffle them again.
    dataset = UnlabeledImageDataset.from_folder(input_folder, exclude_dirs=set(ORGANIZE_DIRNAMES))
    if len(dataset) == 0:
        raise SystemExit(f"No RAW images (.arw/.nef/.cr3) found under {input_folder.resolve()}")
    print(f"Found {len(dataset)} images to rank under {input_folder.resolve()}")

    # Resolved before the crop-cache build so the destination is known up front,
    # not only after a long preprocessing + scoring pass.
    output_csv = timestamped_output_path(args.output_csv, run_started)
    print(f"Ranked CSV for this run will be written to {output_csv.resolve()}")

    device = resolve_device(args.device)

    crop_cache_dir = None
    if args.crop_birds:
        print("=" * 64)
        print("STEP 1/2: Building bird-crop cache for the new folder")
        print("=" * 64)
        params = CropParams(
            margin_frac=args.margin_frac,
            conf_threshold=args.conf_threshold,
            max_side=args.max_side,
        )
        image_paths = [item.image_path for item in dataset.items]
        build_cache(image_paths, args.crop_cache_dir, params, device=device, force=args.force_preprocess)
        crop_cache_dir = args.crop_cache_dir
    else:
        print("Scoring full frames (--no-crop-birds).")

    print("=" * 64)
    print("STEP 2/2: Loading model and ranking")
    print("=" * 64)
    # pretrained=False: the checkpoint supplies all weights (including the frozen
    # backbone), so there is no need to download pretrained weights just to
    # overwrite them - and it keeps ranking usable offline.
    model_config = ModelConfig(backbone=args.backbone, pretrained=False, freeze_backbone=True)
    model = PreferenceHead(model_config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = RawImageLoader(str(input_folder), resize_mode=args.resize_mode, crop_cache_dir=crop_cache_dir)
    ranked = rank_dataset(model, dataset, loader, device=device)

    output_paths = write_results_csv(
        output_csv,
        dataset,
        ranked,
        select_root=str(input_folder),
        reject_root="(inference - no labels)",
        max_rows=args.max_rows,
    )
    print("\nTop-ranked images:")
    for rank, entry in enumerate(ranked[:10], start=1):
        print(f"{rank}. {entry[0]}: {entry[1]:.4f}")
    print(f"\nRanked CSV written to {output_paths[0]}")
    if len(output_paths) > 1:
        print(f"Additional CSV files: {', '.join(str(path) for path in output_paths[1:])}")


if __name__ == "__main__":
    main()
