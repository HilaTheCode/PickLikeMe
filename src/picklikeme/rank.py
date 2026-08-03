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
from typing import Callable

from .analytics import DEFAULT_ANALYTICS_DB, record_run
from .auto_crop import resolve_device
from .bird_crop import IMAGE_FORMAT_EXTENSIONS, CropParams
from .config import DEFAULT_CHECKPOINT_PATH, DEFAULT_CROP_CACHE_DIR, DEFAULT_MAX_CSV_ROWS, cli_prefix
from .dataset import UnlabeledImageDataset
from .model import DINOV3_BACKBONE, ModelConfig, PreferenceHead
from .organize import (
    DEFAULT_SELECTION_PERCENTAGE,
    ORGANIZE_DIRNAMES,
    REJECTED_DIRNAME,
    SELECTED_DIRNAME,
    InvalidSelectionPercentage,
    organize_ranked_images,
    validate_selection_percentage,
)
from .preprocess import CropCacheVersionMismatch, build_cache
from .profiling import PipelineProfiler
from .raw_io import RawImageLoader
from .sidecar import AI_STRATEGY_ID, RANKING_FILENAME, SIDECAR_DIRNAME, ranking_path, write_run_metadata
from .train import load_checkpoint, rank_dataset, timestamped_output_path, write_results_csv


def _folder_already_organized(input_folder: Path) -> bool:
    """True when every RAW under `input_folder` was excluded only because it
    already lives under _Selected/_Rejected - i.e. this folder was ranked
    and arranged before, not never touched. Distinguishes that case from a
    genuinely empty folder so the two can get different error messages."""
    return len(UnlabeledImageDataset.from_folder(input_folder)) > 0


def rank_folder(
    input_folder: str | Path,
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: str | None = None,
    backbone: str = DINOV3_BACKBONE,
    crop_birds: bool = True,
    crop_cache_dir: str | Path = DEFAULT_CROP_CACHE_DIR,
    margin_frac: float = CropParams.margin_frac,
    conf_threshold: float = CropParams.conf_threshold,
    max_side: int | None = CropParams.max_side,
    image_format: str = CropParams.image_format,
    jpeg_quality: int = CropParams.jpeg_quality,
    force_preprocess: bool = False,
    resize_mode: str = "letterbox",
    max_rows: int = DEFAULT_MAX_CSV_ROWS,
    on_stage: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    analytics_db: str | Path = DEFAULT_ANALYTICS_DB,
) -> dict:
    """Programmatic ranking entry point, reused by `main()` and by the
    desktop app's "Rank by AI" action.

    Unlike the CLI, this never organizes the output into Selected/Rejected -
    the review workflow expects every image to stay where it is until the
    photographer reviews it and explicitly arranges the folder. Raises
    FileNotFoundError/ValueError instead of calling sys.exit, so a caller
    (a UI) can catch and display the error itself.

    `analytics_db` defaults to the real shared database - overridable so
    tests (and any caller that wants an isolated run history) never write
    into it. See `analytics.capture.record_run`.
    """
    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    dataset = UnlabeledImageDataset.from_folder(input_folder, exclude_dirs=set(ORGANIZE_DIRNAMES) | {SIDECAR_DIRNAME})
    if len(dataset) == 0:
        if _folder_already_organized(input_folder):
            raise ValueError(
                f"{input_folder.resolve()} has already been organized into "
                f"{SELECTED_DIRNAME}/{REJECTED_DIRNAME} - there are no un-arranged RAW images left to rank."
            )
        raise ValueError(f"No RAW images found under {input_folder.resolve()}")

    resolved_device = resolve_device(device)
    output_csv = ranking_path(input_folder)

    crop_cache = None
    if crop_birds:
        if on_stage is not None:
            on_stage("Building bird-crop cache")
        params = CropParams(
            margin_frac=margin_frac, conf_threshold=conf_threshold, max_side=max_side,
            image_format=image_format, jpeg_quality=jpeg_quality,
        )
        image_paths = [item.image_path for item in dataset.items]
        build_cache(image_paths, crop_cache_dir, params, device=resolved_device, force=force_preprocess)
        crop_cache = crop_cache_dir

    if on_stage is not None:
        on_stage("Loading model")
    model_config = ModelConfig(backbone=backbone, pretrained=False, freeze_backbone=True)
    model = PreferenceHead(model_config).to(resolved_device)
    checkpoint_data = load_checkpoint(checkpoint_path, map_location=resolved_device)
    model.load_state_dict(checkpoint_data["model_state_dict"])

    loader = RawImageLoader(str(input_folder), resize_mode=resize_mode, crop_cache_dir=crop_cache)
    if on_stage is not None:
        on_stage("Scoring images")
    ranked = rank_dataset(model, dataset, loader, device=resolved_device, on_progress=on_progress)

    if on_stage is not None:
        on_stage("Writing ranking CSV")
    output_paths = write_results_csv(
        output_csv,
        dataset,
        ranked,
        select_root=str(input_folder),
        reject_root="(inference - no labels)",
        max_rows=max_rows,
    )

    write_run_metadata(
        input_folder,
        backbone=backbone,
        checkpoint=str(checkpoint_path.resolve()),
        image_count=len(dataset),
        crop_birds=bool(crop_birds),
    )
    record_run(
        input_folder,
        AI_STRATEGY_ID,
        considered=len(dataset),
        accepted=len(dataset),  # the AI model scores every image - no filter stage, see ranking/classic.py's
        reject_counts={},
        image_metrics={path: {"ai_score": score} for _name, score, _label, path in ranked},
        params={"backbone": backbone, "checkpoint": str(checkpoint_path.resolve()), "crop_birds": bool(crop_birds)},
        device=resolved_device,
        db_path=analytics_db,
    )

    return {
        "output_csv": output_paths[0],
        "extra_csv_files": output_paths[1:],
        "image_count": len(dataset),
        "device": resolved_device,
        "top": [(name, score) for name, score, _label, _path in ranked[:10]],
    }



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
        default=None,
        help="Write the ranked CSV here instead of into the folder's own "
        f"{SIDECAR_DIRNAME}/{RANKING_FILENAME}. The run's start date/time is appended to the "
        "stem (e.g. rankings_20260725-143000.csv). Passing this opts out of "
        "`picklikeme review --input <folder>` finding the ranking on its own.",
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
    parser.add_argument(
        "--max-side", type=int, default=CropParams.max_side,
        help="Max cached crop long side in pixels (default: unlimited - the crop's own original resolution)",
    )
    parser.add_argument(
        "--image-format", choices=sorted(IMAGE_FORMAT_EXTENSIONS), default=CropParams.image_format,
        help=f"Vision Cache file format (default: {CropParams.image_format})",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=CropParams.jpeg_quality,
        help=f"JPEG quality when --image-format=jpeg (default: {CropParams.jpeg_quality})",
    )
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild crops even if already cached")
    parser.add_argument(
        "--organize-output",
        type=_boolean,
        nargs="?",
        const=True,
        default=True,
        metavar="true/false",
        help=f"Move the ranked images into {SELECTED_DIRNAME}/ and {REJECTED_DIRNAME}/ under the "
        "input folder (default: true). Pass false to leave every file where it is, which is "
        "what `picklikeme review` expects - it files them once you have reviewed them.",
    )
    parser.add_argument(
        "--selection-percentage",
        type=float,
        default=DEFAULT_SELECTION_PERCENTAGE,
        help=f"Percentage of the highest-ranked images to place in {SELECTED_DIRNAME} "
        f"(default: {DEFAULT_SELECTION_PERCENTAGE:g}; 0-100).",
    )
    parser.add_argument(
        "--organize-dir",
        default=None,
        help=f"Where {SELECTED_DIRNAME}/ and {REJECTED_DIRNAME}/ are created (default: the --input folder)",
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
    # images a previous run had already filed and shuffle them again. The
    # sidecar holds no images, but is excluded explicitly rather than by luck.
    dataset = UnlabeledImageDataset.from_folder(
        input_folder, exclude_dirs=set(ORGANIZE_DIRNAMES) | {SIDECAR_DIRNAME}
    )
    if len(dataset) == 0:
        if _folder_already_organized(input_folder):
            raise SystemExit(
                f"{input_folder.resolve()} has already been organized into "
                f"{SELECTED_DIRNAME}/{REJECTED_DIRNAME} - there are no un-arranged RAW images left to rank."
            )
        raise SystemExit(f"No RAW images (.arw/.nef/.cr3) found under {input_folder.resolve()}")
    print(f"Found {len(dataset)} images to rank under {input_folder.resolve()}")
    profiler = PipelineProfiler(images=len(dataset), decode_workers=1, device=resolve_device(args.device))
    profiler.capture_environment(torch_module=None)

    # Resolved before the crop-cache build so the destination is known up front,
    # not only after a long preprocessing + scoring pass.
    #
    # By default the ranking is written into the folder it describes, so
    # `picklikeme review --input <folder>` finds it by computing one path
    # rather than searching. An explicit --output-csv overrides that entirely.
    if args.output_csv:
        output_csv = timestamped_output_path(args.output_csv, run_started)
    else:
        output_csv = ranking_path(input_folder)
    print(f"Ranked CSV for this run will be written to {output_csv.resolve()}")

    device = resolve_device(args.device)
    profiler.device = device

    crop_cache_dir = None
    if args.crop_birds:
        print("=" * 64)
        print("STEP 1/2: Building bird-crop cache for the new folder")
        print("=" * 64)
        params = CropParams(
            margin_frac=args.margin_frac,
            conf_threshold=args.conf_threshold,
            max_side=args.max_side,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )
        image_paths = [item.image_path for item in dataset.items]
        with profiler.stage("Preprocessing"):
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
    with profiler.stage("Ranking inference"):
        ranked = rank_dataset(model, dataset, loader, device=device)

    with profiler.stage("CSV generation"):
        output_paths = write_results_csv(
            output_csv,
            dataset,
            ranked,
            select_root=str(input_folder),
            reject_root="(inference - no labels)",
            max_rows=args.max_rows,
        )

    total_runtime = (datetime.now() - run_started).total_seconds()
    print(profiler.build_report(total_runtime=total_runtime))
    print("\nTop-ranked images:")
    for rank, entry in enumerate(ranked[:10], start=1):
        print(f"{rank}. {entry[0]}: {entry[1]:.4f}")
    print(f"\nRanked CSV written to {output_paths[0]}")
    if len(output_paths) > 1:
        print(f"Additional CSV files: {', '.join(str(path) for path in output_paths[1:])}")

    if args.output_csv:
        # The ranking is not where review looks, so say how to reach it.
        print(f"\nTo review this shoot:\n  {cli_prefix()} review --input \"{input_folder}\" "
              f"--ranking \"{output_paths[0]}\"")
    else:
        write_run_metadata(
            input_folder,
            backbone=args.backbone,
            checkpoint=str(checkpoint_path.resolve()),
            image_count=len(dataset),
            crop_birds=bool(args.crop_birds),
        )
        print(f"\nTo review this shoot:\n  {cli_prefix()} review --input \"{input_folder}\"")


if __name__ == "__main__":
    try:
        main()
    except CropCacheVersionMismatch as exc:
        raise SystemExit(str(exc)) from None
