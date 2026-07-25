"""End-to-end pipeline: preprocess -> train -> rank, in one command.

Runs the bird-crop preprocessing over the select/reject folders, then trains
(or resumes) the preference model and ranks every image, writing the results
CSV. This is exactly `picklikeme.preprocess` followed by `picklikeme.train`,
chained in a single process so the crop cache is guaranteed to exist before
training reads it.

    python -m picklikeme.run --select-root "..." --reject-root "..."

All training flags are accepted (see `python -m picklikeme.train --help`), plus
a few preprocessing knobs below. Pass --no-crop-birds to train on full frames
(the preprocess step is then skipped automatically), or --skip-preprocess to
reuse an already-built crop cache without rebuilding it.
"""

from __future__ import annotations

import argparse

from .bird_crop import CropParams
from .preprocess import default_decode_workers, preprocess_folders
from .train import build_arg_parser, train_and_rank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess (bird crops) then train and rank, in one command",
        parents=[build_arg_parser(add_help=False)],
    )
    group = parser.add_argument_group("preprocessing (crop cache)")
    group.add_argument("--margin-frac", type=float, default=CropParams.margin_frac, help="Safety margin around the detected bird before cropping")
    group.add_argument("--conf-threshold", type=float, default=CropParams.conf_threshold, help="Detection confidence threshold")
    group.add_argument("--max-side", type=int, default=CropParams.max_side, help="Max cached crop long side (pixels)")
    group.add_argument("--force-preprocess", action="store_true", help="Rebuild crops even if already cached")
    group.add_argument("--skip-preprocess", action="store_true", help="Skip the preprocess step and reuse the existing crop cache")
    group.add_argument(
        "--decode-workers",
        type=int,
        default=None,
        help=f"Decoder threads feeding the GPU during preprocessing (default: min(8, cpu_count) = "
        f"{default_decode_workers()}). Detection stays sequential regardless.",
    )
    args = parser.parse_args()

    if not args.select_root or not args.reject_root:
        raise SystemExit("Both --select-root and --reject-root are required.")

    if args.crop_birds and not args.skip_preprocess:
        print("=" * 64)
        print("STEP 1/2: Building bird-crop cache (preprocessing)")
        print("=" * 64)
        params = CropParams(
            margin_frac=args.margin_frac,
            conf_threshold=args.conf_threshold,
            max_side=args.max_side,
        )
        preprocess_folders(
            args.select_root,
            args.reject_root,
            args.crop_cache_dir,
            params,
            device=args.device or "cuda",
            force=args.force_preprocess,
            decode_workers=args.decode_workers,
        )
    elif not args.crop_birds:
        print("Preprocess step skipped (--no-crop-birds: training on full frames).")
    else:
        print("Preprocess step skipped (--skip-preprocess: reusing existing crop cache).")

    print("=" * 64)
    print("STEP 2/2: Training and ranking")
    print("=" * 64)
    train_and_rank(args)


if __name__ == "__main__":
    main()
