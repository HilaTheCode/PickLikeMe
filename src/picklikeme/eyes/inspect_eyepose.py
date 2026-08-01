"""Visual QA for EyePose-v0's coordinate transform (read-only inspection
tool) - mirrors `picklikeme.inspect_crops`'s own shape and conventions for
the subject detector, applied to the eye detector instead.

For every image, saves the six numbered artifacts needed to verify the
complete transformation by eye, not just by trusting the arithmetic:

    1. original image, with the detected bird box drawn on it
    2. the bird crop (exactly what preprocessing cached)
    3. the image actually fed into the eye model (the 640x640 letterboxed
       tensor input, saved as a viewable image)
    4. the raw model output (every landmark's model-space coordinates and
       confidence, as text)
    5. the eye overlay drawn on the crop
    6. the eye overlay projected back onto the original image

Read-only: never writes to the crop cache, the detections DB, or the eye
cache - only reads what preprocessing already built (falling back to
running the subject detector live, exactly like `inspect_crops.py`'s own
folder mode, if a folder has never been preprocessed at all).

    python -m picklikeme.eyes.inspect_eyepose --input-folder "..."
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..analyzer.contactsheets import EYE_BOX_ACCEPTED, EYE_BOX_REJECTED, SELECTED_BOX, load_source_image
from ..bird_crop import BirdDetector, CropParams, build_crop
from ..config import DEFAULT_INSPECTION_DIR
from .detector import derive_eye_box
from .eyepose_v0 import DEFAULT_EYE_BOX_FRAC, MIN_EYE_BOX_PX, EyePoseV0EyeDetector


@dataclass
class InspectionResult:
    source_path: str
    original_rgb: np.ndarray  # full-resolution, uint8
    crop_rgb: np.ndarray
    expanded_box: tuple[int, int, int, int]  # the crop's own region in ORIGINAL-frame pixels
    debug: dict  # EyePoseV0EyeDetector.debug_predict()'s own return value


def run_one(detector: EyePoseV0EyeDetector, bird_detector: BirdDetector, source_path: str, params: CropParams) -> InspectionResult | None:
    """Decode -> detect -> crop -> eye-detect, reusing bird_crop.build_crop
    directly (no reimplemented detection/crop logic) so the crop this tool
    inspects is byte-identical to what Classic Vision actually scores.
    Returns None if no bird was found (nothing to project an eye onto)."""
    original = np.asarray(load_source_image(source_path).convert("RGB"))
    result = build_crop(original, bird_detector, params)
    if result.detection is None or result.expanded_box is None:
        return None

    debug = detector.debug_predict(result.crop)
    if debug is None:
        return None

    return InspectionResult(
        source_path=source_path,
        original_rgb=original,
        crop_rgb=result.crop,
        expanded_box=tuple(int(round(v)) for v in result.expanded_box),
        debug=debug,
    )


def _font():
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001 - a missing default font must not break inspection
        return None


def save_artifacts(result: InspectionResult, min_confidence: float, output_dir: Path) -> dict[str, Path]:
    """Write the six numbered artifacts for one image; returns their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source_path).stem
    font = _font()
    written: dict[str, Path] = {}

    # 1. Original image, with the detected bird box.
    ex1, ey1, ex2, ey2 = result.expanded_box
    original_annotated = Image.fromarray(result.original_rgb).copy()
    draw = ImageDraw.Draw(original_annotated)
    draw.rectangle([ex1, ey1, ex2, ey2], outline=SELECTED_BOX, width=max(3, original_annotated.width // 250))
    path = output_dir / f"{stem}_1_original.jpg"
    original_annotated.save(path, "JPEG", quality=92)
    written["original"] = path

    # 2. The bird crop, unmodified - exactly what preprocessing cached.
    path = output_dir / f"{stem}_2_crop.jpg"
    Image.fromarray(result.crop_rgb).save(path, "JPEG", quality=95)
    written["crop"] = path

    # 3. The exact tensor input (640x640 letterboxed), as a viewable image.
    path = output_dir / f"{stem}_3_model_input.jpg"
    Image.fromarray(result.debug["padded_input"]).save(path, "JPEG", quality=95)
    written["model_input"] = path

    # 4. Raw model output, as text - every landmark's 640-space coordinate
    # and confidence, plus the winning anchor's own detection confidence.
    lines = [f"detection_confidence: {result.debug['detection_confidence']:.4f}", ""]
    for name, (x, y, vis) in result.debug["landmarks_640"].items():
        lines.append(f"{name:16s} x={x:8.2f} y={y:8.2f} confidence={vis:.4f}")
    path = output_dir / f"{stem}_4_raw_output.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    written["raw_output"] = path

    landmarks_crop = result.debug["landmarks_crop"]
    left, right = landmarks_crop["left_eye"], landmarks_crop["right_eye"]
    primary = left if left.confidence >= right.confidence else right
    crop_h, crop_w = result.crop_rgb.shape[:2]
    box_crop = derive_eye_box(primary.x, primary.y, crop_w, crop_h, DEFAULT_EYE_BOX_FRAC, MIN_EYE_BOX_PX)
    accepted = primary.confidence >= min_confidence
    colour = EYE_BOX_ACCEPTED if accepted else EYE_BOX_REJECTED

    # 5. Eye overlay drawn on the crop - box plus every one of the six
    # landmarks (not just the two the scoring algorithm uses), labelled.
    crop_annotated = Image.fromarray(result.crop_rgb).copy()
    draw = ImageDraw.Draw(crop_annotated)
    draw.rectangle(list(box_crop), outline=colour, width=max(2, crop_annotated.width // 100))
    for name, kp in landmarks_crop.items():
        radius = max(3, crop_annotated.width // 100)
        draw.ellipse([kp.x - radius, kp.y - radius, kp.x + radius, kp.y + radius], outline=colour, width=2)
        draw.text((kp.x + radius + 2, kp.y - radius), f"{name} {kp.confidence:.2f}", fill=colour, font=font)
    path = output_dir / f"{stem}_5_eye_on_crop.jpg"
    crop_annotated.save(path, "JPEG", quality=95)
    written["eye_on_crop"] = path

    # 6. The same overlay, projected back onto the original image - the
    # step that actually exercises the crop<->frame coordinate mapping
    # (review.thumbnails.eye_keypoints_for's own scale-and-offset trick,
    # reproduced here against this tool's own freshly-computed expanded_box
    # rather than a cached detection record, since this tool is read-only).
    scale_x = (ex2 - ex1) / crop_w
    scale_y = (ey2 - ey1) / crop_h

    def to_frame(x: float, y: float) -> tuple[float, float]:
        return (ex1 + x * scale_x, ey1 + y * scale_y)

    box_frame = (*to_frame(box_crop[0], box_crop[1]), *to_frame(box_crop[2], box_crop[3]))
    original_with_eye = original_annotated.copy()
    draw = ImageDraw.Draw(original_with_eye)
    line_w = max(3, original_with_eye.width // 250)
    draw.rectangle(list(box_frame), outline=colour, width=line_w)
    for name, kp in landmarks_crop.items():
        fx, fy = to_frame(kp.x, kp.y)
        radius = max(4, original_with_eye.width // 200)
        draw.ellipse([fx - radius, fy - radius, fx + radius, fy + radius], outline=colour, width=line_w // 2 or 1)
    path = output_dir / f"{stem}_6_eye_on_original.jpg"
    original_with_eye.save(path, "JPEG", quality=92)
    written["eye_on_original"] = path

    return written


def inspect_folder(args) -> None:
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    from ..inspect_crops import SUPPORTED_INPUT_EXTS, resolve_device

    image_paths = sorted(
        str(p) for p in input_folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTS
    )
    if not image_paths:
        raise SystemExit(f"No supported images found in {input_folder.resolve()}")
    print(f"Found {len(image_paths)} images in {input_folder.resolve()}")

    run_dir = Path(args.output_dir) / f"eyepose_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Loading the subject detector and EyePose-v0 on {device}...")
    bird_detector = BirdDetector(device=device)
    eye_detector = EyePoseV0EyeDetector(device=device, min_confidence=args.min_confidence)
    params = CropParams()

    processed, skipped = 0, 0
    for path in image_paths:
        try:
            result = run_one(eye_detector, bird_detector, path, params)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            print(f"  {Path(path).name}: ERROR {type(exc).__name__}: {exc}")
            skipped += 1
            continue
        if result is None:
            print(f"  {Path(path).name}: no bird detected - skipped")
            skipped += 1
            continue
        artifacts = save_artifacts(result, args.min_confidence, run_dir)
        processed += 1
        primary_conf = max(
            result.debug["landmarks_crop"]["left_eye"].confidence,
            result.debug["landmarks_crop"]["right_eye"].confidence,
        )
        print(f"  {Path(path).name}: eye confidence={primary_conf:.3f} -> {artifacts['eye_on_original'].name}")

    print(f"\n{processed} image(s) processed, {skipped} skipped. Artifacts written to {run_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual QA for EyePose-v0's coordinate transform")
    parser.add_argument("--input-folder", required=True, help="Folder of real images to validate against")
    parser.add_argument("--output-dir", default=str(DEFAULT_INSPECTION_DIR))
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Accept/reject threshold shown in the overlay colour")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    args = parser.parse_args()
    inspect_folder(args)


if __name__ == "__main__":
    main()
