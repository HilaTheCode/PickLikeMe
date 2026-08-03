"""Standalone developer script - debugs the EyePose eye-detection pipeline
for exactly ONE image.

NOT part of the PickLikeMe application. Never imported or called from it.
This is a throwaway diagnostic tool a developer runs by hand from the
command line, not a feature, not a reusable framework.

It reuses the real production functions end to end - the same RAW decode,
the same bird detector, the same crop math, the same EyePose-v0 detector -
so every intermediate image/value it saves is exactly what a real
Classic Vision (EyePose) ranking run would have produced for this image.
Nothing here reimplements any detection, cropping, or coordinate-transform
logic; it only calls the real functions and prints/saves what they returned.

Usage:
    python tools/debug_eye_pipeline.py "D:\\Photos\\032A2530.CR3"
    python tools/debug_eye_pipeline.py "D:\\Photos\\032A2530.CR3" --output-dir debug_out --device cuda

Writes, into --output-dir/<image stem>/:
    01_original.jpg          - the decoded full-resolution source image
    02_bird_detection.jpg    - original image with detected box(es) drawn on it
    03_bird_crop.jpg         - the crop the bird detector produced
    04_model_input.jpg       - the exact 640x640 letterboxed tensor EyePose saw
    05_raw_output.json       - the model's own output: per-keypoint (x, y, confidence)
                                in 640-model-space, before any coordinate mapping
    06_all_keypoints.jpg     - every decoded keypoint drawn on the crop, labelled
    07_final_overlay.jpg     - the selected eye, drawn exactly as PickLikeMe's
                                own overlay renders it (contactsheets._draw_eye_overlay)
    report.md                - the coordinate-transform trace, stage by stage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw, ImageFont

from picklikeme.analyzer.contactsheets import OTHER_BOX, SELECTED_BOX, _draw_eye_overlay
from picklikeme.bird_crop import BirdDetector, CropParams, build_crop
from picklikeme.eyes.eyepose_v0 import KPT_NAMES, EyePoseV0EyeDetector
from picklikeme.platform import resolve_torch_device
from picklikeme.ranking.debug import _projected_eye_box
from picklikeme.ranking.filters import FilterCandidate
from picklikeme.raw_io import RawImageLoader

# One distinct colour per keypoint, for stage 6 - purely cosmetic, has no
# effect on anything else.
KEYPOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "beak": (239, 68, 68),
    "left_eye": (59, 130, 246),
    "right_eye": (34, 197, 94),
    "head_top": (234, 179, 8),
    "left_shoulder": (168, 85, 247),
    "right_shoulder": (20, 184, 166),
}


def _font():
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001 - a missing default font must not break debugging
        return None


def _log(stage: str, *, source_file: str, function: str, input_desc: str, output_desc: str) -> None:
    print(f"\n[{stage}]")
    print(f"  source file : {source_file}")
    print(f"  function    : {function}")
    print(f"  input       : {input_desc}")
    print(f"  output      : {output_desc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_path", help="Path to a single source image (RAW or standard)")
    parser.add_argument("--output-dir", default="debug_eye_pipeline_out")
    parser.add_argument("--device", default=None, help="cpu/cuda (default: auto-detect)")
    args = parser.parse_args()

    source_path = Path(args.image_path).resolve()
    if not source_path.is_file():
        raise SystemExit(f"Image not found: {source_path}")

    out_dir = Path(args.output_dir) / source_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Image  : {source_path}")
    print(f"Output : {out_dir.resolve()}")

    device = resolve_torch_device(args.device)
    print(f"Device : {device}")

    # -----------------------------------------------------------------
    # Stage 1: original image - the exact RAW decode preprocess.build_cache
    # uses for every image (RawImageLoader._decode_full_frame), not the
    # embedded-preview shortcut analyzer.contactsheets.load_source_image
    # uses for reports - that would trace a different, lower-resolution
    # image than what Classic Vision actually scores.
    # -----------------------------------------------------------------
    decoder = RawImageLoader(raw_root=".", resize_mode="letterbox")
    original_rgb = decoder._decode_full_frame(str(source_path))
    _log(
        "Stage 1: Original image",
        source_file="src/picklikeme/raw_io.py",
        function="RawImageLoader._decode_full_frame",
        input_desc=f"source_path={source_path.name}",
        output_desc=f"RGB array, shape={original_rgb.shape}, dtype={original_rgb.dtype}",
    )
    Image.fromarray(original_rgb).save(out_dir / "01_original.jpg", "JPEG", quality=92)

    # -----------------------------------------------------------------
    # Stage 2 + 3: bird detection and bird crop - ONE call, because that is
    # how production does it too (preprocess.build_cache calls build_crop
    # once per image with collect_detections=True).
    # -----------------------------------------------------------------
    bird_detector = BirdDetector(device=device)
    crop_params = CropParams()
    result = build_crop(original_rgb, bird_detector, crop_params, collect_detections=True)
    _log(
        "Stage 2: Bird detection",
        source_file="src/picklikeme/bird_crop.py",
        function="build_crop (-> BirdDetector.detect_with_all -> select_best_detection)",
        input_desc=f"original image {original_rgb.shape}, CropParams()={crop_params}",
        output_desc=(
            f"selected={result.detection}, "
            f"{len(result.all_detections)} candidate(s) total, "
            f"expanded_box={result.expanded_box}, source_size={result.source_size}"
        ),
    )
    for i, det in enumerate(result.all_detections):
        marker = " <- selected" if det == result.detection else ""
        print(f"    candidate[{i}]: box={tuple(round(v, 1) for v in det.box)} score={det.score:.3f} label={det.label}{marker}")

    detection_overlay = Image.fromarray(original_rgb).copy()
    draw = ImageDraw.Draw(detection_overlay)
    line = max(3, detection_overlay.width // 250)
    for det in result.all_detections:
        colour = SELECTED_BOX if det == result.detection else OTHER_BOX
        draw.rectangle(list(det.box), outline=colour, width=line)
        draw.text((det.box[0] + 2, det.box[1] + 2), f"{det.label} {det.score:.2f}", fill=colour, font=_font())
    detection_overlay.save(out_dir / "02_bird_detection.jpg", "JPEG", quality=92)

    _log(
        "Stage 3: Bird crop",
        source_file="src/picklikeme/bird_crop.py",
        function="build_crop (crop_to_box + downscale_long_side, inside the same call above)",
        input_desc=f"expanded_box={result.expanded_box}, margin_frac={crop_params.margin_frac}, max_side={crop_params.max_side}",
        output_desc=f"crop shape={result.crop.shape}, dtype={result.crop.dtype}",
    )
    Image.fromarray(result.crop).save(out_dir / "03_bird_crop.jpg", "JPEG", quality=95)

    if result.detection is None or result.expanded_box is None:
        print("\nNo bird detected - nothing to feed EyePose. Stopping here.")
        return

    # -----------------------------------------------------------------
    # Stage 4: exact EyePose model input - EyePoseV0EyeDetector.debug_predict
    # is the real, existing production debug method (see eyepose_v0.py) -
    # one call gives every intermediate value of the real forward pass.
    # -----------------------------------------------------------------
    eye_detector = EyePoseV0EyeDetector(device=device)
    debug = eye_detector.debug_predict(result.crop)
    if debug is None:
        print("\nEyePose produced a degenerate (all-zero) output - nothing to decode. Stopping here.")
        return
    _log(
        "Stage 4: EyePose model input",
        source_file="src/picklikeme/eyes/eyepose_v0.py",
        function="EyePoseV0EyeDetector.debug_predict (-> _letterbox_forward -> _to_tensor -> onnxruntime session.run)",
        input_desc=f"crop shape={result.crop.shape}",
        output_desc=f"padded_input shape={debug['padded_input'].shape} (640x640x3, the exact tensor input before normalize/transpose)",
    )
    Image.fromarray(debug["padded_input"]).save(out_dir / "04_model_input.jpg", "JPEG", quality=95)

    # -----------------------------------------------------------------
    # Stage 5: raw model output - every keypoint's (x, y, confidence) in the
    # model's own 640x640 space, straight from debug_predict's own return
    # value, before any crop/frame coordinate mapping.
    # -----------------------------------------------------------------
    _log(
        "Stage 5: Raw ONNX output",
        source_file="src/picklikeme/eyes/eyepose_v0.py",
        function="EyePoseV0EyeDetector.debug_predict (landmarks_640, detection_confidence)",
        input_desc="(same forward pass as Stage 4 - not re-run)",
        output_desc=f"detection_confidence={debug['detection_confidence']:.4f}, {len(debug['landmarks_640'])} keypoints",
    )
    raw_payload = {
        "detection_confidence": debug["detection_confidence"],
        "landmarks_640_model_space": {
            name: {"x": x, "y": y, "confidence": vis}
            for name, (x, y, vis) in debug["landmarks_640"].items()
        },
    }
    (out_dir / "05_raw_output.json").write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    for index, name in enumerate(KPT_NAMES):
        x, y, vis = debug["landmarks_640"][name]
        print(f"    [{index}] {name:16s} x={x:8.2f} y={y:8.2f} confidence={vis:.4f}  (model-space, 640x640)")

    # -----------------------------------------------------------------
    # Stage 6: every decoded keypoint, drawn on the crop - landmarks_crop is
    # debug_predict's own crop-space projection (via the real
    # _letterbox_inverse), not recomputed here.
    # -----------------------------------------------------------------
    landmarks_crop = debug["landmarks_crop"]
    _log(
        "Stage 6: All decoded keypoints (crop space)",
        source_file="src/picklikeme/eyes/eyepose_v0.py",
        function="EyePoseV0EyeDetector.debug_predict (landmarks_crop, via _letterbox_inverse)",
        input_desc="landmarks_640 + (scale, pad_x, pad_y) from Stage 4's letterbox",
        output_desc=f"{len(landmarks_crop)} EyeKeypoint(x, y, confidence) values, in crop-pixel space",
    )
    crop_annotated = Image.fromarray(result.crop).copy()
    draw = ImageDraw.Draw(crop_annotated)
    radius = max(4, crop_annotated.width // 100)
    for index, name in enumerate(KPT_NAMES):
        kp = landmarks_crop[name]
        colour = KEYPOINT_COLORS.get(name, (255, 255, 255))
        draw.ellipse([kp.x - radius, kp.y - radius, kp.x + radius, kp.y + radius], outline=colour, width=3)
        draw.text((kp.x + radius + 2, kp.y - radius), f"[{index}] {name} {kp.confidence:.2f}", fill=colour, font=_font())
        print(f"    [{index}] {name:16s} x={kp.x:8.2f} y={kp.y:8.2f} confidence={kp.confidence:.4f}  (crop space)")
    crop_annotated.save(out_dir / "06_all_keypoints.jpg", "JPEG", quality=95)

    # -----------------------------------------------------------------
    # Stage 7: the selected eye and the final overlay - EyePoseV0EyeDetector
    # .detect() is the exact method ClassicVisionEyePoseStrategy.rank_folder
    # calls; _projected_eye_box and _draw_eye_overlay are the exact
    # production functions that map a crop-space eye onto the full frame
    # and draw PickLikeMe's own pink/magenta overlay.
    # -----------------------------------------------------------------
    detection = eye_detector.detect(result.crop)
    # detect() picks primary/other as (left, right) if left.confidence >=
    # right.confidence else (right, left), with confidence=primary.confidence
    # - so comparing confidences from THIS SAME call (not cross-comparing
    # against debug_predict's separate forward pass above) is the reliable
    # way to tell which channel became primary.
    primary_name = "left_eye" if detection.left is not None and detection.confidence == detection.left.confidence else "right_eye"
    _log(
        "Stage 7a: Eye selection",
        source_file="src/picklikeme/eyes/eyepose_v0.py",
        function="EyePoseV0EyeDetector.detect",
        input_desc=f"crop shape={result.crop.shape}",
        output_desc=(
            f"primary={primary_name}, confidence={detection.confidence:.4f}, "
            f"accepted={detection.accepted}, box(crop space)={tuple(round(v, 1) for v in detection.box)}, "
            f"head_confidence={detection.head_confidence}, head_visible={detection.head_visible}"
        ),
    )

    # crop_box (not subject_box) is what _projected_eye_box actually scales
    # against - see docs/EyePose_Investigation_Phase_1.md's Q1 finding, the
    # bug this same field rename fixed in production.
    candidate = FilterCandidate(image_path=str(source_path), subject_crop=result.crop, crop_box=result.expanded_box)
    box_frame = _projected_eye_box(candidate, detection.box)

    def project_point(x: float, y: float) -> tuple[float, float]:
        # _projected_eye_box projects a BOX; reused here for a single point
        # by giving it a degenerate zero-size box and reading one corner -
        # the exact same production scale/offset math, not a reimplementation.
        px1, py1, _, _ = _projected_eye_box(candidate, (x, y, x, y))
        return (px1, py1)

    left_frame = project_point(detection.left.x, detection.left.y) if detection.left else None
    right_frame = project_point(detection.right.x, detection.right.y) if detection.right else None
    _log(
        "Stage 7b: Project eye onto full frame",
        source_file="src/picklikeme/ranking/debug.py",
        function="_projected_eye_box",
        input_desc=f"eye box (crop space)={tuple(round(v, 1) for v in detection.box)}, expanded_box={result.expanded_box}",
        output_desc=f"eye box (full frame)={tuple(round(v, 1) for v in box_frame) if box_frame else None}",
    )

    eye_dict = {
        "accepted": detection.accepted,
        "confidence": detection.confidence,
        "box": box_frame,
        "left": {"x": left_frame[0], "y": left_frame[1], "confidence": detection.left.confidence} if left_frame and detection.left else None,
        "right": {"x": right_frame[0], "y": right_frame[1], "confidence": detection.right.confidence} if right_frame and detection.right else None,
    }
    final_overlay = detection_overlay.copy()
    draw = ImageDraw.Draw(final_overlay)
    _draw_eye_overlay(draw, eye_dict, scale=1.0, offset_x=0.0, offset_y=0.0, line=line)
    _log(
        "Stage 7c: Final overlay",
        source_file="src/picklikeme/analyzer/contactsheets.py",
        function="_draw_eye_overlay (the exact function the Gallery/Loupe overlay calls)",
        input_desc=f"eye={eye_dict}",
        output_desc="07_final_overlay.jpg - pink/solid if accepted, dark-magenta/dashed if rejected",
    )
    final_overlay.save(out_dir / "07_final_overlay.jpg", "JPEG", quality=92)

    # -----------------------------------------------------------------
    # Report: the coordinate-transform trace.
    # -----------------------------------------------------------------
    primary_640 = debug["landmarks_640"][primary_name]
    primary_crop = landmarks_crop[primary_name]
    primary_frame = project_point(primary_crop.x, primary_crop.y)

    report = f"""# EyePose pipeline trace - {source_path.name}

Every value below came from a real production function call - see the
console output above for the exact source file / function / input / output
of each stage. Nothing here was recomputed independently.

## Coordinate transformation trace (primary eye = {primary_name})

1. **Original image** ({original_rgb.shape[1]}x{original_rgb.shape[0]} px)
   - `RawImageLoader._decode_full_frame` (src/picklikeme/raw_io.py)

2. **Bird crop rectangle** (in original-image pixels)
   - box = {result.expanded_box}
   - `bird_crop.build_crop` -> `expand_and_clamp_box` + `crop_to_box` (src/picklikeme/bird_crop.py)
   - crop size = {result.crop.shape[1]}x{result.crop.shape[0]} px

3. **Model input** (640x640 letterboxed)
   - `eyepose_v0._letterbox_forward`, called inside `debug_predict`

4. **Raw/decoded keypoint** ({primary_name}, model space, 640x640)
   - x={primary_640[0]:.2f}, y={primary_640[1]:.2f}, confidence={primary_640[2]:.4f}
   - `eyepose_v0._decode_best`

5. **Keypoint in crop space** (`_letterbox_inverse`)
   - x={primary_crop.x:.2f}, y={primary_crop.y:.2f}
   - formula: (x - pad_x) / scale, (y - pad_y) / scale

6. **Keypoint projected onto the original full frame**
   - x={primary_frame[0]:.2f}, y={primary_frame[1]:.2f}
   - `ranking.debug._projected_eye_box`
   - formula: expanded_box_x1 + crop_x * (expanded_box_x2 - expanded_box_x1) / crop_width,
              expanded_box_y1 + crop_y * (expanded_box_y2 - expanded_box_y1) / crop_height
   - expanded_box = {result.expanded_box}, crop size = {result.crop.shape[1]}x{result.crop.shape[0]}

7. **Final overlay** (07_final_overlay.jpg)
   - eye box (full frame) = {tuple(round(v, 1) for v in box_frame) if box_frame else None}
   - accepted = {detection.accepted}, confidence = {detection.confidence:.4f}
   - `analyzer.contactsheets._draw_eye_overlay` - the exact function the
     Gallery/Loupe overlay calls (via `review.thumbnails.eye_keypoints_for`
     in the real app, using the cached equivalent of steps 2-6 above).

## All six decoded keypoints (model space -> crop space)

| index | name | model-space (x, y) | confidence | crop-space (x, y) |
|---|---|---|---|---|
"""
    for index, name in enumerate(KPT_NAMES):
        mx, my, mvis = debug["landmarks_640"][name]
        kp = landmarks_crop[name]
        report += f"| {index} | {name} | ({mx:.1f}, {my:.1f}) | {mvis:.4f} | ({kp.x:.1f}, {kp.y:.1f}) |\n"

    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\nWrote report.md and 7 image/data artifacts to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
