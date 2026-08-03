"""Standalone developer script - debugs the BioCLIP species-classification
pipeline for one or more images.

NOT part of the PickLikeMe application. Never imported or called from it.
This is a throwaway diagnostic tool a developer runs by hand from the
command line to investigate WHY a species prediction is wrong, not a
feature. Modeled directly on debug_eye_pipeline.py - same philosophy: reuse
the real production functions end to end, print/save exactly what they
returned, reimplement nothing.

Two things this tool deliberately does that production `arrange_by_species`
does NOT do, both clearly labeled in every artifact they produce:

1. It also runs the bird-crop subject detector (`bird_crop.BirdDetector`) and
   feeds the resulting crop through the SAME classifier for comparison. This
   is investigation-only - `species/arrange.py` never touches bird_crop, see
   Q1 in the investigation report - and exists specifically to answer
   "would the crop already used for EyePose/ranking help species
   classification too?" with real, measured data instead of a guess.
2. It reconstructs the full Top-N similarity/probability vector by calling
   the classifier's own real `_preprocess`/`_model`/`_text_features`
   directly (the exact same tensors `BioClipSpeciesClassifier.classify()`
   itself uses - copy-verified against its source line by line), since
   `classify()` only returns the single winning answer. No model logic is
   reimplemented; this is the same one-line softmax `classify()` already
   does, just kept instead of discarded.

Usage:
    python tools/debug_species_pipeline.py "D:\\Photos\\032A2530.CR3" [more paths...]
    python tools/debug_species_pipeline.py --output-dir debug_out --device cuda "D:\\Photos"

A directory argument is expanded to every image directly inside it (not
recursive) - convenient for pointing at a folder of investigation samples.

Writes, into --output-dir/<image stem>/:
    01_original.jpg              - analyzer.contactsheets.load_source_image's
                                    own output - the EXACT image production
                                    species classification receives.
    02_subject_detection.jpg     - investigation-only: bird_crop's detector,
                                    every candidate box drawn, selected one
                                    highlighted.
    03_selected_crop.jpg         - investigation-only: the crop build_crop
                                    would produce for this image.
    04_classifier_input.jpg      - the real 224x224 tensor BioCLIP actually
                                    saw, reconstructed from the real
                                    preprocess pipeline (production path -
                                    full frame in).
    04b_classifier_input_cropped.jpg - investigation-only: the same real
                                    preprocess pipeline applied to the
                                    Stage-3 crop instead, for comparison.
    05_top_predictions.csv       - Top-20 species by probability, production
                                    (full-frame) input.
    05b_top_predictions_crop.csv - investigation-only: Top-20 for the
                                    cropped input.
    06_embedding_analysis.md     - margin analysis (top1 vs top2 vs top3),
                                    full-frame vs crop.
    07_crop_analysis.md          - bird-size/crop-occupancy/centering
                                    analysis.
    08_final_decision.md         - summary tying every stage together.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from picklikeme.analyzer.contactsheets import OTHER_BOX, SELECTED_BOX, load_source_image
from picklikeme.bird_crop import BirdDetector, CropParams, build_crop, box_area
from picklikeme.platform import resolve_torch_device
from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier
from picklikeme.species.classifier import UNKNOWN_SPECIES

IMAGE_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TOP_N = 20


def _font():
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001 - a missing default font must not break debugging
        return None


def _log(stage: str, *, source_file: str, function: str, input_desc: str, output_desc: str) -> None:
    print(f"\n  [{stage}]")
    print(f"    source file : {source_file}")
    print(f"    function    : {function}")
    print(f"    input       : {input_desc}")
    print(f"    output      : {output_desc}")


def _expand_inputs(raw_args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for arg in raw_args:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.is_file():
            paths.append(p)
        else:
            raise SystemExit(f"Not found: {p}")
    return paths


def _normalize_stats(classifier: BioClipSpeciesClassifier) -> tuple[list[float], list[float]]:
    """The real Normalize(mean, std) this classifier's own `_preprocess`
    uses - read from the live transform, never hardcoded, so this tool
    stays correct even if BioCLIP's preprocessing ever changes."""
    from torchvision.transforms import Normalize

    for t in classifier._preprocess.transforms:  # noqa: SLF001 - investigation tool, reads real production state
        if isinstance(t, Normalize):
            return list(t.mean), list(t.std)
    raise RuntimeError("No Normalize transform found in the classifier's own preprocess pipeline")


def _tensor_to_image(tensor, mean: list[float], std: list[float]) -> Image.Image:
    """Undo Normalize (the last preprocessing step) so the actual 224x224
    tensor the model saw can be looked at, not just trusted. CenterCrop/
    Resize already happened before Normalize, so this is a faithful,
    lossless-apart-from-the-tensor-roundtrip view of the real model input."""
    array = tensor.detach().cpu().numpy()[0]  # (3, H, W)
    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    array = array * std_arr + mean_arr
    array = np.clip(array, 0.0, 1.0) * 255.0
    array = array.transpose(1, 2, 0).astype(np.uint8)
    return Image.fromarray(array)


def _classify_full(classifier: BioClipSpeciesClassifier, image: Image.Image):
    """Reconstructs the full per-species similarity/probability vector.

    Every tensor here (`_preprocess`, `_model`, `_text_features`) is the
    real classifier's own - only the final matmul+softmax is written here,
    and it is the exact single line `classify()` itself computes (verified
    against species/bioclip_classifier.py's own source), just kept instead
    of collapsed immediately to argmax.
    """
    torch = classifier._torch  # noqa: SLF001
    with torch.no_grad():
        pixels = classifier._preprocess(image).unsqueeze(0).to(classifier.device)  # noqa: SLF001
        image_features = classifier._model.encode_image(pixels)  # noqa: SLF001
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ classifier._text_features.T  # noqa: SLF001
        probs = logits.softmax(dim=-1)[0]
    order = torch.argsort(probs, descending=True)
    ranked = [
        (classifier.species_list[int(i)], float(logits[0, int(i)].item()), float(probs[int(i)].item()))
        for i in order
    ]
    return ranked, pixels


def _write_predictions_csv(path: Path, ranked: list[tuple[str, float, float]]) -> None:
    top1_prob = ranked[0][2] if ranked else 0.0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "species", "similarity_logit", "probability", "diff_from_top1"])
        for rank, (species, logit, prob) in enumerate(ranked[:TOP_N], start=1):
            writer.writerow([rank, species, f"{logit:.4f}", f"{prob:.6f}", f"{top1_prob - prob:.6f}"])


def _draw_predictions(draw: "ImageDraw.ImageDraw", ranked, x: int, y: int, title: str) -> None:
    font = _font()
    draw.text((x, y), title, fill=(255, 255, 255), font=font)
    for i, (species, _logit, prob) in enumerate(ranked[:5], start=1):
        draw.text((x, y + 14 * i), f"{i}. {species} ({prob:.1%})", fill=(255, 255, 255), font=font)


def _process_one(
    source_path: Path,
    out_root: Path,
    detector: BirdDetector,
    classifier: BioClipSpeciesClassifier,
    mean: list[float],
    std: list[float],
) -> dict:
    out_dir = out_root / source_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {source_path.name} -> {out_dir} ===")

    # -----------------------------------------------------------------
    # Stage 1: the EXACT image production species classification receives -
    # analyzer.contactsheets.load_source_image, called by species/cache.py's
    # get_or_classify. Never the cached bird crop - see that function's own
    # docstring, quoted verbatim in the investigation report.
    # -----------------------------------------------------------------
    original = load_source_image(str(source_path))
    _log(
        "Stage 1: Original image (production's own classification input)",
        source_file="src/picklikeme/analyzer/contactsheets.py",
        function="load_source_image",
        input_desc=f"source_path={source_path.name}",
        output_desc=f"PIL Image, size={original.size}, mode={original.mode}",
    )
    original.save(out_dir / "01_original.jpg", "JPEG", quality=92)

    # -----------------------------------------------------------------
    # Stage 2+3: subject detection and crop - INVESTIGATION ONLY.
    # species/arrange.py never calls any of this in production (see Q1 in
    # the report). Run purely to compare against Stage 1's full-frame input.
    # -----------------------------------------------------------------
    from picklikeme.raw_io import RawImageLoader

    decoder = RawImageLoader(raw_root=".", resize_mode="letterbox")
    full_res = decoder._decode_full_frame(str(source_path))
    crop_params = CropParams()
    result = build_crop(full_res, detector, crop_params, collect_detections=True)
    _log(
        "Stage 2: Subject detection (INVESTIGATION ONLY - not run by production species arrange)",
        source_file="src/picklikeme/bird_crop.py",
        function="build_crop (-> BirdDetector.detect_with_all -> select_best_detection)",
        input_desc=f"full-resolution decode {full_res.shape}",
        output_desc=(
            f"selected={result.detection}, {len(result.all_detections)} candidate(s) total"
        ),
    )
    detection_overlay = Image.fromarray(full_res).copy()
    draw = ImageDraw.Draw(detection_overlay)
    line = max(3, detection_overlay.width // 250)
    for det in result.all_detections:
        colour = SELECTED_BOX if det == result.detection else OTHER_BOX
        draw.rectangle(list(det.box), outline=colour, width=line)
        draw.text((det.box[0] + 2, det.box[1] + 2), f"{det.label} {det.score:.2f}", fill=colour, font=_font())
    detection_overlay.save(out_dir / "02_subject_detection.jpg", "JPEG", quality=92)

    crop_image = Image.fromarray(result.crop)
    crop_image.save(out_dir / "03_selected_crop.jpg", "JPEG", quality=95)
    _log(
        "Stage 3: Selected crop (INVESTIGATION ONLY)",
        source_file="src/picklikeme/bird_crop.py",
        function="build_crop",
        input_desc=f"expanded_box={result.expanded_box}",
        output_desc=f"crop shape={result.crop.shape}",
    )

    bird_area_frac = None
    if result.detection is not None and result.expanded_box is not None:
        crop_w = result.expanded_box[2] - result.expanded_box[0]
        crop_h = result.expanded_box[3] - result.expanded_box[1]
        bird_area_frac = box_area(result.detection.box) / max(1.0, crop_w * crop_h)

    # -----------------------------------------------------------------
    # Stage 4: classifier input, real preprocessing, real production path
    # (full frame -> classifier). Plus 4b: the same real preprocessing
    # applied to the crop instead (investigation only).
    # -----------------------------------------------------------------
    ranked_full, pixels_full = _classify_full(classifier, original)
    tensor_image_full = _tensor_to_image(pixels_full, mean, std)
    annotated_full = tensor_image_full.resize((448, 448), Image.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(annotated_full)
    _draw_predictions(draw, ranked_full, 4, 4, "Full-frame prediction:")
    annotated_full.save(out_dir / "04_classifier_input.jpg", "JPEG", quality=95)
    _log(
        "Stage 4: Classifier input (REAL production path - full frame)",
        source_file="src/picklikeme/species/bioclip_classifier.py",
        function="BioClipSpeciesClassifier._preprocess (real transform) + classify()'s own encode_image/similarity math",
        input_desc="Stage 1's original image",
        output_desc=f"top1={ranked_full[0][0]!r} p={ranked_full[0][2]:.3f}",
    )
    _write_predictions_csv(out_dir / "05_top_predictions.csv", ranked_full)

    ranked_crop, pixels_crop = _classify_full(classifier, crop_image)
    tensor_image_crop = _tensor_to_image(pixels_crop, mean, std)
    annotated_crop = tensor_image_crop.resize((448, 448), Image.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(annotated_crop)
    _draw_predictions(draw, ranked_crop, 4, 4, "Crop prediction:")
    annotated_crop.save(out_dir / "04b_classifier_input_cropped.jpg", "JPEG", quality=95)
    _write_predictions_csv(out_dir / "05b_top_predictions_crop.csv", ranked_crop)
    _log(
        "Stage 4b: Classifier input (INVESTIGATION ONLY - cropped)",
        source_file="src/picklikeme/species/bioclip_classifier.py",
        function="same real _preprocess/_model/_text_features, applied to Stage 3's crop",
        input_desc="Stage 3's crop",
        output_desc=f"top1={ranked_crop[0][0]!r} p={ranked_crop[0][2]:.3f}",
    )

    # -----------------------------------------------------------------
    # Stage 6: embedding margin analysis.
    # -----------------------------------------------------------------
    def _margin_block(ranked, label):
        lines = [f"### {label}", ""]
        for i, (species, logit, prob) in enumerate(ranked[:5], start=1):
            lines.append(f"{i}. **{species}** - similarity {logit:.2f}, probability {prob:.1%}")
        if len(ranked) >= 2:
            gap = ranked[0][2] - ranked[1][2]
            verdict = "clear winner" if gap > 0.15 else ("close race" if gap > 0.03 else "near-tie / ambiguous")
            lines.append("")
            lines.append(f"Top1 vs Top2 probability gap: **{gap:.1%}** ({verdict})")
        return "\n".join(lines)

    below_threshold = ranked_full[0][2] < classifier.min_confidence
    predicted_species = UNKNOWN_SPECIES if below_threshold else ranked_full[0][0]

    embedding_md = f"""# Embedding analysis - {source_path.name}

{_margin_block(ranked_full, "Full-frame input (production)")}

{_margin_block(ranked_crop, "Cropped input (investigation only)")}

## Does cropping change the answer?

- Full-frame top-1: **{ranked_full[0][0]}** ({ranked_full[0][2]:.1%})
- Cropped top-1: **{ranked_crop[0][0]}** ({ranked_crop[0][2]:.1%})
- {"Same species, cropping changed confidence only." if ranked_full[0][0] == ranked_crop[0][0] else "**Different species predicted** - the crop changes the answer, not just the confidence."}

## Production accept/reject

`min_confidence` = {classifier.min_confidence:.2f}. Full-frame top-1 probability {ranked_full[0][2]:.1%} is
{"BELOW" if below_threshold else "at or above"} the threshold, so production's real predicted species for this
image is: **{predicted_species}**.
"""
    (out_dir / "06_embedding_analysis.md").write_text(embedding_md, encoding="utf-8")

    # -----------------------------------------------------------------
    # Stage 7: crop analysis.
    # -----------------------------------------------------------------
    crop_lines = [f"# Crop analysis - {source_path.name}", ""]
    if result.detection is None:
        crop_lines.append("No subject detected at all by `BirdDetector` - the full frame is all any crop-based approach would have to work with.")
    else:
        crop_lines.append(f"- Crop dimensions: {result.crop.shape[1]}x{result.crop.shape[0]} px")
        crop_lines.append(f"- Original frame dimensions: {full_res.shape[1]}x{full_res.shape[0]} px")
        if bird_area_frac is not None:
            crop_lines.append(f"- Bird (detection box) occupies **{bird_area_frac:.1%}** of the crop's area")
        box = result.detection.box
        box_cx = (box[0] + box[2]) / 2
        box_cy = (box[1] + box[3]) / 2
        frame_cx, frame_cy = full_res.shape[1] / 2, full_res.shape[0] / 2
        offset_frac_x = abs(box_cx - frame_cx) / full_res.shape[1]
        offset_frac_y = abs(box_cy - frame_cy) / full_res.shape[0]
        crop_lines.append(
            f"- Detection center offset from full-frame center: {offset_frac_x:.1%} horizontally, {offset_frac_y:.1%} vertically"
        )
        full_frame_area_frac = box_area(box) / (full_res.shape[0] * full_res.shape[1])
        crop_lines.append(f"- Bird occupies only **{full_frame_area_frac:.2%}** of the FULL FRAME (what production actually classifies)")
        crop_lines.append("")
        crop_lines.append(
            "A tighter crop would likely help classification when the full-frame bird-area percentage above "
            "is small (BioCLIP's own preprocessing resizes to 224px then center-crops - a small, off-center "
            "subject loses most of its pixel budget to background before the model ever sees it)."
        )
    (out_dir / "07_crop_analysis.md").write_text("\n".join(crop_lines), encoding="utf-8")

    # -----------------------------------------------------------------
    # Stage 8: final decision summary.
    # -----------------------------------------------------------------
    final_md = f"""# Final decision - {source_path.name}

- Selected detection: {result.detection}
- Production species prediction (full frame): **{predicted_species}** (top-1 probability {ranked_full[0][2]:.1%})
- Crop-based prediction (investigation only): **{ranked_crop[0][0]}** (top-1 probability {ranked_crop[0][2]:.1%})
- Full-frame bird-area fraction: {f"{box_area(result.detection.box) / (full_res.shape[0]*full_res.shape[1]):.2%}" if result.detection else "N/A (no detection)"}
"""
    (out_dir / "08_final_decision.md").write_text(final_md, encoding="utf-8")

    return {
        "image": source_path.name,
        "predicted_species": predicted_species,
        "full_frame_top1": ranked_full[0][0],
        "full_frame_prob": ranked_full[0][2],
        "crop_top1": ranked_crop[0][0],
        "crop_prob": ranked_crop[0][2],
        "bird_area_frac_of_frame": (
            box_area(result.detection.box) / (full_res.shape[0] * full_res.shape[1])
            if result.detection is not None else None
        ),
        "detected": result.detection is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="One or more image paths, or a directory to expand (non-recursive)")
    parser.add_argument("--output-dir", default="debug_species_pipeline_out")
    parser.add_argument("--device", default=None, help="cpu/cuda (default: auto-detect)")
    args = parser.parse_args()

    images = _expand_inputs(args.paths)
    if not images:
        raise SystemExit("No images found.")

    device = resolve_torch_device(args.device)
    print(f"Device : {device}")
    print(f"Images : {len(images)}")

    detector = BirdDetector(device=device)
    classifier = BioClipSpeciesClassifier(device=device)
    mean, std = _normalize_stats(classifier)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for source_path in images:
        try:
            summary_rows.append(_process_one(source_path, out_root, detector, classifier, mean, std))
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop a 20-image batch
            print(f"  ERROR on {source_path.name}: {type(exc).__name__}: {exc}")

    summary_path = out_root / "00_batch_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image", "detected", "predicted_species", "full_frame_top1", "full_frame_prob",
                "crop_top1", "crop_prob", "bird_area_frac_of_frame",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\nWrote per-image debug bundles and {summary_path} for {len(summary_rows)}/{len(images)} images.")


if __name__ == "__main__":
    main()
