"""Optional per-image debug images for a Classic Vision run - see
`ClassicVisionStrategy.rank_folder`'s `debug_dir` parameter.

Disabled by default (`debug_dir=None`) and never surfaced in the desktop
UI's generated parameter dialog - this is a development/troubleshooting
tool, not a photographer-facing feature, matching how `eyes.inspect_eyepose`
is a standalone CLI rather than something wired into a review session.

Draws ONLY from `FilterCandidate`/`eyes.detector.EyeDetection` - the shape
every Classic Vision backend already produces - so this needs no per-backend
special-casing and never has to change when a new backend is registered
(see `ranking.classic`'s module docstring). One combined image per
processed candidate, showing everything the reported task asked for: the
bird crop, the eye box, both eye keypoints, confidence values, and the eye
box's coordinates in both the crop's own space and projected back onto the
full original frame.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..analyzer.contactsheets import EYE_BOX_ACCEPTED, EYE_BOX_REJECTED
from .filters import FilterCandidate

PANEL_LINE_HEIGHT = 16
PANEL_BG = (24, 24, 24)
PANEL_TEXT = (230, 230, 230)


def _font():
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover - a missing default font must not break a debug run
        return None


def _projected_eye_box(candidate: FilterCandidate, eye_box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """The eye box, mapped from the subject crop's own pixel space onto the
    full original frame - the same scale-and-offset trick
    `review.thumbnails.eye_keypoints_for` already uses for the Gallery/Loupe
    overlay, reproduced here so the debug image shows the SAME "final
    projected coordinates" a photographer would see there."""
    if candidate.subject_box is None or candidate.subject_crop is None:
        return None
    crop_h, crop_w = candidate.subject_crop.shape[:2]
    if crop_w <= 0 or crop_h <= 0:
        return None
    sx1, sy1, sx2, sy2 = candidate.subject_box
    scale_x, scale_y = (sx2 - sx1) / crop_w, (sy2 - sy1) / crop_h
    x1, y1, x2, y2 = eye_box
    return (sx1 + x1 * scale_x, sy1 + y1 * scale_y, sx1 + x2 * scale_x, sy1 + y2 * scale_y)


def render_debug_image(candidate: FilterCandidate, strategy_id: str) -> Image.Image | None:
    """One combined debug image for a single processed candidate - the crop
    with the eye overlay drawn on it, plus a text panel underneath. None
    when there is no crop at all to show (the NO_SUBJECT case)."""
    if candidate.subject_crop is None:
        return None

    crop = Image.fromarray(candidate.subject_crop).copy()
    draw = ImageDraw.Draw(crop)
    font = _font()
    lines = [f"strategy: {strategy_id}", f"image: {Path(candidate.image_path).name}"]

    eye = candidate.eye
    if eye is None:
        lines.append("eye: not detected (filtered before the eye detector ran)")
    else:
        colour = EYE_BOX_ACCEPTED if eye.accepted else EYE_BOX_REJECTED
        x1, y1, x2, y2 = eye.box
        width = max(2, crop.width // 100)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
        for label, keypoint in (("L", eye.left), ("R", eye.right)):
            if keypoint is None:
                continue
            radius = max(3, crop.width // 120)
            draw.ellipse(
                [keypoint.x - radius, keypoint.y - radius, keypoint.x + radius, keypoint.y + radius],
                outline=colour, width=2,
            )
            draw.text((keypoint.x + radius + 2, keypoint.y), f"{label} {keypoint.confidence:.2f}", fill=colour, font=font)

        lines.append(f"detector: {eye.detector_id}")
        lines.append(f"eye confidence: {eye.confidence:.3f}   accepted: {eye.accepted}")
        lines.append(f"eye box (crop space): {tuple(round(v, 1) for v in eye.box)}")
        projected = _projected_eye_box(candidate, eye.box)
        if projected is not None:
            lines.append(f"eye box (full frame, projected): {tuple(round(v, 1) for v in projected)}")

    panel_height = PANEL_LINE_HEIGHT * len(lines) + 10
    canvas = Image.new("RGB", (crop.width, crop.height + panel_height), PANEL_BG)
    canvas.paste(crop, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((6, crop.height + 4 + index * PANEL_LINE_HEIGHT), line, fill=PANEL_TEXT, font=font)
    return canvas


def debug_image_path(debug_dir: str | Path, image_path: str) -> Path:
    return Path(debug_dir) / f"{Path(image_path).stem}_debug.jpg"


def save_debug_image(candidate: FilterCandidate, strategy_id: str, debug_dir: str | Path) -> Path | None:
    """Render and save one candidate's debug image, or do nothing (return
    None) if there was no crop to show. Failure to write is logged, not
    raised - a debug aid must never be able to abort a real ranking run."""
    image = render_debug_image(candidate, strategy_id)
    if image is None:
        return None
    target = debug_image_path(debug_dir, candidate.image_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "JPEG", quality=90)
    except OSError:
        import logging

        logging.getLogger(__name__).warning("Could not write debug image for %s", candidate.image_path)
        return None
    return target
