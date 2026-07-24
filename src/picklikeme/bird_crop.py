"""Bird detection and crop caching.

This is the "bird-centered input" preprocessing phase: instead of feeding the
full frame to the model, we detect the bird once per image, crop tightly
around it (small safety margin, aspect ratio preserved), and cache the crop so
training never re-detects or re-decodes RAW.

Design notes:
- Detection uses torchvision's COCO-pretrained Faster R-CNN v2, where "bird"
  is class 16. It runs once, in a single process (see picklikeme.preprocess) —
  never inside DataLoader workers, where N workers would each load a detector
  onto the GPU and contend for memory.
- The crop is a true sub-rectangle of the source, so the bird's geometry is
  never distorted; fixed-size model input is produced later by letterbox
  padding in RawImageLoader, not by stretching.
- Cache entries are keyed by the absolute source path only, so one cache is
  reusable across model input sizes (384/512/640): detect once, letterbox to
  any size at load time.
- If no bird is detected, the full frame is cached as the fallback, so every
  image still yields an input and is never re-detected.

Bump CROP_CACHE_VERSION whenever the crop algorithm or its defaults change so a
stale cache is detected instead of silently reused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# COCO category index for "bird" in torchvision's detection weights metadata.
COCO_BIRD_CLASS = 16

CROP_CACHE_VERSION = "v1"
CROP_PARAMS_FILENAME = "crop_params.json"


@dataclass(frozen=True)
class CropParams:
    """Parameters that define how a crop cache was built. Recorded alongside
    the cache so a mismatched configuration can be detected."""

    margin_frac: float = 0.05          # small safety margin around the tight box
    conf_threshold: float = 0.30       # min detection confidence to accept a bird
    max_side: int = 1024               # cap the cached crop's long side (px)
    detector: str = "fasterrcnn_resnet50_fpn_v2"
    version: str = CROP_CACHE_VERSION


# ---------------------------------------------------------------------------
# Bounding-box geometry (pure functions, no torch import needed)
# ---------------------------------------------------------------------------

def expand_and_clamp_box(
    box: tuple[float, float, float, float],
    margin_frac: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Grow a box by margin_frac of its own size on each side, clamped to the
    image. A small margin absorbs detector inaccuracy without pulling in large
    background regions."""
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    mx = box_w * margin_frac
    my = box_h * margin_frac
    x1 = max(0.0, x1 - mx)
    y1 = max(0.0, y1 - my)
    x2 = min(float(width), x2 + mx)
    y2 = min(float(height), y2 + my)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def crop_to_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, image.shape[1]))
    y2 = max(y1 + 1, min(y2, image.shape[0]))
    return image[y1:y2, x1:x2]


def downscale_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale (never upscale) so the longer side is at most max_side,
    preserving aspect ratio. Keeps cached crops small without distortion."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Cache path scheme (shared by the preprocessor and the loader)
# ---------------------------------------------------------------------------

def crop_cache_path(cache_dir: str | Path, source_path: str | Path) -> Path:
    """Deterministic cache file for a source image, keyed by its absolute path.
    Independent of crop parameters (those are recorded in crop_params.json);
    rebuilding the cache overwrites in place."""
    resolved = str(Path(source_path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:20]
    return Path(cache_dir) / f"{digest}.png"


def write_crop_params(cache_dir: str | Path, params: CropParams) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CROP_PARAMS_FILENAME
    path.write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    return path


def read_crop_params(cache_dir: str | Path) -> CropParams | None:
    path = Path(cache_dir) / CROP_PARAMS_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return CropParams(**data)


def save_crop_png(cache_path: Path, crop_rgb: np.ndarray) -> None:
    """Write an RGB crop to the cache as PNG (stored BGR so cv2.imread +
    the loader's BGR->RGB conversion round-trips correctly)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + ".tmp.png")
    cv2.imwrite(str(tmp), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    tmp.replace(cache_path)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class BirdDetector:
    """COCO-pretrained Faster R-CNN v2 restricted to the bird class.

    torch/torchvision are imported lazily so that modules which only need the
    bbox math or cache-path helpers (e.g. RawImageLoader) don't pull the heavy
    detection stack.
    """

    def __init__(self, device: str = "cpu", conf_threshold: float = 0.30):
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self._torch = torch
        self.device = device
        self.conf_threshold = conf_threshold
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()

    def best_bird_box(self, image_rgb: np.ndarray) -> tuple[float, float, float, float] | None:
        """Return the highest-confidence bird box (x1, y1, x2, y2) above the
        confidence threshold, or None if no bird is found."""
        torch = self._torch
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().float().div(255.0)
        with torch.no_grad():
            output = self.model([tensor.to(self.device)])[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        best_box = None
        best_score = self.conf_threshold
        for box, label, score in zip(boxes, labels, scores):
            if label == COCO_BIRD_CLASS and score >= best_score:
                best_score = score
                best_box = tuple(float(v) for v in box)
        return best_box


def build_crop(
    image_rgb: np.ndarray,
    detector: BirdDetector,
    params: CropParams,
) -> tuple[np.ndarray, bool]:
    """Produce the cached crop for one decoded image.

    Returns (crop_rgb, found_bird). When no bird is detected the full frame is
    returned (downscaled) so the image still yields a usable, bird-agnostic
    input rather than being dropped.
    """
    height, width = image_rgb.shape[:2]
    box = detector.best_bird_box(image_rgb)
    if box is None:
        return downscale_long_side(image_rgb, params.max_side), False
    expanded = expand_and_clamp_box(box, params.margin_frac, width, height)
    crop = crop_to_box(image_rgb, expanded)
    return downscale_long_side(crop, params.max_side), True
