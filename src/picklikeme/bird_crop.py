"""Animal detection and crop caching.

This is the "subject-centered input" preprocessing phase: instead of feeding
the full frame to the model, we detect the animal once per image, crop tightly
around it (small safety margin, aspect ratio preserved), and cache the crop so
training never re-detects or re-decodes RAW.

The module, its classes, and its functions keep "bird" in their names for
historical reasons (the project started bird-only) — the detector now accepts
any of the COCO animal classes in SUPPORTED_ANIMAL_CLASSES.

Design notes:
- Detection uses torchvision's COCO-pretrained Faster R-CNN v2. It runs once,
  in a single process (see picklikeme.preprocess) — never inside DataLoader
  workers, where N workers would each load a detector onto the GPU and contend
  for memory.
- The crop is a true sub-rectangle of the source, so the animal's geometry is
  never distorted; fixed-size model input is produced later by letterbox
  padding in RawImageLoader, not by stretching.
- Cache entries are keyed by the absolute source path only, so one cache is
  reusable across model input sizes (384/512/640): detect once, letterbox to
  any size at load time.
- If no supported animal is detected, the full frame is cached as the fallback,
  so every image still yields an input and is never re-detected.

Bump CROP_CACHE_VERSION whenever the crop algorithm, its defaults, or the set
of accepted detection classes changes, so a stale cache is detected instead of
silently reused. v1 = bird only; v2 = SUPPORTED_ANIMAL_CLASSES; v3 = area-
dominant selection among survivors; v4 = group-scene handling (see "Crop
selection policy" below).

Crop selection policy
----------------------
Faster R-CNN's own postprocessing already does the filtering that is NOT
policy: it drops anything below its own score threshold, keeps only the
accepted classes (SUPPORTED_ANIMAL_CLASSES by default), and runs per-class
non-maximum suppression. What is left after that - the *surviving*
detections - still often number more than one (two classes competing for the
same animal, a second animal in frame, a false detection in the background),
and something has to decide what to crop to. That is `select_best_detection`.

- **v2 and earlier (superseded): highest confidence wins, full stop.** A
  single pass tracking a running max score. No box size, aspect ratio,
  position, or anything else about the box ever entered the comparison. This
  under-served a wildlife photography archive because a small, sharp, highly
  confident detection (a bird poking out of a corner, a distant animal caught
  cleanly) would beat the large, obviously-intended subject the photographer
  actually composed the shot around, whenever the large subject's box scored
  even slightly lower - motion blur, an awkward pose, or partial occlusion are
  all things that suppress a detector's confidence on a large, real subject
  without making it any less the photo's subject.

- **v3 (superseded on its own, still the policy below `group_scene_threshold`):
  area dominates; confidence only breaks a near-tie.** The largest surviving
  detection wins, *unless* another detection's area is within `area_tie_frac`
  (default 10%) of the largest, in which case the highest-confidence detection
  among that near-largest group wins. A detection whose area is not close to
  the largest can never win by having higher confidence - there is no
  confidence value large enough to compensate for a much smaller box. This is
  a deliberate size-first policy, not a weighted score: area and confidence
  are never combined into one number.

- **v4 (current): group scenes crop to the whole group, not one member of it.**
  Wildlife photography routinely and *intentionally* frames a flock, a herd, a
  colony - a group of animals is the subject, not any single one of them.
  Picking "the best" individual detection out of a flock of forty birds
  (by area or by confidence, it does not matter which) crops to one bird and
  discards the photograph's actual subject. So when the number of surviving
  detections reaches `group_scene_threshold` (default 10), individual
  selection is skipped entirely: the crop target becomes the smallest box
  enclosing every surviving detection, then the normal margin and downstream
  crop pipeline apply exactly as they do for a single subject. The full-frame
  fallback (see "If no supported animal is detected" above) still only
  applies when *nothing* was detected - a group scene never falls back to the
  full frame, even when the group only occupies a small part of it: the whole
  point is a tight crop around the actual subject, individual or group.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .profiling import PROFILE

# COCO category indices in torchvision's detection weights metadata
# (FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"]). The animal
# classes are contiguous: bird(16) .. giraffe(25).
COCO_BIRD_CLASS = 16

# Wildlife: the primary target of this project (wildlife photography).
WILDLIFE_CLASSES: dict[int, str] = {
    COCO_BIRD_CLASS: "bird",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
}

# The remaining COCO animal classes. Included because supporting them costs
# nothing beyond these five entries (the selection logic is class-agnostic),
# and a horse/cow/sheep in frame is the photo's subject just as much as a
# zebra is. Restrict with BirdDetector(classes=WILDLIFE_CLASSES) if a run
# should consider wildlife only.
DOMESTIC_ANIMAL_CLASSES: dict[int, str] = {
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
}

SUPPORTED_ANIMAL_CLASSES: dict[int, str] = {**WILDLIFE_CLASSES, **DOMESTIC_ANIMAL_CLASSES}


def coco_class_name(label: int) -> str:
    """Human-readable name for a supported COCO animal class (for logging)."""
    return SUPPORTED_ANIMAL_CLASSES.get(int(label), f"class {int(label)}")


CROP_CACHE_VERSION = "v4"
CROP_PARAMS_FILENAME = "crop_params.json"

# Cache entries live in cache_dir/<first 2 hex chars of digest>/<digest>.png.
# Two characters gives 256 shards: ~215 files per shard at 55k images, which
# keeps NTFS directory operations fast without creating a deep tree.
CACHE_SHARD_CHARS = 2

# How close two surviving detections' areas must be, as a fraction of the
# larger one, before confidence is allowed to decide between them. 0.10 means
# a detection needs to reach 90% of the largest detection's area to even be
# considered for the confidence tie-break; anything smaller loses on size
# alone, however much more confident it is. See select_best_detection().
DEFAULT_AREA_TIE_FRAC = 0.10

# At or above this many surviving detections, the image is treated as a group
# scene: the crop target becomes the box enclosing all of them, not a single
# individual. See select_best_detection() and the module docstring's "Crop
# selection policy" section.
DEFAULT_GROUP_SCENE_THRESHOLD = 10


@dataclass(frozen=True)
class CropParams:
    """Parameters that define how a crop cache was built. Recorded alongside
    the cache so a mismatched configuration can be detected."""

    margin_frac: float = 0.05          # small safety margin around the tight box
    conf_threshold: float = 0.30       # min detection confidence to accept a detection
    max_side: int = 1024               # cap the cached crop's long side (px)
    area_tie_frac: float = DEFAULT_AREA_TIE_FRAC  # size-tie tolerance for select_best_detection
    group_scene_threshold: int = DEFAULT_GROUP_SCENE_THRESHOLD  # >= this many detections -> group scene
    detector: str = "fasterrcnn_resnet50_fpn_v2"
    version: str = CROP_CACHE_VERSION


# ---------------------------------------------------------------------------
# Bounding-box geometry (pure functions, no torch import needed)
# ---------------------------------------------------------------------------

def box_area(box: tuple[float, float, float, float]) -> float:
    """Pixel area of an (x1, y1, x2, y2) box. Never negative, even for a
    malformed box (x2 < x1 or y2 < y1), so area-based comparisons - notably
    select_best_detection() - stay well-defined without their own guards."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def enclosing_box(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """The smallest (x1, y1, x2, y2) box containing every given box.

    Used for group scenes: the crop target is the region spanning the whole
    group, not any one member of it. `boxes` must be non-empty - the caller
    (select_best_detection) already knows there is at least one candidate by
    the time this is reached.
    """
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return (x1, y1, x2, y2)


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

    THE single place in the codebase that constructs a cache path: every read
    and every write goes through here, so the layout can never diverge between
    producer and consumer.

    Sharded into 256 subdirectories by the first two hex characters of the
    digest, because a flat directory holding 55k+ entries degrades directory
    operations on NTFS. The path is always *computed* from the digest — the
    cache is never scanned, globbed, or walked to find an entry.

    Independent of crop parameters (those are recorded in crop_params.json);
    rebuilding the cache overwrites in place.
    """
    resolved = str(Path(source_path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:20]
    return Path(cache_dir) / digest[:CACHE_SHARD_CHARS] / f"{digest}.png"


def write_crop_params(cache_dir: str | Path, params: CropParams) -> Path:
    with PROFILE.stage("metadata write"):
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


DETECTIONS_SUFFIX = ".detections.json"


def detections_cache_path(cache_dir: str | Path, source_path: str | Path) -> Path:
    """Where the detection record for an image lives: beside its cached crop.

    Same digest, so the record is found the same way the crop is - by
    computation, never by scanning.
    """
    crop = crop_cache_path(cache_dir, source_path)
    return crop.with_name(crop.stem + DETECTIONS_SUFFIX)


def save_detections(
    cache_dir: str | Path,
    source_path: str | Path,
    result: "CropResult",
) -> Path | None:
    """Record what the detector saw, for later diagnosis.

    Written during preprocessing, when the detector has just run anyway, so no
    consumer ever needs to re-run inference to draw a box. Failure to write is
    not fatal: the record is a convenience, the crop is the product.
    """
    target = detections_cache_path(cache_dir, source_path)
    payload = {
        "version": 1,
        "source_size": list(result.source_size) if result.source_size else None,
        "selected": _detection_dict(result.detection),
        "detections": [_detection_dict(d) for d in result.all_detections],
        "expanded_box": list(result.expanded_box) if result.expanded_box else None,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)
    except OSError:
        return None
    return target


def _detection_dict(detection: "BirdDetection | None") -> dict | None:
    if detection is None:
        return None
    return {"box": list(detection.box), "score": detection.score, "label": int(detection.label)}


def read_detections(cache_dir: str | Path, source_path: str | Path) -> dict | None:
    """The recorded detections for an image, or None if none were recorded."""
    target = detections_cache_path(cache_dir, source_path)
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_crop_png(cache_path: Path, crop_rgb: np.ndarray) -> None:
    """Write an RGB crop to the cache as PNG (stored BGR so cv2.imread +
    the loader's BGR->RGB conversion round-trips correctly)."""
    with PROFILE.stage("png encode + write"):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp.png")
        cv2.imwrite(str(tmp), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        tmp.replace(cache_path)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BirdDetection:
    """A single animal detection: its box (x1, y1, x2, y2), confidence score,
    and the COCO class that matched (bird, elephant, zebra, ...). The rich
    result other code consumes so nobody re-implements the selection or
    confidence logic."""

    box: tuple[float, float, float, float]
    score: float
    label: int = COCO_BIRD_CLASS


def _group_scene_detection(candidates: Sequence[BirdDetection]) -> BirdDetection:
    """A synthetic detection representing an entire group, for select_best_detection()'s
    group-scene branch. Its `box` is the smallest box enclosing every candidate
    - the actual crop target - so it is never just informational the way
    `score`/`label` are here. `score` and `label` (the most confident
    individual member's) exist only so this still behaves like a normal
    BirdDetection for logging and the detector-box overlay: the enclosing box
    renders as the "selected" crop, and every group member still renders as
    a runner-up, which is exactly the picture a group scene should show.
    """
    representative = max(candidates, key=lambda detection: detection.score)
    return BirdDetection(
        box=enclosing_box([candidate.box for candidate in candidates]),
        score=representative.score,
        label=representative.label,
    )


def select_best_detection(
    candidates: Sequence[BirdDetection],
    area_tie_frac: float = DEFAULT_AREA_TIE_FRAC,
    group_scene_threshold: int = DEFAULT_GROUP_SCENE_THRESHOLD,
) -> BirdDetection | None:
    """The crop target build_crop should use, chosen from every detection that
    has already passed class and confidence filtering (and, upstream of this,
    the detector's own per-class NMS - see the module docstring's "Crop
    selection policy" section).

    Two policies, chosen by how many detections survived:

    - **Fewer than `group_scene_threshold`: bounding-box area dominates.** The
      largest candidate wins, unless another candidate's area is within
      `area_tie_frac` of it, in which case confidence breaks the tie among
      that near-largest group only. This is deliberately not a weighted score
      - area and confidence are never combined into one number - so a
      detection that is not close in size to the largest can never win by
      being more confident, however large the gap.

    - **`group_scene_threshold` or more: the image is a group scene.** Picking
      one detection out of a flock, a herd or a colony would crop to a single
      animal and discard the photograph's actual subject, so no individual
      detection is selected at all - the target becomes the smallest box
      enclosing every surviving detection (see `_group_scene_detection`). The
      normal crop margin and downstream pipeline still apply to that box
      exactly as they would to a single detection; there is no full-frame
      fallback here, because a group is still a real, locatable subject.

    The single source of truth for subject selection: BirdDetector.detect_best_bird
    and detect_with_all both call this rather than each implementing their own
    comparison, so the two can never disagree about the crop target.

    Returns None for an empty `candidates` (nothing survived filtering),
    mirroring "no detection" everywhere else in this module.
    """
    if not candidates:
        return None

    if len(candidates) >= group_scene_threshold:
        return _group_scene_detection(candidates)

    areas = [(candidate, box_area(candidate.box)) for candidate in candidates]
    largest_area = max(area for _, area in areas)

    if largest_area <= 0.0:
        # Degenerate boxes only (zero width or height, which a real detector
        # should never emit): area carries no information here, so fall back
        # to confidence alone rather than an arbitrary choice among equally
        # uninformative candidates.
        return max(candidates, key=lambda detection: detection.score)

    tie_threshold = largest_area * (1.0 - area_tie_frac)
    contenders = [candidate for candidate, area in areas if area >= tie_threshold]
    return max(contenders, key=lambda detection: detection.score)


@dataclass
class CropResult:
    """Everything build_crop produces for one image: the crop the model will
    receive, the detection it came from (None on full-frame fallback), and the
    expanded box actually cropped (None on fallback)."""

    crop: np.ndarray
    detection: BirdDetection | None
    expanded_box: tuple[int, int, int, int] | None
    # Every accepted detection, winner included. Recorded so a later diagnostic
    # can show the runners-up without re-running inference; empty when the
    # caller did not ask for them, which changes no cropping behaviour.
    all_detections: list[BirdDetection] = field(default_factory=list)
    source_size: tuple[int, int] | None = None  # (width, height) of the full frame


@dataclass(frozen=True)
class NormalizedCrop:
    """An editor-agnostic crop rectangle in normalized [0, 1] image coordinates
    (fractions of width/height), plus rotation angle. This is the generic crop
    representation the crop engine exposes; exporters translate it into a
    specific editor's format (e.g. Lightroom crs: fields)."""

    left: float
    top: float
    right: float
    bottom: float
    angle: float = 0.0


class BirdDetector:
    """COCO-pretrained Faster R-CNN v2 restricted to a set of animal classes.

    Defaults to SUPPORTED_ANIMAL_CLASSES (wildlife + the remaining COCO
    animals); pass `classes` to restrict it (e.g. WILDLIFE_CLASSES, or
    {COCO_BIRD_CLASS} to reproduce the original bird-only behavior).

    torch/torchvision are imported lazily so that modules which only need the
    bbox math or cache-path helpers (e.g. RawImageLoader) don't pull the heavy
    detection stack.
    """

    def __init__(
        self,
        device: str = "cpu",
        conf_threshold: float = 0.30,
        classes: "dict[int, str] | set[int] | None" = None,
        area_tie_frac: float = DEFAULT_AREA_TIE_FRAC,
        group_scene_threshold: int = DEFAULT_GROUP_SCENE_THRESHOLD,
    ):
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self._torch = torch
        self.device = device
        self.conf_threshold = conf_threshold
        self.classes = frozenset(SUPPORTED_ANIMAL_CLASSES if classes is None else classes)
        self.area_tie_frac = area_tie_frac
        self.group_scene_threshold = group_scene_threshold
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()

    def detect_with_all(self, image_rgb: np.ndarray) -> "tuple[BirdDetection | None, list[BirdDetection]]":
        """(winner, every accepted detection) from a **single** forward pass.

        Exists so a caller that wants the runners-up - the false-negative
        diagnostic overlay - does not have to run inference a second time. The
        winner is chosen by select_best_detection(), the same function
        detect_best_bird() delegates to, so the two can never disagree.
        """
        torch = self._torch
        with PROFILE.stage("detector preprocess"):
            tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().float().div(255.0)
            device_tensor = tensor.to(self.device)
            PROFILE.cuda_sync(torch, self.device)
        with PROFILE.stage("gpu inference"):
            with torch.no_grad():
                output = self.model([device_tensor])[0]
            PROFILE.cuda_sync(torch, self.device)

        with PROFILE.stage("detector postprocess"):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            accepted: list[BirdDetection] = [
                BirdDetection(box=tuple(float(v) for v in box), score=float(score), label=int(label))
                for box, label, score in zip(boxes, labels, scores)
                if int(label) in self.classes and score >= self.conf_threshold
            ]
            best = select_best_detection(accepted, self.area_tie_frac, self.group_scene_threshold)
        return best, accepted

    def detect_best_bird(self, image_rgb: np.ndarray) -> BirdDetection | None:
        """The detection build_crop should crop to, or None if nothing
        survived class/confidence filtering.

        This is the single source of truth for subject selection: everything
        that needs a box goes through here (or through detect_with_all, which
        this delegates to, so the two entry points always agree). What "best"
        means is entirely select_best_detection()'s policy - area-dominant
        with confidence as a tie-break below `group_scene_threshold`
        detections, the enclosing box of the whole group at or above it; see
        the module docstring's "Crop selection policy" section.
        """
        best, _ = self.detect_with_all(image_rgb)
        return best

    def best_bird_box(self, image_rgb: np.ndarray) -> tuple[float, float, float, float] | None:
        """Convenience wrapper over detect_best_bird for callers that only need
        the box (e.g. a presence check). No detection logic of its own."""
        detection = self.detect_best_bird(image_rgb)
        return detection.box if detection is not None else None


def build_crop(
    image_rgb: np.ndarray,
    detector: BirdDetector,
    params: CropParams,
    collect_detections: bool = False,
) -> CropResult:
    """Produce the crop for one decoded image, returning the crop plus the
    detection and expanded box it came from.

    When no supported animal is detected the full frame is returned
    (downscaled) so the image still yields a usable, subject-agnostic input
    rather than being dropped; in that case detection and expanded_box are None.

    `collect_detections=True` additionally records every accepted detection, for
    later diagnosis. It costs nothing - the same forward pass produces them - and
    changes neither the chosen box nor the crop.
    """
    height, width = image_rgb.shape[:2]
    # Opt-in: `collect_detections` asks for the runners-up too, from the same
    # single forward pass. Default False keeps the long-standing
    # `detect_best_bird` contract, which every caller and test double relies on.
    if collect_detections:
        detection, accepted = detector.detect_with_all(image_rgb)
    else:
        detection, accepted = detector.detect_best_bird(image_rgb), []
    with PROFILE.stage("crop generation"):
        if detection is None:
            return CropResult(
                crop=downscale_long_side(image_rgb, params.max_side),
                detection=None,
                expanded_box=None,
                all_detections=accepted,
                source_size=(width, height),
            )
        expanded = expand_and_clamp_box(detection.box, params.margin_frac, width, height)
        crop = downscale_long_side(crop_to_box(image_rgb, expanded), params.max_side)
        return CropResult(
            crop=crop,
            detection=detection,
            expanded_box=expanded,
            all_detections=accepted,
            source_size=(width, height),
        )


def compute_composition_crop(
    detection: BirdDetection,
    image_width: int,
    image_height: int,
    margin_frac: float = 0.0,
) -> NormalizedCrop:
    """A compositional crop for photo editors, derived from the same bird
    detection as training but with a different policy.

    Unlike build_crop (tight, variable aspect, maximizes bird area for the
    model), this expands the bird box by a margin and then grows it to the
    ORIGINAL image aspect ratio, so an editor receives an undistorted crop that
    is never square (as a photo), never letterboxed, never stretched. Returns
    normalized [0, 1] coordinates (the editor-agnostic representation).

    Steps: (1) expand the bird box symmetrically by margin_frac; (2) grow the
    smaller dimension until the box matches the image aspect ratio, keeping the
    center; (3) if that no longer fits, the aspect-correct crop is the whole
    frame; (4) otherwise shift the box back inside the frame, preserving the
    center where possible.
    """
    x1, y1, x2, y2 = detection.box
    box_w = x2 - x1
    box_h = y2 - y1
    x1 -= box_w * margin_frac
    x2 += box_w * margin_frac
    y1 -= box_h * margin_frac
    y2 += box_h * margin_frac

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    target_aspect = image_width / image_height
    if width / height < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect

    if width >= image_width or height >= image_height:
        return NormalizedCrop(0.0, 0.0, 1.0, 1.0)

    center_x = min(max(center_x, width / 2.0), image_width - width / 2.0)
    center_y = min(max(center_y, height / 2.0), image_height - height / 2.0)

    def _clamp01(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    return NormalizedCrop(
        left=_clamp01((center_x - width / 2.0) / image_width),
        top=_clamp01((center_y - height / 2.0) / image_height),
        right=_clamp01((center_x + width / 2.0) / image_width),
        bottom=_clamp01((center_y + height / 2.0) / image_height),
    )
