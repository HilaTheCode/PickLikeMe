"""EyePose-v0 - a second, interchangeable bird-eye detector.

Implements the same `eyes.detector.EyeDetector` protocol
`superanimal_bird.SuperAnimalBirdEyeDetector` does, so `ranking.classic`,
`eyes.cache`, and the Gallery/Loupe overlay never know or care which of the
two produced a given `EyeDetection` - see `eyes.detector`'s module docstring
for that contract, and `eyes.build_eye_detector` for how a caller picks one
by name. SuperAnimal-Bird stays fully intact and selectable; this is an
addition, not a replacement.

The model: `synthet/eye-pose-v0` on Hugging Face - a YOLO11n-pose checkpoint
(Ultralytics architecture) fine-tuned on CUB-200-2011 for six bird
head/body landmarks: beak, left_eye, right_eye, head_top, left_shoulder,
right_shoulder (in that channel order - see KPT_NAMES; matches the
repo's own `wildlife_bird.yaml`). MIT-licensed weights.

Why ONNX Runtime and not the `ultralytics` package at runtime
---------------------------------------------------------------
Running a `.pt` YOLO checkpoint normally means `from ultralytics import
YOLO`. That package is AGPL-3.0 - the exact license `superanimal_bird.py`'s
own module docstring already rejected as "a real consideration for a
distributed desktop app" (which is why SuperAnimal-Bird was hand-reimplemented
against plain torch+timm instead of depending on DeepLabCut). Taking on
`ultralytics` here would reopen that same problem despite the model
*weights* themselves being MIT.

The resolution mirrors SuperAnimal-Bird's own precedent - depend on the
published artifact, not the framework that produced it - one step further:
`ensure_onnx_weights` downloads the published `.pt` (same one-time,
urllib-only pattern `superanimal_bird.ensure_weights` uses) and converts it
to ONNX **once**, on first use, caching the result beside it. That
conversion step is the *only* place `ultralytics` is ever imported, lazily,
and only if the ONNX file isn't already cached - it is never imported by
`detect()`, `__init__`'s session construction, or anything reachable from a
normal ranking run once the `.onnx` file exists. `ultralytics` is therefore
an **optional, one-time setup dependency** (`pip install ultralytics`, once,
the first time this backend runs), never a runtime one; the actual shipped
runtime dependency is `onnxruntime`, MIT-licensed, and inference through it
was verified byte-for-byte identical to the original PyTorch checkpoint's
own raw forward pass on real photos (same static 640x640 letterbox tensor
in, same (1, 23, 8400) tensor out) before this module was written - see
`docs/eyepose_v0_validation.md`.

Coordinate contract
--------------------
`detect(subject_crop_rgb)` returns coordinates in the SAME frame every other
`EyeDetector` does: pixels in the crop it was given, never the model's own
640x640 input tensor. `_predict_landmarks` is the one place that boundary is
crossed - forward transform (`_letterbox_forward`) into model space, decode
(`_decode_best`), inverse transform (`_letterbox_inverse`) back out - and
each of those three is a small, independently unit-tested pure function
rather than inline arithmetic, specifically so the forward/inverse mapping
can be verified on its own (see test_eyepose_v0.py's CoordinateTransform
tests) independent of whether the model's *predictions* are any good.

Accept/reject gate
-------------------
Two independent checks, mirroring SuperAnimal-Bird's own two-gate shape
(confidence, then a geometric plausibility check) without literally reusing
its arithmetic - the two models' landmark schemas are different enough that
a direct port would not mean the same thing:

- **Confidence.** The primary eye's own visibility score must clear
  `min_confidence`.
- **Anatomical plausibility.** The primary eye must lie reasonably close to
  the beak<->head_top axis (the two most reliably-detected, highest-contrast
  points on a bird's head, the same reasoning SuperAnimal-Bird applied to
  its own choice of crown/bill) - distance from that line segment, floored
  and normalised by beak<->head_top's own length (the head-scale reference),
  must stay under `max_head_distance_ratio`. Catches a keypoint that landed
  on a shoulder or the background: confidently, but nowhere near a head.

Unlike SuperAnimal-Bird's `DEFAULT_MAX_EYE_DISAGREEMENT` (tuned against a
30-image hand-adjudicated sample of this project's own archive), the
defaults below are reasonable starting points, not empirically validated
ones - this project has no equivalent adjudicated sample for eye-pose-v0 yet.
Tune `min_confidence`/`max_head_distance_ratio` after reviewing real results
through the Gallery/Loupe debug overlay, the same way SuperAnimal-Bird's own
numbers were originally derived.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..bird_crop import COCO_BIRD_CLASS
from ..config import PROJECT_ROOT
from .detector import EyeDetection, EyeKeypoint, derive_eye_box

if TYPE_CHECKING:  # pragma: no cover - typing only
    import onnxruntime as ort

logger = logging.getLogger(__name__)

# Channel order of the model's pose head - see the repo's own
# wildlife_bird.yaml `kpt_names`. The order IS the channel mapping (index
# into each keypoint's (x, y, vis) triple in the raw output), exactly the
# role BODYPARTS/LEFT_EYE_INDEX play for SuperAnimal-Bird.
KPT_NAMES: tuple[str, ...] = ("beak", "left_eye", "right_eye", "head_top", "left_shoulder", "right_shoulder")

WEIGHTS_REPO = "synthet/eye-pose-v0"
PT_FILENAME = "eye_pose_v0.pt"
ONNX_FILENAME = "eye_pose_v0.onnx"
WEIGHTS_URL = f"https://huggingface.co/{WEIGHTS_REPO}/resolve/main/{PT_FILENAME}"
# Shares cache/eye_models with SuperAnimal-Bird (different filenames) -
# derived data that can always be re-fetched/re-converted, same reasoning as
# superanimal_bird.DEFAULT_WEIGHTS_DIR.
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "cache" / "eye_models"

# The model's own fixed export input size (see the module docstring's ONNX
# section) - not tunable, the exported graph's shape is baked to this.
INPUT_SIZE = 640
# Letterbox padding colour - Ultralytics' own convention (mid-grey), matched
# here so the model sees exactly the padding it was trained/exported against.
PAD_VALUE = 114

# A starting default, not an empirically validated one - see the module
# docstring's "Accept/reject gate" section.
DEFAULT_MIN_CONFIDENCE = 0.5
# Matches SuperAnimal-Bird's own default: the two backends' eye boxes should
# be comparably sized for a fair side-by-side comparison, and 0.08 is not
# tied to anything model-specific - see eyes.detector.derive_eye_box.
DEFAULT_EYE_BOX_FRAC = 0.08
MIN_EYE_BOX_PX = 12.0
# See "Anatomical plausibility" above. A starting default.
DEFAULT_MAX_EYE_HEAD_DISTANCE_RATIO = 1.5
# Floor for the beak<->head_top reference distance, exactly like
# SuperAnimal-Bird's MIN_HEAD_SCALE_PX - guards the pathological case of the
# two points coinciding, never actually reached on a real detection.
MIN_HEAD_SCALE_PX = 3.0


def ensure_onnx_weights(weights_dir: str | Path | None = None) -> Path:
    """The local ONNX graph's path, downloading and converting it once if it
    is not there yet.

    Kept separate from `__init__` for the same reason
    `superanimal_bird.ensure_weights` is: a caller can pre-fetch/pre-convert
    without constructing a detector (and therefore without needing
    onnxruntime importable yet either), and the one-time cost - a small
    download plus, only if `ultralytics` happens to already be installed or
    the caller installs it for this, a several-second export - is an
    obvious, named step rather than a surprise inside a constructor.
    """
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    onnx_path = weights_dir / ONNX_FILENAME
    if onnx_path.is_file():
        return onnx_path
    pt_path = _ensure_pt_weights(weights_dir)
    _export_onnx(pt_path, onnx_path)
    return onnx_path


def _ensure_pt_weights(weights_dir: Path) -> Path:
    target = weights_dir / PT_FILENAME
    if target.is_file():
        return target

    import urllib.request

    weights_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading the eye-pose-v0 checkpoint (~5.4 MB) to %s", target)
    # Downloaded to a temporary name and renamed on success - the same
    # write-then-replace discipline superanimal_bird.ensure_weights and the
    # crop cache both use, so an interrupted transfer can never leave a
    # truncated file that later loads as a corrupt checkpoint.
    tmp = target.with_name(target.name + ".part")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _export_onnx(pt_path: Path, onnx_path: Path) -> None:
    """The one and only place `ultralytics` is ever imported - see the
    module docstring's "Why ONNX Runtime" section. Raises a clear,
    actionable error rather than a bare ImportError if it is not installed;
    this is the single one-time setup step the whole rest of this backend
    never needs again once `onnx_path` exists.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised only without ultralytics installed
        raise RuntimeError(
            "Converting eye-pose-v0's published checkpoint to ONNX requires the "
            "'ultralytics' package for this one-time, offline conversion step only "
            "- it is never imported again afterward (see eyes/eyepose_v0.py's module "
            "docstring for why). Install it once with `pip install ultralytics` and "
            "re-run; the resulting eye_pose_v0.onnx is cached in "
            f"{onnx_path.parent} and every future run uses onnxruntime alone."
        ) from exc

    logger.info("Converting eye-pose-v0 to ONNX (one-time): %s -> %s", pt_path, onnx_path)
    model = YOLO(str(pt_path))
    exported = Path(model.export(format="onnx", imgsz=INPUT_SIZE, opset=12, simplify=True, dynamic=False, nms=False))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != onnx_path.resolve():
        exported.replace(onnx_path)


def _select_providers(device: str) -> list[str]:
    """onnxruntime silently skips a provider it cannot use (e.g. a plain
    `onnxruntime` install with no CUDA build present), so listing the CUDA
    provider first is always safe - it falls back to CPU on its own."""
    if device.startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _letterbox_forward(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, float, float]:
    """Resize `image` to fit within `size`x`size` preserving aspect ratio,
    then centre-pad to exactly `size`x`size` - Ultralytics' own `LetterBox`
    transform (`auto=False`, the mode its ONNX export always uses, since a
    static graph needs a fixed input shape), reproduced here in plain numpy
    so no part of this backend depends on `ultralytics` at runtime. Verified
    to reproduce it exactly - see the module docstring.

    Returns (padded_image, scale, left_pad, top_pad); `_letterbox_inverse`
    is the exact matching inverse of this transform.
    """
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = _resize(image, new_w, new_h)
    pad_w, pad_h = (size - new_w) / 2.0, (size - new_h) / 2.0
    # The -0.1/+0.1 split (not a plain round()) is Ultralytics' own tie-break
    # for an odd total padding amount, reproduced exactly so a pixel-for-pixel
    # match against the original checkpoint holds even in that case.
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    padded = _pad(resized, top, bottom, left, right, PAD_VALUE)
    return padded, scale, float(left), float(top)


def _letterbox_inverse(x: float, y: float, scale: float, pad_x: float, pad_y: float) -> tuple[float, float]:
    """The exact inverse of `_letterbox_forward`'s geometry: subtract the
    pad, then undo the scale - the same two-step `analyzer.contactsheets
    .annotate_thumbnail`'s own frame<->thumbnail mapping uses, applied to
    model-input space instead of a thumbnail."""
    return ((x - pad_x) / scale, (y - pad_y) / scale)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(image, (max(1, width), max(1, height)), interpolation=cv2.INTER_LINEAR)


def _pad(image: np.ndarray, top: int, bottom: int, left: int, right: int, value: int) -> np.ndarray:
    import cv2

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(value, value, value))


def _to_tensor(padded_rgb: np.ndarray) -> np.ndarray:
    """(1, 3, H, W) float32 in [0, 1] - the model's own expected input.
    Already RGB in, RGB expected (see the class docstring's coordinate
    contract) - no colour-channel conversion, unlike a caller starting from
    a BGR-loaded file."""
    normalized = padded_rgb.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))[None, ...]


def _decode_best(raw: np.ndarray) -> tuple[float, dict[str, tuple[float, float, float]]] | None:
    """The single highest-confidence detection from the model's raw
    (1, 23, 8400) output - 4 box values + 1 class score ("bird", the model's
    only class) + 6 keypoints x (x, y, visibility), one column per anchor
    point, all already decoded to 640-space pixel coordinates and
    sigmoid-activated scores by the exported graph itself (verified against
    the checkpoint's own raw forward pass - see the module docstring).

    A simple argmax, not full NMS: `detect()` is always called on an
    already-cropped single subject (see `bird_crop`), so at most one real
    bird is ever in frame and the highest-scoring anchor is it - the same
    "one subject, no multi-object suppression needed" assumption
    SuperAnimal-Bird's own heatmap argmax already makes. Returns None only
    for a degenerate all-zero output (should not happen on real data; guards
    against a crash rather than a wrong answer).
    """
    predictions = raw[0]  # (23, 8400)
    confidences = predictions[4, :]
    best = int(np.argmax(confidences))
    best_confidence = float(confidences[best])
    if best_confidence <= 0.0:
        return None
    keypoints: dict[str, tuple[float, float, float]] = {}
    for index, name in enumerate(KPT_NAMES):
        base = 5 + index * 3
        keypoints[name] = (float(predictions[base, best]), float(predictions[base + 1, best]), float(predictions[base + 2, best]))
    return best_confidence, keypoints


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Shortest distance from (px, py) to the line SEGMENT a<->b (not the
    infinite line) - the standard clamped-projection formula. Used by the
    anatomical-plausibility check (see the class docstring) so an eye
    slightly beyond the beak or head_top end of the segment is still
    measured against the nearest point ON the head, not extrapolated past it."""
    abx, aby = bx - ax, by - ay
    length_sq = abx * abx + aby * aby
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


class EyePoseV0EyeDetector:
    """Bird eye localisation from the `synthet/eye-pose-v0` checkpoint, via
    ONNX Runtime. Implements `eyes.detector.EyeDetector` - see this module's
    own docstring for the model, the licensing rationale, and the accept/
    reject gate.
    """

    detector_id = "eyepose-v0"

    def __init__(
        self,
        device: str = "cpu",
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        eye_box_frac: float = DEFAULT_EYE_BOX_FRAC,
        max_head_distance_ratio: float = DEFAULT_MAX_EYE_HEAD_DISTANCE_RATIO,
        weights_dir: str | Path | None = None,
    ) -> None:
        import onnxruntime as ort

        self.min_confidence = min_confidence
        self.eye_box_frac = eye_box_frac
        self.max_head_distance_ratio = max_head_distance_ratio

        onnx_path = ensure_onnx_weights(weights_dir)
        self._session: "ort.InferenceSession" = ort.InferenceSession(
            str(onnx_path), providers=_select_providers(device)
        )
        self._input_name = self._session.get_inputs()[0].name

    def supports(self, coco_label: int) -> bool:
        """Birds only, exactly like SuperAnimal-Bird - the model was
        fine-tuned exclusively on CUB-200-2011 bird photos."""
        return int(coco_label) == COCO_BIRD_CLASS

    def detect(self, subject_crop_rgb: np.ndarray) -> EyeDetection:
        """See `eyes.detector.EyeDetector.detect` and this module's own
        docstring for the accept/reject gate. Always returns an
        `EyeDetection`, never `None`, even when nothing usable was found -
        the same contract SuperAnimal-Bird's own `detect` documents."""
        height, width = (subject_crop_rgb.shape[:2] if subject_crop_rgb is not None else (0, 0))
        if subject_crop_rgb is None or subject_crop_rgb.size == 0:
            return EyeDetection(
                box=(0.0, 0.0, 1.0, 1.0), confidence=0.0, detector_id=self.detector_id, accepted=False
            )

        landmarks = self._predict_landmarks(subject_crop_rgb)
        if landmarks is None:
            return EyeDetection(
                box=(0.0, 0.0, float(width), float(height)),
                confidence=0.0, detector_id=self.detector_id, accepted=False,
            )

        left, right = landmarks["left_eye"], landmarks["right_eye"]
        primary, other = (left, right) if left.confidence >= right.confidence else (right, left)

        accepted = primary.confidence >= self.min_confidence
        if accepted:
            accepted = self._anatomically_plausible(primary, landmarks)

        box = derive_eye_box(primary.x, primary.y, width, height, self.eye_box_frac, MIN_EYE_BOX_PX)
        return EyeDetection(
            box=box,
            confidence=primary.confidence,
            center=(primary.x, primary.y),
            detector_id=self.detector_id,
            left=left,
            right=right,
            accepted=accepted,
        )

    def _predict_landmarks(self, crop_rgb: np.ndarray) -> dict[str, EyeKeypoint] | None:
        """All six landmarks, in `crop_rgb`'s own pixel space - the forward
        transform, one ONNX Runtime call, decode, then the inverse
        transform. Split out from `detect()` (mirroring
        SuperAnimal-Bird's own `_predict_keypoints`) so a test can
        monkeypatch this one method with controlled, crop-space keypoints
        and exercise the accept/reject arithmetic without a real model."""
        debug = self._run_model(crop_rgb)
        if debug is None:
            return None
        return debug["landmarks_crop"]

    def debug_predict(self, crop_rgb: np.ndarray) -> dict | None:
        """Every intermediate value of one forward pass, for the coordinate-
        transform validation/debug tooling (`eyes.inspect_eyepose`) - not
        part of the `EyeDetector` protocol, and never called by `detect()`
        or anything reachable from a normal ranking run.

        Returns `{"padded_input": the exact 640x640 uint8 RGB array the
        model saw, "detection_confidence": the winning anchor's own "is
        this a bird" score, "landmarks_640": {name: (x, y, vis)} in the
        model's own input-pixel space, "landmarks_crop": {name:
        EyeKeypoint} in `crop_rgb`'s space - the same dict `_predict_landmarks`
        returns}`, or None on a degenerate all-zero model output.
        """
        return self._run_model(crop_rgb)

    def _run_model(self, crop_rgb: np.ndarray) -> dict | None:
        """The one place `detect()`/`_predict_landmarks()`/`debug_predict()`
        all funnel through: forward transform, one ONNX Runtime call,
        decode. Keeping this single-sourced is what guarantees
        `debug_predict`'s intermediate values are ALWAYS exactly what a
        real `detect()` call actually used, never a second, potentially
        drifted recomputation."""
        padded, scale, pad_x, pad_y = _letterbox_forward(crop_rgb, INPUT_SIZE)
        tensor = _to_tensor(padded)
        raw = self._session.run(None, {self._input_name: tensor})[0]
        decoded = _decode_best(raw)
        if decoded is None:
            return None
        detection_confidence, landmarks_640 = decoded
        landmarks_crop = {
            name: EyeKeypoint(*_letterbox_inverse(x, y, scale, pad_x, pad_y), confidence=vis)
            for name, (x, y, vis) in landmarks_640.items()
        }
        return {
            "padded_input": padded,
            "detection_confidence": detection_confidence,
            "landmarks_640": landmarks_640,
            "landmarks_crop": landmarks_crop,
        }

    def _anatomically_plausible(self, eye: EyeKeypoint, landmarks: dict[str, EyeKeypoint]) -> bool:
        """See the class/module docstring's "Anatomical plausibility"."""
        beak, head_top = landmarks["beak"], landmarks["head_top"]
        head_scale = max(MIN_HEAD_SCALE_PX, math.hypot(beak.x - head_top.x, beak.y - head_top.y))
        distance = _point_to_segment_distance(eye.x, eye.y, beak.x, beak.y, head_top.x, head_top.y)
        return distance / head_scale <= self.max_head_distance_ratio
