"""SuperAnimal-Quadruped - the primary mammal-eye detector.

Added alongside EyePose-v0/SuperAnimal-Bird when PeakPic's eye-detection
architecture became domain-aware (see `eyes.domains`): EyePose-v0 was
fine-tuned exclusively on CUB-200-2011 bird photographs and SuperAnimal-Bird
on bird-only data too, so neither is semantically an appropriate model for a
mammal - the fact that EyePose-v0 can sometimes place a plausible-looking
point on a monkey's eye (see the accompanying report) is a coincidence of
general keypoint-regression geometry, not evidence it understands mammal
anatomy, and is explicitly NOT treated as license to use it for mammal
scoring - see `eyes.domains`'s module docstring.

Why this model
--------------
Mirrors `superanimal_bird.py`'s own investigation, repeated for the mammal
domain:

- **AP-10K / the Animal Pose dataset** have real `L_eye`/`R_eye` keypoints
  and cover several safari-relevant species (antelope, cheetah, giraffe -
  though not primates), but every available pretrained checkpoint is
  published through MMPose, whose `mmcv` dependency ships NO prebuilt wheel
  on PyPI at all (source-only, as of this investigation) - a fragile,
  slow, easy-to-break build step this project has already declined once for
  the same reason (see `superanimal_bird.py`'s own "Animal Kingdom" bullet).
- **Animal Kingdom-Birds/Mammals (CVPR 2022)** - same MMPose dependency
  problem, same conclusion.
- **SuperAnimal-Quadruped (this one)**, from the same DeepLabCut Model Zoo
  as SuperAnimal-Bird (Ye et al., *Nature Communications* 2024), is
  bird-agnostic and mammal-general: 39 body parts including `left_eye`/
  `right_eye`, `nose`, `upper_jaw`/`lower_jaw`, ears, and a full limb/torso
  skeleton, trained across a broad mixture of wild and domestic quadrupeds
  (horses, dogs, cats, deer, and others - not primate-specific, which is
  the point: the safari brief explicitly asks for a general mammal
  solution, "not optimised only for Colobus monkeys"). Publishes plain
  PyTorch weights, same LGPL-3.0 license already accepted for
  SuperAnimal-Bird, and needs no dependency this project does not already
  have.

Why it does not drag in DeepLabCut, and why HRNet rather than resnet50_gn
---------------------------------------------------------------------------
Same reasoning as `superanimal_bird.py`'s own "Why it does not drag in
DeepLabCut" section - reimplement the published architecture in plain
torch + timm rather than take on the `deeplabcut` package itself. The
published quadruped checkpoint's backbone is `timm`'s `hrnet_w32` (unlike
the bird checkpoint's `resnet50_gn`) - confirmed by loading the published
`pose_model.pth` and matching its parameter names/shapes against a
`timm.create_model('hrnet_w32', ...)` instance: **every** key the checkpoint
has, this reimplementation also has under an identical name, and
`load_state_dict(..., strict=True)` succeeds with nothing missing and
nothing unexpected. Two differences from the bird checkpoint's head, also
confirmed empirically rather than assumed:

- **No location-refinement head.** The bird checkpoint has a second
  `locref_head` for sub-pixel offsets; the quadruped checkpoint has only a
  single `heatmap_head` (a `ConvTranspose2d(32, 39, kernel_size=1)` reading
  directly off HRNet's own highest-resolution branch, which timm's
  `HighResolutionNet.forward_features` exposes as `stages(x)[0]` when its
  classification head (`incre_modules`/`downsamp_modules`/`final_layer`/
  `classifier`) is bypassed - see `_PoseNet.forward` below). Decode is
  therefore a plain heatmap argmax plus the same half-stride pixel-centre
  correction `eyepose_v0._decode_best` uses, without a locref addition term.
- **Model stride is 4, not 8** - HRNet's stem (two stride-2 convolutions)
  downsamples once before the first stage, versus a stride-32 ResNet
  backbone needing a stride-2 transposed convolution to claw back to 8.
  Verified directly: a 256x256 input produces a 64x64 heatmap.
- **Raw heatmap values are not already probabilities** - passing a real
  crop through and checking the peak value at each landmark showed raw
  values around 0.7-0.95 with `sigmoid()` compressing them to a narrower
  ~0.5-0.7 band; `sigmoid` is applied here (matching the bird model's own
  `apply_sigmoid`-equivalent convention of a bounded [0, 1] confidence)
  rather than clipping the raw value, which would not bound it below 0 or
  above 1 for a different input.

How accurate it actually is on this archive
-------------------------------------------
There is no adjudicated sample for this model yet, unlike SuperAnimal-Bird's
own 30-crop hand-labelled study - this project's only available real,
Burst-grouped photographs at the time this model was added were the same
Colobus monkey shoot the domain-aware architecture itself was motivated by
(see the accompanying report), and a single species is exactly what the
safari brief warns against over-fitting to. A qualitative spot-check on that
shoot (not a substitute for a real adjudicated sample) found `left_eye`/
`right_eye` converging on the same point for a profile view - the same
"invisible channel gravitates onto the one real feature" behaviour
`superanimal_bird.py`'s own investigation found and validated the left/right
disagreement gate on - which is why the same gate shape is reused below,
with the same caveat SuperAnimal-Bird's own docstring gives for a threshold
that has NOT been independently fitted: a reasoned starting point, not an
empirically validated one, to be revisited once a real mammal-domain
adjudicated sample exists (see the accompanying report's limitations).

Known failure modes (expected by construction, not yet measured on a
mammal-domain sample)
---------------------------------------------------------------------------
- **Not primate-specialised.** Trained across quadruped species; a Colobus
  monkey's face is mammalian but not quadruped-typical (up on two legs,
  different jaw/brow proportions) - expect lower reliability than on the
  species this checkpoint was actually trained on (deer, canids, felids,
  ungulates - much of typical African safari megafauna).
- **Antler/ear landmarks will not fire on hornless/earless-relative-to-deer
  species** - harmless (those channels are simply never read here), listed
  because `BODYPARTS` still carries them for correct channel indexing.
- **Group scenes and non-quadruped subjects** - the same caveats
  `superanimal_bird.py`'s own docstring gives apply unchanged; `supports()`
  is the same kind of explicit gate (see `eyes.domains` for how the ranking
  mode decides which subjects reach this detector at all).

The weights (~160 MB) are downloaded from Hugging Face once, into
`cache/eye_models/`, the same one-time-download-then-offline shape every
other model in this project already uses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import PROJECT_ROOT
from .detector import EyeDetection, EyeKeypoint, derive_eye_box

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

logger = logging.getLogger(__name__)

# The 39 body parts SuperAnimal-Quadruped predicts, in the checkpoint's own
# channel order (DeepLabCut's `superanimal_quadruped.yaml` project config -
# see the module docstring). Only the eye/nose/neck-base entries are used
# here; the full list is kept for the same reason `superanimal_bird.
# BODYPARTS` keeps all 42 of its own - the order IS the channel mapping.
BODYPARTS: tuple[str, ...] = (
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end", "back_base", "back_end", "back_middle",
    "tail_base", "tail_end", "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw", "back_left_paw", "back_left_thai",
    "back_right_thai", "back_left_knee", "back_right_knee", "back_right_paw",
    "belly_bottom", "body_middle_right", "body_middle_left",
)
LEFT_EYE_INDEX = BODYPARTS.index("left_eye")
RIGHT_EYE_INDEX = BODYPARTS.index("right_eye")
# The head-scale reference for the left/right agreement check below - nose
# and neck_base are the two head/face landmarks most reliably detected
# across head orientations (mirroring SuperAnimal-Bird's own crown/bill
# choice - see that module's docstring), spanning roughly the whole
# head-to-neck length rather than a narrower feature.
NOSE_INDEX = BODYPARTS.index("nose")
NECK_BASE_INDEX = BODYPARTS.index("neck_base")

WEIGHTS_REPO = "mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped"
WEIGHTS_FILENAME = "superanimal_quadruped_pose.pt"
WEIGHTS_URL = f"https://huggingface.co/{WEIGHTS_REPO}/resolve/main/pose_model.pth"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "cache" / "eye_models"

# --- Inference constants, verified empirically against the published
# checkpoint (see the module docstring) rather than transcribed from a
# config file this project does not have a copy of. ---
INPUT_SIZE = 256
MODEL_STRIDE = 4  # heatmap pixel -> input pixel; verified: 256px in -> 64px heatmap out
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Same defaults as SuperAnimal-Bird's own (see that module's docstring for
# where 0.80/0.5 came from on its own, bird-domain adjudicated sample) -
# reused here as a reasoned starting point pending a real mammal-domain
# sample (see this module's own docstring), not re-derived from scratch.
DEFAULT_MIN_CONFIDENCE = 0.80
DEFAULT_EYE_BOX_FRAC = 0.08
MIN_EYE_BOX_PX = 12
DEFAULT_MAX_EYE_DISAGREEMENT = 0.5
MIN_HEAD_SCALE_PX = 3.0


def _build_network(num_bodyparts: int = len(BODYPARTS)) -> "nn.Module":
    """The published architecture, rebuilt from torch + timm - see the
    module docstring's "Why HRNet rather than resnet50_gn" section for how
    each piece here was confirmed against the real checkpoint's own
    parameter names and shapes."""
    import timm
    import torch.nn as nn

    class _PoseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Module()
            # num_classes=1000 (not 0): the published checkpoint carries a
            # (now-unused) classifier/final_layer pair - see the module
            # docstring - and strict loading requires this reimplementation
            # to declare the identical parameter set, even though forward()
            # below never reads them. incre_modules is the one component the
            # checkpoint genuinely does not have (removed by its own
            # publisher, apparently after confirming forward_features never
            # needs it once the classification path is bypassed) and is
            # therefore the one piece explicitly removed here too.
            self.backbone.model = timm.create_model("hrnet_w32", pretrained=False, num_classes=1000)
            del self.backbone.model.incre_modules
            self.backbone.model.incre_modules = None
            head = nn.Module()
            head.heatmap_head = nn.Module()
            head.heatmap_head.model = nn.ConvTranspose2d(32, num_bodyparts, kernel_size=1)
            self.heads = nn.ModuleDict({"bodypart": head})

        def forward(self, x):
            backbone = self.backbone.model
            x = backbone.conv1(x)
            x = backbone.bn1(x)
            x = backbone.act1(x)
            x = backbone.conv2(x)
            x = backbone.bn2(x)
            x = backbone.act2(x)
            # The raw multi-resolution branch list - HRNet's own stem+stages,
            # with the classification-only incre/downsamp/final/classifier
            # path never invoked (see timm.models.hrnet.HighResolutionNet.
            # forward_features, which returns this same list early when
            # incre_modules is None). Branch 0 is the highest-resolution,
            # 32-channel one the published heatmap head was trained against.
            branches = backbone.stages(x)
            return self.heads["bodypart"].heatmap_head.model(branches[0])

    return _PoseNet()


def ensure_weights(weights_dir: str | Path | None = None) -> Path:
    """The local checkpoint path, downloading it once if it is not there yet
    - see `superanimal_bird.ensure_weights`, same shape, same write-then-
    replace discipline."""
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    target = weights_dir / WEIGHTS_FILENAME
    if target.is_file():
        return target

    import urllib.request

    weights_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading the SuperAnimal-Quadruped eye model (~160 MB) to %s", target)
    tmp = target.with_name(target.name + ".part")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


class SuperAnimalQuadrupedEyeDetector:
    """Mammal eye localisation from the DeepLabCut SuperAnimal-Quadruped
    checkpoint. Implements `eyes.detector.EyeDetector`, exactly like
    SuperAnimal-Bird and EyePose-v0 - the mammal domain's fusion pipeline
    (see `eyes.domains`) consumes it through the same interface, not a
    mammal-specific one.
    """

    detector_id = "superanimal-quadruped"

    def __init__(
        self,
        device: str = "cpu",
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        eye_box_frac: float = DEFAULT_EYE_BOX_FRAC,
        max_eye_disagreement: float = DEFAULT_MAX_EYE_DISAGREEMENT,
        weights_dir: str | Path | None = None,
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.min_confidence = min_confidence
        self.eye_box_frac = eye_box_frac
        self.max_eye_disagreement = max_eye_disagreement

        checkpoint_path = ensure_weights(weights_dir)
        network = _build_network()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
        # strict=True on purpose - see superanimal_bird.py's own identical
        # rationale: a silently partial load must never be allowed to look
        # like a working model.
        network.load_state_dict(state_dict, strict=True)
        self.model = network.to(device).eval()

    def supports(self, coco_label: int) -> bool:
        """Which subjects this detector is willing to run on is decided by
        the ranking-mode/domain layer (`eyes.domains`), not by this class
        checking a COCO class itself - unlike EyePose-v0/SuperAnimal-Bird,
        whose `supports()` can lean on `bird_crop.COCO_BIRD_CLASS` because
        COCO genuinely has a bird class. COCO's 80-class vocabulary has no
        "monkey", "lion", "leopard", "cheetah", "antelope", or "buffalo"
        class at all (see the accompanying report's limitations - this is
        exactly why a Colobus monkey was recorded as COCO class 16/"bird"
        upstream in the first place), so gating on `coco_label` here would
        make this detector unusable on most real safari subjects. This
        always returns True; `eyes.domains.DomainProfile`/the Mammals
        ranking strategy's own filter is what decides eligibility instead,
        based on the selected Ranking Mode rather than an unreliable
        upstream species label.
        """
        return True

    def detect(self, subject_crop_rgb: np.ndarray) -> EyeDetection:
        """See `eyes.detector.EyeDetector.detect`. Always returns an
        `EyeDetection`, never `None` - the same contract every other
        detector in this project documents."""
        height, width = (subject_crop_rgb.shape[:2] if subject_crop_rgb is not None else (0, 0))
        if subject_crop_rgb is None or subject_crop_rgb.size == 0:
            return EyeDetection(
                box=(0.0, 0.0, 1.0, 1.0), confidence=0.0, detector_id=self.detector_id, accepted=False
            )

        keypoints = self._predict_keypoints(subject_crop_rgb)
        left = EyeKeypoint(*(float(v) for v in keypoints[LEFT_EYE_INDEX]))
        right = EyeKeypoint(*(float(v) for v in keypoints[RIGHT_EYE_INDEX]))
        primary, other = (left, right) if left.confidence >= right.confidence else (right, left)

        accepted = primary.confidence >= self.min_confidence
        if accepted:
            nose = keypoints[NOSE_INDEX]
            neck_base = keypoints[NECK_BASE_INDEX]
            head_scale = max(MIN_HEAD_SCALE_PX, float(np.hypot(nose[0] - neck_base[0], nose[1] - neck_base[1])))
            disagreement = float(np.hypot(primary.x - other.x, primary.y - other.y)) / head_scale
            accepted = disagreement <= self.max_eye_disagreement

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

    def _predict_keypoints(self, image_rgb: np.ndarray) -> np.ndarray:
        """`(num_bodyparts, 3)` array of `(x, y, score)` in `image_rgb`
        pixels - plain heatmap argmax plus a half-stride centring
        correction, no location-refinement term (see the module docstring
        for why this checkpoint's head has none, unlike SuperAnimal-Bird's)."""
        import cv2

        torch = self._torch
        height, width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmap_t = self.model(tensor)
        heatmap = heatmap_t[0].permute(1, 2, 0).cpu().numpy()
        map_h, map_w, num_parts = heatmap.shape

        flat_argmax = heatmap.reshape(-1, num_parts).argmax(axis=0)
        rows, cols = flat_argmax // map_w, flat_argmax % map_w
        parts = np.arange(num_parts)
        raw_scores = heatmap[rows, cols, parts]
        scores = 1.0 / (1.0 + np.exp(-raw_scores))  # see the module docstring: not already bounded
        x = (cols + 0.5) * MODEL_STRIDE
        y = (rows + 0.5) * MODEL_STRIDE
        x *= width / INPUT_SIZE
        y *= height / INPUT_SIZE
        return np.stack([x, y, scores], axis=1)
