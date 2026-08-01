"""SuperAnimal-Bird - the local, offline bird-eye detector.

Why this model
--------------
There is no free, pretrained model that regresses a *bird eye bounding box*
directly; the honest options are all animal **keypoint** models, and most of
them do not cover birds at all:

- **AP-10K / APT-36K / the Animal Pose dataset** have `L_Eye`/`R_Eye`
  keypoints and good pretrained checkpoints, but their species lists are
  mammals only - no birds whatsoever. Unusable for the primary subject of
  this archive.
- **COCO-Pose (YOLO-pose, Detectron2, MediaPipe)** has eye keypoints, but
  they are *human* eyes. Also, the strongest packaging of it (Ultralytics)
  is AGPL-3.0, which is a real consideration for a distributed desktop app.
- **Animal Kingdom (CVPR 2022)** does cover birds and reports 77.35 PCK@0.05
  on them, but shipping it means taking on MMPose/mmcv, whose version
  pinning against a specific torch/CUDA build is notoriously fragile on
  Windows - the platform this app targets.
- **SuperAnimal-Bird (this one)**, from the DeepLabCut Model Zoo (Ye et al.,
  *Nature Communications* 2024), is bird-specific, has `left_eye` and
  `right_eye` among its 42 body parts, and publishes plain PyTorch weights.

Why it does not drag in DeepLabCut
----------------------------------
The published checkpoint is a *standard* architecture: a `timm` `resnet50_gn`
backbone at output stride 16, plus two `ConvTranspose2d` heads (a 42-channel
heatmap and an 84-channel location refinement). Both are already reachable
from this project's existing dependencies - `timm` is pinned in
pyproject.toml for the model backbones - so the ~40 lines of `_PoseNet` below
load the published weights with **zero missing and zero unexpected keys**,
and DeepLabCut itself (LGPL-3.0, a heavy install with its own GUI stack) is
never a dependency. The architecture and pre/post-processing constants below
are transcribed from DeepLabCut's own published model configuration so this
reproduces their inference exactly rather than approximating it.

How accurate it actually is on this archive
-------------------------------------------
Measured properly: 30 crops drawn with a fixed seed from this project's own
cache, stratified 15/15 by the photographer's own Selected/Rejected verdict,
birds only, then **each one adjudicated by eye** against a 6x zoom of the
predicted eye box - not scored by the model's own confidence, which is what a
confidence histogram alone would have done.

    clearly correct   14 / 30    the box lands on a real eye
    clearly wrong      7 / 30    the box is on the nape, wing, or background
    undecidable        8 / 30    head too small/blurred/dark to judge
    correctly filtered 1 / 30    no eye visible, confidence fell below cutoff

The important result is not the headline number but the **separation by
confidence**, which is sharp:

    correct   n=14   every one scored >= 0.89
    wrong     n= 7   0.41 0.58 0.66 0.69 0.71 0.78 and one outlier at 0.95

So confidence is a usable gate, but only at a far higher cutoff than the
model's scores naively suggest. Sweeping it over the adjudicated sample:

    cutoff 0.30 (naive)   precision 67%   keeps all 14 correct
    cutoff 0.80           precision 93%   keeps all 14 correct

0.80 removes six of the seven errors and costs none of the correct
detections, which is why DEFAULT_MIN_CONFIDENCE is 0.80 and not something
looser. On a 500-crop seeded sample it keeps 71% of the photographer's
Selected images against 53% of their Rejected ones - it discards rejects
faster than keepers, which is the right direction for a culling tool.

Confidence alone is not enough: the soaring-bird investigation
----------------------------------------------------------------
A photographer reported a bird-in-flight frame, wings fully spread and the
head foreshortened toward the camera with no eye actually visible, that
Classic Vision ranked instead of filtering out. Reproduced on this project's
own cache with a closely matching case - a vulture soaring near head-on,
crop 808x313 - which scores **0.947 confidence**, comfortably above
DEFAULT_MIN_CONFIDENCE, while the keypoint sits on the pale forehead/crown
patch between the wing roots. There is no eye there at all; the model has
confidently guessed a plausible-looking location from the visible head
shape alone. This is a real, reproducible, single-image (not group-scene)
case - the "outlier at 0.95" already called out above.

Three follow-up hypotheses were tested and their results are recorded here
so nobody re-investigates them without re-validating:

- **Aspect-ratio distortion (REJECTED after testing).** The naive
  `cv2.resize` to 256x256 does not preserve aspect ratio, unlike
  DeepLabCut's own `top_down_crop` (which expands the short axis to match
  before resizing, so the subject is never stretched) - a real difference
  from the reference implementation, and the vulture's crop is exactly the
  extreme, wide aspect ratio (2.58:1) that distortion would hit hardest.
  Implementing a matching letterbox-pad-then-resize was measured against
  the same 30-image adjudicated sample: it barely moved the vulture's
  confidence (0.947 -> 0.897, still passes), introduced **two new false
  positives** (previously-correctly-filtered wrong detections that crossed
  0.80 after the change) and **one new false negative** (a correct
  detection that dropped below 0.80). Net negative on real data despite
  being architecturally more faithful to the reference - not shipped.
- **Heatmap peakiness / unimodality (REJECTED after testing).** A
  genuinely confident, well-localised detection should show one sharp
  heatmap peak with no comparable secondary peak elsewhere. Measured
  (masking the primary peak's neighbourhood, then finding the next
  spatially distinct local maximum): the vulture's wrong detection is
  *more* unimodal (ratio 92) than several genuinely correct ones (29-66).
  This model is often most confident exactly when it is hallucinating,
  because its training loss always rewards a single sharp answer, visible
  or not. No usable signal here.
- **Local pixel darkness ("is there really a dark pupil here?")
  (REJECTED after testing).** Sampled the darkest pixel in a small window
  around the claimed eye, relative to the crop's own tones. Did not
  separate correct from wrong cleanly (overlapping ranges on both sides) -
  a real avian pupil is dark, but so are plenty of non-eye plumage
  patches and shadows a photograph is full of.
- **Left/right eye-channel disagreement, normalised by head scale
  (VALIDATED - this is what shipped).** A single forward pass predicts
  BOTH `left_eye` and `right_eye` independently. In every genuinely correct
  detection in the sample (near-universally a side-profile shot with only
  one real eye in frame), the two channels converge on almost the same
  pixel - the "invisible" channel's prediction gravitates onto the one
  strongly eye-like feature actually present. In the vulture's case the two
  channels disagree by 28px, a real, substantial split. Normalising that
  separation by the bird's own head scale (the crown<->bill distance, so
  the same absolute pixel gap means less on a huge frame-filling portrait
  than on a small distant bird) gives a clean separation on the sample:

      correct (n=14)   normalised separation always <= 0.34
      wrong   (n=7)     the vulture's own case: 2.56 - 7x any correct value

  At a threshold of `DEFAULT_MAX_EYE_DISAGREEMENT = 0.5`: **100% of the 14
  correct detections are retained**, while 3 of the 7 wrong ones -
  including the vulture - are now additionally excluded, on top of
  whatever the confidence gate alone already caught. This is a second,
  independent, deterministic geometric check (in keeping with Classic
  Vision's own "deterministic, explainable checks" philosophy), not a
  second opaque confidence number - see `SuperAnimalBirdEyeDetector.detect`.

  It is not a proof and does not catch everything: one wrong detection in
  the sample (index 1, separation 0.122) still slips under this gate too.
  Confidently-wrong-under-occlusion remains a known, only partially
  mitigated limitation of this model - see "Known failure modes" below,
  and see the desktop Loupe/Gallery eye-keypoint overlay
  (`review.thumbnails.eye_keypoints_for`), which exists specifically so a
  photographer can visually catch what statistics alone still miss.

Known failure modes (all observed in the sample, only partially mitigated)
---------------------------------------------------------------------------
- **Group scenes.** `bird_crop`'s group-scene policy crops to the box
  enclosing a whole flock. This is a *top-down, single-animal* pose model, so
  a crop containing forty storks is outside its contract and its output there
  is arbitrary. Confidence does not reliably fall in that case.
- **Upstream misdetections pass straight through.** `supports()` gates on the
  COCO class the subject detector recorded, so when that is itself wrong - a
  fruit bat detected as a bird - nothing downstream catches it, and an eye is
  confidently placed on fur.
- **Small, distant or motion-blurred heads** can score high while the
  keypoint sits on the nape or the background.
- **Confidently-wrong-under-occlusion partially, not fully, mitigated.** The
  left/right disagreement check above catches the reported soaring-bird
  case and similar wide-disagreement guesses, but a hallucinated pair that
  happens to also agree with each other still passes both gates.
- **Never trust it on a non-bird.** On a tiger crop it put both "eyes" on the
  ear at 0.67/0.90. `supports()` restricts this detector to birds and the
  caller turns that into its own explicit reject reason (see
  `ranking.filters.UNSUPPORTED_SUBJECT`) rather than pretending the eye was
  merely not visible.

The weights (~103 MB) are downloaded from Hugging Face once, into
`cache/eye_models/`, and everything after that is fully local - the same
one-time-download-then-offline shape the COCO detector's torchvision weights
and BioCLIP's checkpoint already use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..bird_crop import COCO_BIRD_CLASS
from ..config import PROJECT_ROOT
from .detector import EyeDetection, EyeKeypoint, derive_eye_box

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

logger = logging.getLogger(__name__)

# The 42 body parts SuperAnimal-Bird predicts, in the checkpoint's own channel
# order (DeepLabCut's `superanimal_bird.yaml` project config). Only the two eye
# entries are used here; the full list is kept because the order *is* the
# channel mapping - an abbreviated version would silently mis-index if the two
# indices below were ever recomputed from it.
BODYPARTS: tuple[str, ...] = (
    "back", "bill", "belly", "breast", "crown", "forehead", "left_eye", "left_leg",
    "left_wing_tip", "left_wrist", "nape", "right_eye", "right_leg", "right_wing_tip",
    "right_wrist", "tail_tip", "throat", "neck", "tail_left", "tail_right", "upper_spine",
    "upper_half_spine", "lower_half_spine", "right_foot", "left_foot", "left_half_chest",
    "right_half_chest", "chin", "left_tibia", "right_tibia", "lower_spine",
    "upper_half_neck", "lower_half_neck", "left_chest", "right_chest", "upper_neck",
    "left_wing_shoulder", "left_wing_elbow", "right_wing_shoulder", "right_wing_elbow",
    "upper_cere", "lower_cere",
)
LEFT_EYE_INDEX = BODYPARTS.index("left_eye")
RIGHT_EYE_INDEX = BODYPARTS.index("right_eye")
# The head-scale reference for the left/right agreement check below - two
# landmarks that are almost always confidently detected (a bill in
# particular is large, high-contrast, and present in essentially every
# pose) and whose distance scales with the bird's own size in the crop.
CROWN_INDEX = BODYPARTS.index("crown")
BILL_INDEX = BODYPARTS.index("bill")

# Where the published weights live, and where they are cached locally. Under
# cache/ with the crop cache and the review thumbnails: derived data that can
# always be re-fetched, never something a backup must preserve.
WEIGHTS_REPO = "DeepLabCut/DeepLabCutModelZoo-SuperAnimal-Bird"
WEIGHTS_FILENAME = "superanimal_bird_resnet_50.pt"
WEIGHTS_URL = f"https://huggingface.co/{WEIGHTS_REPO}/resolve/main/{WEIGHTS_FILENAME}"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "cache" / "eye_models"

# --- Inference constants, transcribed from DeepLabCut's model configuration ---
# (`modelzoo/model_configs/resnet_50.yaml` and `config/base/aug_top_down.yaml`).
# These are not tunable knobs: they define what the trained weights expect, so
# changing one silently degrades accuracy rather than trading it off.
INPUT_SIZE = 256                  # top_down_crop width/height
BACKBONE_OUTPUT_STRIDE = 16       # backbone.output_stride
HEAD_UPSAMPLE = 2                 # the heatmap head's ConvTranspose2d stride
MODEL_STRIDE = BACKBONE_OUTPUT_STRIDE // HEAD_UPSAMPLE  # heatmap pixel -> input pixel
LOCREF_STD = 7.2801               # HeatmapPredictor.locref_std
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Below this keypoint confidence the eye is treated as not visible at all.
#
# 0.80, not the ~0.30 a glance at the score distribution would suggest: on a
# hand-adjudicated sample every correct detection scored >= 0.89, while six of
# the seven wrong ones scored below 0.80, so this cutoff raises precision from
# 67% to 93% without losing a single correct detection (see the module
# docstring). It is deliberately strict - a wrong eye box silently corrupts the
# eye-sharpness metric, which carries the heaviest default weight in Classic
# Vision, whereas an over-filtered frame merely shows up as Unranked and can
# still be reviewed by hand.
#
# Exposed as a tunable parameter on the Classic Vision strategy rather than
# hard-coded into a filter; lower it toward 0.5 to rank more of a folder at the
# cost of more misplaced eye boxes.
DEFAULT_MIN_CONFIDENCE = 0.80

# The eye box's side length, as a fraction of the subject crop's *shorter*
# side. The model gives a point, not an extent (see the `detector` module
# docstring), so the region has to be derived - and it has to be derived from
# something that scales with the bird, or the same absolute box would cover a
# whole head on a distant subject and one iris on a frame-filling portrait.
# 0.08 lands a little wider than the eye itself on a typical crop, which is
# what a sharpness measure wants: enough pixels for a Laplacian to be
# meaningful, tight enough that the surrounding plumage does not dominate it.
DEFAULT_EYE_BOX_FRAC = 0.08

# However the fraction works out, an eye box smaller than this many pixels a
# side carries too little signal for a variance-of-Laplacian to mean anything.
MIN_EYE_BOX_PX = 12

# The second, independent gate: how far the LESS confident eye channel may
# disagree with the more confident one, as a fraction of the crown<->bill
# head-scale reference, before the pair is distrusted entirely. See the
# module docstring's "Confidence is not enough" section - validated on the
# same 30-image adjudicated sample as DEFAULT_MIN_CONFIDENCE: 0.5 keeps
# every correct detection (max observed 0.34) while additionally excluding
# the soaring-bird false positive that motivated this check (2.56) and two
# other wrong detections a confidence-only gate could not catch.
DEFAULT_MAX_EYE_DISAGREEMENT = 0.5

# Floor on the crown<->bill reference distance used to normalise the
# disagreement above, purely to keep the division well-defined. Never
# actually reached in the adjudicated sample (smallest observed was 6.6px);
# this only guards the pathological case of crown and bill themselves
# collapsing onto (almost) the same pixel.
MIN_HEAD_SCALE_PX = 3.0


def _build_network(num_bodyparts: int = len(BODYPARTS)) -> "nn.Module":
    """The published architecture, rebuilt from torch + timm.

    Module names deliberately mirror DeepLabCut's own (`backbone.model.*`,
    `heads.bodypart.*`), because that is what the checkpoint's state dict is
    keyed by - matching them is what lets the weights load strictly, with
    nothing missing and nothing left over.
    """
    import timm
    import torch.nn as nn

    class _PoseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.model = timm.create_model(
                "resnet50_gn",
                pretrained=False,
                output_stride=BACKBONE_OUTPUT_STRIDE,
                num_classes=0,
                global_pool="",
            )
            head = nn.Module()
            head.heatmap_head = nn.Module()
            head.heatmap_head.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(2048, num_bodyparts, kernel_size=3, stride=HEAD_UPSAMPLE)
            )
            head.locref_head = nn.Module()
            head.locref_head.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(2048, 2 * num_bodyparts, kernel_size=3, stride=HEAD_UPSAMPLE)
            )
            self.heads = nn.ModuleDict({"bodypart": head})

        def forward(self, x):
            features = self.backbone.model(x)
            bodypart = self.heads["bodypart"]
            return (
                bodypart.heatmap_head.deconv_layers(features),
                bodypart.locref_head.deconv_layers(features),
            )

    return _PoseNet()


def ensure_weights(weights_dir: str | Path | None = None) -> Path:
    """The local checkpoint path, downloading it once if it is not there yet.

    Kept separate from the detector's `__init__` so a caller (the desktop
    app, a test) can pre-fetch or verify the download without constructing a
    model, and so the ~103 MB transfer is an obvious, named step rather than
    a surprise inside a constructor.
    """
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    target = weights_dir / WEIGHTS_FILENAME
    if target.is_file():
        return target

    import urllib.request

    weights_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading the SuperAnimal-Bird eye model (~103 MB) to %s", target)
    # Downloaded to a temporary name and renamed on success, so an interrupted
    # transfer can never leave a truncated file that later loads as a corrupt
    # checkpoint - the same write-then-replace discipline the crop cache uses.
    tmp = target.with_name(target.name + ".part")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


class SuperAnimalBirdEyeDetector:
    """Bird eye localisation from the DeepLabCut SuperAnimal-Bird checkpoint.

    Implements `eyes.detector.EyeDetector`. torch/timm are imported inside
    `__init__` (never at module import), matching `BirdDetector` and
    `BioClipSpeciesClassifier`, so listing the available ranking strategies
    costs nothing.
    """

    detector_id = "superanimal-bird"

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
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        # strict=True on purpose: a silently partial load would produce a model
        # with randomly initialised heads that still returns plausible-looking
        # confidences, which is precisely the failure this project cannot
        # detect downstream. Better to fail loudly at construction.
        network.load_state_dict(state_dict, strict=True)
        self.model = network.to(device).eval()

    def supports(self, coco_label: int) -> bool:
        """Birds only - see the module docstring's tiger example for what
        happens when this model is asked about anything else."""
        return int(coco_label) == COCO_BIRD_CLASS

    def detect(self, subject_crop_rgb: np.ndarray) -> EyeDetection:
        """The detector's full answer for one subject crop: the primary
        (more confident) eye's box/confidence/keypoint, BOTH raw eye
        channels, and whether the result should be trusted at all.

        Always returns an `EyeDetection` - never `None` - even for a crop
        where nothing should be trusted, because that raw data is exactly
        what a debugging overlay needs to show a photographer why an image
        was filtered out (see `review.thumbnails.eye_keypoints_for`).
        `EyeFilter` is what turns `.accepted` into a pass/fail decision.

        Only one eye is required to accept a detection, which is the point:
        wildlife photography - birds especially - is overwhelmingly
        side-profile, where exactly one eye faces the camera and the other
        genuinely is not in the frame. Requiring both would reject the
        majority of perfectly good frames. But BOTH channels are always
        computed (one forward pass produces every body part), and the
        second, independent gate below compares them - see the module
        docstring's "Confidence is not enough" section for why.
        """
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
            crown = keypoints[CROWN_INDEX]
            bill = keypoints[BILL_INDEX]
            head_scale = max(MIN_HEAD_SCALE_PX, float(np.hypot(crown[0] - bill[0], crown[1] - bill[1])))
            disagreement = float(np.hypot(primary.x - other.x, primary.y - other.y)) / head_scale
            accepted = disagreement <= self.max_eye_disagreement

        # Clamped to the crop, and only after clamping is the box guaranteed
        # non-degenerate - an eye detected hard against an edge would otherwise
        # yield a zero-width region the caller cannot crop. Computed
        # regardless of `accepted`, so a rejected image still has a real box
        # to show in a debugging overlay. See eyes.detector.derive_eye_box,
        # shared with eyepose_v0.py so every keypoint-based detector's box
        # means the same thing.
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
        """(num_bodyparts, 3) array of (x, y, score) in `image_rgb` pixels.

        The heatmap -> keypoint decode (argmax, then the location-refinement
        offset, then the half-stride pixel-centre correction) reproduces
        DeepLabCut's `HeatmapPredictor` exactly; `apply_sigmoid` is false and
        scores are clipped to [0, 1], as its published configuration for this
        checkpoint specifies.
        """
        import cv2

        torch = self._torch
        height, width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmap_t, locref_t = self.model(tensor)
        heatmap = heatmap_t[0].permute(1, 2, 0).cpu().numpy()
        map_h, map_w, num_parts = heatmap.shape
        locref = (
            locref_t[0].permute(1, 2, 0).reshape(map_h, map_w, num_parts, 2).cpu().numpy()
            * LOCREF_STD
        )

        flat_argmax = heatmap.reshape(-1, num_parts).argmax(axis=0)
        rows, cols = flat_argmax // map_w, flat_argmax % map_w
        parts = np.arange(num_parts)
        scores = np.clip(heatmap[rows, cols, parts], 0.0, 1.0)
        # +0.5 * stride puts the keypoint at the centre of the heatmap cell
        # rather than its corner, before the sub-pixel locref offset is added.
        x = (cols * MODEL_STRIDE + 0.5 * MODEL_STRIDE + locref[rows, cols, parts, 0])
        y = (rows * MODEL_STRIDE + 0.5 * MODEL_STRIDE + locref[rows, cols, parts, 1])
        # Back from the 256x256 network input to the caller's own crop pixels.
        x *= width / INPUT_SIZE
        y *= height / INPUT_SIZE
        return np.stack([x, y, scores], axis=1)
