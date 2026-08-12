"""Animal detection and crop caching.

This is the "subject-centered input" preprocessing phase: instead of feeding
the full frame to the model, we detect the animal once per image, crop tightly
around it (small safety margin, aspect ratio preserved), and cache the crop so
training never re-detects or re-decodes RAW.

The module, its classes, and its functions keep "bird" in their names for
historical reasons (the project started bird-only) — crop selection is now
completely class-agnostic (see "COCO is a localization tool" below).

**COCO is a localization tool, not the authority on what is in the frame.**
The detector's one job here is to answer "where might the subject be?" — it
is never asked "is this a valid wildlife subject?". Its class labels are
therefore recorded for display and cataloguing (see `detection_category`),
and are read by nothing in the crop path: no class is preferred, and no
class disqualifies a box from becoming the crop. This is not a stylistic
preference, it is a correctness requirement. COCO has no primate class at
all, and the classes it does have are routinely wrong on real wildlife -
gating crops on them means an image full of monkeys is treated exactly like
an empty frame. Measured on this project's own 5,986-image archive under the
old class gate: 1,506 images contained confident detections (median best
confidence 0.886) that produced no crop at all, purely because COCO had
labelled them "person".

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
- If the detector returns no boxes AT ALL, the full frame is cached as the
  fallback, so every image still yields an input and is never re-detected.
  That is the only cause of the fallback - see select_best_detection.

**This cache is Vision Cache infrastructure, not a training-only
optimization.** It is the shared image source for every Computer Vision
consumer - AI-model training, Classic Vision (both eye-detection backends),
and any future vision module (species classification, sharpness analysis,
composition, ...) - never re-decoding the same RAW twice for repeated work
on the same image. Its one job is to hand every consumer the best crop it
can, at the resolution and quality *they* need; it must never become a
resolution ceiling on its own. Concretely: `CropParams.max_side` defaults to
`None` (the crop's own resolution, uncapped) rather than a fixed number, and
any consumer that genuinely needs a smaller input (training's own
`RawImageLoader` in particular) does its own resize at load time - see
`raw_io.RawImageLoader._letterbox`, which already worked this way before
`max_side` became configurable, so no consumer changed. `image_format`/
`jpeg_quality` are similarly consumer-agnostic knobs on the cache itself, not
per-model settings.

Bump CROP_CACHE_VERSION whenever the crop algorithm, its defaults, or the set
of accepted detection classes changes, so a stale cache is detected instead of
silently reused. v1 = bird only; v2 = SUPPORTED_ANIMAL_CLASSES; v3 = area-
dominant selection among survivors; v4 = group-scene handling (see "Crop
selection policy" below); v5 = Vision Cache - `max_side`/`image_format`/
`jpeg_quality` became configurable cache parameters instead of a hardcoded
1024px PNG, so a v4-or-earlier cache (fixed 1024px cap, PNG) is a different,
lower-quality artifact that must never be silently mistaken for a v5 one -
see `build_cache`'s own `crop_params.json` mismatch check, and
`crop_cache_path`'s format-dependent file extension, which is the other half
of that same guarantee (a reader configured for the new default format
simply never finds an old-format file to misread in the first place).
v6/v7 = successive individual-selection policies (see "Crop selection policy"
below); v8 = individual selection became a weighted centre/area/confidence
score with no confidence floor, so a v7-or-earlier cache's `selected`
detection was chosen by a materially different rule and must not be read as
if this one had produced it. v9 = the COCO class gate was removed from crop
selection entirely and the area term became a scaled/capped size score, so a
v8 cache can differ in BOTH which box was selected and whether one was
selected at all.

Crop selection policy
----------------------
Faster R-CNN's own postprocessing already does the filtering that is NOT
policy: it drops anything below its own score threshold and runs per-class
non-maximum suppression. Everything that survives that is a candidate,
whatever class it was labelled - see "COCO is a localization tool" above.
Candidates still often number more than one (two classes competing for the
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

- **v4 (superseded on its own, still the group-scene rule below
  `group_scene_threshold`): group scenes crop to the whole group, not one
  member of it.** Wildlife photography routinely and *intentionally* frames a
  flock, a herd, a colony - a group of animals is the subject, not any single
  one of them. Picking "the best" individual detection out of a flock of
  forty birds (by area or by confidence, it does not matter which) crops to
  one bird and discards the photograph's actual subject. So when the number
  of surviving detections reaches `group_scene_threshold` (default 10),
  individual selection is skipped entirely: the crop target becomes the
  smallest box enclosing every surviving detection, then the normal margin
  and downstream crop pipeline apply exactly as they do for a single subject.
  The full-frame fallback (see "If no supported animal is detected" above)
  still only applies when *nothing* was detected - a group scene never falls
  back to the full frame, even when the group only occupies a small part of
  it: the whole point is a tight crop around the actual subject, individual
  or group. Group-scene selection itself is unchanged by v6 below - it is
  never reached from v3's area-first policy or v6's confidence-first one:
  both apply only below `group_scene_threshold`.

- **v6 (superseded on its own): confidence dominates; area only breaks a
  near-tie.** v3's area-first policy had a real failure mode found during
  the EyePose investigation (`docs/EyePose_Investigation_Phase_1.md`): a
  real bird detected at 0.998 confidence lost the crop to an unrelated,
  much lower confidence (0.458) false-positive detection - mislabelled
  "cow" - simply because the false positive's box happened to be larger.
  Feeding EyePose a crop of the wrong region then produced a
  confidently-reported eye that was not a bird's eye at all (the "no
  visible eye should exist" symptom from the original bug report). v6
  inverted v3 outright: the highest-confidence surviving detection won,
  unless another detection's confidence was within a fraction of the
  winner's own to trigger an area tie-break. This fixed the "cow" case,
  but reopened v2's own original failure mode: a small, clean, very
  confident detection could again beat a larger, legitimately-real subject
  whose confidence was merely *good* rather than exceptional (motion blur,
  an awkward pose, partial occlusion) - exactly why v2 was abandoned for v3
  in the first place. Superseded after one investigation cycle once this
  tension was recognised (`docs/EyePose_Investigation_Phase_1.md`'s
  "Detection selection policy" discussion) - a pure ordering, in *either*
  direction, cannot protect against both failure modes at once: proven
  algebraically (a linear weighted score can't either - the two failure
  modes impose contradictory constraints on the weights) rather than just
  observed.

- **v7 (superseded): an absolute confidence gate, then area decides among
  what survives.** Any candidate whose confidence was below
  `min_crop_confidence` (an absolute floor, default 0.6) was discarded
  outright; among whatever remained, the largest won. Confidence answered
  "is this candidate trustworthy at all", area answered "which trustworthy
  candidate is the subject", and the two were never blended. This fixed the
  "cow" case and restored v3's intent, but kept the property every version
  from v2 onward shared: **nothing in the comparison knew where in the
  frame a candidate was.** A large, confident detection in a corner still
  beat the animal the photographer had centred, because position was simply
  not an input. It also had a second, sharper failure mode of its own - a
  frame in which candidates WERE detected but none cleared 0.6 produced no
  selection at all, and fell through to the full-frame fallback, which is
  the same outcome as "nothing was detected" despite the detector having
  found something in every one of those frames. On this project's own
  archive that was 1,582 of 5,986 images.

- **v8 (superseded by v9's two refinements, otherwise intact): one weighted
  score over three normalised signals;
  candidates are never rejected.** Where a subject sits in the frame is the
  photographer's own compositional statement, and it was the missing input.
  Every candidate is scored on a fixed 50/30/20 blend of centre proximity,
  relative area and detector confidence (see `selection_score`), and the
  highest score wins. Two consequences are deliberate and are the point of
  the version:

  1. **Position dominates.** At 50%, centre proximity outweighs area and
     confidence combined, so a centred subject beats a larger or more
     confident one off to the side - which is what "the photographer framed
     it there on purpose" means.
  2. **There is no reliability gate any more.** If the detector returned
     candidates at all, exactly one of them is selected, always. The
     full-frame fallback now has exactly one cause - zero candidates -
     rather than two that were indistinguishable downstream. Confidence
     still participates, at 20%, as one term among three rather than as a
     veto.

  Unlike v3-v7 this IS a weighted score. The earlier versions' objection to
  blending was that confidence and area answer different questions and
  should not be summed; that objection stands for those two signals alone,
  and is exactly why neither of them is allowed to dominate here. Position
  is a third, independent signal that outranks both, and the blend is what
  lets a candidate that is merely good on all three beat one that is
  extreme on a single axis. Detector- and class-agnostic by construction:
  nothing here reads a detection's `label`.

  Group scenes are unaffected - that branch is evaluated first and never
  reaches the individual scoring path at all.

- **v9 (current): the class gate is gone, and the size term is scaled.**
  v8 removed the confidence floor but left a second, larger filter standing
  upstream of it: `BirdDetector` only offered `select_best_detection` the
  boxes whose COCO class was in SUPPORTED_ANIMAL_CLASSES. That gate, not
  confidence, was what actually produced most of v7's 1,582 fallbacks - on
  re-examination 1,506 of them held confident boxes labelled "person", and
  every one of them was discarded unseen. See "COCO is a localization tool"
  at the top of this module for why gating on COCO's opinion is wrong in
  principle as well as in measurement. v9 therefore scores EVERY detection
  the model returns above `conf_threshold`, whatever its label, and records
  every one of them so the runners-up are visible. Two changes, both
  deliberate:

  1. **No class filter anywhere in the crop path.** `BirdDetector.classes`
     and `.catalogue_classes` are gone rather than left inert, because a
     parameter that silently no longer gates anything is worse than no
     parameter. SUPPORTED_ANIMAL_CLASSES/CATALOGUED_CLASSES survive as what
     they always should have been on their own: the review app's DISPLAY
     taxonomy (see `detection_category`), read by nothing in this path.
  2. **The area term became a scaled, capped size score.** Raw area
     fraction is a terrible 0-1 signal for wildlife: a perfectly framed
     bird occupies ~6.5% of the frame, so the raw fraction handed almost
     every real subject ~0.065 out of a possible 1.0 and the 30% area term
     was, in practice, nearly dead weight. `subject_size_score` scales by
     10 and caps at 1.0, so 10% of the frame or more is a full score and
     the term does real work over the range wildlife photographs actually
     occupy. The same scaled value is what `ranking.crop_sharpness` scores
     its 20% size term on, so "subject size" means one thing in both
     layers.

  Known and accepted consequence, flagged rather than silently absorbed:
  because every class now counts toward the candidate list, a cluttered
  frame reaches `group_scene_threshold` (10) more easily than it did when
  only animals counted. Group-scene handling itself is deliberately
  UNCHANGED here (see `_group_scene_detection`); if that threshold proves
  wrong under the new candidate set it is its own decision, not a silent
  side effect of this one.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .profiling import PROFILE

# ---------------------------------------------------------------------------
# COCO class ids - a DISPLAY TAXONOMY ONLY.
#
# Nothing below this point gates crop selection. These names exist so the
# review app can say "a bird was detected here" instead of "class 16"; see
# `detection_category` and the module docstring's "COCO is a localization
# tool" note. A future reader looking for the crop-eligibility rule will not
# find one here, because there isn't one - `select_best_detection` scores
# every box the detector returns and never reads `label`.
# ---------------------------------------------------------------------------

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

# The remaining COCO animal classes. Named for the same display reason as
# WILDLIFE_CLASSES above - a horse/cow/sheep in frame is the photo's subject
# just as much as a zebra is, and the review app should be able to say so.
DOMESTIC_ANIMAL_CLASSES: dict[int, str] = {
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
}

SUPPORTED_ANIMAL_CLASSES: dict[int, str] = {**WILDLIFE_CLASSES, **DOMESTIC_ANIMAL_CLASSES}

# A person in frame: a birder, a ranger, a researcher handling an animal are
# all common in a wildlife archive, and "who else is in this photo" is
# exactly the kind of thing the review app's filtering/search should be able
# to answer.
#
# A person IS crop-eligible, like every other class. It was not, through v8,
# on the reasoning that a bystander must never steal the crop from the real
# subject - correct as an aim, but enforced with the wrong instrument. COCO
# does not have a primate class, so a monkey is frequently labelled "person",
# and the gate meant to exclude bystanders was in fact excluding the subject.
# The composition-based `selection_score` handles the bystander case on its
# own terms instead: someone standing at the edge of the frame loses to the
# centred animal on position, which is the thing that actually distinguishes
# them.
COCO_PERSON_CLASS = 1

# Every class this project has a display name for. Read by the review app's
# structured subject metadata (see detection_category) and by nothing in the
# crop path. A detection whose class is absent here is still a perfectly
# valid crop candidate; it simply has no category to show.
CATALOGUED_CLASSES: dict[int, str] = {COCO_PERSON_CLASS: "person", **SUPPORTED_ANIMAL_CLASSES}


def coco_class_name(label: int) -> str:
    """Human-readable name for a catalogued COCO class (for logging)."""
    return CATALOGUED_CLASSES.get(int(label), f"class {int(label)}")


# ---------------------------------------------------------------------------
# Subject category taxonomy - the review app's own vocabulary, broader than
# any one detector's class list, so a future detector (trained beyond COCO)
# can populate categories this one structurally cannot recognize without
# changing the taxonomy or anything downstream of it (the review app's
# filters, statistics, and eventual "smart collections" all read the
# category string, never a raw COCO class id).
# ---------------------------------------------------------------------------

DETECTION_CATEGORY_BIRD = "bird"
DETECTION_CATEGORY_MAMMAL = "mammal"
DETECTION_CATEGORY_REPTILE = "reptile"
DETECTION_CATEGORY_AMPHIBIAN = "amphibian"
DETECTION_CATEGORY_FISH = "fish"
DETECTION_CATEGORY_INSECT = "insect"
DETECTION_CATEGORY_ARACHNID = "arachnid"
DETECTION_CATEGORY_HUMAN = "human"

# Every category PickLikeMe knows how to talk about. Order is display order
# (roughly: wildlife first, by how often this project's archives feature
# them, then human last).
DETECTION_CATEGORIES: tuple[str, ...] = (
    DETECTION_CATEGORY_BIRD,
    DETECTION_CATEGORY_MAMMAL,
    DETECTION_CATEGORY_REPTILE,
    DETECTION_CATEGORY_AMPHIBIAN,
    DETECTION_CATEGORY_FISH,
    DETECTION_CATEGORY_INSECT,
    DETECTION_CATEGORY_ARACHNID,
    DETECTION_CATEGORY_HUMAN,
)

# COCO class id -> category. Only bird/mammal/human are actually reachable
# with the current COCO-pretrained detector - COCO has no reptile, amphibian,
# fish, insect or arachnid class at all, full stop, so this project cannot
# honestly report those categories until a different (wildlife-specific)
# detector backs it. The taxonomy above already has a slot for each, ready
# for that day; this mapping is the one place that would grow to fill them.
COCO_CLASS_CATEGORY: dict[int, str] = {
    COCO_PERSON_CLASS: DETECTION_CATEGORY_HUMAN,
    COCO_BIRD_CLASS: DETECTION_CATEGORY_BIRD,
    **{class_id: DETECTION_CATEGORY_MAMMAL for class_id in DOMESTIC_ANIMAL_CLASSES},
    **{class_id: DETECTION_CATEGORY_MAMMAL for class_id in WILDLIFE_CLASSES if class_id != COCO_BIRD_CLASS},
}


def detection_category(label: int) -> str | None:
    """The taxonomy category (see DETECTION_CATEGORIES) for a COCO class id,
    or None if this detector does not catalogue it at all."""
    return COCO_CLASS_CATEGORY.get(int(label))


CROP_CACHE_VERSION = "v9"
CROP_PARAMS_FILENAME = "crop_params.json"

# Cache entries live in cache_dir/<first 2 hex chars of digest>/<digest><ext>,
# <ext> depending on image_format (see crop_cache_path). Two shard characters
# gives 256 shards: ~215 files per shard at 55k images, which keeps NTFS
# directory operations fast without creating a deep tree.
CACHE_SHARD_CHARS = 2

# Vision Cache format - see CropParams.image_format/jpeg_quality and the
# module docstring's "Vision Cache infrastructure" note. JPEG at a high
# quality rather than lossless PNG: measured on this project's own real
# cache, JPEG q98 runs ~3x smaller than PNG for the same pixels (see
# docs/vision_cache.md), which matters once the cache stores crops at their
# original resolution instead of a capped 1024px.
DEFAULT_IMAGE_FORMAT = "jpeg"
DEFAULT_JPEG_QUALITY = 98
IMAGE_FORMAT_EXTENSIONS: dict[str, str] = {"jpeg": ".jpg", "png": ".png"}

# RETAINED, BUT NO LONGER AFFECTS CROP SELECTION (v8 - see the module
# docstring). This was the absolute confidence floor `select_best_detection`
# applied before selecting; v8 removed that gate outright, because a frame
# with real candidates that all fell below it produced no selection at all
# and became indistinguishable from a frame with nothing in it. Confidence
# now participates as a 20% term in `selection_score` instead of as a veto.
#
# The constant and its `CropParams`/`BirdDetector` fields are kept because
# they are threaded through several ranking strategies' own user-facing
# parameters (`ranking.classic`'s `crop_confidence_threshold`, its generated
# parameter dialog, and the `preprocess`/`inspect-crops` CLIs) - removing
# them would change those surfaces, which this change deliberately does not
# touch. Nothing reads the value for selection any more.
DEFAULT_MIN_CROP_CONFIDENCE = 0.6

# The fixed weights of the selection score. Deliberately module-level
# constants rather than CropParams fields: for this experiment they are not
# a per-run knob, and adding them to CropParams would put them into the
# cache-identity comparison and every strategy's parameter dialog for no
# benefit. They sum to 1.0, so `selection_score` is itself in [0, 1].
SELECTION_WEIGHT_CENTER = 0.50
SELECTION_WEIGHT_AREA = 0.30
SELECTION_WEIGHT_CONFIDENCE = 0.20

# How hard `subject_size_score` scales a raw area fraction before capping at
# 1.0. At 10, a subject filling 10% of the frame scores a full 1.0 and
# everything below is linear (1% -> 0.10, 5% -> 0.50, 6.5% -> 0.65).
#
# Chosen from this project's own archive rather than by taste: across 4,404
# real detected subjects the median relative size is 0.0654 and the 90th
# percentile 0.3183. Scored on the raw fraction, half of all real subjects
# contributed under 0.07 of a possible 1.0 to whatever weighted that term,
# making it nearly inert; the cap costs only the top decile, which is
# already unambiguously "fills the frame" and does not need to be ranked
# against itself. See the module docstring's v9 entry.
SIZE_SCORE_SCALE = 10.0

# At or above this many surviving detections, the image is treated as a group
# scene: the crop target becomes the box enclosing all of them, not a single
# individual. See select_best_detection() and the module docstring's "Crop
# selection policy" section.
DEFAULT_GROUP_SCENE_THRESHOLD = 10


@dataclass(frozen=True)
class CropParams:
    """Parameters that define how a crop cache was built. Recorded alongside
    the cache (write_crop_params/read_crop_params) so a mismatched
    configuration is detected rather than silently reused - build_cache
    compares the stored value against the requested one on every run and
    refuses (SystemExit, "pass --force to rebuild") when they differ. Adding
    a field here automatically joins that comparison, which is how
    `image_format`/`jpeg_quality`/max_side's new default are protected
    without any extra version-tracking code - see the module docstring's
    "Vision Cache infrastructure" and CROP_CACHE_VERSION notes.
    """

    margin_frac: float = 0.05          # small safety margin around the tight box
    conf_threshold: float = 0.30       # min detection confidence to accept a detection
    # Cap the cached crop's long side, in pixels - None (the default) means
    # NO cap: the crop is cached at its own, original resolution. Vision
    # Cache quality must never be reduced for a consumer that never asked
    # for that; a consumer that genuinely wants a smaller input (training's
    # RawImageLoader) resizes at load time instead - see the module
    # docstring. Set an explicit int to cap disk usage at the cost of detail,
    # e.g. for a quick experiment or a disk-constrained machine.
    max_side: int | None = None
    min_crop_confidence: float = DEFAULT_MIN_CROP_CONFIDENCE  # absolute reliability floor for select_best_detection
    group_scene_threshold: int = DEFAULT_GROUP_SCENE_THRESHOLD  # >= this many detections -> group scene
    detector: str = "fasterrcnn_resnet50_fpn_v2"
    # "jpeg" (default) or "png". See DEFAULT_IMAGE_FORMAT/crop_cache_path -
    # the cache file's own extension depends on this, so a format change
    # cannot silently collide with an old-format entry on disk.
    image_format: str = DEFAULT_IMAGE_FORMAT
    # Only meaningful when image_format == "jpeg"; ignored for "png" (always
    # lossless). 98 was chosen by measuring real cached crops: visually and
    # numerically close to lossless, at roughly a third of PNG's size - see
    # docs/vision_cache.md. Exposed here, not hardcoded, so a future need
    # (e.g. 100 for a specific analysis, or a lower value to shrink an
    # existing archive) is a parameter, not a code change.
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
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


def _clamp01(value: float) -> float:
    """Numerical safety for every normalised term in `selection_score` -
    each is mathematically already in [0, 1], so this only ever absorbs
    float error or a malformed box, never real signal."""
    if not value == value:  # NaN
        return 0.0
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def center_proximity(
    box: tuple[float, float, float, float], source_size: tuple[int, int]
) -> float:
    """How close `box`'s centre is to the frame's centre, in [0, 1].

    1.0 at the exact centre, 0.0 at a corner, and linear in Euclidean
    distance between them - so 0.5 means "halfway from the centre to a
    corner". Normalised by the centre-to-corner distance, which is the
    largest distance any in-frame point can have, making the value
    resolution-independent: the same relative position in a 6000x4000 frame
    and a 600x400 one scores identically.

    Returns 0.0 for a degenerate frame (zero or negative dimensions) rather
    than dividing by zero - an unusable frame gives every candidate the same
    0.0, which lets the other two terms decide instead of crashing the run.
    """
    width, height = source_size
    if width <= 0 or height <= 0:
        return 0.0
    center_x = float(width) / 2.0
    center_y = float(height) / 2.0
    box_center_x = (box[0] + box[2]) / 2.0
    box_center_y = (box[1] + box[3]) / 2.0
    distance = math.hypot(box_center_x - center_x, box_center_y - center_y)
    max_distance = math.hypot(center_x, center_y)  # centre -> corner
    if max_distance <= 0.0:
        return 1.0
    return _clamp01(1.0 - distance / max_distance)


def relative_box_area(
    box: tuple[float, float, float, float], source_size: tuple[int, int]
) -> float:
    """`box`'s area as a fraction of the whole frame, in [0, 1].

    The same quantity `ranking.metrics.normalized_subject_size` computes for
    the ranking layer, deliberately reimplemented here in these three lines
    rather than imported: `ranking` imports `bird_crop` (every strategy
    does), so importing back the other way would be a circular import. The
    two must stay in agreement - they are the same concept, "relative
    subject size", asked at two different layers.
    """
    width, height = source_size
    frame_area = float(width) * float(height)
    if frame_area <= 0.0:
        return 0.0
    return _clamp01(box_area(box) / frame_area)


def subject_size_score(area_fraction: float) -> float:
    """A raw area fraction turned into a usable 0-1 size signal:
    `clamp01(SIZE_SCORE_SCALE * area_fraction)`.

    THE single definition of "how big is this subject, as a score" for the
    whole project - `selection_score`'s 30% area term and
    `ranking.crop_sharpness`'s 20% size term both call it, so subject size
    cannot come to mean two different things at two different layers. Takes
    a fraction rather than a box so the ranking layer, which stores the
    fraction and no longer has the box, can reach the identical curve.

    See SIZE_SCORE_SCALE for why the scale is 10 and why capping is not a
    loss of signal.
    """
    return _clamp01(SIZE_SCORE_SCALE * float(area_fraction))


def relative_size_score(
    box: tuple[float, float, float, float], source_size: tuple[int, int]
) -> float:
    """`subject_size_score` of `box`'s own area fraction - the box-shaped
    convenience for callers that still have the box."""
    return subject_size_score(relative_box_area(box, source_size))


def selection_score(detection: "BirdDetection", source_size: tuple[int, int]) -> float:
    """How strongly `detection` looks like the photograph's intended subject,
    in [0, 1]. The v9 crop-selection policy, in one place:

        0.50 * center_proximity + 0.30 * size_score + 0.20 * confidence

    Every term is independently normalised to [0, 1] and the weights sum to
    1.0, so the result is directly comparable across images of any
    resolution. See the module docstring's "v8"/"v9" entries for why position
    is weighted above the other two combined, why there is no confidence
    floor that can reject a candidate outright, and why the size term is the
    scaled `subject_size_score` rather than the raw area fraction.

    **This function never reads `detection.label`, and must never start to.**
    It is an ordering over candidate regions, not a judgement about what they
    contain - see the module docstring's "COCO is a localization tool" note.
    """
    return (
        SELECTION_WEIGHT_CENTER * center_proximity(detection.box, source_size)
        + SELECTION_WEIGHT_AREA * relative_size_score(detection.box, source_size)
        + SELECTION_WEIGHT_CONFIDENCE * _clamp01(detection.score)
    )


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


def downscale_long_side(image: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale (never upscale) so the longer side is at most max_side,
    preserving aspect ratio. `max_side=None` means no cap at all - the image
    is returned unchanged, at its own full resolution (see
    CropParams.max_side and the module docstring's "Vision Cache
    infrastructure" note for why that is the default)."""
    if max_side is None:
        return image
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

def crop_cache_path(
    cache_dir: str | Path, source_path: str | Path, *, image_format: str = DEFAULT_IMAGE_FORMAT
) -> Path:
    """Deterministic cache file for a source image, keyed by its absolute path.

    THE single place in the codebase that constructs a cache path: every read
    and every write goes through here, so the layout can never diverge between
    producer and consumer.

    Sharded into 256 subdirectories by the first two hex characters of the
    digest, because a flat directory holding 55k+ entries degrades directory
    operations on NTFS. The path is always *computed* from the digest — the
    cache is never scanned, globbed, or walked to find an entry.

    The digest itself is independent of crop parameters (those are recorded
    in crop_params.json; rebuilding the cache overwrites in place) - but the
    file EXTENSION depends on `image_format`, so a cache rebuilt under a
    different format (see CropParams.image_format) writes to a different
    path rather than silently overwriting - or being silently mistaken for -
    an entry in the old format. A caller reading the cache with the current
    default format therefore naturally gets a cache miss (not a wrong- format
    read) for anything still on disk from before the Vision Cache's format
    became configurable - see the module docstring's CROP_CACHE_VERSION note.
    """
    resolved = str(Path(source_path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:20]
    extension = IMAGE_FORMAT_EXTENSIONS.get(image_format, IMAGE_FORMAT_EXTENSIONS[DEFAULT_IMAGE_FORMAT])
    return Path(cache_dir) / digest[:CACHE_SHARD_CHARS] / f"{digest}{extension}"


def write_crop_params(cache_dir: str | Path, params: CropParams) -> Path:
    with PROFILE.stage("metadata write"):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / CROP_PARAMS_FILENAME
        path.write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
        return path


def read_crop_params(cache_dir: str | Path) -> CropParams | None:
    """The params an existing cache was built with, or None if there is no
    cache yet.

    Silently drops any key in the stored JSON that is not a current
    `CropParams` field, rather than raising `TypeError` - an older cache's
    `crop_params.json` may have a field name from a since-renamed generation
    (`area_tie_frac` -> `confidence_tie_frac` in v6 -> `min_crop_confidence`
    in v7; see the module docstring's policy history), and a raw
    `CropParams(**data)` unpack would crash on an unrecognised key before
    `build_cache`'s own version-mismatch check ever runs. Dropping it here
    does not weaken that check: the stored `version` field (still read
    normally) will not match the current `CROP_CACHE_VERSION`, so
    `build_cache`'s `existing != params` comparison still refuses a stale
    cache exactly as it refuses any other mismatch, with the same
    "pass --force to rebuild" message rather than a stack trace.
    """
    path = Path(cache_dir) / CROP_PARAMS_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    known_fields = {f.name for f in fields(CropParams)}
    return CropParams(**{key: value for key, value in data.items() if key in known_fields})


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


def save_crop_png(cache_path: Path, crop_rgb: np.ndarray, *, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> None:
    """Write an RGB crop to the cache (stored BGR so cv2.imread + the
    loader's BGR->RGB conversion round-trips correctly).

    Format is read from `cache_path`'s own extension - `.jpg`/`.jpeg` writes
    a JPEG at `jpeg_quality`, anything else (`.png`) writes lossless PNG -
    matching whatever `crop_cache_path(..., image_format=...)` decided the
    path should be, so this function never has to be told the format twice.
    Name kept for backward compatibility (many existing call sites and tests
    reference it) even though the default format is no longer literally PNG
    - see CropParams.image_format.
    """
    with PROFILE.stage("image encode + write"):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp" + cache_path.suffix)
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if cache_path.suffix.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        else:
            cv2.imwrite(str(tmp), bgr)
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

    @property
    def category(self) -> str | None:
        """This detection's taxonomy category (see DETECTION_CATEGORIES), or
        None if its class is not catalogued at all."""
        return detection_category(self.label)


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
    source_size: tuple[int, int],
    group_scene_threshold: int = DEFAULT_GROUP_SCENE_THRESHOLD,
) -> BirdDetection | None:
    """The crop target build_crop should use, chosen from every detection the
    model returned above its confidence threshold (and, upstream of this, the
    detector's own per-class NMS - see the module docstring's "Crop selection
    policy" section).

    `candidates` is deliberately unfiltered by class: whatever COCO called
    each box, it competes here on composition alone. See the module
    docstring's "COCO is a localization tool" note for why gating on the
    label is a correctness bug rather than a tuning choice.

    `source_size` is the FULL FRAME's (width, height) in pixels - required,
    because centre proximity is meaningless without it, and a silent default
    would mean a caller that forgot it got a quietly different policy rather
    than an error.

    Two policies, chosen by how many detections survived:

    - **Fewer than `group_scene_threshold`: the highest `selection_score`
      wins.** Every candidate is scored on the fixed 50/30/20 blend of centre
      proximity, scaled size and confidence, and the best one is selected.
      No candidate is ever rejected: if there are candidates at all, exactly
      one of them comes back. See the module docstring's "v8"/"v9" entries
      for why position outweighs the other two combined, and why both the
      confidence floor and the class gate this used to apply were removed
      rather than relaxed. Detector- and class-agnostic: nothing here reads
      `label`, so this applies identically to a bird, a person, a species
      COCO has never heard of, or a misclassification.

    - **`group_scene_threshold` or more: the image is a group scene.**
      UNCHANGED by v8. Picking one detection out of a flock, a herd or a
      colony would crop to a single animal and discard the photograph's
      actual subject, so no individual detection is selected at all - the
      target becomes the smallest box enclosing every surviving detection
      (see `_group_scene_detection`). The normal crop margin and downstream
      pipeline still apply to that box exactly as they would to a single
      detection; there is no full-frame fallback here, because a group is
      still a real, locatable subject. This branch is evaluated first and
      never reaches the individual scoring path, so `selection_score` and
      `source_size` do not participate in it at all.

    The single source of truth for subject selection: BirdDetector.detect_best_bird
    and detect_with_all both call this rather than each implementing their own
    comparison, so the two can never disagree about the crop target.

    Returns None for one reason only: `candidates` is empty, i.e. the
    detector found nothing at all. That is the sole cause of `build_crop`'s
    full-frame fallback. Through v7 there were three indistinguishable
    causes - nothing detected, nothing above the confidence floor, or
    nothing of an accepted COCO class - so a frame the detector HAD found
    something in was filed identically to an empty one (1,582 of 5,986
    images on this project's own archive, 1,506 of them class-gated).
    """
    if not candidates:
        return None

    if len(candidates) >= group_scene_threshold:
        return _group_scene_detection(candidates)

    # Never a filter, only an ordering - so a non-empty candidate list always
    # yields exactly one winner.
    return max(candidates, key=lambda detection: selection_score(detection, source_size))


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
    """COCO-pretrained Faster R-CNN v2, used purely as a region proposer.

    **There is no class filter.** Every box the model returns at or above
    `conf_threshold` is a crop candidate and is recorded, whatever COCO
    labelled it. The `classes`/`catalogue_classes` parameters this had
    through v8 are gone, not defaulted-to-everything: they gated crop
    eligibility on COCO's opinion of the species, which discarded 1,506 real
    subjects on this project's own archive and is structurally incapable of
    handling an animal COCO cannot name (a primate, most importantly). See
    the module docstring's "COCO is a localization tool" note and its v9
    entry.

    `conf_threshold` remains, and is a different thing: it is the model's own
    noise floor, deciding whether a box exists at all rather than whether an
    existing box is allowed to win. Once a box exists it always competes.

    torch/torchvision are imported lazily so that modules which only need the
    bbox math or cache-path helpers (e.g. RawImageLoader) don't pull the heavy
    detection stack.
    """

    def __init__(
        self,
        device: str = "cpu",
        conf_threshold: float = 0.30,
        min_crop_confidence: float = DEFAULT_MIN_CROP_CONFIDENCE,
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
        # Accepted and stored, but NO LONGER USED for crop selection - see
        # DEFAULT_MIN_CROP_CONFIDENCE for why the parameter is retained.
        self.min_crop_confidence = min_crop_confidence
        self.group_scene_threshold = group_scene_threshold
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()

    def detect_with_all(self, image_rgb: np.ndarray) -> "tuple[BirdDetection | None, list[BirdDetection]]":
        """(winner, every detection) from a **single** forward pass.

        Exists so a caller that wants the runners-up - the false-negative
        diagnostic overlay, or the review app's subject cataloguing - does
        not have to run inference a second time.

        The two returned things are now drawn from the SAME list: the winner
        is `select_best_detection`'s pick from exactly the detections also
        returned as the second element. That equality is the point, and it is
        what makes the Loupe's overlay honest - every yellow runner-up box
        really was in contention for the green one. Through v8 the recorded
        list was a superset of the eligible one, so an image could show boxes
        that had been silently barred from winning, which is precisely how a
        frame full of confident detections came to look like an empty frame.
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

            # Confidence only - no class filter (see the class docstring).
            candidates: list[BirdDetection] = [
                BirdDetection(box=tuple(float(v) for v in box), score=float(score), label=int(label))
                for box, label, score in zip(boxes, labels, scores)
                if score >= self.conf_threshold
            ]
            # (width, height) - the frame the boxes are measured against,
            # so centre proximity is computed against the real image.
            height, width = image_rgb.shape[:2]
            best = select_best_detection(candidates, (width, height), self.group_scene_threshold)
        return best, candidates

    def detect_best_bird(self, image_rgb: np.ndarray) -> BirdDetection | None:
        """The detection build_crop should crop to, or None only if the
        detector returned no boxes at all.

        This is the single source of truth for subject selection: everything
        that needs a box goes through here (or through detect_with_all, which
        this delegates to, so the two entry points always agree). What "best"
        means is entirely select_best_detection()'s policy - the highest
        `selection_score` (50% centre proximity, 30% scaled size, 20%
        confidence) below `group_scene_threshold` detections, the enclosing
        box of the whole group at or above it; see the module docstring's
        "Crop selection policy" section.
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
