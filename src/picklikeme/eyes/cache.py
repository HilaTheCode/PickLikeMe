"""Persisted eye-keypoint results, so the Gallery/Loupe debugging overlay
never re-runs the eye model just to draw what a ranking run already
computed.

One JSON sidecar per (cached crop, algorithm), colocated with the crop - the
same convention `bird_crop.save_detections`/`read_detections` use for the
subject detector's boxes, extended with one more key: which STRATEGY
produced this record (see `strategy_id` on every function below), so a
second algorithm run on the same image gets its OWN sidecar rather than
overwriting the first's.

Why this is keyed by strategy_id, not detector_id
----------------------------------------------------
This cache predates PeakPic having more than one eye-detection strategy at
all, and originally had exactly one slot per image - whichever detector ran
on it last silently overwrote whatever an earlier run had recorded. That
became a real, user-visible bug once multiple ranking strategies existed
(Birds/Mammals/Fusion, all independently selectable and independently
cached in every OTHER respect - see `sidecar.py`'s "one file per analysis
module" rule): the Loupe's Elements/Boxes overlay could show a DIFFERENT
run's eye box than the one the photographer had selected as Color Source,
because both strategies were sharing the exact same on-disk slot for the
same image.

`detector_id` (`EyeDetection.detector_id`, still recorded in the payload
below as metadata) is not fine-grained enough to fix this by itself: the
shared Fusion layer's own detector_id (`eyes.fusion.FUSION_DETECTOR_ID`,
`"fusion-v1"`) is identical across the Birds-Fusion, Mammals-Fusion and
Combined strategies despite each running a genuinely different set of
sub-detectors/weights - keying on it would still let one Fusion strategy's
run overwrite another's. `strategy_id` (`ranking.base.StrategyInfo.
strategy_id`, e.g. `"classic-vision-fusion-mammals"`) is the one identifier
that is actually unique per registered strategy, exactly like it already is
for `sidecar.strategy_ranking_path`'s ranking CSVs - so this cache now uses
the same key.

Retention: exactly one record per (image, strategy) - the strategy's own
latest completed run, nothing older. A caller re-running the same strategy
on the same image simply overwrites that strategy's own file (still via the
atomic write-then-replace below), which is correct: only the latest run per
algorithm needs to be retained (see `ranking.classic`'s own module
docstring - Classic Vision is stateless/deterministic, re-running it is
exactly "replace my own last answer with a new one", never "append a
history").

Written whenever the eye detector actually ran on an image, regardless of
whether the result was accepted - see `eyes.detector.EyeDetection.accepted`.
That is deliberate: a REJECTED image's raw keypoints are exactly what a
photographer investigating a filtering decision needs to see, so this is not
a "boxes are found" cache the way `analyzer.detections.DetectionCache` is -
it is a "here is everything the model said, trust it or not" record.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..bird_crop import crop_cache_path
from .detector import EyeDetection, EyeKeypoint

logger = logging.getLogger(__name__)

# Bumped whenever the payload shape changes, so a row written by an older
# version is never misread as the current one - the same discipline
# bird_crop.CROP_CACHE_VERSION and analyzer.detections.DETECTION_CACHE_VERSION
# already apply to their own sidecars.
# v2 (EyePose Investigation Phase 1, Part 2): added head_confidence, the
# independent "is a real head present" signal - see eyepose_v0.head_visible.
# v3 (Image Inspector landmark overlay): added beak/head_top/left_shoulder/
# right_shoulder - the rest of EyePose-v0's own six-landmark set, previously
# computed but discarded before EyeDetection ever returned them.
# The v3 -> per-strategy path change (see the module docstring) is not its
# own version bump: the payload SHAPE did not change, only where it lives -
# an old, pre-this-change `.eye.json` simply becomes unreachable (a cache
# miss, regenerated on the next ranking run), never misread as a different
# strategy's result.
EYE_CACHE_VERSION = 3
EYE_SUFFIX = ".eye.json"
_UNSAFE_STRATEGY_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def eye_cache_path(cache_dir: str | Path, source_path: str | Path, strategy_id: str) -> Path:
    """Where one (image, strategy)'s eye-detector record lives: beside the
    image's cached crop, keyed the same way (`bird_crop.crop_cache_path`) so
    it is found the same way the crop itself is - by computation, never by
    scanning - plus `strategy_id` as an extra filename segment, so a second
    strategy's run on the same image never collides with the first's (see
    the module docstring). `strategy_id` is a registered `StrategyInfo.
    strategy_id` (already filesystem-safe kebab-case, e.g.
    "classic-vision-fusion-mammals") - the character filter is defensive,
    not load-bearing, so a future strategy id can never accidentally escape
    the crop cache's own shard directory.
    """
    crop = crop_cache_path(cache_dir, source_path)
    safe_strategy_id = _UNSAFE_STRATEGY_ID_CHARS.sub("_", strategy_id)
    return crop.with_name(f"{crop.stem}.{safe_strategy_id}{EYE_SUFFIX}")


@dataclass(frozen=True)
class EyeRecord:
    """A persisted eye result, read back for the debugging overlay - the
    on-disk mirror of `eyes.detector.EyeDetection`, plus the subject-crop
    size the coordinates are relative to (needed to scale them onto a
    thumbnail or the Loupe's own, differently-sized, view of the crop)."""

    detector_id: str
    subject_crop_size: tuple[int, int]  # (width, height), matching bird_crop's own convention
    accepted: bool
    box: tuple[float, float, float, float]
    confidence: float
    left: EyeKeypoint | None
    right: EyeKeypoint | None
    # The independent "is a real head present at all" signal - see
    # eyes.detector.EyeDetection.head_confidence's own docstring. None for a
    # backend (or an older cached row) that doesn't have one.
    head_confidence: float | None = None
    # The rest of EyePose-v0's six-landmark set - see
    # eyes.detector.EyeDetection's own fields of the same names. None for a
    # backend (or a pre-v3 cached row) that doesn't have them.
    beak: EyeKeypoint | None = None
    head_top: EyeKeypoint | None = None
    left_shoulder: EyeKeypoint | None = None
    right_shoulder: EyeKeypoint | None = None


def save_eye_detection(
    cache_dir: str | Path,
    source_path: str | Path,
    subject_crop_size: tuple[int, int],
    detection: EyeDetection,
    *,
    strategy_id: str,
) -> Path | None:
    """Persist one (image, strategy)'s eye-detector result beside its cached
    crop - `strategy_id` is the calling ranking strategy's own
    `StrategyInfo.strategy_id` (see the module docstring for why this, not
    `detection.detector_id`, is the cache key), so this strategy's own
    latest run is the only thing this call can ever overwrite.

    Failure to write is not fatal - a missing sidecar just means the
    debugging overlay has nothing to draw for that image, same as a subject
    with no recorded detection at all; it must never interrupt a ranking run.
    """
    target = eye_cache_path(cache_dir, source_path, strategy_id)
    payload = {
        "version": EYE_CACHE_VERSION,
        "detector_id": detection.detector_id,
        "subject_crop_size": list(subject_crop_size),
        "accepted": detection.accepted,
        "box": list(detection.box),
        "confidence": detection.confidence,
        "left": _keypoint_to_dict(detection.left),
        "right": _keypoint_to_dict(detection.right),
        "head_confidence": detection.head_confidence,
        "beak": _keypoint_to_dict(detection.beak),
        "head_top": _keypoint_to_dict(detection.head_top),
        "left_shoulder": _keypoint_to_dict(detection.left_shoulder),
        "right_shoulder": _keypoint_to_dict(detection.right_shoulder),
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        logger.debug("Could not write eye cache for %s: %s", source_path, exc)
        return None
    return target


def read_eye_detection(cache_dir: str | Path, source_path: str | Path, strategy_id: str) -> EyeRecord | None:
    """`strategy_id`'s own last-persisted eye result for an image, or None
    if that strategy has never run on it (or its sidecar is stale/
    unreadable). Never a different strategy's result - see the module
    docstring for why this must be requested explicitly rather than reading
    "whichever one is there"."""
    target = eye_cache_path(cache_dir, source_path, strategy_id)
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read eye cache for %s: %s", source_path, exc)
        return None
    if payload.get("version") != EYE_CACHE_VERSION:
        return None

    size = payload.get("subject_crop_size") or (0, 0)
    box = payload.get("box") or (0.0, 0.0, 0.0, 0.0)
    head_confidence = payload.get("head_confidence")
    return EyeRecord(
        detector_id=payload.get("detector_id", ""),
        subject_crop_size=(int(size[0]), int(size[1])),
        accepted=bool(payload.get("accepted", False)),
        box=tuple(float(v) for v in box),
        confidence=float(payload.get("confidence", 0.0)),
        left=_keypoint_from_dict(payload.get("left")),
        right=_keypoint_from_dict(payload.get("right")),
        head_confidence=float(head_confidence) if head_confidence is not None else None,
        beak=_keypoint_from_dict(payload.get("beak")),
        head_top=_keypoint_from_dict(payload.get("head_top")),
        left_shoulder=_keypoint_from_dict(payload.get("left_shoulder")),
        right_shoulder=_keypoint_from_dict(payload.get("right_shoulder")),
    )


def _keypoint_to_dict(keypoint: EyeKeypoint | None) -> dict | None:
    if keypoint is None:
        return None
    return {"x": keypoint.x, "y": keypoint.y, "confidence": keypoint.confidence}


def _keypoint_from_dict(data: dict | None) -> EyeKeypoint | None:
    if not data:
        return None
    return EyeKeypoint(x=float(data["x"]), y=float(data["y"]), confidence=float(data["confidence"]))
