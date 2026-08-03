"""Persisted eye-keypoint results, so the Gallery/Loupe debugging overlay
never re-runs the eye model just to draw what a Classic Vision run already
computed.

One JSON sidecar per cached crop, colocated with it - the same convention
`bird_crop.save_detections`/`read_detections` already use for the subject
detector's boxes, so this needs no new cache directory and no new
identity-matching layer, and it invalidates alongside the crop it describes
(a rebuilt crop cache entry simply gets a fresh sidecar the next time
Classic Vision runs).

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
EYE_CACHE_VERSION = 2
EYE_SUFFIX = ".eye.json"


def eye_cache_path(cache_dir: str | Path, source_path: str | Path) -> Path:
    """Where one image's eye-detector record lives: beside its cached crop,
    keyed the same way (`bird_crop.crop_cache_path`) so it is found the same
    way the crop itself is - by computation, never by scanning."""
    crop = crop_cache_path(cache_dir, source_path)
    return crop.with_name(crop.stem + EYE_SUFFIX)


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


def save_eye_detection(
    cache_dir: str | Path,
    source_path: str | Path,
    subject_crop_size: tuple[int, int],
    detection: EyeDetection,
) -> Path | None:
    """Persist one image's eye-detector result beside its cached crop.

    Failure to write is not fatal - a missing sidecar just means the
    debugging overlay has nothing to draw for that image, same as a subject
    with no recorded detection at all; it must never interrupt a ranking run.
    """
    target = eye_cache_path(cache_dir, source_path)
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


def read_eye_detection(cache_dir: str | Path, source_path: str | Path) -> EyeRecord | None:
    """The last-persisted eye result for an image, or None if Classic Vision
    has never been run on it (or the sidecar is stale/unreadable)."""
    target = eye_cache_path(cache_dir, source_path)
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
    )


def _keypoint_to_dict(keypoint: EyeKeypoint | None) -> dict | None:
    if keypoint is None:
        return None
    return {"x": keypoint.x, "y": keypoint.y, "confidence": keypoint.confidence}


def _keypoint_from_dict(data: dict | None) -> EyeKeypoint | None:
    if not data:
        return None
    return EyeKeypoint(x=float(data["x"]), y=float(data["y"]), confidence=float(data["confidence"]))
