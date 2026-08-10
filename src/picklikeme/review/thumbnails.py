"""Thumbnails for the gallery - the analyzer's renderers, on demand.

Nothing here draws anything. `contactsheets.build_thumbnail` and
`contactsheets.annotate_thumbnail` already produce exactly the images the
review page needs, and the detector boxes they draw must look identical to the
ones in an analysis report; a second renderer would drift from the first the
moment either changed.

Two differences from how the analyzer calls them:

- **The cache is persistent and shared**, not per-run. An analysis writes
  thumbnails into its own output directory, which is replaced every run; a
  review has no run directory and would otherwise re-decode every RAW on every
  launch. Keyed on resolved path + size + version, so one directory safely
  serves every folder ever reviewed.
- **Built one at a time**, when the browser asks. A folder of several thousand
  images would stall for minutes if they were all rendered up front, and the
  browser only ever requests the ones actually scrolled into view.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from ..config import DEFAULT_CROP_CACHE_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Alongside cache/crops and cache/analyzer_detections.db: derived data that can
# always be rebuilt, so it lives in the cache tree rather than with the shoot.
REVIEW_THUMBNAIL_CACHE = PROJECT_ROOT / "cache" / "review_thumbs"
# The Lightbox's full-size preview cache - see review_preview() below.
REVIEW_PREVIEW_CACHE = PROJECT_ROOT / "cache" / "review_previews"

# Large enough that a distant bird and its detector box stay legible, matching
# the analyzer's own default (AnalysisConfig.thumbnail_size).
REVIEW_THUMBNAIL_SIZE = 400

# Bumped whenever a change makes previously cached previews wrong.
REVIEW_PREVIEW_CACHE_VERSION = 1

# A library of hundreds of thousands of photos would otherwise let this cache
# grow forever - each entry is a full-size JPEG, not a thumbnail. Configurable
# via `picklikeme review --preview-cache-max-gb` (see review/cli.py); the
# default is a reasonable desktop budget, not a hard technical limit.
DEFAULT_PREVIEW_CACHE_MAX_BYTES = 20 * 1024**3  # 20 GB

# How many new cache WRITES pass between budget checks. A cache HIT (the
# overwhelming majority of requests once a folder has been browsed once)
# never triggers this at all - only a miss, which just paid for a real RAW
# decode, so an occasional extra directory walk is proportionally cheap
# there. Checking on every single write would mean walking a cache that may
# hold hundreds of thousands of files on every one of them; checking this
# rarely instead bounds the possible overshoot to a handful of files' worth
# of bytes - negligible against a multi-GB budget - in exchange for that cost
# being paid only once every N misses.
PREVIEW_CACHE_SWEEP_INTERVAL_WRITES = 25

_detection_cache = None
_writes_since_sweep: dict[Path, int] = {}


def _preview_cache_path(cache_dir: Path, image_path: str) -> Path:
    # Same convention as contactsheets._thumbnail_cache_path: resolved path +
    # a version stamp, no mtime-based invalidation. These are camera RAW
    # files a photographer reviews, not documents someone re-saves in place,
    # so that tradeoff - already accepted for the square thumbnails - applies
    # here unchanged. mtime is instead repurposed for LRU eviction (see
    # _enforce_cache_budget): free to do, since nothing here uses it for
    # invalidation.
    key = f"{Path(image_path).resolve()}|v{REVIEW_PREVIEW_CACHE_VERSION}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return cache_dir / digest[:2] / f"{digest}.jpg"


def _cache_entries(cache_dir: Path) -> list[tuple[Path, int, float]]:
    """(path, size_bytes, mtime) for every cached preview - the raw material
    for both size accounting and LRU eviction. Not resolved via the digest
    scheme (unlike a normal read/write) because this one operation
    genuinely needs to see every entry at once; every other function in this
    module still computes its path directly and never walks the directory.
    """
    entries: list[tuple[Path, int, float]] = []
    if not cache_dir.is_dir():
        return entries
    for entry in cache_dir.glob("*/*.jpg"):
        try:
            stat = entry.stat()
        except OSError:
            continue
        entries.append((entry, stat.st_size, stat.st_mtime))
    return entries


def _enforce_cache_budget(cache_dir: Path, max_bytes: int) -> int:
    """Delete the least-recently-used cached previews until `cache_dir` is
    back under `max_bytes`. "Recently used" is each file's own mtime -
    touched on every cache hit (see review_preview) - not filesystem atime,
    which Windows disables updating by default and so cannot be relied on
    for this. Returns the number of files removed (0 if already under
    budget, which is the common case and costs one directory walk).
    """
    entries = _cache_entries(cache_dir)
    total = sum(size for _, size, _ in entries)
    if total <= max_bytes:
        return 0
    entries.sort(key=lambda entry: entry[2])  # oldest (least recently used) first
    removed = 0
    freed = 0
    for path, size, _ in entries:
        if total <= max_bytes:
            break
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        freed += size
        removed += 1
    if removed:
        logger.info(
            "Preview cache over budget (%.1f GB > %.1f GB): removed %d least-recently-used file(s), freed %.1f GB",
            (total + freed) / 1024**3,
            max_bytes / 1024**3,
            removed,
            freed / 1024**3,
        )
    return removed


def review_preview(
    image_path: str,
    *,
    cache_dir: Path | None = None,
    max_bytes: int = DEFAULT_PREVIEW_CACHE_MAX_BYTES,
) -> Path:
    """A cached full-size preview JPEG for one image - the Lightbox's own
    persistent counterpart to the analysis report's `/preview` endpoint
    (analyzer/server.py's `_serve_preview`, `Cache-Control: no-store`).

    That endpoint is deliberately never cached, on disk or in the browser,
    because a stale preview is a real correctness concern for a long-lived
    static report someone might revisit days later. A review session is a
    different shape of use entirely: a photographer flips back and forth
    across the same few dozen images while comparing a burst, and every one
    of those revisits was re-running a full `rawpy` read of the RAW's
    embedded thumbnail plus a PIL JPEG re-encode - real CPU work, identical
    every time for the same file, and the actual bottleneck behind the
    Lightbox feeling slower the longer a session runs (profiled by reading
    `load_source_image`/`_serve_preview`, not guessed at). This persists that
    result once. Raises whatever `load_source_image` raises; the caller
    reports that as a normal per-image failure, same as before this cache
    existed.

    Bounded to `max_bytes` (default DEFAULT_PREVIEW_CACHE_MAX_BYTES) by
    deleting the least-recently-used entries once it grows past that - see
    _enforce_cache_budget. A hit refreshes this file's own "recently used"
    standing (its mtime) so it is not immediately in line for eviction the
    next time the budget is checked.
    """
    from ..analyzer.contactsheets import load_source_image

    cache_dir = cache_dir or REVIEW_PREVIEW_CACHE
    target = _preview_cache_path(cache_dir, image_path)
    # A 0-byte file is what an interrupted write (crash, force-quit, a full
    # disk) leaves behind now that the write itself is atomic (see below) -
    # this only ever catches a file left over from BEFORE that fix, since an
    # atomic write can never itself be observed half-finished. Treated as a
    # cache miss rather than trusted, so a stale corrupt entry heals itself
    # on the next read instead of being served forever - see the Loupe
    # reliability investigation this responds to (a null QPixmap from a
    # truncated cached JPEG, with nothing upstream ever re-checking it).
    if target.exists() and target.stat().st_size > 0:
        try:
            os.utime(target, None)  # mark "recently used" - see _enforce_cache_budget
        except OSError:
            pass
        return target

    image = load_source_image(image_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name and renamed into place on success - the
    # same write-then-replace discipline the Vision Cache's own crop/
    # detection writers already use, so a process killed mid-write can never
    # leave a partial JPEG parked at the real cache path (the failure mode
    # this function's own 0-byte check above exists to heal from).
    tmp = target.with_name(target.name + ".tmp")
    image.save(tmp, format="JPEG", quality=92)
    tmp.replace(target)

    count = _writes_since_sweep.get(cache_dir, 0) + 1
    if count >= PREVIEW_CACHE_SWEEP_INTERVAL_WRITES:
        _enforce_cache_budget(cache_dir, max_bytes)
        count = 0
    _writes_since_sweep[cache_dir] = count
    return target


def _detections():
    """The shared detector-box cache, opened once.

    `allow_detect=False` at every call site: the review application must never
    run the detector, so a box either already exists from preprocessing or the
    overlay is simply not drawn.
    """
    global _detection_cache
    if _detection_cache is None:
        from ..analyzer.detections import DEFAULT_DETECTIONS_DB, DetectionCache

        _detection_cache = DetectionCache(DEFAULT_DETECTIONS_DB, DEFAULT_CROP_CACHE_DIR)
    return _detection_cache


def review_thumbnail(
    image_path: str,
    *,
    with_boxes: bool = False,
    strategy_id: str | None = None,
    size: int = REVIEW_THUMBNAIL_SIZE,
    cache_dir: Path | None = None,
) -> Path | None:
    """A cached thumbnail for one image, or None when it cannot be rendered.

    With `with_boxes`, returns the overlaid copy - a separate file beside the
    plain one, so the page can toggle between them without either being
    rebuilt. Falls back to the plain thumbnail when the image has no recorded
    detections, which is the honest answer: there are no boxes to draw.

    Also draws the eye detector's result (see `eye_keypoints_for`) for
    `strategy_id` - the currently selected ranking strategy/Color Source
    (`None` when no strategy applies, e.g. the AI model, which has no eye
    detector at all - see `eye_keypoints_for`'s own docstring), never
    "whichever strategy happened to run last" (see `eyes.cache`'s module
    docstring for the bug this fixes). The overlaid file's own cache key
    reflects both whether eye data was available AND which strategy it came
    from (see `contactsheets.annotated_thumbnail_path`'s `has_eye`/
    `eye_strategy_id`), so switching Color Source never serves a different
    strategy's stale overlaid thumbnail back.
    """
    from ..analyzer.contactsheets import annotate_thumbnail, annotated_thumbnail_path, build_thumbnail

    cache_dir = cache_dir or REVIEW_THUMBNAIL_CACHE
    plain = build_thumbnail(image_path, size, cache_dir)
    if plain is None or not with_boxes:
        return plain

    eye = eye_keypoints_for(image_path, strategy_id) if strategy_id else None
    overlaid = annotated_thumbnail_path(cache_dir, image_path, size, has_eye=eye is not None, eye_strategy_id=strategy_id)
    if overlaid.exists():
        return overlaid

    try:
        record = _detections().get(image_path, allow_detect=False)
    except Exception as exc:  # noqa: BLE001 - an unreadable cache is not fatal
        logger.debug("No detections for %s: %s", image_path, exc)
        return plain

    return annotate_thumbnail(plain, record, overlaid, size, eye=eye) or plain


def detected_category_for(image_path: str) -> str | None:
    """The subject category (see bird_crop.DETECTION_CATEGORIES) already
    recorded for this image, or None if nothing was ever detected/recorded
    for it. Reuses the same shared, allow_detect=False detection cache
    review_thumbnail's overlay already reads from - review must never run
    the detector itself, only ever read what preprocessing (or an earlier
    false-negative diagnostic backfill) already computed.
    """
    try:
        record = _detections().get(image_path, allow_detect=False)
    except Exception as exc:  # noqa: BLE001 - an unreadable cache is not fatal
        logger.debug("No detections for %s: %s", image_path, exc)
        return None
    selected = record.selected
    return selected.category if selected is not None else None


def detection_boxes_for(image_path: str) -> dict | None:
    """Detector boxes for one image, in full-frame pixel coordinates - the
    same read-only data review_thumbnail's with_boxes overlay draws onto a
    thumbnail file, for a caller (the desktop Loupe) that wants to draw its
    own overlay on a full-size preview instead. None when there is no
    recorded source frame size, i.e. nothing to scale boxes against -
    review must never run the detector itself (allow_detect=False), only
    ever read what preprocessing already computed, same as
    detected_category_for above.
    """
    try:
        record = _detections().get(image_path, allow_detect=False)
    except Exception as exc:  # noqa: BLE001 - an unreadable cache is not fatal
        logger.debug("No detections for %s: %s", image_path, exc)
        return None
    if record.source_size is None:
        return None
    return {
        "source_size": record.source_size,
        "selected": record.selected.as_dict() if record.selected is not None else None,
        "others": [box.as_dict() for box in record.others],
        # The crop's own rectangle (tight detection box grown by
        # CropParams.margin_frac - see DetectionRecord.expanded_box's own
        # docstring) - the region that was actually cropped and cached, as
        # opposed to `selected`, the raw detector box before the margin was
        # applied. None on a full-frame fallback, same as `selected`.
        "expanded_box": list(record.expanded_box) if record.expanded_box else None,
    }


def close_detections() -> None:
    """Release the detector-box cache connection, for a clean shutdown."""
    global _detection_cache
    if _detection_cache is not None:
        _detection_cache.close()
        _detection_cache = None


def eye_keypoints_for(image_path: str, strategy_id: str, *, crop_cache_dir: str | Path | None = None) -> dict | None:
    """`strategy_id`'s own eye-detector result for one image, in full-frame
    pixel coordinates - the same read-only shape `detection_boxes_for`
    returns, for a caller (the Gallery overlay, the Loupe) that wants to
    draw the eye box and both raw left/right keypoints alongside the
    subject box.

    `strategy_id` is REQUIRED, not inferred: `eyes.cache` keys its sidecar
    by (image, strategy) precisely so a caller can never accidentally read
    a different strategy's cached result than the one it means to display
    (see that module's own docstring for the bug this closes - Elements/
    Boxes showing a stale, mismatched run). A caller with no strategy that
    has an eye detector at all (the AI model) simply has nothing to pass
    here that would resolve to a real record - see
    `ranking.eye_detector_names` for which strategies do.

    None when `strategy_id` has never run on this image (no cached eye
    record - see `eyes.cache`) or there is no recorded subject box to map it
    against. Review must never run the eye detector itself, only ever read
    what an earlier ranking run already computed - the same rule
    `detection_boxes_for` and `detected_category_for` already follow.

    The eye record's box/keypoints are in the subject CROP's own pixel
    space (see `eyes.detector.EyeDetection`'s docstring), so they are
    rescaled here onto the full frame using `record.expanded_box` - the
    crop's own rectangle (the tight detection box grown by
    `bird_crop.CropParams.margin_frac` before cropping - see
    `bird_crop.CropResult.expanded_box`), NOT `record.selected` (the tight
    box). Those two rectangles differ by the margin on every image with a
    margin > 0 (the default), and using the tight one here was a real,
    proven bug - see docs/EyePose_Investigation_Phase_1.md's Q1 finding: it
    silently shifted and rescaled every eye overlay by a consistent, wrong
    amount, matching the original "eye systematically displaced" symptom.
    """
    from ..eyes.cache import read_eye_detection

    eye = read_eye_detection(crop_cache_dir or DEFAULT_CROP_CACHE_DIR, image_path, strategy_id)
    if eye is None:
        return None
    try:
        record = _detections().get(image_path, allow_detect=False)
    except Exception as exc:  # noqa: BLE001 - an unreadable cache is not fatal
        logger.debug("No detections for %s: %s", image_path, exc)
        return None
    if record.selected is None or record.expanded_box is None or record.source_size is None:
        return None
    crop_width, crop_height = eye.subject_crop_size
    if crop_width <= 0 or crop_height <= 0:
        return None

    ex1, ey1, ex2, ey2 = record.expanded_box
    scale_x = (ex2 - ex1) / crop_width
    scale_y = (ey2 - ey1) / crop_height

    def to_frame(x: float, y: float) -> tuple[float, float]:
        return (ex1 + x * scale_x, ey1 + y * scale_y)

    def keypoint_dict(keypoint) -> dict | None:
        if keypoint is None:
            return None
        x, y = to_frame(keypoint.x, keypoint.y)
        return {"x": x, "y": y, "confidence": keypoint.confidence}

    x1, y1 = to_frame(eye.box[0], eye.box[1])
    x2, y2 = to_frame(eye.box[2], eye.box[3])
    return {
        "source_size": record.source_size,
        "accepted": eye.accepted,
        "confidence": eye.confidence,
        # Which eye detector actually produced this cached record - see
        # eyes.cache.EyeRecord.detector_id's own docstring. This sidecar
        # holds exactly one slot per image, overwritten by whichever eye
        # detector last ran on it, regardless of which ranking strategy a
        # caller currently has selected (ReviewSession.burst_strategy) - a
        # caller that cares whether this record actually belongs to the
        # CURRENTLY selected run (the desktop Loupe/Gallery overlay does -
        # see desktop.services.ReviewService.eye_keypoints) needs this field
        # to check, since nothing here can know what "currently selected"
        # means on its own.
        "detector_id": eye.detector_id,
        "box": (x1, y1, x2, y2),
        "left": keypoint_dict(eye.left),
        "right": keypoint_dict(eye.right),
        # The rest of EyePose-v0's landmark set, mapped through the same
        # to_frame() transform as left/right - None for any record (older
        # cache version, or a backend like SuperAnimal-Bird that never
        # computed them) that doesn't have them. See eyes.cache.EyeRecord's
        # own fields of the same names.
        "beak": keypoint_dict(eye.beak),
        "head_top": keypoint_dict(eye.head_top),
        "left_shoulder": keypoint_dict(eye.left_shoulder),
        "right_shoulder": keypoint_dict(eye.right_shoulder),
        # The holistic "is a real head instance present at all" scalar (see
        # eyes.detector.EyeDetection.head_confidence's own docstring) -
        # independent of any single landmark's own confidence, so a caller
        # showing a "Head" element alongside "Left Eye"/"Right Eye" pairs it
        # with the confidence that actually answers "is this a head" rather
        # than reusing head_top's own (a landmark-position confidence,
        # answering a narrower question). None for a backend that never
        # computed one - same "no fabricated data" rule every field here
        # already follows.
        "head_confidence": eye.head_confidence,
    }


def eye_keypoints_in_crop_for(image_path: str, strategy_id: str, *, crop_cache_dir: str | Path | None = None) -> dict | None:
    """The same eye-detector result as `eye_keypoints_for` (see its own
    docstring for why `strategy_id` is required), left in the subject
    CROP's own pixel space instead of rescaled onto the full frame.

    Manual QA Issue 3: landmarks drawn on the full photo overlap and become
    unreadable, because the head - the only part of the frame any of them
    sit near - is a small fraction of it. The cached crop file
    (`bird_crop.crop_cache_path`) already IS that region at a much larger
    effective zoom, and every field on `eyes.detector.EyeDetection` is
    already recorded in that exact crop's own coordinate space (see its own
    docstring) - so this needs no transform at all, unlike
    `eye_keypoints_for`'s `to_frame()`. Returned shape matches
    `eye_keypoints_for` exactly except `source_size` is the crop's own
    (width, height) rather than the full frame's, so a caller can reuse one
    scale-to-display calculation for either.
    """
    from ..eyes.cache import read_eye_detection

    eye = read_eye_detection(crop_cache_dir or DEFAULT_CROP_CACHE_DIR, image_path, strategy_id)
    if eye is None:
        return None
    crop_width, crop_height = eye.subject_crop_size
    if crop_width <= 0 or crop_height <= 0:
        return None

    def keypoint_dict(keypoint) -> dict | None:
        if keypoint is None:
            return None
        return {"x": keypoint.x, "y": keypoint.y, "confidence": keypoint.confidence}

    return {
        "source_size": (crop_width, crop_height),
        "accepted": eye.accepted,
        "confidence": eye.confidence,
        "detector_id": eye.detector_id,  # see eye_keypoints_for's own comment
        "box": tuple(eye.box),
        "left": keypoint_dict(eye.left),
        "right": keypoint_dict(eye.right),
        "beak": keypoint_dict(eye.beak),
        "head_top": keypoint_dict(eye.head_top),
        "left_shoulder": keypoint_dict(eye.left_shoulder),
        "right_shoulder": keypoint_dict(eye.right_shoulder),
    }
