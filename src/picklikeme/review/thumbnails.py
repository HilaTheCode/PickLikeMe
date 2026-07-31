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
    if target.exists():
        try:
            os.utime(target, None)  # mark "recently used" - see _enforce_cache_budget
        except OSError:
            pass
        return target

    image = load_source_image(image_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=92)

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
    size: int = REVIEW_THUMBNAIL_SIZE,
    cache_dir: Path | None = None,
) -> Path | None:
    """A cached thumbnail for one image, or None when it cannot be rendered.

    With `with_boxes`, returns the overlaid copy - a separate file beside the
    plain one, so the page can toggle between them without either being
    rebuilt. Falls back to the plain thumbnail when the image has no recorded
    detections, which is the honest answer: there are no boxes to draw.
    """
    from ..analyzer.contactsheets import annotate_thumbnail, annotated_thumbnail_path, build_thumbnail

    cache_dir = cache_dir or REVIEW_THUMBNAIL_CACHE
    plain = build_thumbnail(image_path, size, cache_dir)
    if plain is None or not with_boxes:
        return plain

    overlaid = annotated_thumbnail_path(cache_dir, image_path, size)
    if overlaid.exists():
        return overlaid

    try:
        record = _detections().get(image_path, allow_detect=False)
    except Exception as exc:  # noqa: BLE001 - an unreadable cache is not fatal
        logger.debug("No detections for %s: %s", image_path, exc)
        return plain

    return annotate_thumbnail(plain, record, overlaid, size) or plain


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
    }


def close_detections() -> None:
    """Release the detector-box cache connection, for a clean shutdown."""
    global _detection_cache
    if _detection_cache is not None:
        _detection_cache.close()
        _detection_cache = None
