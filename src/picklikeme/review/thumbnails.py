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

_detection_cache = None


def _preview_cache_path(cache_dir: Path, image_path: str) -> Path:
    # Same convention as contactsheets._thumbnail_cache_path: resolved path +
    # a version stamp, no mtime. These are camera RAW files a photographer
    # reviews, not documents someone re-saves in place, so that tradeoff -
    # already accepted for the square thumbnails - applies here unchanged.
    key = f"{Path(image_path).resolve()}|v{REVIEW_PREVIEW_CACHE_VERSION}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return cache_dir / digest[:2] / f"{digest}.jpg"


def review_preview(image_path: str, *, cache_dir: Path | None = None) -> Path:
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
    """
    from ..analyzer.contactsheets import load_source_image

    cache_dir = cache_dir or REVIEW_PREVIEW_CACHE
    target = _preview_cache_path(cache_dir, image_path)
    if target.exists():
        return target
    image = load_source_image(image_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=92)
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


def close_detections() -> None:
    """Release the detector-box cache connection, for a clean shutdown."""
    global _detection_cache
    if _detection_cache is not None:
        _detection_cache.close()
        _detection_cache = None
