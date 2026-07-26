"""Canonical image identity for PickLikeMe.

Two different questions get asked about an image, and conflating them is what
this module exists to prevent:

**"Which derived artifact belongs to the file at this location?"** - answered by
`cache_key`, a digest of the resolved path. Correct for caches: a crop is
derived from the file *as found*, and if the file moves the crop can simply be
rebuilt. This is what `bird_crop.crop_cache_path` has always used.

**"Which image is this, wherever it now lives?"** - answered by
`image_identity`, a digest of the file's *content*. Required for anything that
must survive the archive being reorganised, renamed, or moved to another drive,
because a path digest changes the instant any of that happens.

Before this module, PickLikeMe only had the first kind, and the annotation
store was keying long-lived human knowledge with it - so a folder rename would
have orphaned every diagnosis. `image_identity` is the single canonical answer
to the second question; nothing else in the codebase should invent another.

Why a partial digest rather than the whole file: a 45 MB NEF takes ~0.15 s to
read fully, and identity is resolved for every image in a report. Size plus the
first and last 512 KB is ~1 MB per image and is effectively collision-free for
camera files - the head holds EXIF (body serial, frame counter, capture
timestamp) and the tail holds image data, so two distinct frames cannot agree on
all three. The digest carries a scheme prefix so a future change (full-file, or
perceptual hashing that survives re-encoding) is distinguishable and migratable
rather than silently incompatible.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when the digest input changes, so stored identities remain attributable
# to the scheme that produced them.
IDENTITY_SCHEME = "p1"

HEAD_BYTES = 512 * 1024
TAIL_BYTES = 512 * 1024

# Files smaller than this are hashed whole - reading them twice would cost more
# than reading them once.
WHOLE_FILE_THRESHOLD = HEAD_BYTES + TAIL_BYTES


class IdentityUnavailable(Exception):
    """Raised when an image's identity cannot be established.

    Never caught and turned into a guess. The annotation store reports these
    explicitly, because attaching a diagnosis to the wrong image is worse than
    losing it.
    """

    def __init__(self, path: str | Path, reason: str):
        self.path = str(path)
        self.reason = reason
        super().__init__(f"{self.path}: {reason}")


def cache_key(source_path: str | Path) -> str:
    """Location-derived key for a *derived artifact* (a crop, a thumbnail).

    Deliberately not the image's identity. Kept here so both notions live in
    one module and the difference is impossible to miss; `bird_crop` continues
    to own the crop-cache path layout built on top of it.
    """
    resolved = str(Path(source_path).resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:20]


def image_identity(path: str | Path) -> str:
    """Content-derived identity: `"p1:<40 hex>"`.

    Independent of filename and location, so it follows the image through any
    reorganisation of the archive.

    Raises IdentityUnavailable if the file is missing, unreadable or empty.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise IdentityUnavailable(path, f"cannot stat file ({exc.strerror or exc})") from exc

    if size == 0:
        raise IdentityUnavailable(path, "file is empty")

    digest = hashlib.sha1()
    # Size participates in the digest so truncation or appending changes identity.
    digest.update(f"{IDENTITY_SCHEME}|{size}|".encode("utf-8"))
    try:
        with path.open("rb") as handle:
            if size <= WHOLE_FILE_THRESHOLD:
                digest.update(handle.read())
            else:
                digest.update(handle.read(HEAD_BYTES))
                handle.seek(-TAIL_BYTES, 2)  # from the end
                digest.update(handle.read(TAIL_BYTES))
    except OSError as exc:
        raise IdentityUnavailable(path, f"cannot read file ({exc.strerror or exc})") from exc

    return f"{IDENTITY_SCHEME}:{digest.hexdigest()}"


def capture_datetime(path: str | Path) -> str | None:
    """EXIF capture time when it can be read cheaply, else None.

    Best-effort and never fatal: it is display metadata for the annotation
    panel, not part of identity. RAW formats mostly need an external tool
    (exiftool, used by the ingest pipeline) which may not be installed, so this
    only covers what Pillow can read directly.
    """
    path = Path(path)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"}:
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return None
            # 36867 DateTimeOriginal, 306 DateTime.
            for tag in (36867, 306):
                value = exif.get(tag)
                if value:
                    return str(value).strip() or None
    except Exception as exc:  # noqa: BLE001 - metadata only
        logger.debug("No capture time for %s: %s", path, exc)
    return None
