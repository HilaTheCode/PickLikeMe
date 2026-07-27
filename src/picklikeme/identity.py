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

Design review (measured on this project's own archive, 52 MB mean NEF):

    hash throughput, in RAM        sha1 2280 MB/s, sha256 4184 MB/s, blake2b 1011 MB/s
    current scheme                 17.6 ms/image  ->  0.27 h for 55k images
    hypothetical full-file sha1   262.6 ms/image  ->  4.01 h for 55k images
    of that full-file cost         6.2% is hashing, 93.8% is reading the file

**Identity is I/O-bound, not CPU-bound.** That single fact settles the algorithm
question: a faster hash (BLAKE3, and note sha256 is already *faster* than sha1
here thanks to CPU SHA extensions) would optimise the 6% and leave the 94%
untouched. What buys the 15x is reading 1 MB instead of 52 MB - the amount read,
not the primitive. BLAKE3 would also add a dependency to a project that
currently needs none for this, and re-keying every stored annotation, in
exchange for no measurable gain.

Collision resistance is likewise not the binding constraint. There is no
adversary - nobody is crafting a RAW to collide with another - and 160 bits over
55k items is astronomically safe. The residual risk is inherent to the *partial*
read, not the primitive: two distinct files would have to agree on size AND the
first 512 KB AND the last 512 KB. For camera files the head carries EXIF (body
serial, frame counter, capture timestamp), so two different frames cannot.
Swapping sha1 for sha256 would not reduce that risk at all.

The one real weakness: a tool that rewrites metadata *inside* the RAW (rather
than into a sidecar .xmp) changes the head and therefore the identity, orphaning
that image's annotation. Uncommon - Lightroom writes sidecars for NEF/ARW - and
the alternatives are worse (hashing only the tail weakens identity; parsing out
the image payload means format-specific code for every camera). The `p1:` prefix
plus the migration machinery in the annotation store means this can be revisited
without data loss if it ever bites.

Conclusion: unchanged, on evidence rather than inertia.
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
