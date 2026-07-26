"""URL construction for the HTML report.

One place, because getting this wrong is invisible until someone clicks a link
and lands on the wrong image - or on nothing at all.

Two bugs lived in the ad-hoc version this replaces:

1. `Path(p).as_uri()` raises ValueError for a relative path, and the `.exists()`
   guard in front of it passes for relative paths (they resolve against the
   CWD). A ranking file carrying relative paths therefore killed report
   generation outright.

2. **Browsers refuse to navigate from an `http://` page to a `file://` URL.**
   That is a fixed security rule, not a setting. So every source-image link
   worked when the report was opened from disk and silently did nothing when the
   same report was served by `picklikeme annotate` - which is the mode the false
   negative panels are designed to be used in. The link was never *wrong*; it
   was unreachable, which looks identical to a wrong link from the outside.

The fix is to emit both forms: a `file://` href that works offline, plus the
absolute path in `data-source` so the served page can rewrite the href to the
server's own `/source` endpoint. One artifact, correct in both modes, on Windows
and on POSIX.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Query parameter the annotation server reads for /source.
SOURCE_ENDPOINT = "source"
SOURCE_PARAM = "path"


def absolute_source_path(image_path: str | Path) -> Path | None:
    """The image's absolute path, or None if it cannot be located.

    Resolution happens *before* any existence check, so a relative path from a
    foreign ranking file is handled rather than crashing later in `as_uri()`.
    """
    try:
        resolved = Path(image_path).resolve()
    except OSError:  # pragma: no cover - malformed path
        return None
    return resolved if resolved.is_file() else None


def source_file_uri(image_path: str | Path) -> str | None:
    """`file://` URI for an image, or None when it cannot be linked.

    Works for relative input, spaces and non-ASCII names: `as_uri()`
    percent-encodes, and resolution makes the path absolute first (which
    `as_uri()` requires).
    """
    resolved = absolute_source_path(image_path)
    if resolved is None:
        return None
    try:
        return resolved.as_uri()
    except ValueError as exc:  # pragma: no cover - resolved paths are absolute
        logger.debug("No file URI for %s: %s", image_path, exc)
        return None


def source_api_url(image_path: str | Path) -> str:
    """Relative URL for the served mode, resolved by the annotation server.

    Relative rather than absolute so it works whatever host and port the server
    happens to be on.
    """
    return f"{SOURCE_ENDPOINT}?{SOURCE_PARAM}={quote(str(Path(image_path)), safe='')}"


def asset_url(target: str | Path, output_dir: str | Path) -> str | None:
    """URL for a generated asset (chart, thumbnail, contact sheet).

    Relative to the report when the asset lives under the output directory - the
    normal case, and what keeps a report directory portable. Falls back to an
    absolute `file://` URI when it does not, instead of raising the way a bare
    `relative_to()` would: an asset outside the report is unusual but is not a
    reason to lose the whole report.

    Both sides are resolved first, so a relative `--output` and an absolute
    asset path (or the reverse) still compare correctly.
    """
    target = Path(target)
    try:
        resolved_target = target.resolve()
        resolved_base = Path(output_dir).resolve()
    except OSError:  # pragma: no cover
        return None
    try:
        return resolved_target.relative_to(resolved_base).as_posix()
    except ValueError:
        logger.debug("Asset %s is outside %s; linking absolutely", target, output_dir)
        try:
            return resolved_target.as_uri()
        except ValueError:  # pragma: no cover
            return None
