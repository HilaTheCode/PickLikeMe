"""Capabilities 10 and 15 - contact sheets, generated efficiently.

Looking at the mistakes is how a photographer actually diagnoses a model, so
this turns each error category into a labelled grid.

Every preview shows the **whole original frame**, never a crop of it. That is
what makes the detector-box overlay meaningful: boxes are recorded in full-frame
coordinates, so drawing them on anything but the full frame puts them in the
wrong place. It is also what a photographer needs in order to judge a miss -
seeing only the region the detector already chose cannot show you that it chose
the wrong region.

Performance is the hard part: full-resolution RAWs are 20-60 MB and cost about
a second each to demosaic, so a naive sheet of 60 images would take a minute.
Three things prevent that:

- **RAWs use their embedded preview.** Cameras store a full-frame JPEG inside
  the file; extracting it costs milliseconds instead of a second, and it covers
  the same field of view. A full demosaic is the fallback for the rare RAW that
  has no embedded preview.
- **thumbnails are cached on disk** under the output directory, keyed by source
  path and size, so re-running an analysis is nearly free.
- **generation is parallel**, using threads because the decoders release the
  GIL (the same property measured for the preprocessing pipeline).

Read-only: nothing outside the analyzer's own output directory is written.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from PIL import Image, ImageDraw, ImageFont

from .model import MatchedImage, Outcome

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .analysis import AnalysisResult

logger = logging.getLogger(__name__)

BACKGROUND = (18, 20, 26)
CAPTION_BACKGROUND = (28, 31, 40)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)
OUTCOME_COLOURS = {
    Outcome.TRUE_POSITIVE: (16, 185, 129),
    Outcome.TRUE_NEGATIVE: (100, 116, 139),
    Outcome.FALSE_POSITIVE: (239, 68, 68),
    Outcome.FALSE_NEGATIVE: (249, 115, 22),
    Outcome.UNKNOWN: (100, 116, 139),
}

CAPTION_LINES = 3
CAPTION_LINE_HEIGHT = 13
CAPTION_HEIGHT = CAPTION_LINES * CAPTION_LINE_HEIGHT + 6
PADDING = 8
MAX_ROWS_PER_SHEET = 6


@dataclass(frozen=True)
class SheetSpec:
    """One contact sheet to produce."""

    name: str
    title: str
    images: list[MatchedImage]


# Bumped whenever a change makes previously cached thumbnails wrong, so stale
# entries are simply never looked up again. v2: previews are built from the full
# frame; every v1 entry was built from the cached bird crop.
THUMBNAIL_CACHE_VERSION = 2

STANDARD_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def _thumbnail_cache_path(cache_dir: Path, source: str, size: int) -> Path:
    key = f"{Path(source).resolve()}|{size}|v{THUMBNAIL_CACHE_VERSION}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return cache_dir / digest[:2] / f"{digest}.jpg"


def _extract_thumb(raw):
    """The RAW's embedded thumbnail, or None if it has none rawpy can read.

    Shared by `load_source_image` (which decodes it into a frame to work
    with) and `export_jpeg_bytes` (which, when the thumbnail is already a
    JPEG, wants the camera's own bytes untouched rather than a decode).
    """
    import rawpy

    try:
        return raw.extract_thumb()
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        return None


def load_source_image(image_path: str) -> Image.Image:
    """The whole original frame, as cheaply as the format allows.

    Never the cached bird crop. Detector boxes are recorded in full-frame
    coordinates, so a crop-based preview would put every box in the wrong place;
    and a crop cannot show what a photographer most needs to see, which is
    whether the detector picked the wrong region in the first place.

    Public (not prefixed with `_`): also used directly by server._serve_preview
    to stream a full-size RAW preview on demand for the report's "Open
    Preview" action, without going through the thumbnail cache at all.
    """
    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(image_path)
    if source.suffix.lower() in STANDARD_IMAGE_SUFFIXES:
        return Image.open(source).convert("RGB")

    # A RAW: the embedded preview is the full frame and costs milliseconds,
    # where a demosaic costs about a second.
    import rawpy

    with rawpy.imread(str(source)) as raw:
        thumb = _extract_thumb(raw)
        if thumb is None:
            # No embedded preview: pay for the demosaic rather than drop the
            # image from the report. Rare, and only for the images a report
            # actually shows.
            logger.debug("No embedded preview in %s; demosaicing instead", source.name)
            return Image.fromarray(raw.postprocess(use_camera_wb=True)).convert("RGB")
        if thumb.format == rawpy.ThumbFormat.JPEG:
            import io

            return Image.open(io.BytesIO(thumb.data)).convert("RGB")
        return Image.fromarray(thumb.data).convert("RGB")


def export_jpeg_bytes(image_path: str) -> bytes:
    """The best available JPEG for `image_path`, extracted as cheaply as the
    format allows - never a RAW development pipeline.

    - Already a JPEG: its own bytes, unchanged.
    - A RAW with an embedded JPEG thumbnail: the camera's own bytes, not a
      decode-then-recompress of them (unlike `/preview`, which always goes
      through PIL because it also has to serve the demosaic fallback through
      the same code path). This is what makes "Save as JPEG" both instant and
      indistinguishable from the camera's own rendering.
    - Anything else (a RAW with no embedded thumbnail, or a non-JPEG image
      format): `load_source_image`'s own fallback, re-encoded.

    Used by review/server.py's `/save-jpeg` - a convenience export for
    sharing, not a step toward RAW editing.
    """
    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(image_path)

    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return source.read_bytes()

    if suffix not in STANDARD_IMAGE_SUFFIXES:
        import rawpy

        with rawpy.imread(str(source)) as raw:
            thumb = _extract_thumb(raw)
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            return thumb.data

    import io

    image = load_source_image(image_path)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def build_thumbnail(image_path: str, size: int, cache_dir: Path) -> Path | None:
    """Return a cached square thumbnail of the full frame, generating it if needed."""
    target = _thumbnail_cache_path(cache_dir, image_path, size)
    if target.exists():
        return target
    try:
        image = load_source_image(image_path)
    except Exception as exc:  # noqa: BLE001 - a missing source must not stop the sheet
        logger.debug("No thumbnail for %s: %s", image_path, exc)
        return None

    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, "JPEG", quality=88)
    return target


def generate_thumbnails(
    images: Sequence[MatchedImage],
    size: int,
    cache_dir: Path,
    workers: int = 8,
) -> dict[str, Path]:
    """Build every thumbnail in parallel, returning path -> thumbnail."""
    unique = list({image.image_path for image in images})
    if not unique:
        return {}
    results: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="analyzer-thumb") as pool:
        for path, thumbnail in zip(unique, pool.map(
            lambda p: build_thumbnail(p, size, cache_dir), unique
        )):
            if thumbnail is not None:
                results[path] = thumbnail
    logger.debug("Thumbnails: %d of %d source images", len(results), len(unique))
    return results


# Overlay colours for the detector-box diagnostic thumbnail.
SELECTED_BOX = (16, 185, 129)   # the box that produced the crop the model scored
OTHER_BOX = (250, 204, 21)      # runners-up: visible, clearly not the choice
NO_DETECTION = (239, 68, 68)


def annotate_thumbnail(
    thumbnail_path: Path,
    record,
    output_path: Path,
    size: int,
) -> Path | None:
    """Draw the detector's boxes onto a copy of an existing thumbnail.

    The point is to answer, at a glance, whether the detector contributed to a
    mistake: solid green is the box that became the crop the model actually
    scored, thin dashed amber are the other candidates it passed over, and a red
    corner marker means it found nothing at all (so the model saw the whole
    frame).

    Boxes come from `record` in full-frame coordinates. They are mapped onto the
    thumbnail by normalising against the frame and scaling onto the letterboxed
    content rectangle, which only holds because the thumbnail shows the whole
    frame - see `load_source_image`. Aspect ratio, thumbnail size and layout are
    untouched: this writes a same-size sibling image, so the report gets no
    heavier.
    """
    if record is None or not record.boxes or record.source_size is None:
        return None
    try:
        base = Image.open(thumbnail_path).convert("RGB")
    except OSError:
        return None

    frame_width, frame_height = record.source_size
    if frame_width <= 0 or frame_height <= 0:
        return None

    # The thumbnail is the full frame letterboxed into a square: the content
    # rectangle keeps the frame's aspect ratio and is centred, with background
    # bars filling the rest. Derived here rather than stored, so the overlay
    # stays correct even if the thumbnail size changes between runs.
    scale = min(base.width / frame_width, base.height / frame_height)
    drawn_width, drawn_height = frame_width * scale, frame_height * scale
    offset_x = (base.width - drawn_width) / 2
    offset_y = (base.height - drawn_height) / 2

    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    line = max(1, round(size / 200))

    def to_thumb(box) -> tuple[float, float, float, float]:
        return (
            offset_x + box.x1 * scale,
            offset_y + box.y1 * scale,
            offset_x + box.x2 * scale,
            offset_y + box.y2 * scale,
        )

    # Runners-up first, so the selected box is never hidden behind one.
    for box in record.others:
        x1, y1, x2, y2 = to_thumb(box)
        _dashed_rectangle(draw, (x1, y1, x2, y2), OTHER_BOX, _stroke(line, x2 - x1, y2 - y1))

    selected = record.selected
    if selected is not None:
        x1, y1, x2, y2 = to_thumb(selected)
        draw.rectangle([x1, y1, x2, y2], outline=SELECTED_BOX, width=_stroke(line + 1, x2 - x1, y2 - y1))
        _draw_label(
            draw, f"{selected.class_name} {selected.score:.2f}", (x1, y1, x2, y2),
            SELECTED_BOX, font, base.size,
        )
    else:
        # Nothing detected: the model was shown the full frame.
        draw.line([(0, 0), (size // 5, 0)], fill=NO_DETECTION, width=line + 2)
        draw.line([(0, 0), (0, size // 5)], fill=NO_DETECTION, width=line + 2)
        draw.text((3, 3), "no detection", fill=NO_DETECTION, font=font)

    if len(record.others):
        draw.text((3, base.height - 12), f"+{len(record.others)} more", fill=OTHER_BOX, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=90)
    return output_path


def _stroke(preferred: int, box_width: float, box_height: float) -> int:
    """Outline width that cannot swallow the box it outlines.

    A full frame puts a distant bird in ten or twenty pixels. At the canvas-wide
    stroke that suits a large box, both edges plus the gap between them exceed
    the box, so it renders as a filled blob and the reader learns nothing about
    where the detector actually drew it. Capped at a quarter of the shorter side.
    """
    return max(1, min(preferred, int(min(box_width, box_height) / 4)))


def _draw_label(draw, text: str, box, colour, font, canvas_size) -> None:
    """Label the box, but only where the label can be read as belonging to it.

    A label wider than its box points at the wrong thing: on a small box the
    text sprawls across neighbouring detections and reads as labelling those
    instead. Better to leave a small box unlabelled - its colour already says
    which one it is, and the report lists the class and score in text.
    """
    x1, y1, x2, y2 = box
    try:
        width = draw.textlength(text, font=font)
    except AttributeError:  # pragma: no cover - very old Pillow
        width = len(text) * 6
    if width > (x2 - x1) * 1.6 or y1 < 11:
        return
    draw.text((max(1, x1 + 1), y1 - 10), text, fill=colour, font=font)


def _dashed_rectangle(draw, box, colour, width: int, dash: int | None = None) -> None:
    """A dashed outline, so a runner-up reads as different from the choice even
    in a greyscale print or for a colour-blind reader.

    The dash length follows the box: a fixed one long enough to read on a large
    box leaves a small box with a single dash per edge, which is just a solid
    outline wearing the wrong colour.
    """
    x1, y1, x2, y2 = box
    if dash is None:
        dash = max(2, int(min(x2 - x1, y2 - y1) / 6))
    for x in range(int(x1), int(x2), dash * 2):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=colour, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=colour, width=width)
    for y in range(int(y1), int(y2), dash * 2):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=colour, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=colour, width=width)


def annotated_thumbnail_path(thumbnails_dir: Path, image_path: str, size: int) -> Path:
    """Where the annotated copy lives: beside the plain one, distinct suffix."""
    plain = _thumbnail_cache_path(thumbnails_dir, image_path, size)
    return plain.with_name(f"{plain.stem}_boxes{plain.suffix}")


def build_thumbnail_overlays(
    result,
    thumbnails: dict[str, Path],
    detection_records: dict,
) -> dict[str, Path]:
    """Annotated (detector-box) thumbnails for every image with a resolved
    detection record - every contact sheet and every HTML thumbnail table, not
    only false negatives.

    Returns image path -> annotated thumbnail. An image with no detection
    record (nothing preprocessing recorded, nothing cached, and detection was
    disabled or failed for it) is simply absent from the result, so its plain
    thumbnail is used wherever it appears - there is no forced fallback.
    """
    config = result.config
    overlays: dict[str, Path] = {}
    for path, plain in thumbnails.items():
        record = detection_records.get(path)
        if record is None or not record.boxes:
            continue
        annotated = annotate_thumbnail(
            plain,
            record,
            annotated_thumbnail_path(config.thumbnails_dir, path, config.thumbnail_size),
            config.thumbnail_size,
        )
        if annotated is not None:
            overlays[path] = annotated
    logger.info("Annotated %d thumbnail(s) with detector boxes", len(overlays))
    return overlays


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 2] + ".."


def render_sheet(
    spec: SheetSpec,
    thumbnails: dict[str, Path],
    output_path: Path,
    thumb_size: int,
    columns: int,
) -> list[Path]:
    """Render one category, paginating so no single image gets enormous."""
    if not spec.images:
        return []

    per_page = columns * MAX_ROWS_PER_SHEET
    pages = [spec.images[i : i + per_page] for i in range(0, len(spec.images), per_page)]
    written: list[Path] = []
    font = _font()
    cell_width = thumb_size + PADDING
    cell_height = thumb_size + CAPTION_HEIGHT + PADDING
    header = 34

    for page_number, page in enumerate(pages, start=1):
        rows = (len(page) + columns - 1) // columns
        width = columns * cell_width + PADDING
        height = header + rows * cell_height + PADDING
        sheet = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(sheet)

        heading = spec.title if len(pages) == 1 else f"{spec.title} - page {page_number}/{len(pages)}"
        draw.text((PADDING, 10), heading, fill=TEXT, font=font)
        draw.text(
            (width - 190, 10),
            f"{len(spec.images)} images",
            fill=MUTED,
            font=font,
        )

        for index, image in enumerate(page):
            column, row = index % columns, index // columns
            x = PADDING + column * cell_width
            y = header + row * cell_height

            thumbnail_path = thumbnails.get(image.image_path)
            if thumbnail_path is not None and thumbnail_path.exists():
                sheet.paste(Image.open(thumbnail_path), (x, y))
            else:
                draw.rectangle([x, y, x + thumb_size, y + thumb_size], fill=CAPTION_BACKGROUND)
                draw.text((x + 8, y + thumb_size // 2), "(no preview)", fill=MUTED, font=font)

            # A coloured bar carries the outcome at a glance, before any text.
            colour = OUTCOME_COLOURS[image.outcome]
            draw.rectangle([x, y, x + thumb_size, y + 4], fill=colour)

            caption_top = y + thumb_size
            draw.rectangle(
                [x, caption_top, x + thumb_size, caption_top + CAPTION_HEIGHT], fill=CAPTION_BACKGROUND
            )
            characters = max(10, thumb_size // 6)
            confidence = "n/a" if image.confidence is None else f"{image.confidence:.2f}"
            truth = "kept" if image.truth == 1 else ("rejected" if image.truth == 0 else "unknown")
            lines = [
                (_truncate(image.filename, characters), TEXT),
                (f"{image.outcome.short}  score {image.score:.3f}", colour),
                (f"you: {truth}  conf {confidence}  #{image.rank:,}", MUTED),
            ]
            for line_index, (text, fill) in enumerate(lines):
                draw.text(
                    (x + 4, caption_top + 3 + line_index * CAPTION_LINE_HEIGHT),
                    text,
                    fill=fill,
                    font=font,
                )

        page_path = (
            output_path
            if len(pages) == 1
            else output_path.with_name(f"{output_path.stem}_p{page_number}{output_path.suffix}")
        )
        page_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(page_path, "PNG")
        written.append(page_path)

    return written


def sheet_specs(result: "AnalysisResult") -> list[SheetSpec]:
    """Every sheet the report wants, in reading order."""
    errors = result.errors
    limit = result.config.max_examples
    return [
        SheetSpec("false_positives", "False positives - model kept, you rejected",
                  [record.image for record in errors.false_positives]),
        SheetSpec("false_negatives", "False negatives - model rejected, you kept",
                  [record.image for record in errors.false_negatives]),
        SheetSpec("borderline", "Borderline - the model has no opinion",
                  [record.image for record in errors.borderline]),
        SheetSpec("rank_disagreements", "Largest ranking disagreements",
                  [record.image for record in errors.largest_rank_disagreements]),
        SheetSpec("most_surprising", "Most surprising predictions",
                  [record.image for record in errors.most_surprising]),
        SheetSpec("top_ranked", "Top of the model's ranking", errors.top_ranked[:limit]),
        SheetSpec("lowest_ranked", "Bottom of the model's ranking", errors.lowest_ranked[:limit]),
        SheetSpec("true_positives", "True positives - agreed keeps",
                  result.match.by_outcome(Outcome.TRUE_POSITIVE)[:limit]),
        SheetSpec("true_negatives", "True negatives - agreed rejects",
                  result.match.by_outcome(Outcome.TRUE_NEGATIVE)[:limit]),
    ]


def render_contact_sheets(result: "AnalysisResult") -> list[Path]:
    """Generate every contact sheet, sharing one thumbnail pass."""
    specs = [spec for spec in sheet_specs(result) if spec.images]
    if not specs:
        logger.info("Nothing to put on a contact sheet.")
        return []

    config = result.config

    # One pass over the union: an image on two sheets is decoded once.
    every_image = [image for spec in specs for image in spec.images]
    thumbnails = generate_thumbnails(
        every_image,
        config.thumbnail_size,
        config.thumbnails_dir,
        workers=config.thumbnail_workers,
    )

    # Detector-box overlays, applied to every sheet: any image with a resolved
    # detection record shows its boxes, whichever category it appears in.
    overlays = build_thumbnail_overlays(result, thumbnails, getattr(result, "detections", {}))
    sheet_thumbs = {**thumbnails, **overlays}

    written: list[Path] = []
    for spec in specs:
        written.extend(
            render_sheet(
                spec,
                sheet_thumbs,
                config.sheets_dir / f"{spec.name}.png",
                config.thumbnail_size,
                config.contact_sheet_columns,
            )
        )
    logger.info("Rendered %d contact sheet page(s)", len(written))
    return written
