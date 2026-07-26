"""Capabilities 10 and 15 - contact sheets, generated efficiently.

Looking at the mistakes is how a photographer actually diagnoses a model, so
this turns each error category into a labelled grid.

Performance is the hard part: full-resolution RAWs are 20-60 MB and cost about
a second each to demosaic, so a naive sheet of 60 images would take a minute.
Three things prevent that:

- **the crop cache is preferred over the RAW.** The cached crop is a small PNG
  that also shows what the model actually saw - both faster and more truthful.
- **thumbnails are cached on disk** under the output directory, keyed by source
  path and size, so re-running an analysis is nearly free.
- **generation is parallel**, using threads because the decoders release the
  GIL (the same property measured for the preprocessing pipeline).

Read-only: the crop cache is only ever read, and thumbnails are written only
under the analyzer's own output directory.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from PIL import Image, ImageDraw, ImageFont

from ..bird_crop import crop_cache_path
from ..config import DEFAULT_CROP_CACHE_DIR
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


def _thumbnail_cache_path(cache_dir: Path, source: str, size: int) -> Path:
    digest = hashlib.sha1(f"{Path(source).resolve()}|{size}".encode("utf-8")).hexdigest()[:20]
    return cache_dir / digest[:2] / f"{digest}.jpg"


def _load_source_image(image_path: str, crop_cache_dir: Path) -> Image.Image:
    """Prefer the cached crop, fall back to the original file.

    The crop is what the model was actually shown, so a mistake is far easier
    to understand from it than from the full frame - and it avoids a RAW
    demosaic entirely.
    """
    cached = crop_cache_path(crop_cache_dir, image_path)
    if cached.exists():
        return Image.open(cached).convert("RGB")

    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(image_path)
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}:
        return Image.open(source).convert("RGB")

    # A RAW with no cached crop: use the embedded preview rather than a full
    # demosaic - it costs milliseconds instead of a second.
    import rawpy

    with rawpy.imread(str(source)) as raw:
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError) as exc:
            raise ValueError(f"no embedded preview in {source.name}") from exc
        if thumb.format == rawpy.ThumbFormat.JPEG:
            import io

            return Image.open(io.BytesIO(thumb.data)).convert("RGB")
        return Image.fromarray(thumb.data).convert("RGB")


def build_thumbnail(image_path: str, size: int, cache_dir: Path, crop_cache_dir: Path) -> Path | None:
    """Return a cached square thumbnail, generating it if needed."""
    target = _thumbnail_cache_path(cache_dir, image_path, size)
    if target.exists():
        return target
    try:
        image = _load_source_image(image_path, crop_cache_dir)
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
    crop_cache_dir: Path,
    workers: int = 8,
) -> dict[str, Path]:
    """Build every thumbnail in parallel, returning path -> thumbnail."""
    unique = list({image.image_path for image in images})
    if not unique:
        return {}
    results: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="analyzer-thumb") as pool:
        for path, thumbnail in zip(unique, pool.map(
            lambda p: build_thumbnail(p, size, cache_dir, crop_cache_dir), unique
        )):
            if thumbnail is not None:
                results[path] = thumbnail
    logger.debug("Thumbnails: %d of %d source images", len(results), len(unique))
    return results


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


def render_contact_sheets(result: "AnalysisResult", crop_cache_dir: Path | None = None) -> list[Path]:
    """Generate every contact sheet, sharing one thumbnail pass."""
    specs = [spec for spec in sheet_specs(result) if spec.images]
    if not specs:
        logger.info("Nothing to put on a contact sheet.")
        return []

    config = result.config
    crop_cache = Path(crop_cache_dir) if crop_cache_dir else DEFAULT_CROP_CACHE_DIR

    # One pass over the union: an image on two sheets is decoded once.
    every_image = [image for spec in specs for image in spec.images]
    thumbnails = generate_thumbnails(
        every_image,
        config.thumbnail_size,
        config.thumbnails_dir,
        crop_cache,
        workers=config.thumbnail_workers,
    )

    written: list[Path] = []
    for spec in specs:
        written.extend(
            render_sheet(
                spec,
                thumbnails,
                config.sheets_dir / f"{spec.name}.png",
                config.thumbnail_size,
                config.contact_sheet_columns,
            )
        )
    logger.info("Rendered %d contact sheet page(s)", len(written))
    return written
