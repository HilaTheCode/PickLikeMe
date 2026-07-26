"""Reading the analyzer's inputs: ranking files and ground-truth folders.

Read-only by construction - nothing here opens a file for writing.

Ranking files are read with **field auto-detection** rather than a fixed
schema. PickLikeMe's own writer emits a metrics preamble followed by
`rank,image_path,score,label`, chunked across `_1`/`_2` files; other producers
emit other column names. Rather than force one format, the loader locates the
header row and maps whatever columns it finds onto RankedImage.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from ..dataset import ALLOWED_RAW_EXTENSIONS
from .model import RankedImage

logger = logging.getLogger(__name__)

# Column aliases, in priority order. The first name present in the header wins,
# so a file carrying both "probability" and "score" uses each for its own field.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "image_path": ("image_path", "path", "filepath", "file_path", "full_path", "filename", "file", "name"),
    "score": ("score", "prediction", "predicted_score", "pred", "value"),
    "rank": ("rank", "position", "ranking", "index"),
    "label": ("label", "ground_truth", "truth", "actual", "y_true"),
    "probability": ("probability", "prob", "predicted_probability", "p", "confidence_score"),
    "predicted_class": ("predicted_class", "prediction_class", "y_pred", "predicted_label"),
    "confidence": ("confidence", "certainty"),
}

# A header row must contain something path-like and something score-like;
# that is what distinguishes it from the key/value metrics preamble.
_REQUIRED = ("image_path", "score")

# Extra image types the analyzer will happily thumbnail, beyond the RAW formats
# the training pipeline ingests.
PREVIEWABLE_EXTENSIONS = ALLOWED_RAW_EXTENSIONS | {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


class RankingFormatError(ValueError):
    """Raised when a ranking file has no recognisable header row."""


@dataclass(frozen=True)
class RankingFile:
    """A loaded ranking, plus what the loader learned about it."""

    path: Path
    images: list[RankedImage]
    detected_columns: dict[str, str]
    preamble: dict[str, str]
    chunk_paths: list[Path]
    warnings: list[str]

    @property
    def has_probabilities(self) -> bool:
        return any(image.probability is not None for image in self.images)

    @property
    def has_labels(self) -> bool:
        return any(image.label is not None for image in self.images)


def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _detect_columns(header: list[str]) -> dict[str, str]:
    """Map canonical field -> actual column name for the columns present."""
    normalised = {_normalise(col): col for col in header if col.strip()}
    detected: dict[str, str] = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised:
                detected[field_name] = normalised[alias]
                break
    return detected


def _looks_like_header(row: list[str]) -> bool:
    return all(key in _detect_columns(row) for key in _REQUIRED)


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    number = _to_float(value)
    return None if number is None else int(number)


def discover_chunks(ranking_path: Path) -> list[Path]:
    """The chunk files that belong with `ranking_path`.

    write_results_csv splits long rankings into `name.csv`, `name_1.csv`,
    `name_2.csv`. They are located by *computing* the successive names, not by
    globbing, so an unrelated `name_final.csv` in the same folder is never
    swept in.
    """
    chunks = [ranking_path]
    index = 1
    while True:
        candidate = ranking_path.with_name(f"{ranking_path.stem}_{index}{ranking_path.suffix}")
        if not candidate.exists():
            break
        chunks.append(candidate)
        index += 1
    return chunks


def _read_chunk(path: Path, warnings: list[str]) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    """Return (rows, detected columns, preamble) for one chunk."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    header_index = next((i for i, row in enumerate(rows) if _looks_like_header(row)), None)
    if header_index is None:
        raise RankingFormatError(
            f"{path}: no header row containing a path column and a score column was found. "
            f"Recognised path columns: {COLUMN_ALIASES['image_path']}; "
            f"score columns: {COLUMN_ALIASES['score']}."
        )

    # Everything above the header is PickLikeMe's key/value metrics preamble.
    preamble = {
        row[0].strip(): row[1].strip()
        for row in rows[:header_index]
        if len(row) >= 2 and row[0].strip() and row[0].strip() != "metric"
    }

    header = rows[header_index]
    detected = _detect_columns(header)
    records = []
    for line_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            warnings.append(f"{path.name}:{line_number}: expected {len(header)} fields, got {len(row)}; row skipped")
            continue
        records.append(dict(zip(header, row)))
    return records, detected, preamble


def load_ranking(ranking_path: str | Path) -> RankingFile:
    """Load a ranking file (plus its chunks) into RankedImage records.

    Rank is taken from the file when present and otherwise assigned by
    descending score, so a bare `path,score` CSV still yields usable ranking
    metrics.
    """
    ranking_path = Path(ranking_path)
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking file not found: {ranking_path}")

    warnings: list[str] = []
    chunks = discover_chunks(ranking_path)
    records: list[dict[str, str]] = []
    detected: dict[str, str] = {}
    preamble: dict[str, str] = {}

    for chunk in chunks:
        chunk_records, chunk_detected, chunk_preamble = _read_chunk(chunk, warnings)
        records.extend(chunk_records)
        detected = detected or chunk_detected
        preamble = preamble or chunk_preamble

    images: list[RankedImage] = []
    seen: set[str] = set()
    for record in records:
        raw_path = record.get(detected["image_path"], "").strip()
        score = _to_float(record.get(detected["score"]))
        if not raw_path or score is None:
            warnings.append(f"Row with missing path or unparsable score skipped: {record}")
            continue
        if raw_path in seen:
            warnings.append(f"Duplicate ranking entry ignored: {raw_path}")
            continue
        seen.add(raw_path)

        probability = _to_float(record.get(detected["probability"])) if "probability" in detected else None
        if probability is None and 0.0 <= score <= 1.0:
            # PickLikeMe regresses toward 1.0/0.0, so an in-range score already
            # is a probability. Outside [0, 1] we refuse to invent one rather
            # than silently squashing a logit through a guessed link function.
            probability = score

        images.append(
            RankedImage(
                image_path=raw_path,
                score=score,
                rank=_to_int(record.get(detected["rank"])) if "rank" in detected else 0,
                label=_to_int(record.get(detected["label"])) if "label" in detected else None,
                probability=probability,
                predicted_class=_to_int(record.get(detected["predicted_class"]))
                if "predicted_class" in detected
                else None,
                confidence=_to_float(record.get(detected["confidence"])) if "confidence" in detected else None,
            )
        )

    if "rank" not in detected or all(image.rank == 0 for image in images):
        order = sorted(range(len(images)), key=lambda i: images[i].score, reverse=True)
        ranked = list(images)
        for position, index in enumerate(order, start=1):
            ranked[index] = RankedImage(**{**images[index].__dict__, "rank": position})
        images = ranked
        warnings.append("No rank column found; ranks assigned by descending score.")

    if not images:
        raise RankingFormatError(f"{ranking_path}: header found but no usable data rows.")

    logger.info("Loaded %d ranked images from %d file(s)", len(images), len(chunks))
    return RankingFile(
        path=ranking_path,
        images=images,
        detected_columns=detected,
        preamble=preamble,
        chunk_paths=chunks,
        warnings=warnings,
    )


def enumerate_ground_truth(root: str | Path, extensions: set[str] | None = None) -> list[Path]:
    """Every image file under a ground-truth folder, recursively.

    Accepts previewable formats as well as RAW, so an analysis can be run
    against a folder of JPEGs without first ingesting them.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Ground-truth folder not found: {root}")
    allowed = extensions if extensions is not None else PREVIEWABLE_EXTENSIONS
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
