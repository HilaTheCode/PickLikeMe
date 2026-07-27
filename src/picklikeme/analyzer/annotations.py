"""False-negative knowledge base: the photographer's own diagnoses, persisted.

Why an image the photographer deliberately kept was rejected by the model is
knowledge only the photographer has. This module stores that knowledge and
nothing else.

A diagnosis is three independent fixed-vocabulary fields - Crop Quality, Image
Quality, and whether you Agree with the Model Decision. Closed vocabularies with
no free-text option, because the purpose is data that can be counted: a growable
tag list (which the superseded `categories` field was) makes frequencies
incomparable over time, since "Backlit" and "backlighting" become two rows in a
breakdown but one phenomenon.

- **Never generated.** No heuristic, no model, nothing in this file infers a
  value. Every field comes from a human via the report UI. That extends to the
  redesign itself: pre-redesign records are preserved verbatim and are *not*
  auto-mapped onto the new fields, because guessing that (say) an old "Subject
  too small" tag meant Crop Quality "Too Small" would invent data the
  photographer never entered - and it would then be counted as if they had.
- **Never influences metrics.** Annotations are attached to an AnalysisResult
  for display only. A test asserts every metric is bit-identical with and
  without an annotation database present.
- **Long-lived.** The database lives outside the analyzer's output directory,
  because output directories are per-run and get replaced; a knowledge base
  accumulated over months must not be inside one.
- **False negatives only**, by request. The annotation workflow is wired to
  false negatives and nothing else; the detector-box thumbnail overlay is a
  separate, unrelated feature (analyzer.detections / analyzer.contactsheets)
  that does apply report-wide.

Identity is the one genuinely hard problem: a diagnosis must follow the image
through a rename, a reorganisation, or a move to another drive. Annotations are
therefore keyed on the project's canonical **content-derived** identity
(`identity.image_identity`), never on a path or a filename. Filename, path and
capture time are stored as display metadata only and are never used to match.

There is deliberately **no fallback**. If identity cannot be established the
condition is reported, because attaching a diagnosis to the wrong image is worse
than losing it.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..config import PROJECT_ROOT
from ..identity import IdentityUnavailable, cache_key, capture_datetime, image_identity

logger = logging.getLogger(__name__)

# Deliberately outside any analysis output directory: those are per-run and get
# overwritten, and this database is meant to outlive every one of them.
DEFAULT_ANNOTATIONS_DB = PROJECT_ROOT / "annotations" / "false_negatives.db"

# ---------------------------------------------------------------------------
# The annotation vocabulary: three independent fixed-value fields.
#
# Every field is a closed set with no free-text escape hatch, because the point
# is data that can be counted. A growable vocabulary (which the superseded
# `categories` field had) makes frequencies incomparable across time: "Backlit"
# and "backlighting" are two rows in a breakdown but one phenomenon.
#
# The three axes are deliberately independent - a technically fine crop of an
# out-of-focus bird, or a badly placed crop of a perfectly sharp one, are
# different diagnoses and must be recordable separately.
# ---------------------------------------------------------------------------

# How good was the crop the detector chose? Judged on its own, independent of
# whether the underlying photograph is any good.
CROP_QUALITY_VALUES: tuple[str, ...] = (
    "Good",
    "Too Small",
    "Wrong Location",
    "Too Large",
)

# How good is the photograph itself? Judged on its own, independent of how the
# crop was placed.
#   Good                - image quality is acceptable
#   Missing Eye         - the bird's eye is not visible
#   Out of Focus        - the bird is not sufficiently sharp
#   No Relevant Subject - nothing meaningful to evaluate (bird too small,
#                         heavily occluded, or effectively absent)
IMAGE_QUALITY_VALUES: tuple[str, ...] = (
    "Good",
    "Missing Eye",
    "Out of Focus",
    "No Relevant Subject",
)

# Having looked at it: was the model right to reject this image?
AGREE_WITH_MODEL_VALUES: tuple[str, ...] = ("Yes", "No")

# field name -> allowed values, so validation, the API and the UI all read the
# same source and cannot drift apart.
ANNOTATION_FIELDS: dict[str, tuple[str, ...]] = {
    "crop_quality": CROP_QUALITY_VALUES,
    "image_quality": IMAGE_QUALITY_VALUES,
    "agree_with_model_decision": AGREE_WITH_MODEL_VALUES,
}

# Human labels for the report UI, keyed the same way.
ANNOTATION_FIELD_LABELS: dict[str, str] = {
    "crop_quality": "Crop Quality",
    "image_quality": "Image Quality",
    "agree_with_model_decision": "Agree with Model Decision",
}

# --- superseded vocabularies, retained for reading old records only ---------
# The growable category checklist and the single primary-cause radio that the
# three fields above replaced. Nothing writes these any more; they are kept so
# a database annotated before the change still renders its history instead of
# appearing empty.
LEGACY_CATEGORIES: tuple[str, ...] = (
    "Wrong crop",
    "Multiple subjects",
    "Subject too small",
    "Foreground obstruction",
    "Out of focus foreground",
    "Subject not centered",
    "Artistic choice",
    "Distracting background",
    "Detector mistake",
    "Pose not appreciated",
    "Action shot",
    "Lighting",
    "Backlit",
    "Animal not in supported categories",
    "Other",
)
LEGACY_PRIMARY_FAILURE_CAUSES: tuple[str, ...] = (
    "Detection crop too small",
    "Head outside crop",
    "Multiple birds",
    "Occlusion",
    "Classifier disagreement",
    "Other",
)

# v1 keyed annotations on a digest of the resolved path, which a rename would
# have orphaned. v2 keys them on content identity and migrates v1 rows across.
SCHEMA_VERSION = 2


class InvalidAnnotationValue(ValueError):
    """Raised when a field is given a value outside its fixed vocabulary.

    Rejected rather than stored, because a stray value would quietly corrupt
    exactly the frequency counts this schema exists to make trustworthy.
    """


def validate_field(field_name: str, value: str | None) -> str | None:
    """Check one field against its vocabulary. None/blank means 'not set'."""
    if field_name not in ANNOTATION_FIELDS:
        raise InvalidAnnotationValue(f"unknown annotation field {field_name!r}")
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    allowed = ANNOTATION_FIELDS[field_name]
    if cleaned not in allowed:
        raise InvalidAnnotationValue(
            f"{ANNOTATION_FIELD_LABELS[field_name]} must be one of {list(allowed)}, got {cleaned!r}"
        )
    return cleaned


@dataclass
class Annotation:
    """One photographer diagnosis for one false negative.

    `image_hash` is the identity. `filename`, `original_path` and
    `capture_datetime` are metadata for display and are never matched on.

    The three diagnostic fields are each optional and independent: a
    photographer may judge the crop without judging the photograph, or record
    only whether they agree with the model.
    """

    image_hash: str
    filename: str
    original_path: str
    crop_quality: str | None = None
    image_quality: str | None = None
    agree_with_model_decision: str | None = None
    capture_datetime: str | None = None
    created_at: str = ""
    updated_at: str = ""
    # Set when the file is no longer at the path recorded with the annotation:
    # identity still matched, so the diagnosis followed the image. Surfaced so
    # the report can say the archive moved.
    relocated: bool = False
    # --- superseded fields, read-only ---------------------------------------
    # Whatever a pre-redesign database recorded. Never written any more and
    # never counted in the new breakdowns, but carried through so a report can
    # show that a record has history rather than silently rendering it blank.
    legacy_categories: list[str] = field(default_factory=list)
    legacy_primary_failure_cause: str | None = None
    legacy_notes: str = ""

    @property
    def is_empty(self) -> bool:
        """True when nothing is recorded, counting only the live fields.

        Legacy content deliberately does not keep a record alive: clearing all
        three dropdowns is how a mistaken annotation is deleted, and a leftover
        old note must not silently block that.
        """
        return not any(
            (self.crop_quality, self.image_quality, self.agree_with_model_decision)
        )

    @property
    def has_legacy_content(self) -> bool:
        return bool(
            self.legacy_categories or self.legacy_primary_failure_cause or self.legacy_notes.strip()
        )

    def as_dict(self) -> dict:
        return {
            "image_hash": self.image_hash,
            "filename": self.filename,
            "original_path": self.original_path,
            # Kept for the report JS, which indexes panels by their current path.
            "image_path": self.original_path,
            "capture_datetime": self.capture_datetime,
            "crop_quality": self.crop_quality,
            "image_quality": self.image_quality,
            "agree_with_model_decision": self.agree_with_model_decision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "relocated": self.relocated,
            "legacy_categories": list(self.legacy_categories),
            "legacy_primary_failure_cause": self.legacy_primary_failure_cause,
            "legacy_notes": self.legacy_notes,
            "has_legacy_content": self.has_legacy_content,
        }


@dataclass(frozen=True)
class UnresolvedImage:
    """An image whose identity could not be established, so it can neither be
    looked up nor annotated. Reported, never guessed at."""

    image_path: str
    reason: str

    @property
    def filename(self) -> str:
        return Path(self.image_path).name

    def as_dict(self) -> dict:
        return {"image_path": self.image_path, "filename": self.filename, "reason": self.reason}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MigrationReport:
    """What the automatic v1 -> v2 re-keying did, so it can be reported."""

    candidates: int = 0
    migrated: int = 0
    merged: int = 0
    unmigrated: list[UnresolvedImage] = field(default_factory=list)

    @property
    def ran(self) -> bool:
        return self.candidates > 0

    def as_dict(self) -> dict:
        return {
            "candidates": self.candidates,
            "migrated": self.migrated,
            "merged": self.merged,
            "unmigrated": [item.as_dict() for item in self.unmigrated],
        }

    def render(self) -> str:
        if not self.ran:
            return ""
        lines = [
            f"Annotation identity migration: {self.candidates} path-keyed record(s) found",
            f"  re-keyed to content identity: {self.migrated}",
        ]
        if self.merged:
            lines.append(f"  merged into existing records:  {self.merged} (same image, two paths)")
        if self.unmigrated:
            lines.append(f"  could not resolve:             {len(self.unmigrated)} (kept, will retry next run)")
            for item in self.unmigrated[:5]:
                lines.append(f"    {item.filename}: {item.reason}")
        return "\n".join(lines)


class AnnotationStore:
    """SQLite-backed store. Safe to open repeatedly; creates its schema."""

    def __init__(self, db_path: str | Path = DEFAULT_ANNOTATIONS_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connect()

    def _connect(self) -> None:
        # check_same_thread=False: the local annotation server is threaded, and
        # every write below is a short committed transaction.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # WAL keeps a reader (report generation) from blocking a writer (the UI).
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS categories (
                    name     TEXT PRIMARY KEY,
                    ordering INTEGER NOT NULL DEFAULT 100,
                    builtin  INTEGER NOT NULL DEFAULT 0
                );

                -- image_hash is the identity; filename/original_path/
                -- capture_datetime are metadata and are never matched on.
                CREATE TABLE IF NOT EXISTS annotations_v2 (
                    image_hash       TEXT PRIMARY KEY,
                    filename         TEXT NOT NULL,
                    original_path    TEXT NOT NULL,
                    capture_datetime TEXT,
                    notes            TEXT NOT NULL DEFAULT '',
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annotation_categories_v2 (
                    image_hash TEXT NOT NULL,
                    category   TEXT NOT NULL,
                    PRIMARY KEY (image_hash, category),
                    FOREIGN KEY (image_hash) REFERENCES annotations_v2(image_hash) ON DELETE CASCADE
                );

                -- Identity costs ~1 MB of reading per image, so it is memoised
                -- against (path, size, mtime): a repeat run over an unchanged
                -- archive does no I/O at all. Purely a cache - deleting it only
                -- costs time.
                CREATE TABLE IF NOT EXISTS identity_cache (
                    path       TEXT PRIMARY KEY,
                    size       INTEGER NOT NULL,
                    mtime_ns   INTEGER NOT NULL,
                    image_hash TEXT NOT NULL
                );

                -- v1 rows that could not be re-keyed (the file was not where
                -- the annotation said). Kept rather than dropped so nothing is
                -- silently lost and a later run can retry.
                CREATE TABLE IF NOT EXISTS unmigrated_v1 (
                    image_key  TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    filename   TEXT NOT NULL,
                    notes      TEXT NOT NULL DEFAULT '',
                    categories TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reason     TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_v2_filename ON annotations_v2(filename);
                CREATE INDEX IF NOT EXISTS idx_v2_updated  ON annotations_v2(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_v2_category ON annotation_categories_v2(category);
                """
            )
            if not self._conn.execute("SELECT 1 FROM schema_info").fetchone():
                self._conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            # The `categories` table is no longer seeded: that vocabulary is
            # superseded, and populating a fresh database with names nothing can
            # write would be misleading. The table itself stays so an existing
            # database keeps its history readable.
        # Additive, guarded upgrades for databases created by earlier versions.
        # The superseded columns are listed too: a database that never had them
        # still needs them to exist so the read path can select uniformly.
        for column in (
            "primary_failure_cause",  # superseded, read-only
            "crop_quality",
            "image_quality",
            "agree_with_model_decision",
        ):
            self._ensure_column("annotations_v2", column, "TEXT")
        self.migration = self._migrate_v1()

    def _ensure_column(self, table: str, column: str, sql_type: str) -> None:
        """Add a column to an existing table if it is not already there.

        Guarded, additive schema changes for a database that already exists on
        disk from before a feature was added - `CREATE TABLE IF NOT EXISTS`
        alone does not retrofit a column onto a table that was created by an
        older version of this module. Checked via PRAGMA rather than attempted
        and caught, so it is silent and idempotent on every open.
        """
        columns = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            with self._conn:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            logger.info("Added column %s.%s (upgrading an existing database)", table, column)

    # -- migration ----------------------------------------------------------

    def _has_v1_table(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='annotations'"
        ).fetchone()
        return row is not None

    def _migrate_v1(self) -> "MigrationReport":
        """Re-key path-identified v1 annotations onto content identity.

        Runs automatically on open, is idempotent (a migrated row is removed
        from the v1 table), and never loses anything: rows whose file cannot be
        found are parked in `unmigrated_v1` with the reason, and two v1 rows
        that turn out to be the same image are merged rather than duplicated.
        """
        report = MigrationReport()
        if not self._has_v1_table():
            return report

        rows = self._conn.execute("SELECT * FROM annotations").fetchall()
        if not rows:
            return report
        report.candidates = len(rows)
        logger.info("Migrating %d path-keyed annotation(s) to content identity", len(rows))

        for row in rows:
            old_key = row["image_key"]
            path = row["image_path"]
            categories = [
                item["category"]
                for item in self._conn.execute(
                    "SELECT category FROM annotation_categories WHERE image_key = ?", (old_key,)
                )
            ]
            try:
                new_hash = self._identity_of(path)
            except IdentityUnavailable as exc:
                with self._conn:
                    self._conn.execute(
                        """INSERT OR REPLACE INTO unmigrated_v1
                           (image_key, image_path, filename, notes, categories,
                            created_at, updated_at, reason)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            old_key,
                            path,
                            row["filename"],
                            row["notes"],
                            "|".join(categories),
                            row["created_at"],
                            row["updated_at"],
                            exc.reason,
                        ),
                    )
                report.unmigrated.append(UnresolvedImage(path, exc.reason))
                continue

            existing = self._conn.execute(
                "SELECT * FROM annotations_v2 WHERE image_hash = ?", (new_hash,)
            ).fetchone()
            with self._conn:
                if existing is None:
                    self._conn.execute(
                        """INSERT INTO annotations_v2
                           (image_hash, filename, original_path, capture_datetime,
                            notes, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            new_hash,
                            row["filename"],
                            path,
                            capture_datetime(path),
                            row["notes"],
                            row["created_at"],
                            row["updated_at"],
                        ),
                    )
                    report.migrated += 1
                else:
                    # Same image annotated twice under different paths (a copy,
                    # or a move recorded before this migration existed). Merge:
                    # union the categories, keep both notes, keep the earliest
                    # creation and the latest update.
                    merged_notes = "\n".join(
                        part for part in (existing["notes"], row["notes"]) if part.strip()
                    )
                    self._conn.execute(
                        """UPDATE annotations_v2
                           SET notes = ?, created_at = MIN(created_at, ?), updated_at = MAX(updated_at, ?)
                           WHERE image_hash = ?""",
                        (merged_notes, row["created_at"], row["updated_at"], new_hash),
                    )
                    report.merged += 1

                for category in categories:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO annotation_categories_v2(image_hash, category) VALUES (?, ?)",
                        (new_hash, category),
                    )
                # Idempotence: consumed rows leave the v1 table.
                self._conn.execute("DELETE FROM annotations WHERE image_key = ?", (old_key,))
                self._conn.execute(
                    "DELETE FROM annotation_categories WHERE image_key = ?", (old_key,)
                )

        with self._conn:
            self._conn.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION,))
        if report.migrated or report.merged or report.unmigrated:
            logger.info(
                "Migration: %d re-keyed, %d merged, %d could not be resolved",
                report.migrated,
                report.merged,
                len(report.unmigrated),
            )
        return report

    # -- identity -----------------------------------------------------------

    def _identity_of(self, image_path: str | Path) -> str:
        """Content identity, memoised against (path, size, mtime).

        Raises IdentityUnavailable - never returns a fallback.
        """
        path = Path(image_path)
        try:
            stat = path.stat()
        except OSError as exc:
            raise IdentityUnavailable(path, f"cannot stat file ({exc.strerror or exc})") from exc

        key = str(path.resolve())
        row = self._conn.execute(
            "SELECT image_hash FROM identity_cache WHERE path = ? AND size = ? AND mtime_ns = ?",
            (key, stat.st_size, stat.st_mtime_ns),
        ).fetchone()
        if row is not None:
            return row["image_hash"]

        digest = image_identity(path)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO identity_cache(path, size, mtime_ns, image_hash) VALUES (?, ?, ?, ?)",
                (key, stat.st_size, stat.st_mtime_ns, digest),
            )
        return digest

    def identity_of(self, image_path: str | Path) -> str:
        """Public identity lookup. Raises IdentityUnavailable."""
        return self._identity_of(image_path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AnnotationStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- vocabularies -------------------------------------------------------

    @staticmethod
    def field_vocabularies() -> dict[str, list[str]]:
        """The allowed values for each diagnostic field.

        A method on the store (rather than the UI importing the constants
        directly) so that whatever validates a save is exactly what the report
        offers, and the two cannot drift.
        """
        return {name: list(values) for name, values in ANNOTATION_FIELDS.items()}

    def legacy_categories(self) -> list[str]:
        """Category names a pre-redesign database recorded. Read-only."""
        rows = self._conn.execute("SELECT name FROM categories ORDER BY ordering, name").fetchall()
        return [row["name"] for row in rows]

    # -- reads --------------------------------------------------------------

    def _row_to_annotation(self, row: sqlite3.Row, current_path: str | None = None) -> Annotation:
        legacy_categories = [
            item["category"]
            for item in self._conn.execute(
                "SELECT category FROM annotation_categories_v2 WHERE image_hash = ? ORDER BY category",
                (row["image_hash"],),
            )
        ]
        relocated = current_path is not None and str(Path(current_path)) != str(
            Path(row["original_path"])
        )
        return Annotation(
            image_hash=row["image_hash"],
            filename=row["filename"],
            original_path=row["original_path"],
            # _ensure_column guarantees every column exists by the time any read
            # runs, even for a database created before these fields existed.
            crop_quality=row["crop_quality"],
            image_quality=row["image_quality"],
            agree_with_model_decision=row["agree_with_model_decision"],
            capture_datetime=row["capture_datetime"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            relocated=relocated,
            legacy_categories=legacy_categories,
            legacy_primary_failure_cause=row["primary_failure_cause"],
            legacy_notes=row["notes"] or "",
        )

    def get(self, image_path: str | Path) -> Annotation | None:
        """Look up one annotation by the image's content identity.

        Matching is by image_hash only - no path or filename fallback exists.
        Raises IdentityUnavailable if identity cannot be established, so the
        caller reports the condition instead of guessing.
        """
        digest = self._identity_of(image_path)
        row = self._conn.execute(
            "SELECT * FROM annotations_v2 WHERE image_hash = ?", (digest,)
        ).fetchone()
        return None if row is None else self._row_to_annotation(row, current_path=str(image_path))

    def get_many(
        self, image_paths: Iterable[str | Path]
    ) -> tuple[dict[str, Annotation], list[UnresolvedImage]]:
        """(path -> annotation, images whose identity could not be established).

        Unresolvable images are returned rather than skipped: "no annotation"
        and "could not tell which image this is" are different facts, and the
        second one needs saying out loud.
        """
        found: dict[str, Annotation] = {}
        unresolved: list[UnresolvedImage] = []
        for path in image_paths:
            try:
                annotation = self.get(path)
            except IdentityUnavailable as exc:
                unresolved.append(UnresolvedImage(str(path), exc.reason))
                continue
            if annotation is not None:
                found[str(path)] = annotation
        return found, unresolved

    def all(self) -> list[Annotation]:
        rows = self._conn.execute("SELECT * FROM annotations_v2 ORDER BY updated_at DESC").fetchall()
        return [self._row_to_annotation(row) for row in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM annotations_v2").fetchone()["n"])

    def unmigrated(self) -> list[UnresolvedImage]:
        """v1 annotations still awaiting a resolvable file."""
        rows = self._conn.execute("SELECT image_path, reason FROM unmigrated_v1").fetchall()
        return [UnresolvedImage(row["image_path"], row["reason"]) for row in rows]

    # -- writes -------------------------------------------------------------

    def save(
        self,
        image_path: str | Path,
        crop_quality: str | None = None,
        image_quality: str | None = None,
        agree_with_model_decision: str | None = None,
    ) -> Annotation:
        """Create or replace one annotation.

        Each field is validated against its fixed vocabulary and an unknown
        value raises rather than being stored - the whole point of the closed
        vocabularies is that the resulting counts can be trusted.

        Saving with all three fields empty deletes the record, so clearing the
        dropdowns in the UI is how a mistaken annotation is removed; there is no
        separate delete gesture to discover.
        """
        digest = self._identity_of(image_path)  # raises IdentityUnavailable
        crop = validate_field("crop_quality", crop_quality)
        image = validate_field("image_quality", image_quality)
        agree = validate_field("agree_with_model_decision", agree_with_model_decision)
        filename = Path(image_path).name

        if not any((crop, image, agree)):
            self.delete(image_path)
            return Annotation(image_hash=digest, filename=filename, original_path=str(image_path))

        now = _now()
        with self._conn:
            existing = self._conn.execute(
                "SELECT created_at, original_path FROM annotations_v2 WHERE image_hash = ?", (digest,)
            ).fetchone()
            created = existing["created_at"] if existing else now
            # The original path is kept as first recorded: it documents where the
            # image was when first diagnosed, and identity does the matching.
            original = existing["original_path"] if existing else str(Path(image_path))
            captured = capture_datetime(image_path)
            self._conn.execute(
                """
                INSERT INTO annotations_v2
                    (image_hash, filename, original_path, capture_datetime,
                     crop_quality, image_quality, agree_with_model_decision,
                     notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    filename                  = excluded.filename,
                    capture_datetime          = COALESCE(excluded.capture_datetime, annotations_v2.capture_datetime),
                    crop_quality              = excluded.crop_quality,
                    image_quality             = excluded.image_quality,
                    agree_with_model_decision = excluded.agree_with_model_decision,
                    updated_at                = excluded.updated_at
                """,
                (digest, filename, original, captured, crop, image, agree, created, now),
            )
            # Superseded columns are deliberately left untouched by the UPDATE
            # above: an old note or category set stays readable as history.

        logger.info(
            "Annotation saved for %s (crop=%s, image=%s, agree=%s)",
            filename,
            crop or "unset",
            image or "unset",
            agree or "unset",
        )
        saved = self.get(image_path)
        return saved if saved is not None else Annotation(
            image_hash=digest,
            filename=filename,
            original_path=original,
            crop_quality=crop,
            image_quality=image,
            agree_with_model_decision=agree,
            capture_datetime=captured,
            created_at=created,
            updated_at=now,
        )

    def delete(self, image_path: str | Path) -> bool:
        digest = self._identity_of(image_path)
        with self._conn:
            cursor = self._conn.execute("DELETE FROM annotations_v2 WHERE image_hash = ?", (digest,))
            self._conn.execute("DELETE FROM annotation_categories_v2 WHERE image_hash = ?", (digest,))
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Summary (the "False Negative Summary" report section)
# ---------------------------------------------------------------------------

@dataclass
class AnnotationSummary:
    """Aggregates over the knowledge base, restricted to the false negatives of
    the current run so the numbers describe this analysis, not the whole DB."""

    total_false_negatives: int = 0
    annotated: int = 0
    # field name -> [(value, count)], one breakdown per diagnostic field, each
    # ordered most frequent first. Only annotations that set that particular
    # field are counted in it, so the three need not sum to the same total.
    field_counts: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    # The distinct (crop, image, agree) triples actually observed, most common
    # first - the cross-field pattern a single-field breakdown cannot show.
    combination_counts: list[tuple[tuple[str, str, str], int]] = field(default_factory=list)
    recent: list[Annotation] = field(default_factory=list)
    unannotated: list[str] = field(default_factory=list)
    # The fixed vocabularies, so the UI renders the same options the store
    # validates against.
    field_vocabularies: dict[str, list[str]] = field(default_factory=dict)
    # Records still carrying pre-redesign content, so it can be reported rather
    # than silently ignored.
    with_legacy_content: int = 0
    database_path: str = ""
    total_in_database: int = 0
    # Images whose identity could not be established: reported explicitly,
    # never quietly treated as unannotated.
    unresolved: list[UnresolvedImage] = field(default_factory=list)
    relocated: int = 0
    migration: MigrationReport | None = None
    pending_migration: list[UnresolvedImage] = field(default_factory=list)

    @property
    def unannotated_count(self) -> int:
        return len(self.unannotated)

    @property
    def coverage(self) -> float | None:
        if not self.total_false_negatives:
            return None
        return self.annotated / self.total_false_negatives

    def as_dict(self) -> dict:
        return {
            "total_false_negatives": self.total_false_negatives,
            "annotated": self.annotated,
            "unannotated": self.unannotated_count,
            "coverage": self.coverage,
            "field_counts": {
                name: [{"value": value, "count": count} for value, count in counts]
                for name, counts in self.field_counts.items()
            },
            "combination_counts": [
                {
                    "crop_quality": combo[0],
                    "image_quality": combo[1],
                    "agree_with_model_decision": combo[2],
                    "count": count,
                }
                for combo, count in self.combination_counts
            ],
            "recent": [annotation.as_dict() for annotation in self.recent],
            "field_vocabularies": self.field_vocabularies,
            "with_legacy_content": self.with_legacy_content,
            "database_path": self.database_path,
            "total_in_database": self.total_in_database,
            "unresolved": [item.as_dict() for item in self.unresolved],
            "relocated": self.relocated,
            "migration": self.migration.as_dict() if self.migration else None,
            "pending_migration": [item.as_dict() for item in self.pending_migration],
        }


def summarise(
    store: AnnotationStore,
    false_negative_paths: Sequence[str],
    *,
    recent_limit: int = 15,
    combination_limit: int = 10,
) -> tuple[dict[str, Annotation], AnnotationSummary]:
    """Load annotations for this run's false negatives and aggregate them.

    Returns (path -> annotation, summary) so the caller does one pass.
    """
    found, unresolved = store.get_many(false_negative_paths)

    field_counters: dict[str, Counter[str]] = {name: Counter() for name in ANNOTATION_FIELDS}
    combination_counter: Counter[tuple[str, str, str]] = Counter()
    for annotation in found.values():
        for name in ANNOTATION_FIELDS:
            value = getattr(annotation, name)
            if value:
                field_counters[name][value] += 1
        # Unset fields participate as an explicit placeholder rather than being
        # dropped, so a combination row always describes the whole record.
        combination_counter[
            (
                annotation.crop_quality or "(unset)",
                annotation.image_quality or "(unset)",
                annotation.agree_with_model_decision or "(unset)",
            )
        ] += 1

    recent = sorted(found.values(), key=lambda a: a.updated_at, reverse=True)[:recent_limit]
    unresolved_paths = {item.image_path for item in unresolved}
    unannotated = [
        path for path in false_negative_paths if path not in found and path not in unresolved_paths
    ]

    return found, AnnotationSummary(
        total_false_negatives=len(false_negative_paths),
        annotated=len(found),
        field_counts={name: counter.most_common() for name, counter in field_counters.items()},
        combination_counts=combination_counter.most_common(combination_limit),
        recent=recent,
        unannotated=unannotated,
        field_vocabularies={name: list(values) for name, values in ANNOTATION_FIELDS.items()},
        with_legacy_content=sum(1 for a in found.values() if a.has_legacy_content),
        database_path=str(store.db_path),
        total_in_database=store.count(),
        unresolved=unresolved,
        relocated=sum(1 for annotation in found.values() if annotation.relocated),
        migration=getattr(store, "migration", None),
        pending_migration=store.unmigrated(),
    )


def render_summary(summary: AnnotationSummary) -> str:
    """Text form, for report.txt and the console."""
    lines = [
        "False negative annotations",
        "==========================",
        f"  database:            {summary.database_path}",
        f"  false negatives:     {summary.total_false_negatives:,}",
        f"  annotated:           {summary.annotated:,}"
        + (f" ({summary.coverage * 100:.1f}%)" if summary.coverage is not None else ""),
        f"  not yet annotated:   {summary.unannotated_count:,}",
        f"  records in database: {summary.total_in_database:,} (all runs)",
        "  identity:            content hash (survives renames, moves, relocation)",
    ]
    if summary.relocated:
        lines.append(
            f"  followed a move:     {summary.relocated:,} (file is no longer where it was annotated)"
        )
    if summary.unresolved:
        lines += ["", f"  IDENTITY UNAVAILABLE for {len(summary.unresolved):,} image(s) - not annotatable:"]
        for item in summary.unresolved[:5]:
            lines.append(f"    {item.filename}: {item.reason}")
        if len(summary.unresolved) > 5:
            lines.append(f"    ... and {len(summary.unresolved) - 5:,} more")
    if summary.migration is not None and summary.migration.ran:
        lines += ["", "  " + summary.migration.render().replace("\n", "\n  ")]
    if summary.pending_migration:
        lines.append(
            f"  {len(summary.pending_migration):,} old path-keyed record(s) await a resolvable file"
        )
    for field_name, counts in summary.field_counts.items():
        if not counts:
            continue
        lines += ["", f"  {ANNOTATION_FIELD_LABELS[field_name]}:"]
        answered = sum(count for _, count in counts)
        for value, count in counts:
            share = count / answered * 100 if answered else 0.0
            lines.append(f"    {value:<28}{count:>5,}  ({share:.0f}% of {answered} answered)")
    if summary.combination_counts:
        lines += ["", "  Most common combinations (crop / image / agree):"]
        for combo, count in summary.combination_counts:
            lines.append(f"    {count:>4,}x  {' / '.join(combo)}")
    if summary.with_legacy_content:
        lines += [
            "",
            f"  {summary.with_legacy_content:,} record(s) also carry pre-redesign notes or",
            "  categories, kept read-only and not counted above.",
        ]
    if summary.recent:
        lines += ["", "  Recently annotated:"]
        for annotation in summary.recent[:8]:
            fields = " / ".join(
                value or "-"
                for value in (
                    annotation.crop_quality,
                    annotation.image_quality,
                    annotation.agree_with_model_decision,
                )
            )
            lines.append(f"    {annotation.updated_at[:16]}  {annotation.filename:<28}{fields}")
    if not summary.annotated:
        lines += [
            "",
            "  No annotations yet. Run `picklikeme annotate --output <dir>` and open the",
            "  report to record why these images were missed.",
        ]
    return "\n".join(lines)
