"""Photographer diagnosis knowledge base, for the model's two kinds of mistake.

Why an image the photographer deliberately kept was rejected by the model - or
deliberately rejected was kept by it - is knowledge only the photographer has.
This module stores that knowledge and nothing else, using the same schema for
both mistakes so the two are directly comparable: a false negative and a false
positive get the same questions, the same fixed answers, and the same
statistics.

A diagnosis is a set of independent fixed-vocabulary fields, defined entirely
by `config/annotations.yaml` (see `annotation_config.py`) - not by this
module. Closed vocabularies with no free-text option, because the purpose is
data that can be counted: a growable tag list (which the superseded
`categories` field was) makes frequencies incomparable over time, since
"Backlit" and "backlighting" become two rows in a breakdown but one
phenomenon.

- **Config, not code.** Which fields exist, their labels, and their allowed
  values all come from `config/annotations.yaml`. The database stores each
  value's stable **id**, never its display label, so relabeling a value in
  config never touches a single existing annotation. Adding a field or a
  value needs no code change - restarting the analyzer is enough.
- **Never generated.** No heuristic, no model, nothing in this file infers a
  value. Every field comes from a human via the report UI. That extends to
  every migration this module does: nothing is ever guessed onto a field the
  photographer never actually answered.
- **Never influences metrics.** Annotations are attached to an AnalysisResult
  for display only. A test asserts every metric is bit-identical with and
  without an annotation database present.
- **Long-lived.** The database lives outside the analyzer's output directory,
  because output directories are per-run and get replaced; a knowledge base
  accumulated over months must not be inside one.
- **False negatives and false positives, identically.** Both are the model
  disagreeing with the photographer; both get the same panel, the same
  fields, the same vocabulary. Nothing in the schema or this module records
  which category an annotation came from - the record is a diagnosis of the
  image, not of the run that flagged it - so the split shown in a report is
  computed by asking the store about two path lists, one per category, not by
  a stored label. The detector-box thumbnail overlay is a separate, unrelated
  feature (analyzer.detections / analyzer.contactsheets) that applies more
  broadly still - report-wide, not just to annotatable images.

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

import json
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import PROJECT_ROOT, cli_prefix
from ..identity import IdentityUnavailable, cache_key, capture_datetime, image_identity
from .annotation_config import AnnotationFieldsConfig, load_annotation_fields

logger = logging.getLogger(__name__)

# Deliberately outside any analysis output directory: those are per-run and get
# overwritten, and this database is meant to outlive every one of them.
DEFAULT_ANNOTATIONS_DB = PROJECT_ROOT / "annotations" / "false_negatives.db"

# --- superseded vocabularies, retained for reading old records only ---------
# The growable category checklist and the single primary-cause radio that the
# config-driven fields replaced. Nothing writes these any more; they are kept
# so a database annotated before either change still renders its history
# instead of appearing empty.
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

# The three field names the pre-config-driven schema hardcoded as columns.
# Used only by the one-time migration into the generic `field_values` column;
# nothing else should ever reference these again.
_LEGACY_FIELD_COLUMNS: tuple[str, ...] = ("crop_quality", "image_quality", "agree_with_model_decision")

# v1 keyed annotations on a digest of the resolved path, which a rename would
# have orphaned. v2 keys them on content identity and migrates v1 rows across.
# v3 replaced three hardcoded per-field columns with one config-driven
# `field_values` JSON column; see `_migrate_legacy_fields`.
SCHEMA_VERSION = 3


class InvalidAnnotationValue(ValueError):
    """Raised when a field is given a value outside its fixed vocabulary, or
    a field id that config does not define.

    Rejected rather than stored, because a stray value would quietly corrupt
    exactly the frequency counts this schema exists to make trustworthy.
    """


# `picklikeme review`'s manual override. Two values and no config: unlike a
# diagnosis vocabulary, "keep or reject" is the workflow itself, not something
# a photographer would want to reword. Absence of a row means "no manual
# decision", which is different from either value.
REVIEW_KEEP = "keep"
REVIEW_REJECT = "reject"
REVIEW_DECISIONS: frozenset[str] = frozenset({REVIEW_KEEP, REVIEW_REJECT})


class InvalidReviewDecision(ValueError):
    """Raised for a review decision outside {keep, reject}."""


# Why a manual Keep/Reject overrides the model - optional, and only ever
# meaningful alongside a decision (see set_review_decision). Fixed, like the
# decision itself, rather than config-driven: this is the review workflow's
# own short list of common overrides, not a photographer's open vocabulary
# the way an annotation field's values are - REVIEW_REASON_OTHER plus
# `reason_note` is the escape hatch for anything this list doesn't cover.
REVIEW_REASON_EYES_NOT_SEEN = "eyes_not_seen"
REVIEW_REASON_CLEAR_EYES_SEEN = "clear_eyes_seen"
REVIEW_REASON_GOOD_QUALITY = "good_quality"
REVIEW_REASON_BAD_QUALITY = "bad_quality"
REVIEW_REASON_OTHER = "other"
REVIEW_REASONS: frozenset[str] = frozenset(
    {
        REVIEW_REASON_EYES_NOT_SEEN,
        REVIEW_REASON_CLEAR_EYES_SEEN,
        REVIEW_REASON_GOOD_QUALITY,
        REVIEW_REASON_BAD_QUALITY,
        REVIEW_REASON_OTHER,
    }
)


class InvalidReviewReason(ValueError):
    """Raised for a review reason outside REVIEW_REASONS (and not None)."""


def validate_field(fields_config: AnnotationFieldsConfig, field_id: str, value_id: str | None) -> str | None:
    """Check one field against the configured vocabulary. None/blank means
    'not set'."""
    field_def = fields_config.get(field_id)
    if field_def is None:
        raise InvalidAnnotationValue(
            f"unknown annotation field {field_id!r} (configured fields: {list(fields_config.field_ids)})"
        )
    cleaned = (value_id or "").strip()
    if not cleaned:
        return None
    if not field_def.has_value(cleaned):
        raise InvalidAnnotationValue(
            f"{field_def.label} must be one of {list(field_def.value_ids)}, got {cleaned!r}"
        )
    return cleaned


@dataclass
class Annotation:
    """One photographer diagnosis for one image - a false negative or a false
    positive; the record itself does not say which, see the module docstring.

    `image_hash` is the identity. `filename`, `original_path` and
    `capture_datetime` are metadata for display and are never matched on.

    `fields` maps a config-defined field id to the value id recorded for it -
    every field is optional and independent: a photographer may judge the
    crop without judging the photograph, or record only whether they agree
    with the model. It may also carry ids for a field or value that is no
    longer in the current config (a "retired" id, kept for display rather
    than silently dropped - see `AnnotationField.label_for`).
    """

    image_hash: str
    filename: str
    original_path: str
    fields: dict[str, str | None] = field(default_factory=dict)
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

        Legacy content deliberately does not keep a record alive: clearing
        every field is how a mistaken annotation is deleted, and a leftover
        old note must not silently block that.
        """
        return not any(self.fields.values())

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
            "fields": dict(self.fields),
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


@dataclass
class LegacyFieldMigrationReport:
    """What the one-time migration from the three hardcoded columns
    (crop_quality/image_quality/agree_with_model_decision) into the generic,
    config-driven `field_values` column did.

    Mirrors `MigrationReport`'s shape and philosophy: never silently drop a
    value. A legacy value that cannot be matched to the current config (the
    field or that exact label is no longer defined) is parked in
    `unmigrated_legacy_field_values`, not lost.
    """

    candidates: int = 0
    migrated: int = 0
    unmapped: int = 0

    @property
    def ran(self) -> bool:
        return self.candidates > 0

    def as_dict(self) -> dict:
        return {"candidates": self.candidates, "migrated": self.migrated, "unmapped": self.unmapped}

    def render(self) -> str:
        if not self.ran:
            return ""
        lines = [f"Legacy annotation field migration: {self.candidates} legacy value(s) found"]
        lines.append(f"  migrated to config-driven ids: {self.migrated}")
        if self.unmapped:
            lines.append(
                f"  could not be matched to config: {self.unmapped} "
                "(kept in unmigrated_legacy_field_values, not lost)"
            )
        return "\n".join(lines)


class AnnotationStore:
    """SQLite-backed store. Safe to open repeatedly; creates its schema."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_ANNOTATIONS_DB,
        fields_config: AnnotationFieldsConfig | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Injectable so a test (or a future caller) can validate against a
        # fixture config instead of the real config/annotations.yaml; the CLI
        # always passes one explicitly (see analyzer.cli / analyzer.server).
        self.fields_config = fields_config if fields_config is not None else load_annotation_fields()
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
                -- field_values is a JSON object {field_id: value_id, ...} -
                -- config-driven, so adding a field never changes this schema.
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

                -- A legacy (crop_quality/image_quality/agree_with_model_decision)
                -- value whose exact label no longer matches anything in the
                -- current config. Kept rather than dropped - see
                -- LegacyFieldMigrationReport.
                CREATE TABLE IF NOT EXISTS unmigrated_legacy_field_values (
                    image_hash      TEXT NOT NULL,
                    legacy_field_id TEXT NOT NULL,
                    legacy_label    TEXT NOT NULL,
                    PRIMARY KEY (image_hash, legacy_field_id),
                    FOREIGN KEY (image_hash) REFERENCES annotations_v2(image_hash) ON DELETE CASCADE
                );

                -- `picklikeme review`: the photographer's manual Keep/Reject,
                -- overriding the model's ordering. Its own table rather than an
                -- annotation field, because the two have different lifetimes
                -- and different meanings - see set_review_decision().
                --
                -- image_path is the last known location, kept so a whole
                -- session's decisions load in one query without resolving
                -- identity for every image; image_hash stays the identity and
                -- is what a lookup falls back to when a file has moved.
                -- Deliberately no FOREIGN KEY to annotations_v2: a decision
                -- must be able to exist for an image that was never diagnosed,
                -- and deleting a diagnosis must not erase a decision.
                CREATE TABLE IF NOT EXISTS review_decisions (
                    image_hash  TEXT PRIMARY KEY,
                    decision    TEXT NOT NULL,
                    reason      TEXT,
                    reason_note TEXT,
                    image_path  TEXT NOT NULL,
                    filename    TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_review_path ON review_decisions(image_path);

                -- Reading a capture timestamp costs a rawpy/PIL open per
                -- file - cheap once, not free at review-session scale (tens
                -- of thousands of images). Memoised against (path, size,
                -- mtime) exactly like identity_cache; captured_at is
                -- nullable because "this file has no EXIF capture time" is
                -- itself a cacheable, permanent answer, not a miss to retry.
                CREATE TABLE IF NOT EXISTS capture_time_cache (
                    path        TEXT PRIMARY KEY,
                    size        INTEGER NOT NULL,
                    mtime_ns    INTEGER NOT NULL,
                    captured_at TEXT
                );

                -- Same reasoning as capture_time_cache, for detected_category:
                -- review.thumbnails.detected_category_for's own cache is a
                -- per-image file on disk (a JSON sidecar written by
                -- preprocessing), which every review session was re-reading
                -- for every image on every single folder load before this
                -- table existed. Memoised here exactly like capture time, so
                -- only the first load after a file last changed pays for it.
                CREATE TABLE IF NOT EXISTS detected_category_cache (
                    path      TEXT PRIMARY KEY,
                    size      INTEGER NOT NULL,
                    mtime_ns  INTEGER NOT NULL,
                    category  TEXT
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
        # still needs them to exist so the migration below can read them
        # uniformly, whether or not it has ever run before.
        for column in (
            "primary_failure_cause",  # superseded, read-only
            "crop_quality",  # superseded (v2), read-only - see _migrate_legacy_fields
            "image_quality",  # superseded (v2), read-only
            "agree_with_model_decision",  # superseded (v2), read-only
            "field_values",  # v3: JSON {field_id: value_id}, config-driven
        ):
            sql_type = "TEXT"
            self._ensure_column("annotations_v2", column, sql_type)
        # review_decisions predates the reason/reason_note columns - a database
        # from before those features has the table but not the columns.
        self._ensure_column("review_decisions", "reason", "TEXT")
        self._ensure_column("review_decisions", "reason_note", "TEXT")
        self.migration = self._migrate_v1()
        self.legacy_field_migration = self._migrate_legacy_fields()
        with self._conn:
            self._conn.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION,))
        for warning in self._retired_id_warnings():
            logger.warning(warning)

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

        if report.migrated or report.merged or report.unmigrated:
            logger.info(
                "Migration: %d re-keyed, %d merged, %d could not be resolved",
                report.migrated,
                report.merged,
                len(report.unmigrated),
            )
        return report

    def _migrate_legacy_fields(self) -> "LegacyFieldMigrationReport":
        """Fold the three legacy per-field columns into the generic,
        config-driven `field_values` JSON column.

        Runs on every open but only fills gaps: a (image_hash, field_id) that
        already has an entry in field_values is left alone, so re-answering a
        field through the new UI is never clobbered by an old legacy value.
        Idempotent and non-destructive - the legacy columns themselves are
        never modified or dropped.

        A legacy value is matched to the current config by exact label text
        (that is literally what the legacy columns stored). When the field or
        that exact label no longer exists in config, the value is parked in
        unmigrated_legacy_field_values rather than lost - see the class
        docstring on LegacyFieldMigrationReport.
        """
        report = LegacyFieldMigrationReport()
        columns = ", ".join(_LEGACY_FIELD_COLUMNS)
        rows = self._conn.execute(
            f"SELECT image_hash, field_values, {columns} FROM annotations_v2"
        ).fetchall()

        for row in rows:
            legacy_values = {name: row[name] for name in _LEGACY_FIELD_COLUMNS if row[name]}
            if not legacy_values:
                continue
            current = json.loads(row["field_values"]) if row["field_values"] else {}
            changed = False
            for legacy_field_id, legacy_label in legacy_values.items():
                if legacy_field_id in current:
                    continue  # already migrated, or answered again since
                report.candidates += 1
                field_def = self.fields_config.get(legacy_field_id)
                matched = None
                if field_def is not None:
                    for value in field_def.values:
                        if value.label == legacy_label:
                            matched = value.id
                            break
                if matched is not None:
                    current[legacy_field_id] = matched
                    report.migrated += 1
                    changed = True
                else:
                    with self._conn:
                        self._conn.execute(
                            """INSERT OR REPLACE INTO unmigrated_legacy_field_values
                               (image_hash, legacy_field_id, legacy_label) VALUES (?, ?, ?)""",
                            (row["image_hash"], legacy_field_id, legacy_label),
                        )
                    report.unmapped += 1
            if changed:
                with self._conn:
                    self._conn.execute(
                        "UPDATE annotations_v2 SET field_values = ? WHERE image_hash = ?",
                        (json.dumps(current), row["image_hash"]),
                    )

        if report.ran:
            logger.info(
                "Legacy field migration: %d migrated, %d could not be matched to config",
                report.migrated,
                report.unmapped,
            )
        return report

    def _retired_id_warnings(self) -> list[str]:
        """Field/value ids with real historical usage that the current config
        no longer defines - a rename or removal of an id already saved.

        Reported as warnings, not a startup failure: the report still renders
        (AnnotationField.label_for falls back to the raw id), only *selecting*
        the retired value again is what's actually lost.
        """
        warnings: list[str] = []
        rows = self._conn.execute(
            "SELECT field_values FROM annotations_v2 WHERE field_values IS NOT NULL"
        ).fetchall()
        usage: dict[tuple[str, str], int] = {}
        for row in rows:
            for field_id, value_id in json.loads(row["field_values"]).items():
                if value_id:
                    usage[(field_id, value_id)] = usage.get((field_id, value_id), 0) + 1

        for (field_id, value_id), count in sorted(usage.items()):
            field_def = self.fields_config.get(field_id)
            if field_def is None:
                warnings.append(
                    f"annotation field {field_id!r} is used by {count} existing annotation(s) but is "
                    f"no longer defined in config/annotations.yaml. Historical annotations still "
                    "display it; restore the field (its label can change safely) to make it "
                    "selectable again."
                )
            elif not field_def.has_value(value_id):
                warnings.append(
                    f"annotation field {field_id!r} value {value_id!r} is used by {count} existing "
                    "annotation(s) but is no longer defined in config/annotations.yaml. Historical "
                    "annotations still display it (as a retired value); it can no longer be selected. "
                    "Restore the same id (its label can change safely) if this was meant to be a rename."
                )
        return warnings

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

    def _cached_per_file(self, table: str, column: str, image_path: str | Path, compute) -> Any:
        """A value computed from one file's current bytes, memoised against
        (path, size, mtime) so it is computed at most once per version of
        that file, not once per lookup - the shape identity_cache and
        capture_time_cache both already use. `table` must exist with columns
        (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, <column>);
        `table`/`column` are always this module's own literal constants,
        never caller input.

        A file that cannot be stat'd (missing, permissions) has nothing to
        memoise against and returns None without calling `compute` at all -
        the same "absence is not itself cached" rule identity_of's raising
        cousin follows, since there is no (size, mtime) to key it by.
        """
        path = Path(image_path)
        try:
            stat = path.stat()
        except OSError:
            return None

        key = str(path.resolve())
        row = self._conn.execute(
            f"SELECT {column} FROM {table} WHERE path = ? AND size = ? AND mtime_ns = ?",  # noqa: S608
            (key, stat.st_size, stat.st_mtime_ns),
        ).fetchone()
        if row is not None:
            return row[column]

        value = compute(str(path))
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table}(path, size, mtime_ns, {column}) VALUES (?, ?, ?, ?)",  # noqa: S608
                (key, stat.st_size, stat.st_mtime_ns, value),
            )
        return value

    def capture_timestamp_of(self, image_path: str | Path) -> str | None:
        """The image's own EXIF capture date/time, memoised against
        (path, size, mtime) - see contactsheets.read_capture_timestamp for
        what is actually read. Absence (no EXIF, or none readable) is cached
        too, so a file with no capture time is also only ever read once.
        """
        from .contactsheets import read_capture_timestamp

        return self._cached_per_file("capture_time_cache", "captured_at", image_path, read_capture_timestamp)

    def detected_category_of(self, image_path: str | Path, compute) -> str | None:
        """The subject category already recorded for this image (see
        bird_crop.DETECTION_CATEGORIES), memoised the same way
        capture_timestamp_of is.

        `compute` is injected rather than imported directly: the actual
        lookup (review.thumbnails.detected_category_for) reads the
        detector's own cache, most often a per-image JSON sidecar file on
        disk - real I/O this module must not pay for on every single lookup,
        the same reasoning that justified capture_time_cache. Injecting it
        also means this module - shared by the whole app - never needs to
        depend on the review package's detection-cache wiring; see
        review/session.py's call site for the actual function passed in.
        """
        return self._cached_per_file("detected_category_cache", "category", image_path, compute)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AnnotationStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- vocabularies -------------------------------------------------------

    def field_vocabularies(self) -> dict[str, list[str]]:
        """The allowed value ids for each configured diagnostic field.

        Reads self.fields_config (rather than the UI importing constants
        directly) so that whatever validates a save is exactly what the report
        offers, and the two cannot drift.
        """
        return {f.id: list(f.value_ids) for f in self.fields_config}

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
        fields = json.loads(row["field_values"]) if row["field_values"] else {}
        return Annotation(
            image_hash=row["image_hash"],
            filename=row["filename"],
            original_path=row["original_path"],
            fields=fields,
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

    def save(self, image_path: str | Path, fields: dict[str, str | None] | None = None) -> Annotation:
        """Create or replace one annotation.

        `fields` maps a configured field id to a value id (or None/blank to
        leave it unset). Each is validated against `self.fields_config` and an
        unknown field id or out-of-vocabulary value raises rather than being
        stored - the whole point of the closed vocabularies is that the
        resulting counts can be trusted.

        Saving with every field empty (or `fields` omitted entirely) deletes
        the record, so clearing every dropdown in the UI is how a mistaken
        annotation is removed; there is no separate delete gesture to discover.
        """
        digest = self._identity_of(image_path)  # raises IdentityUnavailable
        cleaned = {
            field_id: validate_field(self.fields_config, field_id, value_id)
            for field_id, value_id in (fields or {}).items()
        }
        cleaned = {field_id: value_id for field_id, value_id in cleaned.items() if value_id}
        filename = Path(image_path).name

        if not cleaned:
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
            payload = json.dumps(cleaned)
            self._conn.execute(
                """
                INSERT INTO annotations_v2
                    (image_hash, filename, original_path, capture_datetime,
                     field_values, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, '', ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    filename                  = excluded.filename,
                    capture_datetime          = COALESCE(excluded.capture_datetime, annotations_v2.capture_datetime),
                    field_values              = excluded.field_values,
                    updated_at                = excluded.updated_at
                """,
                (digest, filename, original, captured, payload, created, now),
            )
            # Superseded columns (crop_quality/image_quality/agree_with_model_decision,
            # notes, primary_failure_cause) are deliberately left untouched by
            # the UPDATE above: old content stays readable as history.

        logger.info("Annotation saved for %s (%s)", filename, ", ".join(f"{k}={v}" for k, v in cleaned.items()))
        saved = self.get(image_path)
        return saved if saved is not None else Annotation(
            image_hash=digest,
            filename=filename,
            original_path=original,
            fields=cleaned,
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

    # -- review decisions ---------------------------------------------------
    #
    # `picklikeme review`'s manual Keep/Reject. Kept in this store, and this
    # file, because it is irreplaceable human input - the same reason the
    # diagnoses live here rather than in a recomputable cache (see
    # analyzer/detections.py's module docstring for the rule).
    #
    # A separate table rather than a configured annotation field, deliberately:
    # save() replaces field_values wholesale, so a review write would erase the
    # photographer's diagnosis dropdowns; saving with every field empty deletes
    # the row entirely; and summarise() would fold keep/reject counts into the
    # false-negative/false-positive breakdowns those statistics exist to keep
    # trustworthy. The two are different facts with different lifetimes.

    def set_review_decision(
        self,
        image_path: str | Path,
        decision: str,
        *,
        reason: str | None = None,
        reason_note: str | None = None,
    ) -> str:
        """Record a manual Keep or Reject, with an optional reason for the
        override. Raises IdentityUnavailable.

        Keyed on content identity, so the decision follows the image through
        the rename that `organize` performs moments later. `reason` (and
        `reason_note`) fully replace whatever this image already had - there
        is no partial update, matching the decision they are meaningless
        without. `reason_note` is free text and only means anything alongside
        REVIEW_REASON_OTHER; given with any other reason (or none), it is
        silently dropped rather than stored somewhere it can't apply to.
        """
        if decision not in REVIEW_DECISIONS:
            raise InvalidReviewDecision(
                f"decision must be one of {sorted(REVIEW_DECISIONS)}, got {decision!r}"
            )
        if reason is not None and reason not in REVIEW_REASONS:
            raise InvalidReviewReason(
                f"reason must be one of {sorted(REVIEW_REASONS)} or null, got {reason!r}"
            )
        if reason != REVIEW_REASON_OTHER:
            reason_note = None
        digest = self._identity_of(image_path)  # raises IdentityUnavailable
        path = Path(image_path)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO review_decisions
                    (image_hash, decision, reason, reason_note, image_path, filename, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    decision    = excluded.decision,
                    reason      = excluded.reason,
                    reason_note = excluded.reason_note,
                    image_path  = excluded.image_path,
                    filename    = excluded.filename,
                    updated_at  = excluded.updated_at
                """,
                (digest, decision, reason, reason_note, str(path), path.name, _now()),
            )
        return digest

    def clear_review_decision(self, image_path: str | Path) -> bool:
        """Drop back to whatever the threshold says. Raises IdentityUnavailable."""
        digest = self._identity_of(image_path)
        with self._conn:
            cursor = self._conn.execute("DELETE FROM review_decisions WHERE image_hash = ?", (digest,))
        return cursor.rowcount > 0

    def review_decisions(self) -> list[dict]:
        """Every recorded decision, in one query.

        Bulk rather than per-image because a review session opens on thousands
        of images at once: resolving content identity for each of them just to
        discover most have never been decided would cost minutes on a cold
        cache (see identity.py's measurements). The caller matches on
        `image_path` first and falls back to `image_hash` only for the few rows
        that miss - see picklikeme.review.session.
        """
        rows = self._conn.execute(
            "SELECT image_hash, decision, reason, reason_note, image_path, filename, updated_at FROM review_decisions"
        ).fetchall()
        return [dict(row) for row in rows]

    def review_decision_count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) AS n FROM review_decisions").fetchone()["n"]
        )

    def repoint_review_decisions(self, moves: dict[str, "Path"]) -> int:
        """Follow the images that arranging just moved.

        `image_hash` already identifies them, so nothing is *lost* without this
        - but `image_path` is what makes a whole session's decisions load in
        one query, and every path in it goes stale the moment files are filed
        into `_Selected`/`_Rejected`. Repointing from the exact old -> new map
        keeps the fast path fast on the next review, at the cost of one UPDATE
        per moved file. The same reasoning as sidecar.rewrite_ranking_paths.
        """
        if not moves:
            return 0
        updated = 0
        now = _now()
        with self._conn:
            for old, new in moves.items():
                new_path = Path(new)
                cursor = self._conn.execute(
                    """UPDATE review_decisions
                       SET image_path = ?, filename = ?, updated_at = ?
                       WHERE image_path = ?""",
                    (str(new_path), new_path.name, now, str(old)),
                )
                updated += cursor.rowcount
        if updated:
            logger.info("Repointed %d review decision(s) after arranging", updated)
        return updated


# ---------------------------------------------------------------------------
# Summary (the "False Negative summary" / "False Positive summary" report
# sections - one AnnotationSummary per category, built by calling summarise()
# once per path list, so the two are directly comparable field for field.
# ---------------------------------------------------------------------------

UNSET_COMBINATION_PLACEHOLDER = "(unset)"


@dataclass
class AnnotationSummary:
    """Aggregates over the knowledge base, restricted to one outcome category
    (the false negatives, or the false positives, of the current run) so the
    numbers describe this analysis, not the whole DB."""

    total_images: int = 0
    annotated: int = 0
    # field id -> [(value id, count)], one breakdown per configured field,
    # each ordered most frequent first. Only annotations that set that
    # particular field are counted in it, so the fields need not sum to the
    # same total.
    field_counts: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    # The distinct value-id combinations actually observed across every
    # configured field (in config order), most common first - the cross-field
    # pattern a single-field breakdown cannot show.
    combination_counts: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    # The field ids `combination_counts` tuples are positional over, in order.
    combination_fields: tuple[str, ...] = ()
    recent: list[Annotation] = field(default_factory=list)
    unannotated: list[str] = field(default_factory=list)
    # The configured vocabularies (value ids), so the UI renders the same
    # options the store validates against.
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
        if not self.total_images:
            return None
        return self.annotated / self.total_images

    def as_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "annotated": self.annotated,
            "unannotated": self.unannotated_count,
            "coverage": self.coverage,
            "field_counts": {
                name: [{"value": value, "count": count} for value, count in counts]
                for name, counts in self.field_counts.items()
            },
            "combination_counts": [
                {"values": dict(zip(self.combination_fields, combo)), "count": count}
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
    image_paths: Sequence[str],
    *,
    recent_limit: int = 15,
    combination_limit: int = 10,
) -> tuple[dict[str, Annotation], AnnotationSummary]:
    """Load annotations for one outcome category's images and aggregate them.

    `image_paths` is whichever set the caller wants a breakdown for - the
    false negatives of this run, or the false positives, called once each so
    the two summaries are computed identically and are directly comparable.

    Returns (path -> annotation, summary) so the caller does one pass.
    """
    found, unresolved = store.get_many(image_paths)
    field_ids = store.fields_config.field_ids

    field_counters: dict[str, Counter[str]] = {field_id: Counter() for field_id in field_ids}
    combination_counter: Counter[tuple[str, ...]] = Counter()
    for annotation in found.values():
        for field_id in field_ids:
            value = annotation.fields.get(field_id)
            if value:
                field_counters[field_id][value] += 1
        # Unset fields participate as an explicit placeholder rather than being
        # dropped, so a combination row always describes the whole record.
        combination_counter[
            tuple(annotation.fields.get(field_id) or UNSET_COMBINATION_PLACEHOLDER for field_id in field_ids)
        ] += 1

    recent = sorted(found.values(), key=lambda a: a.updated_at, reverse=True)[:recent_limit]
    unresolved_paths = {item.image_path for item in unresolved}
    unannotated = [
        path for path in image_paths if path not in found and path not in unresolved_paths
    ]

    return found, AnnotationSummary(
        total_images=len(image_paths),
        annotated=len(found),
        field_counts={name: counter.most_common() for name, counter in field_counters.items()},
        combination_counts=combination_counter.most_common(combination_limit),
        combination_fields=field_ids,
        recent=recent,
        unannotated=unannotated,
        field_vocabularies=store.field_vocabularies(),
        with_legacy_content=sum(1 for a in found.values() if a.has_legacy_content),
        database_path=str(store.db_path),
        total_in_database=store.count(),
        unresolved=unresolved,
        relocated=sum(1 for annotation in found.values() if annotation.relocated),
        migration=getattr(store, "migration", None),
        pending_migration=store.unmigrated(),
    )


def render_summary(
    summary: AnnotationSummary, fields_config: AnnotationFieldsConfig, *, title: str = "Annotations", item_label: str = "images"
) -> str:
    """Text form, for report.txt and the console.

    `title` and `item_label` let the same renderer serve either category
    ("False negative annotations" / "false negatives", or "False positive
    annotations" / "false positives") without duplicating this function.
    `fields_config` supplies display labels for the field ids `summary`
    carries (which are stable ids, not labels).
    """
    field_labels = {f.id: f.label for f in fields_config}

    def _label_for(field_id: str, value_id: str) -> str:
        field_def = fields_config.get(field_id)
        return field_def.label_for(value_id) if field_def is not None else value_id

    lines = [
        title,
        "=" * len(title),
        f"  database:            {summary.database_path}",
        f"  {item_label + ':':<21}{summary.total_images:,}",
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
    for field_id, counts in summary.field_counts.items():
        if not counts:
            continue
        label = field_labels.get(field_id, field_id)
        lines += ["", f"  {label}:"]
        answered = sum(count for _, count in counts)
        for value, count in counts:
            share = count / answered * 100 if answered else 0.0
            display = _label_for(field_id, value)
            lines.append(f"    {display:<28}{count:>5,}  ({share:.0f}% of {answered} answered)")
    if summary.combination_counts:
        header = " / ".join(field_labels.get(fid, fid) for fid in summary.combination_fields)
        lines += ["", f"  Most common combinations ({header}):"]
        for combo, count in summary.combination_counts:
            display = " / ".join(
                value if value == UNSET_COMBINATION_PLACEHOLDER else _label_for(field_id, value)
                for field_id, value in zip(summary.combination_fields, combo)
            )
            lines.append(f"    {count:>4,}x  {display}")
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
                _label_for(fid, annotation.fields[fid]) if annotation.fields.get(fid) else "-"
                for fid in summary.combination_fields
            )
            lines.append(f"    {annotation.updated_at[:16]}  {annotation.filename:<28}{fields}")
    if not summary.annotated:
        lines += [
            "",
            f"  No annotations yet. Run `{cli_prefix()} annotate --output <dir>` and open",
            "  the report to record your diagnosis of these images.",
        ]
    return "\n".join(lines)
