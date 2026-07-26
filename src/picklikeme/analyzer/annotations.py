"""False-negative knowledge base: the photographer's own diagnoses, persisted.

Why an image the photographer deliberately kept was rejected by the model is
knowledge only the photographer has. This module stores that knowledge and
nothing else:

- **Never generated.** No heuristic, no model, nothing in this file infers a
  category or writes a note. Every field comes from a human via the report UI.
- **Never influences metrics.** Annotations are attached to an AnalysisResult
  for display only. A test asserts every metric is bit-identical with and
  without an annotation database present.
- **Long-lived.** The database lives outside the analyzer's output directory,
  because output directories are per-run and get replaced; a knowledge base
  accumulated over months must not be inside one.
- **False negatives only**, by request. Nothing here is wired to false
  positives.

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

# The starting vocabulary. Stored in the database on first use so it can grow
# without a code change; `builtin` marks these so a future UI could separate
# them from ones the photographer added.
INITIAL_CATEGORIES: tuple[str, ...] = (
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

# v1 keyed annotations on a digest of the resolved path, which a rename would
# have orphaned. v2 keys them on content identity and migrates v1 rows across.
SCHEMA_VERSION = 2


@dataclass
class Annotation:
    """One photographer diagnosis for one false negative.

    `image_hash` is the identity. `filename`, `original_path` and
    `capture_datetime` are metadata for display and are never matched on.
    """

    image_hash: str
    filename: str
    original_path: str
    categories: list[str] = field(default_factory=list)
    notes: str = ""
    capture_datetime: str | None = None
    created_at: str = ""
    updated_at: str = ""
    # Set when the file is no longer at the path recorded with the annotation:
    # identity still matched, so the diagnosis followed the image. Surfaced so
    # the report can say the archive moved.
    relocated: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.categories and not self.notes.strip()

    def as_dict(self) -> dict:
        return {
            "image_hash": self.image_hash,
            "filename": self.filename,
            "original_path": self.original_path,
            # Kept for the report JS, which indexes panels by their current path.
            "image_path": self.original_path,
            "capture_datetime": self.capture_datetime,
            "categories": list(self.categories),
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "relocated": self.relocated,
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
            for ordering, name in enumerate(INITIAL_CATEGORIES):
                self._conn.execute(
                    "INSERT OR IGNORE INTO categories(name, ordering, builtin) VALUES (?, ?, 1)",
                    (name, ordering),
                )
        self.migration = self._migrate_v1()

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

    # -- categories ---------------------------------------------------------

    def categories(self) -> list[str]:
        rows = self._conn.execute("SELECT name FROM categories ORDER BY ordering, name").fetchall()
        return [row["name"] for row in rows]

    def add_category(self, name: str) -> None:
        """Register a category the photographer invented, so it appears in the
        checklist on later runs."""
        name = name.strip()
        if not name:
            return
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO categories(name, ordering, builtin) VALUES (?, 500, 0)", (name,)
            )

    # -- reads --------------------------------------------------------------

    def _row_to_annotation(self, row: sqlite3.Row, current_path: str | None = None) -> Annotation:
        categories = [
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
            categories=categories,
            notes=row["notes"],
            capture_datetime=row["capture_datetime"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            relocated=relocated,
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

    def save(self, image_path: str | Path, categories: Sequence[str], notes: str = "") -> Annotation:
        """Create or replace one annotation.

        Saving with no categories and no notes deletes the record, so clearing
        the panel in the UI is how a mistaken annotation is removed - there is
        no separate delete gesture to discover.
        """
        digest = self._identity_of(image_path)  # raises IdentityUnavailable
        cleaned = [c.strip() for c in categories if c and c.strip()]
        notes = notes.strip()
        filename = Path(image_path).name

        if not cleaned and not notes:
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
                    (image_hash, filename, original_path, capture_datetime, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    filename         = excluded.filename,
                    capture_datetime = COALESCE(excluded.capture_datetime, annotations_v2.capture_datetime),
                    notes            = excluded.notes,
                    updated_at       = excluded.updated_at
                """,
                (digest, filename, original, captured, notes, created, now),
            )
            self._conn.execute("DELETE FROM annotation_categories_v2 WHERE image_hash = ?", (digest,))
            for category in cleaned:
                self._conn.execute(
                    "INSERT OR IGNORE INTO annotation_categories_v2(image_hash, category) VALUES (?, ?)",
                    (digest, category),
                )
                # Free-text categories are remembered so they show up next time.
                self._conn.execute(
                    "INSERT OR IGNORE INTO categories(name, ordering, builtin) VALUES (?, 500, 0)",
                    (category,),
                )

        logger.info("Annotation saved for %s (%d categories)", filename, len(cleaned))
        return Annotation(
            image_hash=digest,
            filename=filename,
            original_path=original,
            categories=sorted(cleaned),
            notes=notes,
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
    category_counts: list[tuple[str, int]] = field(default_factory=list)
    combination_counts: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    recent: list[Annotation] = field(default_factory=list)
    unannotated: list[str] = field(default_factory=list)
    known_categories: list[str] = field(default_factory=list)
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
            "category_counts": [{"category": name, "count": count} for name, count in self.category_counts],
            "combination_counts": [
                {"categories": list(combo), "count": count} for combo, count in self.combination_counts
            ],
            "recent": [annotation.as_dict() for annotation in self.recent],
            "known_categories": self.known_categories,
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

    category_counter: Counter[str] = Counter()
    combination_counter: Counter[tuple[str, ...]] = Counter()
    for annotation in found.values():
        category_counter.update(annotation.categories)
        if len(annotation.categories) > 1:
            # Sorted so {Action shot, Lighting} and {Lighting, Action shot} are
            # counted as the same combination.
            combination_counter[tuple(sorted(annotation.categories))] += 1

    recent = sorted(found.values(), key=lambda a: a.updated_at, reverse=True)[:recent_limit]
    unresolved_paths = {item.image_path for item in unresolved}
    unannotated = [
        path for path in false_negative_paths if path not in found and path not in unresolved_paths
    ]

    return found, AnnotationSummary(
        total_false_negatives=len(false_negative_paths),
        annotated=len(found),
        category_counts=category_counter.most_common(),
        combination_counts=combination_counter.most_common(combination_limit),
        recent=recent,
        unannotated=unannotated,
        known_categories=store.categories(),
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
    if summary.category_counts:
        lines += ["", "  Category frequencies:"]
        for name, count in summary.category_counts:
            share = count / summary.annotated * 100 if summary.annotated else 0.0
            lines.append(f"    {name:<38}{count:>5,}  ({share:.0f}% of annotated)")
    if summary.combination_counts:
        lines += ["", "  Most common combinations:"]
        for combo, count in summary.combination_counts:
            lines.append(f"    {count:>4,}x  {' + '.join(combo)}")
    if summary.recent:
        lines += ["", "  Recently annotated:"]
        for annotation in summary.recent[:8]:
            categories = ", ".join(annotation.categories) or "(notes only)"
            lines.append(f"    {annotation.updated_at[:16]}  {annotation.filename:<28}{categories}")
    if not summary.annotated:
        lines += [
            "",
            "  No annotations yet. Run `picklikeme annotate --output <dir>` and open the",
            "  report to record why these images were missed.",
        ]
    return "\n".join(lines)
