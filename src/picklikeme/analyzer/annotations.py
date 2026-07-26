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

Identity across runs is the one genuinely hard problem: annotations must
survive a folder being renamed or moved to another drive. Lookup therefore
mirrors the matching engine's philosophy - exact key first, then a filename
fallback that refuses to answer when the filename is ambiguous, because a
misattributed diagnosis is worse than a missing one.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..config import PROJECT_ROOT

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

SCHEMA_VERSION = 1


def image_key(image_path: str | Path) -> str:
    """Stable identity for an image.

    The same digest scheme the crop cache uses, so the two stay conceptually
    aligned: derived from the resolved absolute path, never from file contents
    (which would mean reading 45 MB to look up a note).
    """
    resolved = str(Path(image_path).resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:20]


@dataclass
class Annotation:
    """One photographer diagnosis for one false negative."""

    image_key: str
    image_path: str
    filename: str
    categories: list[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    # True when the record was found by filename rather than by path, i.e. the
    # image has moved since it was annotated. Surfaced so the UI can say so.
    matched_by_filename: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.categories and not self.notes.strip()

    def as_dict(self) -> dict:
        return {
            "image_key": self.image_key,
            "image_path": self.image_path,
            "filename": self.filename,
            "categories": list(self.categories),
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "matched_by_filename": self.matched_by_filename,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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

                CREATE TABLE IF NOT EXISTS annotations (
                    image_key  TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    filename   TEXT NOT NULL,
                    notes      TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annotation_categories (
                    image_key TEXT NOT NULL,
                    category  TEXT NOT NULL,
                    PRIMARY KEY (image_key, category),
                    FOREIGN KEY (image_key) REFERENCES annotations(image_key) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_annotations_filename ON annotations(filename);
                CREATE INDEX IF NOT EXISTS idx_annotations_updated  ON annotations(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ann_cat_category     ON annotation_categories(category);
                """
            )
            if not self._conn.execute("SELECT 1 FROM schema_info").fetchone():
                self._conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            for ordering, name in enumerate(INITIAL_CATEGORIES):
                self._conn.execute(
                    "INSERT OR IGNORE INTO categories(name, ordering, builtin) VALUES (?, ?, 1)",
                    (name, ordering),
                )

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

    def _row_to_annotation(self, row: sqlite3.Row, matched_by_filename: bool = False) -> Annotation:
        categories = [
            item["category"]
            for item in self._conn.execute(
                "SELECT category FROM annotation_categories WHERE image_key = ? ORDER BY category",
                (row["image_key"],),
            )
        ]
        return Annotation(
            image_key=row["image_key"],
            image_path=row["image_path"],
            filename=row["filename"],
            categories=categories,
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            matched_by_filename=matched_by_filename,
        )

    def get(self, image_path: str | Path) -> Annotation | None:
        """Look up one annotation: exact key, then unambiguous filename.

        The filename fallback is what lets a knowledge base survive the archive
        being reorganised. It refuses to guess when two annotated images share a
        filename - camera counters reset, so duplicate basenames are common in a
        multi-year archive and a wrong diagnosis is worse than none.
        """
        key = image_key(image_path)
        row = self._conn.execute("SELECT * FROM annotations WHERE image_key = ?", (key,)).fetchone()
        if row is not None:
            return self._row_to_annotation(row)

        filename = Path(image_path).name
        rows = self._conn.execute(
            "SELECT * FROM annotations WHERE filename = ? COLLATE NOCASE", (filename,)
        ).fetchall()
        if len(rows) == 1:
            return self._row_to_annotation(rows[0], matched_by_filename=True)
        if len(rows) > 1:
            logger.debug("Ambiguous annotation filename %r (%d matches); not guessing", filename, len(rows))
        return None

    def get_many(self, image_paths: Iterable[str | Path]) -> dict[str, Annotation]:
        """path -> annotation for the paths that have one."""
        found: dict[str, Annotation] = {}
        for path in image_paths:
            annotation = self.get(path)
            if annotation is not None:
                found[str(path)] = annotation
        return found

    def all(self) -> list[Annotation]:
        rows = self._conn.execute("SELECT * FROM annotations ORDER BY updated_at DESC").fetchall()
        return [self._row_to_annotation(row) for row in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM annotations").fetchone()["n"])

    # -- writes -------------------------------------------------------------

    def save(self, image_path: str | Path, categories: Sequence[str], notes: str = "") -> Annotation:
        """Create or replace one annotation.

        Saving with no categories and no notes deletes the record, so clearing
        the panel in the UI is how a mistaken annotation is removed - there is
        no separate delete gesture to discover.
        """
        key = image_key(image_path)
        cleaned = [c.strip() for c in categories if c and c.strip()]
        notes = notes.strip()

        if not cleaned and not notes:
            self.delete(image_path)
            return Annotation(
                image_key=key, image_path=str(image_path), filename=Path(image_path).name
            )

        now = _now()
        with self._conn:
            existing = self._conn.execute(
                "SELECT created_at FROM annotations WHERE image_key = ?", (key,)
            ).fetchone()
            created = existing["created_at"] if existing else now
            self._conn.execute(
                """
                INSERT INTO annotations(image_key, image_path, filename, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_key) DO UPDATE SET
                    image_path = excluded.image_path,
                    filename   = excluded.filename,
                    notes      = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (key, str(Path(image_path)), Path(image_path).name, notes, created, now),
            )
            self._conn.execute("DELETE FROM annotation_categories WHERE image_key = ?", (key,))
            for category in cleaned:
                self._conn.execute(
                    "INSERT OR IGNORE INTO annotation_categories(image_key, category) VALUES (?, ?)",
                    (key, category),
                )
                # Free-text categories are remembered so they show up next time.
                self._conn.execute(
                    "INSERT OR IGNORE INTO categories(name, ordering, builtin) VALUES (?, 500, 0)",
                    (category,),
                )

        logger.info("Annotation saved for %s (%d categories)", Path(image_path).name, len(cleaned))
        return Annotation(
            image_key=key,
            image_path=str(image_path),
            filename=Path(image_path).name,
            categories=sorted(cleaned),
            notes=notes,
            created_at=created,
            updated_at=now,
        )

    def delete(self, image_path: str | Path) -> bool:
        key = image_key(image_path)
        with self._conn:
            cursor = self._conn.execute("DELETE FROM annotations WHERE image_key = ?", (key,))
            self._conn.execute("DELETE FROM annotation_categories WHERE image_key = ?", (key,))
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
    found = store.get_many(false_negative_paths)

    category_counter: Counter[str] = Counter()
    combination_counter: Counter[tuple[str, ...]] = Counter()
    for annotation in found.values():
        category_counter.update(annotation.categories)
        if len(annotation.categories) > 1:
            # Sorted so {Action shot, Lighting} and {Lighting, Action shot} are
            # counted as the same combination.
            combination_counter[tuple(sorted(annotation.categories))] += 1

    recent = sorted(found.values(), key=lambda a: a.updated_at, reverse=True)[:recent_limit]
    unannotated = [path for path in false_negative_paths if path not in found]

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
    ]
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
