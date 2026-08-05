"""Species predictions, memoised in their own SQLite database.

Keyed by content identity (identity.image_identity), not path - deliberately
different from AnnotationStore's (path, size, mtime) caches. Arrange by
Species's whole job is to *move* the files it classifies, so a path-keyed
cache would be invalidated by the very operation this feature performs. A
separate database, not a table on AnnotationStore or DetectionCache: species
classification is a distinct concern (an optional post-Review workflow, not
a review decision or a detector box) that should be droppable/rebuildable on
its own without touching either of those.

Shaped like analyzer.detections.DetectionCache: a version-like column
(classifier_id) checked on every read so a result computed by one classifier
is never silently served to a caller now using a different one, and
get_or_classify() only pays for a real classification on an actual miss - a
re-run over an already-classified folder is nearly free.

**Schema v2 (this version): keyed by (image_hash, classifier_id) together,
not image_hash alone.** See docs/BioCLIP_Backend_Architecture_Review.md
Section 5 for the full finding: under the v1 schema, `image_hash` alone was
the primary key, so classifying the same image with two different
classifiers (e.g. BioCLIP 2, then the original BioCLIP) made the second
`INSERT OR REPLACE` silently destroy the first classifier's cached row -
two backends could never coexist in the cache. The read path (`get`) was
always safe (a `classifier_id` mismatch was already treated as a cache
miss, never served as a wrong answer) - only storage was broken. This
directly blocked the "same folder classified by multiple backends" use case
multi-backend support (and any future benchmark) needs.

Migration runs automatically, once, the first time a v1 database is opened
under this code (see `_migrate_v1_to_v2`): every existing row is preserved
and copied forward into the new composite-keyed table - no cached
prediction is discarded by this migration itself. Separately, and for an
unrelated reason: the `classifier_id` *format* itself changed in an earlier
fix (the "bioclip2:" prefix used to be hardcoded regardless of which model
actually ran - see bioclip_classifier.py's own docstring), so rows written
before that fix carry an old-format classifier_id string that no current
classifier instance will ever match again. Those rows are harmless, inert
history after this migration, not actively wrong - `get()` still requires
an exact classifier_id match, so an old-format row is simply never served,
the same "safe cache miss" behaviour this cache has always had for a
version mismatch.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from ..config import PROJECT_ROOT
from ..identity import IdentityUnavailable, image_identity
from .classifier import SpeciesClassifier, SpeciesPrediction

logger = logging.getLogger(__name__)

DEFAULT_SPECIES_DB = PROJECT_ROOT / "cache" / "species.db"

# v1: PRIMARY KEY (image_hash) alone - see this module's own docstring for
# why that was a real bug, not just a design nitpick.
# v2 (current): PRIMARY KEY (image_hash, classifier_id) - two different
# classifiers' predictions for the same image can now coexist.
SPECIES_CACHE_SCHEMA_VERSION = 2


class SpeciesCache:
    """Species predictions, keyed by content identity AND classifier_id
    together - see the module docstring for why both are needed."""

    def __init__(self, db_path: str | Path = DEFAULT_SPECIES_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
        self._migrate_if_needed()

    def _stored_schema_version(self) -> int | None:
        row = self._conn.execute("SELECT value FROM schema_info WHERE key = 'species_cache_version'").fetchone()
        return int(row["value"]) if row is not None else None

    def _table_exists(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    def _migrate_if_needed(self) -> None:
        """Creates `species_cache` at the current schema if it does not
        exist yet, or migrates a v1 (pre-composite-key) table forward,
        preserving every row - see the module docstring for exactly what
        this does and does not invalidate."""
        stored_version = self._stored_schema_version()
        if stored_version == SPECIES_CACHE_SCHEMA_VERSION:
            return

        if not self._table_exists("species_cache"):
            # Fresh database - nothing to migrate, just create at the
            # current schema directly.
            self._create_v2_table()
            self._set_schema_version(SPECIES_CACHE_SCHEMA_VERSION)
            return

        # An existing table under an older schema (or, in principle, no
        # recorded version but a table already present - the state of any
        # database written before schema_info existed at all). Detect the
        # old single-column-primary-key shape directly rather than trusting
        # only the version marker, so a database written by an even older,
        # unversioned build is still recognised and migrated correctly.
        columns = self._conn.execute("PRAGMA table_info(species_cache)").fetchall()
        is_v1_shape = any(col["name"] == "image_hash" and col["pk"] == 1 for col in columns) and not any(
            col["name"] == "classifier_id" and col["pk"] > 0 for col in columns
        )
        if is_v1_shape:
            self._migrate_v1_to_v2()
        self._set_schema_version(SPECIES_CACHE_SCHEMA_VERSION)

    def _create_v2_table(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS species_cache (
                    image_hash    TEXT NOT NULL,
                    classifier_id TEXT NOT NULL,
                    species       TEXT NOT NULL,
                    confidence    REAL,
                    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (image_hash, classifier_id)
                )
                """
            )

    def _set_schema_version(self, version: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_info(key, value) VALUES ('species_cache_version', ?)",
                (str(version),),
            )

    def _migrate_v1_to_v2(self) -> None:
        """Renames the old single-primary-key table aside, creates the new
        composite-key one, and copies every row forward unchanged. Every
        row that existed is still present afterward - migration changes
        what counts as a collision going forward, it does not discard
        history. Logged once so a photographer with an existing
        `species.db` sees this happened, not just a silent schema change."""
        row_count = self._conn.execute("SELECT COUNT(*) AS n FROM species_cache").fetchone()["n"]
        logger.info(
            "Migrating %s: species_cache schema v1 -> v2 (%d existing row(s) preserved, "
            "now keyed by image+classifier together instead of image alone)",
            self.db_path, row_count,
        )
        with self._conn:
            self._conn.execute("ALTER TABLE species_cache RENAME TO species_cache_v1")
            self._conn.execute(
                """
                CREATE TABLE species_cache (
                    image_hash    TEXT NOT NULL,
                    classifier_id TEXT NOT NULL,
                    species       TEXT NOT NULL,
                    confidence    REAL,
                    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (image_hash, classifier_id)
                )
                """
            )
            self._conn.execute(
                """
                INSERT INTO species_cache (image_hash, classifier_id, species, confidence, created_at)
                SELECT image_hash, classifier_id, species, confidence, created_at FROM species_cache_v1
                """
            )
            self._conn.execute("DROP TABLE species_cache_v1")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SpeciesCache":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _identity(self, image_path: str | Path) -> str | None:
        try:
            return image_identity(image_path)
        except IdentityUnavailable as exc:
            logger.debug("No identity for %s: %s", image_path, exc)
            return None

    def get(self, image_path: str | Path, classifier_id: str) -> SpeciesPrediction | None:
        """A prior prediction for this exact file from this exact
        classifier, or None if there is nothing on record - including when
        a *different* classifier has a cached row for this same image
        (that row is simply a different cache entry now, never a
        near-miss)."""
        digest = self._identity(image_path)
        if digest is None:
            return None
        row = self._conn.execute(
            "SELECT species, confidence, classifier_id FROM species_cache WHERE image_hash = ? AND classifier_id = ?",
            (digest, classifier_id),
        ).fetchone()
        if row is None:
            return None
        return SpeciesPrediction(
            species=row["species"], confidence=row["confidence"], classifier_id=row["classifier_id"]
        )

    def get_any(self, image_path: str | Path) -> SpeciesPrediction | None:
        """The most recent prediction for this exact file from ANY
        classifier, or None if nothing is on record for it at all.

        For a best-effort caller with no single classifier_id of its own to
        ask for - the Review Window's Advanced Filters panel, which shows
        species purely as a convenience filter over whatever `Organize by
        Species` has already classified in this project, unlike
        AnalyticsStore-backed callers (e.g. AnalyticsDashboard's
        ImageExplorerTab) that always know one specific run's own
        strategy_id/classifier_id to scope `get()` to."""
        digest = self._identity(image_path)
        if digest is None:
            return None
        row = self._conn.execute(
            "SELECT species, confidence, classifier_id FROM species_cache "
            "WHERE image_hash = ? ORDER BY created_at DESC LIMIT 1",
            (digest,),
        ).fetchone()
        if row is None:
            return None
        return SpeciesPrediction(
            species=row["species"], confidence=row["confidence"], classifier_id=row["classifier_id"]
        )

    def store(self, image_path: str | Path, prediction: SpeciesPrediction) -> None:
        """Writes (or replaces) this exact (image, classifier) pair's
        cached prediction - never touches any other classifier's row for
        the same image, since both are now part of the primary key."""
        digest = self._identity(image_path)
        if digest is None:
            return
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO species_cache(image_hash, classifier_id, species, confidence) "
                "VALUES (?, ?, ?, ?)",
                (digest, prediction.classifier_id, prediction.species, prediction.confidence),
            )

    def get_or_classify(self, image_path: str | Path, classifier: SpeciesClassifier) -> SpeciesPrediction:
        """The cached prediction, or a fresh one from `classifier` - decoded
        and classified only on an actual miss, then stored before it is
        returned, so neither this call nor a future one repeats the work.
        Running a second classifier over the same folder afterward is a
        second, independent set of cache entries, not a conflict with the
        first."""
        cached = self.get(image_path, classifier.classifier_id)
        if cached is not None:
            return cached

        from ..analyzer.contactsheets import load_source_image

        image = load_source_image(str(image_path))
        prediction = classifier.classify(image)
        self.store(image_path, prediction)
        return prediction
