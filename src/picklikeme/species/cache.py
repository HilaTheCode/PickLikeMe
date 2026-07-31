"""Species predictions, memoised in their own SQLite database.

Keyed by content identity (identity.image_identity), not path - deliberately
different from AnnotationStore's (path, size, mtime) caches. Arrange by
Species's whole job is to *move* the files it classifies, so a path-keyed
cache would be invalidated by the very operation this feature performs. A
separate database, not a table on AnnotationStore or DetectionCache: species
classification is a distinct concern (an optional post-Review workflow, not
a review decision or a detector box) that should be droppable/rebuildable on
its own without touching either of those.

Shaped like analyzer.detections.DetectionCache: one row per image, a
version-like column (classifier_id here) checked on every read so a result
computed by one classifier is never silently served to a caller now using a
different one, and get_or_classify() only pays for a real classification on
an actual miss - a re-run over an already-classified folder is nearly free.
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


class SpeciesCache:
    """Species predictions, keyed by content identity and classifier_id."""

    def __init__(self, db_path: str | Path = DEFAULT_SPECIES_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS species_cache (
                    image_hash    TEXT PRIMARY KEY,
                    species       TEXT NOT NULL,
                    confidence    REAL,
                    classifier_id TEXT NOT NULL,
                    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )

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
        classifier, or None if there is nothing (compatible) on record."""
        digest = self._identity(image_path)
        if digest is None:
            return None
        row = self._conn.execute(
            "SELECT species, confidence, classifier_id FROM species_cache WHERE image_hash = ?", (digest,)
        ).fetchone()
        if row is None or row["classifier_id"] != classifier_id:
            return None
        return SpeciesPrediction(
            species=row["species"], confidence=row["confidence"], classifier_id=row["classifier_id"]
        )

    def store(self, image_path: str | Path, prediction: SpeciesPrediction) -> None:
        digest = self._identity(image_path)
        if digest is None:
            return
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO species_cache(image_hash, species, confidence, classifier_id) "
                "VALUES (?, ?, ?, ?)",
                (digest, prediction.species, prediction.confidence, prediction.classifier_id),
            )

    def get_or_classify(self, image_path: str | Path, classifier: SpeciesClassifier) -> SpeciesPrediction:
        """The cached prediction, or a fresh one from `classifier` - decoded
        and classified only on an actual miss, then stored before it is
        returned, so neither this call nor a future one repeats the work."""
        cached = self.get(image_path, classifier.classifier_id)
        if cached is not None:
            return cached

        from ..analyzer.contactsheets import load_source_image

        image = load_source_image(str(image_path))
        prediction = classifier.classify(image)
        self.store(image_path, prediction)
        return prediction
