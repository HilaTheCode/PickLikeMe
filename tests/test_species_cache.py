"""SpeciesCache: the v1 -> v2 schema migration and the composite
(image_hash, classifier_id) key it exists to introduce - see
docs/BioCLIP_Backend_Architecture_Review.md Section 5 for the finding that
motivated this (two classifier backends could silently overwrite each
other's cached predictions for the same image under the old, single-column
primary key).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from picklikeme.species.cache import SPECIES_CACHE_SCHEMA_VERSION, SpeciesCache
from picklikeme.species.classifier import SpeciesPrediction


def _seed_v1_database(path: Path, rows: list[tuple[str, str, float, str]]) -> None:
    """Writes a database in the exact old (pre-migration) shape - a plain
    single-column primary key on image_hash - so the migration is tested
    against the real historical schema, not a simplified stand-in."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE species_cache (
            image_hash    TEXT PRIMARY KEY,
            species       TEXT NOT NULL,
            confidence    REAL,
            classifier_id TEXT NOT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.executemany(
        "INSERT INTO species_cache(image_hash, species, confidence, classifier_id) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_fresh_database_is_created_at_the_current_schema(tmp_path: Path) -> None:
    with SpeciesCache(tmp_path / "species.db") as cache:
        columns = cache._conn.execute("PRAGMA table_info(species_cache)").fetchall()
        pk_columns = {col["name"] for col in columns if col["pk"] > 0}
        assert pk_columns == {"image_hash", "classifier_id"}


def test_v1_database_is_migrated_with_every_row_preserved(tmp_path: Path) -> None:
    db_path = tmp_path / "species.db"
    _seed_v1_database(
        db_path,
        [
            ("hash-a", "Kingfisher", 0.9, "bioclip2:bioclip-2:digest"),
            ("hash-b", "Common Tern", 0.7, "bioclip2:bioclip-2:digest"),
        ],
    )

    with SpeciesCache(db_path) as cache:
        columns = cache._conn.execute("PRAGMA table_info(species_cache)").fetchall()
        pk_columns = {col["name"] for col in columns if col["pk"] > 0}
        assert pk_columns == {"image_hash", "classifier_id"}

        rows = cache._conn.execute("SELECT * FROM species_cache ORDER BY image_hash").fetchall()
        assert len(rows) == 2
        assert rows[0]["image_hash"] == "hash-a"
        assert rows[0]["species"] == "Kingfisher"
        assert rows[0]["classifier_id"] == "bioclip2:bioclip-2:digest"

        version = cache._conn.execute(
            "SELECT value FROM schema_info WHERE key = 'species_cache_version'"
        ).fetchone()
        assert int(version["value"]) == SPECIES_CACHE_SCHEMA_VERSION


def test_migration_is_idempotent_on_a_second_open(tmp_path: Path) -> None:
    """Opening an already-migrated database a second time must not re-run
    the migration (which would fail - species_cache_v1 no longer exists)
    or duplicate rows."""
    db_path = tmp_path / "species.db"
    _seed_v1_database(db_path, [("hash-a", "Kingfisher", 0.9, "bioclip2:bioclip-2:digest")])

    with SpeciesCache(db_path):
        pass
    with SpeciesCache(db_path) as cache:
        rows = cache._conn.execute("SELECT * FROM species_cache").fetchall()
        assert len(rows) == 1


def test_two_classifiers_cached_results_coexist_for_the_same_image(tmp_path: Path) -> None:
    """The actual bug this migration fixes: classifying the same image with
    two different backends must never make one overwrite the other."""
    with SpeciesCache(tmp_path / "species.db") as cache:
        cache._identity = lambda path: "same-image-hash"  # noqa: SLF001 - test seam, avoids needing a real file

        cache.store("a.jpg", SpeciesPrediction(species="Kingfisher", confidence=0.9, classifier_id="bioclip-2:x"))
        cache.store("a.jpg", SpeciesPrediction(species="Unknown", confidence=0.3, classifier_id="bioclip:x"))

        v2_result = cache.get("a.jpg", "bioclip-2:x")
        v1_result = cache.get("a.jpg", "bioclip:x")

        assert v2_result is not None and v2_result.species == "Kingfisher"
        assert v1_result is not None and v1_result.species == "Unknown"

        count = cache._conn.execute(
            "SELECT COUNT(*) FROM species_cache WHERE image_hash = 'same-image-hash'"
        ).fetchone()[0]
        assert count == 2


def test_get_returns_none_for_a_classifier_with_no_cached_row(tmp_path: Path) -> None:
    with SpeciesCache(tmp_path / "species.db") as cache:
        cache._identity = lambda path: "same-image-hash"  # noqa: SLF001
        cache.store("a.jpg", SpeciesPrediction(species="Kingfisher", confidence=0.9, classifier_id="bioclip-2:x"))

        assert cache.get("a.jpg", "bioclip:x") is None  # a different classifier - never a near-miss


def test_get_or_classify_only_calls_the_classifier_on_a_real_miss(tmp_path: Path, monkeypatch) -> None:
    import picklikeme.analyzer.contactsheets as contactsheets_module

    calls = []

    class _StubClassifier:
        classifier_id = "stub:1"

        def classify(self, image):
            calls.append(image)
            return SpeciesPrediction(species="Kingfisher", confidence=0.9, classifier_id=self.classifier_id)

    monkeypatch.setattr(contactsheets_module, "load_source_image", lambda path: object())

    with SpeciesCache(tmp_path / "species.db") as cache:
        cache._identity = lambda path: "same-image-hash"  # noqa: SLF001
        classifier = _StubClassifier()

        first = cache.get_or_classify("a.jpg", classifier)
        second = cache.get_or_classify("a.jpg", classifier)

        assert first.species == "Kingfisher"
        assert second.species == "Kingfisher"
        assert len(calls) == 1  # the second call was a cache hit
