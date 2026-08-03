"""arrange_by_species's on_result hook (Part 4 of the BioCLIP multi-backend
infrastructure work) - the seam species.experiment_capture uses to observe
every classification result (for analytics) without a second classifier
call or a second folder enumeration. Uses real small JPEG files and a stub
classifier - no model is loaded.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from picklikeme.species.arrange import arrange_by_species
from picklikeme.species.cache import SpeciesCache
from picklikeme.species.classifier import SpeciesPrediction


def _make_jpeg(path: Path) -> None:
    Image.new("RGB", (16, 16), color="blue").save(path, format="JPEG")


class _StubClassifier:
    classifier_id = "stub:1"

    def classify(self, image):
        return SpeciesPrediction(species="Kingfisher", confidence=0.9, classifier_id=self.classifier_id)


def test_on_result_is_called_once_per_image_with_the_real_prediction(tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    for name in ("a.jpg", "b.jpg"):
        _make_jpeg(folder / name)

    cache = SpeciesCache(tmp_path / "species.db")
    observed: list[tuple[str, SpeciesPrediction, float]] = []

    result = arrange_by_species(
        folder, _StubClassifier(), cache, dry_run=True,
        on_result=lambda path, prediction, elapsed: observed.append((path, prediction, elapsed)),
    )
    cache.close()

    assert result.classified == 2
    assert len(observed) == 2
    for path, prediction, elapsed in observed:
        assert prediction.species == "Kingfisher"
        assert elapsed >= 0.0


def test_on_result_defaults_to_none_and_changes_nothing(tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")

    cache = SpeciesCache(tmp_path / "species.db")
    result = arrange_by_species(folder, _StubClassifier(), cache, dry_run=True)
    cache.close()

    assert result.classified == 1


def test_a_raising_on_result_never_breaks_the_arrange_pass(tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")

    cache = SpeciesCache(tmp_path / "species.db")

    def _boom(path, prediction, elapsed):
        raise RuntimeError("analytics observer exploded")

    result = arrange_by_species(folder, _StubClassifier(), cache, dry_run=False, on_result=_boom)
    cache.close()

    assert result.classified == 1
    assert result.errors == 0  # the on_result failure must not count as a classify/move failure
    assert result.moved == 1


def test_on_result_still_fires_on_a_cache_hit(tmp_path: Path) -> None:
    """A second run over an already-classified folder is (mostly) cache
    hits - analytics must still see every image, not just fresh ones."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")

    cache = SpeciesCache(tmp_path / "species.db")
    arrange_by_species(folder, _StubClassifier(), cache, dry_run=True)  # primes the cache

    observed = []
    arrange_by_species(
        folder, _StubClassifier(), cache, dry_run=True,
        on_result=lambda path, prediction, elapsed: observed.append(prediction.species),
    )
    cache.close()

    assert observed == ["Kingfisher"]
