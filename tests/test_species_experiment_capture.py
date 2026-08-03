"""run_with_analytics - the wrapper that runs arrange_by_species and
records a full analytics experiment for it (Part 4 of the BioCLIP
multi-backend infrastructure work). No real model is used - a stub
classifier stands in, matching every other species test in this suite.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from picklikeme.analytics.store import AnalyticsStore
from picklikeme.species.cache import SpeciesCache
from picklikeme.species.classifier import SpeciesPrediction
from picklikeme.species.experiment_capture import run_with_analytics


def _make_jpeg(path: Path) -> None:
    Image.new("RGB", (16, 16), color="blue").save(path, format="JPEG")


class _StubClassifier:
    classifier_id = "bioclip-2:stubdigest"
    model_id = "hf-hub:imageomics/bioclip-2"
    species_list = ("Kingfisher", "Osprey")
    device = "cpu"
    min_confidence = 0.5
    prompt_template = "a photo of a {}"

    def __init__(self, answers):
        self._answers = iter(answers)

    def classify(self, image):
        return next(self._answers)


def test_run_with_analytics_records_a_full_experiment(tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _make_jpeg(folder / name)

    answers = [
        SpeciesPrediction(species="Kingfisher", confidence=0.9, classifier_id="bioclip-2:x",
                           top_predictions=(("Kingfisher", 0.9), ("Osprey", 0.05))),
        SpeciesPrediction(species="Osprey", confidence=0.7, classifier_id="bioclip-2:x",
                           top_predictions=(("Osprey", 0.7), ("Kingfisher", 0.2))),
        SpeciesPrediction(species="Unknown", confidence=0.3, classifier_id="bioclip-2:x",
                           top_predictions=(("Kingfisher", 0.3), ("Osprey", 0.25))),
    ]
    classifier = _StubClassifier(answers)
    cache = SpeciesCache(tmp_path / "species.db")
    analytics_db = tmp_path / "analytics.db"

    result, run_id, metadata = run_with_analytics(
        folder, classifier, "bioclip2", cache, dry_run=True, analytics_db=analytics_db,
    )
    cache.close()

    assert result.classified == 3
    assert run_id is not None
    assert metadata.classifier_backend == "bioclip2"

    with AnalyticsStore(analytics_db) as store:
        run = store.get_run(run_id)
        assert run["considered"] == 3
        assert run["accepted"] == 2  # 2 confidently classified, 1 Unknown

        species_distribution = store.category_counts(run_id)
        assert species_distribution == {"Kingfisher": 1, "Osprey": 1, "Unknown": 1}

        summary = store.summary_metrics(run_id)
        assert summary["unknown_rate"] == 1 / 3
        assert summary["runtime_seconds"] >= 0
        assert "average_inference_seconds" in summary

        image_paths = store.image_paths(run_id)
        assert len(image_paths) == 3
        first_image_metrics = store.image_metrics(run_id, image_paths[0])
        assert "top1_confidence" in first_image_metrics
        assert "top2_confidence" in first_image_metrics


def test_run_with_analytics_still_returns_the_real_result_if_analytics_db_is_unwritable(tmp_path: Path) -> None:
    """The real classify-and-file pass must never be blocked by an
    analytics failure - matches record_run's own "never fatal" contract."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")

    classifier = _StubClassifier(
        [SpeciesPrediction(species="Kingfisher", confidence=0.9, classifier_id="bioclip-2:x")]
    )
    cache = SpeciesCache(tmp_path / "species.db")

    # A directory in place of the analytics db file - opening it as a
    # SQLite database will fail, but the arrange pass itself must still
    # succeed and return a real result.
    bad_db_path = tmp_path / "not_a_file"
    bad_db_path.mkdir()

    result, run_id, metadata = run_with_analytics(
        folder, classifier, "bioclip2", cache, dry_run=True, analytics_db=bad_db_path,
    )
    cache.close()

    assert result.classified == 1
    assert run_id is None
    assert metadata is not None
