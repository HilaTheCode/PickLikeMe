"""Crop Sharpness: a deliberately eye/head-detection-free ranking strategy -
see crop_sharpness.py's own module docstring. Covers the two pure metrics,
the 80/20 weighted combination, registry/UI-registration integration (the
generic mechanisms every other strategy already relies on - see
ranking.base's own docstring), and one full rank_folder run against a
pre-populated crop cache (no RAW files, no real detector).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from picklikeme.bird_crop import COCO_BIRD_CLASS, crop_cache_path
from picklikeme.ranking import (
    CropSharpnessParams,
    CropSharpnessStrategy,
    available_strategies,
    get_strategy,
    metric_labels,
    score_labels,
)
from picklikeme.ranking.crop_sharpness import ImageMetrics, combine, measure, read_metrics_report
from picklikeme.ranking.filters import FilterCandidate, NO_SUBJECT


def _sharp_image(size: int = 256) -> np.ndarray:
    """Deterministic broadband texture standing in for a sharp photograph -
    same construction as test_ranking_strategies.py's own helper, so a
    canonical downsample cannot flatten it to zero the way a checkerboard
    sitting exactly at Nyquist would."""
    rng = np.random.default_rng(20260811)
    blocks = rng.integers(0, 256, size=(size // 4, size // 4), dtype=np.uint8)
    grown = cv2.resize(blocks, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.repeat(grown[:, :, None], 3, axis=2)


def _blurred_image(size: int = 256) -> np.ndarray:
    return cv2.GaussianBlur(_sharp_image(size), (0, 0), 6.0)


def _candidate(**overrides) -> FilterCandidate:
    defaults = dict(
        image_path="a.nef",
        subject_crop=_sharp_image(64),
        subject_box=(0.0, 0.0, 100.0, 100.0),
        source_size=(1000, 1000),
        subject_label=COCO_BIRD_CLASS,
    )
    defaults.update(overrides)
    return FilterCandidate(**defaults)


# ---------------------------------------------------------------------------
# registration - the generic mechanisms every strategy relies on
# ---------------------------------------------------------------------------


def test_crop_sharpness_is_registered():
    ids = {s.strategy_id for s in available_strategies()}
    assert "crop-sharpness" in ids


def test_get_strategy_constructs_crop_sharpness():
    assert isinstance(get_strategy("crop-sharpness"), CropSharpnessStrategy)


def test_crop_sharpness_has_a_score_label():
    assert score_labels()["crop-sharpness"] == "Crop Sharpness"


def test_crop_sharpness_declares_its_two_metrics_for_the_loupe_diagnostics_line():
    """metric_labels() is read generically by loupe_dialog._diagnostics_text
    (see ranking/__init__.py's own docstring) - declaring metric_labels here
    is the entire integration needed for the two new fields to appear there."""
    labels = metric_labels()["crop-sharpness"]
    assert labels == {"crop_sharpness": "Crop Sharpness", "relative_subject_size": "Relative Subject Size"}


def test_crop_sharpness_has_no_eye_detector():
    """ranking.eye_detector_names() must not list this strategy - it has no
    eye detector at all, by design (see the module docstring)."""
    from picklikeme.ranking import eye_detector_names

    assert "crop-sharpness" not in eye_detector_names()


# ---------------------------------------------------------------------------
# the two metrics
# ---------------------------------------------------------------------------


def test_crop_sharpness_metric_is_the_existing_subject_focus_measure():
    """Reuses ranking.metrics.subject_focus_measure verbatim - not a
    reimplementation - per the task's own instruction to use the project's
    existing sharpness implementation."""
    from picklikeme.ranking.metrics import subject_focus_measure

    crop = _sharp_image(64)
    candidate = _candidate(subject_crop=crop)
    result = measure(candidate)
    assert result.crop_sharpness == pytest.approx(subject_focus_measure(crop))


def test_a_sharp_crop_scores_higher_than_a_blurred_one():
    sharp = measure(_candidate(subject_crop=_sharp_image(64)))
    blurred = measure(_candidate(subject_crop=_blurred_image(64)))
    assert sharp.crop_sharpness > blurred.crop_sharpness


def test_relative_subject_size_is_box_area_over_frame_area():
    candidate = _candidate(subject_box=(0.0, 0.0, 100.0, 200.0), source_size=(1000, 1000))
    result = measure(candidate)
    assert result.relative_subject_size == pytest.approx((100.0 * 200.0) / (1000.0 * 1000.0))


def test_a_larger_subject_box_scores_a_higher_relative_size():
    small = measure(_candidate(subject_box=(0.0, 0.0, 10.0, 10.0)))
    large = measure(_candidate(subject_box=(0.0, 0.0, 500.0, 500.0)))
    assert large.relative_subject_size > small.relative_subject_size


# ---------------------------------------------------------------------------
# the 80/20 weighted combination
# ---------------------------------------------------------------------------


def test_default_weights_are_80_20():
    params = CropSharpnessParams()
    weights = params.normalized_weights()
    assert weights["crop_sharpness_weight"] == pytest.approx(0.8)
    assert weights["relative_subject_size_weight"] == pytest.approx(0.2)


def test_any_weight_scale_normalizes_to_the_same_80_20_split():
    """4/1 means the same thing as 80/20 - WeightedParams.normalized_weights
    scales to sum to 1, never validates a specific range (see base.py's own
    docstring)."""
    params = CropSharpnessParams(crop_sharpness_weight=4.0, relative_subject_size_weight=1.0)
    weights = params.normalized_weights()
    assert weights["crop_sharpness_weight"] == pytest.approx(0.8)
    assert weights["relative_subject_size_weight"] == pytest.approx(0.2)


def test_weights_are_configurable_and_actually_move_the_score():
    metrics = [
        ImageMetrics(image_path="sharp_small.nef", crop_sharpness=100.0, relative_subject_size=0.01,
                     has_subject_detection=True),
        ImageMetrics(image_path="soft_large.nef", crop_sharpness=1.0, relative_subject_size=1.0,
                     has_subject_detection=True),
    ]
    sharpness_heavy = combine(metrics, {"crop_sharpness_weight": 1.0, "relative_subject_size_weight": 0.0})
    size_heavy = combine(metrics, {"crop_sharpness_weight": 0.0, "relative_subject_size_weight": 1.0})

    assert sharpness_heavy[0] > sharpness_heavy[1]  # the sharp-but-small image wins when sharpness is all that counts
    assert size_heavy[1] > size_heavy[0]             # the soft-but-large image wins when size is all that counts


def test_combine_uses_the_80_20_default_end_to_end():
    metrics = [
        ImageMetrics(image_path="a.nef", crop_sharpness=10.0, relative_subject_size=0.1,
                     has_subject_detection=True),
        ImageMetrics(image_path="b.nef", crop_sharpness=1.0, relative_subject_size=0.9,
                     has_subject_detection=True),
    ]
    scores = combine(metrics, CropSharpnessParams().normalized_weights())
    # a.nef is far sharper and only moderately smaller-framed - the heavily
    # sharpness-weighted default must still rank it first.
    assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# the whole strategy, end to end (crop cache pre-populated, no RAW, no torch)
# ---------------------------------------------------------------------------


def _write_cache_entry(cache_dir, image_path, *, box=(10, 10, 60, 60), expanded_box=(5, 5, 65, 65)):
    crop = crop_cache_path(cache_dir, image_path)
    crop.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop), cv2.cvtColor(_sharp_image(64), cv2.COLOR_RGB2BGR))
    payload = {
        "version": 1,
        "source_size": [800, 600],
        "selected": {"box": list(box), "score": 0.9, "label": COCO_BIRD_CLASS},
        "detections": [{"box": list(box), "score": 0.9, "label": COCO_BIRD_CLASS}],
        "expanded_box": list(expanded_box) if expanded_box is not None else None,
    }
    crop.with_name(crop.stem + ".detections.json").write_text(json.dumps(payload), encoding="utf-8")
    return crop


def test_crop_sharpness_ranks_a_folder_with_no_eye_detector_involved(tmp_path, monkeypatch) -> None:
    """Full rank_folder run. Deliberately does NOT monkeypatch
    picklikeme.eyes.build_eye_detector at all - if this strategy ever grew a
    dependency on an eye detector, this test would fail by trying to load a
    real model rather than silently passing."""
    from picklikeme.analyzer.io import load_ranking
    from picklikeme.ranking import crop_sharpness as crop_sharpness_module
    from picklikeme.sidecar import ranking_path, strategy_ranking_path

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"

    scoring = [str(folder / f"animal_{i}.nef") for i in range(3)]
    undetected = str(folder / "empty.nef")
    for path in (*scoring, undetected):
        Path(path).write_bytes(b"not really a raw file")
    for path in scoring:
        _write_cache_entry(cache_dir, path)
    # `undetected` deliberately gets no cache entry - no crop, no subject.

    monkeypatch.setattr(crop_sharpness_module, "build_cache", lambda *a, **k: {})

    stages: list[str] = []
    result = CropSharpnessStrategy().rank_folder(
        folder,
        params=CropSharpnessParams(),
        crop_cache_dir=cache_dir,
        device="cpu",
        on_stage=stages.append,
        analytics_db=tmp_path / "analytics.db",
    )

    assert result["strategy"] == "crop-sharpness"
    assert result["considered"] == 4
    assert result["image_count"] == 3
    assert result["filtered"] == {NO_SUBJECT: 1}
    assert stages  # on_stage was actually called

    ranking_file = strategy_ranking_path(folder, "crop-sharpness")
    assert ranking_file.is_file()
    assert not ranking_path(folder).exists(), "must not have written the AI model's file"
    ranking = load_ranking(ranking_file)
    assert len(ranking.images) == 3
    scores = [image.score for image in ranking.images]
    assert scores == sorted(scores, reverse=True)

    metrics_report = read_metrics_report(folder)
    assert metrics_report["strategy"] == "crop-sharpness"
    assert set(metrics_report["metrics"]) == set(scoring)
    for values in metrics_report["metrics"].values():
        # has_subject_detection rides along so a reader can tell a "subject
        # fills 0% of the frame" from "no subject was located at all".
        assert set(values) == {"crop_sharpness", "relative_subject_size", "has_subject_detection"}
        assert values["has_subject_detection"] is True, "this fixture writes a real detection"
        assert values["relative_subject_size"] is not None

    from picklikeme.ranking.classic import read_filter_report

    assert read_filter_report(folder, strategy_id="crop-sharpness")["counts"] == {NO_SUBJECT: 1}


def test_crop_sharpness_never_touches_the_eye_cache(tmp_path, monkeypatch) -> None:
    """No .eye.json sidecar of any kind must appear for a crop-sharpness run
    - it has no eye detector to write one with."""
    from picklikeme.ranking import crop_sharpness as crop_sharpness_module

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    path = str(folder / "animal.nef")
    Path(path).write_bytes(b"not really a raw file")
    _write_cache_entry(cache_dir, path)

    monkeypatch.setattr(crop_sharpness_module, "build_cache", lambda *a, **k: {})

    CropSharpnessStrategy().rank_folder(
        folder, crop_cache_dir=cache_dir, device="cpu", analytics_db=tmp_path / "analytics.db",
    )

    assert list(cache_dir.rglob("*.eye.json")) == []


def test_crop_sharpness_refuses_a_folder_with_no_images(tmp_path) -> None:
    with pytest.raises(ValueError):
        CropSharpnessStrategy().rank_folder(tmp_path, crop_cache_dir=tmp_path / "crops")


def test_crop_sharpness_refuses_a_folder_that_does_not_exist(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        CropSharpnessStrategy().rank_folder(tmp_path / "nope", crop_cache_dir=tmp_path / "crops")


# ---------------------------------------------------------------------------
# Absolute scoring, and the two shapes a Crop Sharpness score can take.
#
# A real subject crop scores 0.80 * sharpness + 0.20 * relative subject size.
# A full-frame fallback - nothing was located in the frame - scores
# 1.00 * sharpness, with the subject-size term ABSENT rather than zeroed.
# Both terms are fixed maps onto [0, 1], so a crop's score does not depend on
# what it was ranked alongside.
# ---------------------------------------------------------------------------


def _fallback_candidate(crop=None) -> FilterCandidate:
    """What `_load_candidate` produces when no confident detection exists and
    `use_subject_filter=False`: the whole decoded frame is both the crop and
    the "subject box", and `subject_label` is None because no COCO class was
    ever assigned. That last part is the only honest signal that nothing was
    actually found - see FilterCandidate.has_selected_detection."""
    return FilterCandidate(
        image_path="fallback.nef",
        subject_crop=_sharp_image(64) if crop is None else crop,
        subject_box=(0.0, 0.0, 1000.0, 1000.0),
        crop_box=(0.0, 0.0, 1000.0, 1000.0),
        source_size=(1000, 1000),
        subject_label=None,
    )


def test_a_real_detection_is_distinguished_from_the_full_frame_fallback():
    assert _candidate().has_selected_detection is True
    assert _fallback_candidate().has_selected_detection is False


# --- real subject crop: 80% sharpness + 20% relative subject size ----------


def test_a_real_subject_crop_scores_80_percent_sharpness_and_20_percent_size():
    from picklikeme.ranking.metrics import absolute_sharpness_score

    metric = ImageMetrics(
        image_path="a.nef", crop_sharpness=0.70, relative_subject_size=0.25,
        has_subject_detection=True,
    )
    (score,) = combine([metric], CropSharpnessParams().normalized_weights())

    expected = 0.8 * absolute_sharpness_score(0.70) + 0.2 * 0.25
    assert score == pytest.approx(expected)
    # ...and 0.70 is the calibrated midpoint, so the sharpness half is 0.5.
    assert score == pytest.approx(0.8 * 0.5 + 0.2 * 0.25)


def test_the_subject_size_term_actually_moves_a_real_crop_s_score():
    small = ImageMetrics(image_path="s.nef", crop_sharpness=0.7, relative_subject_size=0.02,
                         has_subject_detection=True)
    large = ImageMetrics(image_path="l.nef", crop_sharpness=0.7, relative_subject_size=0.60,
                         has_subject_detection=True)

    scores = combine([small, large], CropSharpnessParams().normalized_weights())

    assert scores[1] > scores[0], "identical sharpness - the size bonus is the only difference"
    assert scores[1] - scores[0] == pytest.approx(0.2 * (0.60 - 0.02))


# --- full-frame fallback: 100% sharpness, no size term ---------------------


def test_the_full_frame_fallback_scores_100_percent_sharpness():
    from picklikeme.ranking.metrics import absolute_sharpness_score

    metric = ImageMetrics(
        image_path="f.nef", crop_sharpness=0.70, relative_subject_size=None,
        has_subject_detection=False,
    )
    (score,) = combine([metric], CropSharpnessParams().normalized_weights())

    assert score == pytest.approx(absolute_sharpness_score(0.70))
    assert score == pytest.approx(0.5), "the calibrated midpoint, undiluted by any size term"


def test_the_full_frame_fallback_never_receives_a_subject_size_contribution():
    """THE regression. The fallback's "subject box" is the whole frame, so a
    subject-size term computed from it would be 1.0 - the maximum possible
    bonus, handed to every image in which nothing was found."""
    metric = measure(_fallback_candidate())

    assert metric.has_subject_detection is False
    assert metric.relative_subject_size is None, "not measured, not 0.0, not 1.0"

    (score,) = combine([metric], CropSharpnessParams().normalized_weights())
    from picklikeme.ranking.metrics import absolute_sharpness_score

    assert score == pytest.approx(absolute_sharpness_score(metric.crop_sharpness))
    # Explicitly NOT the two shapes a size term could have taken.
    assert score != pytest.approx(0.8 * absolute_sharpness_score(metric.crop_sharpness) + 0.2 * 1.0)
    assert score != pytest.approx(0.8 * absolute_sharpness_score(metric.crop_sharpness))


def test_a_fallback_image_is_scored_purely_on_sharpness_whatever_the_weights():
    """Even a size-heavy configuration cannot give a fallback image a size
    bonus - there is nothing to weight."""
    metric = ImageMetrics(image_path="f.nef", crop_sharpness=1.2, relative_subject_size=None,
                          has_subject_detection=False)
    from picklikeme.ranking.metrics import absolute_sharpness_score

    for weights in (
        {"crop_sharpness_weight": 0.8, "relative_subject_size_weight": 0.2},
        {"crop_sharpness_weight": 0.1, "relative_subject_size_weight": 0.9},
        {"crop_sharpness_weight": 1.0, "relative_subject_size_weight": 0.0},
    ):
        (score,) = combine([metric], weights)
        assert score == pytest.approx(absolute_sharpness_score(1.2))


def test_a_fallback_image_and_a_detected_image_of_equal_sharpness_differ_only_by_the_bonus():
    from picklikeme.ranking.metrics import absolute_sharpness_score

    detected = ImageMetrics(image_path="d.nef", crop_sharpness=0.9, relative_subject_size=0.30,
                            has_subject_detection=True)
    fallback = ImageMetrics(image_path="f.nef", crop_sharpness=0.9, relative_subject_size=None,
                            has_subject_detection=False)

    detected_score, fallback_score = combine([detected, fallback], CropSharpnessParams().normalized_weights())

    sharpness = absolute_sharpness_score(0.9)
    assert fallback_score == pytest.approx(sharpness)
    assert detected_score == pytest.approx(0.8 * sharpness + 0.2 * 0.30)


# --- the sharpness score is absolute, in 0-1, and not a percentile ---------


def test_the_sharpness_score_stays_within_zero_and_one():
    from picklikeme.ranking.metrics import absolute_sharpness_score

    for raw in (0.0, 1e-6, 0.041, 0.143, 0.716, 1.369, 1.732, 10.0, 1e6, 1e12):
        score = absolute_sharpness_score(raw)
        assert 0.0 <= score <= 1.0, raw


def test_a_degenerate_or_negative_measure_scores_zero():
    from picklikeme.ranking.metrics import absolute_sharpness_score

    assert absolute_sharpness_score(0.0) == 0.0
    assert absolute_sharpness_score(-1.0) == 0.0
    assert absolute_sharpness_score(float("nan")) == 0.0


def test_the_sharpness_score_is_strictly_increasing_in_sharpness():
    from picklikeme.ranking.metrics import absolute_sharpness_score

    raws = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.4, 1.8, 3.0]
    scores = [absolute_sharpness_score(v) for v in raws]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores), "no two different sharpnesses collapse onto one score"


def test_the_score_does_not_depend_on_the_other_images_in_the_run():
    """The defining property of an absolute score: the SAME image scores the
    same number alone, in a pair, and in a large batch of very different
    images. `robust_normalize` (the previous scheme) fails every one of
    these - it maps each run's own 5th/95th percentiles onto 0 and 1."""
    subject = ImageMetrics(image_path="subject.nef", crop_sharpness=0.65, relative_subject_size=0.2,
                           has_subject_detection=True)
    weights = CropSharpnessParams().normalized_weights()

    alone = combine([subject], weights)[0]

    with_one_other = combine(
        [subject, ImageMetrics(image_path="x.nef", crop_sharpness=9.9, relative_subject_size=0.9,
                               has_subject_detection=True)],
        weights,
    )[0]

    crowd = [subject] + [
        ImageMetrics(image_path=f"n{i}.nef", crop_sharpness=0.01 * i, relative_subject_size=0.01 * i,
                     has_subject_detection=True)
        for i in range(1, 200)
    ]
    in_a_crowd = combine(crowd, weights)[0]

    assert alone == pytest.approx(with_one_other)
    assert alone == pytest.approx(in_a_crowd)


def test_the_score_is_not_a_folder_relative_percentile():
    """A percentile rank would put the sharpest image of any run at the top
    of the scale and the least sharp at the bottom, whatever their absolute
    sharpness. An absolute score does not: a folder of uniformly soft images
    scores low across the board, and a folder of uniformly sharp ones scores
    high, and the two orderings do not both span 0-1."""
    weights = CropSharpnessParams().normalized_weights()
    soft = [ImageMetrics(image_path=f"s{i}.nef", crop_sharpness=v, relative_subject_size=0.1,
                         has_subject_detection=True)
            for i, v in enumerate((0.05, 0.07, 0.09))]
    sharp = [ImageMetrics(image_path=f"h{i}.nef", crop_sharpness=v, relative_subject_size=0.1,
                          has_subject_detection=True)
             for i, v in enumerate((1.4, 1.6, 1.8))]

    soft_scores = combine(soft, weights)
    sharp_scores = combine(sharp, weights)

    assert max(soft_scores) < min(sharp_scores), "a soft folder never reaches a sharp folder's scores"
    assert max(soft_scores) < 0.3, "the best of a soft folder is still a low absolute score"
    assert min(sharp_scores) > 0.5


def test_no_plateau_at_the_top_of_the_scale():
    """The P5/P95-clipping scheme this replaced flattened the top ~5% and the
    bottom ~5% of every run onto identical 1.000 and 0.000 values. These are
    the real P95/P99/max of a 250-crop sample of this project's own crop
    cache: they must remain three distinct displayed scores."""
    from picklikeme.desktop.widgets.design_system import format_score
    from picklikeme.ranking.metrics import absolute_sharpness_score

    displayed = {format_score(absolute_sharpness_score(v)) for v in (1.369, 1.667, 1.732)}
    assert len(displayed) == 3, displayed
    assert "1.000" not in displayed
    # ...and the bottom end stays distinguishable too.
    bottom = {format_score(absolute_sharpness_score(v)) for v in (0.041, 0.065, 0.143)}
    assert len(bottom) == 3, bottom
    assert "0.000" not in bottom


def test_the_calibrated_midpoint_scores_exactly_one_half():
    from picklikeme.ranking.metrics import SHARPNESS_MIDPOINT, absolute_sharpness_score

    assert absolute_sharpness_score(SHARPNESS_MIDPOINT) == pytest.approx(0.5)


def test_measure_records_a_real_subject_size_when_a_detection_exists():
    metric = measure(_candidate())

    assert metric.has_subject_detection is True
    assert metric.relative_subject_size == pytest.approx(100.0 * 100.0 / (1000.0 * 1000.0))


# --- display precision: exactly three decimals -----------------------------


def test_scores_are_displayed_with_exactly_three_decimals():
    from picklikeme.desktop.widgets.design_system import SCORE_FORMAT, format_score

    assert SCORE_FORMAT == ".3f"
    assert format_score(0.0) == "0.000"
    assert format_score(0.2374999) == "0.237"
    assert format_score(0.8123) == "0.812"
    assert format_score(1.0) == "1.000"
    assert format_score(None) == "—"
    for value in (0.0, 0.5, 0.999999, 1.0):
        assert len(format_score(value).split(".")[1]) == 3


def test_the_grid_card_shows_a_crop_sharpness_score_with_three_decimals():
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    delegate.set_color_source("crop-sharpness")
    item = ImageItem(path="/x/a.nef", file_name="a.nef",
                     ranking_results={"crop-sharpness": {"score": 0.5083333, "rank": 1}})

    assert delegate._selected_score_text(item) == "0.508"


def test_a_full_frame_fallback_image_is_scored_end_to_end_without_a_size_term(tmp_path, monkeypatch) -> None:
    """The whole strategy over a cache entry that has NO selected detection -
    the case `use_subject_filter=False` exists to admit. The image must reach
    the ranking, be scored on sharpness alone, and record no subject size."""
    from picklikeme.ranking.metrics import absolute_sharpness_score

    folder = tmp_path / "shoot"
    folder.mkdir()
    image = folder / "nodetection.jpg"
    cv2.imwrite(str(image), cv2.cvtColor(_sharp_image(64), cv2.COLOR_RGB2BGR))

    cache_dir = tmp_path / "crops"
    crop = crop_cache_path(cache_dir, str(image))
    crop.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop), cv2.cvtColor(_sharp_image(64), cv2.COLOR_RGB2BGR))
    # No "selected" key at all: the detector found nothing it trusted, and
    # bird_crop wrote the whole decoded frame as the crop.
    crop.with_name(crop.stem + ".detections.json").write_text(
        json.dumps({"version": 1, "source_size": [800, 600], "selected": None, "detections": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr("picklikeme.ranking.crop_sharpness.build_cache", lambda *a, **k: None)
    monkeypatch.setattr("picklikeme.ranking.crop_sharpness.resolve_device", lambda device: "cpu")
    monkeypatch.setattr("picklikeme.ranking.crop_sharpness.record_run", lambda *a, **k: None)

    result = CropSharpnessStrategy().rank_folder(
        folder,
        params=CropSharpnessParams(use_subject_filter=False),
        crop_cache_dir=cache_dir,
    )

    assert result["image_count"] == 1, "the fallback image must be ranked, not filtered out"
    values = read_metrics_report(folder)["metrics"][str(image)]
    assert values["has_subject_detection"] is False
    assert values["relative_subject_size"] is None, "never measured for a frame with no subject"

    (_name, score) = result["top"][0]
    assert score == pytest.approx(absolute_sharpness_score(values["crop_sharpness"]))
    assert 0.0 <= score <= 1.0
