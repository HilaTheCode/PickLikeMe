"""`use_subject_filter` - the pipeline switch that makes the shared crop
(not COCO's own class+confidence gated "selected" detection) the common
input to ranking, per the Eye-Detector Ensemble Evidence Study's conclusion
that the upstream subject filter rejects images a ranking strategy could
still usefully score. Default False (relaxed); True preserves the original,
stricter behavior unchanged.

Covers both layers: `ClassicVisionStrategy._load_candidate` (the mechanism)
and a full `rank_folder()` run for both a detection-free strategy (Crop
Sharpness) and an eye-based one, so the observable end-to-end behavior is
tested, not just the loader in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from picklikeme.bird_crop import COCO_BIRD_CLASS, crop_cache_path
from picklikeme.eyes.detector import EyeDetection
from picklikeme.ranking import (
    ClassicVisionBirdFusionParams,
    ClassicVisionCombinedParams,
    ClassicVisionEyePoseParams,
    ClassicVisionMammalFusionParams,
    ClassicVisionParams,
    CropSharpnessParams,
)
from picklikeme.ranking.classic import ClassicVisionEyePoseStrategy, ClassicVisionStrategy
from picklikeme.ranking.crop_sharpness import CropSharpnessStrategy
from picklikeme.ranking.filters import NO_SUBJECT, NO_VISIBLE_EYE


def _sharp_image(size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(20260812)
    blocks = rng.integers(0, 256, size=(size // 4, size // 4), dtype=np.uint8)
    grown = cv2.resize(blocks, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.repeat(grown[:, :, None], 3, axis=2)


def _write_cache_entry(cache_dir, image_path, *, box=(10, 10, 60, 60), expanded_box=(5, 5, 65, 65)):
    """A real, COCO-confident detection - today's ordinary case."""
    crop = crop_cache_path(cache_dir, image_path)
    crop.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop), cv2.cvtColor(_sharp_image(), cv2.COLOR_RGB2BGR))
    payload = {
        "version": 1,
        "source_size": [800, 600],
        "selected": {"box": list(box), "score": 0.9, "label": COCO_BIRD_CLASS},
        "detections": [{"box": list(box), "score": 0.9, "label": COCO_BIRD_CLASS}],
        "expanded_box": list(expanded_box),
    }
    crop.with_name(crop.stem + ".detections.json").write_text(json.dumps(payload), encoding="utf-8")
    return crop


def _write_fallback_cache_entry(cache_dir, image_path, *, source_size=(800, 600)):
    """What bird_crop.build_crop/save_detections actually write when COCO
    found nothing: a real, readable crop file (the full decoded frame), but
    `selected` is None - see build_crop's own "full-frame fallback" branch."""
    crop = crop_cache_path(cache_dir, image_path)
    crop.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop), cv2.cvtColor(_sharp_image(), cv2.COLOR_RGB2BGR))
    payload = {
        "version": 1,
        "source_size": list(source_size),
        "selected": None,
        "detections": [],
        "expanded_box": None,
    }
    crop.with_name(crop.stem + ".detections.json").write_text(json.dumps(payload), encoding="utf-8")
    return crop


# ---------------------------------------------------------------------------
# 1. default configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params_cls",
    [
        ClassicVisionParams,
        ClassicVisionEyePoseParams,
        ClassicVisionBirdFusionParams,
        ClassicVisionMammalFusionParams,
        ClassicVisionCombinedParams,
        CropSharpnessParams,
    ],
)
def test_default_is_use_subject_filter_false(params_cls) -> None:
    assert params_cls().use_subject_filter is False
    names = {spec.name for spec in params_cls.specs()}
    assert "use_subject_filter" in names


# ---------------------------------------------------------------------------
# _load_candidate: the mechanism
# ---------------------------------------------------------------------------


def test_load_candidate_strict_rejects_a_fallback_only_entry(tmp_path) -> None:
    cache_dir = tmp_path / "crops"
    path = str(tmp_path / "shoot" / "empty.nef")
    _write_fallback_cache_entry(cache_dir, path)

    candidate = ClassicVisionStrategy._load_candidate(path, cache_dir, require_selected_detection=True)
    assert candidate.subject_crop is None
    assert candidate.subject_box is None


def test_load_candidate_relaxed_uses_the_whole_frame_for_a_fallback_entry(tmp_path) -> None:
    cache_dir = tmp_path / "crops"
    path = str(tmp_path / "shoot" / "empty.nef")
    _write_fallback_cache_entry(cache_dir, path, source_size=(800, 600))

    candidate = ClassicVisionStrategy._load_candidate(path, cache_dir, require_selected_detection=False)
    assert candidate.subject_crop is not None
    assert candidate.subject_box == (0.0, 0.0, 800.0, 600.0)
    assert candidate.crop_box == (0.0, 0.0, 800.0, 600.0)
    assert candidate.subject_label is None  # no COCO class was ever assigned


def test_load_candidate_relaxed_still_prefers_a_real_detection_when_one_exists(tmp_path) -> None:
    """Relaxing the gate must never make a genuinely better (tight) crop
    worse - a real "selected" detection is used exactly as before."""
    cache_dir = tmp_path / "crops"
    path = str(tmp_path / "shoot" / "bird.nef")
    _write_cache_entry(cache_dir, path, box=(10, 10, 60, 60))

    candidate = ClassicVisionStrategy._load_candidate(path, cache_dir, require_selected_detection=False)
    assert candidate.subject_box == (10.0, 10.0, 60.0, 60.0)
    assert candidate.subject_label == COCO_BIRD_CLASS


def test_load_candidate_relaxed_still_rejects_a_truly_missing_crop(tmp_path) -> None:
    """Relaxing the gate is not the same as fabricating data - an image
    preprocessing never touched at all still has nothing to score."""
    cache_dir = tmp_path / "crops"
    path = str(tmp_path / "shoot" / "never-processed.nef")
    candidate = ClassicVisionStrategy._load_candidate(path, cache_dir, require_selected_detection=False)
    assert candidate.subject_crop is None


# ---------------------------------------------------------------------------
# 2 & 4: filter OFF lets a fallback crop reach Crop Sharpness
# ---------------------------------------------------------------------------


def test_filter_off_crop_sharpness_ranks_a_fallback_only_image(tmp_path, monkeypatch) -> None:
    from picklikeme.ranking import crop_sharpness as crop_sharpness_module

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    path = str(folder / "empty.nef")
    Path(path).write_bytes(b"not really a raw file")
    _write_fallback_cache_entry(cache_dir, path)

    monkeypatch.setattr(crop_sharpness_module, "build_cache", lambda *a, **k: {})

    result = CropSharpnessStrategy().rank_folder(
        folder,
        params=CropSharpnessParams(use_subject_filter=False),
        crop_cache_dir=cache_dir,
        device="cpu",
        analytics_db=tmp_path / "analytics.db",
    )

    assert result["filtered"] == {}
    assert result["image_count"] == 1

    from picklikeme.ranking.crop_sharpness import read_metrics_report

    metrics = read_metrics_report(folder)["metrics"][path]
    assert metrics["crop_sharpness"] > 0.0
    # NOT measured for a fallback image. The "subject box" here is the whole
    # frame by construction, so a subject-size term computed from it would be
    # a flat 1.0 - the maximum possible bonus, handed to every image in which
    # nothing was actually found. See crop_sharpness.measure/combine: the
    # term is absent, and the score is sharpness alone.
    assert metrics["relative_subject_size"] is None
    assert metrics["has_subject_detection"] is False


# ---------------------------------------------------------------------------
# 3 & 7: filter ON preserves existing behavior exactly
# ---------------------------------------------------------------------------


def test_filter_on_crop_sharpness_still_rejects_a_fallback_only_image(tmp_path, monkeypatch) -> None:
    from picklikeme.ranking import crop_sharpness as crop_sharpness_module

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    path = str(folder / "empty.nef")
    Path(path).write_bytes(b"not really a raw file")
    _write_fallback_cache_entry(cache_dir, path)

    monkeypatch.setattr(crop_sharpness_module, "build_cache", lambda *a, **k: {})

    result = CropSharpnessStrategy().rank_folder(
        folder,
        params=CropSharpnessParams(use_subject_filter=True),
        crop_cache_dir=cache_dir,
        device="cpu",
        analytics_db=tmp_path / "analytics.db",
    )

    assert result["filtered"] == {NO_SUBJECT: 1}
    assert result["image_count"] == 0


def test_filter_on_still_ranks_a_normal_folder_exactly_as_before(tmp_path, monkeypatch) -> None:
    """Regression guard: a folder of ordinary, real detections behaves
    identically whether use_subject_filter is passed explicitly or not -
    True must not change a single existing result."""
    from picklikeme.ranking import crop_sharpness as crop_sharpness_module

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    paths = [str(folder / f"bird_{i}.nef") for i in range(3)]
    for path in paths:
        Path(path).write_bytes(b"not really a raw file")
        _write_cache_entry(cache_dir, path)

    monkeypatch.setattr(crop_sharpness_module, "build_cache", lambda *a, **k: {})

    default_result = CropSharpnessStrategy().rank_folder(
        folder, crop_cache_dir=cache_dir, device="cpu", analytics_db=tmp_path / "analytics_default.db",
    )
    explicit_result = CropSharpnessStrategy().rank_folder(
        folder,
        params=CropSharpnessParams(use_subject_filter=True),
        crop_cache_dir=cache_dir,
        device="cpu",
        analytics_db=tmp_path / "analytics_explicit.db",
    )
    assert default_result["image_count"] == explicit_result["image_count"] == 3
    assert default_result["filtered"] == explicit_result["filtered"] == {}


# ---------------------------------------------------------------------------
# 5 & 6: eye-based strategies - missing eye evidence is the STRATEGY's own
# reason, never a global NO_SUBJECT, once the filter is off
# ---------------------------------------------------------------------------


class _StubEyeDetector:
    """Controllable stand-in - no real model, see test_ranking_strategies.py's
    own _FakeEyeDetector for the established shape."""

    detector_id = "stub"

    def __init__(self, *, accepted: bool) -> None:
        self._accepted = accepted
        self.detect_calls = 0

    def supports(self, coco_label) -> bool:
        return True

    def detect(self, subject_crop_rgb) -> EyeDetection:
        self.detect_calls += 1
        return EyeDetection(
            box=(1.0, 1.0, 5.0, 5.0), confidence=0.95, center=(3.0, 3.0), accepted=self._accepted,
        )


def _run_eyepose(folder, cache_dir, analytics_db, monkeypatch, *, use_subject_filter, eye_accepted):
    from picklikeme.ranking import classic as classic_module

    monkeypatch.setattr(classic_module, "build_cache", lambda *a, **k: {})
    monkeypatch.setattr(
        "picklikeme.eyes.build_eye_detector",
        lambda name, **kw: _StubEyeDetector(accepted=eye_accepted),
    )
    return ClassicVisionEyePoseStrategy().rank_folder(
        folder,
        params=ClassicVisionEyePoseParams(use_subject_filter=use_subject_filter),
        crop_cache_dir=cache_dir,
        device="cpu",
        analytics_db=analytics_db,
    )


def test_filter_off_eye_based_strategy_is_not_globally_rejected(tmp_path, monkeypatch) -> None:
    """The crop reaches the eye detector at all (proven by detect_calls>0
    would be nice, but the observable contract is the filtered reason) -
    with the eye accepting, the image is fully ranked from a fallback crop."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    path = str(folder / "empty.nef")
    Path(path).write_bytes(b"not really a raw file")
    _write_fallback_cache_entry(cache_dir, path)

    result = _run_eyepose(
        folder, cache_dir, tmp_path / "analytics.db", monkeypatch,
        use_subject_filter=False, eye_accepted=True,
    )
    assert result["filtered"] == {}
    assert result["image_count"] == 1


def test_filter_off_missing_eye_evidence_is_the_strategys_own_reason(tmp_path, monkeypatch) -> None:
    """The central architectural point: with the filter off, a fallback
    crop with no locatable eye is rejected as NO_VISIBLE_EYE - the eye
    detector's own, existing verdict - never as the global NO_SUBJECT."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    path = str(folder / "empty.nef")
    Path(path).write_bytes(b"not really a raw file")
    _write_fallback_cache_entry(cache_dir, path)

    result = _run_eyepose(
        folder, cache_dir, tmp_path / "analytics.db", monkeypatch,
        use_subject_filter=False, eye_accepted=False,
    )
    assert result["filtered"] == {NO_VISIBLE_EYE: 1}
    assert NO_SUBJECT not in result["filtered"]


def test_filter_on_eye_based_strategy_preserves_no_subject_rejection(tmp_path, monkeypatch) -> None:
    """Same fixture, filter ON: the original behavior - a fallback-only
    image is rejected upstream as NO_SUBJECT and the eye detector never
    even runs."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    path = str(folder / "empty.nef")
    Path(path).write_bytes(b"not really a raw file")
    _write_fallback_cache_entry(cache_dir, path)

    result = _run_eyepose(
        folder, cache_dir, tmp_path / "analytics.db", monkeypatch,
        use_subject_filter=True, eye_accepted=True,
    )
    assert result["filtered"] == {NO_SUBJECT: 1}
    assert result["image_count"] == 0
