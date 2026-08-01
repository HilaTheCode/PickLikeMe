"""ranking.debug - the optional per-image debug image for a Classic Vision
run (see ClassicVisionStrategy.rank_folder's `debug_dir` parameter).

Draws only from FilterCandidate/EyeDetection - the shape every backend
already produces - so these tests use a bare EyeDetection/FilterCandidate
rather than any real detector, exactly like test_ranking_strategies.py's
own _FakeEyeDetector-based tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from picklikeme.eyes.detector import EyeDetection, EyeKeypoint
from picklikeme.ranking.debug import debug_image_path, render_debug_image, save_debug_image
from picklikeme.ranking.filters import FilterCandidate


def _crop(size: int = 40) -> np.ndarray:
    return np.full((size, size, 3), 100, dtype=np.uint8)


def _candidate(*, eye=None, subject_crop=True) -> FilterCandidate:
    candidate = FilterCandidate(image_path="/shoot/IMG_0001.NEF")
    if subject_crop:
        candidate.subject_crop = _crop()
        candidate.subject_box = (100.0, 100.0, 500.0, 500.0)
        candidate.source_size = (2000, 1500)
    candidate.eye = eye
    return candidate


def _eye(*, accepted: bool = True, confidence: float = 0.9) -> EyeDetection:
    return EyeDetection(
        box=(10.0, 10.0, 20.0, 20.0),
        confidence=confidence,
        center=(15.0, 15.0),
        detector_id="fake-detector",
        left=EyeKeypoint(x=15.0, y=15.0, confidence=confidence),
        right=EyeKeypoint(x=25.0, y=16.0, confidence=confidence - 0.1),
        accepted=accepted,
    )


class TestRenderDebugImage:
    def test_no_subject_crop_yields_nothing_to_render(self) -> None:
        candidate = _candidate(subject_crop=False)
        assert render_debug_image(candidate, "classic-vision") is None

    def test_a_crop_with_no_eye_still_renders_with_an_explanatory_line(self) -> None:
        candidate = _candidate(eye=None)
        image = render_debug_image(candidate, "classic-vision")
        assert image is not None
        assert image.width == 40  # the crop's own width, unchanged
        assert image.height > 40  # plus the text panel underneath

    def test_an_accepted_eye_is_drawn_in_the_accepted_colour(self) -> None:
        from picklikeme.analyzer.contactsheets import EYE_BOX_ACCEPTED

        candidate = _candidate(eye=_eye(accepted=True))
        image = render_debug_image(candidate, "classic-vision")
        pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
        close = (
            (np.abs(pixels[:, 0].astype(int) - EYE_BOX_ACCEPTED[0]) < 20)
            & (np.abs(pixels[:, 1].astype(int) - EYE_BOX_ACCEPTED[1]) < 20)
            & (np.abs(pixels[:, 2].astype(int) - EYE_BOX_ACCEPTED[2]) < 20)
        ).sum()
        assert close > 5, "no accepted-eye colour found in the rendered debug image"

    def test_a_rejected_eye_is_drawn_in_the_rejected_colour(self) -> None:
        from picklikeme.analyzer.contactsheets import EYE_BOX_REJECTED

        candidate = _candidate(eye=_eye(accepted=False))
        image = render_debug_image(candidate, "classic-vision")
        pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
        close = (
            (np.abs(pixels[:, 0].astype(int) - EYE_BOX_REJECTED[0]) < 20)
            & (np.abs(pixels[:, 1].astype(int) - EYE_BOX_REJECTED[1]) < 20)
            & (np.abs(pixels[:, 2].astype(int) - EYE_BOX_REJECTED[2]) < 20)
        ).sum()
        assert close > 5, "no rejected-eye colour found in the rendered debug image"


class TestSaveDebugImage:
    def test_writes_a_jpeg_named_after_the_source_image(self, tmp_path) -> None:
        candidate = _candidate(eye=_eye())
        target = save_debug_image(candidate, "classic-vision", tmp_path)
        assert target is not None
        assert target == debug_image_path(tmp_path, candidate.image_path)
        assert target.name == "IMG_0001_debug.jpg"
        assert target.is_file()
        Image.open(target).verify()  # a real, readable JPEG, not a stub

    def test_no_crop_writes_nothing(self, tmp_path) -> None:
        candidate = _candidate(subject_crop=False)
        assert save_debug_image(candidate, "classic-vision", tmp_path) is None
        assert not any(tmp_path.iterdir())

    def test_creates_the_debug_directory_if_it_does_not_exist(self, tmp_path) -> None:
        nested = tmp_path / "does" / "not" / "exist"
        candidate = _candidate(eye=_eye())
        target = save_debug_image(candidate, "classic-vision", nested)
        assert target is not None and target.is_file()


if __name__ == "__main__":
    pytest.main([__file__])
