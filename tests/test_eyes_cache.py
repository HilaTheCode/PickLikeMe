"""EyeRecord round-trip coverage for eyes/cache.py, focused on
head_confidence (EyePose Investigation Phase 1, Part 2): the independent
"is a real head present" signal must survive a save/read cycle alongside
the rest of the eye-detector result.
"""

from __future__ import annotations

from pathlib import Path

from picklikeme.eyes.cache import EYE_CACHE_VERSION, read_eye_detection, save_eye_detection
from picklikeme.eyes.detector import EyeDetection, EyeKeypoint


def _detection(head_confidence: float | None, **landmarks) -> EyeDetection:
    return EyeDetection(
        box=(1.0, 2.0, 3.0, 4.0),
        confidence=0.9,
        detector_id="eyepose_v0",
        accepted=True,
        left=EyeKeypoint(x=5.0, y=6.0, confidence=0.9),
        right=EyeKeypoint(x=7.0, y=8.0, confidence=0.1),
        head_confidence=head_confidence,
        **landmarks,
    )


def test_head_confidence_round_trips_through_the_cache(tmp_path: Path) -> None:
    source = tmp_path / "DSC03129.ARW"
    source.write_bytes(b"fake raw")
    save_eye_detection(tmp_path, source, subject_crop_size=(200, 150), detection=_detection(0.026))

    record = read_eye_detection(tmp_path, source)

    assert record is not None
    assert record.head_confidence == 0.026


def test_a_backend_with_no_head_confidence_signal_round_trips_as_none(tmp_path: Path) -> None:
    source = tmp_path / "DSC_1179.ARW"
    source.write_bytes(b"fake raw")
    save_eye_detection(tmp_path, source, subject_crop_size=(200, 150), detection=_detection(None))

    record = read_eye_detection(tmp_path, source)

    assert record is not None
    assert record.head_confidence is None


def test_the_other_four_landmarks_round_trip_through_the_cache(tmp_path: Path) -> None:
    """v3 (Image Inspector landmark overlay): beak/head_top/left_shoulder/
    right_shoulder must survive save/read, exactly like left/right already
    do - these were computed by EyePose-v0 all along but never threaded
    through EyeDetection until now."""
    source = tmp_path / "DSC03130.ARW"
    source.write_bytes(b"fake raw")
    detection = _detection(
        0.9,
        beak=EyeKeypoint(x=9.0, y=10.0, confidence=0.8),
        head_top=EyeKeypoint(x=11.0, y=12.0, confidence=0.7),
        left_shoulder=EyeKeypoint(x=13.0, y=14.0, confidence=0.6),
        right_shoulder=EyeKeypoint(x=15.0, y=16.0, confidence=0.5),
    )
    save_eye_detection(tmp_path, source, subject_crop_size=(200, 150), detection=detection)

    record = read_eye_detection(tmp_path, source)

    assert record is not None
    assert (record.beak.x, record.beak.y, record.beak.confidence) == (9.0, 10.0, 0.8)
    assert (record.head_top.x, record.head_top.y, record.head_top.confidence) == (11.0, 12.0, 0.7)
    assert (record.left_shoulder.x, record.left_shoulder.y, record.left_shoulder.confidence) == (13.0, 14.0, 0.6)
    assert (record.right_shoulder.x, record.right_shoulder.y, record.right_shoulder.confidence) == (15.0, 16.0, 0.5)


def test_a_backend_with_no_body_landmarks_round_trips_them_as_none(tmp_path: Path) -> None:
    """SuperAnimal-Bird (and any older cached row) never computes these -
    None must survive the round trip, not a fabricated zero keypoint."""
    source = tmp_path / "DSC03131.ARW"
    source.write_bytes(b"fake raw")
    save_eye_detection(tmp_path, source, subject_crop_size=(200, 150), detection=_detection(None))

    record = read_eye_detection(tmp_path, source)

    assert record is not None
    assert record.beak is None
    assert record.head_top is None
    assert record.left_shoulder is None
    assert record.right_shoulder is None


def test_a_stale_cache_version_is_ignored(tmp_path: Path) -> None:
    from picklikeme.eyes.cache import eye_cache_path
    import json

    source = tmp_path / "DSC_4264.ARW"
    source.write_bytes(b"fake raw")
    target = eye_cache_path(tmp_path, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"version": EYE_CACHE_VERSION - 1, "head_confidence": 0.9}), encoding="utf-8")

    assert read_eye_detection(tmp_path, source) is None
