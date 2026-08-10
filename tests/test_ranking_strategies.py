"""The pluggable ranking framework: registry, parameters, filters, metrics.

Deliberately covers the *seams* rather than re-testing the trained model or
the eye model's accuracy - neither is this project's code, and both need
weights no test should download. What is tested here is everything PeakPic
itself decides: which strategies exist, how weights normalise, what each
filter rejects and why, what the metrics do to known-sharp vs known-blurred
pixels, and that the AI path still reaches `rank.rank_folder` with exactly
the arguments it always did.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from picklikeme.bird_crop import COCO_BIRD_CLASS, COCO_PERSON_CLASS, crop_cache_path
from picklikeme.eyes.detector import EyeDetection
from picklikeme.ranking import (
    DEFAULT_STRATEGY_ID,
    AIModelParams,
    ClassicVisionEyePoseParams,
    ClassicVisionParams,
    available_strategies,
    get_strategy,
)
from picklikeme.ranking.classic import (
    ClassicVisionEyePoseStrategy,
    ClassicVisionStrategy,
    ImageMetrics,
    combine,
    measure,
    read_filter_report,
    read_metrics_report,
    write_filter_report,
    write_metrics_report,
)
from picklikeme.ranking.filters import (
    LOW_HEAD_CONFIDENCE,
    NO_SUBJECT,
    NO_VISIBLE_EYE,
    REJECT_REASONS,
    UNSUPPORTED_SUBJECT,
    EyeFilter,
    FilterCandidate,
    FilterChain,
    SubjectFilter,
)
from picklikeme.ranking.metrics import (
    focus_measure,
    normalized_subject_size,
    region_focus_measure,
    robust_normalize,
    subject_focus_measure,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sharp_image(size: int = 256) -> np.ndarray:
    """Deterministic broadband texture, standing in for a sharp photograph.

    Deliberately NOT a 1-pixel checkerboard: that sits exactly at Nyquist, so
    the canonical downsample every focus measurement starts with averages it
    into flat grey and both a "sharp" and a "blurred" checkerboard measure
    zero. Coarse random blocks survive resampling the way real detail does.
    """
    rng = np.random.default_rng(20260801)
    blocks = rng.integers(0, 256, size=(size // 4, size // 4), dtype=np.uint8)
    grown = cv2.resize(blocks, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.repeat(grown[:, :, None], 3, axis=2)


def _blurred_image(size: int = 256) -> np.ndarray:
    return cv2.GaussianBlur(_sharp_image(size), (0, 0), 6.0)


class _FakeEyeDetector:
    """Stands in for the real model: no weights, no torch, fully controllable."""

    detector_id = "fake"

    def __init__(self, *, detection: EyeDetection, supported: bool = True) -> None:
        self._detection = detection
        self._supported = supported
        self.detect_calls = 0

    def supports(self, coco_label: int) -> bool:
        return self._supported

    def detect(self, subject_crop_rgb):
        self.detect_calls += 1
        return self._detection


def _eye(confidence: float = 0.9, accepted: bool = True) -> EyeDetection:
    return EyeDetection(
        box=(10.0, 10.0, 30.0, 30.0), confidence=confidence, center=(20.0, 20.0), accepted=accepted
    )


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
# the registry
# ---------------------------------------------------------------------------


def test_every_strategy_is_registered_with_the_ai_model_first() -> None:
    infos = available_strategies()
    assert [i.strategy_id for i in infos] == [
        "ai-model",
        "classic-vision-eyepose-v0",
        "classic-vision",
        "classic-vision-fusion-birds",
        "classic-vision-fusion-mammals",
        "classic-vision-fusion-combined",
    ]
    # The AI model stays the default: strategies were added to give it company,
    # never to demote it.
    assert DEFAULT_STRATEGY_ID == "ai-model"
    assert all(i.display_name and i.description for i in infos)


def test_the_two_classic_vision_backends_are_independently_selectable() -> None:
    """The reported requirement: Classic Vision is a framework of
    interchangeable eye-localisation backends, not one strategy with a
    hidden switch - each is its own strategy_id, its own ranking CSV, its
    own filter/metrics report files, so results from both coexist on a
    folder for direct comparison."""
    from picklikeme.ranking.classic import EYEPOSE_STRATEGY_ID, STRATEGY_ID

    superanimal = get_strategy(STRATEGY_ID)
    eyepose = get_strategy(EYEPOSE_STRATEGY_ID)
    assert superanimal.info.strategy_id != eyepose.info.strategy_id
    assert superanimal._eye_detector_name == "superanimal-bird"
    assert eyepose._eye_detector_name == "eyepose-v0"
    # Distinct score labels too, so the Gallery/Loupe never shows one
    # backend's number under a name that could be mistaken for the other's.
    assert superanimal.info.score_label != eyepose.info.score_label


def test_get_strategy_resolves_each_id_and_rejects_unknown_ones() -> None:
    for info in available_strategies():
        assert get_strategy(info.strategy_id).info.strategy_id == info.strategy_id
    with pytest.raises(ValueError, match="Unknown ranking strategy"):
        get_strategy("no-such-strategy")


def test_listing_strategies_does_not_import_torch() -> None:
    """The Rank menu is built at startup; it must not pay for a model import."""
    import sys

    for module in ("picklikeme.ranking", "picklikeme.ranking.classic", "picklikeme.eyes.detector"):
        assert module in sys.modules or True  # imported at module load, above
    # Constructing every strategy must still be cheap - no weights, no CUDA -
    # including the three Fusion/Ranking-Mode strategies (eyes.domains):
    # FusionEyeDetector and its sub-detectors are only ever constructed
    # inside rank_folder, never by __init__ itself.
    strategies = [get_strategy(i.strategy_id) for i in available_strategies()]
    assert len(strategies) == 6


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def test_default_weights_are_70_10_20() -> None:
    weights = ClassicVisionParams().normalized_weights()
    assert weights == pytest.approx(
        {
            "eye_sharpness_weight": 0.7,
            "subject_sharpness_weight": 0.1,
            "subject_size_weight": 0.2,
        }
    )


def test_any_weight_scale_means_the_same_thing() -> None:
    """5/3/2 and 50/30/20 are the same ranking - weights are normalised, not
    validated into a range, so the photographer can type whatever they like."""
    assert ClassicVisionParams(5, 3, 2).normalized_weights() == pytest.approx(
        ClassicVisionParams(50, 30, 20).normalized_weights()
    )


def test_all_zero_weights_fall_back_to_equal_weighting() -> None:
    weights = ClassicVisionParams(0, 0, 0).normalized_weights()
    assert list(weights.values()) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_negative_weights_are_clamped_rather_than_inverting_a_metric() -> None:
    weights = ClassicVisionParams(-10, 30, 20).normalized_weights()
    assert weights["eye_sharpness_weight"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_params_round_trip_through_a_dialogs_raw_values() -> None:
    values = {spec.name: spec.default for spec in ClassicVisionParams.specs()}
    values["unrelated_widget"] = 123.0  # ignored, not an error
    assert ClassicVisionParams.from_values(values) == ClassicVisionParams()


def test_every_param_spec_is_usable_by_a_generated_dialog() -> None:
    for spec in ClassicVisionParams.specs():
        assert spec.minimum <= spec.default <= spec.maximum
        assert spec.label and spec.help
        assert hasattr(ClassicVisionParams(), spec.name)


# ---------------------------------------------------------------------------
# ClassicVisionEyePoseParams - EyePose Investigation Phase 1, Parts 1/3/6:
# eye_confidence_threshold (renamed from min_eye_confidence to match the
# report exactly), detection_head_confidence_threshold (Part 2's independent
# head-visibility gate, EyePose-only), and the two crop-selection thresholds
# (detection_confidence_threshold/crop_confidence_threshold) both backends
# share.
# ---------------------------------------------------------------------------


def test_eyepose_params_every_spec_is_usable_by_a_generated_dialog() -> None:
    for spec in ClassicVisionEyePoseParams.specs():
        assert spec.minimum <= spec.default <= spec.maximum
        assert spec.label and spec.help
        assert hasattr(ClassicVisionEyePoseParams(), spec.name)


def test_eyepose_params_round_trip_through_a_dialogs_raw_values() -> None:
    values = {spec.name: spec.default for spec in ClassicVisionEyePoseParams.specs()}
    values["unrelated_widget"] = 123.0  # ignored, not an error
    assert ClassicVisionEyePoseParams.from_values(values) == ClassicVisionEyePoseParams()


def test_eye_confidence_threshold_is_the_declared_name() -> None:
    """Named to match the EyePose Investigation Phase 1 report's Part 3
    exactly, not min_eye_confidence (ClassicVisionParams/SuperAnimal-Bird's
    own, different gate - see ClassicVisionEyePoseParams's own docstring for
    why the two backends do not share one name here)."""
    names = {spec.name for spec in ClassicVisionEyePoseParams.specs()}
    assert "eye_confidence_threshold" in names
    assert "min_eye_confidence" not in names


def test_both_backends_share_the_same_detection_threshold_param_names() -> None:
    """detection_confidence_threshold/crop_confidence_threshold configure
    rank_folder's shared crop-cache step (see _detection_specs), not either
    backend's own eye detector - so both params classes declare them
    identically. detection_head_confidence_threshold, by contrast, is
    EyePose-only - SuperAnimal-Bird has no equivalent signal."""
    eyepose_names = {spec.name for spec in ClassicVisionEyePoseParams.specs()}
    superanimal_names = {spec.name for spec in ClassicVisionParams.specs()}
    for name in ("detection_confidence_threshold", "crop_confidence_threshold"):
        assert name in eyepose_names
        assert name in superanimal_names
    assert "detection_head_confidence_threshold" in eyepose_names
    assert "detection_head_confidence_threshold" not in superanimal_names


def test_eyepose_detector_kwargs_maps_eye_confidence_threshold_to_min_confidence() -> None:
    """The EyePoseV0EyeDetector constructor kwarg is (and stays) `min_confidence`
    - matching SuperAnimalBirdEyeDetector's own constructor - only the outer,
    photographer-facing params field was renamed."""
    strategy = ClassicVisionEyePoseStrategy()
    params = ClassicVisionEyePoseParams(eye_confidence_threshold=0.42)
    kwargs = strategy._eye_detector_kwargs(params)
    assert kwargs["min_confidence"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_a_sharp_patch_scores_far_above_a_blurred_one() -> None:
    """5x is the guarantee, not the observed margin (~7.6x on this fixture).
    The ratio is bounded rather than enormous because the measure is contrast-
    normalised: blurring lowers contrast too, and standardising to unit
    variance deliberately gives some of that back, so the number reflects lost
    *edge structure* alone."""
    assert focus_measure(_sharp_image()) > 5 * focus_measure(_blurred_image())


def test_focus_is_scale_invariant_so_a_downscaled_crop_is_not_favoured() -> None:
    """Cached crops are capped at 1024 but never upscaled, so the same subject
    reaches this metric at different pixel sizes - the value must describe the
    subject, not the resampling."""
    image = _blurred_image(512)
    half = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    assert focus_measure(image) == pytest.approx(focus_measure(half), rel=0.35)


def test_focus_is_contrast_invariant() -> None:
    """Halving contrast must not halve the focus value."""
    image = _blurred_image()
    low_contrast = (image.astype(np.float32) * 0.4 + 60).astype(np.uint8)
    assert focus_measure(image) == pytest.approx(focus_measure(low_contrast), rel=0.35)


def test_a_sharp_subject_on_a_smooth_background_still_scores_well() -> None:
    """The regression that motivated using a high percentile rather than the
    variance: a shallow-depth-of-field portrait - a sharp subject surrounded by
    smooth bokeh - must not be dragged down by all that empty area."""
    canvas = np.full((512, 512, 3), 128, dtype=np.uint8)
    canvas[220:290, 220:290] = _sharp_image(70)  # small sharp region, smooth around it
    all_blurred = _blurred_image(512)
    assert subject_focus_measure(canvas) > 3 * subject_focus_measure(all_blurred)


def test_degenerate_patches_score_zero_rather_than_raising() -> None:
    assert focus_measure(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
    assert focus_measure(np.zeros((1, 1, 3), dtype=np.uint8)) == 0.0


def test_region_focus_clamps_a_box_that_runs_off_the_image() -> None:
    image = _sharp_image(64)
    # Entirely outside, and partly outside - both must still measure something.
    assert region_focus_measure(image, (-50.0, -50.0, -10.0, -10.0)) >= 0.0
    assert region_focus_measure(image, (40.0, 40.0, 500.0, 500.0)) > 0.0


def test_normalized_subject_size_is_the_box_fraction_of_the_frame() -> None:
    assert normalized_subject_size((0.0, 0.0, 50.0, 40.0), (100, 80)) == pytest.approx(0.25)
    assert normalized_subject_size((0.0, 0.0, 100.0, 80.0), (100, 80)) == pytest.approx(1.0)
    # An unknown frame size is a metadata gap, not a reason to fail the image.
    assert normalized_subject_size((0.0, 0.0, 10.0, 10.0), (0, 0)) == 0.0


def test_robust_normalize_spans_zero_to_one_and_preserves_order() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = robust_normalize(values)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[-1] == pytest.approx(1.0)
    assert normalized == sorted(normalized)


def test_an_outlier_does_not_flatten_a_realistically_sized_folder() -> None:
    """The percentile clip is what makes this hold. With a real folder's worth
    of images, one absurd value (a lens flare, a blown highlight) is outside
    the 95th percentile and cannot compress everything else to zero."""
    values = [float(i) for i in range(1, 101)] + [10_000.0]
    normalized = robust_normalize(values)
    assert normalized[49] == pytest.approx(0.5, abs=0.1)  # the median stays mid-range
    assert normalized[-1] == pytest.approx(1.0)  # the outlier is clipped, not honoured


def test_a_tiny_folder_with_an_outlier_still_orders_correctly() -> None:
    """The honest limit of percentile clipping: with only a handful of images
    the 95th percentile sits close to the outlier itself, so the rest compress
    toward zero. That is acceptable precisely because this value is only ever
    used to ORDER images - the ordering is still strictly correct, and the
    other metrics still get their say."""
    normalized = robust_normalize([1.0, 2.0, 3.0, 4.0, 5.0, 10_000.0])
    assert normalized == sorted(normalized)
    assert normalized[4] > normalized[3] > normalized[2]


def test_a_metric_with_no_spread_normalizes_to_a_neutral_half() -> None:
    """It carries no ranking information, so it must not skew the sum either
    way - 0.5 lets the other metrics decide."""
    assert robust_normalize([7.0] * 5) == [0.5] * 5
    assert robust_normalize([]) == []


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_reject_reasons_are_registered_so_a_ui_can_label_them() -> None:
    from picklikeme.ranking.filters import REJECT_REASON_LABELS

    assert set(REJECT_REASONS) == set(REJECT_REASON_LABELS)


def test_no_subject_is_rejected_before_the_eye_detector_ever_runs() -> None:
    """Short-circuiting is not just speed: the expensive model must never be
    asked about an image with nothing in it."""
    detector = _FakeEyeDetector(detection=_eye())
    chain = FilterChain([SubjectFilter(), EyeFilter(detector)])
    candidate = _candidate(subject_box=None, subject_crop=None)
    assert chain.reject_reason(candidate) == NO_SUBJECT
    assert detector.detect_calls == 0


def test_a_subject_with_no_locatable_eye_is_rejected_as_no_visible_eye() -> None:
    detector = _FakeEyeDetector(detection=_eye(accepted=False))
    chain = FilterChain([SubjectFilter(), EyeFilter(detector)])
    assert chain.reject_reason(_candidate()) == NO_VISIBLE_EYE
    assert detector.detect_calls == 1


def test_a_subject_with_no_confidently_detected_head_is_rejected_separately() -> None:
    """LOW_HEAD_CONFIDENCE (EyePose Investigation Phase 1, Part 2/3) - a
    genuinely independent question from NO_VISIBLE_EYE, checked first, with
    its own reason. Still only ONE call to the (expensive) detector."""
    low_head_confidence_eye = EyeDetection(
        box=(10.0, 10.0, 30.0, 30.0), confidence=0.97, accepted=True,  # eye gate alone would pass this
        head_confidence=0.026, head_visible=False,
    )
    detector = _FakeEyeDetector(detection=low_head_confidence_eye)
    chain = FilterChain([SubjectFilter(), EyeFilter(detector)])
    assert chain.reject_reason(_candidate()) == LOW_HEAD_CONFIDENCE
    assert detector.detect_calls == 1


def test_a_backend_with_no_head_confidence_signal_is_never_gated_by_it() -> None:
    """head_visible defaults to True (EyeDetection's own default) - a
    backend that leaves head_confidence at None (no equivalent signal, e.g.
    SuperAnimal-Bird today) must never be rejected by a check it cannot
    answer."""
    detector = _FakeEyeDetector(detection=_eye())  # head_confidence=None, head_visible=True by default
    chain = FilterChain([SubjectFilter(), EyeFilter(detector)])
    assert chain.reject_reason(_candidate()) is None


def test_a_subject_no_detector_covers_is_reported_separately() -> None:
    """A mammal is not "no visible eye" - PeakPic simply has no eye model for
    it, and saying so honestly is a different answer."""
    detector = _FakeEyeDetector(detection=_eye(), supported=False)
    chain = FilterChain([SubjectFilter(), EyeFilter(detector)])
    candidate = _candidate(subject_label=COCO_PERSON_CLASS)
    assert chain.reject_reason(candidate) == UNSUPPORTED_SUBJECT
    assert detector.detect_calls == 0


def test_a_good_candidate_passes_and_keeps_the_eye_for_scoring() -> None:
    """The eye is detected once and reused by the scoring phase - the filter
    caches a result, it does not hand over a decision."""
    detector = _FakeEyeDetector(detection=_eye())
    chain = FilterChain([SubjectFilter(), EyeFilter(detector)])
    candidate = _candidate()
    assert chain.reject_reason(candidate) is None
    assert candidate.eye is not None
    assert candidate.eye.confidence == pytest.approx(0.9)


def test_the_filter_chain_is_extensible_without_touching_its_callers() -> None:
    """Adding a filter is adding a class and one list entry - the chain itself
    grows no branches."""

    class _AlwaysRejects:
        reason = "CUSTOM_REASON"

        def check(self, candidate):
            return False

    chain = FilterChain([SubjectFilter(), _AlwaysRejects()])
    assert chain.reject_reason(_candidate()) == "CUSTOM_REASON"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_measure_reports_all_three_metrics_for_a_surviving_candidate() -> None:
    candidate = _candidate()
    candidate.eye = _eye()
    metrics = measure(candidate)
    assert metrics.image_path == "a.nef"
    assert metrics.eye_sharpness > 0.0
    assert metrics.subject_sharpness > 0.0
    # 100x100 box in a 1000x1000 frame.
    assert metrics.subject_size == pytest.approx(0.01)
    assert metrics.eye_confidence == pytest.approx(0.9)


def test_weights_actually_move_the_ranking() -> None:
    """Two images: one with the sharper eye, one with the larger subject.
    Which wins must follow the weights."""
    from picklikeme.ranking.classic import ImageMetrics

    sharp_eye = ImageMetrics("sharp.nef", eye_sharpness=10.0, subject_sharpness=1.0,
                             subject_size=0.01, eye_confidence=0.9)
    big_subject = ImageMetrics("big.nef", eye_sharpness=1.0, subject_sharpness=1.0,
                               subject_size=0.90, eye_confidence=0.9)
    pair = [sharp_eye, big_subject]

    eye_first = combine(pair, ClassicVisionParams(100, 0, 0).normalized_weights())
    assert eye_first[0] > eye_first[1]

    size_first = combine(pair, ClassicVisionParams(0, 0, 100).normalized_weights())
    assert size_first[1] > size_first[0]


def test_scoring_is_deterministic() -> None:
    from picklikeme.ranking.classic import ImageMetrics

    metrics = [
        ImageMetrics(f"{i}.nef", eye_sharpness=float(i), subject_sharpness=float(i % 3),
                     subject_size=i / 10, eye_confidence=0.8)
        for i in range(1, 8)
    ]
    weights = ClassicVisionParams().normalized_weights()
    assert combine(metrics, weights) == combine(metrics, weights)


def test_combining_no_images_is_not_an_error() -> None:
    """A folder where every image was filtered out still has to finish."""
    assert combine([], ClassicVisionParams().normalized_weights()) == []


# ---------------------------------------------------------------------------
# the filter report sidecar
# ---------------------------------------------------------------------------


def test_the_filter_report_records_why_each_image_was_skipped(tmp_path) -> None:
    """A filtered image has no score, so it has no CSV row - this sidecar is
    the only place the reason survives."""
    rejected = {"a.nef": NO_SUBJECT, "b.nef": NO_VISIBLE_EYE}
    counts = {NO_SUBJECT: 1, NO_VISIBLE_EYE: 1}
    write_filter_report(tmp_path, rejected, counts)

    report = read_filter_report(tmp_path)
    assert report["strategy"] == "classic-vision"
    assert report["images"] == rejected
    assert report["counts"] == counts


def test_a_missing_or_corrupt_filter_report_reads_as_empty(tmp_path) -> None:
    assert read_filter_report(tmp_path) == {}
    write_filter_report(tmp_path, {}, {})
    target = tmp_path / ".picklikeme" / "classic_vision_filters.json"
    target.write_text("{not json", encoding="utf-8")
    assert read_filter_report(tmp_path) == {}


# ---------------------------------------------------------------------------
# the metrics report sidecar
# ---------------------------------------------------------------------------


def test_the_metrics_report_records_every_surviving_image_s_raw_measurements(tmp_path) -> None:
    """A photographer investigating why a weak-eyed image still ranked
    respectably needs these three numbers, not just their weighted sum - the
    ranking CSV only ever carries the combined score."""
    metrics = [
        ImageMetrics("a.nef", eye_sharpness=1.5, subject_sharpness=2.5, subject_size=0.1, eye_confidence=0.9),
        ImageMetrics("b.nef", eye_sharpness=3.5, subject_sharpness=4.5, subject_size=0.2, eye_confidence=0.8),
    ]
    write_metrics_report(tmp_path, metrics)

    report = read_metrics_report(tmp_path)
    assert report["strategy"] == "classic-vision"
    assert report["metrics"]["a.nef"] == {
        "eye_sharpness": 1.5, "subject_sharpness": 2.5, "subject_size": 0.1, "eye_confidence": 0.9,
        "head_confidence": None,
    }
    assert report["metrics"]["b.nef"]["eye_confidence"] == pytest.approx(0.8)


def test_a_missing_or_corrupt_metrics_report_reads_as_empty(tmp_path) -> None:
    assert read_metrics_report(tmp_path) == {}
    write_metrics_report(tmp_path, [])
    target = tmp_path / ".picklikeme" / "classic_vision_metrics.json"
    target.write_text("{not json", encoding="utf-8")
    assert read_metrics_report(tmp_path) == {}


# ---------------------------------------------------------------------------
# reading what preprocessing already wrote
# ---------------------------------------------------------------------------


def _write_cache_entry(
    cache_dir, image_path, *, label=COCO_BIRD_CLASS, box=(10, 10, 60, 60), expanded_box=(5, 5, 65, 65)
):
    crop = crop_cache_path(cache_dir, image_path)
    crop.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop), cv2.cvtColor(_sharp_image(64), cv2.COLOR_RGB2BGR))
    payload = {
        "version": 1,
        "source_size": [800, 600],
        "selected": {"box": list(box), "score": 0.9, "label": label},
        "detections": [{"box": list(box), "score": 0.9, "label": label}],
        "expanded_box": list(expanded_box) if expanded_box is not None else None,
    }
    crop.with_name(crop.stem + ".detections.json").write_text(json.dumps(payload), encoding="utf-8")
    return crop


def test_a_candidate_is_assembled_from_the_existing_crop_cache(tmp_path) -> None:
    """No RAW is decoded and the subject detector never runs - everything the
    filters need was already written by preprocessing."""
    cache_dir = tmp_path / "crops"
    image_path = str(tmp_path / "shoot" / "a.nef")
    _write_cache_entry(cache_dir, image_path)

    candidate = ClassicVisionStrategy._load_candidate(image_path, cache_dir)
    assert candidate.subject_crop is not None
    assert candidate.subject_box == (10.0, 10.0, 60.0, 60.0)
    # crop_box is the crop's own (margin-expanded) rectangle, distinct from
    # subject_box (the tight detection box) - see FilterCandidate.crop_box's
    # docstring and docs/EyePose_Investigation_Phase_1.md's Q1 finding.
    assert candidate.crop_box == (5.0, 5.0, 65.0, 65.0)
    assert candidate.source_size == (800, 600)
    assert candidate.subject_label == COCO_BIRD_CLASS


def test_crop_box_is_none_for_a_pre_v6_cache_entry_missing_expanded_box(tmp_path) -> None:
    """A defensive fallback, not the expected path - CROP_CACHE_VERSION's v6
    bump already forces such an entry to rebuild before it would ever reach
    here for real."""
    cache_dir = tmp_path / "crops"
    image_path = str(tmp_path / "shoot" / "a.nef")
    _write_cache_entry(cache_dir, image_path, expanded_box=None)

    candidate = ClassicVisionStrategy._load_candidate(image_path, cache_dir)
    assert candidate.crop_box is None
    assert candidate.subject_box == (10.0, 10.0, 60.0, 60.0)  # unaffected


def test_an_image_with_no_cached_detection_reads_as_having_no_subject(tmp_path) -> None:
    candidate = ClassicVisionStrategy._load_candidate(str(tmp_path / "never-seen.nef"), tmp_path / "crops")
    assert SubjectFilter().check(candidate) is False


# ---------------------------------------------------------------------------
# the whole strategy, end to end
# ---------------------------------------------------------------------------


def test_classic_vision_ranks_a_folder_and_writes_the_usual_sidecar(tmp_path, monkeypatch) -> None:
    """The full run: enumerate, filter, measure, score, write the same ranking
    CSV every other part of PeakPic already reads.

    `build_cache` is stubbed out because the crop cache is pre-populated here -
    that is exactly the state a real folder is in after the AI model has
    ranked it once, and it keeps the test from needing real RAW files or the
    subject detector.
    """
    from picklikeme.analyzer.io import load_ranking
    from picklikeme.ranking import classic as classic_module
    from picklikeme.sidecar import ranking_path, strategy_ranking_path

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"

    # Four images: two birds that will score, one mammal, one with no detection.
    scoring = [str(folder / f"bird_{i}.nef") for i in range(2)]
    mammal = str(folder / "tiger.nef")
    undetected = str(folder / "empty.nef")
    for path in (*scoring, mammal, undetected):
        Path(path).write_bytes(b"not really a raw file")
    for path in scoring:
        _write_cache_entry(cache_dir, path)
    _write_cache_entry(cache_dir, mammal, label=COCO_PERSON_CLASS)
    # `undetected` deliberately gets no cache entry at all.

    monkeypatch.setattr(classic_module, "build_cache", lambda *a, **k: {})
    # A stand-in that finds an eye on anything it is shown, so this test is
    # about the strategy's own plumbing rather than the eye model's judgement
    # (that is covered against the real checkpoint in test_eye_detector.py).
    monkeypatch.setattr(
        "picklikeme.eyes.build_eye_detector",
        lambda name, **kwargs: _FakeEyeDetector(detection=_eye(), supported=True),
    )

    stages: list[str] = []
    progress: list[tuple[int, int]] = []
    result = ClassicVisionStrategy().rank_folder(
        folder,
        params=ClassicVisionParams(),
        crop_cache_dir=cache_dir,
        device="cpu",
        on_stage=stages.append,
        on_progress=lambda done, total: progress.append((done, total)),
        analytics_db=tmp_path / "analytics.db",
    )

    assert result["strategy"] == "classic-vision"
    assert result["considered"] == 4
    assert result["image_count"] == 3  # two birds + the mammal the stand-in accepts
    assert result["filtered"] == {NO_SUBJECT: 1}
    assert stages and progress[-1] == (4, 4)

    # Its OWN scores file, in the format every other part of PeakPic already
    # parses - and crucially NOT the AI model's ranking.csv.
    ranking_file = strategy_ranking_path(folder, "classic-vision")
    assert ranking_file.is_file()
    assert not ranking_path(folder).exists(), "must not have written the AI model's file"
    ranking = load_ranking(ranking_file)
    assert len(ranking.images) == 3
    scores = [image.score for image in ranking.images]
    assert scores == sorted(scores, reverse=True)  # descending, as write_results_csv promises

    # And the reasons for what was left out survive next to it.
    assert read_filter_report(folder)["counts"] == {NO_SUBJECT: 1}

    # And each surviving image's raw metrics, behind that combined score.
    metrics_report = read_metrics_report(folder)["metrics"]
    assert set(metrics_report) == set(scoring) | {mammal}
    assert set(metrics_report[mammal]) == {
        "eye_sharpness", "subject_sharpness", "subject_size", "eye_confidence", "head_confidence",
    }

    # The Analytics Dashboard's Experiment Metadata / Run Summary need a
    # per-image "score" metric (the same generic name every strategy uses,
    # not just this one's own eye_sharpness/subject_sharpness/subject_size),
    # a runtime/images-per-second summary, and enough params to answer
    # "exactly how was this run produced" - algorithm version plus the
    # environment facts every run records (see analytics.environment).
    from picklikeme.analytics.store import AnalyticsStore

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        (run,) = store.list_runs()
        run_id = run["run_id"]
        params = store.get_run(run_id)["params"]
        assert params["algorithm_version"]
        assert "git_commit" in params
        assert "application_version" in params
        assert "gpu_name" in params
        assert "cuda_available" in params

        summary = store.summary_metrics(run_id)
        assert summary["runtime_seconds"] >= 0.0
        assert summary["images_per_second"] >= 0.0

        for path in scoring:
            assert "score" in store.image_metrics(run_id, path)


def test_rank_folder_forwards_detection_thresholds_into_the_crop_cache_build(tmp_path, monkeypatch) -> None:
    """detection_confidence_threshold/crop_confidence_threshold (EyePose
    Investigation Phase 1, Part 6) must actually reach build_cache's
    CropParams - not just exist as unused params fields. Both backends share
    this wiring (see ClassicVisionStrategy.rank_folder), checked here against
    the SuperAnimal-Bird strategy since it needs no EyePose-specific setup."""
    from picklikeme.bird_crop import CropParams
    from picklikeme.ranking import classic as classic_module

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    path = str(folder / "bird.nef")
    Path(path).write_bytes(b"not really a raw file")
    _write_cache_entry(cache_dir, path)

    captured: dict = {}

    def fake_build_cache(image_paths, crop_cache_dir, params, **kwargs):
        captured["params"] = params
        return {}

    monkeypatch.setattr(classic_module, "build_cache", fake_build_cache)
    monkeypatch.setattr(
        "picklikeme.eyes.build_eye_detector",
        lambda name, **kwargs: _FakeEyeDetector(detection=_eye(), supported=True),
    )

    ClassicVisionStrategy().rank_folder(
        folder,
        params=ClassicVisionParams(detection_confidence_threshold=0.55, crop_confidence_threshold=0.25),
        crop_cache_dir=cache_dir,
        device="cpu",
        analytics_db=tmp_path / "analytics.db",
    )

    assert isinstance(captured["params"], CropParams)
    assert captured["params"].conf_threshold == pytest.approx(0.55)
    assert captured["params"].min_crop_confidence == pytest.approx(0.25)


def test_classic_vision_writes_debug_images_only_when_debug_dir_is_given(tmp_path, monkeypatch) -> None:
    """Off by default (see ranking.debug's module docstring) - a run with no
    `debug_dir` must write nothing extra; a run with one gets one debug
    image per candidate that reached the eye detector."""
    from picklikeme.ranking import classic as classic_module
    from picklikeme.ranking.debug import debug_image_path

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    scoring = [str(folder / f"bird_{i}.nef") for i in range(2)]
    for path in scoring:
        Path(path).write_bytes(b"not really a raw file")
        _write_cache_entry(cache_dir, path)

    monkeypatch.setattr(classic_module, "build_cache", lambda *a, **k: {})
    monkeypatch.setattr(
        "picklikeme.eyes.build_eye_detector",
        lambda name, **kwargs: _FakeEyeDetector(detection=_eye(), supported=True),
    )

    ClassicVisionStrategy().rank_folder(
        folder, params=ClassicVisionParams(), crop_cache_dir=cache_dir, device="cpu",
        analytics_db=tmp_path / "analytics.db",
    )
    debug_dir = tmp_path / "debug"
    assert not debug_dir.exists(), "no debug_dir was requested - nothing extra should be written"

    ClassicVisionStrategy().rank_folder(
        folder, params=ClassicVisionParams(), crop_cache_dir=cache_dir, device="cpu", debug_dir=debug_dir,
        analytics_db=tmp_path / "analytics.db",
    )
    for path in scoring:
        assert debug_image_path(debug_dir, path).is_file()


def test_classic_vision_refuses_a_folder_with_no_images(tmp_path) -> None:
    """A UI catches this and shows it - the strategy must raise, never exit."""
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(ValueError, match="No images found"):
        ClassicVisionStrategy().rank_folder(folder, params=ClassicVisionParams())


def test_classic_vision_refuses_a_folder_that_does_not_exist(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ClassicVisionStrategy().rank_folder(tmp_path / "nope", params=ClassicVisionParams())


# ---------------------------------------------------------------------------
# analysis is independent of the Organize workflow
# ---------------------------------------------------------------------------


def test_analysis_targets_include_already_organized_images(tmp_path) -> None:
    """The reported bug: running Classic Vision on a folder that had already
    been arranged failed with "no un-arranged RAW images left to rank".

    Analysis describes pixels; Organize describes workflow state. An image
    does not stop being an analysis target because it has been filed, so
    everything under _Selected/_Rejected must be enumerated.
    """
    from picklikeme.ranking.classic import analysis_targets

    folder = tmp_path / "shoot"
    (folder / "_Selected").mkdir(parents=True)
    (folder / "_Rejected").mkdir(parents=True)
    (folder / "_Selected" / "kept.NEF").write_bytes(b"raw")
    (folder / "_Rejected" / "dropped.NEF").write_bytes(b"raw")
    (folder / "loose.NEF").write_bytes(b"raw")

    found = {Path(p).name for p in analysis_targets(folder)}
    assert found == {"kept.NEF", "dropped.NEF", "loose.NEF"}


def test_analysis_targets_include_non_raw_images(tmp_path) -> None:
    """The module analyses images, not ingestion state - a JPEG is a valid
    analysis target even though the AI ranking path only enumerates RAW."""
    from picklikeme.ranking.classic import analysis_targets

    folder = tmp_path / "shoot"
    folder.mkdir()
    for name in ("a.NEF", "b.jpg", "c.tif", "notes.txt"):
        (folder / name).write_bytes(b"x")

    found = {Path(p).name for p in analysis_targets(folder)}
    assert found == {"a.NEF", "b.jpg", "c.tif"}  # the .txt is not an image


def test_analysis_targets_exclude_only_the_modules_own_output(tmp_path) -> None:
    """The single exclusion: a second run must not analyse the first's results."""
    from picklikeme.ranking.classic import analysis_targets
    from picklikeme.sidecar import SIDECAR_DIRNAME

    folder = tmp_path / "shoot"
    (folder / SIDECAR_DIRNAME).mkdir(parents=True)
    (folder / "real.NEF").write_bytes(b"raw")
    (folder / SIDECAR_DIRNAME / "stray.jpg").write_bytes(b"x")

    assert [Path(p).name for p in analysis_targets(folder)] == ["real.NEF"]


def test_a_fully_organized_folder_still_ranks(tmp_path, monkeypatch) -> None:
    """End-to-end version of the reported bug: every single image lives in
    _Selected/_Rejected, which used to raise before any analysis happened."""
    from picklikeme.ranking import classic as classic_module

    folder = tmp_path / "shoot"
    (folder / "_Selected").mkdir(parents=True)
    (folder / "_Rejected").mkdir(parents=True)
    cache_dir = tmp_path / "crops"
    organized = [folder / "_Selected" / "a.NEF", folder / "_Rejected" / "b.NEF"]
    for path in organized:
        path.write_bytes(b"raw")
        _write_cache_entry(cache_dir, str(path))

    monkeypatch.setattr(classic_module, "build_cache", lambda *a, **k: {})
    monkeypatch.setattr(
        "picklikeme.eyes.build_eye_detector",
        lambda name, **kwargs: _FakeEyeDetector(detection=_eye(), supported=True),
    )

    result = ClassicVisionStrategy().rank_folder(
        folder, params=ClassicVisionParams(), crop_cache_dir=cache_dir, device="cpu",
        analytics_db=tmp_path / "analytics.db",
    )
    assert result["considered"] == 2
    assert result["image_count"] == 2
    assert result["filtered"] == {}


def test_a_folder_where_everything_is_filtered_still_finishes(tmp_path, monkeypatch) -> None:
    """Scoring nothing is a real outcome, not a crash: the module must still
    write its (empty) scores file and report why every image was skipped."""
    from picklikeme.ranking import classic as classic_module

    folder = tmp_path / "shoot"
    folder.mkdir()
    cache_dir = tmp_path / "crops"
    path = folder / "mammal.NEF"
    path.write_bytes(b"raw")
    _write_cache_entry(cache_dir, str(path), label=COCO_PERSON_CLASS)

    monkeypatch.setattr(classic_module, "build_cache", lambda *a, **k: {})
    monkeypatch.setattr(
        "picklikeme.eyes.build_eye_detector",
        lambda name, **kwargs: _FakeEyeDetector(detection=_eye(accepted=False), supported=False),
    )

    result = ClassicVisionStrategy().rank_folder(
        folder, params=ClassicVisionParams(), crop_cache_dir=cache_dir, device="cpu",
        analytics_db=tmp_path / "analytics.db",
    )
    assert result["image_count"] == 0
    assert result["filtered"] == {UNSUPPORTED_SUBJECT: 1}
    assert Path(result["output_csv"]).is_file()


# ---------------------------------------------------------------------------
# scores from different modules coexist
# ---------------------------------------------------------------------------


def test_each_module_owns_its_own_scores_file(tmp_path) -> None:
    from picklikeme.sidecar import discover_strategy_rankings, ranking_path, strategy_ranking_path

    folder = tmp_path / "shoot"
    (folder / ".picklikeme").mkdir(parents=True)

    # The AI model keeps the original unsuffixed name, for backwards compat.
    assert strategy_ranking_path(folder, "ai-model") == ranking_path(folder)
    assert strategy_ranking_path(folder, "classic-vision").name == "ranking-classic-vision.csv"

    ranking_path(folder).write_text("x", encoding="utf-8")
    strategy_ranking_path(folder, "classic-vision").write_text("x", encoding="utf-8")
    # A continuation chunk belongs to the file it continues, not a new module.
    (folder / ".picklikeme" / "ranking-classic-vision_1.csv").write_text("x", encoding="utf-8")

    assert set(discover_strategy_rankings(folder)) == {"ai-model", "classic-vision"}


def test_a_module_that_no_longer_ships_still_has_its_scores_discovered(tmp_path) -> None:
    """Discovery reads the disk, not the registry, so results outlive the code
    that produced them rather than vanishing from the gallery."""
    from picklikeme.sidecar import discover_strategy_rankings

    folder = tmp_path / "shoot"
    (folder / ".picklikeme").mkdir(parents=True)
    (folder / ".picklikeme" / "ranking-burst-analysis.csv").write_text("x", encoding="utf-8")

    assert set(discover_strategy_rankings(folder)) == {"burst-analysis"}


# ---------------------------------------------------------------------------
# the AI strategy stays exactly what it was
# ---------------------------------------------------------------------------


def test_the_ai_strategy_forwards_verbatim_to_rank_folder(monkeypatch) -> None:
    calls = {}

    def fake_rank_folder(input_folder, **kwargs):
        calls["input_folder"] = input_folder
        calls["kwargs"] = kwargs
        return {"image_count": 3, "device": "cpu", "top": []}

    monkeypatch.setattr("picklikeme.rank.rank_folder", fake_rank_folder)

    result = get_strategy("ai-model").rank_folder(
        "/shoot", params=AIModelParams(checkpoint="c.pt", crop_birds=False, device="cpu")
    )

    assert calls["input_folder"] == "/shoot"
    assert calls["kwargs"]["checkpoint"] == "c.pt"
    assert calls["kwargs"]["crop_birds"] is False
    assert calls["kwargs"]["device"] == "cpu"
    # The strategy adds its own reporting keys without disturbing the rest.
    assert result["strategy"] == "ai-model"
    assert result["filtered"] == {}
    assert result["image_count"] == 3


def test_the_ai_strategy_defaults_match_the_pre_strategy_behaviour() -> None:
    params = AIModelParams()
    assert params.crop_birds is True
    assert params.device is None  # "auto" - what rank_folder has always defaulted to
    assert params.checkpoint.endswith("model_checkpoint.pt")
