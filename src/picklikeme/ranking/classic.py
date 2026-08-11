"""Classic Vision Ranking - a deterministic alternative to the trained model.

No learning, no checkpoint, no preference model: two filters decide whether
an image is judged at all, and three measurements decide how it scores. Run
it twice on the same folder with the same parameters and it produces
byte-identical numbers.

    Phase 1 - filtering (ranking.filters) - the "Decision Engine" (see
    docs/EyePose_Investigation_Phase_1.md's Part 3)
        Filter 1  no detected subject          -> NO_SUBJECT
        Filter 2  no eye detector for this class -> UNSUPPORTED_SUBJECT
                  no confident head detected     -> LOW_HEAD_CONFIDENCE
                  no reliable visible eye        -> NO_VISIBLE_EYE
                  (all three from EyeFilter - one detector call, three
                  independent questions, each its own reason)

    Phase 2 - scoring (ranking.metrics), for survivors only
        Eye sharpness      focus inside the eye box alone      (default 50%)
        Subject sharpness  focus across the whole subject crop (default 30%)
        Subject size       subject box area / full frame area  (default 20%)

The two phases never reach into each other: no filter computes a metric, and
no metric decides membership. `EyeFilter` does hand the eye it found to the
scoring phase through the candidate, but that is caching a result, not
sharing a decision - the eye box is needed by both, and detecting it twice
would double the only expensive step in the run.

**This is an analysis module, not a workflow step.** It reads pixels and
writes scores. It never moves a file, never consults whether a folder has been
organized, and never refuses to run because of workflow state. Concretely, it
scores a folder that has never been organized, one already arranged into
`_Selected`/`_Rejected`, one previously ranked by the AI model, one previously
ranked by itself, and one never ranked at all - and in the organized case it
scores the images *inside* `_Selected` and `_Rejected` too, because an image
does not stop being an image once it has been filed. See `analysis_targets`.

**Everything expensive is already built.** `preprocess.build_cache` is the
same call `rank.rank_folder` makes: it decodes each RAW once, runs the
subject detector once, writes the crop, and records the detection beside it.
This strategy then reads those cached crops and detection records - so on a
folder that has already been ranked by the AI model, Classic Vision adds only
the eye-detection pass (~18 images/second on this machine's GPU) and some
cheap OpenCV arithmetic. Nothing here re-decodes a RAW or re-runs the subject
detector.

**Filtered images are absent from the ranking, not scored zero.** A zero
would be a judgement ("this photograph is bad"), and the filters make no such
claim - they say the algorithm has nothing to measure. `ReviewSession` already
builds its gallery from the union of the ranking and the folder, so an
excluded image appears in the review UI as Unranked and completely Neutral,
which is the honest presentation. The per-reason tally, and the per-image
reasons, are written to the sidecar (see `write_filter_report`) so the
photographer can still find out why.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..analytics import DEFAULT_ANALYTICS_DB, record_run
from ..analytics.environment import resolve_environment_info
from ..auto_crop import resolve_device
from ..bird_crop import CropParams, crop_cache_path, read_detections
from ..config import DEFAULT_CROP_CACHE_DIR, DEFAULT_MAX_CSV_ROWS
from ..dataset import UnlabeledImageDataset
from ..eyes.domains import BIRDS_PROFILE, MAMMALS_PROFILE
from ..eyes.eyepose_v0 import (
    DEFAULT_MAX_EYE_HEAD_DISTANCE_RATIO as EYEPOSE_DEFAULT_MAX_HEAD_DISTANCE_RATIO,
)
from ..eyes.eyepose_v0 import DEFAULT_MIN_CONFIDENCE as EYEPOSE_DEFAULT_MIN_CONFIDENCE
from ..eyes.eyepose_v0 import DEFAULT_MIN_HEAD_CONFIDENCE as EYEPOSE_DEFAULT_MIN_HEAD_CONFIDENCE
from ..eyes.fusion import DEFAULT_AGREEMENT_THRESHOLD, DEFAULT_MIN_FUSED_CONFIDENCE, FusionConfig, ModelWeight
from ..eyes.superanimal_bird import DEFAULT_MAX_EYE_DISAGREEMENT, DEFAULT_MIN_CONFIDENCE
from ..preprocess import build_cache
from ..sidecar import (
    SIDECAR_DIRNAME,
    ensure_sidecar_dir,
    strategy_ranking_path,
    write_run_metadata,
)
from .base import GROUP_THRESHOLDS, GROUP_WEIGHTS, ParamSpec, StrategyInfo, WeightedParams, use_subject_filter_spec
from .filters import EyeFilter, FilterCandidate, FilterChain, SubjectFilter
from .metrics import (
    normalized_subject_size,
    region_focus_measure,
    robust_normalize,
    subject_focus_measure,
)

logger = logging.getLogger(__name__)

STRATEGY_ID = "classic-vision"

# Bumped whenever this module's own filtering/scoring logic changes in a way
# that could change a result - independent of the eye-detector backend's own
# identity (see `_eye_detector_name`) and of open_clip/torch versions, the
# same "which exact axis changed" discipline `BioClipSpeciesClassifier.
# CLASSIFIER_VERSION` already applies to species classification. Shown in
# the Analytics Dashboard's Experiment Metadata as "Algorithm Version".
ALGORITHM_VERSION = "1"

# Where the per-image filter verdicts land, beside the ranking CSV the same
# run produced. Not merged into run.json (which is provenance about the run as
# a whole) because this is per-image data that grows with the folder.
#
# These two are the ORIGINAL, un-suffixed names - kept exactly as they were
# before Classic Vision supported more than one eye-detection backend, so a
# folder already analysed by the legacy (SuperAnimal-Bird) strategy keeps
# reading and writing the exact same files. A folder analysed by ANY other
# backend gets its own, backend-specific filenames instead - see
# `_filter_report_filename`/`_metrics_report_filename` - so two backends run
# on the same folder coexist rather than overwriting each other, the same
# requirement `sidecar.strategy_ranking_path` already satisfies for the
# ranking CSV itself.
FILTER_REPORT_FILENAME = "classic_vision_filters.json"

# Where each surviving image's raw, per-metric measurements land - the
# subject-size/eye-sharpness/subject-sharpness breakdown behind the single
# combined score in the ranking CSV. A photographer investigating why a
# weak-eyed image still ranked respectably (or a strong-eyed one ranked low)
# needs these three numbers, not just their weighted sum. Self-describing
# (carries its own "strategy" id), so a generic reader - the Loupe's
# diagnostics line - needs no per-strategy code; see
# sidecar.discover_metric_reports.
METRICS_REPORT_FILENAME = "classic_vision_metrics.json"


def _filter_report_filename(strategy_id: str) -> str:
    """Where one backend's filter report lives - the legacy name for the
    original (SuperAnimal-Bird) strategy id, `<strategy_id>_filters.json`
    for any other, so a second Classic Vision backend never overwrites the
    first's results on a folder analysed by both."""
    return FILTER_REPORT_FILENAME if strategy_id == STRATEGY_ID else f"{strategy_id}_filters.json"


def _metrics_report_filename(strategy_id: str) -> str:
    return METRICS_REPORT_FILENAME if strategy_id == STRATEGY_ID else f"{strategy_id}_metrics.json"

# Display labels for the raw metrics above, for the same generic reader.
METRIC_LABELS: dict[str, str] = {
    "eye_sharpness": "Eye sharpness",
    "subject_sharpness": "Subject sharpness",
    "subject_size": "Subject size",
    "eye_confidence": "Eye confidence",
}


def analysis_targets(input_folder: str | Path) -> list[str]:
    """Every image in a folder that this analysis module should score.

    "Every image" is meant literally, and the exclusions are the whole point
    of this function existing rather than reusing `rank.rank_folder`'s
    enumeration:

    - **`_Selected`/`_Rejected` are included.** An image does not stop being
      an image because the photographer already filed it. Analysis describes
      pixels; Organize describes workflow state, and an analysis module must
      never be able to fail - or silently skip a photograph - because of what
      Organize did earlier. A shoot that has been arranged is still a
      perfectly good thing to score, and re-scoring one must produce results
      for all of it.
    - **JPEGs and TIFFs are included, not only RAW.** Same reasoning: the
      module analyses images, not ingestion state. `enumerate_ground_truth`
      already defines "an image PeakPic can open" (PREVIEWABLE_EXTENSIONS)
      and is what the review gallery itself enumerates with, so using it here
      means the gallery and this module always agree on what exists.
    - **`.picklikeme/` is excluded**, because it holds this module's own
      output. That is the single exclusion, and it exists so a second run
      cannot analyse the results of the first.

    Sorted, so a run is deterministic in the order it processes and reports.
    """
    from ..analyzer.io import enumerate_ground_truth

    input_folder = Path(input_folder)
    found = enumerate_ground_truth(input_folder)
    return [str(path) for path in found if SIDECAR_DIRNAME not in path.parts]


# The three scoring weights are identical in name and meaning across every
# Classic Vision backend - scoring (ranking.metrics) never knows which
# backend produced the eye it measures inside, so there is nothing
# backend-specific about them. Shared here so both params dataclasses below
# declare the exact same three specs rather than two copies that could drift.
def _scoring_weight_specs() -> tuple[ParamSpec, ...]:
    return (
        ParamSpec(
            name="eye_sharpness_weight",
            label="Eye sharpness",
            default=70.0,
            minimum=0.0,
            maximum=1000.0,
            group=GROUP_WEIGHTS,
            help="How sharp the eye itself is, measured inside the eye box only.",
        ),
        ParamSpec(
            name="subject_sharpness_weight",
            label="Subject sharpness",
            default=10.0,
            minimum=0.0,
            maximum=1000.0,
            group=GROUP_WEIGHTS,
            help="How sharp the whole detected subject is.",
        ),
        ParamSpec(
            name="subject_size_weight",
            label="Subject size",
            default=20.0,
            minimum=0.0,
            maximum=1000.0,
            group=GROUP_WEIGHTS,
            help="How much of the frame the subject fills.",
        ),
    )


def _detection_specs() -> tuple[ParamSpec, ...]:
    """The two subject-detection/crop-selection tunables every Classic Vision
    backend shares (see `bird_crop.select_best_detection`'s "v7" policy) -
    factored out exactly like `_scoring_weight_specs()`, so both params
    classes declare them identically rather than risking drift. Generic by
    construction: neither parameter is bird-specific, and both apply the
    same way to any current or future catalogued class - see
    `bird_crop.py`'s module docstring.

    Two genuinely different confidence concepts, not one - both map onto
    `bird_crop.CropParams` fields but answer different questions:
    `detection_confidence_threshold` (-> `CropParams.conf_threshold`) is the
    detector's own, low, "worth recording at all" bar (cataloguing);
    `crop_confidence_threshold` (-> `CropParams.min_crop_confidence`) is the
    higher, "reliable enough to crop to" bar `select_best_detection` actually
    gates on (EyePose Investigation Phase 1, Part 1).
    """
    return (
        ParamSpec(
            name="detection_confidence_threshold",
            label="Detection confidence threshold",
            default=CropParams.conf_threshold,
            minimum=0.0,
            maximum=1.0,
            group=GROUP_THRESHOLDS,
            decimals=2,
            help=(
                "Subject detections below this confidence are ignored entirely before "
                "crop-target selection even runs. Affects the Vision Cache: changing this "
                "rebuilds crops, since a different threshold can select a different subject."
            ),
        ),
        ParamSpec(
            name="crop_confidence_threshold",
            label="Crop-target confidence threshold",
            default=CropParams.min_crop_confidence,
            minimum=0.0,
            maximum=1.0,
            group=GROUP_THRESHOLDS,
            decimals=2,
            help=(
                "Among catalogued detections, only those reaching this confidence are "
                "eligible to become the crop target; the largest-area one among them wins. "
                "Rejects an unreliable detection outright rather than letting a big, "
                "low-confidence box win by size alone (see the EyePose Investigation Phase 1 "
                "report's Q1 - a real bird lost the crop to a much lower-confidence false "
                "positive under the previous, area-first policy)."
            ),
        ),
    )


@dataclass(frozen=True)
class ClassicVisionParams(WeightedParams):
    """Everything the photographer can tune before a Classic Vision
    (SuperAnimal-Bird) run.

    Adding a parameter later is: one field here, one `ParamSpec` in `specs()`.
    The dialog builds itself from `specs()` (see the desktop
    `AlgorithmParametersDialog`), the weights normalise themselves, and no
    other code changes. See `ClassicVisionEyePoseParams` for the EyePose-v0
    backend's own params - the two gates below (confidence, eye-channel
    disagreement) are specific to how SuperAnimal-Bird predicts an eye and do
    not carry over to a different backend's landmark schema unchanged, so
    each backend declares its own tunables rather than sharing one shape that
    would fit neither well - see `ranking.classic`'s module docstring and
    `eyes.eyepose_v0`'s "Accept/reject gate" for why. `detection_confidence_threshold`/
    `crop_confidence_threshold`, by contrast, are shared verbatim (`_detection_specs`)
    - they configure `rank_folder`'s own crop-cache step, not either backend's
    eye detector, so both backends mean exactly the same thing by them.
    """

    eye_sharpness_weight: float = 70.0
    subject_sharpness_weight: float = 10.0
    subject_size_weight: float = 20.0
    min_eye_confidence: float = DEFAULT_MIN_CONFIDENCE
    max_eye_disagreement: float = DEFAULT_MAX_EYE_DISAGREEMENT
    detection_confidence_threshold: float = CropParams.conf_threshold
    crop_confidence_threshold: float = CropParams.min_crop_confidence

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        return (
            *_scoring_weight_specs(),
            ParamSpec(
                name="min_eye_confidence",
                label="Minimum eye confidence",
                default=DEFAULT_MIN_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
                group=GROUP_THRESHOLDS,
                decimals=2,
                help="Below this, the eye counts as not visible and the image is filtered out.",
            ),
            ParamSpec(
                name="max_eye_disagreement",
                label="Max eye disagreement",
                default=DEFAULT_MAX_EYE_DISAGREEMENT,
                minimum=0.0,
                maximum=10.0,
                group=GROUP_THRESHOLDS,
                decimals=2,
                help=(
                    "How much the two independently-predicted eye positions may disagree "
                    "(relative to head size) before the eye is distrusted, even at high "
                    "confidence - catches a confidently-guessed eye on an occluded head."
                ),
            ),
            *_detection_specs(),
            use_subject_filter_spec(),
        )


@dataclass(frozen=True)
class ClassicVisionEyePoseParams(WeightedParams):
    """Everything the photographer can tune before a Classic Vision
    (EyePose-v0) run - see `ClassicVisionParams` for why this is a separate
    dataclass rather than a shared one: the two backends' accept/reject
    gates are genuinely different (see `eyes.eyepose_v0`'s "Accept/reject
    gate"), so their tunable thresholds are too. The three scoring weights
    are identical - see `_scoring_weight_specs`.

    `eye_confidence_threshold` (named to match the EyePose Investigation
    Phase 1 report exactly - see docs/EyePose_Investigation_Phase_1.md's
    Part 3): EyePose-v0 reports a confidence for both the left and right eye
    channel independently; `detect()` already keeps only the higher-confidence
    one as `EyeDetection`'s primary eye/box (the weaker channel is recorded
    for debugging - see `eyes.cache`'s module docstring - but never used for
    scoring). This threshold decides whether that best channel is trusted at
    all: below it, `EyeFilter` rejects the image as NO_VISIBLE_EYE
    ("No reliable visible eye" - see `ranking.filters.REJECT_REASON_LABELS`)
    and it never reaches `measure()`.

    `detection_head_confidence_threshold` is a genuinely independent, earlier
    check (Part 2/3 of the same report): is a real head instance in the crop
    at all, before any individual landmark - including the eye - is trusted?
    Gates `EyeFilter`'s own `LOW_HEAD_CONFIDENCE` rejection, gated on
    `EyeDetection.head_confidence`/`eyes.eyepose_v0.head_visible` - see that
    function's docstring for the measured signal (a wrong crop with no real
    head scored 0.026 there, against 0.82-0.92 for every real head, despite
    its own eye landmark reporting a misleading 0.97). Part 4's first pass
    (weighting in the `head_top` *landmark's* own confidence) found no
    independent signal - see `eyes.eyepose_v0.accepts_eye`'s docstring; this
    is a different, later finding that does carry one.
    """

    eye_sharpness_weight: float = 70.0
    subject_sharpness_weight: float = 10.0
    subject_size_weight: float = 20.0
    eye_confidence_threshold: float = EYEPOSE_DEFAULT_MIN_CONFIDENCE
    max_head_distance_ratio: float = EYEPOSE_DEFAULT_MAX_HEAD_DISTANCE_RATIO
    detection_head_confidence_threshold: float = EYEPOSE_DEFAULT_MIN_HEAD_CONFIDENCE
    detection_confidence_threshold: float = CropParams.conf_threshold
    crop_confidence_threshold: float = CropParams.min_crop_confidence

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        return (
            *_scoring_weight_specs(),
            ParamSpec(
                name="eye_confidence_threshold",
                label="Minimum eye confidence",
                default=EYEPOSE_DEFAULT_MIN_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
                group=GROUP_THRESHOLDS,
                decimals=2,
                help="Below this, the eye counts as not visible and the image is filtered out.",
            ),
            ParamSpec(
                name="max_head_distance_ratio",
                label="Max eye/head distance",
                default=EYEPOSE_DEFAULT_MAX_HEAD_DISTANCE_RATIO,
                minimum=0.0,
                maximum=10.0,
                group=GROUP_THRESHOLDS,
                decimals=2,
                help=(
                    "How far the eye may sit from the beak<->head-top line, relative to "
                    "head size, before it is distrusted - catches a keypoint that landed "
                    "on a shoulder or the background rather than the head."
                ),
            ),
            ParamSpec(
                name="detection_head_confidence_threshold",
                label="Minimum head confidence",
                default=EYEPOSE_DEFAULT_MIN_HEAD_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
                group=GROUP_THRESHOLDS,
                decimals=2,
                help=(
                    "Below this, EyePose-v0 was not confident a real bird head was even in "
                    "the crop, independent of any single landmark's own confidence (including "
                    "the eye's) - the image is filtered out as LOW_HEAD_CONFIDENCE before the "
                    "eye-specific checks above are even consulted."
                ),
            ),
            *_detection_specs(),
            use_subject_filter_spec(),
        )


@dataclass
class ImageMetrics:
    """The raw, un-normalised measurements for one surviving image.

    Raw on purpose: normalisation is folder-relative (see
    `metrics.robust_normalize`), so it cannot happen until every image has
    been measured. Keeping the per-image step pure - pixels in, three numbers
    out - is also what makes it directly testable without a folder.
    """

    image_path: str
    eye_sharpness: float
    subject_sharpness: float
    subject_size: float
    eye_confidence: float
    # The eye detector's own head-visibility confidence (see eyes.detector.
    # EyeDetection.head_confidence) - independent of eye_confidence, which is
    # about the visible eye specifically. None for a detector backend that
    # never computes this (EyeDetection's own default), never fabricated.
    head_confidence: float | None = None


def measure(candidate: FilterCandidate) -> ImageMetrics:
    """The three metrics for one image that passed every filter.

    A pure function of the candidate: no I/O, no model, no folder context. The
    candidate is guaranteed to carry a crop, a subject box, a frame size and an
    eye by the time it gets here - `FilterChain` rejected anything that did
    not.
    """
    assert candidate.subject_crop is not None  # noqa: S101 - guaranteed by FilterChain
    assert candidate.subject_box is not None  # noqa: S101
    assert candidate.eye is not None  # noqa: S101

    return ImageMetrics(
        image_path=candidate.image_path,
        eye_sharpness=region_focus_measure(candidate.subject_crop, candidate.eye.box),
        subject_sharpness=subject_focus_measure(candidate.subject_crop),
        subject_size=normalized_subject_size(
            candidate.subject_box, candidate.source_size or (0, 0)
        ),
        eye_confidence=candidate.eye.confidence,
        head_confidence=candidate.eye.head_confidence,
    )


def combine(metrics: list[ImageMetrics], weights: dict[str, float]) -> list[float]:
    """One score per image, from folder-normalised metrics and normalised weights.

    Each metric is normalised across the run independently (so a metric with
    a huge raw range cannot drown out one with a small range), then combined
    as a plain weighted sum. Subject size is already a fraction of the frame,
    but is normalised alongside the others anyway: an archive of distant birds
    where every subject fills 1-3% of the frame would otherwise contribute
    almost nothing to the ordering however high its weight was set.
    """
    eye = robust_normalize([m.eye_sharpness for m in metrics])
    subject = robust_normalize([m.subject_sharpness for m in metrics])
    size = robust_normalize([m.subject_size for m in metrics])
    return [
        weights["eye_sharpness_weight"] * eye[index]
        + weights["subject_sharpness_weight"] * subject[index]
        + weights["subject_size_weight"] * size[index]
        for index in range(len(metrics))
    ]


def write_filter_report(
    input_folder: Path, rejected: dict[str, str], counts: dict[str, int], strategy_id: str = STRATEGY_ID
) -> Path:
    """Record which images were filtered out and why, beside the ranking.

    The ranking CSV cannot carry this: a filtered image has no score, so it
    has no row. Written as its own sidecar file so "why is this frame
    unranked?" has an answer that survives the run. `strategy_id` defaults
    to the legacy (SuperAnimal-Bird) id for backward compatibility - see
    `_filter_report_filename`.
    """
    ensure_sidecar_dir(input_folder)
    target = input_folder / SIDECAR_DIRNAME / _filter_report_filename(strategy_id)
    payload = {
        "version": 1,
        "strategy": strategy_id,
        "counts": counts,
        "images": rejected,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_filter_report(input_folder: str | Path, strategy_id: str = STRATEGY_ID) -> dict:
    """The last run's filter verdicts for this folder and backend, or `{}`."""
    target = Path(input_folder) / SIDECAR_DIRNAME / _filter_report_filename(strategy_id)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", target, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def write_metrics_report(
    input_folder: str | Path, metrics: list[ImageMetrics], strategy_id: str = STRATEGY_ID
) -> Path:
    """Record every surviving image's raw, per-metric measurements, beside
    the filter report and the ranking itself.

    Not derivable from the ranking CSV, which only ever carries the final
    combined score - a photographer investigating why a weak-eyed image
    still ranked respectably (or a strong-eyed one ranked low) needs to see
    the three numbers `combine()` weighted together, not just their sum.
    `strategy_id` defaults to the legacy (SuperAnimal-Bird) id for backward
    compatibility - see `_metrics_report_filename`.
    """
    ensure_sidecar_dir(input_folder)
    target = Path(input_folder) / SIDECAR_DIRNAME / _metrics_report_filename(strategy_id)
    payload = {
        "version": 1,
        "strategy": strategy_id,
        "metrics": {
            m.image_path: {
                "eye_sharpness": m.eye_sharpness,
                "subject_sharpness": m.subject_sharpness,
                "subject_size": m.subject_size,
                "eye_confidence": m.eye_confidence,
                "head_confidence": m.head_confidence,
            }
            for m in metrics
        },
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_metrics_report(input_folder: str | Path, strategy_id: str = STRATEGY_ID) -> dict:
    """The last run's raw per-image metrics for this folder and backend, or
    `{}`."""
    target = Path(input_folder) / SIDECAR_DIRNAME / _metrics_report_filename(strategy_id)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", target, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


class ClassicVisionStrategy:
    """Implements `ranking.base.RankingStrategy` deterministically, against
    the SuperAnimal-Bird eye-detection backend.

    This is one of potentially several Classic Vision *backends* - see
    `ClassicVisionEyePoseStrategy` for the second (EyePose-v0) - which is
    why `info`/`params_class`/`param_specs` are declared here rather than
    assumed fixed: a subclass overrides exactly these three class
    attributes and `_eye_detector_kwargs` below, and inherits everything
    else (filtering, scoring, CSV/report writing) completely unchanged.
    Registering a third backend later is the same shape again - see
    `eyes.build_eye_detector`'s own module docstring for the matching
    detector-side half of this.
    """

    info = StrategyInfo(
        strategy_id=STRATEGY_ID,
        display_name="Classic Vision Ranking (SuperAnimal)",
        description=(
            "Deterministic scoring from eye sharpness, subject sharpness and subject "
            "size. Filters out frames with no subject or no visible eye. Eye "
            "localisation: SuperAnimal-Bird (DeepLabCut Model Zoo)."
        ),
        score_label="Classic (SuperAnimal)",
    )
    params_class = ClassicVisionParams
    param_specs = ClassicVisionParams.specs()
    # Labels for the raw metrics written to METRICS_REPORT_FILENAME - a
    # diagnostics UI (see `ranking.metric_labels`) reads this class attribute
    # generically, by name, rather than importing classic.py directly.
    metric_labels = METRIC_LABELS
    # Which eyes.build_eye_detector name this backend runs - the ONE thing
    # that actually differs at the detector level; overridden per subclass.
    _eye_detector_name = "superanimal-bird"
    # Whether EyeFilter re-checks the crop's COCO class label against the
    # chosen detector's own supports() before running it (see EyeFilter's
    # own docstring for why this exists at all). True everywhere except
    # ClassicVisionCombinedStrategy, which overrides this to False: it
    # already decides subject eligibility per Burst via a crop-based CLIP
    # domain classification (see ranking.combined) before a detector is ever
    # selected, so re-asking a COCO label the whole point was to stop
    # trusting for exactly this would only reintroduce the bug that
    # architecture exists to avoid.
    _gate_by_subject_label = True

    def _eye_detector_kwargs(self, params: ClassicVisionParams) -> dict:
        """This backend's own params -> `eyes.build_eye_detector` kwargs.

        The only place `rank_folder` below is not 100% shared across
        backends: SuperAnimal-Bird's and EyePose-v0's accept/reject gates
        are genuinely different concepts with different parameter names
        (see `ClassicVisionParams`'s own docstring), so each backend maps
        its own `params` dataclass to its own detector constructor here
        rather than `rank_folder` assuming one shared shape.
        """
        return {"min_confidence": params.min_eye_confidence, "max_eye_disagreement": params.max_eye_disagreement}

    def _eye_detector_metadata(self, params) -> dict:
        """A JSON/CSV-safe summary of this run's eye-detector configuration,
        for `write_run_metadata`/`analytics.record_run` - separate from
        `_eye_detector_kwargs` because that one's return value is allowed to
        contain a real object (see `ClassicVisionBirdFusionStrategy`'s own
        override: a `FusionConfig` instance, not JSON-serialisable) since it
        is only ever passed straight to a detector's constructor, never
        persisted. Defaults to `_eye_detector_kwargs` unchanged, which is
        already flat primitives for every non-Fusion backend.
        """
        return self._eye_detector_kwargs(params)

    def _build_eye_filter_router(
        self,
        params,
        resolved_device: str,
        image_paths: list[str],
        crop_cache_dir: str | Path,
    ) -> Callable[[str], object]:
        """A function mapping an image path to the `EyeDetector` that should
        run on it - one detector for the whole folder by default (today's
        behaviour, unchanged), built once here rather than inside
        `rank_folder`'s own loop.

        The one override point `ranking.combined.ClassicVisionCombinedStrategy`
        needs for per-Burst domain routing (see that module's own docstring):
        it overrides this single method to classify each Burst once and
        return a different detector per path, while `rank_folder`'s loop
        itself never changes - every other strategy still gets exactly one
        shared detector, exactly as before this hook existed.
        """
        from ..eyes import build_eye_detector

        detector = build_eye_detector(
            self._eye_detector_name,
            device=resolved_device,
            **self._eye_detector_kwargs(params),
        )
        return lambda image_path: detector

    def rank_folder(
        self,
        input_folder: str | Path,
        *,
        params: ClassicVisionParams | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        crop_cache_dir: str | Path = DEFAULT_CROP_CACHE_DIR,
        device: str | None = None,
        max_rows: int = DEFAULT_MAX_CSV_ROWS,
        debug_dir: str | Path | None = None,
        force_preprocess: bool = False,
        analytics_db: str | Path = DEFAULT_ANALYTICS_DB,
    ) -> dict:
        """See the class docstring. `debug_dir` is a development/
        troubleshooting aid, off by default and never exposed in the
        desktop UI's generated parameter dialog (see `ranking.debug`'s
        module docstring): when set, one combined debug image per processed
        candidate is written there, showing the crop, the eye box, both eye
        keypoints, confidence values, and the eye box projected onto the
        full frame - drawn from the same `FilterCandidate`/`EyeDetection`
        shape regardless of which backend produced it.

        `force_preprocess` is passed straight through to `build_cache` -
        needed because a Vision Cache built under different cache-affecting
        parameters (resolution, format, quality - see `bird_crop.CropParams`)
        is refused rather than silently reused (see `build_cache`'s own
        `crop_params.json` mismatch check). Same name and meaning as
        `rank.rank_folder`'s own parameter, so the same "rebuild the cache"
        action means the same thing from either ranking strategy.

        `analytics_db` defaults to the real shared database - overridable so
        tests (and any caller that wants an isolated run history) never
        write into it. See `analytics.capture.record_run`.
        """
        from ..eyes import build_eye_detector
        from ..eyes.cache import save_eye_detection
        from ..train import write_results_csv
        from .debug import save_debug_image

        start_time = time.perf_counter()
        params = params or self.params_class()
        input_folder = Path(input_folder)
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

        image_paths = analysis_targets(input_folder)
        if not image_paths:
            raise ValueError(f"No images found under {input_folder.resolve()}")
        # UnlabeledImageDataset is constructed from the paths directly rather
        # than through from_folder(): this module does its own, deliberately
        # workflow-blind enumeration (see analysis_targets), and only needs the
        # dataset as the shape write_results_csv accepts.
        dataset = UnlabeledImageDataset(image_paths)

        resolved_device = resolve_device(device)

        # Step 1: the shared crop cache. Idempotent - on a folder the AI model
        # has already ranked this is a fast pass that decodes nothing.
        if on_stage is not None:
            on_stage("Building subject-crop cache")
        # Everything else (original crop resolution, JPEG q98, ...) picks up
        # the Vision Cache's own current defaults automatically - see
        # bird_crop.py's module docstring. detection_confidence_threshold/
        # crop_confidence_threshold ARE Classic-Vision-configurable (see
        # _detection_specs) because they change which subject gets cropped -
        # unlike eye_confidence_threshold/max_head_distance_ratio below,
        # changing either of these two legitimately requires a crop-cache
        # rebuild (CropParams's own version-mismatch check already enforces
        # this - see docs/EyePose_Investigation_Phase_1.md's Part 5).
        crop_params = CropParams(
            conf_threshold=params.detection_confidence_threshold,
            min_crop_confidence=params.crop_confidence_threshold,
        )
        build_cache(image_paths, crop_cache_dir, crop_params, device=resolved_device, force=force_preprocess)

        if on_stage is not None:
            on_stage("Loading the eye detector")
        detector_for_image = self._build_eye_filter_router(params, resolved_device, image_paths, crop_cache_dir)

        if on_stage is not None:
            on_stage("Filtering and measuring images")
        measurements: list[ImageMetrics] = []
        rejected: dict[str, str] = {}
        counts: dict[str, int] = {}
        for index, image_path in enumerate(image_paths, start=1):
            candidate = self._load_candidate(
                image_path, crop_cache_dir, require_selected_detection=params.use_subject_filter
            )
            eye_filter = EyeFilter(
                detector_for_image(image_path), gate_by_subject_label=self._gate_by_subject_label
            )
            chain = FilterChain([SubjectFilter(), eye_filter])
            reason = chain.reject_reason(candidate)
            # Persisted whenever the eye detector actually ran on this image,
            # regardless of whether EyeFilter accepted the result - see
            # eyes.cache's module docstring. A REJECTED image's raw keypoints
            # are exactly what a photographer investigating a filtering (or
            # non-filtering) decision needs the Gallery/Loupe overlay to show.
            if candidate.eye is not None and candidate.subject_crop is not None:
                crop_height, crop_width = candidate.subject_crop.shape[:2]
                save_eye_detection(
                    crop_cache_dir, image_path, (crop_width, crop_height), candidate.eye,
                    strategy_id=self.info.strategy_id,
                )
            if debug_dir is not None:
                save_debug_image(candidate, self.info.strategy_id, debug_dir)
            if reason is not None:
                rejected[image_path] = reason
                counts[reason] = counts.get(reason, 0) + 1
            else:
                measurements.append(measure(candidate))
            if on_progress is not None:
                on_progress(index, len(image_paths))

        if on_stage is not None:
            on_stage("Scoring and writing the ranking")
        scores = combine(measurements, params.normalized_weights())
        # The (name, score, label, path) tuple shape write_results_csv expects.
        # Label 0 throughout: these images are unlabelled, exactly as in the AI
        # inference path, where rank_dataset passes the dataset's own
        # placeholder label through.
        ranked = [
            (Path(m.image_path).name, score, 0, str(m.image_path))
            for m, score in zip(measurements, scores)
        ]
        # Kept before the sort below reorders `ranked` - zip(measurements,
        # scores) pairs by position, not by name/path.
        score_by_path = {m.image_path: score for m, score in zip(measurements, scores)}
        ranked.sort(key=lambda entry: entry[1], reverse=True)

        strategy_id = self.info.strategy_id
        output_paths = write_results_csv(
            # This backend's OWN scores file, never the AI model's nor
            # another Classic Vision backend's - see sidecar.py. An image
            # can carry every strategy's score at once, and running one must
            # never destroy another's results.
            strategy_ranking_path(input_folder, strategy_id),
            dataset,
            ranked,
            select_root=str(input_folder),
            reject_root="(classic vision - no labels)",
            max_rows=max_rows,
        )
        write_filter_report(input_folder, rejected, counts, strategy_id=strategy_id)
        write_metrics_report(input_folder, measurements, strategy_id=strategy_id)
        write_run_metadata(
            input_folder,
            strategy=strategy_id,
            image_count=len(ranked),
            considered=len(image_paths),
            filtered=counts,
            weights=params.normalized_weights(),
            eye_detector=self._eye_detector_name,
            **self._eye_detector_metadata(params),
        )
        runtime_seconds = time.perf_counter() - start_time
        record_run(
            input_folder,
            strategy_id,
            considered=len(image_paths),
            accepted=len(ranked),
            reject_counts=counts,
            image_metrics={
                m.image_path: {
                    "score": score_by_path[m.image_path],
                    "eye_sharpness": m.eye_sharpness,
                    "subject_sharpness": m.subject_sharpness,
                    "subject_size": m.subject_size,
                    "eye_confidence": m.eye_confidence,
                    "head_confidence": m.head_confidence,
                }
                for m in measurements
            },
            summary_metrics={
                "runtime_seconds": runtime_seconds,
                "images_per_second": len(image_paths) / runtime_seconds if runtime_seconds > 0 else 0.0,
            },
            params={
                "algorithm_version": ALGORITHM_VERSION,
                "weights": params.normalized_weights(),
                "eye_detector": self._eye_detector_name,
                **self._eye_detector_metadata(params),
                **resolve_environment_info(),
            },
            device=resolved_device,
            db_path=analytics_db,
        )

        return {
            "strategy": strategy_id,
            "output_csv": output_paths[0],
            "extra_csv_files": output_paths[1:],
            "image_count": len(ranked),
            "considered": len(image_paths),
            "filtered": counts,
            "device": resolved_device,
            "top": [(name, score) for name, score, _label, _path in ranked[:10]],
        }

    @staticmethod
    def _load_candidate(
        image_path: str, crop_cache_dir: str | Path, *, require_selected_detection: bool = True
    ) -> FilterCandidate:
        """Assemble one candidate from what preprocessing already wrote.

        Reads only - the cached crop PNG and the detection sidecar beside it.
        An image whose crop or record is unreadable comes back with nothing
        filled in, which `SubjectFilter` correctly rejects as NO_SUBJECT: from
        the algorithm's point of view there is indeed no subject it can see.

        `require_selected_detection` (True = today's original, unchanged
        behavior) is `WeightedParams.use_subject_filter` threaded down from
        `rank_folder`. `bird_crop.build_crop` always writes SOME crop for an
        image it could decode - a tight subject crop when the shared COCO
        detector confidently found one ("selected" is present), otherwise the
        full decoded frame as a fallback (see that module's own docstring).
        Historically this method only ever exposed the first case; the second
        made every filter downstream reject the image as NO_SUBJECT even
        though a real, readable crop existed on disk.

        The Eye-Detector Ensemble Evidence Study found that gate - a COCO
        class + confidence judgement - to be the least reliable stage in the
        pipeline, while the crop itself (whichever kind) was consistently
        usable. Passing `require_selected_detection=False` (the
        `use_subject_filter=False` default) relaxes exactly this: the crop
        file is read whenever it exists, and `subject_box`/`crop_box` fall
        back to the whole decoded frame when there is no tight detection to
        use instead - an honest "we don't know where in the frame the
        subject is, only that a photograph exists" default, not a fabricated
        detection. `subject_label` stays `None` in that case (no COCO class
        was ever assigned), which already makes `EyeFilter`'s own class-label
        gate a no-op (see that class's own docstring) - so a strategy that
        still can't find what it needs (e.g. no visible eye) rejects the
        image through its OWN existing reason, never NO_SUBJECT again.
        """
        import cv2

        candidate = FilterCandidate(image_path=image_path)
        record = read_detections(crop_cache_dir, image_path)
        if not record:
            return candidate

        selected = record.get("selected")
        source_size = record.get("source_size")
        if not source_size or len(source_size) != 2:
            return candidate
        if not selected and require_selected_detection:
            return candidate

        box = selected.get("box") if selected else None
        if selected and (not box or len(box) != 4):
            if require_selected_detection:
                return candidate
            box = None  # a malformed "selected" entry is no better than none

        crop_path = crop_cache_path(crop_cache_dir, image_path)
        image_bgr = cv2.imread(str(crop_path))
        if image_bgr is None:
            return candidate

        width, height = int(source_size[0]), int(source_size[1])
        candidate.subject_crop = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        candidate.source_size = (width, height)

        if box is not None:
            expanded = record.get("expanded_box")
            candidate.subject_box = tuple(float(v) for v in box)
            # The crop's own rectangle - see FilterCandidate.crop_box's docstring.
            # None only for a record written before expanded_box existed (a very
            # old, pre-Vision-Cache cache entry); CROP_CACHE_VERSION's v6 bump
            # already forces those to rebuild, so this is a defensive fallback,
            # not the expected path.
            candidate.crop_box = (
                tuple(float(v) for v in expanded) if expanded and len(expanded) == 4 else None
            )
            candidate.subject_label = int(selected.get("label", -1))
        else:
            # No confident detection, and the caller asked for the crop
            # anyway (require_selected_detection=False): the whole frame is
            # both the subject box and the crop's own rectangle, since
            # nothing more specific was ever located. subject_label stays
            # None - see this method's own docstring.
            whole_frame = (0.0, 0.0, float(width), float(height))
            candidate.subject_box = whole_frame
            candidate.crop_box = whole_frame
        return candidate


EYEPOSE_STRATEGY_ID = "classic-vision-eyepose-v0"


class ClassicVisionEyePoseStrategy(ClassicVisionStrategy):
    """Classic Vision against the EyePose-v0 backend instead of
    SuperAnimal-Bird - filtering, scoring, CSV/report writing are all
    inherited from `ClassicVisionStrategy` completely unchanged (see its own
    docstring); only `info`, `params_class`/`param_specs`,
    `_eye_detector_name` and `_eye_detector_kwargs` differ, which is the
    entire adapter surface a new Classic Vision backend needs to override.

    A distinct `strategy_id` means a distinct ranking CSV
    (`sidecar.strategy_ranking_path`) and distinct filter/metrics reports
    (`_filter_report_filename`/`_metrics_report_filename`) - the two
    backends' results coexist on the same folder rather than one
    overwriting the other, so they can be compared directly (the Sort/Color
    Source/score-row UI already does this generically for any two
    strategies - see `ranking.score_labels`/`metric_labels`).
    """

    info = StrategyInfo(
        strategy_id=EYEPOSE_STRATEGY_ID,
        display_name="Classic Vision Ranking (EyePose-v0, recommended)",
        description=(
            "Deterministic scoring from eye sharpness, subject sharpness and subject "
            "size. Filters out frames with no subject or no visible eye. Eye "
            "localisation: EyePose-v0 (YOLO11-pose, six bird head/body landmarks)."
        ),
        score_label="Classic (EyePose)",
    )
    params_class = ClassicVisionEyePoseParams
    param_specs = ClassicVisionEyePoseParams.specs()
    metric_labels = METRIC_LABELS
    _eye_detector_name = "eyepose-v0"

    def _eye_detector_kwargs(self, params: ClassicVisionEyePoseParams) -> dict:
        return {
            "min_confidence": params.eye_confidence_threshold,
            "max_head_distance_ratio": params.max_head_distance_ratio,
            "min_head_confidence": params.detection_head_confidence_threshold,
        }


# ---------------------------------------------------------------------------
# Fusion backends - Ranking Mode = Birds / Mammals (see eyes.domains). Eye
# localisation for these two backends comes from the shared Fusion/
# Validation layer (eyes.fusion.FusionEyeDetector) rather than a single
# model - see that module's own docstring for the algorithm, and
# eyes.domains for which concrete detectors/default weights each Ranking
# Mode selects. Filtering, scoring, CSV/report writing are all inherited
# from ClassicVisionStrategy unchanged, exactly like ClassicVisionEyePoseStrategy
# above - only info/params_class/param_specs/_eye_detector_name/
# _eye_detector_kwargs differ.
# ---------------------------------------------------------------------------

BIRD_FUSION_STRATEGY_ID = "classic-vision-fusion-birds"
MAMMAL_FUSION_STRATEGY_ID = "classic-vision-fusion-mammals"


def _fusion_weight_specs(profile, help_prefix: str) -> tuple[ParamSpec, ...]:
    """One ParamSpec per model in a domain profile's default weights - a
    single-model domain (Mammals, today) still gets a real, adjustable
    weight field, not a hidden constant, so adding a second mammal model
    later needs no new UI wiring (see eyes.domains.DomainProfile's own
    docstring)."""
    return tuple(
        ParamSpec(
            name=f"{model_weight.detector_id.replace('-', '_')}_model_weight",
            label=f"{model_weight.detector_id} model weight",
            default=model_weight.weight,
            minimum=0.0,
            maximum=1.0,
            group=GROUP_WEIGHTS,
            decimals=2,
            help=f"{help_prefix} how much the Fusion Layer trusts {model_weight.detector_id}.",
        )
        for model_weight in profile.default_model_weights
    )


def _fusion_threshold_specs() -> tuple[ParamSpec, ...]:
    return (
        ParamSpec(
            name="agreement_threshold",
            label="Agreement threshold",
            default=DEFAULT_AGREEMENT_THRESHOLD,
            minimum=0.0,
            maximum=5.0,
            group=GROUP_THRESHOLDS,
            decimals=2,
            help="How close (in head-scale units) two models' predictions must be to count as agreeing, rather than a disagreement.",
        ),
        ParamSpec(
            name="min_fused_confidence",
            label="Minimum fused confidence",
            default=DEFAULT_MIN_FUSED_CONFIDENCE,
            minimum=0.0,
            maximum=1.0,
            group=GROUP_THRESHOLDS,
            decimals=2,
            help="Below this, the fused result counts as not visible and the image is filtered out.",
        ),
    )


@dataclass(frozen=True)
class ClassicVisionBirdFusionParams(WeightedParams):
    """Ranking Mode: Birds, via the shared Fusion Layer - EyePose-v0 +
    SuperAnimal-Bird combined (see eyes.fusion, eyes.domains.BIRDS_PROFILE),
    rather than either backend scored alone (see ClassicVisionEyePoseStrategy/
    ClassicVisionStrategy for the two single-model Bird strategies this sits
    alongside - all three remain independently selectable/cached)."""

    eye_sharpness_weight: float = 70.0
    subject_sharpness_weight: float = 10.0
    subject_size_weight: float = 20.0
    eyepose_v0_model_weight: float = BIRDS_PROFILE.default_model_weights[0].weight
    superanimal_bird_model_weight: float = BIRDS_PROFILE.default_model_weights[1].weight
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD
    min_fused_confidence: float = DEFAULT_MIN_FUSED_CONFIDENCE
    detection_confidence_threshold: float = CropParams.conf_threshold
    crop_confidence_threshold: float = CropParams.min_crop_confidence

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        return (
            *_scoring_weight_specs(),
            *_fusion_weight_specs(BIRDS_PROFILE, "Birds Ranking Mode:"),
            *_fusion_threshold_specs(),
            *_detection_specs(),
            use_subject_filter_spec(),
        )


class ClassicVisionBirdFusionStrategy(ClassicVisionStrategy):
    info = StrategyInfo(
        strategy_id=BIRD_FUSION_STRATEGY_ID,
        display_name="Classic Vision Ranking (Birds - Fusion)",
        description=(
            "Ranking Mode: Birds. Eye localisation: EyePose-v0 + SuperAnimal-Bird, "
            "combined through the shared Fusion/Validation layer - agreement, "
            "geometric plausibility and per-model trust decide the final eye "
            "rather than either model alone."
        ),
        score_label="Classic (Birds Fusion)",
    )
    params_class = ClassicVisionBirdFusionParams
    param_specs = ClassicVisionBirdFusionParams.specs()
    metric_labels = METRIC_LABELS
    _eye_detector_name = "fusion-birds"

    def _eye_detector_kwargs(self, params: ClassicVisionBirdFusionParams) -> dict:
        return {
            "config": FusionConfig(
                model_weights=(
                    ModelWeight("eyepose-v0", params.eyepose_v0_model_weight),
                    ModelWeight("superanimal-bird", params.superanimal_bird_model_weight),
                ),
                agreement_threshold=params.agreement_threshold,
                min_fused_confidence=params.min_fused_confidence,
            )
        }

    def _eye_detector_metadata(self, params: ClassicVisionBirdFusionParams) -> dict:
        return {
            "eyepose_v0_model_weight": params.eyepose_v0_model_weight,
            "superanimal_bird_model_weight": params.superanimal_bird_model_weight,
            "agreement_threshold": params.agreement_threshold,
            "min_fused_confidence": params.min_fused_confidence,
        }


@dataclass(frozen=True)
class ClassicVisionMammalFusionParams(WeightedParams):
    """Ranking Mode: Mammals, via the shared Fusion Layer -
    SuperAnimal-Quadruped today (see eyes.domains.MAMMALS_PROFILE); the same
    Fusion Layer ClassicVisionBirdFusionParams uses, just constructed with a
    different domain profile's detectors/weights - never a second fusion
    implementation (see eyes.fusion's own module docstring)."""

    eye_sharpness_weight: float = 70.0
    subject_sharpness_weight: float = 10.0
    subject_size_weight: float = 20.0
    superanimal_quadruped_model_weight: float = MAMMALS_PROFILE.default_model_weights[0].weight
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD
    min_fused_confidence: float = DEFAULT_MIN_FUSED_CONFIDENCE
    detection_confidence_threshold: float = CropParams.conf_threshold
    crop_confidence_threshold: float = CropParams.min_crop_confidence

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        return (
            *_scoring_weight_specs(),
            *_fusion_weight_specs(MAMMALS_PROFILE, "Mammals Ranking Mode:"),
            *_fusion_threshold_specs(),
            *_detection_specs(),
            use_subject_filter_spec(),
        )


class ClassicVisionMammalFusionStrategy(ClassicVisionStrategy):
    info = StrategyInfo(
        strategy_id=MAMMAL_FUSION_STRATEGY_ID,
        display_name="Classic Vision Ranking (Mammals - Fusion)",
        description=(
            "Ranking Mode: Mammals. Eye localisation: SuperAnimal-Quadruped, "
            "through the shared Fusion/Validation layer. Does NOT use "
            "EyePose-v0 - it is a bird-specific model (see eyes.domains)."
        ),
        score_label="Classic (Mammals Fusion)",
    )
    params_class = ClassicVisionMammalFusionParams
    param_specs = ClassicVisionMammalFusionParams.specs()
    metric_labels = METRIC_LABELS
    _eye_detector_name = "fusion-mammals"

    def _eye_detector_kwargs(self, params: ClassicVisionMammalFusionParams) -> dict:
        return {
            "config": FusionConfig(
                model_weights=(
                    ModelWeight("superanimal-quadruped", params.superanimal_quadruped_model_weight),
                ),
                agreement_threshold=params.agreement_threshold,
                min_fused_confidence=params.min_fused_confidence,
            )
        }

    def _eye_detector_metadata(self, params: ClassicVisionMammalFusionParams) -> dict:
        return {
            "superanimal_quadruped_model_weight": params.superanimal_quadruped_model_weight,
            "agreement_threshold": params.agreement_threshold,
            "min_fused_confidence": params.min_fused_confidence,
        }
