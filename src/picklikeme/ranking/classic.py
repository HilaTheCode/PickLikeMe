"""Classic Vision Ranking - a deterministic alternative to the trained model.

No learning, no checkpoint, no preference model: two filters decide whether
an image is judged at all, and three measurements decide how it scores. Run
it twice on the same folder with the same parameters and it produces
byte-identical numbers.

    Phase 1 - filtering (ranking.filters)
        Filter 1  no detected subject          -> NO_SUBJECT
        Filter 2  no visible eye               -> NO_VISIBLE_EYE
                  (or UNSUPPORTED_SUBJECT, when no eye detector covers it)

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..auto_crop import resolve_device
from ..bird_crop import CropParams, crop_cache_path, read_detections
from ..config import DEFAULT_CROP_CACHE_DIR, DEFAULT_MAX_CSV_ROWS
from ..dataset import UnlabeledImageDataset
from ..eyes.superanimal_bird import DEFAULT_MAX_EYE_DISAGREEMENT, DEFAULT_MIN_CONFIDENCE
from ..preprocess import build_cache
from ..sidecar import (
    SIDECAR_DIRNAME,
    ensure_sidecar_dir,
    strategy_ranking_path,
    write_run_metadata,
)
from .base import GROUP_THRESHOLDS, GROUP_WEIGHTS, ParamSpec, StrategyInfo, WeightedParams
from .filters import EyeFilter, FilterCandidate, FilterChain, SubjectFilter
from .metrics import (
    normalized_subject_size,
    region_focus_measure,
    robust_normalize,
    subject_focus_measure,
)

logger = logging.getLogger(__name__)

STRATEGY_ID = "classic-vision"

# Where the per-image filter verdicts land, beside the ranking CSV the same
# run produced. Not merged into run.json (which is provenance about the run as
# a whole) because this is per-image data that grows with the folder.
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


@dataclass(frozen=True)
class ClassicVisionParams(WeightedParams):
    """Everything the photographer can tune before a Classic Vision run.

    Adding a parameter later is: one field here, one `ParamSpec` in `specs()`.
    The dialog builds itself from `specs()` (see the desktop
    `AlgorithmParametersDialog`), the weights normalise themselves, and no
    other code changes.
    """

    eye_sharpness_weight: float = 50.0
    subject_sharpness_weight: float = 30.0
    subject_size_weight: float = 20.0
    min_eye_confidence: float = DEFAULT_MIN_CONFIDENCE
    max_eye_disagreement: float = DEFAULT_MAX_EYE_DISAGREEMENT

    @classmethod
    def specs(cls) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                name="eye_sharpness_weight",
                label="Eye sharpness",
                default=50.0,
                minimum=0.0,
                maximum=1000.0,
                group=GROUP_WEIGHTS,
                help="How sharp the eye itself is, measured inside the eye box only.",
            ),
            ParamSpec(
                name="subject_sharpness_weight",
                label="Subject sharpness",
                default=30.0,
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


def write_filter_report(input_folder: Path, rejected: dict[str, str], counts: dict[str, int]) -> Path:
    """Record which images were filtered out and why, beside the ranking.

    The ranking CSV cannot carry this: a filtered image has no score, so it
    has no row. Written as its own sidecar file so "why is this frame
    unranked?" has an answer that survives the run.
    """
    ensure_sidecar_dir(input_folder)
    target = input_folder / SIDECAR_DIRNAME / FILTER_REPORT_FILENAME
    payload = {
        "version": 1,
        "strategy": STRATEGY_ID,
        "counts": counts,
        "images": rejected,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_filter_report(input_folder: str | Path) -> dict:
    """The last Classic Vision run's filter verdicts for this folder, or `{}`."""
    target = Path(input_folder) / SIDECAR_DIRNAME / FILTER_REPORT_FILENAME
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", target, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def write_metrics_report(input_folder: str | Path, metrics: list[ImageMetrics]) -> Path:
    """Record every surviving image's raw, per-metric measurements, beside
    the filter report and the ranking itself.

    Not derivable from the ranking CSV, which only ever carries the final
    combined score - a photographer investigating why a weak-eyed image
    still ranked respectably (or a strong-eyed one ranked low) needs to see
    the three numbers `combine()` weighted together, not just their sum.
    """
    ensure_sidecar_dir(input_folder)
    target = Path(input_folder) / SIDECAR_DIRNAME / METRICS_REPORT_FILENAME
    payload = {
        "version": 1,
        "strategy": STRATEGY_ID,
        "metrics": {
            m.image_path: {
                "eye_sharpness": m.eye_sharpness,
                "subject_sharpness": m.subject_sharpness,
                "subject_size": m.subject_size,
                "eye_confidence": m.eye_confidence,
            }
            for m in metrics
        },
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_metrics_report(input_folder: str | Path) -> dict:
    """The last Classic Vision run's raw per-image metrics for this folder,
    or `{}`."""
    target = Path(input_folder) / SIDECAR_DIRNAME / METRICS_REPORT_FILENAME
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", target, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


class ClassicVisionStrategy:
    """Implements `ranking.base.RankingStrategy` deterministically."""

    info = StrategyInfo(
        strategy_id=STRATEGY_ID,
        display_name="Classic Vision Ranking",
        description=(
            "Deterministic scoring from eye sharpness, subject sharpness and subject "
            "size. Filters out frames with no subject or no visible eye."
        ),
        score_label="Classic",
    )
    params_class = ClassicVisionParams
    param_specs = ClassicVisionParams.specs()
    # Labels for the raw metrics written to METRICS_REPORT_FILENAME - a
    # diagnostics UI (see `ranking.metric_labels`) reads this class attribute
    # generically, by name, rather than importing classic.py directly.
    metric_labels = METRIC_LABELS

    def __init__(self, *, eye_detector_name: str = "superanimal-bird") -> None:
        self._eye_detector_name = eye_detector_name

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
    ) -> dict:
        from ..eyes import build_eye_detector
        from ..eyes.cache import save_eye_detection
        from ..train import write_results_csv

        params = params or ClassicVisionParams()
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
        build_cache(image_paths, crop_cache_dir, CropParams(), device=resolved_device)

        if on_stage is not None:
            on_stage("Loading the eye detector")
        eye_detector = build_eye_detector(
            self._eye_detector_name,
            device=resolved_device,
            min_confidence=params.min_eye_confidence,
            max_eye_disagreement=params.max_eye_disagreement,
        )
        chain = FilterChain([SubjectFilter(), EyeFilter(eye_detector)])

        if on_stage is not None:
            on_stage("Filtering and measuring images")
        measurements: list[ImageMetrics] = []
        rejected: dict[str, str] = {}
        counts: dict[str, int] = {}
        for index, image_path in enumerate(image_paths, start=1):
            candidate = self._load_candidate(image_path, crop_cache_dir)
            reason = chain.reject_reason(candidate)
            # Persisted whenever the eye detector actually ran on this image,
            # regardless of whether EyeFilter accepted the result - see
            # eyes.cache's module docstring. A REJECTED image's raw keypoints
            # are exactly what a photographer investigating a filtering (or
            # non-filtering) decision needs the Gallery/Loupe overlay to show.
            if candidate.eye is not None and candidate.subject_crop is not None:
                crop_height, crop_width = candidate.subject_crop.shape[:2]
                save_eye_detection(crop_cache_dir, image_path, (crop_width, crop_height), candidate.eye)
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
        ranked.sort(key=lambda entry: entry[1], reverse=True)

        output_paths = write_results_csv(
            # This module's OWN scores file, never the AI model's - see
            # sidecar.py. An image can carry both scores at once, and running
            # one analysis must not destroy the other's results.
            strategy_ranking_path(input_folder, STRATEGY_ID),
            dataset,
            ranked,
            select_root=str(input_folder),
            reject_root="(classic vision - no labels)",
            max_rows=max_rows,
        )
        write_filter_report(input_folder, rejected, counts)
        write_metrics_report(input_folder, measurements)
        write_run_metadata(
            input_folder,
            strategy=STRATEGY_ID,
            image_count=len(ranked),
            considered=len(image_paths),
            filtered=counts,
            weights=params.normalized_weights(),
            min_eye_confidence=params.min_eye_confidence,
            max_eye_disagreement=params.max_eye_disagreement,
            eye_detector=self._eye_detector_name,
        )

        return {
            "strategy": STRATEGY_ID,
            "output_csv": output_paths[0],
            "extra_csv_files": output_paths[1:],
            "image_count": len(ranked),
            "considered": len(image_paths),
            "filtered": counts,
            "device": resolved_device,
            "top": [(name, score) for name, score, _label, _path in ranked[:10]],
        }

    @staticmethod
    def _load_candidate(image_path: str, crop_cache_dir: str | Path) -> FilterCandidate:
        """Assemble one candidate from what preprocessing already wrote.

        Reads only - the cached crop PNG and the detection sidecar beside it.
        An image whose crop or record is unreadable comes back with nothing
        filled in, which `SubjectFilter` correctly rejects as NO_SUBJECT: from
        the algorithm's point of view there is indeed no subject it can see.
        """
        import cv2

        candidate = FilterCandidate(image_path=image_path)
        record = read_detections(crop_cache_dir, image_path)
        if not record:
            return candidate

        selected = record.get("selected")
        source_size = record.get("source_size")
        if not selected or not source_size or len(source_size) != 2:
            return candidate
        box = selected.get("box")
        if not box or len(box) != 4:
            return candidate

        crop_path = crop_cache_path(crop_cache_dir, image_path)
        image_bgr = cv2.imread(str(crop_path))
        if image_bgr is None:
            return candidate

        candidate.subject_crop = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        candidate.subject_box = tuple(float(v) for v in box)
        candidate.source_size = (int(source_size[0]), int(source_size[1]))
        candidate.subject_label = int(selected.get("label", -1))
        return candidate
