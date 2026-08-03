"""Experiment metadata - the reproducibility record for one species-
classification run, built once per run and handed to analytics (see
`analytics.species_capture`) to persist.

Every field answers one question a future benchmark needs answered without
re-running anything: *exactly* what produced these predictions? Four
independent version axes are captured, deliberately kept separate rather
than collapsed into one "version" string, because they can each change
independently and a benchmark needs to know which one moved:

    model_version      - which exact upstream checkpoint (a specific
                          commit of imageomics/bioclip-2 on Hugging Face
                          Hub, resolved from the local cache - see
                          `_resolve_model_commit`). Changes when Imageomics
                          publishes a new commit under the same model_id.
    classifier_version  - which version of THIS project's own wrapper code
                          (BioClipSpeciesClassifier) ran - see
                          bioclip_classifier.CLASSIFIER_VERSION. Changes
                          when this project's own preprocessing/prompt/
                          decision logic changes, independent of the model.
    open_clip_version    - which version of the third-party open_clip
                          library did the actual inference. Changes on a
                          `pip install --upgrade open_clip_torch`.
    application_version   - this project's own release version
                          (pyproject.toml), for correlating an experiment
                          with "which PeakPic build produced this."

None of this is guessed when unavailable - every field that cannot be
resolved (no git repo, no network/cache entry, `pip`-installed as a wheel
with no version metadata) is `None`, not a fabricated placeholder, matching
this project's own "explicit unknown, never guessed" standard (see
docs/BioCLIP_Backend_Architecture_Review.md).
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentMetadata:
    """Everything needed to reproduce, or later distinguish, one species-
    classification run. Constructed once per run by `build_experiment_
    metadata`, never mutated afterward - a run's own record of what it
    was, not a live/updatable configuration object."""

    experiment_id: str
    timestamp: str

    classifier_backend: str  # the registry name, e.g. "bioclip2"
    model_id: str  # e.g. "hf-hub:imageomics/bioclip-2"
    model_version: str | None  # resolved HF Hub commit SHA, or None if unresolvable
    classifier_version: str  # this project's own BioClipSpeciesClassifier code version

    species_list_hash: str  # sha1 of the exact species list, first 12 hex chars
    species_list_filename: str  # the --species-list path, or "(built-in default)"
    species_count: int

    device: str  # the ACTUAL device inference ran on (see Part 3 - never assumed)
    gpu_name: str | None
    cuda_available: bool

    open_clip_version: str | None
    application_version: str | None
    git_commit: str | None

    configuration_hash: str  # sha1 of every threshold below, first 12 hex chars
    thresholds: dict[str, float | str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "classifier_backend": self.classifier_backend,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "classifier_version": self.classifier_version,
            "species_list_hash": self.species_list_hash,
            "species_list_filename": self.species_list_filename,
            "species_count": self.species_count,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "cuda_available": self.cuda_available,
            "open_clip_version": self.open_clip_version,
            "application_version": self.application_version,
            "git_commit": self.git_commit,
            "configuration_hash": self.configuration_hash,
            "thresholds": dict(self.thresholds),
        }


def _resolve_model_commit(model_id: str) -> str | None:
    """The exact Hugging Face Hub commit SHA the local cache resolved
    `model_id` to, read directly from the cache's own `refs/main` file -
    no network call, so this works fully offline (matching this project's
    own offline-after-first-download standard). Returns None if the model
    was never downloaded (should not happen for an already-constructed
    classifier) or the cache layout is not where expected (e.g. a
    customised HF_HOME) - reported as unknown, never guessed.
    """
    if not model_id.startswith("hf-hub:"):
        return None
    repo_id = model_id[len("hf-hub:"):]

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_root = Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 - fall back to the documented default location
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    folder_name = "models--" + repo_id.replace("/", "--")
    ref_file = cache_root / folder_name / "refs" / "main"
    try:
        return ref_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        logger.debug("Could not resolve model commit for %s at %s", model_id, ref_file)
        return None


def _resolve_open_clip_version() -> str | None:
    try:
        import open_clip
        return getattr(open_clip, "__version__", None)
    except Exception:  # noqa: BLE001 - version reporting must never break a run
        return None


def _resolve_application_version() -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version("pick-likeme")
    except Exception:  # noqa: BLE001
        return None


def _resolve_git_commit() -> str | None:
    """Best-effort - a packaged/non-git install must not fail this."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _resolve_gpu_name(torch_module) -> str | None:
    try:
        if torch_module.cuda.is_available():
            return torch_module.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - a GPU query must never break a run
        pass
    return None


def build_experiment_metadata(
    classifier, backend: str, *, species_list_path: str | None = None
) -> ExperimentMetadata:
    """The reproducibility record for a run about to use `classifier`.

    `backend` is the registry name (`species.classifier.available_
    classifiers()`'s own `classifier_id`, e.g. "bioclip2") that produced
    this instance - deliberately passed in by the caller rather than
    derived from the instance itself. `classifier.classifier_id` encodes
    the *model name* (e.g. "bioclip-2"), which is a different string from
    the *registry key* used to build it ("bioclip2") - a classifier
    instance has no way to know which registry name it was constructed
    through (the same reason `eyes.build_eye_detector`'s registry names
    are never stored on the detector instance either), so guessing at this
    from `classifier_id` alone would be wrong for exactly the kind of
    silent-mislabelling reason this whole review started from.

    Reads only public attributes already present on any `SpeciesClassifier`-
    shaped object built by `species.classifier.build_classifier` today
    (`model_id`, `species_list`, `min_confidence`, `prompt_template`,
    `device`, `classifier_id`) - a future non-BioCLIP backend that lacks
    some of these degrades gracefully (missing fields become None), it does
    not crash metadata collection for the whole run.
    """
    import torch

    model_id = getattr(classifier, "model_id", "unknown")
    species_list = tuple(getattr(classifier, "species_list", ()))
    device = str(getattr(classifier, "device", "unknown"))
    min_confidence = getattr(classifier, "min_confidence", None)
    prompt_template = getattr(classifier, "prompt_template", None)

    thresholds: dict[str, float | str] = {}
    if min_confidence is not None:
        thresholds["min_confidence"] = float(min_confidence)
    if prompt_template is not None:
        thresholds["prompt_template"] = str(prompt_template)
    configuration_hash = hashlib.sha1(
        json.dumps(thresholds, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    species_list_hash = hashlib.sha1("\n".join(species_list).encode("utf-8")).hexdigest()[:12]

    from .bioclip_classifier import CLASSIFIER_VERSION

    return ExperimentMetadata(
        experiment_id=str(uuid4()),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        classifier_backend=backend,
        model_id=model_id,
        model_version=_resolve_model_commit(model_id),
        classifier_version=CLASSIFIER_VERSION,
        species_list_hash=species_list_hash,
        species_list_filename=species_list_path or "(built-in default)",
        species_count=len(species_list),
        device=device,
        gpu_name=_resolve_gpu_name(torch),
        cuda_available=bool(torch.cuda.is_available()),
        open_clip_version=_resolve_open_clip_version(),
        application_version=_resolve_application_version(),
        git_commit=_resolve_git_commit(),
        configuration_hash=configuration_hash,
        thresholds=thresholds,
    )
