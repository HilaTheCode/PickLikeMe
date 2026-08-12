"""Thin services that expose the existing review backend to the desktop UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from ..analyzer.annotation_config import DEFAULT_ANNOTATIONS_CONFIG, load_annotation_fields
from ..analyzer.annotations import DEFAULT_ANNOTATIONS_DB, AnnotationStore
from ..auto_crop import generate_lightroom_crops
from ..config import DEFAULT_CHECKPOINT_PATH
from ..ground_truth import GroundTruthPlan
from ..importer import import_selected_images
from ..organize import SELECTED_DIRNAME
from ..ranking import DEFAULT_STRATEGY_ID, AIModelParams, available_strategies, get_strategy
from ..review.session import ReviewSession
from ..review.thumbnails import detection_boxes_for, eye_keypoints_for, review_preview, review_thumbnail


class ReviewService:
    """Backend-facing service for the desktop review workflow."""

    def __init__(self, *, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_ANNOTATIONS_DB
        self.fields_config = load_annotation_fields(DEFAULT_ANNOTATIONS_CONFIG)
        self.store = AnnotationStore(self.db_path, fields_config=self.fields_config)
        self.session = ReviewSession(None, self.store)
        # Set by preview_ground_truth_import, consumed by
        # apply_ground_truth_import - see ground_truth.py's own docstring on
        # why apply must reuse the exact plan preview computed, never
        # re-walk the folders a second time.
        self._ground_truth_plan: GroundTruthPlan | None = None

    def open_folder(self, folder: str | Path) -> dict[str, Any]:
        self.session.open_folder(folder)
        recovered = self.session.reconcile_by_identity()
        state = self.load_session()
        state["recovered"] = recovered
        return state

    def load_session(self) -> dict[str, Any]:
        return self.session.as_dict()

    def loading_state(self) -> dict[str, Any]:
        with self.session._state_lock:  # noqa: SLF001 - session exposes no public accessor
            return dict(self.session._loading_state)

    def set_review_status(self, image_path: str, status: str, *, reason: str | None = None, reason_note: str | None = None) -> str:
        return self.session.set_review_status(image_path, status, reason=reason, reason_note=reason_note)

    def set_keep_percent(self, percent: float) -> dict[str, Any]:
        self.session.set_keep_percent(percent)
        return self.load_session()

    def set_burst_strategy(self, strategy_id: str) -> dict[str, Any]:
        """See ReviewSession.set_burst_strategy - which strategy's score
        Burst Analysis ranks each burst's own members by."""
        self.session.set_burst_strategy(strategy_id)
        return self.load_session()

    def apply_ai_suggestions(self, *, include_decided: bool = False) -> dict[str, Any]:
        """See ReviewSession.apply_ai_suggestions - the one call that lets
        the AI ranking influence review_status at all, and only because the
        photographer explicitly asked it to. include_decided=False (the
        default) only ever touches Neutral images; an already-decided
        image whose status disagrees with the AI's current suggestion is
        only overridden when the caller passes True, after getting
        confirmation from the photographer first (see MainWindow._apply_cutoff)."""
        result = self.session.apply_ai_suggestions(include_decided=include_decided)
        result["state"] = self.load_session()
        return result

    def apply_algorithm_suggestions(self, strategy_id: str, *, include_decided: bool = False) -> dict[str, Any]:
        """See ReviewSession.apply_algorithm_suggestions - the same bulk
        cutoff-application as apply_ai_suggestions above, but against
        `strategy_id`'s own suggestions rather than always the AI model.
        Records ALGORITHM decisions; a photographer's own Keep/Reject is
        never overwritten."""
        result = self.session.apply_algorithm_suggestions(strategy_id, include_decided=include_decided)
        result["state"] = self.load_session()
        return result

    def clear_algorithm_decisions(self) -> int:
        """See ReviewSession.clear_algorithm_decisions - drop every recorded
        algorithm cutoff, leaving User Decisions alone."""
        return self.session.clear_algorithm_decisions()

    def thumbnail_path(self, image_path: str, *, with_boxes: bool = False) -> Path | None:
        return review_thumbnail(image_path, with_boxes=with_boxes, strategy_id=self.session.burst_strategy)

    def preview_path(self, image_path: str) -> Path:
        return review_preview(image_path)

    def detection_boxes(self, image_path: str) -> dict[str, Any] | None:
        # Subject/bird-crop detection - the SAME upstream detection cache
        # (analyzer.detections.DetectionCache) regardless of which eye
        # detector/ranking strategy is currently selected, so this is never
        # scoped to a strategy the way eye_keypoints below is.
        return detection_boxes_for(image_path)

    def eye_keypoints(self, image_path: str, *, strategy_id: str | None = None) -> dict[str, Any] | None:
        """`strategy_id`'s own cached eye-detector result for `image_path` -
        defaults to the currently selected strategy (`ReviewSession.
        burst_strategy`/Color Source), but a caller (a future "Elements
        Source" picker - see the Loupe redesign brief) may request a
        DIFFERENT strategy's result explicitly.

        Structurally impossible to return a mismatched run now: `eyes.cache`
        keys its sidecar by (image, strategy) (see that module's own
        docstring), so this simply asks for the right one rather than
        reading whichever one happens to be cached and then checking it
        after the fact, the way this method used to. A strategy with no eye
        detector at all (the AI model - see `ranking.eye_detector_names`)
        has no sidecar under its own id, so this naturally returns None for
        it without any special-casing.
        """
        strategy_id = strategy_id or self.session.burst_strategy
        if not strategy_id:
            return None
        return eye_keypoints_for(image_path, strategy_id)

    def save_jpeg(self, image_path: str, destination_path: str | Path) -> Path:
        from ..analyzer.contactsheets import export_jpeg_bytes

        data = export_jpeg_bytes(image_path)
        destination = Path(destination_path)
        destination.write_bytes(data)
        return destination

    @staticmethod
    def ranking_strategies() -> list[Any]:
        """Every ranking strategy the Rank menu can offer (see
        `picklikeme.ranking`). Cheap - no model is constructed or imported."""
        return available_strategies()

    @staticmethod
    def ranking_params_class(strategy_id: str) -> type | None:
        """The parameter dataclass a strategy accepts, so the UI can generate
        its dialog from `params_class.specs()`. None for an unknown strategy
        or one that takes no parameters at all."""
        try:
            return getattr(get_strategy(strategy_id), "params_class", None)
        except ValueError:
            return None

    def rank_folder(
        self,
        *,
        strategy: str = DEFAULT_STRATEGY_ID,
        params: Any = None,
        checkpoint: str | Path = DEFAULT_CHECKPOINT_PATH,
        crop_birds: bool = True,
        device: str | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        force_preprocess: bool = False,
    ) -> dict[str, Any]:
        """Rank the open folder with the named strategy, then reload the session.

        `checkpoint`/`crop_birds` are the AI model's own parameters and are
        kept as explicit keywords rather than folded into `params`, so every
        existing caller of this method keeps working unchanged. A strategy
        that has richer parameters (Classic Vision) passes them as `params`
        instead; the reload afterwards is identical either way, because a
        ranking is a ranking however it was produced.

        `force_preprocess` reaches `preprocess.build_cache` unchanged - both
        `rank.rank_folder` and `ranking.classic`'s strategies already accept
        it, this was simply never threaded through the desktop layer before,
        which is exactly why a `CropCacheVersionMismatch` (see preprocess.py)
        used to leave a photographer stuck: the error dialog says "Pass
        --force to rebuild," but nothing in the desktop UI could actually do
        that. `main_window._rank_with_strategy`'s retry-on-mismatch prompt is
        what calls this with `force_preprocess=True`.
        """
        if self.session.input_folder is None:
            raise ValueError("Open a folder before ranking it.")
        if strategy == DEFAULT_STRATEGY_ID and params is None:
            params = AIModelParams(
                checkpoint=str(checkpoint), crop_birds=crop_birds, device=device
            )
        result = get_strategy(strategy).rank_folder(
            self.session.input_folder,
            params=params,
            on_stage=on_stage,
            on_progress=on_progress,
            force_preprocess=force_preprocess,
        )
        self.session.open_folder(self.session.input_folder)
        self.session.reconcile_by_identity()
        result["state"] = self.load_session()
        return result

    def arrange(self, *, dry_run: bool = False) -> dict[str, Any]:
        result = self.session.arrange(dry_run=dry_run)
        return {
            "dry_run": dry_run,
            "ranked": result.ranked,
            "selected": result.selected,
            "rejected": result.rejected,
            "moved": result.moved,
            "skipped": result.skipped,
            "errors": result.errors,
            "renamed": result.renamed,
            "selected_dir": str(result.selected_dir) if result.selected_dir else None,
            "rejected_dir": str(result.rejected_dir) if result.rejected_dir else None,
            "moves": {path: str(dest) for path, dest in result.moves.items()},
            "failures": list(result.failures[:20]),
        }

    def import_selected(
        self,
        destination_root: str | Path,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if self.session.input_folder is None:
            raise ValueError("Open a folder before importing from it.")
        result = import_selected_images(
            source_folder=self.session.input_folder,
            destination_root=destination_root,
            store=self.store,
            on_progress=on_progress,
        )
        result["state"] = self.load_session()
        return result

    def preview_ground_truth_import(
        self,
        *,
        root_folder: str | Path,
        keep_folders: list[str | Path] = (),
        reject_folders: list[str | Path] = (),
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """"Set User Decisions by Subfolders" - step 1. Walks `root_folder`
        once and reports counts only; nothing is written yet. See
        ground_truth.py's own module docstring - this is never an import,
        only `review_decisions` rows are ever touched, and only once
        `apply_ground_truth_import` is called with this exact plan still
        cached. Neutral is never folder-selected - every image under
        `root_folder` not inside a Keep/Reject subfolder becomes Neutral
        automatically."""
        from ..ground_truth import build_plan

        plan = build_plan(
            self.store, root_folder=root_folder, keep_folders=list(keep_folders),
            reject_folders=list(reject_folders), on_progress=on_progress,
        )
        self._ground_truth_plan = plan
        return {
            "totals": plan.totals(),
            "keep": {"will_change": plan.keep.will_change, "already_matching": plan.keep.already_matching, "skipped": len(plan.keep.skipped)},
            "reject": {"will_change": plan.reject.will_change, "already_matching": plan.reject.already_matching, "skipped": len(plan.reject.skipped)},
            "neutral": {"will_change": plan.neutral.will_change, "already_matching": plan.neutral.already_matching, "skipped": len(plan.neutral.skipped)},
            "conflicts": len(plan.conflicts),
        }

    def apply_ground_truth_import(self) -> dict[str, Any]:
        """"Set User Decisions by Subfolders" - step 2. Applies the plan
        `preview_ground_truth_import` computed last - never re-walks the
        folders, so what gets written is provably the same set the
        photographer's confirmation dialog just showed them."""
        from ..ground_truth import apply_plan

        if self._ground_truth_plan is None:
            raise ValueError("Preview the folders before applying - call preview_ground_truth_import first.")
        result = apply_plan(self.store, self._ground_truth_plan)
        self._ground_truth_plan = None
        result["state"] = self.load_session()
        return result

    def auto_crop(
        self,
        *,
        margin_percent: float = 0.0,
        on_progress: Callable[[int, int], None] | None = None,
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.session.input_folder is None:
            raise ValueError("Open a folder before running auto crop.")
        return generate_lightroom_crops(
            self.session.input_folder,
            margin_percent=margin_percent,
            on_progress=on_progress,
            image_paths=image_paths,
        )

    def organize_by_species(
        self,
        *,
        backend: str = "bioclip2",
        language: str = "en",
        min_confidence: float = 0.5,
        device: str | None = None,
        species_list_path: str | Path | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        analytics_db: str | Path | None = None,
    ) -> dict[str, Any]:
        """`backend` is any id `species.classifier.available_classifiers()`
        lists ("bioclip2" - the default - or "bioclip", the original
        BioCLIP kept for comparison). This method never imports a concrete
        classifier class - `build_classifier` is the only thing that
        resolves a backend id to a model, so this stays backend-agnostic
        the same way `ReviewService.rank_folder` stays ranking-strategy-
        agnostic.

        `device=None` (the default) auto-selects CUDA when available,
        exactly like every other model in this project (`resolve_torch_
        device`). Previously this method defaulted to a hardcoded "cpu"
        regardless of GPU availability - see docs/BioCLIP_Backend_
        Architecture_Review.md Section 7/§3 of the follow-up infrastructure
        work for how that was found and confirmed. Desktop's caller
        (`main_window._organize_by_species`) never overrode it, so species
        classification silently ran on CPU only, every time, until now.

        `species_list_path=None` (the default) uses the classifier's own
        built-in vocabulary. Any external text file (see `species.
        classifier.read_species_list`) is used directly, never copied into
        the project - `main_window._organize_by_species` already validates
        it (species count, or a clear error) before this is ever called, so
        a bad path reaching here would already be a UI bug, not an expected
        case; this method still lets whatever `BioClipSpeciesClassifier`
        itself raises propagate, rather than re-validating.
        """
        if self.session.input_folder is None:
            raise ValueError("Open a folder before organizing it by species.")

        from ..species.arrange import sanitize_species_folder_name
        from ..species.cache import DEFAULT_SPECIES_DB, SpeciesCache
        from ..analytics import DEFAULT_ANALYTICS_DB
        from ..species.classifier import build_classifier
        from ..species.experiment_capture import run_with_analytics
        from ..species.translations import localized_species_name

        logger.info(
            "Organize by Species: selected backend=%r (requested device=%r), species_list_path=%r",
            backend, device, species_list_path,
        )
        # device resolution (None -> auto-detect CUDA) happens inside
        # BioClipSpeciesClassifier itself now, not here - see its own
        # __init__ docstring - so every caller gets the right default even
        # if it forgets to resolve one, not just this call site.
        classifier = build_classifier(
            backend, min_confidence=min_confidence, device=device,
            species_list_path=str(species_list_path) if species_list_path else None,
        )
        cache = SpeciesCache(DEFAULT_SPECIES_DB)
        try:
            # run_with_analytics runs the exact same arrange_by_species pass
            # as before (identical file-moving behaviour), plus records a
            # full analytics experiment for it - see species/experiment_
            # capture.py's own docstring. An analytics failure never blocks
            # this (record_run's "never fatal" contract), so `run_id` may
            # legitimately be None; the arrange result itself is unaffected.
            # analytics_db=None (the default) uses the real shared database -
            # overridable so tests never write into it, matching rank_folder's
            # own analytics_db parameter.
            result, run_id, _metadata = run_with_analytics(
                self.session.input_folder,
                classifier,
                backend,
                cache,
                on_progress=on_progress,
                folder_name_fn=lambda species: sanitize_species_folder_name(
                    localized_species_name(species, language=language)
                ),
                species_list_path=str(species_list_path) if species_list_path else None,
                analytics_db=analytics_db or DEFAULT_ANALYTICS_DB,
            )
        finally:
            cache.close()
        return {
            "total": result.total,
            "classified": result.classified,
            "moved": result.moved,
            "skipped": result.skipped,
            "errors": result.errors,
            "species_counts": dict(result.species_counts),
            "experiment_id": run_id,
        }

    def close(self) -> None:
        self.store.close()

