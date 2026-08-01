"""Thin services that expose the existing review backend to the desktop UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..analyzer.annotation_config import DEFAULT_ANNOTATIONS_CONFIG, load_annotation_fields
from ..analyzer.annotations import DEFAULT_ANNOTATIONS_DB, AnnotationStore
from ..auto_crop import generate_lightroom_crops
from ..config import DEFAULT_CHECKPOINT_PATH
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

    def thumbnail_path(self, image_path: str, *, with_boxes: bool = False) -> Path | None:
        return review_thumbnail(image_path, with_boxes=with_boxes)

    def preview_path(self, image_path: str) -> Path:
        return review_preview(image_path)

    def detection_boxes(self, image_path: str) -> dict[str, Any] | None:
        return detection_boxes_for(image_path)

    def eye_keypoints(self, image_path: str) -> dict[str, Any] | None:
        return eye_keypoints_for(image_path)

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
    ) -> dict[str, Any]:
        """Rank the open folder with the named strategy, then reload the session.

        `checkpoint`/`crop_birds` are the AI model's own parameters and are
        kept as explicit keywords rather than folded into `params`, so every
        existing caller of this method keeps working unchanged. A strategy
        that has richer parameters (Classic Vision) passes them as `params`
        instead; the reload afterwards is identical either way, because a
        ranking is a ranking however it was produced.
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
        language: str = "en",
        min_confidence: float = 0.5,
        device: str = "cpu",
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if self.session.input_folder is None:
            raise ValueError("Open a folder before organizing it by species.")

        from ..species.arrange import arrange_by_species, sanitize_species_folder_name
        from ..species.cache import DEFAULT_SPECIES_DB, SpeciesCache
        from ..species.classifier import build_classifier
        from ..species.translations import localized_species_name

        classifier = build_classifier("bioclip2", min_confidence=min_confidence, device=device)
        cache = SpeciesCache(DEFAULT_SPECIES_DB)
        try:
            result = arrange_by_species(
                self.session.input_folder,
                classifier,
                cache,
                on_progress=on_progress,
                folder_name_fn=lambda species: sanitize_species_folder_name(
                    localized_species_name(species, language=language)
                ),
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
        }

    def close(self) -> None:
        self.store.close()

