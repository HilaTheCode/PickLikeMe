"""What the photographer is reviewing, and what would happen if they filed it.

All of the review application's decisions live here: which images exist, what
the AI ranking merely suggests, which review status the photographer has
actually given each image, and what `Arrange` would move where. No HTTP and no
SQL beyond the store's own methods, so every rule below is directly testable.

Three ideas do the work:

- **The gallery is the union of the ranking and the folder.** An image present
  on disk but absent from the ranking still appears, because the alternative is
  a photograph silently missing from a review of its own shoot.
- **Algorithm results and the photographer's User Decision are completely
  separate, on disk as well as in memory.** A score, a rank, and the "suggests
  Keep/Reject" hint derived from them are read-only information a model
  produced. Every image's `user_decision` is exactly one of Keep, Reject or
  Undecided - the photographer's own, independent verdict - and nothing about
  the ranking ever produces one. A decision row carries the `source` that
  wrote it (see `analyzer.annotations.REVIEW_DECISION_SOURCES`), so an
  algorithm cutoff recorded by "Apply Cutoff" can never later be mistaken for
  a photographer's click; only DECISION_SOURCE_USER rows are User Decisions.
  Clearing a decision (one image, or a whole bulk selection) always lands on
  Undecided, never silently on whatever the model would have picked.
- **Only an explicit user Keep/Reject files anything.** `arrange()` reads
  `keep_paths`/`reject_paths`, which read `user_decision` alone; an Undecided
  image - ranked highly, not ranked at all, or carrying a recorded algorithm
  cutoff - is never moved, since "the photographer hasn't looked at it yet" is
  not a basis to sort it.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..analyzer.annotations import (
    DECISION_SOURCE_ALGORITHM,
    DECISION_SOURCE_USER,
    REVIEW_KEEP,
    REVIEW_REASON_OTHER,
    REVIEW_REJECT,
    AnnotationStore,
)
from ..burst_analysis import BurstInfo, ScoredImage, analyze_bursts
from ..identity import IdentityUnavailable
from ..organize import (
    DEFAULT_SELECTION_PERCENTAGE,
    OrganizeResult,
    organize_by_decision,
    selection_count,
    validate_selection_percentage,
)
from ..sidecar import AI_STRATEGY_ID, ranking_path, read_run_metadata, rewrite_ranking_paths

logger = logging.getLogger(__name__)

# The photographer's review status - always exactly one of these three,
# completely independent of the AI ranking. Reuses REVIEW_KEEP/REVIEW_REJECT
# (the values AnnotationStore's `review_decisions` table already stores) so
# writing one is a direct pass-through; REVIEW_STATUS_NEUTRAL has no row in
# that table at all - "no decision recorded" *is* Neutral, not a fourth state.
REVIEW_STATUS_KEEP = REVIEW_KEEP
REVIEW_STATUS_REJECT = REVIEW_REJECT
REVIEW_STATUS_NEUTRAL = "neutral"
REVIEW_STATUSES: frozenset[str] = frozenset({REVIEW_STATUS_KEEP, REVIEW_STATUS_REJECT, REVIEW_STATUS_NEUTRAL})


class InvalidReviewStatus(ValueError):
    """Raised for a review status outside REVIEW_STATUSES."""


@dataclass
class ReviewImage:
    """One row of the gallery."""

    image_path: str
    filename: str
    # Every analysis module's result for this image, keyed by strategy id -
    # {"ai-model": {"score": .., "rank": ..}, "classic-vision": {"score": .., "rank": ..}}.
    # Score and rank always belong together as properties of ONE strategy;
    # there is deliberately no single global rank anywhere in this class -
    # "ranked" only ever means "ranked BY something", and `rank` always means
    # "this strategy's position in its own ordering". Independent results
    # that coexist: running one module never clears another's, and a module
    # PeakPic no longer ships still displays whatever it left behind. Purely
    # informational, read-only, never written by this application beyond
    # `_load_ranked` populating it from disk.
    ranking_results: dict[str, dict] = field(default_factory=dict)
    # Why a strategy did NOT score this image, keyed by strategy id -
    # {"classic-vision": "NO_VISIBLE_EYE"}. An image can be filtered by one
    # strategy and scored by another at the same time, so this is a sibling
    # dict to ranking_results rather than a single flag: absence from BOTH
    # dicts for a strategy means that strategy has simply never run on this
    # folder, while presence here with no matching ranking_results entry
    # means it ran and explicitly excluded this image (see
    # `ranking.classic`'s filter phase and `sidecar.discover_filter_reports`).
    # Purely informational, exactly like ranking_results - never written by
    # this application beyond `_load_ranked` populating it from disk.
    filter_reasons: dict[str, str] = field(default_factory=dict)
    # A strategy's raw, per-metric measurements for this image, keyed by
    # strategy id -> {metric_name: value} - e.g.
    # {"classic-vision": {"eye_sharpness": 0.8, "subject_size": 0.02, ...}}.
    # The breakdown behind a single combined score (see
    # `ranking.classic.write_metrics_report`), for a diagnostics display -
    # "why did this image rank where it did" needs the individual numbers a
    # weighted sum otherwise hides. Purely informational, exactly like
    # ranking_results/filter_reasons.
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # The file's own EXIF capture date/time (ISO-8601), or None if it has
    # none - a sort key, nothing more. See AnnotationStore.capture_timestamp_of.
    captured_at: str | None = None
    # The subject category already recorded for this image (see
    # bird_crop.DETECTION_CATEGORIES: "bird", "mammal", "human", ...), or
    # None if nothing was ever detected/recorded. Read-only, exactly like
    # score/rank - structured metadata for filtering/search/statistics, never
    # written by this application. See thumbnails.detected_category_for.
    detected_category: str | None = None
    # The stored verdict for this image: REVIEW_KEEP, REVIEW_REJECT, or None.
    # None *is* Undecided - see user_decision - not "no opinion yet from
    # somewhere else". Read together with `decision_source`, ALWAYS: a
    # decision string on its own does not say who made it, and only
    # DECISION_SOURCE_USER means the photographer did (see
    # annotations.REVIEW_DECISION_SOURCES). Meaningless without a decision,
    # and always cleared alongside one.
    decision: str | None = None
    # Who recorded `decision` - DECISION_SOURCE_USER or
    # DECISION_SOURCE_ALGORITHM. None exactly when `decision` is None.
    decision_source: str | None = None
    reason: str | None = None
    # Free text, only meaningful alongside REVIEW_REASON_OTHER - see
    # AnnotationStore.set_review_decision.
    reason_note: str | None = None
    # Set when the ranking lists an image that is no longer on disk. Shown as a
    # placeholder rather than dropped, so the photographer can see the gap.
    missing_file: bool = False

    @property
    def score(self) -> float | None:
        """The AI model's own score - never "whichever module scored this
        last". Computed from `ranking_results`, not stored separately, so
        there is exactly one place per image where a strategy's score can
        live. The AI-suggestion threshold, `cut`, and every agreement
        statistic below are defined against this specific strategy's
        ordering, which is why it keeps its own name instead of becoming
        "the most recent score" - see the class docstring.
        """
        return (self.ranking_results.get(AI_STRATEGY_ID) or {}).get("score")

    @property
    def rank(self) -> int | None:
        """The AI model's own rank - see `score`."""
        return (self.ranking_results.get(AI_STRATEGY_ID) or {}).get("rank")

    @property
    def is_ranked(self) -> bool:
        return self.score is not None

    def score_for(self, strategy_id: str) -> float | None:
        """A specific strategy's score, or None if it never scored this
        image - see `score` for why the AI model's own score stays a
        separate, fixed-name property rather than being folded into this."""
        return (self.ranking_results.get(strategy_id) or {}).get("score")

    @property
    def user_decision(self) -> str:
        """The photographer's OWN verdict - KEEP, REJECT or UNDECIDED (see
        `review.user_decision`). The single source of truth for User Decision
        coloring, the counts, and `arrange()`.

        A stored decision only counts here when DECISION_SOURCE_USER recorded
        it. An algorithm's cutoff (DECISION_SOURCE_ALGORITHM - see
        `apply_algorithm_suggestions`) leaves this UNDECIDED no matter how
        confident the score behind it was: an image the photographer has
        never looked at has no verdict, and inventing one from a ranking is
        the exact confusion this property exists to make impossible.
        """
        from .user_decision import UNDECIDED, normalize

        if self.decision_source != DECISION_SOURCE_USER:
            return UNDECIDED
        return normalize(self.decision)

    @property
    def is_decided(self) -> bool:
        """True only for an explicit user Keep or Reject - what `arrange()`
        and every "only the images I actually reviewed" caller tests."""
        from .user_decision import is_decided

        return is_decided(self.user_decision)

    @property
    def algorithm_decision(self) -> str | None:
        """A recorded algorithm cutoff for this image, if one was ever
        applied - purely informational, never a user decision. None when the
        stored decision (if any) is the photographer's own."""
        if self.decision_source != DECISION_SOURCE_ALGORITHM:
            return None
        return self.decision

    @property
    def review_status(self) -> str:
        """Legacy spelling of `user_decision` - "keep"/"reject"/"neutral",
        the vocabulary the web review page, the desktop status filters and
        the K/R/N buttons already speak. Derived from `user_decision`, so an
        algorithm-sourced row reads as Neutral here too and no caller can
        reach a Keep the photographer never made by using the older name.
        """
        from .user_decision import UNDECIDED

        decision = self.user_decision
        return REVIEW_STATUS_NEUTRAL if decision == UNDECIDED else decision

    def as_dict(
        self, ai_suggestion: str | None, burst: "BurstInfo | None" = None, algorithm_suggestion: str | None = None
    ) -> dict:
        return {
            "image_path": self.image_path,
            "filename": self.filename,
            # Kept at the top level for backward compatibility (the web
            # review UI and existing tests read these directly) - always the
            # AI model's own score/rank, exactly what they were before
            # ranking_results existed. See the `score`/`rank` properties.
            "score": self.score,
            "rank": self.rank,
            "ranking_results": self.ranking_results,
            "filter_reasons": self.filter_reasons,
            "metrics": self.metrics,
            "captured_at": self.captured_at,
            "detected_category": self.detected_category,
            "review_status": self.review_status,
            # The same verdict in the explicit three-state vocabulary -
            # "keep"/"reject"/"undecided", never None, never inferred. The
            # desktop Grid's User Decision coloring reads THIS, so "no row in
            # review_decisions" reaches the UI as a value it has to handle
            # rather than as an absence it might default to something else.
            "user_decision": self.user_decision,
            # Who recorded the stored decision, or None if there is none -
            # see annotations.REVIEW_DECISION_SOURCES.
            "decision_source": self.decision_source,
            # A recorded algorithm cutoff for this image, if one was applied
            # (informational - never a user decision, never organized).
            "algorithm_decision": self.algorithm_decision,
            # What the AI ranking alone would recommend at the current
            # threshold - "keep"/"reject", or None if unranked. Informational
            # only; see ReviewSession._ai_suggestions.
            "ai_suggestion": ai_suggestion,
            # The same kind of suggestion, but for whichever strategy is
            # currently selected (ReviewSession.burst_strategy - the same
            # selection Burst Analysis and the desktop Color Source picker
            # already share, see burst_strategy's own docstring). Equal to
            # ai_suggestion whenever that strategy IS the AI model - a
            # distinct field regardless, so a caller never has to guess
            # which one it got. See ReviewSession.suggestions_for.
            "algorithm_suggestion": algorithm_suggestion,
            "reason": self.reason,
            "reason_note": self.reason_note,
            "missing_file": self.missing_file,
            # Burst Analysis's own read-only output (see
            # ReviewSession.burst_info) - every image gets one, even a burst
            # of one, so these are never None once a session has loaded.
            "burst_id": burst.burst_id if burst else None,
            "burst_size": burst.burst_size if burst else 1,
            "burst_rank": burst.burst_rank if burst else 1,
            "burst_best": burst.burst_best if burst else True,
        }


class ReviewSession:
    """The reviewed folder, its ranking, and the photographer's own verdicts."""

    def __init__(
        self,
        input_folder: str | Path | None,
        store: AnnotationStore,
        *,
        ranking_file: str | Path | None = None,
        keep_percent: float = DEFAULT_SELECTION_PERCENTAGE,
    ):
        self.store = store
        self.keep_percent = validate_selection_percentage(keep_percent)
        # Which strategy's score Burst Analysis ranks each burst's members
        # by (see burst_info) - the AI model until the photographer picks a
        # different one (the desktop Gallery ties this to its own Color
        # Source selector, so "the selected ranking strategy" only ever has
        # one meaning across the app - see MainWindow._on_color_source_changed).
        self.burst_strategy = AI_STRATEGY_ID
        self._state_lock = threading.RLock()
        self._background_lock = threading.Lock()
        self._loading_generation = 0
        self._loading_thread: threading.Thread | None = None
        self._loading_state: dict = {}
        if input_folder is None:
            # `picklikeme review` with no --input at all starts here: an
            # empty, folder-less gallery, so the page still has something to
            # render and the photographer picks a folder from inside it (see
            # review.server's /api/review/open-folder) rather than the
            # command failing to start.
            self._clear()
        else:
            self.open_folder(input_folder, ranking_file=ranking_file)

    def _clear(self) -> None:
        with self._state_lock:
            self.input_folder = None
            self.ranking_file = None
            self.run_metadata: dict = {}
            self.warnings: list[str] = ['No folder open yet. Click "Open Folder…" above to choose one.']
            self.images: list[ReviewImage] = []
            self._loading_generation += 1
            self._loading_state = {"stage": "idle", "message": "Ready", "percent": 100, "complete": True}

    # -- loading ------------------------------------------------------------

    def open_folder(self, input_folder: str | Path, *, ranking_file: str | Path | None = None) -> None:
        """Point this session at a different folder - including one that has
        never been ranked at all. `_load_ranked` already treats a missing
        ranking as "every image unranked" rather than an error (see below);
        combined with every image always starting Neutral, that is exactly
        what browsing a fresh, un-ranked shoot needs - nothing but Keep/Reject
        to sort it, with no AI opinion in the mix at all.

        Manual decisions already on record for images under the new folder
        are picked up the same way any reload finds them, by path (see
        `_apply_decisions`) - opening a folder is not a reason to lose a
        photographer's prior work in it.
        """
        self.input_folder = Path(input_folder).resolve()
        self.ranking_file = Path(ranking_file) if ranking_file else ranking_path(self.input_folder)
        self.run_metadata = read_run_metadata(self.input_folder)
        self.burst_strategy = self.latest_run_strategy()
        with self._state_lock:
            self.warnings: list[str] = []
        self.load()

    def latest_run_strategy(self) -> str:
        """"Algorithm Ran Last": whichever strategy actually produced this
        folder's most recent completed ranking run, not a hard-coded
        constant - the ONE centralized definition of that concept, used both
        to seed `burst_strategy` when a folder opens (see `open_folder`
        above) and, on demand, by the desktop Color Source selector's own
        explicit "Algorithm Ran Last" option (see `main_window.
        color_source_options`) so a photographer can return to "whichever
        is latest" after manually picking a specific strategy, without
        needing to know its name.

        Before this existed, `burst_strategy` (and therefore the desktop
        Color Source selector, Grid coloring, filtering, cutoff, and Burst
        ranking - everything downstream of it, see this attribute's own
        docstring) always started at `AI_STRATEGY_ID` regardless of what had
        actually been run on the folder. On a folder ranked only by Classic
        Vision strategies (the AI model never run there at all - no
        `ranking.csv`), that meant the Grid opened showing "no algorithm
        suggestion" for literally every image by default - not because
        nothing was detected, but because the strategy being displayed
        simply had no data - while the Loupe's Elements/Boxes overlay (see
        `review.thumbnails.eye_keypoints_for`) reads a detector-result cache
        that is not gated by `burst_strategy` at all, so it could still show
        real detections for the same images. Two different parts of the app
        reading two different, disconnected notions of "the result" for the
        same folder - this is the fix.

        `run.json`'s own `strategy` field (see `sidecar.write_run_metadata`,
        written by every strategy's own `rank_folder`) already records
        exactly this - it just was not being read as a decision, only shown
        as provenance text. Validated against `discover_strategy_rankings`
        before trusting it: `run.json` could in principle name a strategy
        whose CSV was later deleted, and an explicit fallback to
        `AI_STRATEGY_ID` (still correct for a genuinely fresh, never-ranked
        folder - every image legitimately starts Neutral there) is safer
        than trusting stale provenance blindly.
        """
        from ..sidecar import discover_strategy_rankings

        last_run_strategy = self.run_metadata.get("strategy")
        if last_run_strategy and last_run_strategy in discover_strategy_rankings(self.input_folder):
            return last_run_strategy
        return AI_STRATEGY_ID

    def load(self) -> None:
        """(Re)build the gallery from the ranking, the folder, and the store.

        The initial view is exposed immediately and the rest of the metadata is
        filled in progressively in the background so the UI can start showing
        thumbnails without waiting for the full load to finish.
        """
        if self.input_folder is None:
            self._clear()
            return

        generation = self._loading_generation + 1
        self._loading_generation = generation
        self._set_loading_state("scanning", "Scanning images…", 5, False)

        ranked = self._load_ranked()
        on_disk = self._enumerate_folder()
        images: list[ReviewImage] = []
        seen: set[str] = set()
        for image in ranked:
            images.append(image)
            seen.add(_key(image.image_path))
        for path in on_disk:
            if _key(str(path)) in seen:
                continue
            images.append(ReviewImage(image_path=str(path), filename=path.name))
        images.sort(key=lambda i: (i.score is None, -(i.score or 0.0), i.filename))
        with self._state_lock:
            self.images = images

        self._apply_filter_reasons()
        self._apply_metrics()
        self._apply_decisions()

        if self._loading_thread is not None and self._loading_thread.is_alive():
            self._loading_thread.join(timeout=0)

        self._loading_thread = threading.Thread(target=self._background_load, args=(generation,), daemon=True)
        self._loading_thread.start()
        # Give the first metadata pass a brief chance to finish before the
        # caller returns, so tests and the initial page render still see
        # category/capture values even though the UI itself stays responsive.
        self._loading_thread.join(timeout=0.15)

    def _set_loading_state(self, stage: str, message: str, percent: int, complete: bool) -> None:
        with self._state_lock:
            self._loading_state = {
                "stage": stage,
                "message": message,
                "percent": percent,
                "complete": complete,
            }

    def _background_load(self, generation: int) -> None:
        try:
            with self._background_lock:
                if generation != self._loading_generation:
                    return
                metadata_store = AnnotationStore(self.store.db_path, self.store.fields_config)
                try:
                    self._set_loading_state("loading-metadata", "Loading ranking and review metadata…", 20, False)
                    self._apply_decisions(metadata_store)
                    if generation != self._loading_generation:
                        return

                    from .thumbnails import detected_category_for

                    with self._state_lock:
                        images = list(self.images)
                    for index, image in enumerate(images):
                        if generation != self._loading_generation:
                            return
                        if image.missing_file:
                            continue
                        image.captured_at = metadata_store.capture_timestamp_of(image.image_path)
                        # detected_category_for's own cache is a per-image file on
                        # disk (a JSON sidecar from preprocessing) - real I/O that
                        # must not be paid for on every single load. Memoised in the
                        # store exactly like captured_at (see
                        # AnnotationStore.detected_category_of), so only the first
                        # load after a file last changed re-reads it.
                        image.detected_category = metadata_store.detected_category_of(
                            image.image_path, detected_category_for
                        )
                        if index % 25 == 0 or index == len(images) - 1:
                            with self._state_lock:
                                self.images = list(images)
                            percent = 20 + int(70 * (index + 1) / max(1, len(images)))
                            self._set_loading_state("loading-categories", "Loading categories…", percent, False)

                    if generation != self._loading_generation:
                        return
                    self._set_loading_state("ready", "Ready", 100, True)
                finally:
                    metadata_store.close()
        except Exception as exc:  # noqa: BLE001 - a background failure must not break the page
            logger.exception("Could not finish progressive metadata load for %s", self.input_folder)
            with self._state_lock:
                self.warnings.append(f"Could not finish loading metadata: {exc}")
            self._set_loading_state("error", "Could not finish loading metadata", 100, False)

    def _read_ranking(self, path: Path) -> dict[str, tuple[float, int, str]]:
        """One analysis module's scores, keyed by resolved image path.

        Returns `{}` for anything unreadable rather than raising: one broken
        module's output must not take down a review session, nor prevent the
        other modules' results from being shown.
        """
        from ..analyzer.io import load_ranking

        try:
            ranking = load_ranking(path)
        except Exception as exc:  # noqa: BLE001 - a broken ranking must not end the session
            logger.warning("Could not read %s: %s", path, exc)
            with self._state_lock:
                self.warnings.append(f"Could not read {path.name} ({exc}).")
            return {}
        with self._state_lock:
            self.warnings.extend(ranking.warnings)
        return {
            _key(image.image_path): (image.score, image.rank, image.image_path)
            for image in ranking.images
        }

    def _load_ranked(self) -> list[ReviewImage]:
        """Build the gallery's ranked rows from EVERY analysis module that has
        scored this folder, not just the trained model.

        Each module owns its own CSV (see `sidecar.discover_strategy_rankings`)
        and they are merged per image into `ranking_results`, so a folder
        scored by both carries both results and neither run erased the
        other. `ReviewImage.score`/`.rank` read the AI model's own entry out
        of that dict - there is no separate flat field to keep in sync, which
        is the whole point: one place per image where a strategy's score and
        rank live together.

        An image that only a non-AI module scored still appears here with
        that module's entry in `ranking_results` and no "ai-model" entry at
        all; it shows as Unranked by the AI, which is exactly what it is.
        """
        from ..sidecar import discover_strategy_rankings

        rankings = discover_strategy_rankings(self.input_folder)
        if not rankings:
            with self._state_lock:
                self.warnings.append(
                    f"No analysis results in {self.input_folder}; every image starts Neutral."
                )
            return []

        by_key: dict[str, ReviewImage] = {}
        for strategy_id, path in rankings.items():
            for key, (score, rank, image_path) in self._read_ranking(path).items():
                image = by_key.get(key)
                if image is None:
                    image = ReviewImage(
                        image_path=image_path,
                        filename=Path(image_path).name,
                        missing_file=not Path(image_path).is_file(),
                    )
                    by_key[key] = image
                image.ranking_results[strategy_id] = {"score": score, "rank": rank}
        return list(by_key.values())

    def _enumerate_folder(self) -> list[Path]:
        """Images actually on disk, including any already filed by a previous
        arrange - re-reviewing an organized shoot has to see its own output."""
        from ..analyzer.io import enumerate_ground_truth
        from ..sidecar import SIDECAR_DIRNAME

        try:
            found = enumerate_ground_truth(self.input_folder)
        except FileNotFoundError as exc:
            with self._state_lock:
                self.warnings.append(str(exc))
            return []
        return [p for p in found if SIDECAR_DIRNAME not in p.parts]

    def _apply_filter_reasons(self) -> None:
        """Attach each analysis module's filter verdict (if any) to the
        matching gallery image, by path - mirrors `_apply_decisions` below.

        Runs for every image, not only the ones a ranking CSV produced: an
        image every registered strategy filtered out (so no module's CSV
        ever named it) still reaches the gallery through
        `_enumerate_folder`'s on-disk fallback, and it deserves the same
        "why was this skipped" a partially-filtered image gets.
        """
        from ..sidecar import discover_filter_reports

        reports = discover_filter_reports(self.input_folder)
        if not reports:
            return
        by_key: dict[str, dict[str, str]] = {}
        for strategy_id, images in reports.items():
            for image_path, reason in images.items():
                by_key.setdefault(_key(image_path), {})[strategy_id] = reason
        for image in self.images:
            reasons = by_key.get(_key(image.image_path))
            if reasons:
                image.filter_reasons = reasons

    def _apply_metrics(self) -> None:
        """Attach each analysis module's raw per-metric measurements (if
        any) to the matching gallery image, by path - mirrors
        `_apply_filter_reasons` above."""
        from ..sidecar import discover_metric_reports

        reports = discover_metric_reports(self.input_folder)
        if not reports:
            return
        by_key: dict[str, dict[str, dict[str, float]]] = {}
        for strategy_id, metrics in reports.items():
            for image_path, values in metrics.items():
                by_key.setdefault(_key(image_path), {})[strategy_id] = values
        for image in self.images:
            values = by_key.get(_key(image.image_path))
            if values:
                image.metrics = values

    def _apply_decisions(self, store: AnnotationStore | None = None) -> None:
        """Attach stored manual decisions to the gallery.

        Matched on path, which is one query for the whole session. Content
        identity is the authority and is what a write uses, but resolving it
        for every image just to discover most were never decided would cost
        minutes on a cold cache (see identity.py). `reconcile_by_identity()`
        closes the gap for the few rows a path match misses.
        """
        store = store or self.store
        rows = store.review_decisions()

        def _row(row: dict) -> tuple[str, str, str | None, str | None]:
            # `source` is read with an explicit fallback rather than trusted
            # blind: a row from a database written before the column existed
            # has no origin on record, and DECISION_SOURCE_USER is the
            # conservative reading (see AnnotationStore's own migration).
            return (
                row["decision"],
                row.get("source") or DECISION_SOURCE_USER,
                row.get("reason"),
                row.get("reason_note"),
            )

        empty: tuple[None, None, None, None] = (None, None, None, None)
        by_hash = {row["image_hash"]: _row(row) for row in rows}
        by_path = {_key(row["image_path"]): _row(row) for row in rows}
        matched: set[str] = set()
        with self._state_lock:
            self._decisions_by_hash = by_hash
        for image in self.images:
            decision, source, reason, reason_note = by_path.get(_key(image.image_path), empty)
            image.decision = decision
            image.decision_source = source
            image.reason = reason
            image.reason_note = reason_note
            # A decision matched to a path whose file is gone is not really
            # matched: the image it describes has moved, and its new copy is
            # elsewhere in the gallery waiting to be found by identity.
            if decision is not None and not image.missing_file:
                matched.add(_key(image.image_path))
        with self._state_lock:
            self._unmatched_decisions = len(rows) - len(matched)

    def reconcile_by_identity(self, store: AnnotationStore | None = None) -> int:
        """Recover decisions for images whose file has moved since it was
        reviewed, by falling back to content identity.

        Only runs when a stored decision failed to match any gallery image by
        path, and only hashes gallery images that are currently undecided - so
        the common case (nothing moved) costs nothing at all. Returns the
        number of decisions recovered.
        """
        if not getattr(self, "_unmatched_decisions", 0):
            return 0
        recovered = 0
        for image in self.images:
            if image.decision is not None or image.missing_file:
                continue
            try:
                digest = (store or self.store).identity_of(image.image_path)
            except IdentityUnavailable:
                continue
            decision, source, reason, reason_note = self._decisions_by_hash.get(
                digest, (None, None, None, None)
            )
            if decision is not None:
                image.decision = decision
                image.decision_source = source
                image.reason = reason
                image.reason_note = reason_note
                recovered += 1
        if recovered:
            logger.info("Recovered %d review decision(s) by content identity", recovered)
        return recovered

    @property
    def folder_missing(self) -> bool:
        """True when a folder is set but can no longer be found there -
        moved, renamed, or its drive letter changed. Distinct from no folder
        ever having been opened (`input_folder is None`, the ordinary empty-
        gallery start state) - this is the one that needs a photographer's
        attention (see relocate_folder)."""
        return self.input_folder is not None and not self.input_folder.is_dir()

    def relocate_folder(self, new_folder: str | Path) -> dict:
        """The folder this session was reviewing can no longer be found at
        its old location, and the photographer has pointed at where it went
        instead - moved, renamed, or found on a drive that changed letter.

        Assuming the same relative layout survived the move (the ordinary
        case: this is the exact same shoot, just found somewhere else), every
        ranked image's stored path is repointed at `new_folder` automatically
        by substituting the old root for the new one - nothing new needs
        deciding by hand, and this is transparent to the photographer, who
        only ever sees the gallery load correctly again.

        Reuses exactly the machinery `arrange()` already uses to keep the
        ranking and the store's review decisions correct after files move
        (`rewrite_ranking_paths`, `repoint_review_decisions`), plus
        `reconcile_by_identity` as the same fallback it already is for
        anything the prefix substitution below could not map - a file
        renamed during the move, or one never in the ranking to begin with
        (an "extra" image the folder enumeration found, not the CSV).

        The ranking CSV itself is read from `new_folder` (via `ranking_path`),
        not from `self.ranking_file` (the OLD, now-unreachable location) -
        the CSV moved along with everything else, so it now lives at the new
        location, still full of the OLD absolute paths that need rewriting.
        """
        old_folder = self.input_folder
        new_folder = Path(new_folder).resolve()

        moves: dict[str, str] = {}
        if old_folder is not None:
            new_ranking_file = ranking_path(new_folder)
            if new_ranking_file.is_file():
                from ..analyzer.io import load_ranking

                try:
                    ranking = load_ranking(new_ranking_file)
                except Exception as exc:  # noqa: BLE001 - a broken ranking must not block relocating
                    logger.warning("Could not read %s while relocating: %s", new_ranking_file, exc)
                else:
                    for image in ranking.images:
                        try:
                            relative = Path(image.image_path).resolve().relative_to(old_folder)
                        except (OSError, ValueError):
                            continue
                        candidate = new_folder / relative
                        if candidate.is_file():
                            moves[image.image_path] = str(candidate)

        if moves:
            rewrite_ranking_paths(new_folder, moves)
            self.store.repoint_review_decisions(moves)

        self.open_folder(new_folder)
        recovered = self.reconcile_by_identity()
        return {"relocated": len(moves), "recovered": recovered}

    # -- AI metadata (read-only) ---------------------------------------------

    def set_keep_percent(self, percent: float) -> float:
        """The AI suggestion threshold - what fraction of RANKED images the
        model's own ordering alone would flag as "keep". Purely informational
        (see _ai_suggestions): moving this never changes anyone's
        review_status, only what the AI-suggestion hint next to each image
        says, unless the photographer explicitly applies it (see
        apply_ai_suggestions)."""
        self.keep_percent = validate_selection_percentage(percent)
        return self.keep_percent

    def set_burst_strategy(self, strategy_id: str) -> str:
        """Which strategy's score Burst Analysis ranks bursts by from now
        on - purely a display choice, exactly like keep_percent, and never
        persisted: the next `as_dict()` call recomputes burst_info() fresh."""
        self.burst_strategy = strategy_id
        return self.burst_strategy

    def burst_info(self) -> dict[str, BurstInfo]:
        """Every image's burst membership and in-burst rank, from
        `burst_analysis.analyze_bursts` - see that module for why it is
        handed nothing but (path, captured_at, score) triples, never told
        which strategy produced the score or anything else about this
        session. Recomputed on every call rather than cached: it is cheap
        (pure Python over data already in memory) and must never go stale
        after set_burst_strategy or a new ranking changes the scores it
        reads from `self.images`.
        """
        scored = [
            ScoredImage(image.image_path, image.captured_at, image.score_for(self.burst_strategy))
            for image in self.images
        ]
        return analyze_bursts(scored)

    @property
    def cut(self) -> int:
        """How many ranked images the AI threshold alone would flag as keep."""
        return selection_count(sum(1 for i in self.images if i.is_ranked), self.keep_percent)

    def _ai_suggestions(self) -> dict[str, str | None]:
        """What the AI ranking alone would recommend for each image at the
        current threshold - "keep"/"reject" for a ranked image, None for one
        the model never scored. Never written anywhere, never fed into
        review_status or arrange() by itself - purely the hint shown next to
        the photographer's own, independent verdict.

        Deliberately still AI-specific (not `suggestions_for(AI_STRATEGY_ID)`
        below, even though the two would compute the same thing today) -
        this backs `agreement_stats`/`disagreements`/`apply_ai_suggestions`/
        `evaluation_report.py`, which are their own, explicitly AI-focused
        feature (evaluating the trained model against human judgement over
        time) and were not asked to be generalized. `suggestions_for` is the
        one used by the filter UI's now-per-strategy conflict filters - see
        its own docstring."""
        return self.suggestions_for(AI_STRATEGY_ID)

    def suggestions_for(self, strategy_id: str) -> dict[str, str | None]:
        """What `strategy_id` alone would recommend for each image at the
        current keep-percent threshold - "keep"/"reject" for an image that
        strategy scored, None for one it never scored (never filtered into
        a false "reject" - see `SubjectFilter`/`EyeFilter`'s own NO_SUBJECT/
        NO_VISIBLE_EYE/etc. reasons for why an image has no score to begin
        with). The general form of `_ai_suggestions` - see that method's own
        docstring for why the AI-specific evaluation features keep using a
        fixed AI-only version rather than this one.

        Same threshold (`self.keep_percent`) for every strategy - there is
        only one photographer-facing "how selective" setting, applied to
        whichever strategy's own score ordering is being asked about, not a
        second per-strategy setting to keep in sync.
        """
        scored = [image for image in self.images if image.score_for(strategy_id) is not None]
        cut = selection_count(len(scored), self.keep_percent)
        ranked = sorted(scored, key=lambda image: image.score_for(strategy_id), reverse=True)
        suggestions: dict[str, str | None] = {image.image_path: None for image in self.images}
        for position, image in enumerate(ranked):
            suggestions[image.image_path] = REVIEW_STATUS_KEEP if position < cut else REVIEW_STATUS_REJECT
        return suggestions

    def agreement_stats(self) -> dict:
        """How often the photographer's own review status agrees with the
        AI's suggestion, among images where a real comparison is possible -
        ranked AND actually decided. Neutral is "no opinion formed yet", not
        a disagreement, so it is excluded from the comparison entirely, the
        same way it is excluded from `_ai_suggestions`'s own reasoning.

        Purely informational - for evaluating the model against real human
        judgement over time, e.g. across successive training runs. Never
        read by review_status, ai_suggestion, or arrange(). The one place
        this comparison is computed at all - the panel and
        evaluation_report.py both read this same dict, so the two can never
        disagree about a number.

        The full 2x2 confusion matrix (AI Keep/Reject x User Keep/Reject)
        and precision/recall/F1 treat Keep as the positive class and the
        photographer's own review_status as ground truth: precision is "of
        what the AI suggested keeping, how much did the photographer also
        keep"; recall is "of what the photographer kept, how much did the
        AI also suggest keeping".
        """
        suggestions = self._ai_suggestions()
        compared = 0
        ai_keep_user_keep = 0
        ai_keep_user_reject = 0
        ai_reject_user_keep = 0
        ai_reject_user_reject = 0
        for image in self.images:
            suggestion = suggestions.get(image.image_path)
            if suggestion is None or image.review_status == REVIEW_STATUS_NEUTRAL:
                continue
            compared += 1
            if suggestion == REVIEW_STATUS_KEEP and image.review_status == REVIEW_STATUS_KEEP:
                ai_keep_user_keep += 1
            elif suggestion == REVIEW_STATUS_KEEP:
                ai_keep_user_reject += 1
            elif image.review_status == REVIEW_STATUS_KEEP:
                ai_reject_user_keep += 1
            else:
                ai_reject_user_reject += 1

        agree = ai_keep_user_keep + ai_reject_user_reject
        disagree = ai_keep_user_reject + ai_reject_user_keep
        predicted_keep = ai_keep_user_keep + ai_keep_user_reject
        actual_keep = ai_keep_user_keep + ai_reject_user_keep
        precision = ai_keep_user_keep / predicted_keep if predicted_keep else None
        recall = ai_keep_user_keep / actual_keep if actual_keep else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        return {
            "compared": compared,
            "agree": agree,
            "disagree": disagree,
            "agree_percent": round(100 * agree / compared, 1) if compared else None,
            "disagree_percent": round(100 * disagree / compared, 1) if compared else None,
            "ai_keep_user_keep": ai_keep_user_keep,
            "ai_keep_user_reject": ai_keep_user_reject,
            "ai_reject_user_keep": ai_reject_user_keep,
            "ai_reject_user_reject": ai_reject_user_reject,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def disagreements(self) -> list["ReviewImage"]:
        """Every image where the AI's suggestion and the photographer's own
        review status are both present and do not match - the "Detailed
        Differences" the evaluation report lists individually. Neutral is
        excluded for the same reason agreement_stats() excludes it: it is
        not a disagreement, it is no opinion yet."""
        suggestions = self._ai_suggestions()
        result = []
        for image in self.images:
            suggestion = suggestions.get(image.image_path)
            if suggestion is None or image.review_status == REVIEW_STATUS_NEUTRAL:
                continue
            if image.review_status != suggestion:
                result.append(image)
        return result

    # -- the photographer's review status ------------------------------------

    def keep_paths(self) -> list[str]:
        """Images the PHOTOGRAPHER marked Keep - see ReviewImage.user_
        decision. An algorithm cutoff's own keeps are deliberately absent:
        this list is what `arrange()` files into `_Selected`."""
        return [i.image_path for i in self.images if i.user_decision == REVIEW_STATUS_KEEP]

    def reject_paths(self) -> list[str]:
        """Images the photographer marked Reject - see `keep_paths`."""
        return [i.image_path for i in self.images if i.user_decision == REVIEW_STATUS_REJECT]

    def undecided_paths(self) -> list[str]:
        """Images with NO user decision - never reviewed, or explicitly
        cleared back to Neutral. Nothing files these."""
        return [i.image_path for i in self.images if not i.is_decided]

    # The pre-three-state name for `undecided_paths`, kept because "neutral"
    # is still the wire value the web page and the N button speak.
    neutral_paths = undecided_paths

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.images),
            "keep": len(self.keep_paths()),
            "reject": len(self.reject_paths()),
            "neutral": len(self.undecided_paths()),
            # Same three-state tally under the explicit name, plus how many
            # images carry a recorded ALGORITHM cutoff (informational - not
            # part of keep/reject/undecided, which are user decisions only).
            "undecided": len(self.undecided_paths()),
            "algorithm_decisions": sum(1 for i in self.images if i.algorithm_decision is not None),
            "missing_file": sum(1 for i in self.images if i.missing_file),
        }

    def ai_suggestion_counts(self) -> dict[str, int]:
        """The AI's own tally - how many ranked images it currently suggests
        Keep vs Reject, independent of anything the photographer has
        decided. See _ai_suggestions; a public wrapper exists because
        evaluation_report.py, outside this class, needs the same tally."""
        suggestions = self._ai_suggestions().values()
        return {
            "keep": sum(1 for s in suggestions if s == REVIEW_STATUS_KEEP),
            "reject": sum(1 for s in suggestions if s == REVIEW_STATUS_REJECT),
        }

    def workflow_state(self) -> dict:
        counts = self.counts()
        reviewed = counts["keep"] + counts["reject"] > 0
        ranked = bool(self.ranking_file and self.ranking_file.is_file())
        return {
            "stage": "folder" if self.input_folder is None else "review" if not ranked else "ranked",
            "ranked": ranked,
            "reviewed": reviewed,
            "selected": counts["keep"],
            "rejected": counts["reject"],
            "imported": False,
            "species": "pending",
            "crop": "pending",
        }

    def as_dict(self) -> dict:
        ai_suggestions = self._ai_suggestions()
        # burst_strategy is the one "which ranking strategy is currently
        # selected" the app shares across Burst Analysis, the desktop Color
        # Source picker, and now this - see its own docstring. Recomputing
        # suggestions_for(AI_STRATEGY_ID) here whenever burst_strategy
        # happens to already be the AI model would just repeat the exact
        # same work _ai_suggestions() already did above, so that case is
        # shortcut rather than calling suggestions_for a second time.
        algorithm_suggestions = (
            ai_suggestions if self.burst_strategy == AI_STRATEGY_ID else self.suggestions_for(self.burst_strategy)
        )
        burst = self.burst_info()
        with self._state_lock:
            loading = dict(self._loading_state)
            warnings = list(self.warnings)
            images = list(self.images)
        return {
            "input_folder": str(self.input_folder) if self.input_folder else None,
            "folder_missing": self.folder_missing,
            "ranking_file": str(self.ranking_file) if self.ranking_file else None,
            "has_ranking": bool(self.ranking_file and self.ranking_file.is_file()),
            "keep_percent": self.keep_percent,
            # Which strategy algorithm_suggestion (on every image below) is
            # currently computed against - the UI needs this to label the
            # suggestion/conflict filters correctly (e.g. "Classic Vision
            # suggests Keep" instead of an unconditional "AI suggests...").
            "suggestion_strategy": self.burst_strategy,
            "counts": self.counts(),
            "agreement": self.agreement_stats(),
            "warnings": warnings,
            "run": self.run_metadata,
            "workflow": self.workflow_state(),
            "loading": loading,
            "images": [
                image.as_dict(
                    ai_suggestions.get(image.image_path),
                    burst.get(image.image_path),
                    algorithm_suggestions.get(image.image_path),
                )
                for image in images
            ],
        }

    # -- writes ---------------------------------------------------------------

    def set_review_status(
        self,
        image_path: str,
        status: str,
        *,
        reason: str | None = None,
        reason_note: str | None = None,
        source: str = DECISION_SOURCE_USER,
    ) -> str:
        """Record the photographer's Keep/Reject/Neutral for one image.

        `status` is always one of REVIEW_STATUSES - Neutral is a real,
        explicit choice here (clearing the stored decision), not the absence
        of one. `reason`/`reason_note` only mean anything alongside Keep or
        Reject - a reason is why an override was made, and Neutral is not an
        override of anything, so either is dropped when `status` is Neutral,
        and `reason_note` is dropped unless `reason` is REVIEW_REASON_OTHER
        (mirroring what the store itself enforces).

        Persisted immediately - a review session's work must never exist only
        in a browser tab. Returns the resulting review_status.

        `source` defaults to DECISION_SOURCE_USER: every review-UI path
        (card buttons, Loupe, K/R/N shortcuts, bulk multi-select) is the
        photographer acting, and that default is what makes "I clicked Keep"
        the only way an image becomes Keep. `_apply_suggestions` is the one
        caller that passes DECISION_SOURCE_ALGORITHM instead.
        """
        if status not in REVIEW_STATUSES:
            raise InvalidReviewStatus(f"status must be one of {sorted(REVIEW_STATUSES)}, got {status!r}")
        image = self._image_for(image_path)
        if status == REVIEW_STATUS_NEUTRAL:
            self.store.clear_review_decision(image.image_path)
            reason = None
            reason_note = None
        else:
            if reason != REVIEW_REASON_OTHER:
                reason_note = None
            self.store.set_review_decision(
                image.image_path, status, reason=reason, reason_note=reason_note, source=source
            )
        image.decision = None if status == REVIEW_STATUS_NEUTRAL else status
        image.decision_source = None if status == REVIEW_STATUS_NEUTRAL else source
        image.reason = reason
        image.reason_note = reason_note
        return image.review_status

    def set_review_statuses(self, image_paths: list[str], status: str) -> dict:
        """Apply the same review status to many images at once - the
        multi-select bulk action's backend.

        No reason travels with a bulk action - reason fields exist to record
        a photographer's own judgement about ONE frame (eyes, focus), which
        does not generalise across an arbitrary batch. Reuses
        `set_review_status` per image, so every invariant it enforces still
        holds for each one; an invalid `status` raises immediately, before
        anything is written, since that is a caller bug rather than a
        per-image data issue.

        A path no longer in this gallery (the file moved, or a folder switch
        happened mid-selection) - or one whose identity can no longer be
        established (see IdentityUnavailable) - is skipped and reported
        rather than aborting the whole batch: a large multi-select is exactly
        the case likely to include a few stale ones, and everything already
        applied before that point has already been persisted (each
        `set_review_status` commits on its own), so an abort would silently
        throw away real, saved progress along with the report of what went
        wrong.
        """
        applied = 0
        failed: list[str] = []
        for image_path in image_paths:
            try:
                self.set_review_status(image_path, status)
            except (KeyError, IdentityUnavailable):
                failed.append(image_path)
            else:
                applied += 1
        return {"applied": applied, "failed": failed}

    def apply_ai_suggestions(self, *, include_decided: bool = False) -> dict:
        """Record the AI's CURRENT suggestion for every ranked image as an
        ALGORITHM decision - a snapshot of what the cutoff would pick, kept
        alongside (never inside) the photographer's own decisions.

        This does NOT review anything on the photographer's behalf. Every row
        it writes carries DECISION_SOURCE_ALGORITHM, so none of them colors a
        card as a User Decision, counts toward Keep/Reject, or is filed by
        `arrange()` - see `_apply_suggestions`. An image the photographer has
        already decided is never touched at all.

        `conflicts` in the result is the number of user-decided images whose
        User Decision disagrees with the AI's own suggestion - informational,
        the same comparison `agreement_stats` makes. `include_decided` only
        governs re-writing an image's own EARLIER algorithm decision at a new
        threshold; `overridden` counts those.

        Deliberately always the AI model specifically, never whichever
        strategy `burst_strategy`/the desktop Color Source picker currently
        names - see `_ai_suggestions`'s own docstring for why (this backs
        `agreement_stats`/the evaluation-report tooling, which compare the
        trained model against human judgement specifically, not "whichever
        algorithm was on screen"). A caller that wants the CURRENTLY
        SELECTED strategy's own cutoff applied instead wants
        `apply_algorithm_suggestions` below.
        """
        return self._apply_suggestions(self._ai_suggestions(), include_decided=include_decided)

    def apply_algorithm_suggestions(self, strategy_id: str, *, include_decided: bool = False) -> dict:
        """The general form of `apply_ai_suggestions` - bulk-accepts
        `strategy_id`'s own current suggestion (`suggestions_for`, the same
        per-strategy cutoff `ImageItem.algorithm_suggestion`/the gallery's
        "Color by Algorithm" coloring already use) instead of being fixed to
        the AI model.

        Exists so the desktop's "Apply Cutoff" toolbar action can bulk-write
        against WHICHEVER strategy the Color Source picker currently shows -
        previously it always wrote the AI model's own suggestion regardless
        of Color Source, so applying a cutoff while viewing a different
        algorithm's coloring could leave review_status and that algorithm's
        own top-N% picture visibly disagreeing (a scored-and-clearly-top
        image ending up Reject-colored because the AI model, not the
        strategy on screen, decided the cutoff). Same bulk-write mechanics
        as `apply_ai_suggestions`, just parameterized on which suggestions
        dict feeds it - see `_apply_suggestions`.
        """
        return self._apply_suggestions(self.suggestions_for(strategy_id), include_decided=include_decided)

    def _apply_suggestions(self, suggestions: dict[str, str | None], *, include_decided: bool) -> dict:
        """Record `suggestions` as ALGORITHM decisions (DECISION_SOURCE_
        ALGORITHM), never user ones.

        This method is where a ranking used to become indistinguishable from
        a photographer's own review: it wrote the cutoff's keep/reject
        through the same path a Grid button click uses, into the same rows,
        with nothing on the row saying which was which. One "Apply Cutoff"
        on a 5,986-image folder then left every image looking reviewed - the
        Grid colored them all, the counts claimed them all, and `arrange()`
        would have filed them all.

        Now the write carries its origin. An algorithm decision is shown as
        an algorithm decision and is never read as a User Decision by
        coloring, counts or organizing; a photographer's own Keep/Reject is
        never touched at all (`include_decided` can only ever override
        another ALGORITHM row, which is just this method re-running at a
        different threshold).
        """
        applied = 0
        overridden = 0
        conflicts = 0
        for image in self.images:
            suggestion = suggestions.get(image.image_path)
            if suggestion is None:
                continue
            if image.is_decided:
                # A real user decision. Counted as a conflict when it
                # disagrees, so the caller can still report the comparison,
                # but never overwritten - not even with include_decided.
                if image.user_decision != suggestion:
                    conflicts += 1
                continue
            if image.algorithm_decision is None:
                self.set_review_status(image.image_path, suggestion, source=DECISION_SOURCE_ALGORITHM)
                applied += 1
            elif image.algorithm_decision != suggestion:
                if include_decided:
                    self.set_review_status(image.image_path, suggestion, source=DECISION_SOURCE_ALGORITHM)
                    overridden += 1
        return {"applied": applied, "overridden": overridden, "conflicts": conflicts}

    def clear_algorithm_decisions(self) -> int:
        """Undo every recorded algorithm cutoff at once, leaving the
        photographer's own decisions untouched. Returns how many were
        removed. The escape hatch for a bulk "Apply Cutoff" - see
        `_apply_suggestions`."""
        removed = self.store.clear_review_decisions_by_source(DECISION_SOURCE_ALGORITHM)
        if removed:
            self._apply_decisions()
        return removed

    def _image_for(self, image_path: str) -> ReviewImage:
        key = _key(image_path)
        for image in self.images:
            if _key(image.image_path) == key:
                return image
        raise KeyError(f"{image_path} is not part of this review session")

    def arrange(self, *, dry_run: bool = False) -> OrganizeResult:
        """File the shoot by the photographer's OWN User Decision alone: Keep
        to `_Selected`, Reject to `_Rejected`, Undecided left exactly where
        it is - ranked highly by an algorithm or not, carrying a recorded
        algorithm cutoff or not. Only an explicit user Keep or Reject files
        anything (`keep_paths`/`reject_paths`, which read `user_decision`).

        This is what makes batch reviewing work: decide 200 images, arrange
        those 200, come back and decide 300 more. The other 5,000 the
        ranking scored are not candidates for either folder, because nobody
        has judged them yet.

        On a real run the ranking and the stored decisions are both repointed
        at the new locations, so the folder can be reviewed again afterwards.
        """
        if self.input_folder is None:
            raise ValueError("No folder is open to arrange.")
        result = organize_by_decision(
            self.keep_paths(),
            self.reject_paths(),
            self.input_folder,
            dry_run=dry_run,
            announce=False,
        )
        if not dry_run and result.moves:
            rewrite_ranking_paths(self.input_folder, result.moves)
            self.store.repoint_review_decisions(result.moves)
            self.load()
        return result


def _key(path: str | Path) -> str:
    """Compare paths the way the filesystem does, so a ranking written with a
    different spelling (case, separators) still matches what is on disk."""
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:  # pragma: no cover - unresolvable path
        return str(path).casefold()
