"""What the review application decides, before any of it reaches a browser.

The rules that matter, and that the rest of the app depends on being exact:

- every image's review_status is always exactly Keep, Reject or Neutral -
  the photographer's own, independent verdict;
- the AI ranking (score, rank, ai_suggestion) is read-only metadata and never
  changes review_status by itself - only apply_ai_suggestions does, and only
  because the photographer explicitly asked it to;
- clearing a review status (one image, or a bulk selection) always lands on
  Neutral - never silently on whatever the AI would have picked;
- an image on disk but absent from the ranking still appears in the gallery;
- arrange() files Keep/Reject only; Neutral is never moved, ranked highly or
  not; a review status, once set, survives arranging - the files move
  underneath it.
"""

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import AnnotationStore
from picklikeme.organize import REJECTED_DIRNAME, SELECTED_DIRNAME, selection_count
from picklikeme.review.session import (
    REVIEW_STATUS_KEEP,
    REVIEW_STATUS_NEUTRAL,
    REVIEW_STATUS_REJECT,
    InvalidReviewStatus,
    ReviewSession,
)
from picklikeme.sidecar import AI_STRATEGY_ID, ranking_path


def build_shoot(root: Path, ranked: int = 10, unranked: int = 0) -> tuple[Path, list[Path], list[Path]]:
    """A folder with a ranking, plus optionally images the ranking never saw.

    Scores descend with the index, so `images[0]` is the AI's best pick and
    `images[-1]` its worst - which is what makes the independence tests
    legible (a Keep on the worst-ranked image, a Reject on the best one).
    """
    shoot = root / "shoot"
    shoot.mkdir(parents=True, exist_ok=True)

    ranked_paths = []
    for index in range(ranked):
        target = shoot / f"IMG_{index:04d}.jpg"
        target.write_bytes(f"frame {index}".encode())
        ranked_paths.append(target)

    extra_paths = []
    for index in range(unranked):
        target = shoot / f"EXTRA_{index:04d}.jpg"
        target.write_bytes(f"extra {index}".encode())
        extra_paths.append(target)

    target = ranking_path(shoot)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["select_root", str(shoot)])
        writer.writerow([])
        writer.writerow(["rank", "image_path", "score", "label"])
        for rank, path in enumerate(ranked_paths, start=1):
            writer.writerow([rank, str(path), f"{1.0 - rank * 0.05:.6f}", 0])
    return shoot, ranked_paths, extra_paths


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: Windows holds the SQLite WAL open a moment
        # after close(), which is teardown timing, not a test failure.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.store = AnnotationStore(self.root / "kb.db")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def session(self, shoot: Path, **kwargs) -> ReviewSession:
        return ReviewSession(shoot, self.store, **kwargs)


class AiSuggestionTests(SessionTestCase):
    """The AI's own opinion - read-only, purely informational. review_status
    never depends on it; only ai_suggestion (and, if invoked,
    apply_ai_suggestions) does."""

    def test_the_cut_matches_organize_s_own_arithmetic(self):
        """The AI-suggestion count and organize's own selection_count must
        never disagree, so both come from the same function."""
        shoot, _, _ = build_shoot(self.root, ranked=10)
        for percent in (0, 5, 10, 25, 33.3, 50, 100):
            session = self.session(shoot, keep_percent=percent)
            self.assertEqual(session.cut, selection_count(10, percent))
            keep_suggestions = sum(1 for v in session._ai_suggestions().values() if v == REVIEW_STATUS_KEEP)
            self.assertEqual(keep_suggestions, session.cut)

    def test_the_highest_scoring_images_get_the_keep_suggestion(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=30)
        suggestions = session._ai_suggestions()
        for path in images[:3]:
            self.assertEqual(suggestions[str(path)], REVIEW_STATUS_KEEP)
        for path in images[3:]:
            self.assertEqual(suggestions[str(path)], REVIEW_STATUS_REJECT)

    def test_moving_the_threshold_changes_the_suggestion_never_the_review_status(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=10)
        best = str(images[0])
        self.assertEqual(session._ai_suggestions()[best], REVIEW_STATUS_KEEP)

        session.set_keep_percent(0)

        self.assertEqual(session._ai_suggestions()[best], REVIEW_STATUS_REJECT)
        self.assertEqual(session._image_for(best).review_status, REVIEW_STATUS_NEUTRAL)

    def test_an_unranked_image_gets_no_suggestion(self):
        shoot, _, extra = build_shoot(self.root, ranked=4, unranked=2)
        session = self.session(shoot, keep_percent=50)
        suggestions = session._ai_suggestions()
        for path in extra:
            self.assertIsNone(suggestions[str(path)])


class SuggestionsForAnyStrategyTests(SessionTestCase):
    """suggestions_for(strategy_id) - the generalization behind the
    per-strategy "algorithm_suggestion" field and the desktop conflict
    filters, added so the "AI vs user" conflict filter can compare the
    user's decision against whichever strategy is currently selected
    (ReviewSession.burst_strategy), not only the AI model."""

    def test_suggestions_for_the_ai_strategy_matches_ai_suggestions(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=30)
        self.assertEqual(session.suggestions_for(AI_STRATEGY_ID), session._ai_suggestions())

    def test_suggestions_for_a_second_strategy_uses_its_own_score_order(self):
        """The reported motivation: Classic Vision (or any other module)
        disagreeing entirely with the AI about ranking order must produce
        its own, independently-correct keep/reject suggestions."""
        from picklikeme.sidecar import strategy_ranking_path

        shoot, images, _ = build_shoot(self.root, ranked=3)
        # Classic Vision's favorite is the AI's least favorite, and vice versa.
        write_strategy_ranking(
            strategy_ranking_path(shoot, "classic-vision"),
            [(images[2], 0.99), (images[1], 0.5), (images[0], 0.1)],
        )
        session = self.session(shoot, keep_percent=34)  # top 1 of 3

        ai_suggestions = session.suggestions_for(AI_STRATEGY_ID)
        classic_suggestions = session.suggestions_for("classic-vision")

        self.assertEqual(ai_suggestions[str(images[0])], REVIEW_STATUS_KEEP)
        self.assertEqual(classic_suggestions[str(images[0])], REVIEW_STATUS_REJECT)
        self.assertEqual(classic_suggestions[str(images[2])], REVIEW_STATUS_KEEP)

    def test_an_image_only_the_ai_scored_gets_no_suggestion_from_a_second_strategy(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot, keep_percent=50)
        suggestions = session.suggestions_for("classic-vision")
        for path in images:
            self.assertIsNone(suggestions[str(path)])

    def test_as_dict_exposes_algorithm_suggestion_and_which_strategy_it_is_for(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot, keep_percent=34)
        state = session.as_dict()

        self.assertEqual(state["suggestion_strategy"], AI_STRATEGY_ID)
        by_path = {img["image_path"]: img for img in state["images"]}
        # With no other strategy selected, algorithm_suggestion equals ai_suggestion.
        self.assertEqual(
            by_path[str(images[0])]["algorithm_suggestion"], by_path[str(images[0])]["ai_suggestion"]
        )

    def test_as_dict_reflects_a_non_ai_burst_strategy_selection(self):
        from picklikeme.sidecar import strategy_ranking_path

        shoot, images, _ = build_shoot(self.root, ranked=3)
        write_strategy_ranking(
            strategy_ranking_path(shoot, "classic-vision"),
            [(images[2], 0.99), (images[1], 0.5), (images[0], 0.1)],
        )
        session = self.session(shoot, keep_percent=34)
        session.set_burst_strategy("classic-vision")

        state = session.as_dict()
        self.assertEqual(state["suggestion_strategy"], "classic-vision")
        by_path = {img["image_path"]: img for img in state["images"]}
        self.assertEqual(by_path[str(images[2])]["algorithm_suggestion"], REVIEW_STATUS_KEEP)
        self.assertEqual(by_path[str(images[0])]["algorithm_suggestion"], REVIEW_STATUS_REJECT)
        # ai_suggestion is untouched by the burst_strategy selection - still the AI's own opinion.
        self.assertEqual(by_path[str(images[0])]["ai_suggestion"], REVIEW_STATUS_KEEP)


class ReviewStatusTests(SessionTestCase):
    """The photographer's own verdict - always exactly Keep, Reject or
    Neutral, completely independent of whatever the AI suggests."""

    def test_every_image_starts_neutral(self):
        shoot, images, _ = build_shoot(self.root, ranked=5)
        session = self.session(shoot)
        for path in images:
            self.assertEqual(session._image_for(str(path)).review_status, REVIEW_STATUS_NEUTRAL)

    def test_setting_a_status_is_independent_of_the_ai_suggestion(self):
        """The AI's own top pick can be Rejected, and its own bottom pick can
        be Kept - review_status and ai_suggestion never have to agree."""
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=10)
        best = str(images[0])
        worst = str(images[-1])

        session.set_review_status(best, REVIEW_STATUS_REJECT)
        session.set_review_status(worst, REVIEW_STATUS_KEEP)

        self.assertEqual(session._image_for(best).review_status, REVIEW_STATUS_REJECT)
        self.assertEqual(session._image_for(worst).review_status, REVIEW_STATUS_KEEP)
        suggestions = session._ai_suggestions()
        self.assertEqual(suggestions[best], REVIEW_STATUS_KEEP, "the AI's own opinion is untouched")
        self.assertEqual(suggestions[worst], REVIEW_STATUS_REJECT)

    def test_clearing_a_status_returns_to_neutral_never_to_the_ai_suggestion(self):
        """The bug this model exists to fix: Neutral must be a real, distinct
        status, never silently replaced by whatever the AI would have picked
        at the current threshold."""
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=90)  # the AI would keep nearly all of these
        best = str(images[0])
        session.set_review_status(best, REVIEW_STATUS_REJECT)

        status = session.set_review_status(best, REVIEW_STATUS_NEUTRAL)

        self.assertEqual(status, REVIEW_STATUS_NEUTRAL)
        self.assertEqual(session._image_for(best).review_status, REVIEW_STATUS_NEUTRAL)
        self.assertIn(best, session.neutral_paths())
        self.assertNotIn(best, session.keep_paths())
        self.assertNotIn(best, session.reject_paths())

    def test_moving_the_ai_threshold_never_changes_an_existing_review_status(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=10)
        session.set_review_status(str(images[-1]), REVIEW_STATUS_KEEP)
        session.set_review_status(str(images[0]), REVIEW_STATUS_REJECT)

        for percent in (0, 25, 50, 100):
            session.set_keep_percent(percent)
            self.assertEqual(session._image_for(str(images[-1])).review_status, REVIEW_STATUS_KEEP)
            self.assertEqual(session._image_for(str(images[0])).review_status, REVIEW_STATUS_REJECT)

    def test_decisions_are_persisted_immediately_not_held_in_memory(self):
        """Refreshing the page must restore everything; a browser tab is not
        where a photographer's work is allowed to live."""
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=25)
        session.set_review_status(str(images[4]), REVIEW_STATUS_KEEP)

        reopened = self.session(shoot, keep_percent=25)

        self.assertEqual(reopened._image_for(str(images[4])).review_status, REVIEW_STATUS_KEEP)
        self.assertEqual(reopened.counts()["keep"], 1)

    def test_an_unknown_path_is_refused(self):
        shoot, _, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)
        with self.assertRaises(KeyError):
            session.set_review_status(str(self.root / "elsewhere.jpg"), REVIEW_STATUS_KEEP)

    def test_an_invalid_status_is_refused(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)
        with self.assertRaises(InvalidReviewStatus):
            session.set_review_status(str(images[0]), "maybe")

    def test_a_reason_travels_with_the_status(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=10)
        worst = str(images[-1])

        session.set_review_status(worst, REVIEW_STATUS_REJECT, reason="eyes_not_seen")

        image = next(i for i in session.images if i.image_path == worst)
        self.assertEqual(image.reason, "eyes_not_seen")
        self.assertEqual(image.as_dict(None)["reason"], "eyes_not_seen")

    def test_a_reason_is_optional_and_defaults_to_none(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)
        path = str(images[0])

        session.set_review_status(path, REVIEW_STATUS_KEEP)

        image = next(i for i in session.images if i.image_path == path)
        self.assertIsNone(image.reason)

    def test_setting_neutral_clears_any_reason(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)
        path = str(images[0])
        session.set_review_status(path, REVIEW_STATUS_KEEP, reason="clear_eyes_seen")

        session.set_review_status(path, REVIEW_STATUS_NEUTRAL)

        image = next(i for i in session.images if i.image_path == path)
        self.assertIsNone(image.reason)

    def test_a_reason_is_persisted_immediately_like_the_status_it_belongs_to(self):
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=25)
        session.set_review_status(str(images[4]), REVIEW_STATUS_REJECT, reason="eyes_not_seen")

        reopened = self.session(shoot, keep_percent=25)

        image = next(i for i in reopened.images if i.image_path == str(images[4]))
        self.assertEqual(image.reason, "eyes_not_seen")


class BulkReviewStatusTests(SessionTestCase):
    """The multi-select toolbar's backend - the same set_review_status,
    applied to many images under one call instead of one request per image."""

    def test_applies_the_same_status_to_every_path(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=25)

        result = session.set_review_statuses([str(images[0]), str(images[1])], REVIEW_STATUS_KEEP)

        self.assertEqual(result["applied"], 2)
        self.assertEqual(result["failed"], [])
        self.assertEqual(session._image_for(str(images[0])).review_status, REVIEW_STATUS_KEEP)
        self.assertEqual(session._image_for(str(images[1])).review_status, REVIEW_STATUS_KEEP)

    def test_bulk_clearing_returns_every_path_to_neutral(self):
        """The bulk half of the same bug fix as ReviewStatusTests' single-
        image version: Neutral, never whatever the AI would have picked."""
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=90)  # the AI would keep almost all of these
        session.set_review_statuses([str(images[0]), str(images[1])], REVIEW_STATUS_REJECT)

        session.set_review_statuses([str(images[0]), str(images[1])], REVIEW_STATUS_NEUTRAL)

        self.assertEqual(session._image_for(str(images[0])).review_status, REVIEW_STATUS_NEUTRAL)
        self.assertEqual(session._image_for(str(images[1])).review_status, REVIEW_STATUS_NEUTRAL)

    def test_a_path_not_in_the_gallery_is_reported_and_skipped(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot, keep_percent=25)

        result = session.set_review_statuses(
            [str(images[0]), str(self.root / "nowhere.jpg")], REVIEW_STATUS_KEEP
        )

        self.assertEqual(result["applied"], 1)
        self.assertEqual(result["failed"], [str(self.root / "nowhere.jpg")])
        self.assertEqual(session._image_for(str(images[0])).review_status, REVIEW_STATUS_KEEP)

    def test_each_status_is_persisted_immediately_not_only_in_memory(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot, keep_percent=25)

        session.set_review_statuses([str(images[0]), str(images[1])], REVIEW_STATUS_KEEP)
        reopened = self.session(shoot, keep_percent=25)

        self.assertEqual(reopened._image_for(str(images[0])).review_status, REVIEW_STATUS_KEEP)
        self.assertEqual(reopened._image_for(str(images[1])).review_status, REVIEW_STATUS_KEEP)

    def test_no_reason_is_ever_recorded_by_a_bulk_action(self):
        shoot, images, _ = build_shoot(self.root, ranked=2)
        session = self.session(shoot, keep_percent=25)

        session.set_review_statuses([str(images[0])], REVIEW_STATUS_KEEP)

        self.assertIsNone(session._image_for(str(images[0])).reason)

    def test_an_invalid_status_raises_before_writing_anything(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)

        with self.assertRaises(InvalidReviewStatus):
            session.set_review_statuses([str(images[0]), str(images[1])], "maybe")

        for path in images[:2]:
            self.assertEqual(session._image_for(str(path)).review_status, REVIEW_STATUS_NEUTRAL)


class ApplyAiSuggestionsTests(SessionTestCase):
    """Bulk-accepting the AI's current suggestion - the ONE path by which the
    ranking is ever allowed to set a review status, and only because the
    photographer explicitly asked for it, once."""

    def test_applies_the_suggestion_to_every_neutral_ranked_image(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=30)

        result = session.apply_ai_suggestions()

        self.assertEqual(result["applied"], 10)
        for path in images[:3]:
            self.assertEqual(session._image_for(str(path)).review_status, REVIEW_STATUS_KEEP)
        for path in images[3:]:
            self.assertEqual(session._image_for(str(path)).review_status, REVIEW_STATUS_REJECT)

    def test_never_touches_an_image_already_decided(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=30)
        # Deliberately disagree with the AI on its own top pick.
        session.set_review_status(str(images[0]), REVIEW_STATUS_REJECT)

        session.apply_ai_suggestions()

        self.assertEqual(session._image_for(str(images[0])).review_status, REVIEW_STATUS_REJECT)

    def test_never_touches_an_unranked_image(self):
        shoot, _, extra = build_shoot(self.root, ranked=4, unranked=2)
        session = self.session(shoot, keep_percent=50)

        result = session.apply_ai_suggestions()

        self.assertEqual(result["applied"], 4, "only the ranked images have a suggestion to apply")
        for path in extra:
            self.assertEqual(session._image_for(str(path)).review_status, REVIEW_STATUS_NEUTRAL)

    def test_is_persisted_immediately(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)
        session.apply_ai_suggestions()

        reopened = self.session(shoot, keep_percent=50)
        self.assertEqual(reopened._image_for(str(images[0])).review_status, REVIEW_STATUS_KEEP)

    def test_running_it_twice_is_a_no_op_the_second_time(self):
        shoot, _, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=33)
        session.apply_ai_suggestions()

        again = session.apply_ai_suggestions()

        self.assertEqual(again["applied"], 0, "nothing is Neutral any more")

    def test_conflicts_are_reported_but_not_touched_by_default(self):
        """Phase 9: never silently overwrite a photographer's own Keep/Reject
        - a disagreeing, already-decided image is counted, not changed,
        unless include_decided is explicitly True."""
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=30)
        best = str(images[0])  # the AI's own top pick
        session.set_review_status(best, REVIEW_STATUS_REJECT)  # disagrees on purpose

        result = session.apply_ai_suggestions()

        self.assertEqual(result["conflicts"], 1)
        self.assertEqual(result["overridden"], 0)
        self.assertEqual(session._image_for(best).review_status, REVIEW_STATUS_REJECT)
        # The other 9 Neutral images were still applied normally.
        self.assertEqual(result["applied"], 9)

    def test_include_decided_overrides_only_the_disagreeing_images(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=30)
        best = str(images[0])
        agreeing = str(images[1])  # also AI-suggested Keep; agree with it up front
        session.set_review_status(best, REVIEW_STATUS_REJECT)
        session.set_review_status(agreeing, REVIEW_STATUS_KEEP)

        result = session.apply_ai_suggestions(include_decided=True)

        self.assertEqual(result["conflicts"], 1)
        self.assertEqual(result["overridden"], 1)
        self.assertEqual(session._image_for(best).review_status, REVIEW_STATUS_KEEP, "now matches the AI")
        self.assertEqual(session._image_for(agreeing).review_status, REVIEW_STATUS_KEEP, "already agreed, untouched")

    def test_include_decided_still_leaves_agreeing_decided_images_alone(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)
        agreeing = str(images[0])
        session.set_review_status(agreeing, REVIEW_STATUS_KEEP)  # matches the AI's own top-half suggestion

        result = session.apply_ai_suggestions(include_decided=True)

        self.assertEqual(result["conflicts"], 0)
        self.assertEqual(result["overridden"], 0)
        self.assertEqual(session._image_for(agreeing).review_status, REVIEW_STATUS_KEEP)


class ApplyAlgorithmSuggestionsTests(SessionTestCase):
    """apply_algorithm_suggestions - the general form of apply_ai_suggestions,
    parameterized on which strategy's own cutoff to bulk-apply instead of
    always the AI model. Added for the desktop "Apply Cutoff" toolbar
    action: previously it always wrote the AI model's own suggestion no
    matter which strategy the Color Source picker showed, so applying a
    cutoff while viewing a different algorithm's coloring could leave a
    clearly-top-scored image (under THAT algorithm) Reject-colored, because
    the AI model - not the algorithm on screen - had decided the cutoff.
    """

    def test_applies_the_given_strategy_s_own_suggestion_not_the_ai_model_s(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)
        # A second strategy whose ranking is the exact REVERSE of the AI
        # model's (images[0] is the AI's best pick and images[-1] its
        # worst - see build_shoot's own docstring).
        other_strategy = "classic-vision"
        for index, path in enumerate(images):
            image = session._image_for(str(path))
            image.ranking_results[other_strategy] = {"score": index * 0.1, "rank": None}

        result = session.apply_algorithm_suggestions(other_strategy)

        self.assertEqual(result["applied"], 4)
        # Reversed order: the AI's WORST pick (images[-1]) has the highest
        # "classic-vision" score, so it - not images[0] - is Kept.
        self.assertEqual(session._image_for(str(images[-1])).review_status, REVIEW_STATUS_KEEP)
        self.assertEqual(session._image_for(str(images[0])).review_status, REVIEW_STATUS_REJECT)

    def test_never_touches_an_image_the_given_strategy_never_scored(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)
        other_strategy = "classic-vision"
        # Only scores TWO of the four images with the other strategy.
        for index, path in enumerate(images[:2]):
            image = session._image_for(str(path))
            image.ranking_results[other_strategy] = {"score": 1.0 - index * 0.1, "rank": index + 1}

        result = session.apply_algorithm_suggestions(other_strategy)

        self.assertEqual(result["applied"], 2, "only the two images that strategy actually scored")
        for path in images[2:]:
            self.assertEqual(session._image_for(str(path)).review_status, REVIEW_STATUS_NEUTRAL)

    def test_never_touches_an_image_already_decided_unless_told_to(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)
        other_strategy = "classic-vision"
        for index, path in enumerate(images):
            image = session._image_for(str(path))
            image.ranking_results[other_strategy] = {"score": index * 0.1, "rank": None}
        # Deliberately disagree with classic-vision's own top pick.
        session.set_review_status(str(images[-1]), REVIEW_STATUS_REJECT)

        result = session.apply_algorithm_suggestions(other_strategy, include_decided=False)
        self.assertEqual(result["conflicts"], 1)
        self.assertEqual(session._image_for(str(images[-1])).review_status, REVIEW_STATUS_REJECT)

        result = session.apply_algorithm_suggestions(other_strategy, include_decided=True)
        self.assertEqual(result["overridden"], 1)
        self.assertEqual(session._image_for(str(images[-1])).review_status, REVIEW_STATUS_KEEP)

    def test_apply_ai_suggestions_is_unaffected_and_stays_ai_specific(self):
        """Regression: adding apply_algorithm_suggestions must not change
        apply_ai_suggestions's own behavior - it is still always the AI
        model, regardless of any other strategy's scores on the same
        images (see apply_ai_suggestions's own docstring on why - it backs
        agreement_stats/evaluation_report.py, which are deliberately
        AI-specific)."""
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)
        for index, path in enumerate(images):
            image = session._image_for(str(path))
            image.ranking_results["classic-vision"] = {"score": index * 0.1, "rank": None}

        session.apply_ai_suggestions()

        # AI model order preserved (images[0] best -> Keep), unaffected by
        # classic-vision's reversed scores on the very same images.
        self.assertEqual(session._image_for(str(images[0])).review_status, REVIEW_STATUS_KEEP)
        self.assertEqual(session._image_for(str(images[-1])).review_status, REVIEW_STATUS_REJECT)


class AgreementStatsTests(SessionTestCase):
    """How often the photographer's own review status matches the AI's
    suggestion - informational, for evaluating the model over time."""

    def test_neutral_images_are_excluded_from_the_comparison(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)

        stats = session.agreement_stats()

        self.assertEqual(stats["compared"], 0)
        self.assertIsNone(stats["agree_percent"])
        self.assertIsNone(stats["disagree_percent"])

    def test_agreement_and_disagreement_are_counted_correctly(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)  # AI keeps images[0:2], rejects images[2:4]
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)     # agrees (AI: keep)
        session.set_review_status(str(images[1]), REVIEW_STATUS_REJECT)   # disagrees (AI: keep, user: reject)
        session.set_review_status(str(images[2]), REVIEW_STATUS_KEEP)     # disagrees (AI: reject, user: keep)
        # images[3] left Neutral - excluded entirely.

        stats = session.agreement_stats()

        self.assertEqual(stats["compared"], 3)
        self.assertEqual(stats["agree"], 1)
        self.assertEqual(stats["disagree"], 2)
        self.assertEqual(stats["ai_keep_user_reject"], 1)
        self.assertEqual(stats["ai_reject_user_keep"], 1)
        self.assertAlmostEqual(stats["agree_percent"], 100 / 3, places=1)
        self.assertAlmostEqual(stats["disagree_percent"], 200 / 3, places=1)

    def test_unranked_images_are_excluded_even_if_decided(self):
        shoot, _, extra = build_shoot(self.root, ranked=2, unranked=1)
        session = self.session(shoot, keep_percent=50)
        session.set_review_status(str(extra[0]), REVIEW_STATUS_KEEP)

        stats = session.agreement_stats()

        self.assertEqual(stats["compared"], 0, "the AI has no opinion about an unranked image")

    def test_the_full_confusion_matrix_is_reported(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)  # AI keeps images[0:2], rejects images[2:4]
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)     # AI keep / user keep
        session.set_review_status(str(images[1]), REVIEW_STATUS_REJECT)  # AI keep / user reject
        session.set_review_status(str(images[2]), REVIEW_STATUS_KEEP)    # AI reject / user keep
        session.set_review_status(str(images[3]), REVIEW_STATUS_REJECT)  # AI reject / user reject

        stats = session.agreement_stats()

        self.assertEqual(stats["ai_keep_user_keep"], 1)
        self.assertEqual(stats["ai_keep_user_reject"], 1)
        self.assertEqual(stats["ai_reject_user_keep"], 1)
        self.assertEqual(stats["ai_reject_user_reject"], 1)
        self.assertEqual(stats["agree"], 2)
        self.assertEqual(stats["disagree"], 2)

    def test_precision_recall_and_f1_use_review_status_as_ground_truth(self):
        """Precision: of what the AI suggested keeping, how much did the
        photographer also keep. Recall: of what the photographer kept, how
        much did the AI also suggest keeping. Keep is the positive class."""
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)     # TP
        session.set_review_status(str(images[1]), REVIEW_STATUS_REJECT)  # FP (AI keep, user reject)
        session.set_review_status(str(images[2]), REVIEW_STATUS_KEEP)    # FN (AI reject, user keep)
        session.set_review_status(str(images[3]), REVIEW_STATUS_REJECT)  # TN

        stats = session.agreement_stats()

        # precision = TP / (TP + FP) = 1 / 2; recall = TP / (TP + FN) = 1 / 2
        self.assertAlmostEqual(stats["precision"], 0.5)
        self.assertAlmostEqual(stats["recall"], 0.5)
        self.assertAlmostEqual(stats["f1"], 0.5)

    def test_precision_recall_and_f1_are_none_when_nothing_is_comparable(self):
        shoot, _, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)

        stats = session.agreement_stats()

        self.assertIsNone(stats["precision"])
        self.assertIsNone(stats["recall"])
        self.assertIsNone(stats["f1"])

    def test_precision_is_none_when_the_ai_never_suggested_keep(self):
        shoot, images, _ = build_shoot(self.root, ranked=2)
        session = self.session(shoot, keep_percent=0)  # AI rejects everything
        session.set_review_status(str(images[0]), REVIEW_STATUS_REJECT)
        session.set_review_status(str(images[1]), REVIEW_STATUS_KEEP)

        stats = session.agreement_stats()

        self.assertIsNone(stats["precision"], "nothing was predicted keep, so precision is undefined")
        self.assertEqual(stats["recall"], 0.0, "the one real keep was missed entirely")


class DisagreementsTests(SessionTestCase):
    """The evaluation report's "Detailed Differences" - every image where
    the AI and the photographer's own review status disagree."""

    def test_lists_only_the_disagreeing_images(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)     # agrees
        session.set_review_status(str(images[1]), REVIEW_STATUS_REJECT)  # disagrees

        disagreeing = session.disagreements()

        self.assertEqual([i.image_path for i in disagreeing], [str(images[1])])

    def test_neutral_and_unranked_images_are_never_listed(self):
        shoot, images, extra = build_shoot(self.root, ranked=2, unranked=1)
        session = self.session(shoot, keep_percent=50)
        session.set_review_status(str(extra[0]), REVIEW_STATUS_KEEP)

        self.assertEqual(session.disagreements(), [])


class MissingDataTests(SessionTestCase):
    def test_an_image_absent_from_the_ranking_still_appears(self):
        shoot, ranked, extra = build_shoot(self.root, ranked=4, unranked=2)
        session = self.session(shoot, keep_percent=50)

        self.assertEqual(session.counts()["total"], 6)
        for path in extra:
            self.assertEqual(session._image_for(str(path)).review_status, REVIEW_STATUS_NEUTRAL)

    def test_unranked_images_are_never_selected_automatically(self):
        """Ranked images start Neutral too now - a high AI threshold suggests
        nothing about review_status by itself - so all 7 (4 ranked + 3
        unranked) are Neutral until someone actually decides."""
        shoot, _, extra = build_shoot(self.root, ranked=4, unranked=3)
        session = self.session(shoot, keep_percent=100)

        for path in extra:
            self.assertNotIn(str(path), session.keep_paths())
            self.assertNotIn(str(path), session.reject_paths())
            self.assertIn(str(path), session.neutral_paths())
        self.assertEqual(session.counts()["neutral"], 7)

    def test_a_manual_decision_gives_an_unranked_image_a_destination(self):
        shoot, ranked, extra = build_shoot(self.root, ranked=4, unranked=1)
        session = self.session(shoot, keep_percent=25)

        session.set_review_status(str(extra[0]), REVIEW_STATUS_KEEP)

        self.assertIn(str(extra[0]), session.keep_paths())
        # Only the one explicitly decided image left Neutral - the 4 ranked
        # ones are untouched by the AI's own opinion of them.
        self.assertEqual(session.counts()["neutral"], len(ranked))

    def test_a_ranking_row_whose_file_is_gone_is_shown_not_dropped(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        images[1].unlink()

        session = self.session(shoot, keep_percent=50)

        self.assertEqual(session.counts()["total"], 4, "the row is kept so the gap is visible")
        self.assertEqual(session.counts()["missing_file"], 1)

    def test_a_folder_with_no_ranking_still_opens(self):
        """Every image Neutral is a reviewable state, not a crash."""
        shoot = self.root / "bare"
        shoot.mkdir()
        (shoot / "a.jpg").write_bytes(b"a")

        session = self.session(shoot)

        self.assertEqual(session.counts()["total"], 1)
        self.assertEqual(session.counts()["neutral"], 1)
        self.assertTrue(session.warnings)

    def test_an_unreadable_ranking_degrades_instead_of_failing(self):
        shoot, _, _ = build_shoot(self.root, ranked=3)
        ranking_path(shoot).write_text("this is not a ranking", encoding="utf-8")

        session = self.session(shoot)

        self.assertEqual(session.counts()["total"], 3, "images are still found on disk")
        self.assertEqual(session.counts()["neutral"], 3)
        self.assertTrue(any("ranking" in w.lower() for w in session.warnings))


class DetectedCategoryTests(SessionTestCase):
    """The subject category (bird/mammal/human/...) already recorded for an
    image - structured, read-only metadata, populated the same way
    captured_at is: once per load, never by running the detector itself."""

    def test_a_recorded_category_is_attached_to_the_matching_image(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        with mock.patch(
            "picklikeme.review.thumbnails.detected_category_for",
            side_effect=lambda path: "bird" if path == str(images[0]) else None,
        ):
            session = self.session(shoot)

        self.assertEqual(session._image_for(str(images[0])).detected_category, "bird")
        self.assertIsNone(session._image_for(str(images[1])).detected_category)

    def test_a_missing_file_is_never_asked_for_a_category(self):
        """Consistent with captured_at: there is nothing to read a category
        from for a file that is not there, and it must not raise."""
        shoot, images, _ = build_shoot(self.root, ranked=4)
        images[1].unlink()

        with mock.patch("picklikeme.review.thumbnails.detected_category_for", return_value=None) as detected:
            session = self.session(shoot)

        queried_paths = [call.args[0] for call in detected.call_args_list]
        self.assertNotIn(str(images[1]), queried_paths, "a missing file has nothing to read a category from")
        self.assertIsNone(session._image_for(str(images[1])).detected_category)

    def test_is_exposed_in_the_wire_format(self):
        shoot, images, _ = build_shoot(self.root, ranked=1)
        with mock.patch("picklikeme.review.thumbnails.detected_category_for", return_value="mammal"):
            session = self.session(shoot)

        payload = session.as_dict()
        self.assertEqual(payload["images"][0]["detected_category"], "mammal")


class ArrangeTests(SessionTestCase):
    def test_dry_run_reports_the_plan_and_moves_nothing(self):
        shoot, images, _ = build_shoot(self.root, ranked=8)
        session = self.session(shoot, keep_percent=25)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)
        session.set_review_status(str(images[1]), REVIEW_STATUS_KEEP)
        for path in images[2:]:
            session.set_review_status(str(path), REVIEW_STATUS_REJECT)

        result = session.arrange(dry_run=True)

        self.assertEqual(result.selected, 2)
        self.assertEqual(result.rejected, 6)
        for path in images:
            self.assertTrue(path.exists(), "a dry run must not move a file")
        self.assertFalse((shoot / SELECTED_DIRNAME).exists())

    def test_neutral_is_never_filed_no_matter_how_highly_the_ai_ranked_it(self):
        """The central behaviour this UX model is built around: an
        unreviewed (Neutral) image is never moved by Arrange, however highly
        the AI ranked it - only an explicit Keep or Reject does."""
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=100)  # the AI "suggests" keeping all 6
        # Nobody has reviewed any of them - every image is still Neutral.

        result = session.arrange()

        self.assertEqual(result.moved, 0)
        for path in images:
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, shoot)

    def test_arranging_files_by_review_status_alone(self):
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=33)
        session.set_review_status(str(images[5]), REVIEW_STATUS_KEEP)  # worst-ranked, kept anyway
        session.set_review_status(str(images[0]), REVIEW_STATUS_REJECT)  # best-ranked, rejected anyway

        session.arrange()

        self.assertTrue((shoot / SELECTED_DIRNAME / "IMG_0005.jpg").exists())
        self.assertTrue((shoot / REJECTED_DIRNAME / "IMG_0000.jpg").exists())

    def test_unranked_neutral_images_are_left_where_they_are(self):
        shoot, ranked, extra = build_shoot(self.root, ranked=4, unranked=2)
        session = self.session(shoot, keep_percent=50)
        for path in ranked:
            session.set_review_status(str(path), REVIEW_STATUS_REJECT)

        result = session.arrange()

        self.assertEqual(result.ranked, 4, "only the reviewed images were filed")
        for path in extra:
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, shoot)

    def test_decisions_and_ranking_both_follow_the_files(self):
        """The load-bearing re-review test: arrange moves everything, so both
        the ranking and the stored review status must be repointed or the
        next review would look like a folder nobody had ever touched."""
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=33)
        session.set_review_status(str(images[5]), REVIEW_STATUS_KEEP)

        session.arrange()
        reopened = self.session(shoot, keep_percent=33)

        self.assertEqual(reopened.counts()["total"], 6, "the arranged file is found again")
        self.assertEqual(reopened.counts()["missing_file"], 0, "the ranking was repointed")
        moved = shoot / SELECTED_DIRNAME / "IMG_0005.jpg"
        self.assertEqual(reopened._image_for(str(moved)).review_status, REVIEW_STATUS_KEEP)

    def test_arranging_twice_is_a_no_op_rather_than_a_shuffle(self):
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=50)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)
        session.arrange()

        again = session.arrange()

        self.assertEqual(again.moved, 0)
        self.assertEqual(again.errors, 0)


class IdentityRecoveryTests(SessionTestCase):
    def test_a_decision_is_recovered_after_the_file_moves_behind_our_back(self):
        """Path matching is the fast path; content identity is the truth. A
        file moved outside the app still carries its review status."""
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=25)
        session.set_review_status(str(images[3]), REVIEW_STATUS_KEEP)

        moved = shoot / "renamed_by_hand.jpg"
        images[3].rename(moved)

        reopened = self.session(shoot, keep_percent=25)
        self.assertEqual(
            reopened._image_for(str(moved)).review_status, REVIEW_STATUS_NEUTRAL, "path match cannot find it"
        )

        recovered = reopened.reconcile_by_identity()

        self.assertEqual(recovered, 1)
        self.assertEqual(reopened._image_for(str(moved)).review_status, REVIEW_STATUS_KEEP)

    def test_reconciling_costs_nothing_when_nothing_moved(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=25)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)

        reopened = self.session(shoot, keep_percent=25)

        self.assertEqual(reopened.reconcile_by_identity(), 0)

    def test_a_reason_is_recovered_along_with_the_status_it_belongs_to(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=25)
        session.set_review_status(str(images[3]), REVIEW_STATUS_KEEP, reason="clear_eyes_seen")

        moved = shoot / "renamed_by_hand.jpg"
        images[3].rename(moved)

        reopened = self.session(shoot, keep_percent=25)
        reopened.reconcile_by_identity()

        self.assertEqual(reopened._image_for(str(moved)).reason, "clear_eyes_seen")


class OpenFolderTests(SessionTestCase):
    """Switching a live session to a different folder - the way a photo
    folder that was never ranked at all gets reviewed, without restarting the
    server or losing the one shared annotations store."""

    def test_opening_a_different_folder_replaces_the_gallery(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)

        other = self.root / "unranked"
        other.mkdir()
        (other / "a.jpg").write_bytes(b"a")
        (other / "b.jpg").write_bytes(b"b")
        session.open_folder(other)

        self.assertEqual(session.input_folder, other.resolve())
        self.assertEqual(session.counts()["total"], 2)
        self.assertEqual(session.counts()["neutral"], 2)
        self.assertTrue(session.warnings, "the new folder has no ranking of its own")

    def test_a_manual_decision_in_the_new_folder_is_unaffected_by_the_old_one(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)

        other = self.root / "unranked"
        other.mkdir()
        photo = other / "a.jpg"
        photo.write_bytes(b"a")
        session.open_folder(other)

        self.assertEqual(session._image_for(str(photo)).review_status, REVIEW_STATUS_NEUTRAL)

    def test_a_decision_already_recorded_for_the_new_folder_is_picked_up(self):
        """Re-opening a folder reviewed before (e.g. switching away and back)
        must not look like a first visit - the store remembers regardless."""
        other = self.root / "unranked"
        other.mkdir()
        photo = other / "a.jpg"
        photo.write_bytes(b"a")
        session = self.session(other)
        session.set_review_status(str(photo), REVIEW_STATUS_REJECT)

        shoot, _, _ = build_shoot(self.root, ranked=2)
        session.open_folder(shoot)
        session.open_folder(other)

        self.assertEqual(session._image_for(str(photo)).review_status, REVIEW_STATUS_REJECT)

    def test_the_ranking_file_is_recomputed_for_the_new_folder(self):
        shoot, _, _ = build_shoot(self.root, ranked=2)
        session = self.session(shoot)

        other = self.root / "unranked"
        other.mkdir()
        session.open_folder(other)

        self.assertEqual(session.ranking_file, ranking_path(other))


class FolderRelocationTests(SessionTestCase):
    """The folder moved, was renamed, or its drive letter changed - the
    photographer points at the new location and every stored path (the
    ranking, any review decisions) is repointed automatically."""

    def test_folder_missing_is_false_for_a_folder_that_exists(self):
        shoot, _, _ = build_shoot(self.root, ranked=2)
        session = self.session(shoot)

        self.assertFalse(session.folder_missing)

    def test_folder_missing_is_true_once_the_folder_is_gone(self):
        shoot, _, _ = build_shoot(self.root, ranked=2)
        session = self.session(shoot)

        shutil.rmtree(shoot)

        self.assertTrue(session.folder_missing)

    def test_folder_missing_is_false_with_no_folder_open_at_all(self):
        """Distinct states: never having opened a folder is not an error,
        unlike one that was opened and then disappeared."""
        session = self.session(None)

        self.assertFalse(session.folder_missing)

    def test_relocating_repoints_the_ranking_and_reloads_correctly(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=50)

        moved_to = self.root / "moved_shoot"
        shutil.move(str(shoot), str(moved_to))

        result = session.relocate_folder(moved_to)

        self.assertEqual(result["relocated"], 4)
        self.assertEqual(session.input_folder, moved_to.resolve())
        self.assertFalse(session.folder_missing)
        self.assertEqual(session.counts()["total"], 4)
        self.assertEqual(session.counts()["missing_file"], 0, "every path was repointed, nothing looks missing")

    def test_relocating_preserves_ai_suggestions_after_the_move(self):
        """The ranking itself (scores) must survive the move too, not just
        the file existing - otherwise every image would silently lose its
        AI suggestion."""
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=33)
        moved_to = self.root / "moved_shoot"
        shutil.move(str(shoot), str(moved_to))

        session.relocate_folder(moved_to)

        best = str(moved_to / images[0].name)
        self.assertEqual(session._ai_suggestions()[best], REVIEW_STATUS_KEEP)

    def test_relocating_recovers_a_manual_decision_by_identity(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=25)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP, reason="clear_eyes_seen")

        moved_to = self.root / "moved_shoot"
        shutil.move(str(shoot), str(moved_to))
        result = session.relocate_folder(moved_to)

        moved_image = moved_to / images[0].name
        self.assertEqual(session._image_for(str(moved_image)).review_status, REVIEW_STATUS_KEEP)
        self.assertEqual(session._image_for(str(moved_image)).reason, "clear_eyes_seen")
        # Repointed directly via the ranking-derived move map, not the
        # (slower) content-identity fallback - recovered should be 0.
        self.assertEqual(result["recovered"], 0)

    def test_relocating_a_folder_with_no_ranking_still_works(self):
        """The common case for a folder that was never ranked at all -
        nothing to repoint in a ranking that doesn't exist, but the
        photographer's own decisions still follow by content identity."""
        original = self.root / "unranked_shoot"
        original.mkdir()
        photo = original / "a.jpg"
        photo.write_bytes(b"a")
        session = self.session(original)
        session.set_review_status(str(photo), REVIEW_STATUS_REJECT)

        moved_to = self.root / "moved_unranked"
        shutil.move(str(original), str(moved_to))
        result = session.relocate_folder(moved_to)

        self.assertEqual(result["relocated"], 0)
        moved_photo = moved_to / "a.jpg"
        self.assertEqual(session._image_for(str(moved_photo)).review_status, REVIEW_STATUS_REJECT)
        self.assertEqual(result["recovered"], 1)


class NoFolderOpenTests(SessionTestCase):
    """`picklikeme review` with no --input at all: the session has to exist
    and answer every query before any folder has ever been opened."""

    def test_starts_empty_rather_than_failing(self):
        session = self.session(None)

        self.assertIsNone(session.input_folder)
        self.assertIsNone(session.ranking_file)
        self.assertEqual(session.images, [])
        self.assertEqual(session.counts()["total"], 0)
        self.assertTrue(session.warnings, "must prompt the photographer to open one")

    def test_as_dict_is_json_safe_with_nothing_open(self):
        session = self.session(None)

        payload = session.as_dict()

        self.assertIsNone(payload["input_folder"])
        self.assertIsNone(payload["ranking_file"])
        self.assertFalse(payload["has_ranking"])
        self.assertEqual(payload["images"], [])

    def test_arranging_with_nothing_open_is_a_clean_error_not_a_crash(self):
        session = self.session(None)

        with self.assertRaises(ValueError):
            session.arrange(dry_run=True)

    def test_opening_a_folder_afterwards_populates_the_gallery(self):
        session = self.session(None)
        shoot, images, _ = build_shoot(self.root, ranked=2)

        session.open_folder(shoot)

        self.assertEqual(session.input_folder, shoot.resolve())
        self.assertEqual(session.counts()["total"], 2)
        self.assertFalse(session.warnings)


def write_strategy_ranking(target: Path, entries: list[tuple[Path, float]]) -> None:
    """A second analysis module's scores file, in the same format
    `build_shoot` writes the AI's - so a test can attach an independent
    strategy's results to a shoot without depending on `picklikeme.ranking`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow([])
        writer.writerow(["rank", "image_path", "score", "label"])
        for rank, (path, score) in enumerate(entries, start=1):
            writer.writerow([rank, str(path), f"{score:.6f}", 0])


class RankingResultsSymmetryTests(SessionTestCase):
    """Score and rank always belong together as properties of ONE strategy -
    there is no single global rank anywhere in this data model, only
    `ranking_results[strategy_id] = {"score": .., "rank": ..}` per module.

    `ReviewImage.score`/`.rank` (and the AI cutoff, agreement stats, etc. that
    are built on them) are a read-only VIEW onto `ranking_results["ai-model"]`,
    not a second, separately-maintained pair - these tests pin exactly that,
    so a future change cannot let the two drift apart again.
    """

    def test_the_ai_models_score_and_rank_live_together_under_its_own_key(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)

        image = session._image_for(str(images[0]))
        self.assertIn("ai-model", image.ranking_results)
        entry = image.ranking_results["ai-model"]
        self.assertEqual(entry["score"], image.score)
        self.assertEqual(entry["rank"], image.rank)
        self.assertEqual(image.rank, 1)

    def test_a_second_strategys_results_do_not_touch_the_ais(self):
        """The reported motivation: two independent analysis modules coexist
        without one's score/rank pair overwriting the other's."""
        from picklikeme.sidecar import strategy_ranking_path

        shoot, images, _ = build_shoot(self.root, ranked=3)
        # Classic Vision disagrees with the AI about the order entirely.
        write_strategy_ranking(
            strategy_ranking_path(shoot, "classic-vision"),
            [(images[2], 0.99), (images[1], 0.5), (images[0], 0.1)],
        )

        session = self.session(shoot)
        best_by_ai = session._image_for(str(images[0]))

        self.assertEqual(best_by_ai.rank, 1)  # unchanged: still the AI's #1
        self.assertEqual(best_by_ai.ranking_results["ai-model"]["rank"], 1)
        # Its OWN rank under classic-vision is independent and different.
        self.assertEqual(best_by_ai.ranking_results["classic-vision"]["rank"], 3)
        self.assertAlmostEqual(best_by_ai.ranking_results["classic-vision"]["score"], 0.1)

    def test_an_image_only_a_second_strategy_scored_is_unranked_by_ai(self):
        """`score`/`rank`/`is_ranked` mean specifically "the AI model has an
        opinion" - an image with only a Classic Vision result must not read
        as AI-ranked just because SOME module scored it."""
        from picklikeme.sidecar import strategy_ranking_path

        shoot, images, extra = build_shoot(self.root, ranked=2, unranked=1)
        write_strategy_ranking(
            strategy_ranking_path(shoot, "classic-vision"),
            [(extra[0], 0.75)],
        )

        session = self.session(shoot)
        image = session._image_for(str(extra[0]))

        self.assertIsNone(image.score)
        self.assertIsNone(image.rank)
        self.assertFalse(image.is_ranked)
        self.assertEqual(image.ranking_results["classic-vision"], {"score": 0.75, "rank": 1})

    def test_as_dict_exposes_ranking_results_plus_backward_compatible_flat_fields(self):
        """`ranking_results` is the real data; the flat `score`/`rank` keys
        stay in the JSON for whatever already reads them directly (the web
        review UI, existing tests) - but they are DERIVED, not separately
        stored, so they can never disagree with `ranking_results["ai-model"]`.
        """
        shoot, images, _ = build_shoot(self.root, ranked=1)
        session = self.session(shoot)

        payload = session._image_for(str(images[0])).as_dict(ai_suggestion=None)

        self.assertIn("ranking_results", payload)
        self.assertNotIn("scores", payload, "the old key name must not linger")
        self.assertEqual(payload["score"], payload["ranking_results"]["ai-model"]["score"])
        self.assertEqual(payload["rank"], payload["ranking_results"]["ai-model"]["rank"])


class BurstInfoTests(SessionTestCase):
    """ReviewSession.burst_info / as_dict()'s burst_* fields - the session's
    own wiring on top of burst_analysis.analyze_bursts (tested directly, in
    isolation, in test_burst_analysis.py). These fixtures write plain byte
    files with no real EXIF, so captured_at is None throughout unless a test
    sets it directly - every image therefore starts as its own singleton
    burst, which is itself a case worth pinning."""

    def test_every_image_gets_burst_fields_even_with_no_captured_at(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)

        payload = session._image_for(str(images[0])).as_dict(
            ai_suggestion=None, burst=session.burst_info()[str(images[0])]
        )
        self.assertIsNotNone(payload["burst_id"])
        self.assertEqual(payload["burst_size"], 1)
        self.assertEqual(payload["burst_rank"], 1)
        self.assertTrue(payload["burst_best"])

    def test_images_close_in_capture_time_are_grouped_and_ranked_by_ai_score(self):
        """build_shoot's fixture scores descend with index (images[0] best),
        so grouping the first three into one burst must make images[0] the
        burst's best - the AI model is burst_strategy's default."""
        shoot, images, _ = build_shoot(self.root, ranked=5)
        session = self.session(shoot)
        for index, offset in enumerate((0, 1, 2)):
            session._image_for(str(images[index])).captured_at = f"2024-01-01T10:00:0{offset}"

        info = session.burst_info()
        burst_id = info[str(images[0])].burst_id
        self.assertEqual(burst_id, info[str(images[1])].burst_id)
        self.assertEqual(burst_id, info[str(images[2])].burst_id)
        self.assertEqual(info[str(images[0])].burst_size, 3)
        self.assertTrue(info[str(images[0])].burst_best)  # highest AI score of the three
        self.assertFalse(info[str(images[1])].burst_best)
        self.assertFalse(info[str(images[2])].burst_best)
        # Untouched images stay outside that burst entirely.
        self.assertNotEqual(burst_id, info[str(images[3])].burst_id)

    def test_set_burst_strategy_switches_which_score_ranks_the_burst(self):
        """The reported requirement: burst_rank is determined by whichever
        strategy is selected, not hard-coded to the AI model. A second
        strategy that disagrees with the AI about which image is best must
        change burst_best once selected."""
        from picklikeme.sidecar import strategy_ranking_path

        shoot, images, _ = build_shoot(self.root, ranked=3)
        # AI ranks images[0] highest; Classic Vision ranks images[2] highest.
        write_strategy_ranking(
            strategy_ranking_path(shoot, "classic-vision"),
            [(images[2], 0.99), (images[1], 0.5), (images[0], 0.1)],
        )
        session = self.session(shoot)
        for index, offset in enumerate((0, 1, 2)):
            session._image_for(str(images[index])).captured_at = f"2024-01-01T10:00:0{offset}"

        by_ai = session.burst_info()
        self.assertTrue(by_ai[str(images[0])].burst_best)

        session.set_burst_strategy("classic-vision")
        by_classic = session.burst_info()
        self.assertTrue(by_classic[str(images[2])].burst_best)
        self.assertFalse(by_classic[str(images[0])].burst_best)

    def test_as_dict_carries_burst_fields_for_every_image(self):
        shoot, images, _ = build_shoot(self.root, ranked=2)
        session = self.session(shoot)

        payload = session.as_dict()
        for image_payload in payload["images"]:
            self.assertIn("burst_id", image_payload)
            self.assertIn("burst_size", image_payload)
            self.assertIn("burst_rank", image_payload)
            self.assertIn("burst_best", image_payload)


if __name__ == "__main__":
    unittest.main()
