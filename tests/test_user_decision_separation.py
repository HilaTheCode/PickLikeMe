"""Algorithm Result, User Decision and Crop/Detection state are three things.

The regression suite for the bug where they were one. Symptom: a photographer
reviewed ~20 images out of a 5,986-image folder, selected the Color mode for
their own decisions, and every single image was colored as though they had
decided it. Two independent causes, both of which are asserted against here:

1. **The write.** "Apply Cutoff" recorded the ranking's own keep/reject
   through the same call a Grid button click uses, into the same
   `review_decisions` rows, with nothing on the row saying which was which.
   One click turned an entire ranking into what every later reader - Grid
   coloring, the counts, `arrange()` - could only read as "reviewed by hand".
   The fix is `source` (DECISION_SOURCE_USER / DECISION_SOURCE_ALGORITHM) and
   `ReviewImage.user_decision`, which reads user rows alone.

2. **The read.** Grid coloring answered in one blended five-value vocabulary:
   the photographer's decision won, and failing that the selected algorithm's
   binary keep/reject-at-a-threshold suggestion borrowed the very same
   Keep/Reject colors. So an unreviewed image could be painted "Keep" by a
   cutoff, and an algorithm-colored grid was tinted by whatever had been
   reviewed instead of by the scores it claimed to show. The fix is two
   independent modes - `resolve_user_decision` and `resolve_algorithm_state`.

The final test walks the photographer's exact scenario end to end.
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import (  # noqa: E402
    DECISION_SOURCE_ALGORITHM,
    DECISION_SOURCE_USER,
    AnnotationStore,
    InvalidDecisionSource,
)
from picklikeme.desktop.models.image_item import ImageItem  # noqa: E402
from picklikeme.desktop.widgets.design_system import (  # noqa: E402
    ALGORITHM_FILTERED,
    ALGORITHM_SCORED,
    ALGORITHM_SKIPPED,
    USER_DECISION_KEEP,
    USER_DECISION_REJECT,
    USER_DECISION_UNDECIDED,
    resolve_algorithm_state,
    resolve_status,
    resolve_user_decision,
)
from picklikeme.organize import REJECTED_DIRNAME, SELECTED_DIRNAME  # noqa: E402
from picklikeme.review.session import (  # noqa: E402
    REVIEW_STATUS_KEEP,
    REVIEW_STATUS_NEUTRAL,
    REVIEW_STATUS_REJECT,
    ReviewSession,
)
from picklikeme.review.user_decision import KEEP, REJECT, UNDECIDED, is_decided, normalize  # noqa: E402
from picklikeme.sidecar import AI_STRATEGY_ID, SIDECAR_DIRNAME  # noqa: E402

STRATEGY = "crop-sharpness"


def build_ranked_shoot(root: Path, count: int, *, strategy_id: str = STRATEGY) -> list[Path]:
    """A folder of `count` images with a real ranking CSV for `strategy_id` -
    every image scored, nobody reviewed. The state a freshly ranked shoot is
    actually in, which is the state the bug misrepresented."""
    shoot = root / "shoot"
    shoot.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = shoot / f"IMG_{index:04d}.jpg"
        # Distinct bytes per file: decisions are keyed on content identity,
        # so identical files would collapse into one row and hide bugs.
        path.write_bytes(b"peakpick-test-image-" + str(index).encode() * 4)
        paths.append(path)

    sidecar = shoot / SIDECAR_DIRNAME
    sidecar.mkdir(exist_ok=True)
    name = "ranking.csv" if strategy_id == AI_STRATEGY_ID else f"ranking-{strategy_id}.csv"
    with (sidecar / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "score", "image_path"])
        for index, path in enumerate(paths):
            # Descending score, so paths[0] is the best-ranked image.
            writer.writerow([index + 1, f"{1.0 - index / (count + 1):.6f}", str(path)])
    return paths


class SessionCase(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / "annotations.sqlite"
        self.store = AnnotationStore(self.db)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self.store.close)

    def session(self, folder: Path, *, store: AnnotationStore | None = None) -> ReviewSession:
        return ReviewSession(folder, store or self.store)

    def reopened(self, folder: Path) -> ReviewSession:
        """A brand-new store on the same database file - what "a new
        application session" means for anything that persists."""
        store = AnnotationStore(self.db)
        self.addCleanup(store.close)
        return ReviewSession(folder, store)


# ---------------------------------------------------------------------------
# 1-4: the three states, and that they survive.
# ---------------------------------------------------------------------------


class UserDecisionStateTests(SessionCase):
    def test_1_an_unreviewed_image_is_undecided(self):
        """Every image in a fully ranked folder that nobody has reviewed."""
        paths = build_ranked_shoot(self.root, 5)
        session = self.session(paths[0].parent)

        for path in paths:
            image = session._image_for(str(path))
            self.assertEqual(image.user_decision, UNDECIDED)
            self.assertFalse(image.is_decided)
            self.assertIsNone(image.decision, "no stored decision at all")
            self.assertEqual(image.review_status, REVIEW_STATUS_NEUTRAL, "the legacy spelling agrees")
        self.assertEqual(session.counts()["undecided"], 5)
        self.assertEqual(session.counts()["keep"], 0)

    def test_2_keep_creates_a_user_decision_of_keep(self):
        paths = build_ranked_shoot(self.root, 3)
        session = self.session(paths[0].parent)

        session.set_review_status(str(paths[0]), REVIEW_STATUS_KEEP)

        image = session._image_for(str(paths[0]))
        self.assertEqual(image.user_decision, KEEP)
        self.assertEqual(image.decision_source, DECISION_SOURCE_USER)
        self.assertTrue(image.is_decided)
        self.assertEqual(session.keep_paths(), [str(paths[0])])

    def test_3_reject_creates_a_user_decision_of_reject(self):
        paths = build_ranked_shoot(self.root, 3)
        session = self.session(paths[0].parent)

        session.set_review_status(str(paths[1]), REVIEW_STATUS_REJECT)

        image = session._image_for(str(paths[1]))
        self.assertEqual(image.user_decision, REJECT)
        self.assertEqual(image.decision_source, DECISION_SOURCE_USER)
        self.assertEqual(session.reject_paths(), [str(paths[1])])

    def test_4_user_decisions_survive_a_new_application_session(self):
        """Closed and reopened against the same database file, through a
        different AnnotationStore instance - and the untouched images come
        back Undecided rather than defaulting to anything."""
        paths = build_ranked_shoot(self.root, 6)
        shoot = paths[0].parent
        session = self.session(shoot)
        session.set_review_status(str(paths[0]), REVIEW_STATUS_KEEP)
        session.set_review_status(str(paths[1]), REVIEW_STATUS_REJECT)

        reopened = self.reopened(shoot)

        self.assertEqual(reopened._image_for(str(paths[0])).user_decision, KEEP)
        self.assertEqual(reopened._image_for(str(paths[1])).user_decision, REJECT)
        for path in paths[2:]:
            self.assertEqual(reopened._image_for(str(path)).user_decision, UNDECIDED)
        self.assertEqual(reopened.counts(), {**reopened.counts(), "keep": 1, "reject": 1, "undecided": 4})

    def test_4b_clearing_a_decision_returns_the_image_to_undecided(self):
        """Neutral is a real choice that deletes the row - it must not land
        on "whatever the algorithm would have said"."""
        paths = build_ranked_shoot(self.root, 3)
        shoot = paths[0].parent
        session = self.session(shoot)
        session.set_review_status(str(paths[0]), REVIEW_STATUS_KEEP)

        session.set_review_status(str(paths[0]), REVIEW_STATUS_NEUTRAL)

        self.assertEqual(session._image_for(str(paths[0])).user_decision, UNDECIDED)
        self.assertIsNone(session._image_for(str(paths[0])).decision_source)
        self.assertEqual(self.reopened(shoot)._image_for(str(paths[0])).user_decision, UNDECIDED)


# ---------------------------------------------------------------------------
# 5-7, 10-11: coloring reads exactly one mode, and reads nothing else.
# ---------------------------------------------------------------------------


class UserDecisionColoringTests(unittest.TestCase):
    def test_5_user_decision_coloring_colors_only_explicitly_decided_images(self):
        kept = ImageItem(path="/x/a.nef", file_name="a.nef", review_status="keep")
        rejected = ImageItem(path="/x/b.nef", file_name="b.nef", review_status="reject")
        undecided = ImageItem(path="/x/c.nef", file_name="c.nef")

        self.assertEqual(resolve_user_decision(kept), USER_DECISION_KEEP)
        self.assertEqual(resolve_user_decision(rejected), USER_DECISION_REJECT)
        self.assertEqual(resolve_user_decision(undecided), USER_DECISION_UNDECIDED)

    def test_6_undecided_images_stay_neutral_however_the_algorithm_rated_them(self):
        """THE reported symptom, as a unit test: a top-scored, top-ranked,
        cutoff-suggests-Keep image that nobody reviewed is Undecided."""
        top = ImageItem(
            path="/x/top.nef", file_name="top.nef",
            ai_suggestion="keep", algorithm_suggestion="keep", algorithm_decision="keep",
            ranking_results={STRATEGY: {"score": 0.99, "rank": 1}},
        )
        bottom = ImageItem(
            path="/x/bottom.nef", file_name="bottom.nef",
            ai_suggestion="reject", algorithm_suggestion="reject", algorithm_decision="reject",
            ranking_results={STRATEGY: {"score": 0.01, "rank": 999}},
        )

        for item in (top, bottom):
            self.assertEqual(resolve_status(item, None), USER_DECISION_UNDECIDED)

    def test_7_a_ranking_score_never_creates_a_user_decision(self):
        """Neither a score, a rank, a suggestion, a filter reason, a crop
        detection, nor a recorded algorithm cutoff makes an image decided."""
        for kwargs in (
            {"ranking_results": {STRATEGY: {"score": 1.0, "rank": 1}}},
            {"algorithm_suggestion": "keep"},
            {"ai_suggestion": "keep"},
            {"algorithm_decision": "keep"},
            {"filter_reasons": {STRATEGY: "NO_SUBJECT"}},
            {"metrics": {STRATEGY: {"crop_sharpness": 0.9}}},
            {"burst_best": True, "burst_size": 5},
        ):
            with self.subTest(**kwargs):
                item = ImageItem(path="/x/i.nef", file_name="i.nef", **kwargs)
                self.assertFalse(item.is_decided)
                self.assertEqual(item.user_decision, UNDECIDED)
                self.assertEqual(resolve_user_decision(item), USER_DECISION_UNDECIDED)

    def test_10_algorithm_coloring_uses_the_selected_algorithms_own_result(self):
        scored = ImageItem(path="/x/s.nef", file_name="s.nef",
                           ranking_results={STRATEGY: {"score": 0.4, "rank": 2}})
        filtered = ImageItem(path="/x/f.nef", file_name="f.nef",
                             filter_reasons={STRATEGY: "NO_SUBJECT"})
        untouched = ImageItem(path="/x/u.nef", file_name="u.nef",
                              ranking_results={"other-strategy": {"score": 0.9}})

        self.assertEqual(resolve_algorithm_state(scored, STRATEGY), ALGORITHM_SCORED)
        self.assertEqual(resolve_algorithm_state(filtered, STRATEGY), ALGORITHM_FILTERED)
        self.assertEqual(resolve_algorithm_state(untouched, STRATEGY), ALGORITHM_SKIPPED)
        # ...and a different algorithm gives a different answer for the same
        # image, which is what "the selected algorithm's own score" means.
        self.assertEqual(resolve_algorithm_state(untouched, "other-strategy"), ALGORITHM_SCORED)
        self.assertEqual(resolve_algorithm_state(scored, "other-strategy"), ALGORITHM_SKIPPED)

    def test_10b_algorithm_coloring_ignores_the_users_decision(self):
        """The contamination in the other direction: an algorithm mode
        reports what the ALGORITHM did, whatever the photographer said."""
        user_kept = ImageItem(path="/x/a.nef", file_name="a.nef", review_status="keep",
                              filter_reasons={STRATEGY: "NO_SUBJECT"})
        self.assertEqual(resolve_status(user_kept, STRATEGY), ALGORITHM_FILTERED)

    def test_11_changing_color_mode_never_mutates_a_user_decision(self):
        items = [
            ImageItem(path="/x/a.nef", file_name="a.nef", review_status="keep"),
            ImageItem(path="/x/b.nef", file_name="b.nef", review_status="reject"),
            ImageItem(path="/x/c.nef", file_name="c.nef"),
        ]
        before = [item.review_status for item in items]

        for mode in (None, STRATEGY, "other-strategy", AI_STRATEGY_ID, None):
            for item in items:
                resolve_status(item, mode)

        self.assertEqual([item.review_status for item in items], before)


class VocabularyTests(unittest.TestCase):
    def test_normalize_never_invents_a_keep(self):
        self.assertEqual(normalize("keep"), KEEP)
        self.assertEqual(normalize("reject"), REJECT)
        for value in (None, "", "neutral", "undecided", "maybe", "filtered", "skipped"):
            with self.subTest(value=value):
                self.assertEqual(normalize(value), UNDECIDED)
                self.assertFalse(is_decided(value))


# ---------------------------------------------------------------------------
# 8-9, 12: organizing, and the write path that started all this.
# ---------------------------------------------------------------------------


class OrganizeByUserDecisionTests(SessionCase):
    def test_8_organizing_moves_only_explicitly_decided_images(self):
        paths = build_ranked_shoot(self.root, 10)
        shoot = paths[0].parent
        session = self.session(shoot)
        session.set_review_status(str(paths[0]), REVIEW_STATUS_KEEP)
        session.set_review_status(str(paths[1]), REVIEW_STATUS_KEEP)
        session.set_review_status(str(paths[2]), REVIEW_STATUS_REJECT)

        result = session.arrange(dry_run=False)

        self.assertEqual(result.selected, 2)
        self.assertEqual(result.rejected, 1)
        self.assertEqual(result.moved, 3)
        self.assertTrue((shoot / SELECTED_DIRNAME / paths[0].name).is_file())
        self.assertTrue((shoot / SELECTED_DIRNAME / paths[1].name).is_file())
        self.assertTrue((shoot / REJECTED_DIRNAME / paths[2].name).is_file())

    def test_9_undecided_images_are_never_moved(self):
        paths = build_ranked_shoot(self.root, 10)
        shoot = paths[0].parent
        session = self.session(shoot)
        session.set_review_status(str(paths[0]), REVIEW_STATUS_KEEP)

        session.arrange(dry_run=False)

        for path in paths[1:]:
            self.assertTrue(path.is_file(), f"{path.name} was moved despite being Undecided")

    def test_9b_a_recorded_algorithm_cutoff_organizes_nothing(self):
        """The scenario that would have filed 5,986 images: rank a folder,
        Apply Cutoff, then Organize. Nothing is a candidate."""
        paths = build_ranked_shoot(self.root, 10)
        shoot = paths[0].parent
        session = self.session(shoot)
        session.set_keep_percent(30)

        applied = session.apply_algorithm_suggestions(STRATEGY)
        self.assertEqual(applied["applied"], 10, "all ten carry the cutoff's own decision")

        preview = session.arrange(dry_run=True)

        self.assertEqual((preview.selected, preview.rejected, preview.moved), (0, 0, 0))
        self.assertEqual(session.keep_paths(), [])
        for path in paths:
            self.assertTrue(path.is_file())

    def test_9c_batch_review_organizes_only_the_batch(self):
        """The workflow this protects: decide a batch, organize the batch,
        come back and decide more."""
        paths = build_ranked_shoot(self.root, 12)
        shoot = paths[0].parent
        session = self.session(shoot)

        for path in paths[:3]:
            session.set_review_status(str(path), REVIEW_STATUS_KEEP)
        first = session.arrange(dry_run=False)
        self.assertEqual(first.moved, 3)

        for path in paths[3:5]:
            session.set_review_status(str(path), REVIEW_STATUS_REJECT)
        second = session.arrange(dry_run=False)

        self.assertEqual(second.moved, 2, "only the newly decided images")
        self.assertEqual(len(list((shoot / SELECTED_DIRNAME).iterdir())), 3)
        self.assertEqual(len(list((shoot / REJECTED_DIRNAME).iterdir())), 2)
        for path in paths[5:]:
            self.assertTrue(path.is_file(), "still undecided, still where it was")


class DecisionSourceTests(SessionCase):
    def test_12_applying_a_cutoff_never_becomes_a_user_decision(self):
        paths = build_ranked_shoot(self.root, 10)
        shoot = paths[0].parent
        session = self.session(shoot)
        session.set_keep_percent(30)

        session.apply_algorithm_suggestions(STRATEGY)

        rows = self.store.review_decisions()
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["source"] == DECISION_SOURCE_ALGORITHM for row in rows))
        self.assertEqual(session.counts()["keep"], 0)
        self.assertEqual(session.counts()["reject"], 0)
        self.assertEqual(session.counts()["undecided"], 10)
        self.assertEqual(session.counts()["algorithm_decisions"], 10)

    def test_12b_a_cutoff_survives_a_reload_still_as_an_algorithm_decision(self):
        """Reloading must not launder an algorithm decision into a user one -
        the read path has to carry `source` too, not just the write."""
        paths = build_ranked_shoot(self.root, 6)
        shoot = paths[0].parent
        session = self.session(shoot)
        session.apply_algorithm_suggestions(STRATEGY)

        reopened = self.reopened(shoot)

        for path in paths:
            image = reopened._image_for(str(path))
            self.assertIn(image.algorithm_decision, (REVIEW_STATUS_KEEP, REVIEW_STATUS_REJECT))
            self.assertEqual(image.user_decision, UNDECIDED)
        self.assertEqual(reopened.keep_paths(), [])

    def test_12c_a_cutoff_never_overwrites_a_user_decision(self):
        paths = build_ranked_shoot(self.root, 10)
        session = self.session(paths[0].parent)
        session.set_keep_percent(30)
        session.set_review_status(str(paths[0]), REVIEW_STATUS_REJECT)  # disagrees on purpose

        result = session.apply_algorithm_suggestions(STRATEGY, include_decided=True)

        self.assertEqual(result["conflicts"], 1)
        self.assertEqual(result["overridden"], 0)
        self.assertEqual(session._image_for(str(paths[0])).user_decision, REJECT)

    def test_12d_clearing_algorithm_decisions_keeps_the_users_own(self):
        paths = build_ranked_shoot(self.root, 8)
        shoot = paths[0].parent
        session = self.session(shoot)
        session.set_review_status(str(paths[0]), REVIEW_STATUS_KEEP)
        session.set_review_status(str(paths[1]), REVIEW_STATUS_REJECT)
        session.apply_algorithm_suggestions(STRATEGY)

        removed = session.clear_algorithm_decisions()

        self.assertEqual(removed, 6, "the six that had no user decision")
        self.assertEqual(session.keep_paths(), [str(paths[0])])
        self.assertEqual(session.reject_paths(), [str(paths[1])])
        self.assertEqual(self.store.review_decision_count(), 2)
        self.assertEqual(self.store.review_decision_count(source=DECISION_SOURCE_ALGORITHM), 0)

    def test_the_store_rejects_an_unknown_source(self):
        image = self.root / "one.jpg"
        image.write_bytes(b"one")
        with self.assertRaises(InvalidDecisionSource):
            self.store.set_review_decision(image, "keep", source="the-vibes")

    def test_a_database_written_before_source_existed_backfills_to_user(self):
        """A pre-existing row's origin genuinely cannot be recovered, so it
        backfills to `user` - never silently discarding real work. Simulated
        by dropping the column and reopening, which is what an older
        database looks like on disk."""
        image = self.root / "legacy.jpg"
        image.write_bytes(b"legacy")
        self.store.set_review_decision(image, "keep")
        self.store._conn.execute("ALTER TABLE review_decisions DROP COLUMN source")
        self.store._conn.commit()
        self.store.close()

        upgraded = AnnotationStore(self.db)
        self.addCleanup(upgraded.close)

        rows = upgraded.review_decisions()
        self.assertEqual([row["source"] for row in rows], [DECISION_SOURCE_USER])


# ---------------------------------------------------------------------------
# The reported scenario, end to end.
# ---------------------------------------------------------------------------


class ReportedScenarioTests(SessionCase):
    """~20 reviewed images among many ranked-but-unreviewed ones - a small
    controlled stand-in for the photographer's 5,986-image folder."""

    IMAGES = 120
    REVIEWED = 20

    def setUp(self) -> None:
        super().setUp()
        self.paths = build_ranked_shoot(self.root, self.IMAGES)
        self.shoot = self.paths[0].parent
        self.session_ = self.session(self.shoot)
        # A cutoff was applied at some point, exactly as it was on the real
        # folder - this must not survive as anybody's decision.
        self.session_.set_keep_percent(25)
        self.session_.apply_algorithm_suggestions(STRATEGY)
        # ...and then 20 images were actually reviewed, keep/reject alternating.
        self.reviewed = self.paths[: self.REVIEWED]
        for index, path in enumerate(self.reviewed):
            self.session_.set_review_status(
                str(path), REVIEW_STATUS_KEEP if index % 2 == 0 else REVIEW_STATUS_REJECT
            )
        self.unreviewed = self.paths[self.REVIEWED:]

    def items(self, session: ReviewSession) -> list[ImageItem]:
        """The desktop's own view of the session - the exact conversion
        MainWindow._refresh_from_state performs."""
        return [
            ImageItem(
                path=image["image_path"],
                file_name=Path(image["image_path"]).name,
                review_status=image["review_status"],
                algorithm_decision=image["algorithm_decision"],
                ai_suggestion=image["ai_suggestion"],
                algorithm_suggestion=image["algorithm_suggestion"],
                ranking_results=image["ranking_results"],
                filter_reasons=image["filter_reasons"],
            )
            for image in session.as_dict()["images"]
        ]

    def test_only_the_reviewed_images_receive_user_decision_colors(self):
        by_path = {item.path: item for item in self.items(self.session_)}

        colored = [p for p, item in by_path.items()
                   if resolve_status(item, None) != USER_DECISION_UNDECIDED]

        self.assertEqual(len(colored), self.REVIEWED)
        self.assertEqual(sorted(colored), sorted(str(p) for p in self.reviewed))

    def test_every_unreviewed_image_stays_neutral(self):
        by_path = {item.path: item for item in self.items(self.session_)}

        for path in self.unreviewed:
            item = by_path[str(path)]
            self.assertEqual(resolve_status(item, None), USER_DECISION_UNDECIDED)
            self.assertIsNotNone(item.score_for(STRATEGY), "it IS ranked - that is the point")

    def test_algorithm_coloring_is_independent_of_who_was_reviewed(self):
        items = self.items(self.session_)

        states = {resolve_status(item, STRATEGY) for item in items}

        self.assertEqual(states, {ALGORITHM_SCORED}, "every image was scored by this strategy")

    def test_organizing_touches_only_the_twenty_decided_images(self):
        preview = self.session_.arrange(dry_run=True)

        self.assertEqual(preview.selected + preview.rejected, self.REVIEWED)
        self.assertEqual(preview.selected, 10)
        self.assertEqual(preview.rejected, 10)

    def test_it_all_survives_a_new_session(self):
        reopened = self.reopened(self.shoot)

        self.assertEqual(reopened.counts()["keep"], 10)
        self.assertEqual(reopened.counts()["reject"], 10)
        self.assertEqual(reopened.counts()["undecided"], self.IMAGES - self.REVIEWED)
        by_path = {item.path: item for item in self.items(reopened)}
        for path in self.unreviewed:
            self.assertEqual(resolve_status(by_path[str(path)], None), USER_DECISION_UNDECIDED)

    def test_changing_ranking_or_sorting_never_mutates_a_user_decision(self):
        """Requirement 12: re-ranking, re-scoring and re-sorting are display
        and analysis operations. None of them writes a decision."""
        before = {p: self.session_._image_for(str(p)).user_decision for p in self.paths}

        self.session_.set_keep_percent(5)
        self.session_.set_burst_strategy(AI_STRATEGY_ID)
        self.session_.suggestions_for(STRATEGY)
        self.session_.as_dict()
        self.session_.agreement_stats()
        self.session_.load()

        after = {p: self.session_._image_for(str(p)).user_decision for p in self.paths}
        self.assertEqual(after, before)
        self.assertEqual(sum(1 for v in after.values() if v != UNDECIDED), self.REVIEWED)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
