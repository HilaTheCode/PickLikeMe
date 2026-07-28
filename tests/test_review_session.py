"""What the review application decides, before any of it reaches a browser.

The rules that matter, and that the rest of the app depends on being exact:

- a manual Keep/Reject always beats the threshold, however the threshold moves;
- an image with no ranking is never selected automatically and never filed
  without an explicit decision;
- an image on disk but absent from the ranking still appears in the gallery;
- a decision, once made, survives arranging - the files move underneath it.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import AnnotationStore
from picklikeme.organize import REJECTED_DIRNAME, SELECTED_DIRNAME, selection_count
from picklikeme.review.session import (
    STATE_AUTO_REJECTED,
    STATE_AUTO_SELECTED,
    STATE_MANUAL_KEEP,
    STATE_MANUAL_REJECT,
    STATE_UNRANKED,
    ReviewSession,
)
from picklikeme.sidecar import ranking_path


def build_shoot(root: Path, ranked: int = 10, unranked: int = 0) -> tuple[Path, list[Path], list[Path]]:
    """A folder with a ranking, plus optionally images the ranking never saw.

    Scores descend with the index, so `images[0]` is the model's best pick and
    `images[-1]` its worst - which is what makes the override tests legible.
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


class ThresholdTests(SessionTestCase):
    def test_the_cut_matches_organize_s_own_arithmetic(self):
        """The UI's count and the arrange must never disagree, so both come
        from selection_count rather than each rounding for themselves."""
        shoot, _, _ = build_shoot(self.root, ranked=10)
        for percent in (0, 5, 10, 25, 33.3, 50, 100):
            session = self.session(shoot, keep_percent=percent)
            self.assertEqual(session.cut, selection_count(10, percent))
            self.assertEqual(len(session.selected_paths()), session.cut)

    def test_the_highest_scoring_images_are_the_selected_ones(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=30)
        selected = session.selected_paths()
        self.assertEqual(selected, [str(p) for p in images[:3]])

    def test_moving_the_percentage_reclassifies_without_any_inference(self):
        shoot, _, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=10)
        self.assertEqual(session.counts()["selected"], 1)
        session.set_keep_percent(50)
        self.assertEqual(session.counts()["selected"], 5)
        session.set_keep_percent(0)
        self.assertEqual(session.counts()["selected"], 0)
        self.assertEqual(session.counts()["rejected"], 10)

    def test_states_explain_every_image(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        states = self.session(shoot, keep_percent=50).states()
        self.assertEqual(states[str(images[0])], STATE_AUTO_SELECTED)
        self.assertEqual(states[str(images[3])], STATE_AUTO_REJECTED)


class ManualOverrideTests(SessionTestCase):
    def test_a_manual_keep_beats_the_threshold(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=10)
        worst = str(images[-1])

        state = session.set_decision(worst, "keep")

        self.assertEqual(state, STATE_MANUAL_KEEP)
        self.assertIn(worst, session.selected_paths())
        self.assertEqual(session.counts()["selected"], 2, "the auto pick plus the manual one")

    def test_a_manual_reject_beats_the_threshold(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=50)
        best = str(images[0])

        self.assertEqual(session.set_decision(best, "reject"), STATE_MANUAL_REJECT)
        self.assertIn(best, session.rejected_paths())
        self.assertNotIn(best, session.selected_paths())

    def test_manual_decisions_survive_the_percentage_changing(self):
        """The threshold re-sorts everything the photographer has not ruled on,
        and nothing they have."""
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=10)
        session.set_decision(str(images[-1]), "keep")
        session.set_decision(str(images[0]), "reject")

        for percent in (0, 25, 50, 100):
            session.set_keep_percent(percent)
            states = session.states()
            self.assertEqual(states[str(images[-1])], STATE_MANUAL_KEEP)
            self.assertEqual(states[str(images[0])], STATE_MANUAL_REJECT)

    def test_clearing_a_decision_returns_the_image_to_the_threshold(self):
        shoot, images, _ = build_shoot(self.root, ranked=10)
        session = self.session(shoot, keep_percent=50)
        best = str(images[0])
        session.set_decision(best, "reject")
        self.assertEqual(session.states()[best], STATE_MANUAL_REJECT)

        self.assertEqual(session.set_decision(best, None), STATE_AUTO_SELECTED)
        self.assertEqual(session.counts()["manual"], 0)

    def test_decisions_are_persisted_immediately_not_held_in_memory(self):
        """Refreshing the page must restore everything; a browser tab is not
        where a photographer's work is allowed to live."""
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=25)
        session.set_decision(str(images[4]), "keep")

        reopened = self.session(shoot, keep_percent=25)

        self.assertEqual(reopened.states()[str(images[4])], STATE_MANUAL_KEEP)
        self.assertEqual(reopened.counts()["manual"], 1)

    def test_an_unknown_path_is_refused(self):
        shoot, _, _ = build_shoot(self.root, ranked=3)
        session = self.session(shoot)
        with self.assertRaises(KeyError):
            session.set_decision(str(self.root / "elsewhere.jpg"), "keep")


class MissingDataTests(SessionTestCase):
    def test_an_image_absent_from_the_ranking_still_appears(self):
        shoot, ranked, extra = build_shoot(self.root, ranked=4, unranked=2)
        session = self.session(shoot, keep_percent=50)

        self.assertEqual(session.counts()["total"], 6)
        states = session.states()
        for path in extra:
            self.assertEqual(states[str(path)], STATE_UNRANKED)

    def test_unranked_images_are_never_selected_automatically(self):
        shoot, _, extra = build_shoot(self.root, ranked=4, unranked=3)
        session = self.session(shoot, keep_percent=100)

        for path in extra:
            self.assertNotIn(str(path), session.selected_paths())
            self.assertNotIn(str(path), session.rejected_paths())
        self.assertEqual(session.counts()["untouched"], 3)

    def test_a_manual_decision_gives_an_unranked_image_a_destination(self):
        shoot, _, extra = build_shoot(self.root, ranked=4, unranked=1)
        session = self.session(shoot, keep_percent=25)

        session.set_decision(str(extra[0]), "keep")

        self.assertIn(str(extra[0]), session.selected_paths())
        self.assertEqual(session.counts()["untouched"], 0)

    def test_a_ranking_row_whose_file_is_gone_is_shown_not_dropped(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        images[1].unlink()

        session = self.session(shoot, keep_percent=50)

        self.assertEqual(session.counts()["total"], 4, "the row is kept so the gap is visible")
        self.assertEqual(session.counts()["missing_file"], 1)

    def test_a_folder_with_no_ranking_still_opens(self):
        """Every image unranked is a reviewable state, not a crash."""
        shoot = self.root / "bare"
        shoot.mkdir()
        (shoot / "a.jpg").write_bytes(b"a")

        session = self.session(shoot)

        self.assertEqual(session.counts()["total"], 1)
        self.assertEqual(session.counts()["untouched"], 1)
        self.assertTrue(session.warnings)

    def test_an_unreadable_ranking_degrades_instead_of_failing(self):
        shoot, _, _ = build_shoot(self.root, ranked=3)
        ranking_path(shoot).write_text("this is not a ranking", encoding="utf-8")

        session = self.session(shoot)

        self.assertEqual(session.counts()["total"], 3, "images are still found on disk")
        self.assertEqual(session.counts()["untouched"], 3)
        self.assertTrue(any("ranking" in w.lower() for w in session.warnings))


class ArrangeTests(SessionTestCase):
    def test_dry_run_reports_the_plan_and_moves_nothing(self):
        shoot, images, _ = build_shoot(self.root, ranked=8)
        session = self.session(shoot, keep_percent=25)

        result = session.arrange(dry_run=True)

        self.assertEqual(result.selected, 2)
        self.assertEqual(result.rejected, 6)
        for path in images:
            self.assertTrue(path.exists(), "a dry run must not move a file")
        self.assertFalse((shoot / SELECTED_DIRNAME).exists())

    def test_arranging_files_by_the_final_verdict_including_overrides(self):
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=33)
        session.set_decision(str(images[5]), "keep")   # worst, kept anyway
        session.set_decision(str(images[0]), "reject")  # best, rejected anyway

        session.arrange()

        self.assertTrue((shoot / SELECTED_DIRNAME / "IMG_0005.jpg").exists())
        self.assertTrue((shoot / REJECTED_DIRNAME / "IMG_0000.jpg").exists())

    def test_unranked_undecided_images_are_left_where_they_are(self):
        shoot, _, extra = build_shoot(self.root, ranked=4, unranked=2)
        session = self.session(shoot, keep_percent=50)

        result = session.arrange()

        self.assertEqual(result.ranked, 4, "only the ranked images were filed")
        for path in extra:
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, shoot)

    def test_decisions_and_ranking_both_follow_the_files(self):
        """The load-bearing re-review test: arrange moves everything, so both
        the ranking and the stored decisions must be repointed or the next
        review would look like a folder nobody had ever touched."""
        shoot, images, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=33)
        session.set_decision(str(images[5]), "keep")

        session.arrange()
        reopened = self.session(shoot, keep_percent=33)

        self.assertEqual(reopened.counts()["total"], 6, "the arranged files are found again")
        self.assertEqual(reopened.counts()["missing_file"], 0, "the ranking was repointed")
        self.assertEqual(reopened.counts()["manual"], 1, "the decision followed its file")
        moved = shoot / SELECTED_DIRNAME / "IMG_0005.jpg"
        self.assertEqual(reopened.states()[str(moved)], STATE_MANUAL_KEEP)

    def test_arranging_twice_is_a_no_op_rather_than_a_shuffle(self):
        shoot, _, _ = build_shoot(self.root, ranked=6)
        session = self.session(shoot, keep_percent=50)
        session.arrange()

        again = self.session(shoot, keep_percent=50).arrange()

        self.assertEqual(again.moved, 0)
        self.assertEqual(again.errors, 0)


class IdentityRecoveryTests(SessionTestCase):
    def test_a_decision_is_recovered_after_the_file_moves_behind_our_back(self):
        """Path matching is the fast path; content identity is the truth. A
        file moved outside the app still carries its decision."""
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=25)
        session.set_decision(str(images[3]), "keep")

        moved = shoot / "renamed_by_hand.jpg"
        images[3].rename(moved)

        reopened = self.session(shoot, keep_percent=25)
        self.assertIsNone(reopened._image_for(str(moved)).decision, "path match cannot find it")

        recovered = reopened.reconcile_by_identity()

        self.assertEqual(recovered, 1)
        self.assertEqual(reopened.states()[str(moved)], STATE_MANUAL_KEEP)

    def test_reconciling_costs_nothing_when_nothing_moved(self):
        shoot, images, _ = build_shoot(self.root, ranked=4)
        session = self.session(shoot, keep_percent=25)
        session.set_decision(str(images[0]), "keep")

        reopened = self.session(shoot, keep_percent=25)

        self.assertEqual(reopened.reconcile_by_identity(), 0)


if __name__ == "__main__":
    unittest.main()
