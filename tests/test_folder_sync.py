"""The Grid must show what is on the disk, right now - nothing else.

The bug these tests exist for: `ReviewSession.load` used to build the gallery
from every strategy CSV first and then *add* whatever the folder scan found
that the CSVs had missed. Membership therefore came from a file written in the
past, and the disk could only ever grow it. Three consequences, all observed on
a real 5,986-image archive:

- a photograph filed away by Arrange stayed in the grid, because removing it
  from the folder cannot remove it from last week's CSV;
- worse, when it was still findable at its new path the SAME photograph
  appeared TWICE - once as its stale ranked row (scored, pointing at a file no
  longer there) and once as a freshly-enumerated on-disk row (real, but
  matching no CSV row, therefore scoreless and Undecided). "An undecided image
  with no score sitting next to an identical scored one" is that pair;
- opening a folder could not clear either, because a reopen ran the same load.

The rule now: the folder scan decides MEMBERSHIP, the rankings only supply
SCORES. These tests pin that in both directions - present-and-ranked, and
absent-but-still-in-the-CSV - across Arrange, manual Refresh, and Open Folder.

Persistence is the other half: an absent file's ranking row and its stored
decision are deliberately NOT deleted. They are simply not rendered, and they
come back with the file.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import AnnotationStore
from picklikeme.organize import REJECTED_DIRNAME, SELECTED_DIRNAME
from picklikeme.review.session import REVIEW_STATUS_KEEP, REVIEW_STATUS_REJECT, ReviewSession
from picklikeme.review.user_decision import KEEP, UNDECIDED
from picklikeme.sidecar import ranking_path


def _write_ranking(shoot: Path, paths: list[Path]) -> None:
    target = ranking_path(shoot)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["select_root", str(shoot)])
        writer.writerow([])
        writer.writerow(["rank", "image_path", "score", "label"])
        for rank, path in enumerate(paths, start=1):
            writer.writerow([rank, str(path), f"{1.0 - rank * 0.05:.6f}", 0])


class FolderSyncTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.store = AnnotationStore(self.root / "kb.db")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def shoot(self, name: str = "shoot", count: int = 5, *, ranked: bool = True) -> tuple[Path, list[Path]]:
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        images = []
        for index in range(count):
            target = folder / f"IMG_{index:04d}.jpg"
            target.write_bytes(f"{name} frame {index}".encode())
            images.append(target)
        if ranked:
            _write_ranking(folder, images)
        return folder, images

    def names(self, session: ReviewSession) -> set[str]:
        return {Path(image.image_path).name for image in session.images}

    def paths(self, session: ReviewSession) -> set[Path]:
        return {Path(image.image_path).resolve() for image in session.images}


# ---------------------------------------------------------------------------
# 1 + 2. Arrange moves files -> the grid follows, immediately
# ---------------------------------------------------------------------------


class ArrangeRefreshesTheGridTests(FolderSyncTestCase):
    """Arrange files Keep/Reject into `_Selected`/`_Rejected` UNDER the open
    folder, so those images legitimately stay in the grid at their new paths
    (the Open Folder contract has always included subfolders, so an organized
    shoot can be re-reviewed). What must not survive is the OLD path.
    """

    def test_arrange_removes_the_old_paths_and_keeps_the_new_ones(self):
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)
        session.set_review_status(str(images[1]), REVIEW_STATUS_REJECT)
        before = self.paths(session)

        session.arrange()

        after = self.paths(session)
        self.assertNotIn(images[0].resolve(), after, "the pre-arrange path must be gone")
        self.assertNotIn(images[1].resolve(), after)
        self.assertIn((shoot / SELECTED_DIRNAME / images[0].name).resolve(), after)
        self.assertIn((shoot / REJECTED_DIRNAME / images[1].name).resolve(), after)
        self.assertEqual(len(after), len(before), "moved, not duplicated and not lost")

    def test_no_image_ever_appears_twice_after_arrange(self):
        """THE duplicate. Under the old load order the moved file appeared at
        both its old (ranked) and new (on-disk) path."""
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)

        session.arrange()

        filenames = [Path(image.image_path).name for image in session.images]
        self.assertEqual(len(filenames), len(set(filenames)), filenames)

    def test_undecided_files_stay_exactly_where_they_are_and_stay_visible(self):
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)

        session.arrange()

        for untouched in images[1:]:
            self.assertTrue(untouched.is_file(), "an undecided file is never moved")
            self.assertIn(untouched.resolve(), self.paths(session), "and never disappears")

    def test_a_file_arranged_out_of_the_tree_entirely_leaves_the_grid(self):
        """The stronger case: Arrange keeps its output under the open folder,
        but any move OUT of it (a sibling `_Selected`, Finder, another tool)
        must drop the image from the grid, and it is the same code path."""
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        elsewhere = self.root / "somewhere_else"
        elsewhere.mkdir()
        images[0].rename(elsewhere / images[0].name)

        session.refresh()

        self.assertNotIn(images[0].name, self.names(session))
        self.assertEqual(len(session.images), 3)

    def test_arrange_leaves_the_moved_images_decisions_intact(self):
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)

        session.arrange()

        moved = next(i for i in session.images if i.filename == images[0].name)
        self.assertEqual(moved.user_decision, KEEP, "the decision follows the file")


# ---------------------------------------------------------------------------
# 3 + 4. Manual Refresh
# ---------------------------------------------------------------------------


class ManualRefreshTests(FolderSyncTestCase):
    def test_refresh_picks_up_files_added_externally(self):
        shoot, images = self.shoot(count=3)
        session = ReviewSession(shoot, self.store)
        self.assertEqual(len(session.images), 3)

        (shoot / "NEW_0001.jpg").write_bytes(b"dropped in by Finder")
        session.refresh()

        self.assertIn("NEW_0001.jpg", self.names(session))
        self.assertEqual(len(session.images), 4)

    def test_refresh_drops_files_deleted_externally(self):
        shoot, images = self.shoot(count=3)
        session = ReviewSession(shoot, self.store)

        images[1].unlink()
        session.refresh()

        self.assertNotIn(images[1].name, self.names(session))
        self.assertEqual(len(session.images), 2)

    def test_refresh_drops_a_file_that_is_still_named_by_the_ranking_csv(self):
        """The precise failure: the CSV still lists it, so the old load added
        it back on every single reload."""
        shoot, images = self.shoot(count=3)
        images[0].unlink()

        session = ReviewSession(shoot, self.store)

        self.assertNotIn(images[0].name, self.names(session))
        ranking_text = ranking_path(shoot).read_text(encoding="utf-8")
        self.assertIn(images[0].name, ranking_text, "the CSV row is deliberately NOT deleted")

    def test_a_returning_file_gets_its_score_and_decision_back(self):
        """Requirement 4: history is preserved, not rendered while absent."""
        shoot, images = self.shoot(count=3)
        session = ReviewSession(shoot, self.store)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)
        scored = next(i for i in session.images if i.filename == images[0].name)
        original_score = scored.score
        self.assertIsNotNone(original_score)

        parked = self.root / "parked.jpg"
        images[0].rename(parked)
        session.refresh()
        self.assertNotIn(images[0].name, self.names(session))

        parked.rename(images[0])
        session.refresh()

        restored = next(i for i in session.images if i.filename == images[0].name)
        self.assertEqual(restored.score, original_score, "the score came back with the file")
        self.assertEqual(restored.user_decision, KEEP, "and so did the decision")

    def test_refresh_is_a_no_op_when_no_folder_is_open(self):
        session = ReviewSession(None, self.store)
        session.refresh()
        self.assertEqual(session.images, [])

    def test_a_refresh_is_never_undone_by_the_previous_loads_background_pass(self):
        """A SECOND cause of the same symptom, found while testing the first.

        `load` publishes the gallery synchronously and then starts a
        background pass to fill in capture times and categories. That pass
        snapshots the image list when it starts and periodically writes the
        snapshot back. A refresh that lands while the previous pass is still
        running would therefore have its result overwritten by the older
        snapshot - putting the just-removed files straight back into the grid.

        This is exactly the "Refresh did nothing" case, and it is timing
        dependent, so it is asserted by driving the collision directly rather
        than by hoping the scheduler reproduces it.
        """
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        stale_snapshot = list(session.images)
        self.assertEqual(len(stale_snapshot), 4)
        stale_generation = session._loading_generation

        images[0].unlink()
        session.refresh()
        self.assertEqual(len(session.images), 3)

        # Replay the old pass's write-back, exactly as _background_load does.
        with session._state_lock:
            if stale_generation == session._loading_generation:
                session.images = stale_snapshot

        self.assertEqual(len(session.images), 3, "the stale snapshot must not win")
        self.assertNotIn(images[0].name, self.names(session))

    def test_repeated_refreshes_are_stable(self):
        shoot, _ = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        first = self.paths(session)
        for _ in range(3):
            session.refresh()
        self.assertEqual(self.paths(session), first)


# ---------------------------------------------------------------------------
# 5 + 6 + 8. Open Folder starts from the disk
# ---------------------------------------------------------------------------


class OpenFolderStartsFromTheDiskTests(FolderSyncTestCase):
    def test_open_folder_shows_exactly_the_supported_files_present_now(self):
        shoot, images = self.shoot(count=4)
        images[2].unlink()
        (shoot / "notes.txt").write_text("not an image")

        session = ReviewSession(shoot, self.store)

        self.assertEqual(
            self.names(session), {p.name for p in images if p.is_file()},
        )
        self.assertNotIn("notes.txt", self.names(session))

    def test_opening_a_second_folder_retains_nothing_from_the_first(self):
        first, first_images = self.shoot("first", count=3)
        second, second_images = self.shoot("second", count=2)

        session = ReviewSession(first, self.store)
        self.assertEqual(self.names(session), {p.name for p in first_images})

        session.open_folder(second)

        self.assertEqual(self.names(session), {p.name for p in second_images})
        self.assertEqual(
            self.paths(session), {p.resolve() for p in second_images},
            "not one path from the first folder survives",
        )

    def test_reopening_the_first_folder_after_files_moved_away_shows_the_remainder(self):
        first, first_images = self.shoot("first", count=3)
        second, _ = self.shoot("second", count=1)
        session = ReviewSession(first, self.store)

        first_images[0].rename(second / first_images[0].name)
        session.open_folder(second)
        session.open_folder(first)

        self.assertNotIn(first_images[0].name, self.names(session))
        self.assertEqual(len(session.images), 2)

    def test_an_unranked_folder_still_shows_every_file_on_disk(self):
        """Membership comes from the scan, so a folder nothing has ever
        ranked is fully visible - the behaviour that made the disk scan
        additive in the first place, preserved."""
        shoot, images = self.shoot(count=3, ranked=False)

        session = ReviewSession(shoot, self.store)

        self.assertEqual(self.names(session), {p.name for p in images})
        self.assertTrue(all(i.score is None for i in session.images))

    def test_images_on_disk_but_absent_from_the_ranking_still_appear(self):
        shoot, images = self.shoot(count=3)
        extra = shoot / "UNRANKED.jpg"
        extra.write_bytes(b"never ranked")

        session = ReviewSession(shoot, self.store)

        self.assertIn("UNRANKED.jpg", self.names(session))
        self.assertEqual(len(session.images), 4)


# ---------------------------------------------------------------------------
# 7. Refresh and Open Folder mutate nothing
# ---------------------------------------------------------------------------


class RefreshMutatesNothingTests(FolderSyncTestCase):
    def test_refresh_preserves_every_user_decision(self):
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)
        session.set_review_status(str(images[1]), REVIEW_STATUS_REJECT)

        before = {i.filename: i.user_decision for i in session.images}
        session.refresh()
        after = {i.filename: i.user_decision for i in session.images}

        self.assertEqual(before, after)
        self.assertEqual(after[images[0].name], KEEP)
        self.assertEqual(after[images[2].name], UNDECIDED, "and an undecided image stays undecided")

    def test_refresh_preserves_scores_and_ranks(self):
        shoot, _ = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)

        before = {i.filename: (i.score, i.rank, dict(i.ranking_results)) for i in session.images}
        session.refresh()
        after = {i.filename: (i.score, i.rank, dict(i.ranking_results)) for i in session.images}

        self.assertEqual(before, after)

    def test_refresh_does_not_rewrite_the_ranking_csv(self):
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        images[0].unlink()
        before = ranking_path(shoot).read_bytes()

        session.refresh()

        self.assertEqual(ranking_path(shoot).read_bytes(), before, "refresh is read-only on disk")

    def test_refresh_does_not_add_or_remove_stored_decisions(self):
        shoot, images = self.shoot(count=4)
        session = ReviewSession(shoot, self.store)
        session.set_review_status(str(images[0]), REVIEW_STATUS_KEEP)
        before = self.store.review_decision_count()

        session.refresh()
        images[1].unlink()
        session.refresh()

        self.assertEqual(
            self.store.review_decision_count(), before,
            "an absent file's decision is preserved, not purged",
        )

    def test_opening_another_folder_preserves_the_first_folders_decisions(self):
        first, first_images = self.shoot("first", count=3)
        second, _ = self.shoot("second", count=2)
        session = ReviewSession(first, self.store)
        session.set_review_status(str(first_images[0]), REVIEW_STATUS_KEEP)

        session.open_folder(second)
        session.open_folder(first)

        restored = next(i for i in session.images if i.filename == first_images[0].name)
        self.assertEqual(restored.user_decision, KEEP)

    def test_refresh_writes_nothing_at_all_under_the_folder(self):
        """The broad guarantee, checked by snapshot rather than by reading
        the implementation: whatever refresh touches, it does not WRITE.
        Covers the ranking CSV, the run/filter/metric sidecars and anything
        else living under the shoot, in one assertion that keeps holding as
        those files change shape.
        """
        shoot, images = self.shoot(count=4)
        crop_cache = self.root / "crops"
        crop_cache.mkdir()
        (crop_cache / "aa.jpg").write_bytes(b"a cached crop")
        (crop_cache / "crop_params.json").write_text('{"version": "v9"}')

        def snapshot(folder: Path) -> dict[str, tuple[int, int]]:
            return {
                str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(folder.rglob("*"))
                if p.is_file()
            }

        session = ReviewSession(shoot, self.store)
        before_shoot, before_cache = snapshot(shoot), snapshot(crop_cache)

        images[0].unlink()
        (shoot / "ADDED.jpg").write_bytes(b"new")
        session.refresh()

        after_shoot = snapshot(shoot)
        # The two images this test moved itself are the only differences.
        self.assertEqual(
            {k: v for k, v in after_shoot.items() if k != "ADDED.jpg"},
            {k: v for k, v in before_shoot.items() if k != images[0].name},
            "refresh wrote to something under the folder",
        )
        self.assertEqual(snapshot(crop_cache), before_cache, "the crop cache is untouched")


# ---------------------------------------------------------------------------
# The desktop wiring: the action exists, and Arrange goes through it
# ---------------------------------------------------------------------------


class ServiceRefreshTests(FolderSyncTestCase):
    def _service(self):
        from picklikeme.desktop.services import ReviewService

        service = ReviewService(db_path=self.root / "svc.db")
        return service

    def test_refresh_folder_returns_the_resynced_state(self):
        shoot, images = self.shoot(count=3)
        service = self._service()
        try:
            service.open_folder(shoot)
            self.assertEqual(len(service.load_session()["images"]), 3)

            images[0].unlink()
            (shoot / "ADDED.jpg").write_bytes(b"new")
            state = service.refresh_folder()

            names = {Path(i["image_path"]).name for i in state["images"]}
            self.assertNotIn(images[0].name, names)
            self.assertIn("ADDED.jpg", names)
            self.assertEqual(len(names), 3)
        finally:
            service.store.close()

    def test_refresh_folder_with_no_folder_open_does_not_raise(self):
        service = self._service()
        try:
            self.assertEqual(service.refresh_folder()["images"], [])
        finally:
            service.store.close()


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# The desktop window: the Refresh control exists and Arrange rescans
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


def _window(tmp_path):
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    return MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    ), service


def _shoot(tmp_path, name="shoot", count=4):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    images = []
    for index in range(count):
        target = folder / f"IMG_{index:04d}.jpg"
        target.write_bytes(f"frame {index}".encode())
        images.append(target)
    _write_ranking(folder, images)
    return folder, images


def test_the_refresh_action_exists_and_is_reachable(app, tmp_path) -> None:
    """A user-accessible control, not just a service method: an action with a
    shortcut, on the File menu AND as a primary-toolbar button."""
    window, service = _window(tmp_path)
    try:
        action = window._refresh_action
        assert action.text() == "Refresh Folder"
        assert not action.shortcut().isEmpty(), "Refresh must have a keyboard shortcut"
        assert window._refresh_button is not None

        menu_texts = {
            entry.text()
            for top in window.menuBar().actions()
            if top.menu() is not None
            for entry in top.menu().actions()
        }
        assert "Refresh Folder" in menu_texts
    finally:
        window.close()
        service.close()


def test_refresh_action_resyncs_the_grid_with_the_disk(app, tmp_path) -> None:
    window, service = _window(tmp_path)
    try:
        folder, images = _shoot(tmp_path, count=4)
        window.open_folder(str(folder))
        assert len(window._all_items) == 4

        images[0].unlink()
        (folder / "ADDED.jpg").write_bytes(b"new")
        window._refresh_action.trigger()

        names = {Path(item.path).name for item in window._all_items}
        assert images[0].name not in names
        assert "ADDED.jpg" in names
        assert len(names) == 4
    finally:
        window.close()
        service.close()


def test_refresh_with_no_folder_open_reports_instead_of_raising(app, tmp_path) -> None:
    window, service = _window(tmp_path)
    try:
        window._refresh_action.trigger()
        assert "Open a folder" in window._status_message_label.text()
    finally:
        window.close()
        service.close()


def test_arrange_automatically_rescans_the_folder(app, tmp_path, monkeypatch) -> None:
    """The whole point of requirement 1: no reopen, no manual Refresh. The
    grid must be rebuilt FROM THE DISK as part of the Organize action - so
    this asserts both that refresh_folder was the call made, and that the
    resulting grid matches the disk.
    """
    from PySide6.QtWidgets import QMessageBox

    window, service = _window(tmp_path)
    try:
        folder, images = _shoot(tmp_path, count=4)
        window.open_folder(str(folder))
        window.apply_review_status("keep", paths=[str(images[0])])
        window.apply_review_status("reject", paths=[str(images[1])])

        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
        )
        calls: list[str] = []
        real_refresh = service.refresh_folder
        monkeypatch.setattr(
            service, "refresh_folder", lambda: (calls.append("refresh"), real_refresh())[1]
        )

        window._organize()

        assert calls == ["refresh"], "Organize must rescan the disk, not re-serialise state"
        on_disk = {p.name for p in folder.rglob("*.jpg")}
        in_grid = {Path(item.path).name for item in window._all_items}
        assert in_grid == on_disk
        grid_paths = {item.path for item in window._all_items}
        assert str(images[0]) not in grid_paths, "the pre-arrange path is gone from the grid"
        assert str(images[1]) not in grid_paths
    finally:
        window.close()
        service.close()
