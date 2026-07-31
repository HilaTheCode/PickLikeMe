import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import REVIEW_KEEP, AnnotationStore
from picklikeme.importer import import_selected_images
from picklikeme.organize import SELECTED_DIRNAME
from picklikeme.review.page import build_page
from picklikeme.review.session import ReviewSession
from picklikeme.workspace import WorkspaceManager


class PeakPicWorkflowTests(unittest.TestCase):
    def test_import_selected_copies_files_and_updates_review_decisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_folder = root / "source"
            selected_dir = source_folder / SELECTED_DIRNAME
            selected_dir.mkdir(parents=True)
            source_file = selected_dir / "IMG_0001.CR2"
            source_file.write_bytes(b"raw-data")

            destination_root = root / "imported"
            destination_root.mkdir()

            store = AnnotationStore(root / "kb.db")
            try:
                store.set_review_decision(source_file, REVIEW_KEEP)
                result = import_selected_images(source_folder=source_folder, destination_root=destination_root, store=store)

                self.assertEqual(result["copied"], 1)
                destination_file = destination_root / "IMG_0001.CR2"
                self.assertTrue(destination_file.exists())
                self.assertEqual(destination_file.read_bytes(), b"raw-data")

                rows = store.review_decisions()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["image_path"], str(destination_file))
            finally:
                store.close()

    def test_workspace_manager_releases_previous_workspace(self):
        manager = WorkspaceManager()
        closed = []

        first = Path("/tmp/first")
        second = Path("/tmp/second")

        manager.open_workspace(first, on_close=lambda: closed.append("first"))
        manager.open_workspace(second, on_close=lambda: closed.append("second"))

        self.assertEqual(manager.current_workspace, second)
        self.assertEqual(closed, ["first"])
        self.assertEqual(manager.current_workspace.name, second.name)

    def test_review_session_exposes_workflow_state(self):
        store = AnnotationStore(Path(tempfile.gettempdir()) / "peakpic-workflow-test.db")
        try:
            session = ReviewSession(None, store)
            workflow = session.as_dict()["workflow"]
            self.assertIn("stage", workflow)
            self.assertFalse(workflow["ranked"])
            self.assertFalse(workflow["reviewed"])
            self.assertFalse(workflow["imported"])
        finally:
            store.close()

    def test_review_page_uses_peakpic_branding(self):
        html = build_page()
        self.assertIn("PeakPic", html)
        self.assertNotIn("PickLikeMe", html)
