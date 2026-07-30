"""evaluation_report.py: the standalone HTML/CSV reports built from
ReviewSession.agreement_stats and .disagreements. Since both of those are
already covered directly in test_review_session.py, these tests focus on
what this module adds - rendering, escaping, and the CSV/HTML data staying
in agreement with each other - using a real ReviewSession/AnnotationStore
rather than mocks, the same way test_review_session.py does.
"""

import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import AnnotationStore
from picklikeme.review.evaluation_report import (
    build_evaluation_report_csv,
    build_evaluation_report_html,
)
from picklikeme.review.session import ReviewSession
from picklikeme.sidecar import write_run_metadata
from test_review_session import build_shoot


class EvaluationReportTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.store = AnnotationStore(self.root / "kb.db")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class NothingDecidedYetTests(EvaluationReportTestCase):
    """Before any decision is made, agreement_stats().compared is 0 - the
    report must say so plainly rather than showing 0%/0% as if that were a
    real (perfect or terrible) result."""

    def test_the_report_says_there_is_nothing_to_compare_yet(self):
        shoot, _, _ = build_shoot(self.root, ranked=5)
        session = ReviewSession(shoot, self.store, keep_percent=40)

        body = build_evaluation_report_html(session)

        self.assertIn("nothing to compare", body)
        self.assertIn(shoot.name, body)

    def test_the_csv_has_only_a_header_row(self):
        shoot, _, _ = build_shoot(self.root, ranked=5)
        session = ReviewSession(shoot, self.store, keep_percent=40)

        rows = list(csv.reader(io.StringIO(build_evaluation_report_csv(session))))

        self.assertEqual(rows, [["file_name", "ai_decision", "user_decision", "ai_score"]])


class PopulatedReportTests(EvaluationReportTestCase):
    """A shoot with real agreement and disagreement - the report's numbers
    must match ReviewSession's own agreement_stats()/disagreements(), since
    those are the single source of truth both the panel and this report
    read from."""

    def setUp(self):
        super().setUp()
        self.shoot, self.images, _ = build_shoot(self.root, ranked=10)
        write_run_metadata(self.shoot, backbone="resnet50-v2", image_count=10)
        # keep_percent=30 -> cut=3: images[0..2] are "AI Keep", images[3..9] "AI Reject".
        self.session = ReviewSession(self.shoot, self.store, keep_percent=30)
        self.session.set_review_status(str(self.images[0]), "keep")  # agrees (AI: keep)
        self.session.set_review_status(str(self.images[1]), "reject")  # disagrees (AI: keep)
        self.session.set_review_status(str(self.images[-1]), "keep")  # disagrees (AI: reject)

    def test_general_information_names_the_folder_and_model(self):
        body = build_evaluation_report_html(self.session)

        self.assertIn(self.shoot.name, body)
        self.assertIn("resnet50-v2", body)
        self.assertIn("Total images", body)

    def test_summary_counts_match_the_session(self):
        body = build_evaluation_report_html(self.session)
        counts = self.session.counts()
        ai_counts = self.session.ai_suggestion_counts()

        self.assertIn(f"{counts['total']:,}", body)
        self.assertIn(f"{ai_counts['keep']:,}", body)
        self.assertIn(f"{ai_counts['reject']:,}", body)

    def test_the_confusion_matrix_cells_match_agreement_stats(self):
        body = build_evaluation_report_html(self.session)
        agreement = self.session.agreement_stats()

        self.assertIn(f"{agreement['ai_keep_user_keep']:,}", body)
        self.assertIn(f"{agreement['ai_keep_user_reject']:,}", body)
        self.assertIn(f"{agreement['ai_reject_user_keep']:,}", body)
        self.assertIn(f"{agreement['ai_reject_user_reject']:,}", body)

    def test_performance_metrics_are_rendered_to_three_decimals(self):
        body = build_evaluation_report_html(self.session)
        agreement = self.session.agreement_stats()

        self.assertIn(f"{agreement['precision']:.3f}", body)
        self.assertIn(f"{agreement['recall']:.3f}", body)
        self.assertIn(f"{agreement['f1']:.3f}", body)

    def test_every_disagreement_is_listed_with_its_filename(self):
        body = build_evaluation_report_html(self.session)

        for image in self.session.disagreements():
            self.assertIn(image.filename, body)

    def test_the_html_and_csv_report_the_same_disagreements(self):
        html_body = build_evaluation_report_html(self.session)
        csv_rows = list(csv.reader(io.StringIO(build_evaluation_report_csv(self.session))))[1:]

        self.assertEqual(len(csv_rows), len(self.session.disagreements()))
        for image in self.session.disagreements():
            self.assertIn(image.filename, html_body)
            self.assertTrue(any(row[0] == image.filename for row in csv_rows))

    def test_csv_ai_decision_is_always_the_opposite_of_the_user_s(self):
        rows = list(csv.reader(io.StringIO(build_evaluation_report_csv(self.session))))[1:]

        for file_name, ai_decision, user_decision, _score in rows:
            self.assertNotEqual(ai_decision, user_decision)
            self.assertIn(ai_decision, ("keep", "reject"))
            self.assertIn(user_decision, ("keep", "reject"))

    def test_the_report_is_self_contained(self):
        """Offline, exactly like the analyzer's own HTML report - no CDN, no
        external asset of any kind."""
        body = build_evaluation_report_html(self.session)

        self.assertNotIn("http://", body)
        self.assertNotIn("https://", body)


class EscapingTests(EvaluationReportTestCase):
    def test_a_filename_with_html_special_characters_is_escaped(self):
        shoot, images, _ = build_shoot(self.root, ranked=3)
        # "<", ">", "\"" etc. are illegal in a Windows filename, but "&" is a
        # real HTML metacharacter photographers do use (e.g. "Smith & Jones").
        dangerous = shoot / "Smith & Jones.jpg"
        images[0].rename(dangerous)
        write_run_metadata(shoot)  # no backbone - exercises the "unknown model" branch

        # Rewrite the ranking so it points at the renamed file.
        from picklikeme.sidecar import ranking_path

        original = ranking_path(shoot).read_text(encoding="utf-8")
        ranking_path(shoot).write_text(original.replace(str(images[0]), str(dangerous)), encoding="utf-8")

        session = ReviewSession(shoot, self.store, keep_percent=100)
        session.set_review_status(str(dangerous), "reject")  # AI keep_percent=100 -> AI suggests keep

        body = build_evaluation_report_html(session)

        self.assertNotIn("Smith & Jones.jpg<", body)
        self.assertIn("Smith &amp; Jones.jpg", body)
        self.assertIn("unknown", body)


if __name__ == "__main__":
    unittest.main()
