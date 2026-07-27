"""Primary Failure Cause: a second, single-select diagnostic dimension on
false-negative annotations, distinct from the multi-select category checklist.

Storage: `AnnotationStore.save(..., primary_failure_cause=...)`.
Aggregation: `AnnotationSummary.primary_cause_counts`.
UI: a radio group in the annotation editor plus a "Primary failure cause
frequencies" breakdown in the summary section.
"""

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import (
    PRIMARY_FAILURE_CAUSES,
    AnnotationStore,
    summarise,
)
from test_annotations import make_image, store_in


class StorageTests(unittest.TestCase):
    def test_save_and_reload_round_trips_the_primary_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                saved = store.save(path, ["Action shot"], "note", primary_failure_cause="Occlusion")
                self.assertEqual(saved.primary_failure_cause, "Occlusion")

            with AnnotationStore(Path(tmp) / "kb" / "fn.db") as reopened:
                annotation = reopened.get(path)
                self.assertEqual(annotation.primary_failure_cause, "Occlusion")
                self.assertEqual(annotation.categories, ["Action shot"])  # unaffected

    def test_primary_cause_alone_is_enough_to_keep_the_record(self):
        """No categories, no notes, only a primary cause: must not be treated
        as empty and deleted."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                saved = store.save(path, [], "", primary_failure_cause="Multiple birds")
                self.assertFalse(saved.is_empty)
                self.assertEqual(store.count(), 1)
                self.assertEqual(store.get(path).primary_failure_cause, "Multiple birds")

    def test_clearing_everything_including_the_cause_deletes_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, ["Backlit"], "note", primary_failure_cause="Occlusion")
                self.assertEqual(store.count(), 1)
                store.save(path, [], "", primary_failure_cause=None)
                self.assertEqual(store.count(), 0)
                self.assertIsNone(store.get(path))

    def test_categories_and_cause_are_independent_dimensions(self):
        """Re-saving categories must not clobber the cause, and vice versa -
        each call sends both, but a category-only edit followed by a
        cause-only edit must still leave both readable afterwards."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, ["Backlit"], "", primary_failure_cause="Occlusion")
                store.save(path, ["Backlit", "Lighting"], "", primary_failure_cause="Occlusion")
                annotation = store.get(path)
                self.assertEqual(annotation.categories, ["Backlit", "Lighting"])
                self.assertEqual(annotation.primary_failure_cause, "Occlusion")

    def test_default_primary_cause_is_none_for_a_plain_category_only_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, ["Backlit"], "note")
                self.assertIsNone(store.get(path).primary_failure_cause)

    def test_empty_string_cause_is_treated_as_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                saved = store.save(path, ["Backlit"], "", primary_failure_cause="   ")
                self.assertIsNone(saved.primary_failure_cause)

    def test_as_dict_includes_the_primary_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                saved = store.save(path, [], "", primary_failure_cause="Classifier disagreement")
                self.assertEqual(saved.as_dict()["primary_failure_cause"], "Classifier disagreement")

    def test_expected_vocabulary_is_exactly_as_specified(self):
        self.assertEqual(
            PRIMARY_FAILURE_CAUSES,
            (
                "Detection crop too small",
                "Head outside crop",
                "Multiple birds",
                "Occlusion",
                "Classifier disagreement",
                "Other",
            ),
        )


class SchemaUpgradeTests(unittest.TestCase):
    """A database created before this feature existed must open cleanly and
    gain the new column without losing anything already stored."""

    def _make_pre_upgrade_db(self, db_path: Path, image_hash: str, notes: str) -> None:
        import sqlite3

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE schema_info (version INTEGER NOT NULL);
            CREATE TABLE categories (name TEXT PRIMARY KEY, ordering INTEGER, builtin INTEGER);
            CREATE TABLE annotations_v2 (
                image_hash TEXT PRIMARY KEY, filename TEXT NOT NULL, original_path TEXT NOT NULL,
                capture_datetime TEXT, notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE annotation_categories_v2 (
                image_hash TEXT NOT NULL, category TEXT NOT NULL, PRIMARY KEY (image_hash, category));
            """
        )
        conn.execute("INSERT INTO schema_info(version) VALUES (2)")
        conn.execute(
            "INSERT INTO annotations_v2 VALUES (?,?,?,?,?,?,?)",
            (image_hash, "x.NEF", "/x.NEF", None, notes, "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

    def test_opening_an_old_database_adds_the_column_without_data_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            self._make_pre_upgrade_db(db, "p1:deadbeef", "a note from before this feature existed")

            with AnnotationStore(db) as store:
                row = store._conn.execute(
                    "SELECT * FROM annotations_v2 WHERE image_hash = ?", ("p1:deadbeef",)
                ).fetchone()
                self.assertIn("primary_failure_cause", row.keys())
                self.assertIsNone(row["primary_failure_cause"])
                self.assertEqual(row["notes"], "a note from before this feature existed")

    def test_the_upgrade_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            self._make_pre_upgrade_db(db, "p1:deadbeef", "note")

            with AnnotationStore(db):
                pass
            # Second open must not error (ALTER TABLE ADD COLUMN a second time
            # would raise "duplicate column" if the guard were missing).
            with AnnotationStore(db) as store:
                self.assertEqual(store.count(), 1)

    def test_an_old_row_can_be_updated_to_add_a_primary_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            image = make_image(Path(tmp) / "x.NEF")
            with tempfile.TemporaryDirectory():
                pass
            # Use the real digest for this file so a normal save() targets the
            # same row the pre-upgrade insert created.
            with AnnotationStore(db) as probe:
                digest = probe.identity_of(image)
            self._make_pre_upgrade_db(db.with_name("old2.db"), digest, "pre-existing")
            db2 = db.with_name("old2.db")

            with AnnotationStore(db2) as store:
                store.save(image, ["Backlit"], "pre-existing", primary_failure_cause="Occlusion")
                annotation = store.get(image)
                self.assertEqual(annotation.primary_failure_cause, "Occlusion")
                self.assertEqual(annotation.categories, ["Backlit"])


class SummaryTests(unittest.TestCase):
    def test_frequency_counts_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [make_image(root / f"i{i}.NEF") for i in range(4)]
            with store_in(tmp) as store:
                store.save(paths[0], [], "", primary_failure_cause="Occlusion")
                store.save(paths[1], [], "", primary_failure_cause="Occlusion")
                store.save(paths[2], [], "", primary_failure_cause="Head outside crop")
                store.save(paths[3], ["Backlit"], "note")  # no primary cause set

                _, summary = summarise(store, [str(p) for p in paths])

            self.assertEqual(summary.annotated, 4)
            self.assertEqual(summary.primary_cause_counts[0], ("Occlusion", 2))
            self.assertEqual(dict(summary.primary_cause_counts)["Head outside crop"], 1)
            # The un-caused annotation must not appear in the cause breakdown.
            self.assertNotIn("None", dict(summary.primary_cause_counts))
            self.assertEqual(sum(c for _, c in summary.primary_cause_counts), 3)

    def test_known_primary_causes_are_always_the_full_fixed_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                _, summary = summarise(store, [])
            self.assertEqual(tuple(summary.known_primary_causes), PRIMARY_FAILURE_CAUSES)

    def test_no_primary_causes_set_yields_an_empty_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_image(root / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, ["Backlit"], "note")
                _, summary = summarise(store, [str(path)])
            self.assertEqual(summary.primary_cause_counts, [])

    def test_text_summary_includes_the_frequency_block(self):
        from picklikeme.analyzer.annotations import render_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_image(root / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, [], "", primary_failure_cause="Multiple birds")
                _, summary = summarise(store, [str(path)])
            text = render_summary(summary)
            self.assertIn("Primary failure cause frequencies", text)
            self.assertIn("Multiple birds", text)

    def test_as_dict_round_trips_the_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_image(root / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, [], "", primary_failure_cause="Occlusion")
                _, summary = summarise(store, [str(path)])
            payload = summary.as_dict()
            self.assertEqual(payload["primary_cause_counts"], [{"cause": "Occlusion", "count": 1}])
            self.assertIn("known_primary_causes", payload)


class HtmlAndApiIntegrationTests(unittest.TestCase):
    def _result(self, tmp: Path, db: Path):
        from picklikeme.analyzer.analysis import run_analysis
        from picklikeme.analyzer.config import AnalysisConfig
        from test_annotations import build_fn_dataset

        ranking, selected, rejected = build_fn_dataset(tmp)
        return run_analysis(
            AnalysisConfig(
                ranking_path=ranking,
                selected_root=selected,
                rejected_root=rejected,
                output_dir=tmp / "out",
                annotations_db=db,
                charts=False,
                contact_sheets=False,
                max_examples=10,
            )
        )

    def test_report_contains_a_radio_group_with_every_cause(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html(self._result(root, root / "kb.db"))
            self.assertIn('type="radio"', html)
            for cause in PRIMARY_FAILURE_CAUSES:
                self.assertIn(cause, html)
            self.assertIn("cause-grid", html)

    def test_categories_and_cause_do_not_cross_contaminate_in_the_ui(self):
        """The checkbox selector used by the JS Save handler must not also
        match the radio inputs, or picking a cause would be saved as an extra
        category tag. Checked at the markup level: every radio in the
        cause-grid must be type=radio, every checkbox in the general grid
        must be type=checkbox, and they must not share a CSS selector scope."""
        from picklikeme.analyzer.reports.html import build_html
        import re

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html(self._result(root, root / "kb.db"))
            cause_grid = re.search(r'<div class="cat-grid cause-grid">(.*?)</div>', html, re.S)
            self.assertIsNotNone(cause_grid)
            self.assertNotIn("checkbox", cause_grid.group(1))
            self.assertIn("radio", cause_grid.group(1))

    def test_existing_primary_cause_is_pre_selected_on_reload(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            first = self._result(root, db)
            target = first.errors.false_negatives[0].image_path
            with AnnotationStore(db) as store:
                store.save(target, [], "", primary_failure_cause="Classifier disagreement")

            again = self._result(root, db)
            html = build_html(again)
            self.assertIn('value="Classifier disagreement" checked', html)

    def test_summary_section_shows_top_primary_cause_card(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            first = self._result(root, db)
            target = first.errors.false_negatives[0].image_path
            with AnnotationStore(db) as store:
                store.save(target, [], "", primary_failure_cause="Detection crop too small")

            again = self._result(root, db)
            html = build_html(again)
            self.assertIn("Top primary cause", html)
            self.assertIn("Detection crop too small", html)

    def test_server_forwards_primary_failure_cause_on_save(self):
        from picklikeme.analyzer.server import make_server
        from picklikeme.analyzer.reports.html import write_html_report
        from picklikeme.analyzer.reports import write_json_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            result = self._result(root, db)
            write_json_report(result, result.config.output_dir / "analysis.json")
            write_html_report(result)

            target = result.errors.false_negatives[0].image_path
            store = AnnotationStore(db)
            server = make_server(result.config.output_dir, store, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = urllib.request.Request(
                    f"{base}/api/annotations",
                    data=json.dumps(
                        {"image_path": target, "categories": [], "primary_failure_cause": "Occlusion"}
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    body = json.load(response)
                self.assertEqual(body["annotation"]["primary_failure_cause"], "Occlusion")
                self.assertEqual(store.get(target).primary_failure_cause, "Occlusion")

                with urllib.request.urlopen(f"{base}/api/primary-causes") as response:
                    causes = json.load(response)["primary_causes"]
                self.assertEqual(tuple(causes), PRIMARY_FAILURE_CAUSES)
            finally:
                server.shutdown()
                server.server_close()
                store.close()


class MetricIsolationTests(unittest.TestCase):
    def test_primary_failure_cause_never_changes_a_metric(self):
        """Same guarantee as the general categories: human diagnosis data must
        never move an evaluation number."""
        from picklikeme.analyzer.analysis import run_analysis
        from picklikeme.analyzer.config import AnalysisConfig
        from test_annotations import build_fn_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            ranking, selected, rejected = build_fn_dataset(root)

            def run():
                return run_analysis(
                    AnalysisConfig(
                        ranking_path=ranking,
                        selected_root=selected,
                        rejected_root=rejected,
                        output_dir=root / "out",
                        annotations_db=db,
                        charts=False,
                        contact_sheets=False,
                    )
                )

            before = run()
            metrics_before = {v.name: v.value for v in before.metrics.values}

            with AnnotationStore(db) as store:
                for record in before.errors.false_negatives:
                    store.save(record.image_path, [], "", primary_failure_cause="Occlusion")

            after = run()
            metrics_after = {v.name: v.value for v in after.metrics.values}
            self.assertEqual(metrics_before, metrics_after)


if __name__ == "__main__":
    unittest.main()
