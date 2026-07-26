"""False-negative knowledge base: storage, summary, HTML integration, API.

The load-bearing test here is `test_annotations_never_change_any_metric`: the
whole feature is only safe if human notes cannot move the numbers they explain.
"""

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import (
    DEFAULT_ANNOTATIONS_DB,
    INITIAL_CATEGORIES,
    AnnotationStore,
    image_key,
    render_summary,
    summarise,
)
from picklikeme.analyzer.config import AnalysisConfig
from test_analyzer import build_dataset, write_ranking  # reuse the analyzer fixtures


def store_in(tmp) -> AnnotationStore:
    return AnnotationStore(Path(tmp) / "kb" / "fn.db")


# Scores are chosen rather than sampled, so the fixture is guaranteed to contain
# false negatives (kept images the model scored below 0.5) - the entire feature
# under test hangs off those existing.
FN_SCORES = [
    # (keep?, score) - four false negatives, plus TPs, FPs and TNs.
    (True, 0.95), (True, 0.88), (True, 0.71), (True, 0.62),
    (True, 0.44), (True, 0.31), (True, 0.18), (True, 0.07),   # <- false negatives
    (False, 0.91), (False, 0.66),                             # <- false positives
    (False, 0.39), (False, 0.28), (False, 0.16), (False, 0.04),
]


def build_fn_dataset(tmp: Path):
    """A ranking with a known number of false negatives, plus its folders."""
    selected, rejected = tmp / "keep", tmp / "drop"
    selected.mkdir(parents=True, exist_ok=True)
    rejected.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (keep, score) in enumerate(FN_SCORES):
        target = (selected if keep else rejected) / f"IMG_{index:04d}.NEF"
        target.write_bytes(b"x")
        rows.append([str(target), score, 1 if keep else 0])

    rows.sort(key=lambda row: -row[1])
    ranked = [[position, row[0], f"{row[1]:.6f}", row[2]] for position, row in enumerate(rows, start=1)]
    return write_ranking(tmp / "rankings.csv", ranked), selected, rejected


EXPECTED_FALSE_NEGATIVES = sum(1 for keep, score in FN_SCORES if keep and score < 0.5)


class StorageTests(unittest.TestCase):
    def test_database_is_created_with_the_initial_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                self.assertTrue(store.db_path.exists())
                self.assertEqual(store.categories()[: len(INITIAL_CATEGORIES)], list(INITIAL_CATEGORIES))
                self.assertIn("Action shot", store.categories())
                self.assertIn("Animal not in supported categories", store.categories())

    def test_save_and_reload_multiple_categories_and_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "IMG_0001.NEF"
            with store_in(tmp) as store:
                store.save(path, ["Action shot", "Artistic choice"], "Great wing position.\nSecond line.")

            # A fresh store must see it: the point is surviving future runs.
            with AnnotationStore(Path(tmp) / "kb" / "fn.db") as reopened:
                annotation = reopened.get(path)
                self.assertIsNotNone(annotation)
                self.assertEqual(annotation.categories, ["Action shot", "Artistic choice"])
                self.assertIn("Great wing position.", annotation.notes)
                self.assertIn("\n", annotation.notes, "multi-line notes must round-trip")

    def test_saving_again_replaces_categories_and_keeps_created_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.NEF"
            with store_in(tmp) as store:
                first = store.save(path, ["Backlit"], "one")
                second = store.save(path, ["Lighting", "Out of focus foreground"], "two")

                self.assertEqual(second.created_at, first.created_at)
                self.assertEqual(
                    store.get(path).categories, ["Lighting", "Out of focus foreground"]
                )
                self.assertEqual(store.get(path).notes, "two")

    def test_clearing_everything_deletes_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.NEF"
            with store_in(tmp) as store:
                store.save(path, ["Backlit"], "note")
                self.assertEqual(store.count(), 1)
                store.save(path, [], "")
                self.assertEqual(store.count(), 0)
                self.assertIsNone(store.get(path))

    def test_custom_category_is_remembered_for_next_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                store.save(Path(tmp) / "a.NEF", ["Wing blur I like"], "")
                self.assertIn("Wing blur I like", store.categories())

    def test_annotation_survives_the_file_moving(self):
        """The archive gets reorganised; a diagnosis must not be orphaned."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "old" / "shoot" / "IMG_9.NEF"
            moved = root / "new" / "elsewhere" / "IMG_9.NEF"
            with store_in(tmp) as store:
                store.save(original, ["Subject too small"], "tiny bird")

                found = store.get(moved)
                self.assertIsNotNone(found, "should be found by filename after the move")
                self.assertTrue(found.matched_by_filename)
                self.assertEqual(found.categories, ["Subject too small"])

    def test_ambiguous_filename_is_never_guessed(self):
        """Camera counters reset, so duplicate basenames are normal in a
        multi-year archive. A wrong diagnosis is worse than a missing one."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with store_in(tmp) as store:
                store.save(root / "shootA" / "DSC_0001.NEF", ["Backlit"], "")
                store.save(root / "shootB" / "DSC_0001.NEF", ["Wrong crop"], "")

                self.assertIsNone(store.get(root / "shootC" / "DSC_0001.NEF"))
                # Exact paths still resolve unambiguously.
                self.assertEqual(store.get(root / "shootA" / "DSC_0001.NEF").categories, ["Backlit"])

    def test_image_key_is_stable_and_path_derived(self):
        self.assertEqual(image_key("/a/b.nef"), image_key("/a/b.nef"))
        self.assertNotEqual(image_key("/a/b.nef"), image_key("/a/c.nef"))

    def test_default_database_lives_outside_any_output_directory(self):
        # Output dirs are per-run and get replaced; the knowledge base must not
        # be inside one.
        self.assertNotIn("analysis", DEFAULT_ANNOTATIONS_DB.parts[-2:])
        self.assertEqual(DEFAULT_ANNOTATIONS_DB.name, "false_negatives.db")


class SummaryTests(unittest.TestCase):
    def test_frequencies_combinations_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"i{i}.NEF" for i in range(5)]
            with store_in(tmp) as store:
                store.save(paths[0], ["Action shot", "Artistic choice"], "a")
                store.save(paths[1], ["Action shot", "Artistic choice"], "b")
                store.save(paths[2], ["Action shot"], "")
                store.save(paths[3], ["Backlit"], "")

                found, summary = summarise(store, [str(p) for p in paths])

            self.assertEqual(summary.total_false_negatives, 5)
            self.assertEqual(summary.annotated, 4)
            self.assertEqual(summary.unannotated_count, 1)
            self.assertAlmostEqual(summary.coverage, 0.8)
            self.assertEqual(summary.category_counts[0], ("Action shot", 3))
            self.assertEqual(
                summary.combination_counts[0], (("Action shot", "Artistic choice"), 2)
            )

    def test_combination_order_does_not_create_two_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with store_in(tmp) as store:
                store.save(root / "a.NEF", ["Lighting", "Backlit"], "")
                store.save(root / "b.NEF", ["Backlit", "Lighting"], "")
                _, summary = summarise(store, [str(root / "a.NEF"), str(root / "b.NEF")])
            self.assertEqual(len(summary.combination_counts), 1)
            self.assertEqual(summary.combination_counts[0][1], 2)

    def test_single_category_is_not_a_combination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with store_in(tmp) as store:
                store.save(root / "a.NEF", ["Backlit"], "")
                _, summary = summarise(store, [str(root / "a.NEF")])
            self.assertEqual(summary.combination_counts, [])

    def test_recent_is_ordered_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with store_in(tmp) as store:
                store.save(root / "old.NEF", ["Backlit"], "")
                store.save(root / "new.NEF", ["Lighting"], "")
                _, summary = summarise(store, [str(root / "old.NEF"), str(root / "new.NEF")])
            self.assertEqual(len(summary.recent), 2)
            self.assertGreaterEqual(summary.recent[0].updated_at, summary.recent[1].updated_at)

    def test_text_summary_explains_an_empty_knowledge_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                _, summary = summarise(store, ["/a.NEF"])
            text = render_summary(summary)
            self.assertIn("No annotations yet", text)
            self.assertIn("picklikeme annotate", text)


class IsolationTests(unittest.TestCase):
    """Annotations are human knowledge for humans. They must not leak into the
    evaluation."""

    def _run(self, tmp: Path, db: Path):
        from picklikeme.analyzer.analysis import run_analysis

        ranking, selected, rejected = build_fn_dataset(tmp)
        return run_analysis(
            AnalysisConfig(
                ranking_path=ranking,
                selected_root=selected,
                rejected_root=rejected,
                output_dir=tmp / "out",
                annotations_db=db,
                max_examples=20,
            )
        )

    def test_annotations_never_change_any_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"

            clean = self._run(root, db)
            before = {value.name: value.value for value in clean.metrics.values}
            confusion_before = clean.confusion.as_dict()
            sweep_before = clean.sweep.recommended.as_dict()

            # Annotate every false negative, then re-run.
            with AnnotationStore(db) as store:
                for record in clean.errors.false_negatives:
                    store.save(record.image_path, ["Action shot", "Backlit"], "human note")
            self.assertGreater(len(clean.errors.false_negatives), 0, "fixture must produce FNs")

            annotated = self._run(root, db)
            after = {value.name: value.value for value in annotated.metrics.values}

            self.assertEqual(before, after, "a metric changed once annotations existed")
            self.assertEqual(confusion_before, annotated.confusion.as_dict())
            self.assertEqual(sweep_before, annotated.sweep.recommended.as_dict())
            self.assertEqual(
                [s.title for s in clean.suggestions], [s.title for s in annotated.suggestions]
            )
            # ...but the annotations must actually have loaded.
            self.assertGreater(annotated.annotation_summary.annotated, 0)

    def test_missing_database_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._run(root, root / "does" / "not" / "exist" / "kb.db")
            self.assertIsNotNone(result.metrics.get("accuracy"))
            self.assertEqual(result.annotations, {})

    def test_no_annotations_flag_skips_the_database_entirely(self):
        from picklikeme.analyzer.analysis import run_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_fn_dataset(root)
            db = root / "never_created.db"
            result = run_analysis(
                AnalysisConfig(
                    ranking_path=ranking,
                    selected_root=selected,
                    rejected_root=rejected,
                    output_dir=root / "out",
                    annotations_db=db,
                    annotations_enabled=False,
                )
            )
            self.assertIsNone(result.annotation_summary)
            self.assertFalse(db.exists(), "the database must not be created when disabled")

    def test_only_false_negatives_are_annotated(self):
        """False positives are deliberately out of scope."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            result = self._run(root, db)
            fn_paths = {r.image_path for r in result.errors.false_negatives}
            fp_paths = {r.image_path for r in result.errors.false_positives}

            self.assertEqual(
                result.annotation_summary.total_false_negatives, len(fn_paths)
            )
            for path in result.annotation_summary.unannotated:
                self.assertNotIn(path, fp_paths)


class HtmlIntegrationTests(unittest.TestCase):
    def _result(self, tmp: Path, db: Path):
        from picklikeme.analyzer.analysis import run_analysis

        ranking, selected, rejected = build_fn_dataset(tmp)
        return run_analysis(
            AnalysisConfig(
                ranking_path=ranking,
                selected_root=selected,
                rejected_root=rejected,
                output_dir=tmp / "out",
                annotations_db=db,
                max_examples=10,
            )
        )

    def test_panels_include_edit_save_checklist_and_notes(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._result(root, root / "kb.db")
            html = build_html(result)

            self.assertIn("btn-edit", html)
            self.assertIn("btn-save", html)
            self.assertIn("<textarea", html)
            for category in ("Action shot", "Artistic choice", "Detector mistake"):
                self.assertIn(category, html)
            self.assertIn("False negative summary", html)
            # Filters
            self.assertIn('id="f-state"', html)
            self.assertIn('id="f-cats"', html)
            self.assertIn("not annotated", html)

    def test_existing_annotations_are_reloaded_into_the_page(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            first = self._result(root, db)
            target = first.errors.false_negatives[0].image_path
            with AnnotationStore(db) as store:
                store.save(target, ["Backlit", "Subject too small"], "reloaded note")

            again = self._result(root, db)
            html = build_html(again)

            self.assertIn("reloaded note", html)
            self.assertIn("checked", html, "a stored category must render pre-ticked")
            # Inlined so a file:// report still shows what is known.
            self.assertIn("window.PLM_ANNOTATIONS=", html)
            self.assertIn("reloaded note", json.dumps(again.annotations[target].as_dict()))

    def test_report_stays_offline_with_the_annotation_ui(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html(self._result(root, root / "kb.db"))
            self.assertNotIn("http://", html)
            self.assertNotIn("https://", html)


class ServerTests(unittest.TestCase):
    def _serve(self, report_dir: Path, db: Path):
        from picklikeme.analyzer.server import make_server

        store = AnnotationStore(db)
        server = make_server(report_dir, store, port=0)  # port 0 = pick a free one
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, store, f"http://127.0.0.1:{server.server_address[1]}"

    def _prepare(self, tmp: Path) -> Path:
        report_dir = tmp / "out"
        report_dir.mkdir(parents=True)
        (report_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")
        return report_dir

    def test_health_categories_and_save_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = self._prepare(root)
            server, store, base = self._serve(report_dir, root / "kb.db")
            try:
                with urllib.request.urlopen(f"{base}/api/health") as response:
                    self.assertTrue(json.load(response)["ok"])

                with urllib.request.urlopen(f"{base}/api/categories") as response:
                    self.assertIn("Action shot", json.load(response)["categories"])

                payload = json.dumps(
                    {
                        "image_path": str(root / "IMG_1.NEF"),
                        "categories": ["Action shot", "Lighting"],
                        "notes": "posted from the report",
                    }
                ).encode("utf-8")
                request = urllib.request.Request(
                    f"{base}/api/annotations",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    body = json.load(response)
                self.assertTrue(body["ok"])
                self.assertEqual(body["annotation"]["categories"], ["Action shot", "Lighting"])

                # Persisted, not just echoed.
                self.assertEqual(store.get(root / "IMG_1.NEF").notes, "posted from the report")

                with urllib.request.urlopen(f"{base}/api/annotations") as response:
                    self.assertEqual(len(json.load(response)["annotations"]), 1)
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_bad_request_is_reported_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, store, base = self._serve(self._prepare(root), root / "kb.db")
            try:
                request = urllib.request.Request(
                    f"{base}/api/annotations",
                    data=json.dumps({"categories": []}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(request)
                self.assertEqual(ctx.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_path_traversal_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secret.txt").write_text("do not serve me", encoding="utf-8")
            server, store, base = self._serve(self._prepare(root), root / "kb.db")
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base}/../secret.txt")
                self.assertIn(ctx.exception.code, (403, 404))
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_server_binds_loopback_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, store, _ = self._serve(self._prepare(root), root / "kb.db")
            try:
                self.assertEqual(server.server_address[0], "127.0.0.1")
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_refuses_a_directory_without_a_report(self):
        from picklikeme.analyzer.server import make_server

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty"
            empty.mkdir()
            with AnnotationStore(root / "kb.db") as store:
                with self.assertRaises(SystemExit):
                    make_server(empty, store, port=0)


class CliTests(unittest.TestCase):
    def test_annotate_parser_defaults(self):
        from picklikeme.analyzer.cli import build_annotate_parser

        args = build_annotate_parser().parse_args([])
        self.assertIn("analysis", args.output)
        self.assertIsNone(args.port)

    def test_analyze_flags_reach_the_config(self):
        from picklikeme.analyzer.cli import build_parser, config_from_args

        args = build_parser().parse_args(
            ["--ranking", "r.csv", "--annotations-db", "kb.db", "--no-annotations"]
        )
        config = config_from_args(args)
        self.assertEqual(config.annotations_db, Path("kb.db"))
        self.assertFalse(config.annotations_enabled)

    def test_annotate_is_registered_on_the_main_cli(self):
        from picklikeme.ingest.cli import main

        with self.assertRaises(SystemExit):
            main(["annotate", "--help"])


if __name__ == "__main__":
    unittest.main()
