"""The config-driven annotation fields (config/annotations.yaml).

Crop Quality, Image Quality and Agree with Model Decision replaced the growable
category checklist plus the single primary-failure-cause radio, and are now
themselves defined by config rather than hardcoded in this package. The point
of the redesign is countable data with a stable identity, so the tests that
matter here are about *closedness* (a value outside the vocabulary must be
refused rather than stored), *ids vs labels* (the database stores an id, the
UI shows a label, and the two must never be confused), and that pre-redesign
records are preserved rather than silently re-interpreted as new answers.

These tests exercise the *real* shipped config/annotations.yaml (loaded via
`store_in()`'s default), since it is itself part of what's being verified.
"""

import json
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotation_config import DEFAULT_ANNOTATIONS_CONFIG, load_annotation_fields
from picklikeme.analyzer.annotations import (
    AnnotationStore,
    InvalidAnnotationValue,
    render_summary,
    summarise,
)
from test_annotations import make_image, store_in

# Legacy column names the pre-config-driven schema bolted onto annotations_v2.
# Fixed, historical identifiers - not part of the configurable surface.
LEGACY_FIELD_COLUMNS = ("crop_quality", "image_quality", "agree_with_model_decision")


class VocabularyTests(unittest.TestCase):
    def test_the_shipped_config_defines_the_expected_fields_in_order(self):
        config = load_annotation_fields(DEFAULT_ANNOTATIONS_CONFIG)
        self.assertEqual(config.field_ids, ("crop_quality", "image_quality", "agree_with_model_decision"))

    def test_the_shipped_config_s_values_are_exactly_as_specified(self):
        config = load_annotation_fields(DEFAULT_ANNOTATIONS_CONFIG)
        self.assertEqual(
            config.get("crop_quality").value_ids, ("good", "too_small", "wrong_location", "too_large")
        )
        self.assertEqual(
            config.get("image_quality").value_ids,
            ("good", "missing_eye", "out_of_focus", "no_relevant_subject", "group_scene"),
        )
        self.assertEqual(config.get("agree_with_model_decision").value_ids, ("yes", "no"))

    def test_store_exposes_the_same_vocabularies_as_its_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                self.assertEqual(
                    store.field_vocabularies(),
                    {f.id: list(f.value_ids) for f in store.fields_config},
                )


class StorageTests(unittest.TestCase):
    def test_save_and_reload_round_trips_every_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                saved = store.save(
                    path,
                    fields={
                        "crop_quality": "too_small",
                        "image_quality": "out_of_focus",
                        "agree_with_model_decision": "no",
                    },
                )
                self.assertTrue(saved.image_hash.startswith("p1:"))

            with AnnotationStore(Path(tmp) / "kb" / "fn.db") as reopened:
                annotation = reopened.get(path)
                self.assertEqual(annotation.fields["crop_quality"], "too_small")
                self.assertEqual(annotation.fields["image_quality"], "out_of_focus")
                self.assertEqual(annotation.fields["agree_with_model_decision"], "no")

    def test_fields_are_independent(self):
        """Answering one question must not disturb the answers to the others."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, fields={"crop_quality": "good", "image_quality": "missing_eye"})
                store.save(
                    path,
                    fields={
                        "crop_quality": "good",
                        "image_quality": "missing_eye",
                        "agree_with_model_decision": "yes",
                    },
                )
                annotation = store.get(path)
                self.assertEqual(annotation.fields["crop_quality"], "good")
                self.assertEqual(annotation.fields["image_quality"], "missing_eye")
                self.assertEqual(annotation.fields["agree_with_model_decision"], "yes")

    def test_any_single_field_is_enough_to_keep_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                for index, field_def in enumerate(store.fields_config):
                    path = make_image(Path(tmp) / f"only_{index}.NEF", f"pixels {index}".encode())
                    saved = store.save(path, fields={field_def.id: field_def.value_ids[0]})
                    self.assertFalse(saved.is_empty)
                    self.assertIsNotNone(store.get(path))

    def test_clearing_every_field_deletes_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, fields={"crop_quality": "good", "agree_with_model_decision": "yes"})
                self.assertEqual(store.count(), 1)
                store.save(path)
                self.assertEqual(store.count(), 0)
                self.assertIsNone(store.get(path))

    def test_blank_and_whitespace_are_treated_as_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                saved = store.save(path, fields={"crop_quality": "   ", "image_quality": "good"})
                self.assertIsNone(saved.fields.get("crop_quality"))
                self.assertEqual(saved.fields["image_quality"], "good")

    def test_a_value_outside_the_vocabulary_is_refused_not_stored(self):
        """The whole redesign exists to make the data countable, so free text
        must fail loudly rather than quietly becoming a one-off row."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                with self.assertRaises(InvalidAnnotationValue):
                    store.save(path, fields={"crop_quality": "a bit tight honestly"})
                self.assertEqual(store.count(), 0)

    def test_validation_is_case_and_spelling_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                for wrong in ("Too Small", "TOO_SMALL", "too  small", "yes"):
                    with self.assertRaises(InvalidAnnotationValue):
                        store.save(path, fields={"crop_quality": wrong})

    def test_an_unknown_field_id_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                with self.assertRaises(InvalidAnnotationValue):
                    store.save(path, fields={"nonexistent_field": "anything"})

    def test_annotation_follows_a_renamed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = make_image(root / "IMG_9.NEF", b"the same bird")
            with store_in(tmp) as store:
                store.save(original, fields={"crop_quality": "too_small"})
                original.unlink()
                renamed = make_image(root / "best_of_shoot.NEF", b"the same bird")

                found = store.get(renamed)
                self.assertIsNotNone(found, "identity must survive a rename")
                self.assertEqual(found.fields["crop_quality"], "too_small")
                self.assertTrue(found.relocated)

    def test_as_dict_exposes_every_field_under_the_fields_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                payload = store.save(
                    path,
                    fields={
                        "crop_quality": "wrong_location",
                        "image_quality": "no_relevant_subject",
                        "agree_with_model_decision": "yes",
                    },
                ).as_dict()
            self.assertEqual(payload["fields"]["crop_quality"], "wrong_location")
            self.assertEqual(payload["fields"]["image_quality"], "no_relevant_subject")
            self.assertEqual(payload["fields"]["agree_with_model_decision"], "yes")


class LegacyRecordTests(unittest.TestCase):
    """Records written before either redesign (fields as hardcoded columns, or
    now as config-driven ids) are preserved, shown, and *not* guessed at.
    Inferring "Subject too small" meant Crop Quality "Too Small" would
    fabricate an answer the photographer never gave, and then count it."""

    def _make_pre_redesign_db(self, db_path: Path, image_hash: str, notes: str, cause: str) -> None:
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
                primary_failure_cause TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE annotation_categories_v2 (
                image_hash TEXT NOT NULL, category TEXT NOT NULL, PRIMARY KEY (image_hash, category));
            """
        )
        conn.execute("INSERT INTO schema_info(version) VALUES (2)")
        conn.execute(
            "INSERT INTO annotations_v2 VALUES (?,?,?,?,?,?,?,?)",
            (
                image_hash,
                "x.NEF",
                "/x.NEF",
                None,
                notes,
                cause,
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ),
        )
        conn.execute(
            "INSERT INTO annotation_categories_v2 VALUES (?,?)", (image_hash, "Subject too small")
        )
        conn.commit()
        conn.close()

    def test_opening_an_old_database_adds_the_columns_without_data_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            self._make_pre_redesign_db(db, "p1:deadbeef", "an old note", "Occlusion")

            with AnnotationStore(db) as store:
                row = store._conn.execute(
                    "SELECT * FROM annotations_v2 WHERE image_hash = ?", ("p1:deadbeef",)
                ).fetchone()
                for column in (*LEGACY_FIELD_COLUMNS, "field_values"):
                    self.assertIn(column, row.keys())
                    self.assertIsNone(row[column])
                self.assertEqual(row["notes"], "an old note")
                self.assertEqual(store.count(), 1)

    def test_the_upgrade_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            self._make_pre_redesign_db(db, "p1:deadbeef", "note", "Occlusion")
            with AnnotationStore(db):
                pass
            with AnnotationStore(db) as store:
                self.assertEqual(store.count(), 1)

    def test_old_content_is_readable_as_legacy_and_never_as_a_new_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = make_image(root / "x.NEF")
            db = root / "kb.db"
            with AnnotationStore(root / "probe.db") as probe:
                digest = probe.identity_of(image)
            self._make_pre_redesign_db(db, digest, "an old note", "Occlusion")

            with AnnotationStore(db) as store:
                annotation = store.get(image)
                self.assertTrue(annotation.has_legacy_content)
                self.assertTrue(annotation.is_empty, "legacy content is not a new answer")
                self.assertEqual(annotation.legacy_notes, "an old note")
                self.assertEqual(annotation.legacy_primary_failure_cause, "Occlusion")
                self.assertEqual(annotation.legacy_categories, ["Subject too small"])
                self.assertEqual(annotation.fields, {}, "no legacy field columns had data to migrate")

    def test_answering_the_new_fields_leaves_the_legacy_content_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = make_image(root / "x.NEF")
            db = root / "kb.db"
            with AnnotationStore(root / "probe.db") as probe:
                digest = probe.identity_of(image)
            self._make_pre_redesign_db(db, digest, "an old note", "Occlusion")

            with AnnotationStore(db) as store:
                store.save(image, fields={"crop_quality": "good", "agree_with_model_decision": "no"})
                annotation = store.get(image)
                self.assertEqual(annotation.fields["crop_quality"], "good")
                self.assertEqual(annotation.legacy_notes, "an old note")
                self.assertEqual(annotation.legacy_primary_failure_cause, "Occlusion")

    def test_legacy_content_is_counted_separately_in_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = make_image(root / "x.NEF")
            db = root / "kb.db"
            with AnnotationStore(root / "probe.db") as probe:
                digest = probe.identity_of(image)
            self._make_pre_redesign_db(db, digest, "an old note", "Occlusion")

            with AnnotationStore(db) as store:
                _, summary = summarise(store, [str(image)])
            self.assertEqual(summary.with_legacy_content, 1)
            for counts in summary.field_counts.values():
                self.assertEqual(counts, [], "legacy content must not enter the breakdowns")


class SummaryTests(unittest.TestCase):
    def test_per_field_frequency_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [make_image(root / f"i{i}.NEF", f"pixels {i}".encode()) for i in range(4)]
            with store_in(tmp) as store:
                store.save(paths[0], fields={"crop_quality": "too_small", "agree_with_model_decision": "no"})
                store.save(paths[1], fields={"crop_quality": "too_small", "agree_with_model_decision": "no"})
                store.save(paths[2], fields={"crop_quality": "good", "image_quality": "missing_eye"})
                store.save(paths[3], fields={"image_quality": "missing_eye"})

                _, summary = summarise(store, [str(p) for p in paths])

            self.assertEqual(summary.annotated, 4)
            self.assertEqual(summary.field_counts["crop_quality"][0], ("too_small", 2))
            self.assertEqual(dict(summary.field_counts["image_quality"])["missing_eye"], 2)
            self.assertEqual(dict(summary.field_counts["agree_with_model_decision"]), {"no": 2})
            # Unanswered fields never appear as a value of their own.
            self.assertNotIn("", dict(summary.field_counts["crop_quality"]))

    def test_combinations_describe_the_whole_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [make_image(root / f"i{i}.NEF", f"pixels {i}".encode()) for i in range(2)]
            with store_in(tmp) as store:
                store.save(
                    paths[0],
                    fields={"crop_quality": "good", "image_quality": "out_of_focus", "agree_with_model_decision": "yes"},
                )
                store.save(paths[1], fields={"crop_quality": "good"})
                _, summary = summarise(store, [str(p) for p in paths])

            self.assertEqual(
                summary.combination_fields, ("crop_quality", "image_quality", "agree_with_model_decision")
            )
            combos = dict(summary.combination_counts)
            self.assertEqual(combos[("good", "out_of_focus", "yes")], 1)
            self.assertEqual(combos[("good", "(unset)", "(unset)")], 1)

    def test_no_annotations_yields_empty_breakdowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_image(root / "a.NEF")
            with store_in(tmp) as store:
                _, summary = summarise(store, [str(path)])
            self.assertEqual(summary.annotated, 0)
            self.assertEqual(summary.combination_counts, [])
            for counts in summary.field_counts.values():
                self.assertEqual(counts, [])

    def test_text_summary_has_one_block_per_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_image(root / "a.NEF")
            with store_in(tmp) as store:
                store.save(
                    path,
                    fields={"crop_quality": "too_large", "image_quality": "good", "agree_with_model_decision": "no"},
                )
                _, summary = summarise(store, [str(path)])
                fields_config = store.fields_config
            text = render_summary(summary, fields_config)
            for label in ("Crop Quality:", "Image Quality:", "Agree with Model Decision:"):
                self.assertIn(label, text)
            self.assertIn("Too Large", text)  # the label, not the raw id

    def test_as_dict_round_trips_the_breakdowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_image(root / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, fields={"crop_quality": "good", "image_quality": "good", "agree_with_model_decision": "yes"})
                _, summary = summarise(store, [str(path)])
            payload = summary.as_dict()
            self.assertEqual(
                payload["field_counts"]["crop_quality"], [{"value": "good", "count": 1}]
            )
            self.assertEqual(
                payload["field_vocabularies"]["agree_with_model_decision"], ["yes", "no"]
            )
            # Must survive a JSON round-trip: the report inlines this.
            self.assertEqual(json.loads(json.dumps(payload))["with_legacy_content"], 0)


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

    def test_report_offers_one_dropdown_per_field_with_every_value(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._result(root, root / "kb.db")
            html = build_html(result)
            for field_def in result.annotation_fields_config:
                self.assertIn(f'data-field="{field_def.id}"', html)
                for value in field_def.values:
                    self.assertIn(f'<option value="{value.id}"', html)
                    self.assertIn(value.label, html)
            self.assertIn("ann-field", html)

    def test_the_report_offers_no_free_text_input(self):
        """A textarea or text box would reopen exactly the hole the fixed
        vocabularies were introduced to close."""
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html(self._result(root, root / "kb.db"))
            self.assertNotIn("<textarea", html)
            self.assertNotIn('type="text"', html)

    def test_stored_values_render_pre_selected(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            first = self._result(root, db)
            target = first.errors.false_negatives[0].image_path
            with AnnotationStore(db) as store:
                store.save(
                    target,
                    fields={
                        "crop_quality": "wrong_location",
                        "image_quality": "missing_eye",
                        "agree_with_model_decision": "no",
                    },
                )

            html = build_html(self._result(root, db))
            # The option's value is the stored id; its visible text is the label.
            self.assertIn('<option value="wrong_location" selected>Wrong Location</option>', html)
            self.assertIn('<option value="missing_eye" selected>Missing Eye</option>', html)
            self.assertIn('data-crop-quality="wrong_location"', html)

    def test_summary_section_reports_disagreements(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            first = self._result(root, db)
            target = first.errors.false_negatives[0].image_path
            with AnnotationStore(db) as store:
                store.save(target, fields={"crop_quality": "too_small", "agree_with_model_decision": "no"})

            html = build_html(self._result(root, db))
            self.assertIn("Disagree with model", html)
            self.assertIn("Crop Quality", html)

    def test_report_with_the_field_ui_stays_offline(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html(self._result(root, root / "kb.db"))
            self.assertNotIn("http://", html)
            self.assertNotIn("https://", html)

    def _serve(self, report_dir: Path, db: Path):
        from picklikeme.analyzer.server import make_server

        store = AnnotationStore(db)
        server = make_server(report_dir, store, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, store, f"http://127.0.0.1:{server.server_address[1]}"

    def _post(self, base: str, payload: dict):
        return urllib.request.Request(
            f"{base}/api/annotations",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def test_server_saves_every_field_and_publishes_the_vocabularies(self):
        from picklikeme.analyzer.reports import write_json_report
        from picklikeme.analyzer.reports.html import write_html_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            result = self._result(root, db)
            write_json_report(result, result.config.output_dir / "analysis.json")
            write_html_report(result)

            target = result.errors.false_negatives[0].image_path
            server, store, base = self._serve(result.config.output_dir, db)
            try:
                with urllib.request.urlopen(
                    self._post(
                        base,
                        {
                            "image_path": target,
                            "crop_quality": "too_small",
                            "image_quality": "out_of_focus",
                            "agree_with_model_decision": "no",
                        },
                    )
                ) as response:
                    body = json.load(response)
                self.assertEqual(body["annotation"]["fields"]["crop_quality"], "too_small")
                self.assertEqual(store.get(target).fields["agree_with_model_decision"], "no")

                with urllib.request.urlopen(f"{base}/api/fields") as response:
                    fields = json.load(response)["fields"]
                by_id = {f["id"]: f for f in fields}
                self.assertEqual(
                    set(by_id["crop_quality"].keys()), {"id", "label", "values"}
                )
                self.assertEqual(by_id["crop_quality"]["label"], "Crop Quality")
                self.assertIn({"id": "too_small", "label": "Too Small"}, by_id["crop_quality"]["values"])
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_server_rejects_an_out_of_vocabulary_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "out"
            report_dir.mkdir(parents=True)
            (report_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")
            image = make_image(root / "IMG_1.NEF")

            server, store, base = self._serve(report_dir, root / "kb.db")
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(
                        self._post(base, {"image_path": str(image), "crop_quality": "smallish"})
                    )
                self.assertEqual(ctx.exception.code, 400)
                self.assertTrue(json.load(ctx.exception)["invalid_value"])
                self.assertEqual(store.count(), 0, "an invalid value must not be stored")
            finally:
                server.shutdown()
                server.server_close()
                store.close()


class MetricIsolationTests(unittest.TestCase):
    def test_the_new_fields_never_change_a_metric(self):
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
                    store.save(
                        record.image_path,
                        fields={
                            "crop_quality": "too_small",
                            "image_quality": "out_of_focus",
                            "agree_with_model_decision": "no",
                        },
                    )

            after = run()
            self.assertEqual(metrics_before, {v.name: v.value for v in after.metrics.values})
            self.assertGreater(after.annotation_summary.annotated, 0)


class FalsePositiveAnnotationTests(unittest.TestCase):
    """False positives get the same configured fields as false negatives -
    same schema, same vocabulary, same editor - so the two are directly
    comparable."""

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

    def test_a_false_positive_can_be_saved_with_the_same_fields_as_a_false_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._result(root, root / "kb.db")
            self.assertGreater(len(result.errors.false_positives), 0, "fixture must produce FPs")
            target = result.errors.false_positives[0].image_path

            with store_in(tmp) as store:
                saved = store.save(
                    target,
                    fields={
                        "crop_quality": "too_large",
                        "image_quality": "group_scene",
                        "agree_with_model_decision": "no",
                    },
                )
            self.assertEqual(saved.fields["crop_quality"], "too_large")
            self.assertEqual(saved.fields["image_quality"], "group_scene")

    def test_the_report_renders_an_annotation_panel_for_false_positives(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._result(root, root / "kb.db")
            html = build_html(result)

            self.assertIn('data-scope="false_negative"', html)
            self.assertIn('data-scope="false_positive"', html)
            self.assertIn("False positives - annotate why they were kept", html)
            self.assertIn("False positive summary", html)
            # The decision label differs by direction: FN images were kept by
            # the photographer, FP images were rejected.
            self.assertIn("you kept it", html)
            self.assertIn("you rejected it", html)

    def test_every_false_positive_gets_its_own_panel(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._result(root, root / "kb.db")
            html = build_html(result)

            fp_paths = {r.image_path for r in result.errors.false_positives}
            panels = re.findall(r'<div class="fn" data-path="([^"]*)"', html)
            for path in fp_paths:
                self.assertIn(path, panels)

    def test_group_scene_is_offered_as_an_image_quality_option(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = build_html(self._result(root, root / "kb.db"))
            self.assertIn('<option value="group_scene">Group Scene</option>', html)

    def test_false_negative_and_false_positive_filters_do_not_cross_contaminate(self):
        """Each category has its own filter bar; saving/filtering one must not
        touch the other's panels. Checked at the markup level: each `.ann-scope`
        contains only panels for its own category's paths."""
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._result(root, root / "kb.db")
            html = build_html(result)

            fn_paths = {r.image_path for r in result.errors.false_negatives}
            fp_paths = {r.image_path for r in result.errors.false_positives}

            scopes = re.findall(
                r'<div class="ann-scope" data-scope="([^"]+)">(.*?)</div>\s*</div>\s*</section>',
                html,
                re.S,
            )
            found = {name: body for name, body in scopes}
            self.assertIn("false_negative", found)
            self.assertIn("false_positive", found)
            for path in fn_paths:
                self.assertIn(path, found["false_negative"])
                self.assertNotIn(path, found["false_positive"])
            for path in fp_paths:
                self.assertIn(path, found["false_positive"])
                self.assertNotIn(path, found["false_negative"])

    def test_summary_sections_are_scoped_to_their_own_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            first = self._result(root, db)
            fn_target = first.errors.false_negatives[0].image_path
            fp_target = first.errors.false_positives[0].image_path
            with AnnotationStore(db) as store:
                store.save(fn_target, fields={"crop_quality": "good"})
                store.save(fp_target, fields={"crop_quality": "wrong_location"})

            again = self._result(root, db)
            self.assertEqual(dict(again.annotation_summary.field_counts["crop_quality"]), {"good": 1})
            self.assertEqual(
                dict(again.fp_annotation_summary.field_counts["crop_quality"]), {"wrong_location": 1}
            )

    def test_json_report_carries_both_categories_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            first = self._result(root, db)
            fn_target = first.errors.false_negatives[0].image_path
            fp_target = first.errors.false_positives[0].image_path
            with AnnotationStore(db) as store:
                store.save(fn_target, fields={"crop_quality": "good"})
                store.save(fp_target, fields={"crop_quality": "wrong_location"})

            payload = json.loads(json.dumps(self._result(root, db).as_dict()))
            self.assertEqual(
                payload["false_negative_annotations"]["by_image"][fn_target]["fields"]["crop_quality"],
                "good",
            )
            self.assertNotIn(fp_target, payload["false_negative_annotations"]["by_image"])
            self.assertEqual(
                payload["false_positive_annotations"]["by_image"][fp_target]["fields"]["crop_quality"],
                "wrong_location",
            )
            self.assertNotIn(fn_target, payload["false_positive_annotations"]["by_image"])

    def test_server_saves_a_false_positive_annotation(self):
        from picklikeme.analyzer.reports import write_json_report
        from picklikeme.analyzer.reports.html import write_html_report
        from picklikeme.analyzer.server import make_server

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            result = self._result(root, db)
            write_json_report(result, result.config.output_dir / "analysis.json")
            write_html_report(result)

            target = result.errors.false_positives[0].image_path
            store = AnnotationStore(db)
            server = make_server(result.config.output_dir, store, port=0)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = urllib.request.Request(
                    f"{base}/api/annotations",
                    data=json.dumps(
                        {"image_path": target, "image_quality": "group_scene"}
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    body = json.load(response)
                self.assertEqual(body["annotation"]["fields"]["image_quality"], "group_scene")
                self.assertEqual(store.get(target).fields["image_quality"], "group_scene")
            finally:
                server.shutdown()
                server.server_close()
                store.close()


if __name__ == "__main__":
    unittest.main()
