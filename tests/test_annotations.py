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

from picklikeme.analyzer.annotations import DEFAULT_ANNOTATIONS_DB, AnnotationStore
from picklikeme.identity import IdentityUnavailable, cache_key, image_identity
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
        target.write_bytes(f"frame {index} score {score}".encode())
        rows.append([str(target), score, 1 if keep else 0])

    rows.sort(key=lambda row: -row[1])
    ranked = [[position, row[0], f"{row[1]:.6f}", row[2]] for position, row in enumerate(rows, start=1)]
    return write_ranking(tmp / "rankings.csv", ranked), selected, rejected


EXPECTED_FALSE_NEGATIVES = sum(1 for keep, score in FN_SCORES if keep and score < 0.5)


def make_image(path: Path, content: bytes | None = None) -> Path:
    """A real file, since identity is derived from content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if content is not None else f"pixels:{path.name}".encode())
    return path


class IdentityTests(unittest.TestCase):
    """The canonical identity must depend on content and nothing else."""

    def test_identity_is_independent_of_name_and_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = make_image(root / "shoot" / "IMG_1.NEF", b"same pixels")
            renamed = make_image(root / "archive" / "2026" / "wing_shot.NEF", b"same pixels")
            self.assertEqual(image_identity(original), image_identity(renamed))

    def test_different_content_gives_different_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertNotEqual(
                image_identity(make_image(root / "a.NEF", b"pixels A")),
                image_identity(make_image(root / "b.NEF", b"pixels B")),
            )

    def test_same_name_different_content_is_not_confused(self):
        """Camera counters reset, so two shoots hold DSC_0001.NEF. The old
        filename fallback could mix these up; content identity cannot."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = make_image(root / "shootA" / "DSC_0001.NEF", b"frame from shoot A")
            b = make_image(root / "shootB" / "DSC_0001.NEF", b"frame from shoot B")
            self.assertNotEqual(image_identity(a), image_identity(b))

    def test_identity_carries_a_scheme_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(image_identity(make_image(Path(tmp) / "a.NEF")).startswith("p1:"))

    def test_size_participates_so_truncation_changes_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF", b"0123456789")
            before = image_identity(path)
            path.write_bytes(b"01234")
            self.assertNotEqual(before, image_identity(path))

    def test_large_files_use_head_and_tail(self):
        """Beyond the threshold only the ends are read, but a change at either
        end must still be detected."""
        from picklikeme.identity import WHOLE_FILE_THRESHOLD

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = bytes(WHOLE_FILE_THRESHOLD + 5000)
            base = make_image(root / "big.NEF", b"HEAD" + body + b"TAIL")
            head_changed = make_image(root / "big2.NEF", b"HEAx" + body + b"TAIL")
            tail_changed = make_image(root / "big3.NEF", b"HEAD" + body + b"TAIx")
            self.assertNotEqual(image_identity(base), image_identity(head_changed))
            self.assertNotEqual(image_identity(base), image_identity(tail_changed))

    def test_missing_and_empty_files_are_explicit_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(IdentityUnavailable):
                image_identity(root / "nope.NEF")
            empty = root / "empty.NEF"
            empty.write_bytes(b"")
            with self.assertRaises(IdentityUnavailable) as ctx:
                image_identity(empty)
            self.assertIn("empty", ctx.exception.reason)

    def test_cache_key_remains_path_derived_and_distinct_from_identity(self):
        """The crop cache legitimately keys on location; identity does not.
        Keeping both in one module makes the difference explicit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = make_image(root / "one" / "x.NEF", b"same pixels")
            b = make_image(root / "two" / "x.NEF", b"same pixels")
            self.assertNotEqual(cache_key(a), cache_key(b))          # location differs
            self.assertEqual(image_identity(a), image_identity(b))   # content does not

    def test_cache_key_matches_the_crop_cache(self):
        """One implementation, not two: bird_crop must use this exact key."""
        from picklikeme.bird_crop import crop_cache_path

        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            self.assertEqual(crop_cache_path(tmp, path).stem, cache_key(path))


class StorageTests(unittest.TestCase):
    def test_database_is_created_ready_to_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                self.assertTrue(store.db_path.exists())
                self.assertEqual(store.count(), 0)

    def test_save_and_reload_keeps_display_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "IMG_0001.NEF")
            with store_in(tmp) as store:
                saved = store.save(path, fields={"crop_quality": "too_small", "image_quality": "out_of_focus"})
                self.assertTrue(saved.image_hash.startswith("p1:"))

            with AnnotationStore(Path(tmp) / "kb" / "fn.db") as reopened:
                annotation = reopened.get(path)
                self.assertIsNotNone(annotation)
                self.assertEqual(annotation.fields["crop_quality"], "too_small")
                self.assertEqual(annotation.filename, "IMG_0001.NEF")

    def test_saving_again_replaces_the_answers_and_keeps_created_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                first = store.save(path, fields={"crop_quality": "good"})
                second = store.save(path, fields={"crop_quality": "too_large", "image_quality": "missing_eye"})

                self.assertEqual(second.created_at, first.created_at)
                self.assertEqual(store.get(path).fields["crop_quality"], "too_large")
                self.assertEqual(store.get(path).fields["image_quality"], "missing_eye")

    def test_clearing_everything_deletes_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF")
            with store_in(tmp) as store:
                store.save(path, fields={"crop_quality": "good"})
                self.assertEqual(store.count(), 1)
                store.save(path)
                self.assertEqual(store.count(), 0)
                self.assertIsNone(store.get(path))

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

    def test_annotation_follows_a_reorganised_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = make_image(root / "D_drive" / "2026" / "india" / "a.NEF", b"elephant frame")
            with store_in(tmp) as store:
                store.save(original, fields={"crop_quality": "wrong_location"})
                moved = make_image(root / "E_drive" / "archive" / "asia" / "a.NEF", b"elephant frame")
                self.assertEqual(store.get(moved).fields["crop_quality"], "wrong_location")

    def test_duplicate_filenames_are_never_confused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with store_in(tmp) as store:
                a = make_image(root / "shootA" / "DSC_0001.NEF", b"frame A")
                b = make_image(root / "shootB" / "DSC_0001.NEF", b"frame B")
                store.save(a, fields={"image_quality": "good"})

                self.assertEqual(store.get(a).fields["image_quality"], "good")
                self.assertIsNone(store.get(b), "a same-named different image must not match")

    def test_unreadable_image_is_an_explicit_failure_not_a_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                missing = Path(tmp) / "gone.NEF"
                with self.assertRaises(IdentityUnavailable):
                    store.get(missing)
                with self.assertRaises(IdentityUnavailable):
                    store.save(missing, fields={"crop_quality": "good"})

    def test_get_many_separates_unresolved_from_unannotated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = make_image(root / "good.NEF")
            plain = make_image(root / "plain.NEF")
            missing = root / "missing.NEF"
            with store_in(tmp) as store:
                store.save(good, fields={"crop_quality": "good"})
                found, unresolved = store.get_many([str(good), str(plain), str(missing)])

            self.assertEqual(list(found), [str(good)])
            self.assertEqual([item.filename for item in unresolved], ["missing.NEF"])

    def test_identity_is_memoised_against_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_image(Path(tmp) / "a.NEF", b"original")
            with store_in(tmp) as store:
                first = store.identity_of(path)
                self.assertEqual(first, store.identity_of(path))  # cache hit

                # Rewriting changes size, so the cache entry must be invalidated.
                path.write_bytes(b"completely different content")
                self.assertNotEqual(first, store.identity_of(path))

    def test_default_database_lives_outside_any_output_directory(self):
        self.assertNotIn("analysis", DEFAULT_ANNOTATIONS_DB.parts[-2:])
        self.assertEqual(DEFAULT_ANNOTATIONS_DB.name, "false_negatives.db")


class MigrationTests(unittest.TestCase):
    """v1 keyed on a path digest; those records must be re-keyed automatically."""

    def _make_v1_db(self, db_path: Path, entries):
        """Build a database in the old path-keyed shape."""
        import hashlib
        import sqlite3

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE schema_info (version INTEGER NOT NULL);
            CREATE TABLE categories (name TEXT PRIMARY KEY, ordering INTEGER, builtin INTEGER);
            CREATE TABLE annotations (
                image_key TEXT PRIMARY KEY, image_path TEXT NOT NULL, filename TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE annotation_categories (
                image_key TEXT NOT NULL, category TEXT NOT NULL, PRIMARY KEY (image_key, category));
            """
        )
        conn.execute("INSERT INTO schema_info(version) VALUES (1)")
        for path, categories, notes in entries:
            key = hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:20]
            conn.execute(
                "INSERT INTO annotations VALUES (?,?,?,?,?,?)",
                (key, str(path), Path(path).name, notes, "2026-01-01T00:00:00", "2026-01-02T00:00:00"),
            )
            for category in categories:
                conn.execute("INSERT INTO annotation_categories VALUES (?,?)", (key, category))
        conn.commit()
        conn.close()

    def test_existing_annotations_are_rekeyed_preserving_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = make_image(root / "a.NEF", b"bird")
            db = root / "kb.db"
            self._make_v1_db(db, [(image, ["Action shot", "Backlit"], "keep this note")])

            with AnnotationStore(db) as store:
                self.assertEqual(store.migration.migrated, 1)
                self.assertEqual(store.count(), 1)
                annotation = store.get(image)
                # Re-keyed, not re-interpreted: v1 content stays legacy content.
                self.assertEqual(annotation.legacy_categories, ["Action shot", "Backlit"])
                self.assertEqual(annotation.legacy_notes, "keep this note")
                self.assertEqual(annotation.created_at, "2026-01-01T00:00:00")
                self.assertTrue(annotation.image_hash.startswith("p1:"))

    def test_migration_is_idempotent_and_creates_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = make_image(root / "a.NEF", b"bird")
            db = root / "kb.db"
            self._make_v1_db(db, [(image, ["Backlit"], "note")])

            for _ in range(3):
                with AnnotationStore(db) as store:
                    self.assertEqual(store.count(), 1)

    def test_two_old_records_for_the_same_image_are_merged(self):
        """The same file copied to two paths was two v1 records; it is one image."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_image(root / "one" / "a.NEF", b"identical pixels")
            second = make_image(root / "two" / "b.NEF", b"identical pixels")
            db = root / "kb.db"
            self._make_v1_db(
                db, [(first, ["Backlit"], "from one"), (second, ["Wrong crop"], "from two")]
            )

            with AnnotationStore(db) as store:
                self.assertEqual(store.count(), 1, "must not duplicate the same image")
                self.assertEqual(store.migration.merged, 1)
                annotation = store.get(first)
                self.assertEqual(annotation.legacy_categories, ["Backlit", "Wrong crop"])
                self.assertIn("from one", annotation.legacy_notes)
                self.assertIn("from two", annotation.legacy_notes)

    def test_unresolvable_records_are_kept_not_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            self._make_v1_db(db, [(root / "vanished.NEF", ["Backlit"], "precious note")])

            with AnnotationStore(db) as store:
                self.assertEqual(store.count(), 0)
                self.assertEqual(len(store.migration.unmigrated), 1)
                self.assertEqual(store.unmigrated()[0].filename, "vanished.NEF")

            # The note survives in the parked table, so nothing was destroyed.
            import sqlite3

            conn = sqlite3.connect(db)
            row = conn.execute("SELECT notes, categories FROM unmigrated_v1").fetchone()
            conn.close()
            self.assertEqual(row[0], "precious note")
            self.assertEqual(row[1], "Backlit")

    def test_a_fresh_database_reports_no_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store_in(tmp) as store:
                self.assertFalse(store.migration.ran)
                self.assertEqual(store.migration.render(), "")


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
                    store.save(
                        record.image_path,
                        fields={
                            "crop_quality": "too_small",
                            "image_quality": "out_of_focus",
                            "agree_with_model_decision": "no",
                        },
                    )
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
                    annotate_detections=False,
                )
            )
            self.assertIsNone(result.annotation_summary)
            self.assertFalse(db.exists(), "the database must not be created when disabled")

    def test_false_negatives_and_false_positives_are_both_annotated_but_kept_separate(self):
        """Both categories are in scope (same schema, same fields), but each
        summary is scoped to its own images - a false positive must never
        appear in the false-negative breakdown or vice versa."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            result = self._run(root, db)
            fn_paths = {r.image_path for r in result.errors.false_negatives}
            fp_paths = {r.image_path for r in result.errors.false_positives}

            self.assertGreater(len(fn_paths), 0)
            self.assertGreater(len(fp_paths), 0)
            self.assertEqual(result.annotation_summary.total_images, len(fn_paths))
            self.assertEqual(result.fp_annotation_summary.total_images, len(fp_paths))
            for path in result.annotation_summary.unannotated:
                self.assertNotIn(path, fp_paths)
            for path in result.fp_annotation_summary.unannotated:
                self.assertNotIn(path, fn_paths)


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

    def test_panels_show_save_and_the_field_editors_with_no_edit_click_required(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._result(root, root / "kb.db")
            html = build_html(result)

            self.assertNotIn("btn-edit", html)
            self.assertNotIn("btn-cancel", html)
            self.assertIn("btn-save", html)
            self.assertIn("ann-field", html)
            for label in ("Crop Quality", "Image Quality", "Agree with Model Decision"):
                self.assertIn(label, html)
            self.assertIn("False negative summary", html)
            # Filters
            self.assertIn('class="f-state"', html)
            self.assertIn('class="f-field"', html)
            self.assertIn("not annotated", html)

    def test_existing_annotations_are_reloaded_into_the_page(self):
        from picklikeme.analyzer.reports.html import build_html

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "kb.db"
            first = self._result(root, db)
            target = first.errors.false_negatives[0].image_path
            with AnnotationStore(db) as store:
                store.save(target, fields={"crop_quality": "too_small", "agree_with_model_decision": "no"})

            again = self._result(root, db)
            html = build_html(again)

            # The option's value is the stored id; its text is the configured label.
            self.assertIn('<option value="too_small" selected>Too Small</option>', html)
            # Inlined so a file:// report still shows what is known.
            self.assertIn("window.PLM_ANNOTATIONS=", html)
            self.assertEqual(
                json.loads(json.dumps(again.annotations[target].as_dict()))["fields"]["crop_quality"],
                "too_small",
            )

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

    def test_health_fields_and_save_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = self._prepare(root)
            server, store, base = self._serve(report_dir, root / "kb.db")
            try:
                with urllib.request.urlopen(f"{base}/api/health") as response:
                    self.assertTrue(json.load(response)["ok"])

                with urllib.request.urlopen(f"{base}/api/fields") as response:
                    fields = json.load(response)["fields"]
                    self.assertIn("crop_quality", [f["id"] for f in fields])
                    crop_field = next(f for f in fields if f["id"] == "crop_quality")
                    self.assertIn({"id": "too_small", "label": "Too Small"}, crop_field["values"])

                image = root / "IMG_1.NEF"
                image.write_bytes(b"real pixels")
                payload = json.dumps(
                    {
                        "image_path": str(image),
                        "crop_quality": "good",
                        "image_quality": "missing_eye",
                        "agree_with_model_decision": "yes",
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
                self.assertEqual(body["annotation"]["fields"]["image_quality"], "missing_eye")

                # Persisted, not just echoed.
                self.assertEqual(store.get(image).fields["agree_with_model_decision"], "yes")

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
                    data=json.dumps({"crop_quality": "Good"}).encode("utf-8"),
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

    def test_saving_without_identity_is_refused_explicitly(self):
        """No fallback: an unreadable image cannot be annotated, and the UI is
        told why rather than silently attaching the note to a guess."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, store, base = self._serve(self._prepare(root), root / "kb.db")
            try:
                request = urllib.request.Request(
                    f"{base}/api/annotations",
                    data=json.dumps(
                        {"image_path": str(root / "not_on_disk.NEF"), "crop_quality": "Good"}
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(request)
                self.assertEqual(ctx.exception.code, 409)
                body = json.load(ctx.exception)
                self.assertTrue(body["identity_unavailable"])
                self.assertIn("identity", body["error"])
                self.assertEqual(store.count(), 0, "nothing may be stored without identity")
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
