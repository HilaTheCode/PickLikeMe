"""The review server's HTTP surface.

Two things are being defended here. First, that the endpoints do what the page
assumes - because the page is the only client and a silent shape change would
break it invisibly. Second, that subclassing the annotation server did not
weaken it: the same path confinement must still apply, and the review page must
not have opened a way to read files outside the folder under review.
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
from unittest import mock
from urllib.parse import quote

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotations import AnnotationStore
from picklikeme.organize import REJECTED_DIRNAME, SELECTED_DIRNAME
from picklikeme.review.server import make_review_server
from picklikeme.review.session import ReviewSession
from test_review_session import build_shoot


def make_real_images(paths: list[Path]) -> None:
    """Replace the byte-blob fixtures with decodable JPEGs, so the thumbnail
    endpoint has something real to render."""
    for index, path in enumerate(paths):
        Image.fromarray(np.full((40, 60, 3), (index * 37) % 255, dtype=np.uint8)).save(path)


class ReviewServerTestCase(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: Windows holds the SQLite WAL open briefly
        # after close(); that is teardown timing, not a failure.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.store = AnnotationStore(self.root / "kb.db")
        self.shoot, self.images, self.extra = build_shoot(self.root, ranked=6, unranked=1)
        make_real_images(self.images)
        make_real_images(self.extra)
        self.session = ReviewSession(self.shoot, self.store, keep_percent=33)
        self.server = make_review_server(self.session, self.store, port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.store.close()
        self._tmp.cleanup()

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path) as response:
            return json.load(response)

    def post(self, path: str, payload: dict):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.load(response)


class PageTests(ReviewServerTestCase):
    def test_the_root_serves_a_self_contained_page(self):
        with urllib.request.urlopen(self.base + "/") as response:
            self.assertEqual(response.headers["Content-Type"], "text/html; charset=utf-8")
            body = response.read().decode("utf-8")

        self.assertIn("Arrange Files On Disk", body)
        self.assertIn("Keep", body)
        self.assertIn("Reject", body)
        # Offline and dependency-free: no CDN, no external asset of any kind.
        self.assertNotIn("https://", body)
        self.assertNotIn("<script src=", body)
        self.assertNotIn("<link ", body)

    def test_the_page_ships_empty_and_fetches_its_own_state(self):
        """A folder of thousands of images must render instantly; the gallery
        is filled by api/review/state, not baked into the document."""
        with urllib.request.urlopen(self.base + "/") as response:
            body = response.read().decode("utf-8")
        self.assertIn("api/review/state", body)
        for image in self.images:
            self.assertNotIn(str(image), body)


class PageMarkupTests(unittest.TestCase):
    """Structural checks on the generated document, with no server involved.

    These exist because the suite cannot run the page's JavaScript, so a
    scripting error is invisible to every other test here: asserting that a
    string appears in the HTML says nothing about whether the page works. One
    such bug shipped - `setTheme()` dereferenced a `#theme` button that was
    never in the markup, and because it runs at parse time the exception
    aborted the whole script before `DOMContentLoaded` was ever registered, so
    the gallery silently never loaded.
    """

    def _ids_the_script_reaches_for(self, js: str) -> set[str]:
        return set(re.findall(r"""q\(['"]#([A-Za-z0-9_-]+)['"]\)""", js)) | set(
            re.findall(r"""getElementById\(['"]([A-Za-z0-9_-]+)['"]\)""", js)
        )

    def test_every_element_the_script_looks_up_exists_in_the_markup(self):
        from picklikeme.review.page import build_js, build_page

        html = build_page()
        present = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))

        missing = sorted(self._ids_the_script_reaches_for(build_js()) - present)

        self.assertEqual(
            missing,
            [],
            f"the page script queries element(s) that the markup never defines: {missing}. "
            "A null dereference here aborts the whole script and the gallery never loads.",
        )

    def test_the_theme_toggle_is_present_and_survives_a_missing_button(self):
        """The specific regression, from both directions: the button is in the
        markup, and the code that labels it is written so it could not kill the
        page even if a future edit removed it again."""
        from picklikeme.review.page import build_js, build_page

        self.assertIn('id="theme"', build_page())
        js = build_js()
        self.assertIn("const button = q('#theme');", js)
        self.assertIn("if(button) button.textContent", js)


class StateEndpointTests(ReviewServerTestCase):
    def test_state_describes_every_image_and_its_counts(self):
        payload = self.get("/api/review/state")["state"]

        self.assertEqual(payload["counts"]["total"], 7)
        self.assertEqual(payload["counts"]["selected"], 2)
        self.assertEqual(payload["counts"]["untouched"], 1)
        self.assertEqual(len(payload["images"]), 7)
        first = payload["images"][0]
        for key in ("image_path", "filename", "score", "rank", "decision", "state", "missing_file"):
            self.assertIn(key, first)

    def test_images_arrive_best_first(self):
        images = self.get("/api/review/state")["state"]["images"]
        scores = [i["score"] for i in images if i["score"] is not None]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIsNone(images[-1]["score"], "unranked images sort last")


class DecisionEndpointTests(ReviewServerTestCase):
    def test_a_decision_is_saved_and_the_new_state_returned(self):
        worst = str(self.images[-1])

        payload = self.post("/api/review/decision", {"image_path": worst, "decision": "keep"})

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["state"]["counts"]["manual"], 1)
        self.assertIn(worst, self.session.selected_paths())
        # Persisted, not just echoed.
        self.assertEqual(self.store.review_decision_count(), 1)

    def test_a_decision_can_be_cleared_with_null(self):
        best = str(self.images[0])
        self.post("/api/review/decision", {"image_path": best, "decision": "reject"})

        payload = self.post("/api/review/decision", {"image_path": best, "decision": None})

        self.assertEqual(payload["state"]["counts"]["manual"], 0)
        self.assertEqual(self.store.review_decision_count(), 0)

    def test_an_invalid_decision_is_refused_and_nothing_is_stored(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/decision", {"image_path": str(self.images[0]), "decision": "maybe"})

        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.store.review_decision_count(), 0)

    def test_a_missing_image_path_is_a_bad_request(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/decision", {"decision": "keep"})
        self.assertEqual(ctx.exception.code, 400)

    def test_an_image_outside_the_session_is_refused(self):
        outside = self.root / "elsewhere.jpg"
        outside.write_bytes(b"not part of this shoot")

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/decision", {"image_path": str(outside), "decision": "keep"})

        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.store.review_decision_count(), 0)


class KeepPercentEndpointTests(ReviewServerTestCase):
    def test_changing_the_percentage_reclassifies(self):
        payload = self.post("/api/review/keep-percent", {"keep_percent": 100})
        self.assertEqual(payload["state"]["counts"]["selected"], 6)
        self.assertEqual(payload["state"]["counts"]["rejected"], 0)

    def test_an_out_of_range_percentage_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/keep-percent", {"keep_percent": 150})
        self.assertEqual(ctx.exception.code, 400)


class ThumbnailEndpointTests(ReviewServerTestCase):
    def test_a_thumbnail_is_served_for_an_image_in_the_folder(self):
        url = "/thumb?path=" + quote(str(self.images[0]), safe="")
        with urllib.request.urlopen(self.base + url) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            data = response.read()
        self.assertGreater(len(data), 0)

    def test_the_boxes_variant_is_served_too(self):
        """Toggling the overlay must not fail when an image has no recorded
        detections - it falls back to the plain thumbnail."""
        url = "/thumb?path=" + quote(str(self.images[0]), safe="") + "&boxes=1"
        with urllib.request.urlopen(self.base + url) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertGreater(len(response.read()), 0)

    def test_a_thumbnail_outside_the_folder_is_refused(self):
        """The confinement inherited from the annotation server must still
        hold: the review page must not become a way to read the whole disk."""
        secret = self.root / "secret.jpg"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(secret)

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/thumb?path=" + quote(str(secret), safe=""))
        self.assertEqual(ctx.exception.code, 403)

    def test_a_missing_path_parameter_is_a_bad_request(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/thumb")
        self.assertEqual(ctx.exception.code, 400)


class ArrangeEndpointTests(ReviewServerTestCase):
    def test_a_dry_run_reports_the_plan_without_moving_anything(self):
        payload = self.post("/api/review/arrange", {"dry_run": True})

        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["result"]["selected"], 2)
        self.assertEqual(payload["result"]["rejected"], 4)
        self.assertIn(SELECTED_DIRNAME, payload["result"]["selected_dir"])
        for path in self.images:
            self.assertTrue(path.exists(), "a dry run must not move a file")

    def test_arranging_moves_the_files_and_returns_fresh_state(self):
        payload = self.post("/api/review/arrange", {"dry_run": False})

        self.assertEqual(payload["result"]["moved"], 6)
        self.assertEqual(len(list((self.shoot / SELECTED_DIRNAME).glob("*.jpg"))), 2)
        self.assertEqual(len(list((self.shoot / REJECTED_DIRNAME).glob("*.jpg"))), 4)
        # The gallery now describes the new locations.
        self.assertEqual(payload["state"]["counts"]["missing_file"], 0)

    def test_unranked_images_are_left_in_place_by_arranging(self):
        self.post("/api/review/arrange", {"dry_run": False})

        self.assertTrue(self.extra[0].exists())
        self.assertEqual(self.extra[0].parent, self.shoot)

    def test_manual_decisions_are_honoured_by_arranging(self):
        worst = str(self.images[-1])
        self.post("/api/review/decision", {"image_path": worst, "decision": "keep"})

        self.post("/api/review/arrange", {"dry_run": False})

        self.assertTrue((self.shoot / SELECTED_DIRNAME / Path(worst).name).exists())


class InheritedBehaviourTests(ReviewServerTestCase):
    """Subclassing must not have cost anything the parent provided, nor added
    a way around its guards."""

    def test_the_annotation_api_still_works(self):
        payload = self.get("/api/health")
        self.assertTrue(payload["ok"])
        self.assertIn("annotations", self.get("/api/annotations"))

    def test_open_folder_reaches_the_os(self):
        with mock.patch("picklikeme.analyzer.server.open_in_file_manager") as opened:
            payload = self.get("/open-folder?path=" + quote(str(self.shoot), safe=""))
        self.assertEqual(payload, {"ok": True})
        opened.assert_called_once()

    def test_open_folder_outside_the_shoot_is_refused(self):
        outside = self.root / "elsewhere"
        outside.mkdir()
        with mock.patch("picklikeme.analyzer.server.open_in_file_manager") as opened:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(self.base + "/open-folder?path=" + quote(str(outside), safe=""))
            self.assertEqual(ctx.exception.code, 403)
        opened.assert_not_called()

    def test_no_static_file_can_be_read_through_the_server(self):
        """The page has no assets, so the inherited static-file fallback is
        closed off entirely rather than left pointing at the photo folder."""
        (self.shoot / "notes.txt").write_text("private", encoding="utf-8")

        for path in ("/notes.txt", "/../kb.db", "/.picklikeme/ranking.csv"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(self.base + path)
            self.assertIn(ctx.exception.code, (403, 404))

    def test_an_unknown_endpoint_is_a_clean_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/nonsense", {})
        self.assertEqual(ctx.exception.code, 404)

    def test_the_server_binds_loopback_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
