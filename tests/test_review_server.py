"""The review server's HTTP surface.

Two things are being defended here. First, that the endpoints do what the page
assumes - because the page is the only client and a silent shape change would
break it invisibly. Second, that subclassing the annotation server did not
weaken it: the same path confinement must still apply, and the review page must
not have opened a way to read files outside the folder under review.
"""

import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from io import BytesIO
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

        self.assertIn("Arrange Files", body)
        self.assertIn("Keep", body)
        self.assertIn("Reject", body)
        self.assertIn("Neutral", body)
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

    def test_the_toolbar_never_repeats_the_application_s_own_name(self):
        """The window/tab title (<title>) already identifies the app - the
        in-page toolbar must not duplicate it (Phase 3 of the redesign)."""
        with urllib.request.urlopen(self.base + "/") as response:
            body = response.read().decode("utf-8")

        # <title> is allowed to say it; strip that one occurrence and check
        # the rest of the document (the visible toolbar/body) does not.
        title_match = re.search(r"<title>(.*?)</title>", body)
        self.assertIsNotNone(title_match)
        body_without_title = body.replace(title_match.group(0), "", 1)
        self.assertNotIn("PickLikeMe", body_without_title)
        self.assertNotIn("<h1>", body_without_title)


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

    def test_the_generated_script_is_syntactically_valid_javascript(self):
        """The same failure mode as the class docstring's #theme bug, from a
        different cause: an unescaped apostrophe inside a JS string literal
        (e.g. an f-string/Python-string writer using `\\'` where `\\\\'` was
        needed - Python's own parser silently eats the single backslash,
        leaving a bare `'` that ends the JS string early) also aborts the
        whole script at parse time before DOMContentLoaded is ever
        registered, and every other test here only pattern-matches source
        text, so none of them would notice. `node --check` parses without
        executing, so this needs no DOM and stays fast; skipped if `node`
        is not on PATH rather than failing the suite over a missing tool.
        """
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not on PATH")

        from picklikeme.review.page import build_js

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(build_js())
            script_path = handle.name
        try:
            result = subprocess.run(
                [node, "--check", script_path],
                capture_output=True,
                text=True,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

        self.assertEqual(result.returncode, 0, f"generated JS has a syntax error:\n{result.stderr}")

    def test_every_endpoint_answers_with_an_envelope_not_the_state_itself(self):
        """Every review endpoint answers `{ok, ..., state}` - arrange and
        reconcile carry a sibling (`result`, `recovered`) alongside `state`,
        which is exactly why `api()` hands back the whole envelope rather than
        unwrapping it once for every caller. That makes ".state" a contract
        each call site must honour for itself: assigning the raw envelope to
        PLM.state (skipping the unwrap) type-checks fine and fails silently
        until `render()` dereferences `undefined.total` - the exact shape of
        bug that shipped in the initial `api/review/state` load in boot().

        This can't run the page's JS to prove it at runtime (see the class
        docstring), so it encodes the same invariant as a source pattern: every
        `PLM.state = ` assignment must go through a `.state` unwrap, and none
        may assign a raw `api(...)` result directly.
        """
        from picklikeme.review.page import build_js

        js = build_js()
        # The right-hand side of each assignment, not the whole statement: every
        # line's left-hand side is the literal text "PLM.state", so searching
        # the whole line for ".state" is trivially true and catches nothing -
        # exactly how this check first shipped, and exactly why the regression
        # it was meant to catch reached the user instead.
        right_hand_sides = [
            match.group(1).strip() for match in re.finditer(r"PLM\.state\s*=\s*(.+?);", js, re.S)
        ]
        self.assertGreaterEqual(len(right_hand_sides), 4, "expected an assignment per endpoint call site")

        not_unwrapped = [rhs for rhs in right_hand_sides if not rhs.endswith(".state")]
        self.assertEqual(
            not_unwrapped,
            [],
            "every `PLM.state = ...` assignment must unwrap `.state` from the API envelope "
            f"on its right-hand side; found one that does not: {not_unwrapped}",
        )


class StateEndpointTests(ReviewServerTestCase):
    def test_state_describes_every_image_and_its_counts(self):
        payload = self.get("/api/review/state")["state"]

        self.assertEqual(payload["counts"]["total"], 7)
        # Nobody has reviewed anything yet - every image starts Neutral,
        # regardless of what the AI ranking would suggest.
        self.assertEqual(payload["counts"]["keep"], 0)
        self.assertEqual(payload["counts"]["reject"], 0)
        self.assertEqual(payload["counts"]["neutral"], 7)
        self.assertEqual(len(payload["images"]), 7)
        first = payload["images"][0]
        for key in (
            "image_path", "filename", "score", "rank", "captured_at", "detected_category",
            "review_status", "ai_suggestion", "missing_file",
        ):
            self.assertIn(key, first)

    def test_images_arrive_best_first(self):
        images = self.get("/api/review/state")["state"]["images"]
        scores = [i["score"] for i in images if i["score"] is not None]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIsNone(images[-1]["score"], "unranked images sort last")

    def test_every_image_starts_neutral_with_an_independent_ai_suggestion(self):
        """The AI's own opinion (ai_suggestion) is present from the very
        first load, but never pre-decides review_status."""
        images = self.get("/api/review/state")["state"]["images"]
        ranked = [i for i in images if i["score"] is not None]
        unranked = [i for i in images if i["score"] is None]

        self.assertTrue(all(i["review_status"] == "neutral" for i in images))
        self.assertTrue(all(i["ai_suggestion"] in ("keep", "reject") for i in ranked))
        self.assertTrue(all(i["ai_suggestion"] is None for i in unranked))


class StatusEndpointTests(ReviewServerTestCase):
    """The photographer's own Keep/Reject/Neutral for a single image."""

    def test_a_status_is_saved_and_the_new_state_returned(self):
        worst = str(self.images[-1])

        payload = self.post("/api/review/status", {"image_path": worst, "status": "keep"})

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["state"]["counts"]["keep"], 1)
        self.assertIn(worst, self.session.keep_paths())
        # Persisted, not just echoed.
        self.assertEqual(self.store.review_decision_count(), 1)

    def test_a_status_can_be_cleared_back_to_neutral(self):
        best = str(self.images[0])
        self.post("/api/review/status", {"image_path": best, "status": "reject"})

        payload = self.post("/api/review/status", {"image_path": best, "status": "neutral"})

        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        self.assertEqual(by_path[best]["review_status"], "neutral")
        self.assertEqual(payload["state"]["counts"]["keep"], 0)
        self.assertEqual(payload["state"]["counts"]["reject"], 0)
        self.assertEqual(self.store.review_decision_count(), 0)

    def test_clearing_returns_to_neutral_even_when_the_ai_would_suggest_otherwise(self):
        """The bug this whole model exists to fix, at the HTTP boundary: a
        cleared status must read back as Neutral, never as whatever the AI's
        own ranking would have picked at the current threshold."""
        self.post("/api/review/keep-percent", {"keep_percent": 100})  # AI "suggests" keeping everything
        best = str(self.images[0])
        self.post("/api/review/status", {"image_path": best, "status": "reject"})

        payload = self.post("/api/review/status", {"image_path": best, "status": "neutral"})

        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        self.assertEqual(by_path[best]["review_status"], "neutral")
        self.assertEqual(by_path[best]["ai_suggestion"], "keep", "the AI's own opinion is untouched")

    def test_an_invalid_status_is_refused_and_nothing_is_stored(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/status", {"image_path": str(self.images[0]), "status": "maybe"})

        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.store.review_decision_count(), 0)

    def test_a_missing_image_path_is_a_bad_request(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/status", {"status": "keep"})
        self.assertEqual(ctx.exception.code, 400)

    def test_a_missing_status_is_a_bad_request(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/status", {"image_path": str(self.images[0])})
        self.assertEqual(ctx.exception.code, 400)

    def test_an_image_outside_the_session_is_refused(self):
        outside = self.root / "elsewhere.jpg"
        outside.write_bytes(b"not part of this shoot")

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/status", {"image_path": str(outside), "status": "keep"})

        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.store.review_decision_count(), 0)

    def test_a_reason_is_saved_alongside_keep_or_reject(self):
        worst = str(self.images[-1])

        payload = self.post(
            "/api/review/status", {"image_path": worst, "status": "reject", "reason": "eyes_not_seen"}
        )

        image = next(i for i in payload["state"]["images"] if i["image_path"] == worst)
        self.assertEqual(image["reason"], "eyes_not_seen")
        self.assertEqual(self.store.review_decisions()[0]["reason"], "eyes_not_seen")

    def test_an_invalid_reason_is_refused_and_nothing_is_stored(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post(
                "/api/review/status",
                {"image_path": str(self.images[0]), "status": "keep", "reason": "squinting"},
            )

        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.store.review_decision_count(), 0)

    def test_clearing_to_neutral_clears_its_reason_too(self):
        best = str(self.images[0])
        self.post("/api/review/status", {"image_path": best, "status": "keep", "reason": "clear_eyes_seen"})

        self.post("/api/review/status", {"image_path": best, "status": "neutral"})

        self.assertEqual(self.store.review_decision_count(), 0)


class BulkStatusEndpointTests(ReviewServerTestCase):
    """The multi-select toolbar's one request for many images."""

    def test_marking_several_images_keep_in_one_call(self):
        paths = [str(self.images[4]), str(self.images[5])]

        payload = self.post("/api/review/bulk-status", {"image_paths": paths, "status": "keep"})

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["applied"], 2)
        self.assertEqual(payload["failed"], [])
        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        for path in paths:
            self.assertEqual(by_path[path]["review_status"], "keep")

    def test_marking_several_images_reject_in_one_call(self):
        paths = [str(self.images[0]), str(self.images[1])]

        payload = self.post("/api/review/bulk-status", {"image_paths": paths, "status": "reject"})

        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        for path in paths:
            self.assertEqual(by_path[path]["review_status"], "reject")

    def test_bulk_clearing_returns_every_path_to_neutral(self):
        """The bulk half of the same bug fix as StatusEndpointTests' single-
        image version - both single and multiple images must return to a
        real Neutral, not silently to whatever the AI would have picked."""
        paths = [str(self.images[0]), str(self.images[1])]
        self.post("/api/review/keep-percent", {"keep_percent": 100})
        self.post("/api/review/bulk-status", {"image_paths": paths, "status": "reject"})

        payload = self.post("/api/review/bulk-status", {"image_paths": paths, "status": "neutral"})

        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        for path in paths:
            self.assertEqual(by_path[path]["review_status"], "neutral")

    def test_a_stale_path_is_reported_not_fatal_to_the_rest_of_the_batch(self):
        good = str(self.images[0])
        stale = str(self.root / "moved_away.jpg")

        payload = self.post("/api/review/bulk-status", {"image_paths": [good, stale], "status": "keep"})

        self.assertEqual(payload["applied"], 1)
        self.assertEqual(payload["failed"], [stale])
        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        self.assertEqual(by_path[good]["review_status"], "keep")

    def test_an_empty_list_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/bulk-status", {"image_paths": [], "status": "keep"})
        self.assertEqual(ctx.exception.code, 400)

    def test_a_non_list_image_paths_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/bulk-status", {"image_paths": str(self.images[0]), "status": "keep"})
        self.assertEqual(ctx.exception.code, 400)

    def test_a_non_string_entry_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post(
                "/api/review/bulk-status", {"image_paths": [str(self.images[0]), 123], "status": "keep"}
            )
        self.assertEqual(ctx.exception.code, 400)

    def test_a_missing_status_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/bulk-status", {"image_paths": [str(self.images[0])]})
        self.assertEqual(ctx.exception.code, 400)

    def test_an_invalid_status_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post(
                "/api/review/bulk-status",
                {"image_paths": [str(self.images[0])], "status": "maybe"},
            )
        self.assertEqual(ctx.exception.code, 400)

    def test_no_reason_can_be_sent_with_a_bulk_action(self):
        """A bulk action never records a per-image reason - see
        ReviewSession.set_review_statuses - so a caller trying to sneak one
        in is simply ignored, not an error."""
        payload = self.post(
            "/api/review/bulk-status",
            {"image_paths": [str(self.images[0])], "status": "keep", "reason": "clear_eyes_seen"},
        )
        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        self.assertIsNone(by_path[str(self.images[0])]["reason"])


class ApplyAiSuggestionsEndpointTests(ReviewServerTestCase):
    """Bulk-accepting the AI's current suggestion - the one endpoint that
    lets the ranking set a review status at all, and only because the
    photographer explicitly asked for it."""

    def test_applies_the_suggestion_to_every_neutral_ranked_image(self):
        payload = self.post("/api/review/apply-ai-suggestions", {})

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["applied"], 6, "the 6 ranked images; the 1 unranked has no suggestion")
        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        for image in payload["state"]["images"]:
            if image["ai_suggestion"] is not None:
                self.assertEqual(image["review_status"], image["ai_suggestion"])
        self.assertEqual(by_path[str(self.extra[0])]["review_status"], "neutral")

    def test_never_touches_an_image_already_decided(self):
        best = str(self.images[0])
        self.post("/api/review/status", {"image_path": best, "status": "reject"})

        self.post("/api/review/apply-ai-suggestions", {})

        payload = self.get("/api/review/state")["state"]
        by_path = {i["image_path"]: i for i in payload["images"]}
        self.assertEqual(by_path[best]["review_status"], "reject")

    def test_running_it_twice_is_a_no_op_the_second_time(self):
        self.post("/api/review/apply-ai-suggestions", {})

        payload = self.post("/api/review/apply-ai-suggestions", {})

        self.assertEqual(payload["applied"], 0)

    def test_conflicts_are_reported_but_never_overridden_by_default(self):
        """Phase 9: never silently overwrite a photographer's own Keep/Reject."""
        best = str(self.images[0])
        self.post("/api/review/status", {"image_path": best, "status": "reject"})

        payload = self.post("/api/review/apply-ai-suggestions", {})

        self.assertEqual(payload["conflicts"], 1)
        self.assertEqual(payload["overridden"], 0)
        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        self.assertEqual(by_path[best]["review_status"], "reject")

    def test_include_decided_overrides_the_disagreeing_image(self):
        best = str(self.images[0])
        self.post("/api/review/status", {"image_path": best, "status": "reject"})

        payload = self.post("/api/review/apply-ai-suggestions", {"include_decided": True})

        self.assertEqual(payload["overridden"], 1)
        by_path = {i["image_path"]: i for i in payload["state"]["images"]}
        self.assertEqual(by_path[best]["review_status"], "keep")


class AgreementStatsEndpointTests(ReviewServerTestCase):
    """AI <-> user agreement, surfaced for evaluating the model over time -
    see ReviewSession.agreement_stats."""

    def test_state_reports_agreement_once_images_are_actually_decided(self):
        best = str(self.images[0])  # the AI's own top pick at keep_percent=33
        worst = str(self.images[-1])
        self.post("/api/review/status", {"image_path": best, "status": "keep"})  # agrees
        self.post("/api/review/status", {"image_path": worst, "status": "keep"})  # disagrees (AI: reject)

        payload = self.get("/api/review/state")["state"]

        self.assertEqual(payload["agreement"]["compared"], 2)
        self.assertEqual(payload["agreement"]["agree"], 1)
        self.assertEqual(payload["agreement"]["disagree"], 1)
        self.assertEqual(payload["agreement"]["ai_reject_user_keep"], 1)

    def test_agreement_is_null_percent_with_nothing_compared_yet(self):
        payload = self.get("/api/review/state")["state"]

        self.assertEqual(payload["agreement"]["compared"], 0)
        self.assertIsNone(payload["agreement"]["agree_percent"])


class KeepPercentEndpointTests(ReviewServerTestCase):
    def test_changing_the_percentage_changes_the_ai_suggestion_not_review_status(self):
        """keep_percent is the AI's own threshold - read-only metadata. It
        must never set anyone's review status by itself."""
        payload = self.post("/api/review/keep-percent", {"keep_percent": 100})

        images = payload["state"]["images"]
        ranked = [i for i in images if i["score"] is not None]
        self.assertTrue(all(i["ai_suggestion"] == "keep" for i in ranked))
        self.assertTrue(all(i["review_status"] == "neutral" for i in images), "nobody has reviewed anything")

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


class ClickThroughToFullSizeTests(ReviewServerTestCase):
    """Clicking a card's thumbnail opens the in-app Lightbox, which decodes
    the RAW's own full-size preview via /preview. Unlike the analysis
    report's own use of the same URL (still `Cache-Control: no-store`,
    inherited from AnnotationRequestHandler unchanged), the review server
    overrides `_serve_preview` to back it with a persistent on-disk cache
    (see thumbnails.review_preview) - the fix for the Lightbox getting
    slower the longer a session ran, traced to a full RAW decode + JPEG
    re-encode on every single request for images the reviewer had often
    only just looked at. Still confined to the reviewed folder by the same
    `source_roots` check as /thumb and /open-folder, and the photographer
    never leaves the page (no target="_blank" any more - see Lightbox in
    page.py for the markup-level checks on that)."""

    def test_the_preview_endpoint_serves_the_full_frame_not_the_square_thumb(self):
        url = "/preview?path=" + quote(str(self.images[0]), safe="")
        with urllib.request.urlopen(self.base + url) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            data = response.read()

        decoded = Image.open(BytesIO(data))
        self.assertEqual(decoded.size, (60, 40), "the fixture's real dimensions, not a cropped square")

    def test_the_preview_endpoint_is_confined_to_the_reviewed_folder(self):
        """The Lightbox must not become a new way to read the whole disk."""
        secret = self.root / "secret.jpg"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(secret)

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/preview?path=" + quote(str(secret), safe=""))
        self.assertEqual(ctx.exception.code, 403)

    def test_the_review_preview_is_cacheable_unlike_the_reports_own_use_of_it(self):
        """The whole point of the override: this /preview may be cached
        (content-addressed by resolved path, see review_preview), unlike the
        analysis report's, which must stay no-store."""
        url = "/preview?path=" + quote(str(self.images[0]), safe="")
        with urllib.request.urlopen(self.base + url) as response:
            self.assertIn("max-age", response.headers["Cache-Control"])

    def test_a_second_request_for_the_same_image_is_served_from_the_disk_cache(self):
        """Not just an HTTP header - load_source_image (the expensive rawpy/
        PIL path) must not run a second time for a request the on-disk cache
        already has an answer for."""
        url = "/preview?path=" + quote(str(self.images[0]), safe="")
        with urllib.request.urlopen(self.base + url):
            pass  # first request populates the cache

        with mock.patch("picklikeme.analyzer.contactsheets.load_source_image") as load:
            with urllib.request.urlopen(self.base + url) as response:
                self.assertEqual(response.status, 200)
            load.assert_not_called()

    def test_a_missing_file_is_a_clean_404_not_a_crash(self):
        missing = self.shoot / "gone.jpg"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/preview?path=" + quote(str(missing), safe=""))
        self.assertEqual(ctx.exception.code, 404)


class SaveJpegEndpointTests(ReviewServerTestCase):
    """The Lightbox's "Save JPEG" button - a download, not a preview: the
    response must carry Content-Disposition: attachment so the browser's own
    Save As / download handling takes over, and it must respect the same
    dataset confinement every other path-taking endpoint has."""

    def test_the_response_is_offered_as_a_download_not_rendered_inline(self):
        url = "/save-jpeg?path=" + quote(str(self.images[0]), safe="")
        with urllib.request.urlopen(self.base + url) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertIn(".jpg", response.headers["Content-Disposition"])
            data = response.read()
        self.assertGreater(len(data), 0)

    def test_it_is_confined_to_the_reviewed_folder(self):
        secret = self.root / "secret.jpg"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(secret)

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/save-jpeg?path=" + quote(str(secret), safe=""))
        self.assertEqual(ctx.exception.code, 403)

    def test_a_missing_file_is_a_clean_404(self):
        missing = self.shoot / "gone.jpg"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/save-jpeg?path=" + quote(str(missing), safe=""))
        self.assertEqual(ctx.exception.code, 404)


class EvaluationReportEndpointTests(ReviewServerTestCase):
    """The "Export Evaluation Report" button - see evaluation_report.py.
    Both formats are downloads, the same as /save-jpeg: the response must
    carry Content-Disposition: attachment, not render inline."""

    def test_the_html_report_is_offered_as_a_download(self):
        with urllib.request.urlopen(self.base + "/evaluation-report.html") as response:
            self.assertEqual(response.headers["Content-Type"], "text/html; charset=utf-8")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertIn(".html", response.headers["Content-Disposition"])
            body = response.read().decode("utf-8")
        self.assertIn(self.shoot.name, body)
        self.assertIn("Evaluation Report", body)

    def test_the_csv_report_is_offered_as_a_download(self):
        self.post("/api/review/status", {"image_path": str(self.images[0]), "status": "keep"})
        self.post("/api/review/status", {"image_path": str(self.images[-1]), "status": "keep"})

        with urllib.request.urlopen(self.base + "/evaluation-report.csv") as response:
            self.assertEqual(response.headers["Content-Type"], "text/csv; charset=utf-8")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertIn(".csv", response.headers["Content-Disposition"])
            body = response.read().decode("utf-8")

        rows = list(csv.reader(io.StringIO(body)))
        self.assertEqual(rows[0], ["file_name", "ai_decision", "user_decision", "ai_score"])
        self.assertEqual(len(rows) - 1, len(self.session.disagreements()))


class LightboxMarkupTests(unittest.TestCase):
    """Structural checks on the Lightbox module, with no server involved.

    Like PageMarkupTests, these exist because the suite has no JS engine (the
    actual interactive behaviour - zoom math, pan clamping, keep/reject/
    neutral then advance, drag-vs-click disambiguation - is verified
    separately by executing the real generated JS in Node/jsdom during
    development; that harness is not committed since this project's test
    stack has none). What runs here is what a plain source-pattern check
    actually can catch: the wiring between the gallery and the viewer, and
    that nothing needed by that wiring was left out of the markup.
    """

    def test_card_click_targets_are_wired_to_the_lightbox_not_a_link(self):
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertNotIn("target=\"_blank\"", js, "the viewer must not fall back to opening a new tab")
        self.assertIn("Lightbox.open(Number(el.dataset.index))", js)
        self.assertIn("class=\"thumb-link\" data-index=", js)

    def test_every_lightbox_control_the_script_looks_up_exists_in_the_markup(self):
        """The same class of bug the theme-button fix caught, applied to the
        whole new viewer: a null dereference in any of these aborts the script
        the same way, taking the entire page down with it - not just the
        viewer."""
        from picklikeme.review.page import build_js, build_page

        html = build_page()
        present = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
        wanted = set(re.findall(r"""q\(['"]#([A-Za-z0-9_-]+)['"]\)""", build_js()))

        missing = sorted(wanted - present)
        self.assertEqual(missing, [], f"the script queries element(s) the markup never defines: {missing}")

    def test_keep_reject_neutral_reuse_the_existing_setstatus_function(self):
        """Not three separate POST implementations - decideAndAdvance calls
        the same setStatus() the gallery cards use, so validation, PLM.state
        refresh and status reporting can never drift between the two
        surfaces."""
        from picklikeme.review.page import build_js

        self.assertIn("await setStatus(image.image_path, status, reason, reasonNote)", build_js())

    def test_the_dialog_element_is_used_for_native_escape_and_focus_handling(self):
        """Reuses the platform's own modal semantics (already established by
        the arrange confirmation dialog in this same page) rather than
        reimplementing focus trapping and Escape-to-close by hand."""
        from picklikeme.review.page import build_page

        self.assertIn('<dialog id="lightbox">', build_page())

    def test_escape_is_intercepted_to_share_the_fade_out_with_every_other_close(self):
        from picklikeme.review.page import build_js

        self.assertIn("addEventListener('cancel'", build_js())

    def test_keep_reject_neutral_have_keyboard_shortcuts(self):
        """5/K keep, 0/R reject, U neutral - 5/0 mirror a lot of RAW viewers'
        star-rating keys, K/R are the mnemonic pair, and U is the same
        "unflag/undecided" convention Lightroom's own U key uses."""
        from picklikeme.review.page import build_js

        js = build_js()
        handler = re.search(r"function onKeyDown\(e\)\{(.*?)\n  \}", js, re.S)
        self.assertIsNotNone(handler, "onKeyDown not found")
        body = handler.group(1)
        self.assertIn("key === '5' || key === 'k'", body)
        self.assertIn("decideAndAdvance('keep')", body)
        self.assertIn("key === '0' || key === 'r'", body)
        self.assertIn("decideAndAdvance('reject')", body)
        self.assertIn("key === 'u'", body)
        self.assertIn("decideAndAdvance('neutral')", body)

    def test_keyboard_shortcuts_do_not_hijack_the_reason_note_or_dropdown(self):
        """Typing "reject" as a free-text note, or using the reason
        dropdown's own letter/arrow handling, must not also fire a
        Keep/Reject/Neutral or navigate away from the image being
        annotated."""
        from picklikeme.review.page import build_js

        js = build_js()
        handler = re.search(r"function onKeyDown\(e\)\{(.*?)\n  \}", js, re.S)
        body = handler.group(1)
        self.assertIn("if(isTypingTarget(e.target)) return", body)

        guard = re.search(r"function isTypingTarget\(target\)\{(.*?)\n  \}", js, re.S)
        self.assertIsNotNone(guard, "isTypingTarget not found")
        self.assertIn("'SELECT'", guard.group(1))
        self.assertIn("'INPUT'", guard.group(1))

    def test_a_bounded_preload_cache_is_used_instead_of_relying_on_http_caching(self):
        """/preview is Cache-Control: no-store (shared with the analysis
        report's own use of it - see analyzer/server.py), so naive preloading
        via `new Image().src = ...` would refetch and redecode on every
        navigation regardless. The cache must be bounded (evicted), not just
        present, or a long session would leak a blob URL per image ever
        visited."""
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("URL.createObjectURL", js)
        self.assertIn("URL.revokeObjectURL", js)
        self.assertIn("evictFarFromCurrent", js)

    def test_plain_wheel_navigates_and_ctrl_wheel_zooms(self):
        """A plain wheel is the fast, default gesture for browsing (steps
        through images); zooming is the deliberate, less frequent action, so
        it only happens with Ctrl held. The navigate branch must short-
        circuit before any scale change, or the two gestures would fight
        over the same event."""
        from picklikeme.review.page import build_js

        js = build_js()
        wheel_body = re.search(r"function onWheel\(e\)\{(.*?)\n  \}", js, re.S)
        self.assertIsNotNone(wheel_body, "onWheel not found")
        body = wheel_body.group(1)
        self.assertIn("!e.ctrlKey", body)
        self.assertIn("next()", body)
        self.assertIn("prev()", body)
        # The navigate branch must return before reaching the zoom factor/setScale.
        self.assertLess(body.index("!e.ctrlKey"), body.index("setScale"))

    def test_overlay_chrome_stays_above_the_image_at_any_zoom(self):
        """A CSS transform doesn't resize .lb-img-wrap's own layout box, so a
        zoomed-in image can paint outside it - and img-wrap comes after the
        close button, nav buttons and bottom bar in the markup, so without an
        explicit z-index a zoomed image would paint over and hide them."""
        from picklikeme.review.page import CSS

        for selector in (".lb-close", ".lb-nav", ".lb-bottom"):
            rule = re.search(re.escape(selector) + r"[^{]*\{([^}]*)\}", CSS)
            self.assertIsNotNone(rule, f"no CSS rule for {selector}")
            self.assertIn("z-index", rule.group(1), f"{selector} has no z-index to stay above the zoomed image")

    def test_the_info_row_lives_in_the_bottom_bar_with_keep_reject_neutral(self):
        from picklikeme.review.page import CSS, build_page

        html = build_page()
        bottom = re.search(r'<div class="lb-bottom">(.*?)</div>\s*</div>\s*</dialog>', html, re.S)
        self.assertIsNotNone(bottom, "lb-bottom not found")
        body = bottom.group(1)
        self.assertIn('class="lb-info"', body)
        self.assertIn('id="lb-counter"', body)
        self.assertIn('id="lb-exp-value"', body)
        self.assertIn('id="lb-save-jpeg"', body)
        self.assertIn('id="lb-keep"', body)
        self.assertIn('id="lb-reject"', body)
        self.assertIn('id="lb-neutral"', body)
        # The info row's own text is black on its own light backing.
        info_rule = re.search(r"\.lb-info\{([^}]*)\}", CSS)
        self.assertIsNotNone(info_rule, "no .lb-info CSS rule")
        self.assertIn("color:#000", info_rule.group(1))

    def test_keep_reject_neutral_live_inside_the_same_white_box_centred_above_the_film_strip(self):
        from picklikeme.review.page import CSS, build_page

        html = build_page()
        bottom = re.search(r'<div class="lb-bottom">(.*?)</div>\s*</div>\s*</dialog>', html, re.S)
        self.assertIsNotNone(bottom, "lb-bottom not found")
        body = bottom.group(1)

        info = re.search(r'<div class="lb-info">(.*?)</div>\s*<div class="lb-film"', body, re.S)
        self.assertIsNotNone(info, "lb-info not found directly above the film strip")
        info_body = info.group(1)
        self.assertIn('id="lb-save-jpeg"', info_body)
        self.assertIn('id="lb-keep"', info_body)
        self.assertIn('id="lb-reject"', info_body)
        self.assertIn('id="lb-neutral"', info_body)
        self.assertIn('id="lb-status"', info_body)

        info_rule = re.search(r"\.lb-info\{([^}]*)\}", CSS)
        self.assertIsNotNone(info_rule, "no .lb-info CSS rule")
        self.assertIn("justify-content:center", info_rule.group(1))

    def test_exposure_is_a_css_filter_never_a_decision_or_a_network_call(self):
        """The whole point of "display only": adjustExposure()/applyExposure()
        must never call setStatus(), api(), or fetch - if it ever needed to,
        that would mean exposure had started writing something down
        somewhere."""
        from picklikeme.review.page import build_js

        js = build_js()
        exposure_block = re.search(
            r"function applyExposure\(\)\{(.*?)\n  \}\n\n  function adjustExposure",
            js,
            re.S,
        )
        self.assertIsNotNone(exposure_block, "applyExposure not found")
        body = exposure_block.group(1)
        self.assertIn("style.filter", body)
        self.assertIn("brightness(", body)
        for forbidden in ("fetch(", "api(", "setStatus(", "PLM.state ="):
            self.assertNotIn(forbidden, body)

    def test_exposure_is_clamped_to_plus_minus_three_ev(self):
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("EXPOSURE_MIN_STEPS = -9", js)
        self.assertIn("EXPOSURE_MAX_STEPS = 9", js)

    def test_save_jpeg_triggers_a_real_download_not_a_fetch(self):
        """A same-origin navigation, not fetch()+blob: the server answers with
        Content-Disposition: attachment (see review/server.py), so the
        browser's own Save As / download handling does the actual saving -
        this only has to point it at the right URL."""
        from picklikeme.review.page import build_js

        js = build_js()
        save_block = re.search(r"function saveJpeg\(\)\{(.*?)\n  \}", js, re.S)
        self.assertIsNotNone(save_block, "saveJpeg not found")
        body = save_block.group(1)
        self.assertIn("'save-jpeg?path=' + encodeURIComponent(image.image_path)", body)
        self.assertIn("a.click()", body)
        self.assertNotIn("fetch(", body)

    def test_save_jpeg_button_is_wired_and_present_in_the_markup(self):
        from picklikeme.review.page import build_js, build_page

        self.assertIn('id="lb-save-jpeg"', build_page())
        self.assertIn("addEventListener('click', saveJpeg)", build_js())


class ZoomWindowCleanupTests(unittest.TestCase):
    """The Lightbox's floating top-left status pill is gone - the Keep/
    Reject/Neutral buttons' own `.on` highlight and the reason/status text
    already say the same thing, so showing it a second time was pure
    clutter. The Light/Dark toggle moved out of the toolbar into the side
    panel, reclaiming the toolbar row it used to occupy alone."""

    def test_the_floating_status_badge_is_gone(self):
        from picklikeme.review.page import CSS, build_js, build_page

        self.assertNotIn('id="lb-badge"', build_page())
        self.assertNotIn(".lb-badge", CSS)
        self.assertNotIn("updateBadge", build_js())

    def test_the_theme_toggle_lives_in_the_side_panel_not_the_toolbar(self):
        from picklikeme.review.page import build_page

        html = build_page()
        toolbar = re.search(r'<div class="toolbar">(.*?)\n  </div>', html, re.S)
        panel = re.search(r'<aside class="panel" id="side-panel">(.*?)</aside>', html, re.S)
        self.assertIsNotNone(toolbar)
        self.assertIsNotNone(panel)
        self.assertNotIn('id="theme"', toolbar.group(1))
        self.assertIn('id="theme"', panel.group(1))


class MainWindowCleanupTests(unittest.TestCase):
    """The opened folder is shown exactly once (the toolbar's #folder chip),
    and a gallery card no longer repeats its own Keep/Reject/Neutral status
    as text - the card's colored border (.card.keep/.reject/.neutral)
    already carries it, freeing room for a slightly larger thumbnail."""

    def test_switching_folders_does_not_repeat_the_path_in_the_status_line(self):
        from picklikeme.review.page import build_js

        js = build_js()
        switch_block = re.search(r"async function switchFolder\(\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(switch_block, "switchFolder not found")
        body = switch_block.group(1)
        self.assertNotIn("j.state.input_folder", body, "the folder path must not be repeated in say()")
        self.assertIn("say('Folder opened.'", body)

    def test_the_card_has_no_text_status_badge(self):
        from picklikeme.review.page import build_js

        card_fn = re.search(r"function card\(image, index\)\{(.*?)\n\}", build_js(), re.S)
        self.assertIsNotNone(card_fn)
        body = card_fn.group(1)
        self.assertNotIn('class="badge ', body)
        self.assertIn("title=", body, "the status should still be reachable on hover")

    def test_the_badge_css_class_is_gone_entirely(self):
        from picklikeme.review.page import CSS

        self.assertNotIn(".badge{", CSS)
        self.assertNotIn(".badge.keep", CSS)

    def test_the_colored_border_is_still_how_status_is_shown(self):
        from picklikeme.review.page import CSS

        self.assertIn(".card.keep{", CSS)
        self.assertIn(".card.reject{", CSS)
        self.assertIn(".card.neutral{", CSS)

    def test_thumbnails_get_slightly_larger_once_the_badge_text_is_gone(self):
        from picklikeme.review.page import CSS

        self.assertIn("minmax(240px,1fr)", CSS)


class LoadingFeedbackMarkupTests(unittest.TestCase):
    """Folder loading feedback - a spinner overlay for the initial load,
    switching folders, and relocating one, all sharing one setLoading()
    helper rather than a bespoke indicator per action."""

    def test_the_overlay_exists_and_starts_hidden(self):
        from picklikeme.review.page import build_page

        html = build_page()
        overlay = re.search(r'<div class="loading-overlay" id="loading-overlay"([^>]*)>', html)
        self.assertIsNotNone(overlay, "loading-overlay not found")
        self.assertIn("display:none", overlay.group(1))

    def test_set_loading_toggles_the_overlay(self):
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"function setLoading\(active, message\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "setLoading not found")
        body = fn.group(1)
        self.assertIn("'flex' : 'none'", body)

    def test_every_folder_replacing_call_shows_the_overlay(self):
        from picklikeme.review.page import build_js

        js = build_js()
        for fn_name in ("switchFolder", "relocateFolder", "boot"):
            block = re.search(r"async function " + fn_name + r"\(\)\{(.*?)\n\}", js, re.S)
            self.assertIsNotNone(block, f"{fn_name} not found")
            self.assertIn("setLoading(true", block.group(1), f"{fn_name} never shows the loading overlay")
            self.assertIn("setLoading(false)", block.group(1), f"{fn_name} never hides the loading overlay")


class ArrangeProgressFeedbackTests(unittest.TestCase):
    """Arrange is a real file-moving operation for potentially thousands of
    files - the dialog must show it is working and refuse a second click
    (Arrange or Cancel) while one move is already in flight."""

    def test_the_dialog_has_a_progress_indicator_that_starts_hidden(self):
        from picklikeme.review.page import build_page

        html = build_page()
        progress = re.search(r'<div class="dlg-progress" id="dlg-progress"([^>]*)>', html)
        self.assertIsNotNone(progress, "dlg-progress not found")
        self.assertIn("display:none", progress.group(1))

    def test_do_arrange_disables_both_dialog_buttons_and_shows_the_spinner(self):
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"async function doArrange\(\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "doArrange not found")
        body = fn.group(1)
        self.assertIn("q('#dlg-cancel').disabled = true", body)
        self.assertIn("q('#dlg-go').disabled = true", body)
        self.assertIn("q('#dlg-progress').style.display = 'flex'", body)
        # And both must be restored afterwards, success or failure.
        self.assertIn("q('#dlg-cancel').disabled = false", body)
        self.assertIn("q('#dlg-go').disabled = false", body)

    def test_do_arrange_sets_a_wait_cursor_while_busy(self):
        from picklikeme.review.page import build_js

        fn = re.search(r"async function doArrange\(\)\{(.*?)\n\}", build_js(), re.S)
        body = fn.group(1)
        self.assertIn("document.body.style.cursor = 'wait'", body)
        self.assertIn("document.body.style.cursor = ''", body)


class ReasonFieldTests(unittest.TestCase):
    """The override-reason dropdown next to Keep/Reject/Neutral - structural
    checks only, same rationale as LightboxMarkupTests."""

    def test_both_reasons_are_offered_as_options(self):
        from picklikeme.review.page import build_page

        html = build_page()
        self.assertIn('id="lb-reason"', html)
        self.assertIn('value="eyes_not_seen"', html)
        self.assertIn("Eyes not seen", html)
        self.assertIn('value="clear_eyes_seen"', html)
        self.assertIn("Clear Eyes Seen", html)
        # A reason is optional - blank must be a real, selectable option, not
        # just whatever <select> defaults to.
        self.assertIn('value=""', html)

    def test_deciding_from_the_lightbox_sends_whatever_the_dropdown_currently_shows(self):
        """The dropdown, not the last-saved reason - so a reason picked right
        before clicking Keep/Reject is the one that gets sent, and clicking
        with nothing picked sends none."""
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("await setStatus(image.image_path, status, reason, reasonNote)", js)
        self.assertIn("const reason = q('#lb-reason').value || null", js)

    def test_changing_the_reason_updates_it_without_re_advancing(self):
        """onReasonChange must call setStatus() directly rather than
        decideAndAdvance(), which would also move to the next image - only a
        real Keep/Reject click should ever advance."""
        from picklikeme.review.page import build_js

        js = build_js()
        handler = re.search(r"async function onReasonChange\(\)\{(.*?)\n  \}", js, re.S)
        self.assertIsNotNone(handler, "onReasonChange not found")
        body = handler.group(1)
        self.assertIn("if(!image || image.review_status === 'neutral') return", body)
        self.assertIn("setStatus(image.image_path, image.review_status, reason, null)", body)
        self.assertNotIn("decideAndAdvance(", body, "must not advance to the next image")

    def test_the_dropdown_resets_to_the_current_image_s_own_reason_on_every_navigation(self):
        """Otherwise a reason picked for one image (even if never saved,
        because Keep/Reject was never clicked) would leak onto the next one
        just by browsing past it."""
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("function updateReasonSelect(image)", js)
        self.assertIn("updateReasonSelect(image);", js, "must run on every renderAll(), not just on a decision")
        select_fn = re.search(r"function updateReasonSelect\(image\)\{(.*?)\n  \}", js, re.S)
        self.assertIsNotNone(select_fn)
        self.assertIn("(image && image.reason) || ''", select_fn.group(1))

    def test_a_reason_is_meaningless_and_disabled_for_a_neutral_image(self):
        from picklikeme.review.page import build_js

        js = build_js()
        select_fn = re.search(r"function updateReasonSelect\(image\)\{(.*?)\n  \}", js, re.S)
        self.assertIsNotNone(select_fn)
        self.assertIn("image.review_status === 'neutral'", select_fn.group(1))

    def test_reason_is_never_used_by_the_gallery_cards_only_the_lightbox(self):
        """The grid has no reason control - a card's status button click must
        not silently invent one."""
        from picklikeme.review.page import build_js

        js = build_js()
        card_click_binding = re.search(
            r"b\.addEventListener\('click', \(\) => setStatus\(b\.dataset\.path, b\.dataset\.status\)\);", js
        )
        self.assertIsNotNone(card_click_binding, "gallery card setStatus() call changed shape unexpectedly")


class GalleryFilterTests(unittest.TestCase):
    """Structural checks for the Keep/Reject/Neutral/All filter, with no
    server involved - same rationale as LightboxMarkupTests: the interactive
    part is covered by a Node/jsdom harness during development."""

    def test_filter_buttons_exist_for_all_keep_reject_and_neutral(self):
        from picklikeme.review.page import build_page

        html = build_page()
        self.assertIn('data-filter="all"', html)
        self.assertIn('data-filter="keep"', html)
        self.assertIn('data-filter="reject"', html)
        self.assertIn('data-filter="neutral"', html)

    def test_the_filter_values_match_review_status_directly(self):
        """No separate classification function any more - a card's CSS class
        IS image.review_status, so a filter value and a review_status value
        are the exact same string; there is nothing left that could disagree."""
        from picklikeme.review.page import build_js

        js = build_js()
        filter_images = re.search(r"function filterImages\(images\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(filter_images)
        body = filter_images.group(1)
        self.assertIn("if(PLM.filter === 'all') return images;", body)
        self.assertIn("i.review_status === PLM.filter", body)

    def test_ai_keep_ai_reject_and_differences_filters_exist(self):
        from picklikeme.review.page import build_page, build_js

        html = build_page()
        self.assertIn('data-filter="ai_keep"', html)
        self.assertIn('data-filter="ai_reject"', html)
        self.assertIn('data-filter="differences"', html)

        js = build_js()
        filter_images = re.search(r"function filterImages\(images\)\{(.*?)\n\}", js, re.S)
        body = filter_images.group(1)
        self.assertIn("i.ai_suggestion === 'keep'", body)
        self.assertIn("i.ai_suggestion === 'reject'", body)
        # A difference requires the AI to have an opinion AND the photographer
        # to have actually decided (Neutral is "no opinion", not disagreement).
        self.assertIn("i.ai_suggestion != null && i.review_status !== 'neutral' && i.review_status !== i.ai_suggestion", body)

    def test_filter_buttons_are_wired_to_set_filter(self):
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("querySelectorAll('.filter')", js)
        self.assertIn("setFilter(b.dataset.filter)", js)

    def test_the_gallery_and_the_lightbox_share_one_filtered_list(self):
        """visibleImages() is what both render() and the Lightbox's own
        navigation read - so the viewer can never step outside what is
        currently on screen under the active filter."""
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("function visibleImages()", js)
        self.assertIn("function images(){ return visibleImages(); }", js)


class DetectedCategoryMarkupTests(unittest.TestCase):
    """The subject-category chip and its dynamic panel filters - structured
    metadata, not a judgement (see bird_crop.DETECTION_CATEGORIES)."""

    def test_the_category_section_exists_and_starts_hidden(self):
        """Data-driven, not a fixed list (Phase 3): with no categories
        present in the folder at all (the common case today - most folders
        have never been preprocessed), the section shows nothing."""
        from picklikeme.review.page import build_page

        html = build_page()
        section = re.search(r'<div class="panel-section" id="category-section"([^>]*)>', html)
        self.assertIsNotNone(section, "category-section not found")
        self.assertIn("display:none", section.group(1))
        self.assertIn('id="category-filters"', html)

    def test_filter_buttons_are_generated_per_category_actually_present(self):
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"function renderCategoryFilters\(images\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "renderCategoryFilters not found")
        body = fn.group(1)
        self.assertIn("images.map(i => i.detected_category)", body)
        self.assertIn("if(!present.length){ section.style.display = 'none'; return; }", body)

    def test_a_category_filter_matches_on_detected_category(self):
        from picklikeme.review.page import build_js

        js = build_js()
        filter_images = re.search(r"function filterImages\(images\)\{(.*?)\n\}", js, re.S)
        body = filter_images.group(1)
        self.assertIn("PLM.filter.indexOf('category:') === 0", body)
        self.assertIn("i.detected_category === category", body)

    def test_every_card_shows_a_category_chip_when_one_is_recorded(self):
        from picklikeme.review.page import build_js

        js = build_js()
        card_fn = re.search(r"function card\(image, index\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(card_fn)
        body = card_fn.group(1)
        self.assertIn("image.detected_category", body)
        self.assertIn("category-chip", body)

    def test_the_category_chip_is_visually_distinct_from_the_ai_chip(self):
        from picklikeme.review.page import CSS

        self.assertIn(".category-chip{", CSS)
        ai_rule = re.search(r"\.ai-chip\{([^}]*)\}", CSS)
        category_rule = re.search(r"\.category-chip\{([^}]*)\}", CSS)
        self.assertIsNotNone(ai_rule)
        self.assertIsNotNone(category_rule)
        self.assertNotEqual(ai_rule.group(1), category_rule.group(1))


class BulkActionsMarkupTests(unittest.TestCase):
    """Structural checks for the multi-select bulk actions bar, with no
    server involved - same rationale as GalleryFilterTests/LightboxMarkupTests."""

    def test_every_card_gets_a_tri_state_status_row_not_a_toggle(self):
        """Keep/Reject/Neutral are three explicit buttons, each setting an
        exact target status - no hidden "click again to undo" gesture."""
        from picklikeme.review.page import build_js

        js = build_js()
        card_fn = re.search(r"function card\(image, index\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(card_fn)
        body = card_fn.group(1)
        self.assertIn('data-status="keep"', body)
        self.assertIn('data-status="reject"', body)
        self.assertIn('data-status="neutral"', body)

    def test_every_card_gets_a_pick_checkbox(self):
        from picklikeme.review.page import build_js

        js = build_js()
        card_fn = re.search(r"function card\(image, index\)\{(.*?)\n\}", js, re.S)
        self.assertIn("data-pick=", card_fn.group(1))

    def test_the_checkbox_is_not_inside_the_thumb_link(self):
        """A click on the checkbox must never also open the lightbox - see
        the pick/thumb-link sibling comment in card(). Checked structurally:
        the returned markup concatenates pick as a sibling BEFORE visual
        (which contains the thumb-link), not inside visual's own string."""
        from picklikeme.review.page import build_js

        js = build_js()
        card_fn = re.search(r"function card\(image, index\)\{(.*?)\n\}", js, re.S)
        body = card_fn.group(1)
        self.assertIn("+ pick + visual", body)

    def test_the_bulk_bar_is_entirely_absent_from_the_layout_by_default(self):
        """Phase 2: hidden, not merely disabled - there must be no dead
        chrome sitting on screen when nothing is picked."""
        from picklikeme.review.page import build_page

        html = build_page()
        bar = re.search(r'<div class="bulkbar" id="bulk-bar"([^>]*)>', html)
        self.assertIsNotNone(bar, "bulk-bar not found")
        self.assertIn("display:none", bar.group(1))
        for control in ("bulk-count", "bulk-keep", "bulk-reject", "bulk-neutral", "bulk-clear-sel"):
            self.assertIn(f'id="{control}"', html)

    def test_the_dismiss_button_was_removed(self):
        """Clear Selection is sufficient on its own (Phase 8 of this round) -
        a second, similar-but-different button next to it was just clutter."""
        from picklikeme.review.page import build_page

        self.assertNotIn('id="bulk-dismiss"', build_page())

    def test_the_bar_is_shown_or_hidden_by_updatebulkbar_from_the_picked_count(self):
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"function updateBulkBar\(\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "updateBulkBar not found")
        body = fn.group(1)
        self.assertIn("n === 0", body)
        self.assertIn("bar.style.display = 'none'", body)

    def test_clear_selection_empties_the_picked_set(self):
        from picklikeme.review.page import build_js

        js = build_js()
        clear_fn = re.search(r"function clearPicked\(\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(clear_fn, "clearPicked not found")
        self.assertIn("PLM.picked.clear()", clear_fn.group(1))

    def test_the_confirmation_dialog_exists_and_is_wired_before_any_request(self):
        """The whole point of the feature: a bulk action must never reach the
        server before the photographer has confirmed it in this dialog - the
        same generic dialog Apply AI Suggestions also reuses."""
        from picklikeme.review.page import build_js, build_page

        html = build_page()
        self.assertIn('<dialog id="confirm-dlg">', html)

        js = build_js()
        self.assertIn("function askConfirm(title, body, action)", js)
        self.assertIn("q('#confirm-dlg').showModal()", js)
        confirm_fn = re.search(r"function confirmBulkStatus\(status\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(confirm_fn, "confirmBulkStatus not found")
        self.assertNotIn("api(", confirm_fn.group(1), "confirmBulkStatus must only stage the dialog, never post")
        self.assertIn("async function runBulkStatus(status)", js)
        self.assertIn("api/review/bulk-status", js)

    def test_the_three_bulk_actions_map_to_keep_reject_and_neutral(self):
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("confirmBulkStatus('keep')", js)
        self.assertIn("confirmBulkStatus('reject')", js)
        self.assertIn("confirmBulkStatus('neutral')", js)

    def test_apply_ai_suggestions_only_confirms_before_overriding_manual_work(self):
        """Neutral images are updated immediately - nothing manual is at risk
        there. Only overriding an already-decided image goes through the
        shared confirmation dialog (Phase 5: no second, near-duplicate
        dialog for this action; Phase 9: never silently overwrite a
        photographer's own Keep/Reject)."""
        from picklikeme.review.page import build_js

        js = build_js()
        apply_fn = re.search(r"async function applyAiSuggestions\(\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(apply_fn, "applyAiSuggestions not found")
        body = apply_fn.group(1)
        self.assertIn("include_decided: false", body)
        self.assertIn("if(j.conflicts > 0)", body)
        self.assertIn("askConfirm(", body)

        decided_fn = re.search(r"async function applyAiSuggestionsToDecided\(\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(decided_fn, "applyAiSuggestionsToDecided not found")
        self.assertIn("include_decided: true", decided_fn.group(1))
        self.assertNotIn("showModal", decided_fn.group(1), "must go through askConfirm, not open a dialog directly")


class SelectAllVisibleMarkupTests(unittest.TestCase):
    """The primary bulk-selection command (Phase 1 of this round): every
    image currently on screen, after both the active filter and sort."""

    def test_the_button_exists_and_is_wired(self):
        from picklikeme.review.page import build_js, build_page

        self.assertIn('id="select-all-visible"', build_page())
        js = build_js()
        self.assertIn("q('#select-all-visible').addEventListener('click', selectAllVisible)", js)

    def test_it_picks_exactly_visibleimages_never_the_unfiltered_full_list(self):
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"function selectAllVisible\(\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "selectAllVisible not found")
        self.assertIn("visibleImages()", fn.group(1))
        self.assertNotIn("s.images", fn.group(1), "must not bypass the active filter/sort")

    def test_the_word_all_is_kept_in_the_command_s_name(self):
        """Explicit per the request: the label must keep saying "All"."""
        from picklikeme.review.page import build_page

        self.assertIn(">Select All Visible<", build_page())


class SortingMarkupTests(unittest.TestCase):
    """Sort by file name, capture date, or AI score, each ascending or
    descending - a missing value (no date, no score) always sorts last."""

    def test_the_three_sort_keys_are_offered(self):
        from picklikeme.review.page import build_page

        html = build_page()
        self.assertIn('id="sort-key"', html)
        for key in ("score", "name", "date"):
            self.assertIn(f'value="{key}"', html)

    def test_default_sort_matches_the_server_s_own_default_ordering(self):
        """Preserves the existing default (best AI score first) rather than
        surprising anyone who never touches the new sort controls at all."""
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("sort: {key: 'score', dir: 'desc'}", js)

    def test_a_missing_value_always_sorts_last_regardless_of_direction(self):
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"function sortImages\(images\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "sortImages not found")
        body = fn.group(1)
        self.assertIn("if(missingA) return 1;", body)
        self.assertIn("if(missingB) return -1;", body)

    def test_toggling_direction_and_changing_key_both_trigger_a_rerender(self):
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("function setSortKey(key){\n  PLM.sort.key = key;\n  render();\n}", js)
        self.assertIn("function toggleSortDir(){", js)
        self.assertIn("q('#sort-key').addEventListener('change', e => setSortKey(e.target.value))", js)
        self.assertIn("q('#sort-dir').addEventListener('click', toggleSortDir)", js)


class FiltersPanelMarkupTests(unittest.TestCase):
    """Filters, sorting and view options live in a dedicated, collapsible
    side panel (Phase 7 of this round) rather than crowding the toolbar."""

    def test_the_panel_exists_and_holds_the_filter_buttons(self):
        from picklikeme.review.page import build_page

        html = build_page()
        panel = re.search(r'<aside class="panel" id="side-panel">(.*?)</aside>', html, re.S)
        self.assertIsNotNone(panel, "side-panel not found")
        body = panel.group(1)
        self.assertIn('data-filter="all"', body)
        self.assertIn('id="boxes"', body)
        self.assertIn('id="sort-key"', body)
        self.assertIn('id="select-all-visible"', body)

    def test_the_toolbar_itself_no_longer_holds_the_view_group(self):
        """Decluttered per Phase 7 - View moved out entirely, not duplicated."""
        from picklikeme.review.page import build_page

        html = build_page()
        toolbar = re.search(r'<div class="toolbar">(.*?)</div>\s*<div class="bulkbar"', html, re.S)
        self.assertIsNotNone(toolbar, "toolbar not found")
        self.assertNotIn('data-filter=', toolbar.group(1))

    def test_the_panel_is_collapsible_and_its_state_persists(self):
        from picklikeme.review.page import build_js, build_page

        self.assertIn('id="panel-toggle"', build_page())
        js = build_js()
        self.assertIn("function togglePanel(){", js)
        self.assertIn("localStorage.setItem('plm-panel-open'", js)
        self.assertIn("q('#panel-toggle').addEventListener('click', togglePanel)", js)


class AgreementStatsMarkupTests(unittest.TestCase):
    """The AI <-> user agreement readout in the panel - purely
    informational, built from session.py's agreement_stats()."""

    def test_the_section_exists_and_starts_hidden(self):
        from picklikeme.review.page import build_page

        html = build_page()
        section = re.search(r'<div class="panel-section" id="agreement-section"([^>]*)>', html)
        self.assertIsNotNone(section, "agreement-section not found")
        self.assertIn("display:none", section.group(1))

    def test_it_is_hidden_again_whenever_nothing_has_been_compared_yet(self):
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"function renderAgreementStats\(agreement\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "renderAgreementStats not found")
        self.assertIn("!agreement.compared", fn.group(1))
        self.assertIn("section.style.display = 'none'", fn.group(1))


class EvaluationReportMarkupTests(unittest.TestCase):
    """The "Export Evaluation Report" buttons - live in the same panel
    section as AI Agreement, since a report only means something once there
    is something to compare."""

    def test_both_export_buttons_exist_in_the_agreement_section(self):
        from picklikeme.review.page import build_page

        html = build_page()
        section = re.search(r'<div class="panel-section" id="agreement-section".*?</div>\s*</div>', html, re.S)
        self.assertIsNotNone(section, "agreement-section not found")
        self.assertIn('id="export-report"', section.group(0))
        self.assertIn('id="export-report-csv"', section.group(0))

    def test_both_buttons_are_wired_to_the_download_helper(self):
        from picklikeme.review.page import build_js

        js = build_js()
        self.assertIn("function exportEvaluationReport(fmt){", js)
        self.assertIn("q('#export-report').addEventListener('click', () => exportEvaluationReport('html'))", js)
        self.assertIn("q('#export-report-csv').addEventListener('click', () => exportEvaluationReport('csv'))", js)

    def test_the_download_is_a_real_navigation_not_a_fetch(self):
        """Same pattern as saveJpeg: an <a download> click, so the browser's
        own Content-Disposition handling takes over rather than JS trying to
        save the response itself."""
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"function exportEvaluationReport\(fmt\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "exportEvaluationReport not found")
        self.assertIn("a.href = 'evaluation-report.' + fmt", fn.group(1))
        self.assertIn("a.download = ''", fn.group(1))


class RelocateFolderMarkupTests(unittest.TestCase):
    """The "folder could not be found" recovery dialog - see
    ReviewSession.relocate_folder for the backend side of this."""

    def test_the_dialog_exists_with_the_requested_wording(self):
        from picklikeme.review.page import build_page

        html = build_page()
        self.assertIn('<dialog id="relocate-dlg">', html)
        self.assertIn("could not be found", html)
        self.assertIn("select its new location", html)
        self.assertIn('id="relocate-go"', html)
        self.assertIn('id="relocate-later"', html)

    def test_it_auto_shows_when_the_folder_is_missing_and_not_yet_dismissed(self):
        from picklikeme.review.page import build_js

        js = build_js()
        render_fn = re.search(r"^function render\(\)\{(.*?)\n\}", js, re.S | re.M)
        self.assertIsNotNone(render_fn, "render() not found")
        body = render_fn.group(1)
        self.assertIn("s.folder_missing && !PLM.relocateDismissed", body)
        self.assertIn("q('#relocate-dlg').showModal()", body)

    def test_relocating_reuses_the_native_folder_picker_flow(self):
        from picklikeme.review.page import build_js

        js = build_js()
        fn = re.search(r"async function relocateFolder\(\)\{(.*?)\n\}", js, re.S)
        self.assertIsNotNone(fn, "relocateFolder not found")
        self.assertIn("api/review/relocate-folder", fn.group(1))


class ArrangeEndpointTests(ReviewServerTestCase):
    def test_a_dry_run_reports_the_plan_without_moving_anything(self):
        self.post("/api/review/status", {"image_path": str(self.images[0]), "status": "keep"})
        self.post("/api/review/status", {"image_path": str(self.images[1]), "status": "keep"})
        for image in self.images[2:]:
            self.post("/api/review/status", {"image_path": str(image), "status": "reject"})

        payload = self.post("/api/review/arrange", {"dry_run": True})

        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["result"]["selected"], 2)
        self.assertEqual(payload["result"]["rejected"], 4)
        self.assertIn(SELECTED_DIRNAME, payload["result"]["selected_dir"])
        for path in self.images:
            self.assertTrue(path.exists(), "a dry run must not move a file")

    def test_neutral_images_are_never_moved_no_matter_how_the_ai_ranked_them(self):
        """Arrange reads review_status alone - an all-Neutral gallery (the
        default, before anyone reviews anything) must move nothing."""
        self.post("/api/review/keep-percent", {"keep_percent": 100})

        payload = self.post("/api/review/arrange", {"dry_run": False})

        self.assertEqual(payload["result"]["moved"], 0)
        for path in self.images + [self.extra[0]]:
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, self.shoot)

    def test_arranging_moves_the_files_and_returns_fresh_state(self):
        for image in self.images[:2]:
            self.post("/api/review/status", {"image_path": str(image), "status": "keep"})
        for image in self.images[2:]:
            self.post("/api/review/status", {"image_path": str(image), "status": "reject"})

        payload = self.post("/api/review/arrange", {"dry_run": False})

        self.assertEqual(payload["result"]["moved"], 6)
        self.assertEqual(len(list((self.shoot / SELECTED_DIRNAME).glob("*.jpg"))), 2)
        self.assertEqual(len(list((self.shoot / REJECTED_DIRNAME).glob("*.jpg"))), 4)
        # The gallery now describes the new locations.
        self.assertEqual(payload["state"]["counts"]["missing_file"], 0)

    def test_neutral_undecided_images_are_left_in_place_by_arranging(self):
        for image in self.images:
            self.post("/api/review/status", {"image_path": str(image), "status": "reject"})

        self.post("/api/review/arrange", {"dry_run": False})

        self.assertTrue(self.extra[0].exists())
        self.assertEqual(self.extra[0].parent, self.shoot)

    def test_a_keep_on_the_ai_s_own_worst_pick_is_honoured_by_arranging(self):
        worst = str(self.images[-1])
        self.post("/api/review/status", {"image_path": worst, "status": "keep"})

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


class OpenFolderEndpointTests(ReviewServerTestCase):
    """Switching the whole review to a different folder - the main way a
    folder that was never ranked at all gets reviewed."""

    def test_opening_a_folder_by_path_skips_the_native_dialog(self):
        other = self.root / "unranked_shoot"
        other.mkdir()
        make_real_images([other / "IMG_0001.jpg"])

        with mock.patch("picklikeme.review.server.choose_folder") as dialog:
            payload = self.post("/api/review/open-folder", {"path": str(other)})

        dialog.assert_not_called()
        self.assertTrue(payload["ok"])
        self.assertEqual(Path(payload["state"]["input_folder"]).resolve(), other.resolve())
        self.assertEqual(len(payload["state"]["images"]), 1)
        self.assertEqual(payload["state"]["images"][0]["review_status"], "neutral")
        self.assertIn("recovered", payload)

    def test_omitting_the_path_shows_the_native_dialog(self):
        original = self.session.input_folder
        other = self.root / "unranked_shoot"
        other.mkdir()
        make_real_images([other / "IMG_0001.jpg"])

        with mock.patch("picklikeme.review.server.choose_folder", return_value=other) as dialog:
            payload = self.post("/api/review/open-folder", {})

        dialog.assert_called_once()
        self.assertEqual(dialog.call_args.kwargs["initial_dir"], original)
        self.assertEqual(Path(payload["state"]["input_folder"]).resolve(), other.resolve())

    def test_cancelling_the_dialog_leaves_the_current_folder_untouched(self):
        with mock.patch("picklikeme.review.server.choose_folder", return_value=None):
            payload = self.post("/api/review/open-folder", {})

        self.assertTrue(payload["cancelled"])
        self.assertEqual(Path(payload["state"]["input_folder"]).resolve(), self.shoot.resolve())

    def test_a_path_that_is_not_a_folder_is_refused(self):
        missing = self.root / "does_not_exist"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/open-folder", {"path": str(missing)})
        self.assertEqual(ctx.exception.code, 404)

    def test_a_non_string_path_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/open-folder", {"path": 123})
        self.assertEqual(ctx.exception.code, 400)

    def test_after_switching_the_new_folder_s_own_images_become_servable(self):
        """The dataset confinement `/thumb`, `/source` etc. all enforce must
        move with the session - otherwise every endpoint would keep refusing
        the very folder the photographer just opened."""
        other = self.root / "unranked_shoot"
        other.mkdir()
        photo = other / "IMG_0001.jpg"
        make_real_images([photo])
        self.post("/api/review/open-folder", {"path": str(other)})

        with urllib.request.urlopen(self.base + "/thumb?path=" + quote(str(photo), safe="")) as response:
            self.assertEqual(response.status, 200)

    def test_after_switching_the_old_folder_s_images_are_no_longer_servable(self):
        other = self.root / "unranked_shoot"
        other.mkdir()
        make_real_images([other / "IMG_0001.jpg"])
        self.post("/api/review/open-folder", {"path": str(other)})

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/thumb?path=" + quote(str(self.images[0]), safe=""))
        self.assertEqual(ctx.exception.code, 403)


class RelocateFolderEndpointTests(ReviewServerTestCase):
    """The folder this session was reviewing can no longer be found - moved,
    renamed, or a changed drive letter - and the photographer has pointed at
    where it went. Shares its path/dialog resolution with /open-folder, but
    every stored path is repointed automatically rather than starting over."""

    def test_relocating_by_path_repoints_the_ranking_and_reloads(self):
        moved_to = self.root / "moved_shoot"
        shutil.move(str(self.shoot), str(moved_to))

        with mock.patch("picklikeme.review.server.choose_folder") as dialog:
            payload = self.post("/api/review/relocate-folder", {"path": str(moved_to)})

        dialog.assert_not_called()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["relocated"], 6)
        self.assertEqual(Path(payload["state"]["input_folder"]).resolve(), moved_to.resolve())
        self.assertFalse(payload["state"]["folder_missing"])
        self.assertEqual(payload["state"]["counts"]["missing_file"], 0)

    def test_omitting_the_path_shows_the_native_dialog_seeded_at_the_old_folder(self):
        original = self.session.input_folder
        moved_to = self.root / "moved_shoot"
        shutil.move(str(self.shoot), str(moved_to))

        with mock.patch("picklikeme.review.server.choose_folder", return_value=moved_to) as dialog:
            payload = self.post("/api/review/relocate-folder", {})

        dialog.assert_called_once()
        self.assertEqual(dialog.call_args.kwargs["initial_dir"], original)
        self.assertEqual(Path(payload["state"]["input_folder"]).resolve(), moved_to.resolve())

    def test_cancelling_leaves_the_missing_folder_state_untouched(self):
        shutil.rmtree(self.shoot)

        with mock.patch("picklikeme.review.server.choose_folder", return_value=None):
            payload = self.post("/api/review/relocate-folder", {})

        self.assertTrue(payload["cancelled"])
        self.assertTrue(payload["state"]["folder_missing"])

    def test_after_relocating_the_new_folder_s_images_become_servable(self):
        moved_to = self.root / "moved_shoot"
        shutil.move(str(self.shoot), str(moved_to))
        moved_image = moved_to / Path(self.images[0]).name

        self.post("/api/review/relocate-folder", {"path": str(moved_to)})

        with urllib.request.urlopen(self.base + "/thumb?path=" + quote(str(moved_image), safe="")) as response:
            self.assertEqual(response.status, 200)


class NoFolderOpenServerTests(unittest.TestCase):
    """`picklikeme review` with no --input at all: the server must come up
    and serve a working page with nothing open yet, purely so
    /api/review/open-folder has something to switch away from."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.store = AnnotationStore(self.root / "kb.db")
        self.session = ReviewSession(None, self.store)
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

    def test_the_page_still_loads(self):
        with urllib.request.urlopen(self.base + "/") as response:
            self.assertEqual(response.status, 200)

    def test_the_state_endpoint_reports_an_empty_folder_less_gallery(self):
        payload = self.get("/api/review/state")

        self.assertIsNone(payload["state"]["input_folder"])
        self.assertEqual(payload["state"]["images"], [])
        self.assertTrue(payload["state"]["warnings"])

    def test_every_path_taking_endpoint_refuses_until_a_folder_is_opened(self):
        somewhere = self.root / "irrelevant.jpg"
        somewhere.write_bytes(b"x")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/thumb?path=" + quote(str(somewhere), safe=""))
        self.assertEqual(ctx.exception.code, 403)

    def test_arranging_with_nothing_open_is_a_clean_400_not_a_crash(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/review/arrange", {"dry_run": True})
        self.assertEqual(ctx.exception.code, 400)

    def test_opening_a_folder_makes_it_reviewable(self):
        shoot = self.root / "shoot"
        shoot.mkdir()
        make_real_images([shoot / "IMG_0001.jpg"])

        payload = self.post("/api/review/open-folder", {"path": str(shoot)})

        self.assertEqual(Path(payload["state"]["input_folder"]).resolve(), shoot.resolve())
        self.assertEqual(len(payload["state"]["images"]), 1)
        with urllib.request.urlopen(self.base + "/thumb?path=" + quote(str(shoot / "IMG_0001.jpg"), safe="")) as response:
            self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
