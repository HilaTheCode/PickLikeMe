"""The review server: the existing annotation server, with a gallery on top.

`ReviewRequestHandler` subclasses `AnnotationRequestHandler` rather than
restating it, so `/source`, `/preview`, `/open-folder`, `/api/health` and the
whole annotation API arrive already written, already confined to the dataset,
and already tested. What is added here is the review itself: the page, the
gallery state, a decision write, and arranging.

Same constraints as its parent: loopback only, stdlib only, and every
path-taking endpoint confined to the reviewed folder by the inherited
`_within_dataset` check.

The one thing this server does that the annotation server never does is **move
files** - and only from `POST /api/review/arrange`, only after the browser has
shown the photographer an exact plan produced by a real dry run.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import HTTPServer, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..analyzer.annotations import AnnotationStore, InvalidReviewDecision, InvalidReviewReason
from ..analyzer.server import HOST, AnnotationRequestHandler
from ..identity import IdentityUnavailable
from .page import build_page
from .session import ReviewSession

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_PORT = 8757  # one above the annotation server, so both can run


class ReviewRequestHandler(AnnotationRequestHandler):
    """Serves the review page and its API.

    `session` is injected as a class attribute by `make_review_server()`, the
    same way the parent injects `store` and `root`. `ThreadingHTTPServer`
    builds a fresh handler per request, so per-server state cannot live on
    `self`.
    """

    session: ReviewSession
    # Serialises writes. The session is a single mutable object and the server
    # is threaded, so two decisions arriving together must not interleave.
    lock: threading.Lock = threading.Lock()

    # -- helpers ------------------------------------------------------------

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        # The gallery is a live view; a cached copy would show stale decisions.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _resolve(self, url_path: str):
        """The review page has no static assets - everything is inline or comes
        from a named endpoint. Refusing here keeps the inherited static-file
        fallback from ever reaching the filesystem."""
        return None

    def _query_path(self) -> str:
        params = parse_qs(urlparse(self.path).query)
        return (params.get("path") or [""])[0]

    def _state_payload(self) -> dict:
        return {"ok": True, "state": self.session.as_dict()}

    # -- routes -------------------------------------------------------------

    def _serve_thumbnail(self) -> None:
        """A cached thumbnail, with or without the detector's boxes drawn on.

        Built on demand rather than up front: a folder of several thousand
        images would otherwise stall the review for minutes before showing
        anything, and the browser only ever asks for the ones on screen.

        Both variants are separate files on disk, so toggling boxes is a
        second cached read rather than a re-render.
        """
        target = self._within_dataset(self._query_path())
        if target is None:
            return
        if not target.is_file():
            self._send_json({"error": "file not found (has it moved?)"}, status=404)
            return

        params = parse_qs(urlparse(self.path).query)
        want_boxes = (params.get("boxes") or ["0"])[0] not in ("", "0", "false")

        from .thumbnails import review_thumbnail

        try:
            thumbnail = review_thumbnail(str(target), with_boxes=want_boxes)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not break the gallery
            logger.warning("No thumbnail for %s: %s", target, exc)
            self._send_json({"error": f"could not build a thumbnail: {exc}"}, status=500)
            return
        if thumbnail is None:
            self._send_json({"error": "no preview available for this file"}, status=404)
            return

        data = thumbnail.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        # Content-addressed by path+size+version, so it is safe to cache hard.
        self.send_header("Cache-Control", "max-age=300")
        self.end_headers()
        self.wfile.write(data)

    def _serve_save_jpeg(self) -> None:
        """The currently-viewed image as a downloadable JPEG.

        Not a preview: the response carries `Content-Disposition: attachment`
        so the browser's own download/Save As handling takes over rather than
        rendering it inline, and the bytes come from `export_jpeg_bytes` -
        the camera's own embedded JPEG where one exists, so this is a share-
        ready copy, not a step toward RAW development.
        """
        target = self._within_dataset(self._query_path())
        if target is None:
            return
        if not target.is_file():
            self._send_json({"error": "file not found (has it moved?)"}, status=404)
            return

        from ..analyzer.contactsheets import export_jpeg_bytes

        try:
            data = export_jpeg_bytes(str(target))
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never a crash
            logger.warning("Could not export a JPEG for %s: %s", target, exc)
            self._send_json({"error": f"could not export a JPEG: {exc}"}, status=500)
            return

        filename = target.with_suffix(".jpg").name
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        route = urlparse(self.path).path
        if route in ("/", "/review.html", "/index.html"):
            self._send_html(build_page())
            return
        if route == "/thumb":
            self._serve_thumbnail()
            return
        if route == "/save-jpeg":
            self._serve_save_jpeg()
            return
        if route == "/api/review/state":
            self._send_json(self._state_payload())
            return
        # /source, /preview, /open-folder, /api/health, /api/fields and
        # /api/annotations are all inherited unchanged.
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        route = urlparse(self.path).path
        if not route.startswith("/api/review/"):
            super().do_POST()
            return

        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"bad request: {exc}"}, status=400)
            return

        handlers = {
            "/api/review/decision": self._post_decision,
            "/api/review/keep-percent": self._post_keep_percent,
            "/api/review/arrange": self._post_arrange,
            "/api/review/reconcile": self._post_reconcile,
        }
        handler = handlers.get(route)
        if handler is None:
            self._send_json({"error": f"no such endpoint: {route}"}, status=404)
            return

        try:
            with self.lock:
                handler(payload)
        except IdentityUnavailable as exc:
            # Explicit, never a fallback: a decision attached to a guess is
            # worse than no decision at all.
            logger.warning("Refusing to record a decision without identity: %s", exc)
            self._send_json(
                {
                    "error": (
                        f"Cannot establish this image's identity ({exc.reason}), so the decision "
                        "was not saved. The file must be readable to be reviewed."
                    ),
                    "identity_unavailable": True,
                },
                status=409,
            )
        except (InvalidReviewDecision, InvalidReviewReason, KeyError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never fatal
            logger.exception("Review request failed")
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _post_decision(self, payload: dict) -> None:
        image_path = (payload.get("image_path") or "").strip()
        if not image_path:
            raise ValueError("image_path is required")
        decision = payload.get("decision")
        if decision is not None and not isinstance(decision, str):
            raise ValueError("decision must be 'keep', 'reject', or null")
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or null")
        self.session.set_decision(image_path, decision, reason=reason)
        self._send_json(self._state_payload())

    def _post_keep_percent(self, payload: dict) -> None:
        self.session.set_keep_percent(payload.get("keep_percent"))
        self._send_json(self._state_payload())

    def _post_arrange(self, payload: dict) -> None:
        dry_run = bool(payload.get("dry_run", True))
        result = self.session.arrange(dry_run=dry_run)
        self._send_json(
            {
                "ok": True,
                "dry_run": dry_run,
                "result": {
                    "selected": result.selected,
                    "rejected": result.rejected,
                    "moved": result.moved,
                    "skipped": result.skipped,
                    "renamed": result.renamed,
                    "errors": result.errors,
                    "selected_dir": str(result.selected_dir) if result.selected_dir else None,
                    "rejected_dir": str(result.rejected_dir) if result.rejected_dir else None,
                    "failures": [{"path": p, "reason": r} for p, r in result.failures[:20]],
                },
                "state": self.session.as_dict(),
            }
        )

    def _post_reconcile(self, payload: dict) -> None:
        recovered = self.session.reconcile_by_identity()
        self._send_json({"ok": True, "recovered": recovered, "state": self.session.as_dict()})


def make_review_server(
    session: ReviewSession, store: AnnotationStore, port: int = DEFAULT_REVIEW_PORT
) -> HTTPServer:
    """Build (but do not start) the loopback review server.

    Unlike `analyzer.server.make_server` there is no `report.html`
    precondition: the review page is generated per request, and the folder
    being reviewed is a shoot rather than a report directory.
    """
    folder = session.input_folder
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    handler = type(
        "BoundReviewRequestHandler",
        (ReviewRequestHandler,),
        {
            "session": session,
            "store": store,
            # Everything path-taking is confined to the reviewed folder.
            "root": folder,
            "source_roots": (folder,),
            "lock": threading.Lock(),
        },
    )
    return ThreadingHTTPServer((HOST, port), handler)


def serve_review(
    session: ReviewSession,
    store: AnnotationStore,
    port: int = DEFAULT_REVIEW_PORT,
    open_browser: bool = True,
) -> None:
    """Serve a review session until interrupted."""
    server = make_review_server(session, store, port)
    url = f"http://{HOST}:{server.server_address[1]}/"
    counts = session.counts()

    print(f"Review: {url}")
    print(f"  folder:   {session.input_folder}")
    print(f"  ranking:  {session.ranking_file}")
    print(f"  images:   {counts['total']:,} ({counts['untouched']:,} without a ranking)")
    print(f"  database: {store.db_path}")
    print("  Ctrl+C to stop. No file is moved until you click Arrange.")
    for warning in session.warnings:
        print(f"  ! {warning}")

    if open_browser:
        # From a thread so a slow browser launch cannot delay serving.
        threading.Thread(target=lambda: __import__("webbrowser").open(url), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
        final = session.counts()
        print(f"{final['manual']:,} manual decision(s) recorded in {store.db_path}")
