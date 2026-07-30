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
from ..analyzer.os_actions import choose_folder
from ..analyzer.server import HOST, AnnotationRequestHandler
from ..identity import IdentityUnavailable
from .page import build_page
from .session import InvalidReviewStatus, ReviewSession
from .thumbnails import DEFAULT_PREVIEW_CACHE_MAX_BYTES

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
    # The preview cache's size budget (see thumbnails.review_preview) -
    # configurable via `picklikeme review --preview-cache-max-gb`, set as a
    # class attribute by make_review_server() the same way session/store are.
    preview_cache_max_bytes: int = DEFAULT_PREVIEW_CACHE_MAX_BYTES

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

    def _serve_evaluation_report(self, fmt: str) -> None:
        """The evaluation report - see evaluation_report.py - as a download.

        Same `Content-Disposition: attachment` pattern as `_serve_save_jpeg`:
        a real navigation (see page.py's `exportEvaluationReport`), not a
        fetch, so the browser's own Save As/download handling takes over.
        Standalone by design - meant to be archived next to a shoot or a
        training run and compared against another version's report later.
        """
        from .evaluation_report import build_evaluation_report_csv, build_evaluation_report_html

        folder = self.session.input_folder
        stem = folder.name if folder else "review"
        try:
            if fmt == "csv":
                body = build_evaluation_report_csv(self.session)
                content_type = "text/csv; charset=utf-8"
            else:
                body = build_evaluation_report_html(self.session)
                content_type = "text/html; charset=utf-8"
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never a crash
            logger.exception("Could not build the evaluation report")
            self._send_json({"error": f"could not build the evaluation report: {exc}"}, status=500)
            return

        data = body.encode("utf-8")
        filename = f"{stem}_evaluation_report.{fmt}"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_preview(self) -> None:
        """The Lightbox's full-size preview - overrides the inherited
        version (still used, unchanged, by the analysis report) to back it
        with a persistent on-disk cache (see thumbnails.review_preview), so
        revisiting the same image while flipping back and forth through a
        burst doesn't redo a real RAW decode + JPEG re-encode every time.
        Dispatched here automatically: `AnnotationRequestHandler.do_GET`
        calls `self._serve_preview()` for `/preview`, and Python's normal
        method resolution picks this override over the inherited one.
        """
        target = self._within_dataset(self._query_path())
        if target is None:
            return
        if not target.is_file():
            self._send_json({"error": "file not found (has the dataset moved?)"}, status=404)
            return

        from .thumbnails import review_preview

        try:
            cached = review_preview(str(target), max_bytes=self.preview_cache_max_bytes)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not break the viewer
            logger.warning("Could not build a preview for %s: %s", target, exc)
            self._send_json({"error": f"could not extract a preview: {exc}"}, status=500)
            return

        data = cached.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        # Safe to cache, unlike the inherited no-store version: content-
        # addressed by resolved path (see thumbnails.review_preview), the
        # same tradeoff the square thumbnail cache already makes.
        self.send_header("Cache-Control", "max-age=300")
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
        if route == "/evaluation-report.html":
            self._serve_evaluation_report("html")
            return
        if route == "/evaluation-report.csv":
            self._serve_evaluation_report("csv")
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
            "/api/review/status": self._post_status,
            "/api/review/bulk-status": self._post_bulk_status,
            "/api/review/apply-ai-suggestions": self._post_apply_ai_suggestions,
            "/api/review/keep-percent": self._post_keep_percent,
            "/api/review/arrange": self._post_arrange,
            "/api/review/reconcile": self._post_reconcile,
            "/api/review/open-folder": self._post_open_folder,
            "/api/review/relocate-folder": self._post_relocate_folder,
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
        except (InvalidReviewDecision, InvalidReviewReason, InvalidReviewStatus, KeyError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never fatal
            logger.exception("Review request failed")
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _post_status(self, payload: dict) -> None:
        image_path = (payload.get("image_path") or "").strip()
        if not image_path:
            raise ValueError("image_path is required")
        status = payload.get("status")
        if not isinstance(status, str):
            raise ValueError("status must be a string ('keep', 'reject', or 'neutral')")
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or null")
        reason_note = payload.get("reason_note")
        if reason_note is not None and not isinstance(reason_note, str):
            raise ValueError("reason_note must be a string or null")
        self.session.set_review_status(image_path, status, reason=reason, reason_note=reason_note)
        self._send_json(self._state_payload())

    def _post_bulk_status(self, payload: dict) -> None:
        """The multi-select toolbar's one request for many images - see
        ReviewSession.set_review_statuses. No `reason`: a bulk action is not
        the place to record a per-image judgement call.
        """
        image_paths = payload.get("image_paths")
        if not isinstance(image_paths, list) or not image_paths or not all(
            isinstance(p, str) for p in image_paths
        ):
            raise ValueError("image_paths must be a non-empty list of strings")
        status = payload.get("status")
        if not isinstance(status, str):
            raise ValueError("status must be a string ('keep', 'reject', or 'neutral')")
        result = self.session.set_review_statuses(image_paths, status)
        self._send_json({"ok": True, **result, "state": self.session.as_dict()})

    def _post_apply_ai_suggestions(self, payload: dict) -> None:
        """Bulk-accept the AI's current suggestion - see
        ReviewSession.apply_ai_suggestions. The one endpoint that lets the AI
        ranking influence review_status at all, and only because the
        photographer explicitly asked it to.

        `include_decided` defaults to False: a Neutral image is updated
        either way (nothing manual is at risk there), but an already-decided
        image is only overridden when the caller passes this explicitly true
        - the page's own flow only does so after showing the photographer
        how many images that would affect and asking first.
        """
        include_decided = bool(payload.get("include_decided", False))
        result = self.session.apply_ai_suggestions(include_decided=include_decided)
        self._send_json({"ok": True, **result, "state": self.session.as_dict()})

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

    def _pick_folder(self, payload: dict) -> Path | None:
        """Resolve which folder a caller means - shared by `/open-folder` and
        `/relocate-folder`, since both boil down to the same "an explicit
        `path`, else the native picker, else cancelled" resolution.

        `path` lets a caller (a test, or a photographer who'd rather paste a
        path than click through a dialog) skip the native picker entirely.
        Without it, `choose_folder` shows the OS's own folder browser - the
        server-side bridge a served page needs to reach the OS at all (see
        os_actions.py) - seeded at the folder currently under review.

        Returns None once a response has already been sent (the dialog was
        cancelled, or the chosen path is not a real directory) - the caller
        should simply return at that point, like any other early exit.
        """
        raw_path = payload.get("path")
        if raw_path is not None and not isinstance(raw_path, str):
            raise ValueError("path must be a string or null")
        if raw_path:
            folder = Path(raw_path)
        else:
            folder = choose_folder(initial_dir=self.session.input_folder)
            if folder is None:
                self._send_json({"ok": True, "cancelled": True, "state": self.session.as_dict()})
                return None
        if not folder.is_dir():
            self._send_json({"error": f"folder not found: {folder}"}, status=404)
            return None
        return folder

    def _retarget_dataset_roots(self) -> None:
        """The dataset confinement `/source`, `/thumb`, `/preview` and
        `/open-folder` all check - set on the class, not `self`, since a
        fresh handler instance is built per request (see the class
        docstring) and every one of them must see the new folder."""
        cls = type(self)
        cls.root = self.session.input_folder
        cls.source_roots = (self.session.input_folder,)

    def _post_open_folder(self, payload: dict) -> None:
        """Switch the review to a different folder - typically one that has
        never been ranked, so its photos can only be sorted by hand."""
        folder = self._pick_folder(payload)
        if folder is None:
            return
        self.session.open_folder(folder)
        self._retarget_dataset_roots()
        recovered = self.session.reconcile_by_identity()
        self._send_json({"ok": True, "recovered": recovered, "state": self.session.as_dict()})

    def _post_relocate_folder(self, payload: dict) -> None:
        """The folder this session was reviewing can no longer be found at
        its old location (moved, renamed, or a changed drive letter) - see
        ReviewSession.relocate_folder. Shares `_pick_folder` with
        `_post_open_folder`: picking a folder to point at is the same
        problem either way."""
        folder = self._pick_folder(payload)
        if folder is None:
            return
        result = self.session.relocate_folder(folder)
        self._retarget_dataset_roots()
        self._send_json({"ok": True, **result, "state": self.session.as_dict()})


def make_review_server(
    session: ReviewSession,
    store: AnnotationStore,
    port: int = DEFAULT_REVIEW_PORT,
    preview_cache_max_bytes: int = DEFAULT_PREVIEW_CACHE_MAX_BYTES,
) -> HTTPServer:
    """Build (but do not start) the loopback review server.

    Unlike `analyzer.server.make_server` there is no `report.html`
    precondition: the review page is generated per request, and the folder
    being reviewed is a shoot rather than a report directory.

    `session.input_folder` may be None - `picklikeme review` with no
    `--input` starts with nothing open at all. `source_roots` is then empty,
    so every path-taking endpoint refuses everything (correctly: there is no
    folder yet), until `/api/review/open-folder` sets a real one (see
    `ReviewRequestHandler._post_open_folder`).
    """
    folder = session.input_folder
    if folder is not None and not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    handler = type(
        "BoundReviewRequestHandler",
        (ReviewRequestHandler,),
        {
            "session": session,
            "store": store,
            # Everything path-taking is confined to the reviewed folder.
            # `root` is never actually used to serve a file here (see
            # ReviewRequestHandler._resolve) - it just needs a real Path.
            "root": folder or Path("."),
            "source_roots": (folder,) if folder else (),
            "lock": threading.Lock(),
            "preview_cache_max_bytes": preview_cache_max_bytes,
        },
    )
    return ThreadingHTTPServer((HOST, port), handler)


def serve_review(
    session: ReviewSession,
    store: AnnotationStore,
    port: int = DEFAULT_REVIEW_PORT,
    open_browser: bool = True,
    preview_cache_max_bytes: int = DEFAULT_PREVIEW_CACHE_MAX_BYTES,
) -> None:
    """Serve a review session until interrupted."""
    server = make_review_server(session, store, port, preview_cache_max_bytes=preview_cache_max_bytes)
    url = f"http://{HOST}:{server.server_address[1]}/"
    counts = session.counts()

    print(f"Review: {url}")
    print(f"  folder:   {session.input_folder or '(none yet - use Open Folder in the page)'}")
    print(f"  ranking:  {session.ranking_file or '(none)'}")
    print(f"  images:   {counts['total']:,} ({counts['neutral']:,} still Neutral)")
    print(f"  database: {store.db_path}")
    print(f"  preview cache: max {preview_cache_max_bytes / 1024**3:.1f} GB")
    print("  Ctrl+C to stop. No file is moved until you click Arrange.")
    for warning in session.warnings:
        print(f"  ! {warning}")

    # A one-time check at startup, unthrottled (unlike the one review_preview
    # itself does on every Nth write - see PREVIEW_CACHE_SWEEP_INTERVAL_WRITES):
    # if the budget was lowered since the last run, or the cache simply grew
    # past it while the app was closed, the invariant should hold again
    # immediately, not only once enough new previews trickle in to trigger it.
    from .thumbnails import REVIEW_PREVIEW_CACHE, _enforce_cache_budget

    _enforce_cache_budget(REVIEW_PREVIEW_CACHE, preview_cache_max_bytes)

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
        print(f"{final['keep']:,} kept, {final['reject']:,} rejected, in {store.db_path}")
