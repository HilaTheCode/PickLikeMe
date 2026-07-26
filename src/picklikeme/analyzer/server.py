"""Local annotation server: makes Save in the HTML report actually persist.

A report opened as `file://` cannot write to SQLite - browsers have no such
API, and giving a page one would be a bad idea anyway. So annotating runs
through a tiny loopback HTTP server that serves the already-generated report
directory and exposes a small JSON API over the annotation store.

Deliberate constraints:

- **Loopback only.** Bound to 127.0.0.1 and never to 0.0.0.0. This is a
  personal knowledge base on a workstation; it has no authentication and must
  not be reachable from the network.
- **stdlib only.** `http.server` plus `sqlite3`. No Flask, no new dependency
  for a tool that runs for a few minutes at a time.
- **Serves only the analysis directory.** Paths are resolved and checked to be
  inside it, so a crafted URL cannot read elsewhere on disk.
- **Writes only annotations.** The API has no endpoint that can touch a
  ranking, a checkpoint, a report or an image.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..identity import IdentityUnavailable
from .annotations import AnnotationStore

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
DEFAULT_PORT = 8756
MAX_BODY_BYTES = 256 * 1024  # a note plus categories; anything larger is a mistake


class AnnotationRequestHandler(SimpleHTTPRequestHandler):
    """Serves the report directory and the annotation API.

    `store` and `root` are injected as class attributes by `serve()`.
    """

    store: AnnotationStore
    root: Path
    # Roots the /source endpoint may read from, taken from the report's own
    # analysis.json. Empty means originals are not served at all.
    source_roots: tuple[Path, ...] = ()
    protocol_version = "HTTP/1.1"

    # -- helpers ------------------------------------------------------------

    def _send_json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # No caching: the page must always see the current database.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body too large ({length} bytes)")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _resolve(self, url_path: str) -> Path | None:
        """Map a URL to a file inside the report directory, or None if it
        escapes - the check is done after resolution so `..` cannot slip out."""
        relative = unquote(urlparse(url_path).path).lstrip("/")
        candidate = (self.root / (relative or "report.html")).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            logger.warning("Refused path traversal attempt: %s", url_path)
            return None
        return candidate

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib signature
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- routes -------------------------------------------------------------

    def _serve_source_image(self) -> None:
        """Stream an original image from the dataset this report describes.

        Exists because browsers refuse to follow a `file://` link from an
        `http://` page, so a served report cannot open originals any other way.

        Confined to the dataset: the requested path must resolve inside one of
        the roots recorded in the report's own analysis.json. Anything else is
        403, so this endpoint cannot be turned into a general file reader.
        """
        params = parse_qs(urlparse(self.path).query)
        raw = (params.get("path") or [""])[0]
        if not raw:
            self._send_json({"error": "path is required"}, status=400)
            return

        try:
            target = Path(raw).resolve()
        except OSError:
            self._send_json({"error": "unusable path"}, status=400)
            return

        if not any(_is_within(target, root) for root in self.source_roots):
            logger.warning("Refused source outside the dataset roots: %s", target)
            self._send_json(
                {"error": "path is outside this report's dataset"}, status=403
            )
            return
        if not target.is_file():
            self._send_json({"error": "file not found (has the dataset moved?)"}, status=404)
            return

        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Let the browser offer a sensible filename for formats it cannot render.
        self.send_header("Content-Disposition", f'inline; filename="{target.name}"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        route = urlparse(self.path).path
        if route == "/source":
            self._serve_source_image()
            return
        if route == "/api/health":
            self._send_json({"ok": True, "database": str(self.store.db_path)})
            return
        if route == "/api/categories":
            self._send_json({"categories": self.store.categories()})
            return
        if route == "/api/annotations":
            self._send_json({"annotations": [a.as_dict() for a in self.store.all()]})
            return

        target = self._resolve(self.path)
        if target is None:
            self._send_json({"error": "forbidden"}, status=403)
            return
        if not target.is_file():
            self._send_json({"error": f"not found: {route}"}, status=404)
            return

        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if target.suffix.lower() in {".html", ".json"}:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        route = urlparse(self.path).path
        if route != "/api/annotations":
            self._send_json({"error": f"no such endpoint: {route}"}, status=404)
            return
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"bad request: {exc}"}, status=400)
            return

        image_path = (payload.get("image_path") or "").strip()
        if not image_path:
            self._send_json({"error": "image_path is required"}, status=400)
            return

        categories = payload.get("categories") or []
        if not isinstance(categories, list):
            self._send_json({"error": "categories must be a list"}, status=400)
            return

        try:
            annotation = self.store.save(image_path, categories, payload.get("notes") or "")
        except IdentityUnavailable as exc:
            # Explicit, not a fallback: without identity the annotation could
            # only be attached to a guess, and a wrong diagnosis is worse than
            # no diagnosis.
            logger.warning("Refusing to save without identity: %s", exc)
            self._send_json(
                {
                    "error": (
                        f"Cannot establish this image's identity ({exc.reason}), so the "
                        "annotation was not saved. The file must be readable to be annotated."
                    ),
                    "identity_unavailable": True,
                },
                status=409,
            )
            return
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never fatal
            logger.exception("Failed to save annotation")
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
            return

        self._send_json({"ok": True, "annotation": annotation.as_dict(), "deleted": annotation.is_empty})


def _is_within(candidate: Path, root: Path) -> bool:
    """True when `candidate` is inside `root`, both already resolved."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def dataset_roots(report_dir: Path) -> tuple[Path, ...]:
    """Folders the report's images legitimately live in.

    Read from the report's own analysis.json rather than configured separately,
    so the allowlist can never drift from the report being served. Falls back to
    an empty tuple - and therefore to serving no originals - if the file is
    missing or unreadable.
    """
    payload_path = report_dir / "analysis.json"
    if not payload_path.is_file():
        logger.info("No analysis.json in %s; original images will not be served", report_dir)
        return ()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        config = payload.get("config") or {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", payload_path, exc)
        return ()

    roots: list[Path] = []
    for key in ("selected_root", "rejected_root"):
        value = config.get(key)
        if value:
            try:
                resolved = Path(value).resolve()
            except OSError:
                continue
            if resolved.is_dir():
                roots.append(resolved)
    # No ground-truth folders (labels came from the ranking file): fall back to
    # the folders the ranked images actually sit in, which is still a bounded set.
    if not roots:
        seen: set[Path] = set()
        for group in ("false_negatives", "false_positives", "borderline"):
            for record in (payload.get("errors") or {}).get(group, []) or []:
                parent = Path(record.get("image_path", "")).parent
                if parent.parts:
                    try:
                        seen.add(parent.resolve())
                    except OSError:
                        continue
        roots.extend(sorted(seen))
    return tuple(roots)


def make_server(report_dir: Path, store: AnnotationStore, port: int = DEFAULT_PORT) -> HTTPServer:
    """Build (but do not start) the loopback server."""
    report_dir = Path(report_dir).resolve()
    if not report_dir.is_dir():
        raise SystemExit(f"Report directory not found: {report_dir}")
    if not (report_dir / "report.html").is_file():
        raise SystemExit(
            f"No report.html in {report_dir}. Run `picklikeme analyze --output {report_dir}` first."
        )

    roots = dataset_roots(report_dir)
    logger.info("Serving originals from %d dataset root(s)", len(roots))
    handler = type(
        "BoundAnnotationRequestHandler",
        (AnnotationRequestHandler,),
        {"store": store, "root": report_dir, "source_roots": roots},
    )
    return ThreadingHTTPServer((HOST, port), handler)


def serve(
    report_dir: str | Path,
    db_path: str | Path | None = None,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    """Serve a report directory until interrupted."""
    from .annotations import DEFAULT_ANNOTATIONS_DB

    store = AnnotationStore(db_path or DEFAULT_ANNOTATIONS_DB)
    server = make_server(Path(report_dir), store, port)
    url = f"http://{HOST}:{port}/report.html"

    print(f"Annotation server: {url}")
    print(f"  report:   {Path(report_dir).resolve()}")
    print(f"  database: {store.db_path}")
    print(f"  {store.count():,} annotation(s) already recorded")
    print("  Ctrl+C to stop. Nothing but the annotation database is written.")

    if open_browser:
        # Opened from a thread so a slow browser launch cannot delay serving.
        threading.Thread(
            target=lambda: __import__("webbrowser").open(url), daemon=True
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
        print(f"{store.count():,} annotation(s) in {store.db_path}")
        store.close()
