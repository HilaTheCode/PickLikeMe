"""Part 1 regression tests: every link in the report must be correct.

Two real bugs are pinned here:

1. A ranking file with relative image paths crashed report generation
   (`as_uri()` rejects relative paths, and the `exists()` guard in front of it
   passes because relative paths resolve against the CWD).

2. Source links were unreachable in the served report, because browsers refuse
   to follow a local-file URL from a served page. The link was never *wrong* -
   it did nothing, which is indistinguishable from wrong when you click it.
"""

import csv
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.analysis import run_analysis
from picklikeme.analyzer.annotations import AnnotationStore
from picklikeme.analyzer.config import AnalysisConfig
from picklikeme.analyzer.contactsheets import _thumbnail_cache_path, render_contact_sheets
from picklikeme.analyzer.links import (
    asset_url,
    folder_api_url,
    folder_file_uri,
    preview_api_url,
    source_api_url,
    source_file_uri,
)
from picklikeme.analyzer.reports.html import build_html, write_html_report
from picklikeme.analyzer.server import dataset_roots, make_server

# Distinct scores so the false-negative set is deterministic.
LAYOUT = [
    (True, 0.95), (True, 0.88), (True, 0.44), (True, 0.21),   # last two are FNs
    (False, 0.91), (False, 0.30), (False, 0.12), (False, 0.05),
]


def build_dataset(root: Path, subdir: str = "", jpeg: bool = True):
    """Ranking CSV plus ground-truth folders, optionally deeply nested."""
    selected = root / subdir / "keep" if subdir else root / "keep"
    rejected = root / subdir / "drop" if subdir else root / "drop"
    selected.mkdir(parents=True, exist_ok=True)
    rejected.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (keep, score) in enumerate(LAYOUT):
        target = (selected if keep else rejected) / f"IMG_{index:04d}.jpg"
        if jpeg:
            Image.fromarray(
                np.full((40, 60, 3), (index * 29) % 256, dtype=np.uint8)
            ).save(target, "JPEG")
        else:
            target.write_bytes(f"frame {index}".encode())
        rows.append([str(target), score, 1 if keep else 0])

    rows.sort(key=lambda row: -row[1])
    csv_path = root / "rankings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["relevant_images", len(rows)])
        writer.writerow([])
        writer.writerow(["rank", "image_path", "score", "label"])
        for position, row in enumerate(rows, start=1):
            writer.writerow([position, row[0], f"{row[1]:.6f}", row[2]])
    return csv_path, selected, rejected


def analyse(root: Path, output_dir: Path, **kwargs):
    ranking, selected, rejected = build_dataset(root, **kwargs)
    config = AnalysisConfig(
        ranking_path=ranking,
        selected_root=selected,
        rejected_root=rejected,
        output_dir=output_dir,
        charts=False,
        thumbnail_size=48,
        annotations_db=root / "kb.db",
    )
    return run_analysis(config), config


def all_links(html: str) -> list[str]:
    return re.findall(r'(?:src|href)="([^"]+)"', html)


class LinkHelperTests(unittest.TestCase):
    def test_relative_image_path_yields_a_usable_uri(self):
        """The crash case: a relative path must not blow up link generation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sub" / "a.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"x")
            previous = os.getcwd()
            os.chdir(root)
            try:
                uri = source_file_uri(Path("sub") / "a.jpg")
            finally:
                os.chdir(previous)
            self.assertIsNotNone(uri)
            self.assertTrue(uri.startswith("file:"))
            self.assertTrue(uri.endswith("a.jpg"))

    def test_missing_image_yields_no_link_rather_than_raising(self):
        self.assertIsNone(source_file_uri("/definitely/not/here.jpg"))

    def test_spaces_and_non_ascii_are_percent_encoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "my folder" / "IMG שלי.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"x")
            uri = source_file_uri(image)
            self.assertNotIn(" ", uri)
            self.assertIn("%20", uri)
            # Round-trips back to the same file.
            self.assertEqual(Path(unquote(urlparse(uri).path).lstrip("/")), image)

    def test_asset_url_is_relative_to_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            asset = out / "charts" / "roc.png"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"x")
            self.assertEqual(asset_url(asset, out), "charts/roc.png")

    def test_asset_url_survives_a_relative_output_dir(self):
        """The mixed absolute/relative case a bare relative_to() would reject."""
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                asset = Path(tmp) / "out" / "thumbnails" / "ab" / "x.jpg"
                asset.parent.mkdir(parents=True)
                asset.write_bytes(b"x")
                self.assertEqual(asset_url(asset, Path("out")), "thumbnails/ab/x.jpg")
                self.assertEqual(asset_url(Path("out/thumbnails/ab/x.jpg"), Path(tmp) / "out"),
                                 "thumbnails/ab/x.jpg")
            finally:
                os.chdir(previous)

    def test_asset_outside_the_report_falls_back_to_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out").mkdir()
            outside = root / "elsewhere" / "x.png"
            outside.parent.mkdir()
            outside.write_bytes(b"x")
            url = asset_url(outside, root / "out")
            self.assertTrue(url.startswith("file:"), url)

    def test_api_url_encodes_the_whole_path(self):
        url = source_api_url(r"D:\a b\ג.jpg")
        self.assertTrue(url.startswith("source?path="))
        self.assertNotIn(" ", url)
        self.assertNotIn("\\", url)

    def test_folder_file_uri_points_at_the_parent_with_a_trailing_slash(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "shoot" / "a.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"x")
            uri = folder_file_uri(image)
            self.assertTrue(uri.startswith("file:"))
            self.assertTrue(uri.endswith("/"), "a directory URI must end in / for browsers to list it")
            self.assertNotIn("a.jpg", uri)
            self.assertTrue(uri.endswith("shoot/"))

    def test_folder_file_uri_is_none_when_the_image_cannot_be_located(self):
        """Same rule as source_file_uri: an unlocatable image means its folder
        is not trustworthy either, so no link is offered rather than a guess."""
        self.assertIsNone(folder_file_uri("/definitely/not/here.jpg"))

    def test_folder_api_url_targets_the_parent_directory(self):
        url = folder_api_url(r"D:\shoot\a b.jpg")
        self.assertTrue(url.startswith("folder?path="))
        self.assertNotIn("a%20b.jpg", url)  # the file itself must not appear
        self.assertIn(quote(r"D:\shoot", safe=""), url)

    def test_preview_api_url_targets_the_file_itself(self):
        url = preview_api_url(r"D:\shoot\a b.jpg")
        self.assertTrue(url.startswith("preview?path="))
        self.assertIn(quote(r"D:\shoot\a b.jpg", safe=""), url)


class ReportLinkTests(unittest.TestCase):
    def test_relative_paths_in_the_ranking_do_not_break_the_report(self):
        """Regression: this raised ValueError and produced no report at all."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep = root / "keep"
            keep.mkdir()
            rows = []
            for index in range(6):
                image = keep / f"IMG_{index}.jpg"
                Image.fromarray(np.full((30, 30, 3), index * 30, dtype=np.uint8)).save(image)
                rows.append([index + 1, os.path.relpath(image, root), f"{0.9 - index * 0.15:.3f}",
                             1 if index < 4 else 0])
            csv_path = root / "r.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["rank", "image_path", "score", "label"])
                writer.writerows(rows)

            previous = os.getcwd()
            os.chdir(root)
            try:
                config = AnalysisConfig(
                    ranking_path=csv_path, output_dir=root / "out",
                    charts=False, contact_sheets=False, annotations_db=root / "kb.db",
                )
                html = build_html(run_analysis(config))
            finally:
                os.chdir(previous)
            self.assertIn("<!doctype html>", html)
            self.assertTrue([link for link in all_links(html) if link.startswith("file:")])

    def test_every_link_in_the_report_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config = analyse(root, root / "out", subdir="2026/india/birds")
            render_contact_sheets(result)
            html = write_html_report(result).read_text(encoding="utf-8")

            checked = 0
            for link in set(all_links(html)):
                if link.startswith("api/") or link.startswith("source?") or link.startswith("#"):
                    continue
                if link.startswith("file:"):
                    target = Path(unquote(urlparse(link).path).lstrip("/"))
                else:
                    target = config.output_dir / unquote(link)
                self.assertTrue(target.exists(), f"broken link: {link}")
                checked += 1
            self.assertGreater(checked, 5, "the report should contain links to check")

    def test_each_false_negative_thumbnail_belongs_to_its_own_image(self):
        """The failure mode a user would describe as a wrong link: the panel
        shows someone else's picture."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config = analyse(root, root / "out")
            render_contact_sheets(result)
            html = write_html_report(result).read_text(encoding="utf-8")

            # The false-positive category now gets `.fn` panels too (same
            # markup, same class - see analyzer.reports.html), so restrict the
            # check to panels whose path is actually a false negative.
            all_panels = re.findall(r'<div class="fn" data-path="([^"]*)".*?<img src="([^"]*)"', html, re.S)
            fn_paths = {r.image_path for r in result.errors.false_negatives}
            panels = [(path, src) for path, src in all_panels if path in fn_paths]
            self.assertEqual(len(panels), len(result.errors.false_negatives))
            for path, src in panels:
                expected = _thumbnail_cache_path(
                    config.thumbnails_dir, path, config.thumbnail_size
                ).resolve()
                actual = (config.output_dir / unquote(src)).resolve()
                self.assertEqual(actual, expected, f"thumbnail does not belong to {path}")
                # And the pixels really are that image's.
                index = int(Path(path).stem.split("_")[1])
                colour = int(np.asarray(Image.open(actual))[24, 24, 0])
                self.assertLessEqual(abs(colour - (index * 29) % 256), 3)

    def test_open_folder_and_preview_carry_the_absolute_path_for_the_served_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, _ = analyse(root, root / "out")
            render_contact_sheets(result)  # Open Preview's offline href needs a thumbnail to exist
            html = write_html_report(result).read_text(encoding="utf-8")

            folders = re.findall(r'data-folder="([^"]*)"', html)
            self.assertTrue(folders, "Open Folder must expose data-folder for the served rewrite")
            for path in folders:
                self.assertTrue(Path(path).is_absolute(), path)
                self.assertTrue(Path(path).is_dir(), path)

            previews = re.findall(r'data-preview="([^"]*)"', html)
            self.assertTrue(previews, "Open Preview must expose data-preview for the served rewrite")
            for path in previews:
                self.assertTrue(Path(path).is_absolute(), path)
                self.assertTrue(Path(path).exists(), path)

    def test_no_raw_hyperlink_remains(self):
        """The old direct link to the RAW file - unreliable since browsers
        cannot render RAW at all - must be gone, replaced by the two actions."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, _ = analyse(root, root / "out")
            html = build_html(result)
            self.assertNotIn("data-source=", html)
            self.assertIn("Open Folder", html)
            self.assertIn("Open Preview", html)

    def test_relocated_dataset_disables_open_folder_but_keeps_the_filename_readable(self):
        """If the images are gone, the filename must still be readable text
        and Open Folder must degrade to disabled rather than a dead link."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, _ = analyse(root, root / "out")
            for folder in ("keep", "drop"):
                for image in (root / folder).iterdir():
                    image.unlink()
            html = build_html(result)
            self.assertNotIn('data-folder="', html)
            self.assertIn("<span title=", html)
            self.assertIn('title="original file not found">Open Folder</span>', html)


class SourceEndpointTests(unittest.TestCase):
    # Windows holds the SQLite WAL open a moment after close(), which would
    # otherwise make temp-dir teardown flaky rather than reveal a real bug.
    def _tempdir(self):
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    """The served report reaches originals through /source, because a browser
    will not follow a local-file link from a served page."""

    def _serve(self, root: Path, output_dir: Path):
        store = AnnotationStore(root / "kb.db")
        server = make_server(output_dir, store, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, store, f"http://127.0.0.1:{server.server_address[1]}"

    def test_dataset_roots_come_from_the_report(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            result, config = analyse(root, root / "out")
            from picklikeme.analyzer.reports import write_json_report

            write_json_report(result, config.output_dir / "analysis.json")
            write_html_report(result)

            roots = dataset_roots(config.output_dir)
            self.assertIn((root / "keep").resolve(), roots)
            self.assertIn((root / "drop").resolve(), roots)

    def test_original_image_is_served_and_matches_the_file(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            result, config = analyse(root, root / "out")
            from picklikeme.analyzer.reports import write_json_report

            write_json_report(result, config.output_dir / "analysis.json")
            write_html_report(result)

            target = result.errors.false_negatives[0].image_path
            server, store, base = self._serve(root, config.output_dir)
            try:
                url = f"{base}/source?path={quote(target, safe='')}"
                with urllib.request.urlopen(url) as response:
                    served = response.read()
                    self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                self.assertEqual(served, Path(target).read_bytes())
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_paths_outside_the_dataset_are_refused(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            result, config = analyse(root, root / "out")
            from picklikeme.analyzer.reports import write_json_report

            write_json_report(result, config.output_dir / "analysis.json")
            write_html_report(result)
            secret = root / "secret.txt"
            secret.write_text("not an image", encoding="utf-8")

            server, store, base = self._serve(root, config.output_dir)
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base}/source?path={quote(str(secret), safe='')}")
                self.assertEqual(ctx.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_missing_path_parameter_is_a_bad_request(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            result, config = analyse(root, root / "out")
            from picklikeme.analyzer.reports import write_json_report

            write_json_report(result, config.output_dir / "analysis.json")
            write_html_report(result)
            server, store, base = self._serve(root, config.output_dir)
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base}/source")
                self.assertEqual(ctx.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_no_analysis_json_means_no_originals_are_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "bare"
            output.mkdir()
            (output / "report.html").write_text("<html></html>", encoding="utf-8")
            self.assertEqual(dataset_roots(output), ())


class FolderAndPreviewEndpointTests(unittest.TestCase):
    """Open Folder and Open Preview's served-mode counterparts: /folder and
    /preview, the equivalents of /source for a directory listing and a
    full-size RAW extraction respectively. Same confinement to the dataset
    roots as /source, checked the same way."""

    def _tempdir(self):
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def _serve(self, root: Path, output_dir: Path):
        store = AnnotationStore(root / "kb.db")
        server = make_server(output_dir, store, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, store, f"http://127.0.0.1:{server.server_address[1]}"

    def _prepare(self, root: Path):
        result, config = analyse(root, root / "out")
        from picklikeme.analyzer.reports import write_json_report

        write_json_report(result, config.output_dir / "analysis.json")
        write_html_report(result)
        return result, config

    def test_folder_lists_the_files_it_contains(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            result, config = self._prepare(root)
            target = result.errors.false_negatives[0].image_path
            folder = str(Path(target).parent)

            server, store, base = self._serve(root, config.output_dir)
            try:
                url = f"{base}/folder?path={quote(folder, safe='')}"
                with urllib.request.urlopen(url) as response:
                    self.assertEqual(response.headers["Content-Type"], "text/html; charset=utf-8")
                    body = response.read().decode("utf-8")
                self.assertIn(Path(target).name, body)
                # Every listed entry links through /source, since a served
                # page cannot follow a file:// link at all, folder or file.
                self.assertIn(f"source?path={quote(target, safe='')}", body)
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_folder_outside_the_dataset_is_refused(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            _, config = self._prepare(root)
            outside = root / "elsewhere"
            outside.mkdir()

            server, store, base = self._serve(root, config.output_dir)
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base}/folder?path={quote(str(outside), safe='')}")
                self.assertEqual(ctx.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_folder_missing_path_parameter_is_a_bad_request(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            _, config = self._prepare(root)
            server, store, base = self._serve(root, config.output_dir)
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base}/folder")
                self.assertEqual(ctx.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_preview_streams_a_full_size_image_not_the_small_thumbnail(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            result, config = self._prepare(root)
            target = result.errors.false_negatives[0].image_path

            server, store, base = self._serve(root, config.output_dir)
            try:
                url = f"{base}/preview?path={quote(target, safe='')}"
                with urllib.request.urlopen(url) as response:
                    self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                    data = response.read()
                # The fixture's images are 40x60 JPEGs (see build_dataset) -
                # this must be that image, not the (much smaller) thumbnail.
                decoded = Image.open(__import__("io").BytesIO(data))
                self.assertEqual(decoded.size, (60, 40))
            finally:
                server.shutdown()
                server.server_close()
                store.close()

    def test_preview_outside_the_dataset_is_refused(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            _, config = self._prepare(root)
            secret = root / "secret.jpg"
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(secret)

            server, store, base = self._serve(root, config.output_dir)
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base}/preview?path={quote(str(secret), safe='')}")
                self.assertEqual(ctx.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                store.close()


if __name__ == "__main__":
    unittest.main()
