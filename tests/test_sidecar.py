"""The per-shoot state directory: `<folder>/.picklikeme/`.

The point of the sidecar is that `review --input <folder>` finds a ranking by
computing one path, so the tests that matter are the ones about determinism
(the path is a pure function of the folder) and about surviving `organize`'s
moves, which is what lets a shoot be reviewed a second time.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.organize import organize_by_decision
from picklikeme.sidecar import (
    RANKING_FILENAME,
    SIDECAR_DIRNAME,
    has_ranking,
    ranking_path,
    read_run_metadata,
    rewrite_ranking_paths,
    sidecar_dir,
    write_run_metadata,
)


def write_ranking(folder: Path, image_paths: list[str], chunk_paths: list[str] | None = None) -> Path:
    """A ranking in the real on-disk shape: metrics preamble, then rows."""
    target = ranking_path(folder)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, paths: list[str], first_rank: int) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerow(["select_root", str(folder)])
            writer.writerow(["reject_root", "(inference - no labels)"])
            writer.writerow(["relevant_images", len(paths)])
            writer.writerow([])
            writer.writerow(["rank", "image_path", "score", "label"])
            for offset, image in enumerate(paths, start=first_rank):
                writer.writerow([offset, image, f"{1.0 / offset:.6f}", 0])

    _write(target, image_paths, 1)
    if chunk_paths:
        _write(target.with_name(f"{target.stem}_1{target.suffix}"), chunk_paths, len(image_paths) + 1)
    return target


class PathConventionTests(unittest.TestCase):
    def test_the_path_is_a_pure_function_of_the_folder(self):
        folder = Path("D:/Shoot")
        self.assertEqual(sidecar_dir(folder), folder / SIDECAR_DIRNAME)
        self.assertEqual(ranking_path(folder), folder / SIDECAR_DIRNAME / RANKING_FILENAME)

    def test_the_name_carries_no_timestamp(self):
        """A timestamped name would put the user back in the business of
        knowing which file is 'the' ranking."""
        self.assertEqual(RANKING_FILENAME, "ranking.csv")

    def test_accepts_a_plain_string_folder(self):
        self.assertEqual(ranking_path("D:/Shoot"), Path("D:/Shoot") / SIDECAR_DIRNAME / RANKING_FILENAME)

    def test_has_ranking_is_false_until_one_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertFalse(has_ranking(folder))
            write_ranking(folder, [str(folder / "a.NEF")])
            self.assertTrue(has_ranking(folder))


class RunMetadataTests(unittest.TestCase):
    def test_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_run_metadata(folder, backbone="cnn", image_count=42)
            payload = read_run_metadata(folder)
            self.assertEqual(payload["backbone"], "cnn")
            self.assertEqual(payload["image_count"], 42)
            self.assertIn("written_at", payload)

    def test_missing_metadata_is_empty_not_an_error(self):
        """Provenance is for display; a review must never fail without it."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_run_metadata(Path(tmp)), {})

    def test_unreadable_metadata_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_run_metadata(folder, backbone="cnn")
            (folder / SIDECAR_DIRNAME / "run.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(read_run_metadata(folder), {})


class RewriteAfterMoveTests(unittest.TestCase):
    """Arranging moves every file. Without repointing, the ranking would
    describe paths that no longer exist and the shoot could never be reviewed
    again."""

    def _images(self, folder: Path, count: int) -> list[str]:
        paths = []
        for index in range(count):
            target = folder / f"IMG_{index:04d}.NEF"
            target.write_bytes(f"frame {index}".encode())
            paths.append(str(target))
        return paths

    def _rows(self, path: Path) -> list[list[str]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.reader(handle))

    def test_paths_are_repointed_to_where_organize_moved_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "shoot"
            folder.mkdir(parents=True)
            images = self._images(folder, 4)
            target = write_ranking(folder, images)

            result = organize_by_decision(images[:2], images[2:], folder)
            rewritten = rewrite_ranking_paths(folder, result.moves)

            self.assertEqual(rewritten, 4)
            body = target.read_text(encoding="utf-8")
            for new_path in result.moves.values():
                self.assertIn(str(new_path), body)
            # And the ranking now describes files that actually exist.
            from picklikeme.analyzer.io import load_ranking

            for image in load_ranking(target).images:
                self.assertTrue(Path(image.image_path).is_file())

    def test_the_preamble_survives_a_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "shoot"
            folder.mkdir(parents=True)
            images = self._images(folder, 2)
            target = write_ranking(folder, images)

            result = organize_by_decision(images[:1], images[1:], folder)
            rewrite_ranking_paths(folder, result.moves)

            rows = self._rows(target)
            self.assertEqual(rows[0], ["metric", "value"])
            self.assertEqual(rows[1][0], "select_root")
            self.assertEqual(rows[3], ["relevant_images", "2"])

    def test_every_chunk_is_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "shoot"
            folder.mkdir(parents=True)
            images = self._images(folder, 4)
            target = write_ranking(folder, images[:2], chunk_paths=images[2:])
            chunk = target.with_name(f"{target.stem}_1{target.suffix}")
            self.assertTrue(chunk.is_file())

            result = organize_by_decision(images, [], folder)
            rewrite_ranking_paths(folder, result.moves)

            chunk_body = chunk.read_text(encoding="utf-8")
            for original in images[2:]:
                self.assertNotIn(original, chunk_body)
                self.assertIn(str(result.moves[original]), chunk_body)

    def test_rows_that_did_not_move_are_left_exactly_as_they_were(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "shoot"
            folder.mkdir(parents=True)
            images = self._images(folder, 3)
            target = write_ranking(folder, images)
            before = self._rows(target)

            # Only the first image is filed; the other two are untouched.
            result = organize_by_decision(images[:1], [], folder)
            rewrite_ranking_paths(folder, result.moves)

            after = self._rows(target)
            self.assertEqual(after[7:], before[7:], "unmoved rows must not be rewritten")

    def test_no_moves_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "shoot"
            folder.mkdir(parents=True)
            images = self._images(folder, 2)
            target = write_ranking(folder, images)
            before = target.read_bytes()

            self.assertEqual(rewrite_ranking_paths(folder, {}), 0)
            self.assertEqual(target.read_bytes(), before)

    def test_a_missing_ranking_is_not_an_error(self):
        """Filing succeeded; bookkeeping that cannot be updated must not turn
        that into a failure."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(rewrite_ranking_paths(folder, {"a": Path("b")}), 0)


if __name__ == "__main__":
    unittest.main()
