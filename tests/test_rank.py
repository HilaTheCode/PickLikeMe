import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.dataset import UnlabeledImageDataset


class UnlabeledDatasetTests(unittest.TestCase):
    def test_from_folder_finds_raw_recursively_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            for n in ["a.NEF", "b.arw", "c.CR3", "sub/d.Nef"]:
                (root / n).write_bytes(b"raw")
            (root / "note.txt").write_bytes(b"skip")
            (root / "prev.jpg").write_bytes(b"skip")

            ds = UnlabeledImageDataset.from_folder(root)
            names = {Path(item.image_path).name for item in ds.items}
            self.assertEqual(names, {"a.NEF", "b.arw", "c.CR3", "d.Nef"})

    def test_items_are_unlabeled_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.nef").write_bytes(b"raw")
            ds = UnlabeledImageDataset.from_folder(root)
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds[0].label, 0)

    def test_count_sequences_groups_by_parent_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "burstA").mkdir()
            (root / "burstB").mkdir()
            for n in ["burstA/1.nef", "burstA/2.nef", "burstB/3.nef"]:
                (root / n).write_bytes(b"raw")
            ds = UnlabeledImageDataset.from_folder(root)
            self.assertEqual(ds.count_sequences(), 2)


class RankCliWiringTests(unittest.TestCase):
    def test_missing_checkpoint_exits_cleanly(self):
        # rank should refuse to run without a trained checkpoint, with a clear message.
        import picklikeme.rank as rank_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.nef").write_bytes(b"raw")
            argv = ["rank", "--input", str(root), "--checkpoint", str(root / "nope.pt")]
            old = sys.argv
            sys.argv = argv
            try:
                with self.assertRaises(SystemExit) as ctx:
                    rank_module.main()
                self.assertIn("Checkpoint not found", str(ctx.exception))
            finally:
                sys.argv = old

    def _run_rank(self, root: Path, extra_argv: list[str]):
        """Drive rank.main() with everything expensive mocked out, returning the
        path it handed to write_results_csv."""
        import picklikeme.rank as rank_module

        checkpoint = root / "ckpt.pt"
        checkpoint.write_bytes(b"fake")
        argv = [
            "rank",
            "--input", str(root),
            "--checkpoint", str(checkpoint),
            "--no-crop-birds",
            "--device", "cpu",
        ] + extra_argv
        old = sys.argv
        sys.argv = argv
        try:
            with mock.patch.object(rank_module, "PreferenceHead"), \
                    mock.patch.object(rank_module, "load_checkpoint", return_value={"model_state_dict": {}}), \
                    mock.patch.object(rank_module, "RawImageLoader"), \
                    mock.patch.object(rank_module, "rank_dataset", return_value=[]), \
                    mock.patch.object(rank_module, "write_results_csv", return_value=[root / "written.csv"]) as write_csv:
                rank_module.main()
        finally:
            sys.argv = old
        return Path(write_csv.call_args.args[0])

    def test_the_ranking_is_written_into_the_folder_it_describes(self):
        """The default: `review --input <folder>` must find the ranking by
        computing one path, so rank puts it there rather than in the CWD."""
        from picklikeme.sidecar import RANKING_FILENAME, SIDECAR_DIRNAME, ranking_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.nef").write_bytes(b"raw")

            written_path = self._run_rank(root, [])

            self.assertEqual(written_path, ranking_path(root))
            self.assertEqual(written_path.name, RANKING_FILENAME)
            self.assertEqual(written_path.parent.name, SIDECAR_DIRNAME)
            self.assertNotIn("rankings_", written_path.name, "the default carries no timestamp")

    def test_run_metadata_is_recorded_beside_the_ranking(self):
        from picklikeme.sidecar import read_run_metadata

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.nef").write_bytes(b"raw")

            self._run_rank(root, [])

            metadata = read_run_metadata(root)
            self.assertEqual(metadata["image_count"], 1)
            self.assertIn("written_at", metadata)
            self.assertIn("checkpoint", metadata)

    def test_explicit_output_csv_still_overrides_and_is_timestamped(self):
        """The escape hatch: an explicit path opts out of the sidecar entirely,
        and keeps the timestamp so consecutive runs never overwrite."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.nef").write_bytes(b"raw")

            written_path = self._run_rank(root, ["--output-csv", str(root / "rankings.csv")])

            self.assertRegex(written_path.name, r"^rankings_\d{8}-\d{6}\.csv$")
            self.assertEqual(written_path.parent, root)


if __name__ == "__main__":
    unittest.main()
