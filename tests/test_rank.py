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

    def test_output_csv_is_timestamped_so_runs_never_overwrite_each_other(self):
        """The ranked CSV handed to write_results_csv must carry the run's
        date/time, not the bare --output-csv name."""
        import picklikeme.rank as rank_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.nef").write_bytes(b"raw")
            checkpoint = root / "ckpt.pt"
            checkpoint.write_bytes(b"fake")

            argv = [
                "rank",
                "--input", str(root),
                "--checkpoint", str(checkpoint),
                "--output-csv", str(root / "rankings.csv"),
                "--no-crop-birds",
                "--device", "cpu",
            ]
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

            written_path = Path(write_csv.call_args.args[0])
            self.assertRegex(written_path.name, r"^rankings_\d{8}-\d{6}\.csv$")
            self.assertEqual(written_path.parent, root)


if __name__ == "__main__":
    unittest.main()
