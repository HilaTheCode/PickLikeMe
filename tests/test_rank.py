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

    def test_analytics_capture_records_score_runtime_and_environment(self):
        """The Analytics Dashboard's Experiment Metadata / Run Summary need
        the same shape of data from the AI model path that ranking.classic
        already records: a per-image "score", a runtime/images-per-second
        summary, and environment facts (git commit, application version,
        GPU) in params - see analytics.environment."""
        import picklikeme.rank as rank_module
        from picklikeme.analytics.store import AnalyticsStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.nef").write_bytes(b"raw")
            checkpoint = root / "ckpt.pt"
            checkpoint.write_bytes(b"fake")
            analytics_db = root / "analytics.db"

            with mock.patch.object(rank_module, "PreferenceHead"), \
                    mock.patch.object(
                        rank_module, "load_checkpoint", return_value={"model_state_dict": {}, "epoch": 7}
                    ), \
                    mock.patch.object(rank_module, "RawImageLoader"), \
                    mock.patch.object(
                        rank_module, "rank_dataset",
                        return_value=[("x.nef", 0.75, 0, str(root / "x.nef"))],
                    ), \
                    mock.patch.object(rank_module, "write_results_csv", return_value=[root / "written.csv"]):
                rank_module.rank_folder(
                    root, checkpoint=checkpoint, device="cpu", crop_birds=False, analytics_db=analytics_db,
                )

            with AnalyticsStore(analytics_db) as store:
                (run,) = store.list_runs()
                params = store.get_run(run["run_id"])["params"]
                self.assertEqual(params["algorithm_version"], "epoch-7")
                self.assertIn("git_commit", params)
                self.assertIn("application_version", params)
                self.assertIn("gpu_name", params)
                self.assertIn("cuda_available", params)

                summary = store.summary_metrics(run["run_id"])
                self.assertGreaterEqual(summary["runtime_seconds"], 0.0)
                self.assertGreaterEqual(summary["images_per_second"], 0.0)

                self.assertEqual(store.image_metrics(run["run_id"], str(root / "x.nef")), {"score": 0.75})

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
