import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.split import assign_burst_splits, create_split, load_split


def _manifest_frame() -> pd.DataFrame:
    rows = []
    for burst in range(10):
        for frame in range(5):
            rows.append(
                {
                    "image_path": f"shoot/{'Keep' if frame == 0 else 'Reject'}/b{burst}_f{frame}.arw",
                    "label": 1 if frame == 0 else 0,
                    "burst_id": f"shoot::b{burst:04d}",
                }
            )
    rows.append({"image_path": "shoot/Keep/no_burst.arw", "label": 1, "burst_id": None})
    return pd.DataFrame(rows)


class AssignBurstSplitsTests(unittest.TestCase):
    def test_bursts_are_never_split_across_train_and_test(self):
        result = assign_burst_splits(_manifest_frame(), test_fraction=0.3, seed=7)
        for _, group in result[result["burst_id"].notna()].groupby("burst_id"):
            self.assertEqual(len(set(group["split"])), 1)

    def test_deterministic_for_same_seed(self):
        first = assign_burst_splits(_manifest_frame(), test_fraction=0.3, seed=7)
        second = assign_burst_splits(_manifest_frame(), test_fraction=0.3, seed=7)
        self.assertTrue(first.equals(second))

    def test_different_seed_changes_assignment(self):
        first = assign_burst_splits(_manifest_frame(), test_fraction=0.3, seed=7)
        second = assign_burst_splits(_manifest_frame(), test_fraction=0.3, seed=8)
        self.assertFalse(first["split"].equals(second["split"]))

    def test_test_fraction_is_approximately_respected(self):
        result = assign_burst_splits(_manifest_frame(), test_fraction=0.3, seed=7)
        test_count = (result["split"] == "test").sum()
        self.assertGreaterEqual(test_count, 0.3 * len(result))
        self.assertLess(test_count, 0.3 * len(result) + 5)


class CreateSplitTests(unittest.TestCase):
    def test_refuses_to_overwrite_existing_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.csv"
            split_path = Path(tmpdir) / "split.csv"
            _manifest_frame().to_csv(manifest_path, index=False)

            create_split(manifest_path, split_path)
            with self.assertRaises(FileExistsError):
                create_split(manifest_path, split_path)
            create_split(manifest_path, split_path, force=True)

    def test_load_split_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.csv"
            split_path = Path(tmpdir) / "split.csv"
            _manifest_frame().to_csv(manifest_path, index=False)

            created = create_split(manifest_path, split_path)
            loaded = load_split(split_path)
            self.assertEqual(len(created), len(loaded))
            self.assertIn("split", loaded.columns)


if __name__ == "__main__":
    unittest.main()
