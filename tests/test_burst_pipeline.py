import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.burst import BurstEntry, reconstruct_bursts


class BurstPipelineTests(unittest.TestCase):
    def test_reconstruct_bursts_groups_close_timestamps(self):
        entries = [
            BurstEntry(path="a.CR2", timestamp="2024-01-01T10:00:00.000000", burst_id=None),
            BurstEntry(path="b.CR2", timestamp="2024-01-01T10:00:00.500000", burst_id=None),
            BurstEntry(path="c.CR2", timestamp="2024-01-01T10:00:05.000000", burst_id=None),
        ]

        bursts = reconstruct_bursts(entries, max_gap_seconds=2.0)

        self.assertEqual(len(bursts), 2)
        self.assertEqual([e.path for e in bursts[0]], ["a.CR2", "b.CR2"])
        self.assertEqual([e.path for e in bursts[1]], ["c.CR2"])


if __name__ == "__main__":
    unittest.main()
