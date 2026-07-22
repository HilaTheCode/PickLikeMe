import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.evaluate import ScoredImage, burst_top_k_accuracy, compute_metrics, roc_auc


class RocAucTests(unittest.TestCase):
    def test_known_value(self):
        # positive/negative pairs: (0.9,0.8) win, (0.9,0.1) win, (0.7,0.8) loss, (0.7,0.1) win
        self.assertAlmostEqual(roc_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]), 0.75)

    def test_perfect_separation(self):
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_ties_count_half(self):
        self.assertAlmostEqual(roc_auc([1, 0], [0.5, 0.5]), 0.5)

    def test_single_class_returns_none(self):
        self.assertIsNone(roc_auc([1, 1], [0.5, 0.6]))


class BurstTopKTests(unittest.TestCase):
    def _scored(self):
        return [
            # burst A: model ranks the selected frame first
            ScoredImage("a1.arw", 0.9, 1, "A"),
            ScoredImage("a2.arw", 0.8, 0, "A"),
            ScoredImage("a3.arw", 0.2, 0, "A"),
            # burst B: selected frame is ranked second
            ScoredImage("b1.arw", 0.7, 0, "B"),
            ScoredImage("b2.arw", 0.6, 1, "B"),
            ScoredImage("b3.arw", 0.1, 0, "B"),
            # burst C: all rejected -> not eligible
            ScoredImage("c1.arw", 0.5, 0, "C"),
            ScoredImage("c2.arw", 0.4, 0, "C"),
            # no burst id -> excluded from burst metrics
            ScoredImage("d1.arw", 0.3, 1, None),
        ]

    def test_top1(self):
        accuracy, eligible = burst_top_k_accuracy(self._scored(), k=1)
        self.assertEqual(eligible, 2)
        self.assertAlmostEqual(accuracy, 0.5)

    def test_top3(self):
        accuracy, eligible = burst_top_k_accuracy(self._scored(), k=3)
        self.assertEqual(eligible, 2)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_no_eligible_bursts(self):
        accuracy, eligible = burst_top_k_accuracy([ScoredImage("x.arw", 0.5, 1, None)], k=1)
        self.assertIsNone(accuracy)
        self.assertEqual(eligible, 0)


class ComputeMetricsTests(unittest.TestCase):
    def test_precision_recall_and_confusion(self):
        scored = [
            ScoredImage("a.arw", 0.9, 1, None),  # tp
            ScoredImage("b.arw", 0.8, 0, None),  # fp
            ScoredImage("c.arw", 0.2, 1, None),  # fn
            ScoredImage("d.arw", 0.1, 0, None),  # tn
        ]
        metrics = compute_metrics(scored, threshold=0.5)
        self.assertEqual(metrics["confusion"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["num_images"], 4)
        self.assertEqual(metrics["num_selected"], 2)


if __name__ == "__main__":
    unittest.main()
