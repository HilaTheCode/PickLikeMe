"""Burst Analysis: the review-time processing layer that groups images into
bursts (reusing burst.reconstruct_bursts wholesale) and ranks each burst's
own members by whatever score it was given - see the module docstring for
why it is deliberately blind to which ranking strategy produced that score.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.burst_analysis import BurstInfo, ScoredImage, analyze_bursts


class GroupingReusesBurstPyTests(unittest.TestCase):
    """Burst identification is not reimplemented here - it is burst.py's own
    capture-time-gap clustering, unchanged."""

    def test_close_timestamps_are_grouped_into_one_burst(self):
        images = [
            ScoredImage("a.NEF", "2024-01-01T10:00:00", 0.5),
            ScoredImage("b.NEF", "2024-01-01T10:00:00.500000", 0.9),
            ScoredImage("c.NEF", "2024-01-01T10:00:05", 0.1),
        ]

        result = analyze_bursts(images, max_gap_seconds=2.0)

        self.assertEqual(result["a.NEF"].burst_id, result["b.NEF"].burst_id)
        self.assertNotEqual(result["a.NEF"].burst_id, result["c.NEF"].burst_id)
        self.assertEqual(result["a.NEF"].burst_size, 2)
        self.assertEqual(result["c.NEF"].burst_size, 1)

    def test_a_wide_gap_splits_into_separate_bursts(self):
        images = [
            ScoredImage("a.NEF", "2024-01-01T10:00:00", 0.5),
            ScoredImage("b.NEF", "2024-01-01T10:05:00", 0.9),
        ]
        result = analyze_bursts(images, max_gap_seconds=2.0)
        self.assertNotEqual(result["a.NEF"].burst_id, result["b.NEF"].burst_id)


class UntimedImagesAreSingletonBurstsTests(unittest.TestCase):
    """Never guess a burst for an image with no reliable capture time - see
    ingest.burst's own documented policy, which this mirrors."""

    def test_no_captured_at_is_its_own_burst(self):
        images = [
            ScoredImage("a.NEF", "2024-01-01T10:00:00", 0.5),
            ScoredImage("b.NEF", None, 0.9),
        ]
        result = analyze_bursts(images)
        self.assertEqual(result["a.NEF"].burst_size, 1)
        self.assertEqual(result["b.NEF"].burst_size, 1)
        self.assertNotEqual(result["a.NEF"].burst_id, result["b.NEF"].burst_id)
        self.assertTrue(result["a.NEF"].burst_best)
        self.assertTrue(result["b.NEF"].burst_best)

    def test_an_unparseable_timestamp_is_treated_as_missing_not_fatal(self):
        images = [
            ScoredImage("a.NEF", "not-a-real-timestamp", 0.5),
            ScoredImage("b.NEF", "2024-01-01T10:00:00", 0.9),
        ]
        result = analyze_bursts(images)  # must not raise
        self.assertEqual(result["a.NEF"].burst_size, 1)
        self.assertEqual(result["b.NEF"].burst_size, 1)


class BurstRankingTests(unittest.TestCase):
    """burst_rank/burst_best - this module's own addition on top of the
    reused grouping."""

    def test_the_highest_score_in_a_burst_is_rank_one_and_best(self):
        images = [
            ScoredImage("a.NEF", "2024-01-01T10:00:00", 0.2),
            ScoredImage("b.NEF", "2024-01-01T10:00:00.3", 0.9),
            ScoredImage("c.NEF", "2024-01-01T10:00:00.6", 0.5),
        ]
        result = analyze_bursts(images)

        self.assertEqual(result["b.NEF"].burst_rank, 1)
        self.assertTrue(result["b.NEF"].burst_best)
        self.assertEqual(result["c.NEF"].burst_rank, 2)
        self.assertFalse(result["c.NEF"].burst_best)
        self.assertEqual(result["a.NEF"].burst_rank, 3)
        self.assertFalse(result["a.NEF"].burst_best)

    def test_exactly_one_member_per_burst_is_best(self):
        images = [
            ScoredImage("a.NEF", "2024-01-01T10:00:00", 0.2),
            ScoredImage("b.NEF", "2024-01-01T10:00:00.3", 0.9),
            ScoredImage("c.NEF", "2024-01-01T10:05:00", 0.5),  # a separate burst
        ]
        result = analyze_bursts(images)
        self.assertEqual(sum(1 for info in result.values() if info.burst_best), 2)

    def test_an_unscored_member_ranks_after_every_scored_one(self):
        """An image the chosen strategy never scored (filtered out, or the
        strategy simply never ran) can still belong to a burst and be
        ranked - just last among its members, never mistaken for a score
        of zero or excluded from the burst."""
        images = [
            ScoredImage("a.NEF", "2024-01-01T10:00:00", 0.1),
            ScoredImage("b.NEF", "2024-01-01T10:00:00.3", None),
            ScoredImage("c.NEF", "2024-01-01T10:00:00.6", 0.9),
        ]
        result = analyze_bursts(images)

        self.assertEqual(result["c.NEF"].burst_rank, 1)
        self.assertTrue(result["c.NEF"].burst_best)
        self.assertEqual(result["a.NEF"].burst_rank, 2)
        self.assertEqual(result["b.NEF"].burst_rank, 3)
        self.assertFalse(result["b.NEF"].burst_best)

    def test_every_image_gets_an_entry_even_a_burst_of_one(self):
        images = [ScoredImage("solo.NEF", "2024-01-01T10:00:00", 0.5)]
        result = analyze_bursts(images)
        self.assertEqual(result["solo.NEF"], BurstInfo(
            burst_id=result["solo.NEF"].burst_id, burst_size=1, burst_rank=1, burst_best=True
        ))

    def test_an_empty_input_produces_no_entries(self):
        self.assertEqual(analyze_bursts([]), {})


class StrategyBlindnessTests(unittest.TestCase):
    """The architectural property the whole module exists to guarantee: two
    runs differing only in where the numbers came from produce identical
    burst structure and rank ordering for the same numbers - nothing here
    ever branches on where a score came from."""

    def test_identical_scores_from_different_sources_produce_identical_results(self):
        ai_scores = [
            ScoredImage("a.NEF", "2024-01-01T10:00:00", 0.3),
            ScoredImage("b.NEF", "2024-01-01T10:00:00.4", 0.7),
        ]
        classic_vision_scores = [
            ScoredImage("a.NEF", "2024-01-01T10:00:00", 0.3),
            ScoredImage("b.NEF", "2024-01-01T10:00:00.4", 0.7),
        ]
        self.assertEqual(analyze_bursts(ai_scores), analyze_bursts(classic_vision_scores))


if __name__ == "__main__":
    unittest.main()
