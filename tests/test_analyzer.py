"""Tests for the evaluation & analysis module.

Metric values are asserted against hand-computed numbers rather than against
whatever the code currently returns, so a refactor that changes a formula fails
here instead of silently redefining "precision".
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer import metrics as metrics_package
from picklikeme.analyzer.analysis import run_analysis
from picklikeme.analyzer.comparison import compare_runs
from picklikeme.analyzer.config import AnalysisConfig
from picklikeme.analyzer.errors import analyse_errors, severity_of
from picklikeme.analyzer.io import RankingFormatError, discover_chunks, load_ranking
from picklikeme.analyzer.matching import classify, match_dataset, predict
from picklikeme.analyzer.metrics.base import Metric, counts_of
from picklikeme.analyzer.metrics.classification import average_precision
from picklikeme.analyzer.metrics.ranking import (
    kendall_tau_b,
    ndcg,
    precision_at_k,
    rank_displacement,
    recall_at_k,
    spearman,
    top_k_count,
)
from picklikeme.analyzer.model import MatchedImage, Outcome, RankedImage
from picklikeme.analyzer.thresholds import confusion_matrix, evaluate_threshold, sweep_thresholds


def make(score, truth, rank=1, path=None, probability=None):
    """A MatchedImage with the outcome derived at threshold 0.5."""
    ranked = RankedImage(
        image_path=path or f"/img/{score}_{truth}.arw",
        score=score,
        rank=rank,
        probability=probability if probability is not None else score,
    )
    predicted = 1 if score >= 0.5 else 0
    return MatchedImage(ranked=ranked, truth=truth, predicted=predicted, outcome=classify(truth, predicted))


def write_ranking(path: Path, rows, header=("rank", "image_path", "score", "label"), preamble=True):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if preamble:
            writer.writerow(["metric", "value"])
            writer.writerow(["relevant_images", len(rows)])
            writer.writerow([])
        writer.writerow(header)
        writer.writerows(rows)
    return path


def build_dataset(tmp: Path, positives=12, negatives=28, seed=3):
    """A small but realistic fixture: folders of files plus a ranking CSV."""
    import random

    rng = random.Random(seed)
    selected, rejected = tmp / "keep", tmp / "drop"
    selected.mkdir(parents=True, exist_ok=True)
    rejected.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(positives + negatives):
        keep = index < positives
        target = (selected if keep else rejected) / f"IMG_{index:04d}.jpg"
        target.write_bytes(b"x")
        score = min(1.0, max(0.0, rng.gauss(0.75 if keep else 0.3, 0.15)))
        rows.append([str(target), score, 1 if keep else 0])

    rows.sort(key=lambda row: -row[1])
    ranked = [[position, row[0], f"{row[1]:.6f}", row[2]] for position, row in enumerate(rows, start=1)]
    return write_ranking(tmp / "rankings.csv", ranked), selected, rejected


# ---------------------------------------------------------------------------
# Capability 1 - matching
# ---------------------------------------------------------------------------

class MatchingTests(unittest.TestCase):
    def test_every_image_gets_exactly_one_outcome(self):
        images = [make(0.9, 1), make(0.8, 0), make(0.2, 1), make(0.1, 0)]
        counts = counts_of(images)
        self.assertEqual((counts.tp, counts.fp, counts.fn, counts.tn), (1, 1, 1, 1))

    def test_classify_covers_the_matrix(self):
        self.assertIs(classify(1, 1), Outcome.TRUE_POSITIVE)
        self.assertIs(classify(0, 1), Outcome.FALSE_POSITIVE)
        self.assertIs(classify(1, 0), Outcome.FALSE_NEGATIVE)
        self.assertIs(classify(0, 0), Outcome.TRUE_NEGATIVE)
        self.assertIs(classify(None, 1), Outcome.UNKNOWN)

    def test_unmatched_images_become_unknown_and_do_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep").mkdir()
            (root / "keep" / "a.jpg").write_bytes(b"x")
            images = [
                RankedImage(image_path=str(root / "keep" / "a.jpg"), score=0.9, rank=1),
                RankedImage(image_path=str(root / "nowhere" / "ghost.jpg"), score=0.8, rank=2),
            ]
            result = match_dataset(images, root / "keep", None, 0.5)

            self.assertEqual(len(result.images), 2)
            self.assertEqual(len(result.evaluable), 1)
            self.assertEqual(result.counts["unknown"], 1)
            self.assertTrue(any("ghost.jpg" in w for w in result.warnings))

    def test_matches_after_the_folder_moved(self):
        """The ranking holds a stale absolute path; the file is now elsewhere.
        A suffix match must still find it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shoot = root / "archive" / "shoot1"
            shoot.mkdir(parents=True)
            (shoot / "b.jpg").write_bytes(b"x")
            stale = RankedImage(image_path=r"D:\old\drive\shoot1\b.jpg", score=0.9, rank=1)

            result = match_dataset([stale], shoot.parent, None, 0.5)
            self.assertEqual(len(result.evaluable), 1)
            self.assertEqual(result.evaluable[0].truth, 1)

    def test_ambiguous_filename_in_both_folders_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep, drop = root / "keep", root / "drop"
            keep.mkdir()
            drop.mkdir()
            (keep / "dup.jpg").write_bytes(b"x")
            (drop / "dup.jpg").write_bytes(b"x")
            # A path that matches neither folder exactly, only by filename.
            stale = RankedImage(image_path=r"D:\elsewhere\dup.jpg", score=0.9, rank=1)

            result = match_dataset([stale], keep, drop, 0.5)
            self.assertEqual(result.counts["unknown"], 1, "an ambiguous name must not be guessed")

    def test_labels_in_the_ranking_are_used_when_no_folders_given(self):
        images = [
            RankedImage(image_path="/a.arw", score=0.9, rank=1, label=1),
            RankedImage(image_path="/b.arw", score=0.1, rank=2, label=0),
        ]
        result = match_dataset(images, None, None, 0.5)
        self.assertEqual(len(result.evaluable), 2)
        self.assertEqual(result.counts["true_positive"], 1)
        self.assertEqual(result.counts["true_negative"], 1)

    def test_ground_truth_images_missing_from_the_ranking_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep = root / "keep"
            keep.mkdir()
            for name in ("a.jpg", "b.jpg"):
                (keep / name).write_bytes(b"x")
            images = [RankedImage(image_path=str(keep / "a.jpg"), score=0.9, rank=1)]

            result = match_dataset(images, keep, None, 0.5)
            self.assertEqual(len(result.unranked_selected), 1)
            self.assertTrue(any("absent from the ranking" in w for w in result.warnings))

    def test_predict_prefers_an_explicit_predicted_class(self):
        image = RankedImage(image_path="/a", score=0.9, rank=1, predicted_class=0)
        self.assertEqual(predict(image, 0.5), 0)


# ---------------------------------------------------------------------------
# Capability 2 - classification metrics
# ---------------------------------------------------------------------------

class ClassificationMetricTests(unittest.TestCase):
    def setUp(self):
        # 3 TP, 2 FP, 1 FN, 4 TN - every metric below is hand-computed from these.
        self.images = (
            [make(0.9, 1), make(0.8, 1), make(0.7, 1)]
            + [make(0.9, 0), make(0.6, 0)]
            + [make(0.2, 1)]
            + [make(0.4, 0), make(0.3, 0), make(0.2, 0), make(0.1, 0)]
        )
        self.result = metrics_package.compute(self.images)

    def test_counts(self):
        counts = counts_of(self.images)
        self.assertEqual((counts.tp, counts.fp, counts.fn, counts.tn), (3, 2, 1, 4))

    def test_core_metrics_match_hand_computation(self):
        self.assertAlmostEqual(self.result.get("accuracy"), 7 / 10)
        self.assertAlmostEqual(self.result.get("precision"), 3 / 5)
        self.assertAlmostEqual(self.result.get("recall"), 3 / 4)
        self.assertAlmostEqual(self.result.get("specificity"), 4 / 6)
        self.assertAlmostEqual(self.result.get("f1"), 2 * 0.6 * 0.75 / (0.6 + 0.75))
        self.assertAlmostEqual(self.result.get("balanced_accuracy"), (0.75 + 4 / 6) / 2)
        self.assertAlmostEqual(self.result.get("false_positive_rate"), 2 / 6)
        self.assertAlmostEqual(self.result.get("false_negative_rate"), 1 / 4)
        self.assertAlmostEqual(self.result.get("negative_predictive_value"), 4 / 5)
        self.assertAlmostEqual(self.result.get("youden_j"), 0.75 + 4 / 6 - 1)

    def test_mcc_matches_the_closed_form(self):
        import math

        expected = (3 * 4 - 2 * 1) / math.sqrt(5 * 4 * 6 * 5)
        self.assertAlmostEqual(self.result.get("mcc"), expected)

    def test_undefined_metrics_are_none_not_zero(self):
        """All-negative predictions leave precision undefined; reporting 0.0
        would claim the model was wrong rather than silent."""
        images = [make(0.1, 1), make(0.2, 0)]
        result = metrics_package.compute(images)
        self.assertIsNone(result.get("precision"))

    def test_roc_auc_is_perfect_on_separable_data(self):
        images = [make(0.9, 1), make(0.8, 1), make(0.2, 0), make(0.1, 0)]
        self.assertAlmostEqual(metrics_package.compute(images).get("roc_auc"), 1.0)

    def test_roc_auc_reports_not_applicable_with_one_class(self):
        value = metrics_package.compute([make(0.9, 1), make(0.8, 1)]).by_name("roc_auc")
        self.assertIsNone(value.value)
        self.assertIn("both", value.detail)

    def test_average_precision_matches_hand_computation(self):
        # Ranked: +, -, +, -  ->  (1/1 + 2/3) / 2
        images = [make(0.9, 1), make(0.8, 0), make(0.7, 1), make(0.6, 0)]
        self.assertAlmostEqual(average_precision(images), (1.0 + 2 / 3) / 2)


# ---------------------------------------------------------------------------
# Capability 3 - ranking metrics
# ---------------------------------------------------------------------------

class RankingMetricTests(unittest.TestCase):
    def test_top_k_count_rounds_up_and_never_returns_zero(self):
        self.assertEqual(top_k_count(1000, 1.0), 10)
        self.assertEqual(top_k_count(140, 1.0), 2)
        self.assertEqual(top_k_count(40, 1.0), 1)

    def test_precision_and_recall_at_k(self):
        images = [make(0.9, 1), make(0.8, 1), make(0.7, 0), make(0.6, 1), make(0.1, 0)]
        self.assertAlmostEqual(precision_at_k(images, 2), 1.0)
        self.assertAlmostEqual(precision_at_k(images, 4), 3 / 4)
        self.assertAlmostEqual(recall_at_k(images, 2), 2 / 3)
        self.assertAlmostEqual(recall_at_k(images, 5), 1.0)

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        self.assertAlmostEqual(ndcg([make(0.9, 1), make(0.8, 1), make(0.2, 0)]), 1.0)

    def test_ndcg_drops_when_a_keeper_is_ranked_last(self):
        self.assertLess(ndcg([make(0.9, 0), make(0.8, 0), make(0.2, 1)]), 0.6)

    def test_spearman_is_capped_below_one_by_ties_in_binary_truth(self):
        """Perfect ordering cannot reach rho = 1 here: the scores are all
        distinct while truth has two big tie groups, so the tie-corrected ranks
        can never align exactly. 0.894 IS the perfect score for this shape, and
        a naive implementation that reported 1.0 would be wrong."""
        images = [make(0.9, 1), make(0.8, 1), make(0.2, 0), make(0.1, 0)]
        value = spearman([i.score for i in images], [float(i.truth) for i in images])
        self.assertAlmostEqual(value, 0.8944271909999159, places=6)

    def test_spearman_is_the_exact_negative_when_scores_invert_truth(self):
        images = [make(0.9, 0), make(0.8, 0), make(0.2, 1), make(0.1, 1)]
        value = spearman([i.score for i in images], [float(i.truth) for i in images])
        self.assertAlmostEqual(value, -0.8944271909999159, places=6)

    def test_kendall_tau_handles_ties_without_dividing_by_zero(self):
        # All truths identical: tau-b is undefined, not a crash.
        self.assertIsNone(kendall_tau_b([0.9, 0.5, 0.1], [1.0, 1.0, 1.0]))

    def test_rank_displacement_is_zero_for_a_perfect_ranking(self):
        displacement = rank_displacement([make(0.9, 1), make(0.8, 1), make(0.2, 0)])
        self.assertEqual(displacement.maximum, 0)
        self.assertEqual(displacement.average, 0)

    def test_rank_displacement_finds_the_worst_offender(self):
        images = [make(0.95, 0), make(0.9, 0), make(0.85, 0), make(0.1, 1, path="/late.arw")]
        displacement = rank_displacement(images)
        self.assertEqual(displacement.maximum, 3)
        self.assertEqual(displacement.worst_image.image_path, "/late.arw")


# ---------------------------------------------------------------------------
# Capability 4 / 5 - thresholds and confusion matrix
# ---------------------------------------------------------------------------

class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.images = [make(0.9, 1), make(0.7, 1), make(0.6, 0), make(0.4, 1), make(0.2, 0), make(0.1, 0)]

    def test_threshold_changes_the_counts(self):
        low = evaluate_threshold(self.images, 0.3)
        high = evaluate_threshold(self.images, 0.8)
        self.assertEqual((low.tp, low.fp), (3, 1))
        self.assertEqual((high.tp, high.fp), (1, 0))

    def test_sweep_recommends_the_best_threshold_for_the_target(self):
        sweep = sweep_thresholds(self.images, current_threshold=0.5, steps=101, optimize_for="f1")
        best_elsewhere = max(p.f1 for p in sweep.points if p.f1 is not None)
        self.assertAlmostEqual(sweep.recommended.f1, best_elsewhere)

    def test_recall_target_prefers_a_lower_threshold_than_precision(self):
        recall_sweep = sweep_thresholds(self.images, 0.5, 101, "recall")
        precision_sweep = sweep_thresholds(self.images, 0.5, 101, "precision")
        self.assertLessEqual(recall_sweep.recommended.threshold, precision_sweep.recommended.threshold)

    def test_tiny_improvements_are_not_recommended(self):
        perfect = [make(0.99, 1), make(0.98, 1), make(0.01, 0), make(0.02, 0)]
        self.assertFalse(sweep_thresholds(perfect, 0.5, 101, "f1").is_worth_changing)

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            sweep_thresholds(self.images, 0.5, 11, "nonsense")

    def test_confusion_matrix_percentages_are_row_normalised(self):
        matrix = confusion_matrix(self.images, 0.5)
        self.assertEqual(matrix.total, 6)
        self.assertIn("you: KEPT", matrix.render())
        self.assertEqual(matrix.cells[0][0], matrix.tp)


# ---------------------------------------------------------------------------
# Capabilities 7 / 8 / 9 - errors, hard mistakes, borderline
# ---------------------------------------------------------------------------

class ErrorAnalysisTests(unittest.TestCase):
    def test_severity_treats_both_error_directions_equally(self):
        confident_fp = make(0.98, 0)
        confident_fn = make(0.02, 1)
        self.assertAlmostEqual(severity_of(confident_fp), severity_of(confident_fn))

    def test_mistakes_are_sorted_most_confident_first(self):
        images = [make(0.55, 0), make(0.99, 0), make(0.75, 0), make(0.9, 1)]
        analysis = analyse_errors(images)
        scores = [record.image.score for record in analysis.false_positives]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(analysis.false_positives[0].image.score, 0.99)

    def test_borderline_band_is_configurable_and_ordered_by_uncertainty(self):
        images = [make(0.50, 1), make(0.47, 0), make(0.54, 1), make(0.95, 1), make(0.05, 0)]
        analysis = analyse_errors(images, borderline_low=0.45, borderline_high=0.55)
        self.assertEqual(len(analysis.borderline), 3)
        self.assertAlmostEqual(analysis.borderline[0].image.probability, 0.50)

    def test_narrow_band_selects_fewer_images(self):
        images = [make(0.46, 1), make(0.50, 0), make(0.54, 1)]
        wide = analyse_errors(images, borderline_low=0.45, borderline_high=0.55)
        narrow = analyse_errors(images, borderline_low=0.49, borderline_high=0.51)
        self.assertEqual(len(wide.borderline), 3)
        self.assertEqual(len(narrow.borderline), 1)

    def test_limit_caps_every_list(self):
        images = [make(0.9, 0) for _ in range(50)]
        analysis = analyse_errors(images, limit=10)
        self.assertEqual(len(analysis.false_positives), 10)


# ---------------------------------------------------------------------------
# IO - field auto-detection
# ---------------------------------------------------------------------------

class RankingIoTests(unittest.TestCase):
    def test_reads_the_pipeline_format_with_its_preamble(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_ranking(
                Path(tmp) / "r.csv",
                [[1, "/a.arw", "0.90", 1], [2, "/b.arw", "0.10", 0]],
            )
            ranking = load_ranking(path)
            self.assertEqual(len(ranking.images), 2)
            self.assertEqual(ranking.images[0].rank, 1)
            self.assertEqual(ranking.images[0].label, 1)
            self.assertEqual(ranking.preamble["relevant_images"], "2")

    def test_detects_alternative_column_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_ranking(
                Path(tmp) / "r.csv",
                [["/a.arw", "0.9", "0.88"], ["/b.arw", "0.1", "0.12"]],
                header=("filepath", "prediction", "prob"),
                preamble=False,
            )
            ranking = load_ranking(path)
            self.assertEqual(ranking.detected_columns["image_path"], "filepath")
            self.assertEqual(ranking.detected_columns["score"], "prediction")
            self.assertAlmostEqual(ranking.images[0].probability, 0.88)

    def test_rank_is_assigned_by_score_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_ranking(
                Path(tmp) / "r.csv",
                [["/low.arw", "0.10"], ["/high.arw", "0.90"]],
                header=("image_path", "score"),
                preamble=False,
            )
            ranking = load_ranking(path)
            ranks = {image.filename: image.rank for image in ranking.images}
            self.assertEqual(ranks["high.arw"], 1)
            self.assertEqual(ranks["low.arw"], 2)

    def test_in_range_scores_become_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_ranking(
                Path(tmp) / "r.csv", [["/a.arw", "0.75"]], header=("image_path", "score"), preamble=False
            )
            self.assertAlmostEqual(load_ranking(path).images[0].probability, 0.75)

    def test_out_of_range_scores_do_not_invent_a_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_ranking(
                Path(tmp) / "r.csv", [["/a.arw", "4.2"]], header=("image_path", "score"), preamble=False
            )
            self.assertIsNone(load_ranking(path).images[0].probability)

    def test_chunked_files_are_all_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ranking(root / "r.csv", [[1, "/a.arw", "0.9", 1]])
            write_ranking(root / "r_1.csv", [[2, "/b.arw", "0.5", 0]])
            write_ranking(root / "r_2.csv", [[3, "/c.arw", "0.1", 0]])

            self.assertEqual(len(discover_chunks(root / "r.csv")), 3)
            self.assertEqual(len(load_ranking(root / "r.csv").images), 3)

    def test_unrelated_similar_filename_is_not_swept_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ranking(root / "r.csv", [[1, "/a.arw", "0.9", 1]])
            write_ranking(root / "r_final.csv", [[1, "/z.arw", "0.9", 1]])
            self.assertEqual(len(discover_chunks(root / "r.csv")), 1)

    def test_a_file_with_no_header_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("alpha,beta\n1,2\n", encoding="utf-8")
            with self.assertRaises(RankingFormatError):
                load_ranking(path)

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_ranking(Path("no_such_ranking.csv"))


# ---------------------------------------------------------------------------
# Capability 16 - plugin discovery
# ---------------------------------------------------------------------------

class PluginDiscoveryTests(unittest.TestCase):
    def test_discovery_finds_metrics_from_every_module(self):
        names = {metric.name for metric in metrics_package.all_metrics()}
        for expected in ("accuracy", "roc_auc", "ndcg", "spearman", "expected_calibration_error"):
            self.assertIn(expected, names)

    def test_every_metric_declares_its_contract(self):
        for metric in metrics_package.all_metrics():
            self.assertTrue(metric.name, f"{type(metric).__name__} has no name")
            self.assertTrue(metric.description, f"{metric.name} has no description")
            self.assertIn(metric.category, {"classification", "ranking", "calibration", "general"})

    def test_metric_names_are_unique(self):
        names = [metric.name for metric in metrics_package.all_metrics()]
        self.assertEqual(len(names), len(set(names)))

    def test_subclassing_registers_a_new_metric_with_no_other_change(self):
        class SillyMetric(Metric):
            name = "silly_test_metric"
            description = "counts images"
            category = "general"

            def compute(self, images):
                return float(len(images))

        try:
            self.assertIn("silly_test_metric", {m.name for m in metrics_package.registered_metrics()})
            result = metrics_package.compute([make(0.9, 1)])
            self.assertEqual(result.get("silly_test_metric"), 1.0)
        finally:
            Metric._registry.remove(SillyMetric)

    def test_a_failing_metric_is_reported_not_raised(self):
        class ExplodingMetric(Metric):
            name = "exploding_test_metric"
            description = "always fails"

            def compute(self, images):
                raise RuntimeError("kaboom")

        try:
            value = metrics_package.compute([make(0.9, 1)]).by_name("exploding_test_metric")
            self.assertIsNone(value.value)
            self.assertIn("kaboom", value.detail)
        finally:
            Metric._registry.remove(ExplodingMetric)


# ---------------------------------------------------------------------------
# End to end, plus HTML / contact sheets / comparison
# ---------------------------------------------------------------------------

class EndToEndTests(unittest.TestCase):
    def test_full_analysis_produces_every_artefact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            config = AnalysisConfig(
                ranking_path=ranking,
                selected_root=selected,
                rejected_root=rejected,
                output_dir=root / "out",
                max_examples=8,
                thumbnail_size=80,
            )
            result = run_analysis(config)

            self.assertEqual(len(result.evaluable), 40)
            self.assertIsNotNone(result.metrics.get("accuracy"))
            self.assertTrue(result.suggestions)

            from picklikeme.analyzer.reports import write_json_report, write_text_report

            text_path = write_text_report(result, config.output_dir / "report.txt")
            json_path = write_json_report(result, config.output_dir / "analysis.json")
            self.assertIn("Confusion matrix", text_path.read_text(encoding="utf-8"))

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("metrics", payload)
            self.assertIn("accuracy", payload["metrics"])
            self.assertEqual(payload["matching"]["counts"]["unknown"], 0)

    def test_html_report_is_self_contained_and_themed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            config = AnalysisConfig(
                ranking_path=ranking,
                selected_root=selected,
                rejected_root=rejected,
                output_dir=root / "out",
                max_examples=5,
            )
            result = run_analysis(config)

            from picklikeme.analyzer.reports.html import write_html_report

            html_text = write_html_report(result).read_text(encoding="utf-8")

            self.assertIn("<!doctype html>", html_text)
            self.assertIn("prefers-color-scheme: dark", html_text)
            self.assertIn("data-theme", html_text)
            # Offline: no external resource of any kind.
            self.assertNotIn("http://", html_text)
            self.assertNotIn("https://", html_text)
            self.assertNotIn("cdn", html_text.lower())

    def test_contact_sheets_render_and_thumbnails_are_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Real JPEGs, so thumbnails can actually be produced.
            import numpy as np
            from PIL import Image

            ranking, selected, rejected = build_dataset(root, positives=6, negatives=10)
            for folder in (selected, rejected):
                for path in folder.iterdir():
                    Image.fromarray(
                        np.full((40, 60, 3), 120, dtype=np.uint8)
                    ).save(path.with_suffix(".jpg"), "JPEG")

            config = AnalysisConfig(
                ranking_path=ranking,
                selected_root=selected,
                rejected_root=rejected,
                output_dir=root / "out",
                max_examples=6,
                thumbnail_size=64,
                thumbnail_workers=2,
            )
            result = run_analysis(config)

            from picklikeme.analyzer.contactsheets import render_contact_sheets

            sheets = render_contact_sheets(result, crop_cache_dir=root / "no_cache")
            self.assertTrue(sheets)
            self.assertTrue(all(path.exists() for path in sheets))
            self.assertTrue(any(config.thumbnails_dir.rglob("*.jpg")))

            # Second pass must reuse the cache rather than re-decode.
            before = {p: p.stat().st_mtime_ns for p in config.thumbnails_dir.rglob("*.jpg")}
            render_contact_sheets(result, crop_cache_dir=root / "no_cache")
            after = {p: p.stat().st_mtime_ns for p in config.thumbnails_dir.rglob("*.jpg")}
            self.assertEqual(before, after, "thumbnails were regenerated instead of reused")

    def test_comparison_detects_fixed_and_broken_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep, drop = root / "keep", root / "drop"
            keep.mkdir()
            drop.mkdir()
            (keep / "a.jpg").write_bytes(b"x")
            (drop / "b.jpg").write_bytes(b"x")

            # Baseline: a wrong (FN), b right (TN). Candidate: a fixed, b broken.
            baseline_csv = write_ranking(
                root / "base.csv",
                [[1, str(drop / "b.jpg"), "0.40", 0], [2, str(keep / "a.jpg"), "0.20", 1]],
            )
            candidate_csv = write_ranking(
                root / "cand.csv",
                [[1, str(keep / "a.jpg"), "0.90", 1], [2, str(drop / "b.jpg"), "0.80", 0]],
            )

            config = AnalysisConfig(
                ranking_path=baseline_csv,
                selected_root=keep,
                rejected_root=drop,
                output_dir=root / "out",
                compare_ranking_path=candidate_csv,
            )
            result = run_analysis(config)
            comparison = result.comparison

            self.assertIsNotNone(comparison)
            self.assertEqual(comparison.common_images, 2)
            self.assertEqual([c.filename for c in comparison.fixed], ["a.jpg"])
            self.assertEqual([c.filename for c in comparison.broken], ["b.jpg"])
            # One fixed and one broken nets to zero, so the verdict must be
            # driven by the metrics and must state the net image count.
            self.assertIn("+0 images net corrected", comparison.verdict)
            self.assertTrue(comparison.improvements)

    def test_analysis_writes_nothing_outside_its_output_directory(self):
        """The analyzer is read-only: the ranking, the ground-truth folders and
        their mtimes must be untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_dataset(root)
            before = {
                path: path.stat().st_mtime_ns
                for path in list(selected.rglob("*")) + list(rejected.rglob("*")) + [ranking]
            }

            config = AnalysisConfig(
                ranking_path=ranking,
                selected_root=selected,
                rejected_root=rejected,
                output_dir=root / "out",
            )
            run_analysis(config)

            after = {
                path: path.stat().st_mtime_ns
                for path in list(selected.rglob("*")) + list(rejected.rglob("*")) + [ranking]
            }
            self.assertEqual(before, after)


class ConfigTests(unittest.TestCase):
    def test_invalid_values_are_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            AnalysisConfig(ranking_path=Path("r.csv"), threshold=1.5)
        with self.assertRaises(ValueError):
            AnalysisConfig(ranking_path=Path("r.csv"), borderline_low=0.6, borderline_high=0.4)
        with self.assertRaises(ValueError):
            AnalysisConfig(ranking_path=Path("r.csv"), optimize_for="vibes")

    def test_round_trips_through_a_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(
                json.dumps({"ranking_path": "r.csv", "threshold": 0.6, "optimize_for": "recall"}),
                encoding="utf-8",
            )
            config = AnalysisConfig.from_file(path)
            self.assertEqual(config.threshold, 0.6)
            self.assertEqual(config.optimize_for, "recall")

    def test_explicit_overrides_beat_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({"ranking_path": "r.csv", "threshold": 0.6}), encoding="utf-8")
            self.assertEqual(AnalysisConfig.from_file(path, threshold=0.8).threshold, 0.8)

    def test_unknown_keys_are_rejected_rather_than_ignored(self):
        with self.assertRaises(ValueError):
            AnalysisConfig.from_dict({"ranking_path": "r.csv", "typo_here": 1})


class CliTests(unittest.TestCase):
    def test_ranking_is_required(self):
        from picklikeme.analyzer.cli import build_parser, config_from_args

        args = build_parser().parse_args([])
        with self.assertRaises(SystemExit):
            config_from_args(args)

    def test_flags_map_onto_the_config(self):
        from picklikeme.analyzer.cli import build_parser, config_from_args

        args = build_parser().parse_args(
            ["--ranking", "r.csv", "--threshold", "0.7", "--optimize-for", "recall", "--no-html"]
        )
        config = config_from_args(args)
        self.assertEqual(config.ranking_path, Path("r.csv"))
        self.assertEqual(config.threshold, 0.7)
        self.assertEqual(config.optimize_for, "recall")
        self.assertFalse(config.html_report)

    def test_analyze_is_registered_on_the_main_cli(self):
        from picklikeme.ingest.cli import main

        with self.assertRaises(SystemExit):
            main(["analyze", "--help"])


if __name__ == "__main__":
    unittest.main()
