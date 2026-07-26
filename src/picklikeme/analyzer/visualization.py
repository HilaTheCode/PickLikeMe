"""Capability 12 - publication-quality charts.

Matplotlib with the Agg backend, chosen explicitly: the analyzer runs headless
(over SSH, in CI, beside a training job) and must never try to open a window.

One shared style keeps every chart legible in both the light and dark HTML
themes: transparent backgrounds, mid-grey axes that read on either, and no
reliance on colour alone to distinguish series.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # must precede pyplot; headless by design
import matplotlib.pyplot as plt  # noqa: E402

from .metrics.ranking import by_model_rank, top_k_count  # noqa: E402
from .thresholds import evaluate_threshold  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .analysis import AnalysisResult

logger = logging.getLogger(__name__)

# Readable on white and on near-black.
POSITIVE = "#3b82f6"
NEGATIVE = "#f97316"
ACCENT = "#10b981"
DANGER = "#ef4444"
NEUTRAL = "#94a3b8"
GRID = "#94a3b8"

FIGSIZE = (7.5, 4.6)
DPI = 130


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=NEUTRAL, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=NEUTRAL, fontsize=10)
    ax.set_ylabel(ylabel, color=NEUTRAL, fontsize=10)
    ax.tick_params(colors=NEUTRAL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_alpha(0.35)
    ax.grid(True, color=GRID, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # Transparent so one PNG serves both HTML themes.
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return path


def _legend(ax) -> None:
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(NEUTRAL)


def confusion_heatmap(result: "AnalysisResult", path: Path) -> Path:
    matrix = result.confusion
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cells = matrix.cells
    ax.imshow(cells, cmap="Blues", alpha=0.85)

    labels = [["TP", "FN"], ["FP", "TN"]]
    for row in range(2):
        row_total = sum(cells[row]) or 1
        for column in range(2):
            count = cells[row][column]
            ax.text(
                column,
                row,
                f"{labels[row][column]}\n{count:,}\n{count / row_total * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                color="#0f172a" if count < max(max(cells)) * 0.6 else "white",
                fontweight="bold",
            )
    ax.set_xticks([0, 1], ["model: KEEP", "model: REJECT"], color=NEUTRAL)
    ax.set_yticks([0, 1], ["you: KEPT", "you: REJECTED"], color=NEUTRAL)
    ax.set_title(f"Confusion matrix @ {matrix.threshold:.2f}", color=NEUTRAL, fontsize=12, pad=12)
    ax.tick_params(colors=NEUTRAL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, path)


def roc_curve(result: "AnalysisResult", path: Path) -> Path:
    points = result.sweep.points
    fpr = [p.false_positive_rate for p in points if p.false_positive_rate is not None]
    tpr = [p.recall for p in points if p.recall is not None]
    pairs = sorted(zip(fpr, tpr))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([x for x, _ in pairs], [y for _, y in pairs], color=POSITIVE, linewidth=2.2, label="model")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1.2, alpha=0.6, label="chance")
    auc = result.metrics.get("roc_auc")
    if auc is not None:
        ax.fill_between([x for x, _ in pairs], [y for _, y in pairs], alpha=0.12, color=POSITIVE)
        ax.text(0.62, 0.12, f"AUC = {auc:.4f}", color=POSITIVE, fontsize=12, fontweight="bold")
    _style(ax, "ROC curve", "False positive rate", "True positive rate (recall)")
    _legend(ax)
    return _save(fig, path)


def precision_recall_curve(result: "AnalysisResult", path: Path) -> Path:
    points = [p for p in result.sweep.points if p.precision is not None and p.recall is not None]
    pairs = sorted(((p.recall, p.precision) for p in points))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([r for r, _ in pairs], [p for _, p in pairs], color=ACCENT, linewidth=2.2, label="model")

    positives = result.match.num_selected
    total = len(result.evaluable)
    if total:
        baseline = positives / total
        ax.axhline(baseline, color=NEUTRAL, linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"chance ({baseline:.3f})")
    pr_auc = result.metrics.get("pr_auc")
    if pr_auc is not None:
        ax.text(0.05, 0.08, f"PR AUC = {pr_auc:.4f}", color=ACCENT, fontsize=12, fontweight="bold")
    _style(ax, "Precision-recall curve", "Recall", "Precision")
    _legend(ax)
    return _save(fig, path)


def threshold_curves(result: "AnalysisResult", path: Path) -> Path:
    points = result.sweep.points
    thresholds = [p.threshold for p in points]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for values, label, colour in (
        ([p.precision for p in points], "precision", POSITIVE),
        ([p.recall for p in points], "recall", ACCENT),
        ([p.f1 for p in points], "F1", DANGER),
        ([p.accuracy for p in points], "accuracy", NEUTRAL),
    ):
        xs = [t for t, v in zip(thresholds, values) if v is not None]
        ys = [v for v in values if v is not None]
        ax.plot(xs, ys, label=label, color=colour, linewidth=2.0)

    ax.axvline(result.sweep.current.threshold, color=NEUTRAL, linestyle=":", linewidth=1.6, label="current")
    ax.axvline(result.sweep.recommended.threshold, color=DANGER, linestyle="--", linewidth=1.6, label="recommended")
    _style(ax, "Metrics across every threshold", "Threshold", "Value")
    _legend(ax)
    return _save(fig, path)


def score_distribution(result: "AnalysisResult", path: Path) -> Path:
    dist = result.distribution
    centres = [(dist.edges[i] + dist.edges[i + 1]) / 2 for i in range(len(dist.edges) - 1)]
    width = (dist.edges[1] - dist.edges[0]) * 0.92
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(centres, dist.positive_counts, width=width, color=POSITIVE, alpha=0.75, label="you kept")
    ax.bar(centres, dist.negative_counts, width=width, color=NEGATIVE, alpha=0.6,
           bottom=dist.positive_counts, label="you rejected")
    ax.axvline(result.config.threshold, color=DANGER, linestyle="--", linewidth=1.8, label="threshold")
    _style(ax, "Score distribution by ground truth", "Model score", "Images")
    _legend(ax)
    return _save(fig, path)


def confidence_distribution(result: "AnalysisResult", path: Path) -> Path:
    dist = result.distribution
    if not any(dist.confidence_counts):
        raise ValueError("no confidence values available")
    centres = [
        (dist.confidence_edges[i] + dist.confidence_edges[i + 1]) / 2
        for i in range(len(dist.confidence_edges) - 1)
    ]
    width = (dist.confidence_edges[1] - dist.confidence_edges[0]) * 0.92
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(centres, dist.confidence_counts, width=width, color=ACCENT, alpha=0.8)
    _style(ax, "How committed the model is", "Confidence (0.5 = no opinion)", "Images")
    return _save(fig, path)


def calibration_plot(result: "AnalysisResult", path: Path) -> Path:
    populated = result.calibration.populated
    if not populated:
        raise ValueError("no calibration bins populated")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1.3, alpha=0.7, label="perfect")
    ax.plot(
        [b.mean_probability for b in populated],
        [b.observed_rate for b in populated],
        marker="o",
        color=POSITIVE,
        linewidth=2.0,
        label="model",
    )
    # Bar heights show how much data supports each point - a bin holding three
    # images should not be read like one holding three thousand.
    biggest = max(b.count for b in populated)
    for b in populated:
        ax.bar(b.mean_probability, b.count / biggest * 0.12, width=0.045, bottom=0,
               color=NEUTRAL, alpha=0.3)
    ece = result.calibration.expected_calibration_error
    if ece is not None:
        ax.text(0.05, 0.9, f"ECE = {ece:.4f}", color=POSITIVE, fontsize=12, fontweight="bold")
    _style(ax, "Calibration (reliability diagram)", "Predicted probability", "Observed keep rate")
    _legend(ax)
    return _save(fig, path)


def top_k_performance(result: "AnalysisResult", path: Path) -> Path:
    images = result.evaluable
    if not images:
        raise ValueError("no matched images")
    percents = list(result.config.top_percents)
    from .metrics.ranking import precision_at_k, recall_at_k

    precisions, recalls = [], []
    for percent in percents:
        k = top_k_count(len(images), percent)
        precisions.append(precision_at_k(images, k) or 0.0)
        recalls.append(recall_at_k(images, k) or 0.0)

    positions = range(len(percents))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar([p - 0.2 for p in positions], precisions, width=0.4, color=POSITIVE, label="precision@K")
    ax.bar([p + 0.2 for p in positions], recalls, width=0.4, color=ACCENT, label="recall@K")
    ax.set_xticks(list(positions), [f"top {p:g}%" for p in percents])
    _style(ax, "Quality of the top slice", "Cut-off", "Value")
    _legend(ax)
    return _save(fig, path)


def ranking_distribution(result: "AnalysisResult", path: Path) -> Path:
    """Where the keepers actually sit in the model's ordering.

    The clearest single picture of ranking quality: keepers bunched at the left
    means the top of the ranking is worth reviewing.
    """
    ordered = by_model_rank(result.evaluable)
    if not ordered:
        raise ValueError("no matched images")
    buckets = min(50, max(10, len(ordered) // 20))
    size = len(ordered) / buckets
    rates = []
    for index in range(buckets):
        chunk = ordered[int(index * size) : int((index + 1) * size)]
        rates.append(sum(1 for image in chunk if image.truth == 1) / len(chunk) if chunk else 0.0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(range(buckets), rates, width=0.9, color=POSITIVE, alpha=0.85)
    overall = result.match.num_selected / len(ordered)
    ax.axhline(overall, color=DANGER, linestyle="--", linewidth=1.6, label=f"overall keep rate ({overall:.3f})")
    ax.set_xticks(
        [0, buckets // 2, buckets - 1],
        ["top of ranking", "middle", "bottom"],
    )
    _style(ax, "Keep rate across the ranking", "Position in the model's ordering", "Share you kept")
    _legend(ax)
    return _save(fig, path)


def comparison_chart(result: "AnalysisResult", path: Path) -> Path:
    comparison = result.comparison
    if comparison is None:
        raise ValueError("not in comparison mode")
    deltas = [d for d in comparison.deltas if d.delta is not None][:14]
    if not deltas:
        raise ValueError("no comparable metrics")
    names = [d.name for d in deltas]
    values = [d.delta for d in deltas]
    colours = [ACCENT if d.improved else DANGER for d in deltas]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.38 * len(deltas) + 1.2)))
    ax.barh(range(len(deltas)), values, color=colours, alpha=0.85)
    ax.axvline(0, color=NEUTRAL, linewidth=1.2)
    ax.set_yticks(range(len(deltas)), names)
    ax.invert_yaxis()
    _style(
        ax,
        f"{comparison.candidate_label} vs {comparison.baseline_label}",
        "Change (green = better)",
        "",
    )
    return _save(fig, path)


# Every chart, with the filename it lands on. Failures are tolerated
# individually: a report missing one chart is far better than no report.
CHARTS = {
    "confusion_matrix.png": confusion_heatmap,
    "roc_curve.png": roc_curve,
    "precision_recall_curve.png": precision_recall_curve,
    "threshold_curves.png": threshold_curves,
    "score_distribution.png": score_distribution,
    "confidence_distribution.png": confidence_distribution,
    "calibration.png": calibration_plot,
    "top_k_performance.png": top_k_performance,
    "ranking_distribution.png": ranking_distribution,
    "comparison.png": comparison_chart,
}


def render_charts(result: "AnalysisResult") -> list[Path]:
    """Render every applicable chart into the config's charts directory."""
    if not result.evaluable:
        logger.info("No matched images; skipping charts.")
        return []

    output_dir = result.config.charts_dir
    written: list[Path] = []
    for filename, renderer in CHARTS.items():
        try:
            written.append(renderer(result, output_dir / filename))
        except Exception as exc:  # noqa: BLE001 - one chart must not lose the rest
            logger.debug("Chart %s skipped: %s", filename, exc)
    logger.info("Rendered %d chart(s) to %s", len(written), output_dir)
    return written
