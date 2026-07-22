"""Protocol metrics shared by every model version.

docs/roadmap.md fixes one evaluation protocol for all versions so their
results are comparable. This module implements the metric core:

- Top-1 / Top-3 burst accuracy (primary metric): for each held-out burst
  containing at least one selected and one rejected frame, does one of the
  model's k highest-scored frames match a frame I actually selected?
- Image-level ROC AUC (rank-based, tie-aware, no sklearn dependency).
- Precision / recall / confusion counts at a fixed score threshold.

Richer reporting (per-burst visual comparisons, confusion matrices as
rendered reports) is deliberately deferred to V9; this module only provides
the numbers the protocol needs from V1 onward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch


@dataclass
class ScoredImage:
    image_path: str
    score: float
    label: int
    burst_id: Optional[str] = None


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    n = len(scores)
    num_pos = sum(1 for label in labels if label == 1)
    num_neg = n - num_pos
    if num_pos == 0 or num_neg == 0:
        return None

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)


def confusion_counts(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for label, score in zip(labels, scores):
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 1:
            counts["tp"] += 1
        elif predicted == 1 and label == 0:
            counts["fp"] += 1
        elif predicted == 0 and label == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def burst_top_k_accuracy(scored: Sequence[ScoredImage], k: int) -> tuple[Optional[float], int]:
    """Fraction of eligible bursts whose top-k scored frames include a selected
    frame. Eligible bursts have a burst_id, at least one selected frame, and at
    least one rejected frame — anything else has no decision to get right."""
    bursts: dict[str, list[ScoredImage]] = {}
    for item in scored:
        if item.burst_id:
            bursts.setdefault(item.burst_id, []).append(item)

    eligible = 0
    hits = 0
    for members in bursts.values():
        labels = {member.label for member in members}
        if labels != {0, 1}:
            continue
        eligible += 1
        top_k = sorted(members, key=lambda m: m.score, reverse=True)[:k]
        if any(member.label == 1 for member in top_k):
            hits += 1

    if eligible == 0:
        return None, 0
    return hits / eligible, eligible


def compute_metrics(scored: Sequence[ScoredImage], threshold: float = 0.5) -> dict:
    labels = [item.label for item in scored]
    scores = [item.score for item in scored]

    counts = confusion_counts(labels, scores, threshold)
    predicted_positive = counts["tp"] + counts["fp"]
    actual_positive = counts["tp"] + counts["fn"]
    top1, eligible = burst_top_k_accuracy(scored, k=1)
    top3, _ = burst_top_k_accuracy(scored, k=3)

    return {
        "num_images": len(scored),
        "num_selected": actual_positive,
        "num_rejected": len(scored) - actual_positive,
        "threshold": threshold,
        "roc_auc": roc_auc(labels, scores),
        "precision": counts["tp"] / predicted_positive if predicted_positive else None,
        "recall": counts["tp"] / actual_positive if actual_positive else None,
        "confusion": counts,
        "top1_burst_accuracy": top1,
        "top3_burst_accuracy": top3,
        "num_eligible_bursts": eligible,
    }


def score_items(model, items, loader, device: str = "cpu") -> list[ScoredImage]:
    model.eval()
    scored: list[ScoredImage] = []
    with torch.no_grad():
        for item in items:
            image = loader.load_image(item.image_path)
            tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float()
            logits = model(tensor.to(device))
            value = logits.squeeze(-1).squeeze(0)
            score = float(value.mean().cpu().item()) if value.ndim > 0 else float(value.cpu().item())
            scored.append(ScoredImage(image_path=item.image_path, score=score, label=int(item.label), burst_id=item.burst_id))
    return scored


def format_metrics(metrics: dict) -> str:
    def fmt(value) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        f"Images evaluated:     {metrics['num_images']} ({metrics['num_selected']} selected / {metrics['num_rejected']} rejected)",
        f"Eligible bursts:      {metrics['num_eligible_bursts']}",
        f"Top-1 burst accuracy: {fmt(metrics['top1_burst_accuracy'])}",
        f"Top-3 burst accuracy: {fmt(metrics['top3_burst_accuracy'])}",
        f"ROC AUC:              {fmt(metrics['roc_auc'])}",
        f"Precision@{metrics['threshold']}:        {fmt(metrics['precision'])}",
        f"Recall@{metrics['threshold']}:           {fmt(metrics['recall'])}",
        f"Confusion (tp/fp/tn/fn): {metrics['confusion']['tp']}/{metrics['confusion']['fp']}/{metrics['confusion']['tn']}/{metrics['confusion']['fn']}",
    ]
    return "\n".join(lines)


def write_metrics_json(metrics: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
