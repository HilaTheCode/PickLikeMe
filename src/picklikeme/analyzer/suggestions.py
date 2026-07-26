"""Capability 13 - actionable recommendations, each backed by a measurement.

The rule this module lives by: **no suggestion without evidence**. Every
Suggestion carries the statistic that triggered it and the threshold that
statistic had to cross, so a reader can always check the reasoning instead of
trusting it. Nothing here is a heuristic hunch dressed as advice; if a rule's
condition is not met, the rule stays silent.

Rules are functions registered in RULES, so adding advice is one function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .analysis import AnalysisResult

# Severity ordering used for display.
CRITICAL, WARNING, INFO = "critical", "warning", "info"
_SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}


@dataclass(frozen=True)
class Suggestion:
    """One recommendation plus the evidence for it."""

    title: str
    detail: str
    evidence: str
    severity: str = INFO
    category: str = "general"
    action: str = ""
    images: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "detail": self.detail,
            "evidence": self.evidence,
            "severity": self.severity,
            "category": self.category,
            "action": self.action,
            "images": self.images,
        }


Rule = Callable[["AnalysisResult"], list[Suggestion]]
RULES: list[Rule] = []


def rule(func: Rule) -> Rule:
    RULES.append(func)
    return func


def _names(records, limit: int = 12) -> list[str]:
    return [record.filename for record in records[:limit]]


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

@rule
def threshold_advice(result: "AnalysisResult") -> list[Suggestion]:
    sweep = result.sweep
    if not sweep.is_worth_changing:
        return []
    current, best = sweep.current, sweep.recommended
    return [
        Suggestion(
            title=f"Move the decision threshold to {best.threshold:.2f}",
            detail=(
                f"At the current {current.threshold:.2f} the model scores "
                f"{sweep.optimize_for}={current.get(sweep.optimize_for):.4f}. At "
                f"{best.threshold:.2f} it scores {best.get(sweep.optimize_for):.4f} - "
                f"precision {best.precision:.3f} and recall {best.recall:.3f}, versus "
                f"{current.precision:.3f}/{current.recall:.3f} today."
            ),
            evidence=(
                f"Full sweep of {len(sweep.points)} thresholds; improvement in "
                f"{sweep.optimize_for} = {sweep.improvement:+.4f} (reported only above +0.01)."
            ),
            severity=WARNING,
            category="threshold",
            action=f"Re-run ranking with a {best.threshold:.2f} cut-off, or pass --threshold {best.threshold:.2f}.",
        )
    ]


# ---------------------------------------------------------------------------
# Dataset composition
# ---------------------------------------------------------------------------

@rule
def class_imbalance(result: "AnalysisResult") -> list[Suggestion]:
    positive, negative = result.match.num_selected, result.match.num_rejected
    total = positive + negative
    if total == 0:
        return []
    minority = min(positive, negative)
    share = minority / total
    if share >= 0.15:
        return []
    label = "selected" if positive < negative else "rejected"
    return [
        Suggestion(
            title=f"Training set is imbalanced - only {share * 100:.1f}% {label}",
            detail=(
                f"{positive:,} selected against {negative:,} rejected. A model can reach "
                f"{max(positive, negative) / total * 100:.1f}% accuracy by always predicting the "
                "majority class, so accuracy is not a meaningful headline number here."
            ),
            evidence=f"Matched ground truth: {positive:,} positive / {negative:,} negative.",
            severity=WARNING,
            category="dataset",
            action="Read balanced accuracy, MCC and PR AUC instead of accuracy; consider class weighting.",
        )
    ]


@rule
def unmatched_images(result: "AnalysisResult") -> list[Suggestion]:
    unmatched = len(result.match.unmatched)
    total = len(result.match.images)
    if total == 0 or unmatched / total < 0.05:
        return []
    return [
        Suggestion(
            title=f"{unmatched:,} ranked images have no ground truth ({unmatched / total * 100:.1f}%)",
            detail=(
                "These are excluded from every metric. If the ranking was produced before the "
                "folders were reorganised, the reported numbers describe only the images that "
                "still matched."
            ),
            evidence=f"{unmatched:,} of {total:,} ranked images matched no file in either folder.",
            severity=WARNING if unmatched / total < 0.25 else CRITICAL,
            category="data-quality",
            action="Check --selected/--rejected point at the folders the ranking was produced from.",
        )
    ]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@rule
def calibration_advice(result: "AnalysisResult") -> list[Suggestion]:
    ece = result.calibration.expected_calibration_error
    if ece is None:
        return []
    suggestions: list[Suggestion] = []
    if ece > 0.10:
        suggestions.append(
            Suggestion(
                title=f"Predicted probabilities are poorly calibrated (ECE {ece:.3f})",
                detail=(
                    "Stated confidence does not match observed accuracy, so a probability of 0.8 "
                    "does not mean 80% of such images are keepers. Thresholds chosen from these "
                    "numbers will not behave as expected."
                ),
                evidence=f"Expected calibration error {ece:.4f} over {len(result.calibration.populated)} populated bins (>0.10 flags).",
                severity=WARNING,
                category="calibration",
                action="Pick the threshold from the sweep rather than from an intuition about probability.",
            )
        )

    gap = result.metrics.get("overconfidence")
    if gap is not None and abs(gap) > 0.10:
        over = gap > 0
        suggestions.append(
            Suggestion(
                title=f"Model is {'over' if over else 'under'}confident by {abs(gap) * 100:.1f} points",
                detail=(
                    f"Mean confidence is {result.metrics.get('mean_confidence'):.3f} while accuracy is "
                    f"{result.metrics.get('accuracy'):.3f}. "
                    + (
                        "It commits hard to answers it gets wrong, so its confident mistakes are worth reviewing first."
                        if over
                        else "It hedges on answers it gets right, so the uncertainty band contains more usable images than expected."
                    )
                ),
                evidence=f"mean_confidence - accuracy = {gap:+.4f} (|gap| > 0.10 flags).",
                severity=INFO,
                category="calibration",
                action="Review the confident-mistakes sheet" if over else "Review the borderline sheet",
            )
        )
    return suggestions


# ---------------------------------------------------------------------------
# What to label next
# ---------------------------------------------------------------------------

@rule
def borderline_labelling(result: "AnalysisResult") -> list[Suggestion]:
    borderline = result.errors.borderline
    if not borderline:
        return []
    config = result.config
    total = len(result.evaluable)
    return [
        Suggestion(
            title=f"{len(borderline):,} borderline images are the best next labels",
            detail=(
                f"Their probability sits in [{config.borderline_low}, {config.borderline_high}], where the "
                "model has effectively no opinion. Labelling these changes the decision boundary far "
                "more per image than labelling images it already scores confidently."
            ),
            evidence=(
                f"{len(borderline):,} of {total:,} evaluated images ({len(borderline) / total * 100:.1f}%) "
                f"fall in the uncertainty band."
            ),
            severity=INFO,
            category="training-data",
            action="Start with contact_sheets/borderline_*.png - they are ordered from most uncertain.",
            images=_names(borderline),
        )
    ]


@rule
def most_valuable_mistakes(result: "AnalysisResult") -> list[Suggestion]:
    suggestions: list[Suggestion] = []
    for records, kind, explanation in (
        (
            result.errors.confident_false_positives,
            "false positives",
            "the model was sure these were keepers and you threw them away",
        ),
        (
            result.errors.confident_false_negatives,
            "false negatives",
            "the model was sure these were rejects and you kept them",
        ),
    ):
        if not records:
            continue
        worst = records[0]
        suggestions.append(
            Suggestion(
                title=f"{len(records)} high-confidence {kind} worth reviewing",
                detail=(
                    f"Ordered by how wrong the model was - {explanation}. The worst is "
                    f"{worst.filename} at probability "
                    f"{worst.image.probability if worst.image.probability is not None else worst.image.score:.3f}. "
                    "Confident mistakes cluster around blind spots, so they usually share a subject or condition."
                ),
                evidence=f"Severity = |probability - truth|; top record scores {worst.severity:.3f}.",
                severity=WARNING if worst.severity > 0.8 else INFO,
                category="training-data",
                action=f"Add the worst of these to the training set; see the {kind.replace(' ', '_')} contact sheet.",
                images=_names(records),
            )
        )
    return suggestions


@rule
def ranking_disagreements(result: "AnalysisResult") -> list[Suggestion]:
    records = result.errors.largest_rank_disagreements
    if not records or records[0].rank_displacement == 0:
        return []
    total = len(result.evaluable)
    worst = records[0]
    if worst.rank_displacement < max(10, total * 0.05):
        return []
    return [
        Suggestion(
            title=f"Worst ranking disagreement is {worst.rank_displacement:,} positions",
            detail=(
                f"{worst.filename} sits {worst.rank_displacement:,} places from where your decision puts it "
                f"(out of {total:,}). Large displacements matter even when the threshold happens to get the "
                "image right, because culling is done by scrolling the ranking, not by reading probabilities."
            ),
            evidence=(
                f"Displacement measured against the ideal ordering (all keepers first); "
                f"median displacement is {result.metrics.get('median_rank_displacement', 0):,.1f}."
            ),
            severity=INFO,
            category="ranking",
            action="Review contact_sheets/rank_disagreements_*.png.",
            images=_names(records),
        )
    ]


# ---------------------------------------------------------------------------
# Model quality
# ---------------------------------------------------------------------------

@rule
def discrimination_quality(result: "AnalysisResult") -> list[Suggestion]:
    auc = result.metrics.get("roc_auc")
    if auc is None:
        return []
    if auc < 0.6:
        return [
            Suggestion(
                title=f"The model barely discriminates (ROC AUC {auc:.3f})",
                detail=(
                    "0.5 is a coin flip. At this level the ordering carries little signal, and no threshold "
                    "will produce a useful cull. This is a training problem, not a threshold problem."
                ),
                evidence=f"ROC AUC = {auc:.4f} over {len(result.evaluable):,} matched images (<0.60 flags).",
                severity=CRITICAL,
                category="model",
                action="Train longer, unfreeze the backbone, or check that the crop cache matches the training data.",
            )
        ]
    if auc > 0.95:
        return [
            Suggestion(
                title=f"Ranking quality is excellent (ROC AUC {auc:.3f})",
                detail=(
                    "The model orders keepers above rejects almost perfectly. Remaining gains are in "
                    "threshold placement, not in the model."
                ),
                evidence=f"ROC AUC = {auc:.4f} over {len(result.evaluable):,} matched images (>0.95 flags).",
                severity=INFO,
                category="model",
                action="Tune the threshold for your preferred precision/recall trade-off.",
            )
        ]
    return []


@rule
def precision_recall_tradeoff(result: "AnalysisResult") -> list[Suggestion]:
    precision = result.metrics.get("precision")
    recall = result.metrics.get("recall")
    if precision is None or recall is None:
        return []
    if recall < 0.5 and precision > 0.8:
        return [
            Suggestion(
                title=f"The model is too strict - it misses {(1 - recall) * 100:.0f}% of your keepers",
                detail=(
                    f"Precision {precision:.3f} with recall {recall:.3f}: nearly everything it keeps is "
                    "right, but it discards too much. For culling, a missed keeper is usually more "
                    "expensive than an extra image to review."
                ),
                evidence=f"recall={recall:.4f} (<0.50) with precision={precision:.4f} (>0.80).",
                severity=WARNING,
                category="threshold",
                action="Lower the threshold, or optimise with --optimize-for recall.",
            )
        ]
    if precision < 0.5 and recall > 0.8:
        return [
            Suggestion(
                title=f"The model is too permissive - {(1 - precision) * 100:.0f}% of its keeps are wrong",
                detail=(
                    f"Recall {recall:.3f} with precision {precision:.3f}: it finds your keepers but buries "
                    "them among rejects, so the shortlist still needs heavy manual culling."
                ),
                evidence=f"precision={precision:.4f} (<0.50) with recall={recall:.4f} (>0.80).",
                severity=WARNING,
                category="threshold",
                action="Raise the threshold, or optimise with --optimize-for precision.",
            )
        ]
    return []


@rule
def error_patterns(result: "AnalysisResult") -> list[Suggestion]:
    """Repeated mistakes in one folder point at a systematic weakness."""
    from collections import Counter
    from pathlib import Path

    errors = result.errors.false_positives + result.errors.false_negatives
    if len(errors) < 5:
        return []

    folders = Counter(str(Path(record.image_path).parent) for record in errors)
    folder, count = folders.most_common(1)[0]
    share = count / len(errors)
    if share < 0.4 or len(folders) < 2:
        return []
    return [
        Suggestion(
            title=f"{share * 100:.0f}% of mistakes come from one folder",
            detail=(
                f"{count} of {len(errors)} errors are in {folder}. A single folder is usually a single "
                "shoot - one subject, one light, one background - so this looks like a systematic "
                "weakness rather than scattered noise."
            ),
            evidence=f"Errors grouped by parent folder across {len(folders)} folders; top folder holds {count}.",
            severity=INFO,
            category="training-data",
            action="Add examples from that shoot to the training set.",
        )
    ]


def generate_suggestions(result: "AnalysisResult") -> list[Suggestion]:
    """Run every rule, most severe first."""
    suggestions: list[Suggestion] = []
    for rule_func in RULES:
        try:
            suggestions.extend(rule_func(result))
        except Exception:  # noqa: BLE001 - a broken rule must not lose the report
            import logging

            logging.getLogger(__name__).warning("Suggestion rule %s failed", rule_func.__name__, exc_info=True)
    suggestions.sort(key=lambda s: _SEVERITY_ORDER.get(s.severity, 9))
    return suggestions
