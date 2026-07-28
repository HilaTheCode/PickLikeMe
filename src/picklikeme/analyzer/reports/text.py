"""Console and plain-text reporting.

The text report is the one that always works - no matplotlib, no Pillow, no
browser - so it carries every headline number rather than being a teaser for
the HTML.
"""

from __future__ import annotations

from pathlib import Path

from ...config import format_duration
from ..analysis import AnalysisResult
from ..suggestions import CRITICAL, WARNING

_RULE = "=" * 78
_THIN = "-" * 78


def _fmt(value: float | None, spec: str = "{:.4f}") -> str:
    return "n/a" if value is None else spec.format(value)


def _section(title: str) -> list[str]:
    return ["", _RULE, title, _RULE]


def render_summary(result: AnalysisResult) -> str:
    """The headline block printed at the end of every run."""
    lines: list[str] = []
    lines += _section(result.config.report_title)
    lines.append(f"  ranking:    {result.ranking.path}")
    lines.append(f"  images:     {len(result.ranking.images):,} ranked, {len(result.evaluable):,} with ground truth")
    lines.append(f"  threshold:  {result.config.threshold:.3f}")
    lines.append(f"  generated:  {result.generated_at} in {format_duration(result.elapsed_seconds)}")

    if not result.has_ground_truth:
        lines.append("")
        lines.append("  NO GROUND TRUTH MATCHED - metrics cannot be computed.")
        lines.append("  Pass --selected/--rejected, or use a ranking file that carries labels.")
        return "\n".join(lines)

    lines.append("")
    lines.append("  Headline")
    for name, value in result.headline().items():
        lines.append(f"    {name:<22}{_fmt(value)}")

    lines.append("")
    lines.append(result.confusion.render())
    return "\n".join(lines)


def render_metrics(result: AnalysisResult) -> str:
    lines: list[str] = []
    for category in result.metrics.categories:
        lines += _section(f"{category.title()} metrics")
        for value in result.metrics.in_category(category):
            arrow = "" if value.value is None else ("up" if value.higher_is_better else "dn")
            detail = f"   {value.detail}" if value.detail else ""
            lines.append(f"  {value.name:<34}{value.rendered():>12}  {arrow:<3}{value.description}{detail}")
    return "\n".join(lines)


def render_thresholds(result: AnalysisResult) -> str:
    sweep = result.sweep
    lines = _section(f"Threshold analysis (optimising {sweep.optimize_for})")
    lines.append(
        f"  {'threshold':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'accuracy':>10}"
        f"{'bal.acc':>9}{'fpr':>8}{'fnr':>8}"
    )
    # Show a readable subset: every 10th point, plus current and recommended.
    interesting = {round(p.threshold, 4) for p in sweep.points[::10]}
    interesting.add(round(sweep.current.threshold, 4))
    interesting.add(round(sweep.recommended.threshold, 4))
    for point in sweep.points:
        if round(point.threshold, 4) not in interesting:
            continue
        marker = ""
        if point.threshold == sweep.current.threshold:
            marker = " <- current"
        if point.threshold == sweep.recommended.threshold:
            marker += " <- RECOMMENDED"
        lines.append(
            f"  {point.threshold:>10.3f}{_fmt(point.precision, '{:.3f}'):>11}{_fmt(point.recall, '{:.3f}'):>9}"
            f"{_fmt(point.f1, '{:.3f}'):>9}{_fmt(point.accuracy, '{:.3f}'):>10}"
            f"{_fmt(point.balanced_accuracy, '{:.3f}'):>9}{_fmt(point.false_positive_rate, '{:.3f}'):>8}"
            f"{_fmt(point.false_negative_rate, '{:.3f}'):>8}{marker}"
        )
    lines.append("")
    if sweep.is_worth_changing:
        lines.append(
            f"  Recommended threshold {sweep.recommended.threshold:.3f} "
            f"({sweep.optimize_for} {_fmt(sweep.current.get(sweep.optimize_for))} -> "
            f"{_fmt(sweep.recommended.get(sweep.optimize_for))}, {sweep.improvement:+.4f})"
        )
    else:
        lines.append(f"  Current threshold {sweep.current.threshold:.3f} is already near-optimal for {sweep.optimize_for}.")
    return "\n".join(lines)


def render_errors(result: AnalysisResult, limit: int = 15) -> str:
    lines: list[str] = []
    errors = result.errors

    def table(title: str, records, note: str = "") -> None:
        lines.extend(_section(title))
        if note:
            lines.append(f"  {note}")
        if not records:
            lines.append("  (none)")
            return
        lines.append(f"  {'#':>4}{'score':>9}{'conf':>8}{'rank':>8}{'disp':>8}  filename")
        for position, record in enumerate(records[:limit], start=1):
            image = record.image
            lines.append(
                f"  {position:>4}{image.score:>9.4f}"
                f"{_fmt(image.confidence, '{:.3f}'):>8}{image.rank:>8,}{record.rank_displacement:>8,}  "
                f"{record.filename}"
            )
        if len(records) > limit:
            lines.append(f"  ... and {len(records) - limit:,} more")

    table(
        "Highest-confidence false positives",
        errors.confident_false_positives,
        "The model was most certain these were keepers; you rejected them.",
    )
    table(
        "Highest-confidence false negatives",
        errors.confident_false_negatives,
        "The model was most certain these were rejects; you kept them.",
    )
    table(
        "Largest ranking disagreements",
        errors.largest_rank_disagreements,
        "Furthest from where your decision would place them.",
    )
    table(
        "Most surprising predictions",
        errors.most_surprising,
        "Confidently wrong AND badly misplaced in the ranking.",
    )
    table(
        f"Borderline images ({result.config.borderline_low}-{result.config.borderline_high})",
        errors.borderline,
        "The model has no opinion on these - the most informative images to label next.",
    )
    return "\n".join(lines)


def render_suggestions(result: AnalysisResult) -> str:
    lines = _section("Recommendations")
    if not result.suggestions:
        lines.append("  Nothing flagged - no rule's evidence threshold was crossed.")
        return "\n".join(lines)

    for suggestion in result.suggestions:
        badge = {CRITICAL: "[CRITICAL]", WARNING: "[WARNING] "}.get(suggestion.severity, "[info]    ")
        lines.append("")
        lines.append(f"  {badge} {suggestion.title}")
        lines.append(f"      {suggestion.detail}")
        lines.append(f"      evidence: {suggestion.evidence}")
        if suggestion.action:
            lines.append(f"      action:   {suggestion.action}")
    return "\n".join(lines)


def render_matching(result: AnalysisResult) -> str:
    lines = _section("Dataset matching")
    lines.append(result.match.summary())
    if result.match.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for warning in result.match.warnings[:10]:
            lines.append(f"    - {warning}")
    if result.ranking.warnings:
        lines.append("")
        lines.append("  Ranking file notes:")
        for warning in result.ranking.warnings[:5]:
            lines.append(f"    - {warning}")
    lines.append("")
    lines.append(f"  Detected columns: {result.ranking.detected_columns}")
    return "\n".join(lines)


def render_annotations(result: AnalysisResult) -> str:
    """Annotation knowledge base for both mistake categories, if a database was
    readable. Same fields and vocabulary for false negatives and false
    positives, rendered as two independent blocks so they read side by side."""
    from ..annotations import render_summary as render_annotation_summary

    blocks = []
    if result.annotation_summary is not None:
        blocks.append(
            render_annotation_summary(
                result.annotation_summary,
                result.annotation_fields_config,
                title="False negative annotations",
                item_label="false negatives",
            )
        )
    if result.fp_annotation_summary is not None:
        blocks.append(
            render_annotation_summary(
                result.fp_annotation_summary,
                result.annotation_fields_config,
                title="False positive annotations",
                item_label="false positives",
            )
        )
    if not blocks:
        return ""
    return "\n".join(_section("Annotations")[:-1]) + "\n" + "\n\n".join(blocks)


def render_full(result: AnalysisResult) -> str:
    """The complete text report, in the order a reader wants it."""
    parts = [
        render_summary(result),
        render_matching(result),
    ]
    if result.has_ground_truth:
        parts += [
            render_metrics(result),
            render_thresholds(result),
            render_errors(result),
            render_suggestions(result),
        ]
        annotations = render_annotations(result)
        if annotations:
            parts.append(annotations)
    if result.comparison is not None:
        parts.append("\n" + _RULE + "\n" + result.comparison.render())
    return "\n".join(parts) + "\n"


def write_text_report(result: AnalysisResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_full(result), encoding="utf-8")
    return output_path
