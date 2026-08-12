"""Standalone, archivable evaluation reports for a reviewed folder.

The Review application already *is* the evaluation tool for the model: every
number a report needs - agreement, the confusion matrix, precision/recall/F1,
the list of disagreements - is computed once by `ReviewSession.agreement_stats`
and `ReviewSession.disagreements`, the same methods the side panel and the "AI
<-> User Differences" filter already use. A report can therefore never show a
number that disagrees with what the photographer saw on screen while making
those decisions.

Two output formats, for two different purposes:

- **HTML** (`build_evaluation_report_html`) - the primary format: a single
  self-contained file (no CDN, inline CSS only, exactly like the analyzer's
  own HTML report) meant to be archived next to a shoot or a training run and
  opened side by side with another version's report for comparison.
- **CSV** (`build_evaluation_report_csv`) - just the per-image "Detailed
  Differences" table. That is the only part of the report that is naturally
  tabular; the rest is a handful of summary numbers, not a spreadsheet.

Both are built from one `_ReportData` snapshot so the two formats can never
disagree with each other either.

Adding a new statistic later: write one `_section_*(data) -> str` function and
add it to `HTML_SECTIONS` below. Nothing else here, or in the server routes
that call this module, needs to change.
"""

from __future__ import annotations

import csv
import html
import io
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .session import ReviewImage, ReviewSession

CSS = """
:root{--bg:#f8fafc;--panel:#fff;--panel-2:#f1f5f9;--text:#0f172a;--muted:#64748b;
--border:#e2e8f0;--accent:#2563eb;--good:#059669;--bad:#dc2626;}
@media(prefers-color-scheme:dark){:root{--bg:#0b0f17;--panel:#121826;--panel-2:#1a2233;
--text:#e2e8f0;--muted:#94a3b8;--border:#1f2937;--accent:#60a5fa;--good:#34d399;--bad:#f87171;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px 60px}
header{border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:24px}
h1{font-size:22px;margin:0 0 6px}
.sub{color:var(--muted);font-size:13px}
section{background:var(--panel);border:1px solid var(--border);border-radius:12px;
padding:16px 20px;margin-bottom:18px}
h2{font-size:14px;margin:0 0 12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
td,th{padding:6px 10px;border-bottom:1px solid var(--border);text-align:left}
th{color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:none}
.kv td:first-child{color:var(--muted);width:45%}
.kv td:last-child{font-variant-numeric:tabular-nums}
.num{text-align:right;font-variant-numeric:tabular-nums}
.good{color:var(--good)}.bad{color:var(--bad)}
.matrix td,.matrix th{text-align:center}
.matrix .hit{background:var(--panel-2);font-weight:600}
.empty{color:var(--muted);font-style:italic}
"""


@dataclass
class _ReportData:
    """One immutable snapshot every section (and both writers) reads from."""

    folder_name: str
    generated_at: datetime
    model_name: str | None
    keep_percent: float
    counts: dict
    ai_counts: dict
    agreement: dict
    disagreements: list["ReviewImage"]


def _collect(session: "ReviewSession", generated_at: datetime | None = None) -> _ReportData:
    folder = session.input_folder
    return _ReportData(
        folder_name=folder.name if folder else "(no folder open)",
        generated_at=generated_at or datetime.now(),
        model_name=session.run_metadata.get("backbone"),
        keep_percent=session.keep_percent,
        counts=session.counts(),
        ai_counts=session.ai_suggestion_counts(),
        agreement=session.agreement_stats(),
        disagreements=session.disagreements(),
    )


def _pct(value: int, total: int) -> str:
    return f"{100 * value / total:.1f}%" if total else "&mdash;"


def _ratio(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "&mdash;"


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _ai_decision_for(image: "ReviewImage") -> str:
    """The AI's decision for one row of the Detailed Differences table.

    Every image here came from `ReviewSession.disagreements()`, which only
    ever includes a "keep"/"reject" review_status paired with the opposite
    AI suggestion (Neutral and unranked images are excluded there) - so the
    AI's side of a disagreement is always simply the other one.
    """
    return "reject" if image.review_status == "keep" else "keep"


# -- report sections ----------------------------------------------------------
# Each takes the same _ReportData and returns one complete <section>...
# </section>. Order here is the order they render in.


def _section_general_info(data: _ReportData) -> str:
    rows = [
        ("Folder", _e(data.folder_name)),
        ("Evaluation date/time", _e(data.generated_at.strftime("%Y-%m-%d %H:%M:%S"))),
        ("Model", _e(data.model_name) if data.model_name else "<span class=empty>unknown</span>"),
        ("AI keep threshold", f"{data.keep_percent:g}%"),
        ("Total images", f"{data.counts['total']:,}"),
    ]
    body = "".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows)
    return f"<section><h2>General Information</h2><table class=kv>{body}</table></section>"


def _section_summary(data: _ReportData) -> str:
    counts = data.counts
    rows = [
        ("Total Images", counts["total"]),
        ("AI Keep", data.ai_counts["keep"]),
        ("AI Reject", data.ai_counts["reject"]),
        ("User Keep", counts["keep"]),
        ("User Reject", counts["reject"]),
        ("User Neutral", counts["neutral"]),
    ]
    body = "".join(f"<tr><td>{label}</td><td class=num>{value:,}</td></tr>" for label, value in rows)
    return f"<section><h2>Summary Statistics</h2><table class=kv>{body}</table></section>"


def _section_agreement(data: _ReportData) -> str:
    agreement = data.agreement
    if not agreement["compared"]:
        return (
            "<section><h2>Agreement</h2>"
            "<p class=empty>No images are both ranked and decided yet, so there is nothing to compare.</p>"
            "</section>"
        )
    rows = [
        ("Compared (ranked &amp; decided)", f"{agreement['compared']:,}"),
        ("AI / User Agreement", f"<span class=good>{agreement['agree_percent']:.1f}%</span>"),
        ("AI / User Disagreement", f"<span class=bad>{agreement['disagree_percent']:.1f}%</span>"),
    ]
    body = "".join(f"<tr><td>{label}</td><td class=num>{value}</td></tr>" for label, value in rows)
    return f"<section><h2>Agreement</h2><table class=kv>{body}</table></section>"


def _section_confusion_matrix(data: _ReportData) -> str:
    agreement = data.agreement
    if not agreement["compared"]:
        return (
            "<section><h2>Confusion Matrix</h2>"
            "<p class=empty>No images are both ranked and decided yet, so there is nothing to compare.</p>"
            "</section>"
        )
    kk, kr = agreement["ai_keep_user_keep"], agreement["ai_keep_user_reject"]
    rk, rr = agreement["ai_reject_user_keep"], agreement["ai_reject_user_reject"]
    return f"""<section><h2>Confusion Matrix</h2>
<table class=matrix>
<tr><th></th><th>User Keep</th><th>User Reject</th></tr>
<tr><th>AI Keep</th><td class="hit">{kk:,}</td><td>{kr:,}</td></tr>
<tr><th>AI Reject</th><td>{rk:,}</td><td class="hit">{rr:,}</td></tr>
</table></section>"""


def _section_performance(data: _ReportData) -> str:
    agreement = data.agreement
    if not agreement["compared"]:
        return (
            "<section><h2>Performance Metrics</h2>"
            "<p class=empty>No images are both ranked and decided yet, so there is nothing to score.</p>"
            "</section>"
        )
    rows = [
        ("Precision", _ratio(agreement["precision"])),
        ("Recall", _ratio(agreement["recall"])),
        ("F1 Score", _ratio(agreement["f1"])),
    ]
    body = "".join(f"<tr><td>{label}</td><td class=num>{value}</td></tr>" for label, value in rows)
    return (
        f"<section><h2>Performance Metrics</h2>"
        f"<p class=sub>User review status is treated as ground truth; Keep is the positive class.</p>"
        f"<table class=kv>{body}</table></section>"
    )


def _section_differences(data: _ReportData) -> str:
    if not data.disagreements:
        return (
            "<section><h2>Detailed Differences</h2>"
            "<p class=empty>No disagreements between the AI and your review status.</p></section>"
        )
    rows = []
    for image in data.disagreements:
        score = f"{image.score:.3f}" if image.score is not None else "&mdash;"
        ai = _ai_decision_for(image).capitalize()
        rows.append(
            f"<tr><td>{_e(image.filename)}</td><td>{ai}</td>"
            f"<td>{_e(image.review_status.capitalize())}</td><td class=num>{score}</td></tr>"
        )
    body = "".join(rows)
    return f"""<section><h2>Detailed Differences</h2>
<table>
<tr><th>File Name</th><th>AI Decision</th><th>User Decision</th><th>AI Score</th></tr>
{body}
</table></section>"""


HTML_SECTIONS: tuple[tuple[str, Callable[[_ReportData], str]], ...] = (
    ("General Information", _section_general_info),
    ("Summary Statistics", _section_summary),
    ("Agreement", _section_agreement),
    ("Confusion Matrix", _section_confusion_matrix),
    ("Performance Metrics", _section_performance),
    ("Detailed Differences", _section_differences),
)


def build_evaluation_report_html(session: "ReviewSession", *, generated_at: datetime | None = None) -> str:
    """A complete, self-contained HTML report - see the module docstring."""
    data = _collect(session, generated_at)
    sections = "".join(render(data) for _title, render in HTML_SECTIONS)
    title = f"PickLikeMe Evaluation Report - {_e(data.folder_name)}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style></head><body>
<div class="wrap">
<header>
  <h1>Evaluation Report</h1>
  <div class="sub">{_e(data.folder_name)} &middot; generated {_e(data.generated_at.strftime("%Y-%m-%d %H:%M:%S"))}</div>
</header>
{sections}
</div>
</body></html>"""


def build_evaluation_report_csv(session: "ReviewSession") -> str:
    """The "Detailed Differences" table alone, as CSV - the one part of the
    report naturally suited to a spreadsheet. See the module docstring for
    why the summary numbers are not also duplicated here."""
    data = _collect(session)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["file_name", "ai_decision", "user_decision", "ai_score"])
    for image in data.disagreements:
        writer.writerow(
            [image.filename, _ai_decision_for(image), image.review_status, image.score if image.score is not None else ""]
        )
    return buffer.getvalue()
