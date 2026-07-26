"""Capability 11 - a self-contained, offline HTML report.

Constraints that shaped this:

- **Offline.** No CDN, no webfont, no framework. All CSS and JS is inline, so
  the file works from a USB stick or an air-gapped machine.
- **Both themes.** Follows the OS preference and offers a toggle; charts are
  transparent PNGs so one image serves both.
- **Links, not copies.** Thumbnails and charts are referenced by relative path
  rather than base64-embedded: a 60-image error sheet would otherwise produce a
  multi-megabyte HTML file that no browser enjoys.

Tables are sortable and sections collapsible with a few lines of vanilla JS.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..suggestions import CRITICAL, WARNING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..analysis import AnalysisResult

logger = logging.getLogger(__name__)

CSS = """
:root{--bg:#f8fafc;--panel:#fff;--panel-2:#f1f5f9;--text:#0f172a;--muted:#64748b;
--border:#e2e8f0;--accent:#2563eb;--good:#059669;--bad:#dc2626;--warn:#d97706;}
:root[data-theme=dark]{--bg:#0b0f17;--panel:#121826;--panel-2:#1a2233;--text:#e2e8f0;
--muted:#94a3b8;--border:#1f2937;--accent:#60a5fa;--good:#34d399;--bad:#f87171;--warn:#fbbf24;}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b0f17;--panel:#121826;
--panel-2:#1a2233;--text:#e2e8f0;--muted:#94a3b8;--border:#1f2937;--accent:#60a5fa;
--good:#34d399;--bad:#f87171;--warn:#fbbf24;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;
border-bottom:1px solid var(--border);padding-bottom:18px;margin-bottom:26px}
h1{font-size:23px;margin:0 0 6px}h2{font-size:18px;margin:0}
.sub{color:var(--muted);font-size:13px}
button{font:inherit;cursor:pointer;background:var(--panel);color:var(--text);
border:1px solid var(--border);border-radius:8px;padding:7px 13px}
button:hover{border-color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin-bottom:26px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.card .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card .value{font-size:26px;font-weight:650;margin-top:5px;font-variant-numeric:tabular-nums}
.card .note{color:var(--muted);font-size:12px;margin-top:3px}
section{background:var(--panel);border:1px solid var(--border);border-radius:12px;margin-bottom:18px;
overflow:hidden}
.head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:14px 18px;
cursor:pointer;user-select:none}
.head:hover{background:var(--panel-2)}
.head .chev{color:var(--muted);transition:transform .18s}
section.collapsed .chev{transform:rotate(-90deg)}
section.collapsed .body{display:none}
.body{padding:0 18px 18px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em;
cursor:pointer;white-space:nowrap;position:sticky;top:0;background:var(--panel)}
th:hover{color:var(--accent)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:hover{background:var(--panel-2)}
.scroll{overflow-x:auto;max-height:560px;overflow-y:auto}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}.muted{color:var(--muted)}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;
border:1px solid currentColor}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.chart{background:var(--panel-2);border:1px solid var(--border);border-radius:10px;padding:10px}
.chart img{width:100%;height:auto;display:block}
.chart .cap{color:var(--muted);font-size:12px;margin-top:6px;text-align:center}
.sug{border-left:3px solid var(--muted);background:var(--panel-2);border-radius:0 8px 8px 0;
padding:12px 15px;margin-bottom:11px}
.sug.critical{border-left-color:var(--bad)}.sug.warning{border-left-color:var(--warn)}
.sug .t{font-weight:650;margin-bottom:5px}
.sug .d{font-size:13.5px}
.sug .e{color:var(--muted);font-size:12.5px;margin-top:6px;font-family:ui-monospace,monospace}
.sug .a{font-size:13px;margin-top:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.tile{background:var(--panel-2);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.tile img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#0b0f17}
.tile .m{padding:6px 8px;font-size:11.5px;line-height:1.4}
.tile .fn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
a{color:var(--accent)}
footer{color:var(--muted);font-size:12.5px;text-align:center;margin-top:34px}
"""

JS = """
document.querySelectorAll('.head').forEach(h=>h.addEventListener('click',()=>
  h.parentElement.classList.toggle('collapsed')));

function setTheme(t){document.documentElement.setAttribute('data-theme',t);
  try{localStorage.setItem('plm-theme',t)}catch(e){}
  document.getElementById('theme-btn').textContent = t==='dark'?'Light mode':'Dark mode';}
(function(){let t;try{t=localStorage.getItem('plm-theme')}catch(e){}
  if(!t)t=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
  setTheme(t);})();
document.getElementById('theme-btn').addEventListener('click',()=>
  setTheme(document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark'));

// Sortable tables: numeric when the column is tagged .num, else lexicographic.
document.querySelectorAll('table').forEach(table=>{
  table.querySelectorAll('th').forEach((th,i)=>{
    th.addEventListener('click',()=>{
      const body=table.tBodies[0];
      const rows=[...body.rows];
      const numeric=th.classList.contains('num');
      const dir=th.dataset.dir==='asc'?-1:1;
      table.querySelectorAll('th').forEach(o=>{if(o!==th)delete o.dataset.dir});
      th.dataset.dir=dir===1?'asc':'desc';
      rows.sort((a,b)=>{
        const x=a.cells[i]?.dataset.v??a.cells[i]?.textContent??'';
        const y=b.cells[i]?.dataset.v??b.cells[i]?.textContent??'';
        return dir*(numeric?(parseFloat(x)||0)-(parseFloat(y)||0):String(x).localeCompare(String(y)));
      });
      rows.forEach(r=>body.appendChild(r));
    });
  });
});
"""


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _num(value: float | None, spec: str = "{:.4f}") -> str:
    return '<span class="muted">n/a</span>' if value is None else spec.format(value)


def _section(title: str, body: str, subtitle: str = "", collapsed: bool = False) -> str:
    cls = " collapsed" if collapsed else ""
    sub = f'<span class="sub">{_e(subtitle)}</span>' if subtitle else ""
    return (
        f'<section class="{cls.strip()}"><div class="head"><h2>{_e(title)}</h2>'
        f'<div style="display:flex;gap:12px;align-items:center">{sub}'
        f'<span class="chev">&#9660;</span></div></div><div class="body">{body}</div></section>'
    )


def _cards(result: "AnalysisResult") -> str:
    counts = result.match.counts
    cards = [
        ("Images analysed", f"{len(result.evaluable):,}", f"{len(result.ranking.images):,} ranked"),
        ("Accuracy", _num(result.metrics.get("accuracy")), "at the configured threshold"),
        ("Precision", _num(result.metrics.get("precision")), "of what the model keeps"),
        ("Recall", _num(result.metrics.get("recall")), "of what you kept"),
        ("F1", _num(result.metrics.get("f1")), "precision/recall balance"),
        ("ROC AUC", _num(result.metrics.get("roc_auc")), "ranking quality"),
        ("False positives", f"{counts['false_positive']:,}", "kept but shouldn't be"),
        ("False negatives", f"{counts['false_negative']:,}", "rejected but shouldn't be"),
    ]
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="label">{_e(label)}</div>'
        f'<div class="value">{value}</div><div class="note">{_e(note)}</div></div>'
        for label, value, note in cards
    ) + "</div>"


def _metrics_table(result: "AnalysisResult") -> str:
    out = []
    for category in result.metrics.categories:
        rows = "".join(
            f"<tr><td>{_e(value.name)}</td>"
            f'<td class="num" data-v="{value.value if value.value is not None else -999}">{_e(value.rendered())}</td>'
            f'<td class="muted">{_e(value.description)}</td>'
            f'<td class="muted">{_e(value.detail)}</td></tr>'
            for value in result.metrics.in_category(category)
        )
        out.append(
            f"<h3 style='margin:16px 0 6px;font-size:15px'>{_e(category.title())}</h3>"
            f'<div class="scroll"><table><thead><tr><th>Metric</th><th class="num">Value</th>'
            f"<th>Meaning</th><th>Note</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    return "".join(out)


def _confusion(result: "AnalysisResult") -> str:
    matrix = result.confusion
    kept, rejected = matrix.tp + matrix.fn, matrix.fp + matrix.tn

    def cell(count: int, total: int, klass: str) -> str:
        share = f"{count / total * 100:.1f}%" if total else "-"
        return f'<td class="num {klass}"><strong>{count:,}</strong><br><span class="muted">{share}</span></td>'

    return (
        f'<p class="sub">Threshold {matrix.threshold:.3f}, {matrix.total:,} images. '
        "Percentages are row-normalised.</p>"
        "<table><thead><tr><th></th><th class='num'>model: KEEP</th>"
        "<th class='num'>model: REJECT</th></tr></thead><tbody>"
        f"<tr><th>you: KEPT</th>{cell(matrix.tp, kept, 'good')}{cell(matrix.fn, kept, 'bad')}</tr>"
        f"<tr><th>you: REJECTED</th>{cell(matrix.fp, rejected, 'bad')}{cell(matrix.tn, rejected, 'good')}</tr>"
        "</tbody></table>"
    )


def _thresholds(result: "AnalysisResult") -> str:
    sweep = result.sweep
    if sweep.is_worth_changing:
        banner = (
            f'<div class="sug warning"><div class="t">Recommended threshold: '
            f"{sweep.recommended.threshold:.3f}</div>"
            f'<div class="d">Raises {_e(sweep.optimize_for)} from '
            f"{_num(sweep.current.get(sweep.optimize_for))} to "
            f"{_num(sweep.recommended.get(sweep.optimize_for))} "
            f"({sweep.improvement:+.4f}).</div></div>"
        )
    else:
        banner = (
            f'<p class="sub">The current threshold ({sweep.current.threshold:.3f}) is already '
            f"near-optimal for {_e(sweep.optimize_for)}.</p>"
        )

    rows = []
    for point in sweep.points:
        marks = []
        if point.threshold == sweep.current.threshold:
            marks.append('<span class="pill muted">current</span>')
        if point.threshold == sweep.recommended.threshold:
            marks.append('<span class="pill warn">recommended</span>')
        rows.append(
            f'<tr><td class="num" data-v="{point.threshold}">{point.threshold:.3f}</td>'
            f'<td class="num" data-v="{point.precision or 0}">{_num(point.precision, "{:.3f}")}</td>'
            f'<td class="num" data-v="{point.recall or 0}">{_num(point.recall, "{:.3f}")}</td>'
            f'<td class="num" data-v="{point.f1 or 0}">{_num(point.f1, "{:.3f}")}</td>'
            f'<td class="num" data-v="{point.accuracy or 0}">{_num(point.accuracy, "{:.3f}")}</td>'
            f'<td class="num" data-v="{point.tp}">{point.tp:,}</td>'
            f'<td class="num" data-v="{point.fp}">{point.fp:,}</td>'
            f'<td class="num" data-v="{point.fn}">{point.fn:,}</td>'
            f"<td>{' '.join(marks)}</td></tr>"
        )
    return (
        banner
        + '<div class="scroll"><table><thead><tr><th class="num">Threshold</th>'
        '<th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th>'
        '<th class="num">Accuracy</th><th class="num">TP</th><th class="num">FP</th>'
        '<th class="num">FN</th><th></th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _charts(result: "AnalysisResult", charts_dir: Path, output_dir: Path) -> str:
    captions = {
        "confusion_matrix.png": "Agreement and disagreement at the current threshold",
        "roc_curve.png": "Ranking quality independent of threshold",
        "precision_recall_curve.png": "The precision/recall trade-off available to you",
        "threshold_curves.png": "How each metric responds to the threshold",
        "score_distribution.png": "Overlap between what you kept and rejected",
        "confidence_distribution.png": "How strongly the model commits",
        "calibration.png": "Whether stated probabilities can be trusted",
        "top_k_performance.png": "Quality of the top slice of the ranking",
        "ranking_distribution.png": "Where your keepers sit in the ordering",
        "comparison.png": "Metric changes between the two runs",
    }
    tiles = []
    for name, caption in captions.items():
        path = charts_dir / name
        if not path.exists():
            continue
        rel = path.relative_to(output_dir).as_posix()
        tiles.append(
            f'<div class="chart"><a href="{_e(rel)}" target="_blank">'
            f'<img src="{_e(rel)}" alt="{_e(caption)}" loading="lazy"></a>'
            f'<div class="cap">{_e(caption)}</div></div>'
        )
    return f'<div class="charts">{"".join(tiles)}</div>' if tiles else '<p class="sub">No charts rendered.</p>'


def _error_table(records, output_dir: Path, thumbs: dict[str, Path]) -> str:
    if not records:
        return '<p class="sub">None.</p>'
    rows = []
    for record in records:
        image = record.image
        thumb = thumbs.get(image.image_path)
        cell = ""
        if thumb is not None and thumb.exists():
            rel = thumb.relative_to(output_dir).as_posix()
            cell = f'<img src="{_e(rel)}" style="width:52px;height:52px;object-fit:cover;border-radius:5px" loading="lazy">'
        file_url = Path(image.image_path).as_uri() if Path(image.image_path).exists() else None
        name = (
            f'<a href="{_e(file_url)}" title="{_e(image.image_path)}">{_e(image.filename)}</a>'
            if file_url
            else f'<span title="{_e(image.image_path)}">{_e(image.filename)}</span>'
        )
        confidence = image.confidence
        rows.append(
            f"<tr><td>{cell}</td><td>{name}</td>"
            f'<td class="num" data-v="{image.score}">{image.score:.4f}</td>'
            f'<td class="num" data-v="{confidence or 0}">{_num(confidence, "{:.3f}")}</td>'
            f'<td class="num" data-v="{image.rank}">{image.rank:,}</td>'
            f'<td class="num" data-v="{record.rank_displacement}">{record.rank_displacement:,}</td>'
            f'<td><span class="pill {"bad" if image.is_error else "good"}">{_e(image.outcome.short)}</span></td>'
            f'<td class="muted">{"kept" if image.truth == 1 else "rejected"}</td></tr>'
        )
    return (
        '<div class="scroll"><table><thead><tr><th></th><th>File</th><th class="num">Score</th>'
        '<th class="num">Confidence</th><th class="num">Rank</th><th class="num">Displacement</th>'
        "<th>Outcome</th><th>You</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _suggestions(result: "AnalysisResult") -> str:
    if not result.suggestions:
        return '<p class="sub">Nothing flagged - no rule\'s evidence threshold was crossed.</p>'
    blocks = []
    for suggestion in result.suggestions:
        klass = {CRITICAL: "critical", WARNING: "warning"}.get(suggestion.severity, "")
        action = f'<div class="a"><strong>Action:</strong> {_e(suggestion.action)}</div>' if suggestion.action else ""
        blocks.append(
            f'<div class="sug {klass}"><div class="t">{_e(suggestion.title)}</div>'
            f'<div class="d">{_e(suggestion.detail)}</div>'
            f'<div class="e">evidence: {_e(suggestion.evidence)}</div>{action}</div>'
        )
    return "".join(blocks)


def _comparison(result: "AnalysisResult") -> str:
    comparison = result.comparison
    if comparison is None:
        return ""
    rows = []
    for delta in comparison.deltas:
        if delta.delta is None:
            continue
        klass = "good" if delta.improved else ("bad" if delta.improved is False else "muted")
        rows.append(
            f"<tr><td>{_e(delta.name)}</td>"
            f'<td class="num">{_num(delta.baseline)}</td>'
            f'<td class="num">{_num(delta.candidate)}</td>'
            f'<td class="num {klass}" data-v="{delta.delta}">{delta.delta:+.4f}</td></tr>'
        )
    changed = "".join(
        f"<tr><td>{_e(change.filename)}</td><td>{_e(change.baseline_outcome)}</td>"
        f"<td>{_e(change.candidate_outcome)}</td>"
        f'<td class="num">{change.score_delta:+.4f}</td>'
        f'<td class="num">{change.rank_delta:+,}</td></tr>'
        for change in (comparison.broken[:25] + comparison.fixed[:25])
    )
    return (
        f'<p><span class="pill">{_e(comparison.verdict)}</span></p>'
        f'<div class="scroll"><table><thead><tr><th>Metric</th>'
        f'<th class="num">{_e(comparison.baseline_label)}</th>'
        f'<th class="num">{_e(comparison.candidate_label)}</th>'
        f'<th class="num">Change</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        f"<h3 style='margin:18px 0 6px;font-size:15px'>Images that changed</h3>"
        f'<div class="scroll"><table><thead><tr><th>File</th><th>Was</th><th>Now</th>'
        f'<th class="num">Score change</th><th class="num">Rank change</th></tr></thead>'
        f"<tbody>{changed}</tbody></table></div>"
    )


def build_html(result: "AnalysisResult", thumbs: dict[str, Path] | None = None) -> str:
    config = result.config
    output_dir = config.output_dir
    thumbs = thumbs or {}
    errors = result.errors

    sections = [
        _section("Recommendations", _suggestions(result), f"{len(result.suggestions)} item(s)"),
        _section("Confusion matrix", _confusion(result)),
        _section("Charts", _charts(result, config.charts_dir, output_dir)),
        _section("All metrics", _metrics_table(result), f"{len(result.metrics.values)} metrics", collapsed=True),
        _section("Threshold analysis", _thresholds(result), f"optimising {config.optimize_for}", collapsed=True),
        _section(
            "False positives",
            _error_table(errors.false_positives, output_dir, thumbs),
            f"{len(errors.false_positives)} shown",
            collapsed=True,
        ),
        _section(
            "False negatives",
            _error_table(errors.false_negatives, output_dir, thumbs),
            f"{len(errors.false_negatives)} shown",
            collapsed=True,
        ),
        _section(
            "Borderline images",
            _error_table(errors.borderline, output_dir, thumbs),
            f"{len(errors.borderline)} in [{config.borderline_low}, {config.borderline_high}]",
            collapsed=True,
        ),
        _section(
            "Largest ranking disagreements",
            _error_table(errors.largest_rank_disagreements, output_dir, thumbs),
            collapsed=True,
        ),
        _section("Dataset matching", f"<pre style='white-space:pre-wrap;font-size:13px'>"
                 f"{_e(result.match.summary())}</pre>", collapsed=True),
    ]
    if result.comparison is not None:
        sections.insert(1, _section("Model comparison", _comparison(result), result.comparison.verdict))

    sheets = sorted(config.sheets_dir.glob("*.png")) if config.sheets_dir.exists() else []
    if sheets:
        links = "".join(
            f'<li><a href="{_e(p.relative_to(output_dir).as_posix())}" target="_blank">{_e(p.stem)}</a></li>'
            for p in sheets
        )
        sections.append(_section("Contact sheets", f"<ul>{links}</ul>", f"{len(sheets)} sheet(s)", collapsed=True))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(config.report_title)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<header><div><h1>{_e(config.report_title)}</h1>
<div class="sub">{_e(result.generated_at)} &middot; {len(result.evaluable):,} images with ground truth
&middot; threshold {config.threshold:.3f} &middot; ranking: {_e(config.ranking_path.name)}</div></div>
<button id="theme-btn">Dark mode</button></header>
{_cards(result)}
{''.join(sections)}
<footer>Generated by picklikeme analyze &middot; read-only: no model, cache or source image was modified</footer>
</div><script>{JS}</script></body></html>"""


def write_html_report(result: "AnalysisResult", thumbs: dict[str, Path] | None = None) -> Path:
    output_dir = result.config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse whatever the contact-sheet pass already generated; never decode
    # anything a second time just to fill a table.
    if thumbs is None:
        thumbs = {}
        thumbnails_dir = result.config.thumbnails_dir
        if thumbnails_dir.exists():
            from ..contactsheets import _thumbnail_cache_path

            for record in (
                result.errors.false_positives
                + result.errors.false_negatives
                + result.errors.borderline
                + result.errors.largest_rank_disagreements
            ):
                candidate = _thumbnail_cache_path(
                    thumbnails_dir, record.image_path, result.config.thumbnail_size
                )
                if candidate.exists():
                    thumbs[record.image_path] = candidate

    path = output_dir / "report.html"
    path.write_text(build_html(result, thumbs), encoding="utf-8")
    logger.info("HTML report: %s", path)
    return path
