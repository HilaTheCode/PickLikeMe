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

from ...config import cli_prefix
from ..annotations import ANNOTATION_FIELD_LABELS, ANNOTATION_FIELDS
from ..links import asset_url, source_api_url, source_file_uri
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
.tile img{width:100%;aspect-ratio:1;object-fit:contain;display:block;background:#0b0f17}
.tile .m{padding:6px 8px;font-size:11.5px;line-height:1.4}
.tile .fn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
a{color:var(--accent)}
footer{color:var(--muted);font-size:12.5px;text-align:center;margin-top:34px}

/* --- False-negative annotation panels --- */
.fn{border:1px solid var(--border);border-radius:10px;margin-bottom:12px;background:var(--panel-2)}
.fn-top{display:flex;gap:13px;padding:11px 13px;align-items:flex-start}
/* Large enough to judge a detector box on a whole-frame preview; `contain`
   because the preview is already letterboxed and must never be cropped again. */
.fn-top img{width:232px;height:232px;object-fit:contain;border-radius:7px;background:#0b0f17;flex:none}
@media (max-width:640px){.fn-top img{width:160px;height:160px}}
.fn-meta{flex:1;min-width:0}
.fn-name{font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fn-nums{color:var(--muted);font-size:12.5px;margin-top:3px;font-variant-numeric:tabular-nums}
.fn-tags{margin-top:7px;display:flex;flex-wrap:wrap;gap:5px}
.tag{background:var(--accent);color:#fff;border:0;padding:2px 9px;border-radius:999px;font-size:11.5px;
font-weight:600}
.fn-actions{display:flex;flex-direction:column;gap:6px;flex:none}
.fn-note{white-space:pre-wrap;color:var(--text);font-size:13px;margin-top:7px;
border-left:2px solid var(--border);padding-left:9px}
.fn-editor{display:none;padding:0 13px 13px}
.fn.editing .fn-editor{display:block}
.fn.editing{border-color:var(--accent)}
.field-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;
margin:11px 0 4px}
.field{display:flex;flex-direction:column;gap:5px}
.field-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;
font-weight:600}
.field select{width:100%}
.tag.field-tag{background:var(--panel);color:var(--text);border:1px solid var(--border);
font-weight:600}
.fn-legacy{margin-top:7px;font-size:12px;color:var(--muted);font-style:italic;
border-left:2px dashed var(--border);padding-left:9px}
.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:9px}
.status{font-size:12.5px;color:var(--muted)}
.status.saved{color:var(--good)}.status.error{color:var(--bad)}
.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.offline{background:var(--panel-2);border:1px dashed var(--warn);border-radius:9px;padding:11px 14px;
margin-bottom:14px;font-size:13.5px}
.filters{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:14px;
padding-bottom:13px;border-bottom:1px solid var(--border)}
select,.filters select{font:inherit;font-size:13px;padding:6px 9px;border-radius:7px;
border:1px solid var(--border);background:var(--panel);color:var(--text)}
.freq{display:flex;align-items:center;gap:10px;margin-bottom:5px}
.freq .n{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12.5px;width:44px;text-align:right}
.freq .bar{height:15px;background:var(--accent);border-radius:4px;min-width:2px}
.freq .lbl{font-size:13px}
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

// ---------------------------------------------------------------------------
// Annotations - false negatives and false positives, identically.
//
// Annotations are written by the photographer only. Nothing here derives,
// suggests or pre-fills a category - the page renders what the database holds
// and posts back exactly what was selected. `.fn` names one annotation panel
// (kept from when this was false-negative-only); `.ann-scope` wraps one
// category's filter bar and panels, so the two categories filter and count
// independently while sharing every other function below.
//
// Saving needs the local server (a file:// page cannot reach SQLite), so the
// page probes /api/health once: online enables Save, offline shows the existing
// annotations read-only and says how to enable editing.
// ---------------------------------------------------------------------------
const PLM = {online:false, annotations: window.PLM_ANNOTATIONS || {}};

async function plmProbe(){
  try{
    const r = await fetch('api/health',{cache:'no-store'});
    if(!r.ok) return false;
    const j = await r.json();
    return !!j.ok;
  }catch(e){ return false; }
}

// The three diagnostic fields, and their display labels. Kept in one place so
// save, render and filter all agree; the server validates the same set.
const PLM_FIELDS = ['crop_quality','image_quality','agree_with_model_decision'];
const PLM_LABELS = {
  crop_quality:'Crop Quality',
  image_quality:'Image Quality',
  agree_with_model_decision:'Agree with Model Decision'
};
const plmAttr = f => 'data-'+f.replace(/_/g,'-');

function plmRenderTags(el, annotation){
  const tags = el.querySelector('.fn-tags');
  const values = {};
  PLM_FIELDS.forEach(f=>{ values[f] = (annotation && annotation[f]) || ''; });
  if(tags){
    const html = PLM_FIELDS.filter(f=>values[f])
      .map(f=>`<span class="tag field-tag">${plmEsc(PLM_LABELS[f])}: ${plmEsc(values[f])}</span>`)
      .join('');
    tags.innerHTML = html || '<span class="status">not yet annotated</span>';
  }
  PLM_FIELDS.forEach(f=>{ el.setAttribute(plmAttr(f), values[f]); });
  el.dataset.annotated = PLM_FIELDS.some(f=>values[f]) ? '1' : '0';
}

function plmEsc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}

function plmToggleEdit(el){ el.classList.toggle('editing'); }

async function plmSave(el){
  const status = el.querySelector('.status');
  const path = el.dataset.path;
  const payload = {image_path:path};
  el.querySelectorAll('select.ann-field').forEach(sel=>{
    payload[sel.dataset.field] = sel.value;   // '' means "not set"
  });

  if(!PLM.online){
    status.textContent = 'Read-only: see the banner above for the command that enables Save.';
    status.className = 'status error';
    return;
  }
  status.textContent = 'Saving...'; status.className='status';
  try{
    const r = await fetch('api/annotations',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if(!r.ok || j.error){ throw new Error(j.error || ('HTTP '+r.status)); }
    plmRenderTags(el, j.annotation);
    status.textContent = j.deleted ? 'Annotation cleared.' : 'Saved.';
    status.className = 'status saved';
    el.classList.remove('editing');
    // Each category (false negatives, false positives) has its own filter bar
    // and count, so only the panel's own scope needs refreshing.
    const scope = el.closest('.ann-scope');
    if(scope){ plmUpdateCounts(scope); plmApplyFilters(scope); }
  }catch(e){
    status.textContent = 'Save failed: '+e.message;
    status.className = 'status error';
  }
}

function plmUpdateCounts(scope){
  const all=[...scope.querySelectorAll('.fn')];
  const done=all.filter(e=>e.dataset.annotated==='1').length;
  const el=scope.querySelector('.ann-coverage');
  if(el) el.textContent = `${done} of ${all.length} annotated`;
}

// Filtering: by annotated / not annotated, and by an exact value per field.
// Scoped to one `.ann-scope` container so the false-negative and false-positive
// filter bars never affect each other's panels.
function plmApplyFilters(scope){
  const mode=scope.querySelector('.f-state')?.value||'all';
  // One optional value filter per field; blank means "any".
  const wanted={};
  scope.querySelectorAll('select.f-field').forEach(sel=>{
    if(sel.value) wanted[sel.dataset.field]=sel.value;
  });
  let shown=0;
  scope.querySelectorAll('.fn').forEach(el=>{
    const annotated=el.dataset.annotated==='1';
    let ok=true;
    if(mode==='annotated'&&!annotated) ok=false;
    if(mode==='unannotated'&&annotated) ok=false;
    if(ok){
      for(const f of Object.keys(wanted)){
        if((el.getAttribute(plmAttr(f))||'') !== wanted[f]){ ok=false; break; }
      }
    }
    el.style.display = ok ? '' : 'none';
    if(ok) shown++;
  });
  const out=scope.querySelector('.f-count');
  if(out) out.textContent = `${shown} shown`;
}

document.addEventListener('DOMContentLoaded', async ()=>{
  document.querySelectorAll('.fn').forEach(el=>{
    el.querySelector('.btn-edit')?.addEventListener('click',()=>plmToggleEdit(el));
    el.querySelector('.btn-save')?.addEventListener('click',()=>plmSave(el));
    el.querySelector('.btn-cancel')?.addEventListener('click',()=>el.classList.remove('editing'));
  });
  document.querySelectorAll('.ann-scope').forEach(scope=>{
    scope.querySelector('.f-state')?.addEventListener('change',()=>plmApplyFilters(scope));
    scope.querySelectorAll('select.f-field').forEach(sel=>
      sel.addEventListener('change',()=>plmApplyFilters(scope)));
    plmUpdateCounts(scope);
  });

  PLM.online = await plmProbe();
  const banners=[...document.querySelectorAll('.ann-offline')];
  if(PLM.online){
    banners.forEach(b=>b.style.display='none');
    // Browsers refuse to navigate from a served page to a local-file URL, so a
    // served report must reach originals through the server. Rewrite them once.
    document.querySelectorAll('a[data-source]').forEach(a=>{
      a.href='source?path='+encodeURIComponent(a.dataset.source);
      a.target='_blank';
    });
    // Refresh from the database in case it changed since the report was written.
    try{
      const r=await fetch('api/annotations',{cache:'no-store'});
      const j=await r.json();
      // Match on image_hash first: it is the identity and survives the file
      // having moved since the annotation was written. Path is only a fallback
      // for panels that have never been annotated (and so carry no hash).
      const byHash={}, byPath={};
      (j.annotations||[]).forEach(a=>{byHash[a.image_hash]=a; byPath[a.original_path]=a;});
      document.querySelectorAll('.fn').forEach(el=>{
        const a=byHash[el.dataset.hash]||byPath[el.dataset.path];
        if(a){
          el.dataset.hash=a.image_hash;
          plmRenderTags(el,a);
          el.querySelectorAll('select.ann-field').forEach(sel=>{
            sel.value=a[sel.dataset.field]||'';
          });
        }
      });
      document.querySelectorAll('.ann-scope').forEach(scope=>plmUpdateCounts(scope));
    }catch(e){}
  }else{
    banners.forEach(b=>b.style.display='');
  }
});

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


def _source_link(image_path: str, label: str) -> str:
    """Anchor to an original image, usable in both report modes.

    `href` is a file:// URI so the link works when the report is opened from
    disk. `data-source` carries the absolute path so the served page can rewrite
    the href to the server's /source endpoint - browsers block http:// -> file://
    navigation outright, which is why a served report's links appeared broken.
    """
    uri = source_file_uri(image_path)
    if uri is None:
        return f'<span title="{_e(image_path)}" class="muted">{_e(label)}</span>'
    return (
        f'<a href="{_e(uri)}" data-source="{_e(image_path)}" '
        f'title="{_e(image_path)}">{_e(label)}</a>'
    )


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
        rel = asset_url(path, output_dir)
        if rel is None:
            continue
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
            rel = asset_url(thumb, output_dir)
            if rel:
                cell = (
                    f'<img src="{_e(rel)}" style="width:96px;height:96px;object-fit:contain;'
                    'border-radius:5px;background:#0b0f17" loading="lazy">'
                )
        name = _source_link(image.image_path, image.filename)
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


def _annotated_thumb(result: "AnalysisResult", image_path: str) -> Path | None:
    """The detector-box overlay for an image, if one was rendered.

    Generic - applies to any image in the report, not only false negatives.
    Whether an overlay exists at all depends on whether a detection record with
    boxes was resolved for that image (see analysis._attach_detections).
    """
    from ..contactsheets import annotated_thumbnail_path

    config = result.config
    candidate = annotated_thumbnail_path(config.thumbnails_dir, image_path, config.thumbnail_size)
    return candidate if candidate.exists() else None


DETECTOR_BOX_LEGEND = (
    '<p class="sub" style="margin:0 0 16px">Thumbnails throughout this report show the whole '
    'frame, with what the detector found drawn on it when known: '
    '<span style="color:#10b981;font-weight:650">solid green</span> is the box that became the '
    'crop the model actually scored, '
    '<span style="color:#facc15;font-weight:650">dashed amber</span> are detections it passed over, '
    'and <span style="color:#ef4444;font-weight:650">red</span> means nothing was detected (the '
    "model saw the whole frame). An image with no overlay has no resolved detection record.</p>"
)


def _field_select(field_name: str, values: list[str], current: str | None) -> str:
    """One labelled dropdown for a fixed-vocabulary annotation field.

    A `<select>` rather than radios: three fields of four options each would be
    twelve radio targets per panel, which crowds a list meant to be scanned
    quickly. The blank option is what "not answered yet" looks like, and is what
    a photographer picks to clear a field.
    """
    options = '<option value="">- not set -</option>' + "".join(
        f'<option value="{_e(value)}"{" selected" if value == current else ""}>{_e(value)}</option>'
        for value in values
    )
    return (
        f'<label class="field"><span class="field-label">{_e(ANNOTATION_FIELD_LABELS[field_name])}</span>'
        f'<select class="ann-field" data-field="{_e(field_name)}">{options}</select></label>'
    )


def _annotation_panels(
    records: list,
    result: "AnalysisResult",
    thumbs: dict[str, Path],
    *,
    scope: str,
    decision_label: str,
    empty_message: str,
) -> str:
    """Annotation panels for one outcome category - false negatives or false
    positives - identical fields, vocabulary and markup for both, so the two
    are directly comparable. The detector-box overlay drawn on the thumbnail is
    not scoped to either category; it is whatever `thumbs` already carries,
    which build_html() has pre-merged with overlays for every image in the
    report.

    Every field renders unset unless the database already holds a value for
    that image - nothing is ever pre-selected on the model's behalf.

    Wrapped in a `.ann-scope` container so its filter bar only ever filters its
    own panels: this function is called once per category, and the two sets of
    filters must not interfere with each other.
    """
    if not records:
        return f'<p class="sub">{_e(empty_message)}</p>'

    vocabularies = {name: list(values) for name, values in ANNOTATION_FIELDS.items()}
    output_dir = result.config.output_dir

    field_filters = "".join(
        f'<label>{_e(ANNOTATION_FIELD_LABELS[name])} '
        f'<select class="f-field" data-field="{_e(name)}">'
        '<option value="">any</option>'
        + "".join(f'<option value="{_e(v)}">{_e(v)}</option>' for v in values)
        + "</select></label>"
        for name, values in vocabularies.items()
    )
    filters = (
        '<div class="filters">'
        '<label>Show <select class="f-state">'
        '<option value="all">all</option>'
        '<option value="unannotated">not annotated</option>'
        '<option value="annotated">annotated only</option>'
        "</select></label>"
        + field_filters
        + '<span class="status f-count"></span>'
        '<span class="status ann-coverage"></span>'
        "</div>"
    )

    offline = (
        '<div class="offline ann-offline" style="display:none">'
        "<strong>Read-only.</strong> This report was opened directly from disk, so Save cannot reach "
        "the annotation database. Existing annotations are shown. To edit, run "
        f'<code>{_e(cli_prefix())} annotate --output "{_e(output_dir)}"</code> and open the address it prints.'
        "</div>"
    )

    panels = []
    for record in records:
        image = record.image
        path = image.image_path
        detection = (result.detections or {}).get(path)
        annotation = result.annotations.get(path)
        values = {
            name: (getattr(annotation, name) if annotation else None) for name in ANNOTATION_FIELDS
        }
        answered = [v for v in values.values() if v]

        # `thumbs` is pre-merged with overlays in build_html(), so this is
        # already the annotated copy when one exists.
        thumb = thumbs.get(path)
        thumb_html = '<img alt="no preview" style="display:flex">'
        if thumb is not None and thumb.exists():
            rel = asset_url(thumb, output_dir)
            if rel:
                thumb_html = f'<img src="{_e(rel)}" alt="{_e(image.filename)}" loading="lazy">'
        name = _source_link(path, image.filename)
        boxes = ""
        if detection is not None and detection.boxes:
            selected = detection.selected
            parts = [
                f"detector: {len(detection.boxes)} box"
                + ("es" if len(detection.boxes) != 1 else "")
            ]
            if selected is not None:
                parts.append(f"cropped {selected.class_name} @ {selected.score:.2f}")
            else:
                parts.append("nothing detected - full frame used")
            boxes = f' &middot; {_e(" &middot; ".join(parts))}'.replace("&amp;middot;", "&middot;")
        moved = (
            ' <span class="pill warn" title="matched by content identity; the file is no longer '
            'at the path recorded with the annotation">followed a move</span>'
            if annotation and annotation.relocated
            else ""
        )
        captured = (
            f" &middot; shot {_e(annotation.capture_datetime)}"
            if annotation and annotation.capture_datetime
            else ""
        )
        tags = (
            "".join(
                f'<span class="tag field-tag">{_e(ANNOTATION_FIELD_LABELS[field_name])}: '
                f"{_e(value)}</span>"
                for field_name, value in values.items()
                if value
            )
            or '<span class="status">not yet annotated</span>'
        )
        # Pre-redesign content, shown read-only so a record's history is
        # visible rather than silently dropped. Never editable, never counted.
        legacy = ""
        if annotation is not None and annotation.has_legacy_content:
            bits = []
            if annotation.legacy_primary_failure_cause:
                bits.append(f"cause: {annotation.legacy_primary_failure_cause}")
            if annotation.legacy_categories:
                bits.append(", ".join(annotation.legacy_categories))
            if annotation.legacy_notes.strip():
                bits.append(annotation.legacy_notes.strip())
            legacy = (
                '<div class="fn-legacy" title="recorded before the current three fields; '
                'kept for reference, not counted">'
                f"earlier annotation &middot; {_e(' &middot; '.join(bits))}</div>"
            ).replace("&amp;middot;", "&middot;")

        selects = "".join(
            _field_select(field_name, vocabularies.get(field_name, []), values[field_name])
            for field_name in ANNOTATION_FIELDS
        )

        panels.append(
            f'<div class="fn" data-path="{_e(path)}" '
            f'data-hash="{_e(annotation.image_hash if annotation else "")}" '
            + "".join(
                f'data-{field_name.replace("_", "-")}="{_e(values[field_name] or "")}" '
                for field_name in ANNOTATION_FIELDS
            )
            + f'data-annotated="{"1" if answered else "0"}">'
            f'<div class="fn-top">{thumb_html}'
            f'<div class="fn-meta"><div class="fn-name">{name}{moved}</div>'
            f'<div class="fn-nums">score {image.score:.4f} &middot; confidence '
            f'{_num(image.confidence, "{:.3f}")} &middot; rank {image.rank:,} &middot; '
            f"displaced {record.rank_displacement:,} &middot; {_e(decision_label)}{captured}{boxes}</div>"
            f'<div class="fn-tags">{tags}</div>{legacy}'
            f"</div>"
            f'<div class="fn-actions"><button class="btn-edit">Edit</button></div></div>'
            f'<div class="fn-editor">'
            f'<div class="sub">Your diagnosis - judge the crop and the photograph separately.</div>'
            f'<div class="field-grid">{selects}</div>'
            f'<div class="row"><button class="btn-save primary">Save</button>'
            f'<button class="btn-cancel">Cancel</button>'
            f'<span class="status"></span></div>'
            f"</div></div>"
        )

    return (
        f'<div class="ann-scope" data-scope="{_e(scope)}">'
        + filters
        + offline
        + "".join(panels)
        + "</div>"
    )


def _annotation_summary(summary, *, category_label: str, section_hint: str) -> str:
    """One category's summary section (false negatives, or false positives) -
    same layout for both, so the numbers can be compared side by side.
    """
    if summary is None:
        return '<p class="sub">Annotation database unavailable.</p>'

    coverage = (
        f"{summary.annotated:,} of {summary.total_images:,} annotated"
        + (f" ({summary.coverage * 100:.1f}%)" if summary.coverage is not None else "")
    )
    # The headline number: how often you looked at a mistake and disagreed with
    # the model's call. None when nobody has answered that field yet.
    agree_counts = dict(summary.field_counts.get("agree_with_model_decision") or [])
    disagreements = agree_counts.get("No", 0) if agree_counts else None
    cards = (
        '<div class="cards">'
        f'<div class="card"><div class="label">{_e(category_label)}</div>'
        f'<div class="value">{summary.total_images:,}</div>'
        f'<div class="note">in this analysis</div></div>'
        f'<div class="card"><div class="label">Annotated</div>'
        f'<div class="value">{summary.annotated:,}</div><div class="note">{_e(coverage)}</div></div>'
        f'<div class="card"><div class="label">Not annotated</div>'
        f'<div class="value">{summary.unannotated_count:,}</div>'
        f'<div class="note">candidates for review</div></div>'
        f'<div class="card"><div class="label">Knowledge base</div>'
        f'<div class="value">{summary.total_in_database:,}</div>'
        f'<div class="note">records across all runs</div></div>'
        + (
            f'<div class="card"><div class="label">Disagree with model</div>'
            f'<div class="value">{disagreements:,}</div>'
            f'<div class="note">of {summary.annotated:,} annotated</div></div>'
            if disagreements is not None
            else ""
        )
        + "</div>"
    )

    # One breakdown per field: the fields are independent judgements, so they get
    # independent charts rather than being pooled into a single tag frequency.
    blocks = []
    for field_name, label in ANNOTATION_FIELD_LABELS.items():
        counts = summary.field_counts.get(field_name) or []
        if not counts:
            continue
        biggest = counts[0][1]
        bars = "".join(
            f'<div class="freq"><span class="n">{count:,}</span>'
            f'<span class="bar" style="width:{count / biggest * 260:.0f}px"></span>'
            f'<span class="lbl">{_e(name)}</span></div>'
            for name, count in counts
        )
        blocks.append(f"<h3 style='margin:18px 0 8px;font-size:15px'>{_e(label)}</h3>{bars}")
    if blocks:
        frequencies = "".join(blocks)
    else:
        frequencies = (
            f"<p class=\"sub\">No annotations yet. Open the {_e(section_hint)} section, click "
            "<strong>Edit</strong> on an image and record your diagnosis.</p>"
        )

    combinations = ""
    if summary.combination_counts:
        rows = "".join(
            f'<tr><td class="num">{count:,}</td>'
            + "".join(f"<td>{_e(value)}</td>" for value in combo)
            + "</tr>"
            for combo, count in summary.combination_counts
        )
        headers = "".join(f"<th>{_e(label)}</th>" for label in ANNOTATION_FIELD_LABELS.values())
        combinations = (
            "<h3 style='margin:18px 0 6px;font-size:15px'>Most common combinations</h3>"
            f'<table><thead><tr><th class="num">Count</th>{headers}</tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )

    legacy_note = ""
    if summary.with_legacy_content:
        legacy_note = (
            '<div class="sug"><div class="t">'
            f"{summary.with_legacy_content:,} record(s) carry pre-redesign notes or categories</div>"
            '<div class="d">Kept verbatim and shown read-only beside the image. They are not '
            "counted above: mapping a free-text note onto a fixed field would invent a judgement "
            "you never made. Re-annotate those images to bring them into the counts.</div></div>"
        )

    def _recent_tags(a) -> str:
        return "".join(
            f'<span class="tag field-tag">{_e(value)}</span> '
            for value in (a.crop_quality, a.image_quality, a.agree_with_model_decision)
            if value
        )

    recent = ""
    if summary.recent:
        rows = "".join(
            f"<tr><td>{_e(a.updated_at[:16])}</td><td>{_e(a.filename)}</td>"
            f"<td>{_recent_tags(a)}</td></tr>"
            for a in summary.recent
        )
        recent = (
            "<h3 style='margin:18px 0 6px;font-size:15px'>Recently annotated</h3>"
            '<div class="scroll"><table><thead><tr><th>Updated</th><th>File</th>'
            f"<th>Diagnosis</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )

    unannotated = ""
    if summary.unannotated:
        items = "".join(f"<li>{_e(Path(p).name)}</li>" for p in summary.unannotated[:40])
        more = (
            f"<p class='sub'>... and {len(summary.unannotated) - 40:,} more</p>"
            if len(summary.unannotated) > 40
            else ""
        )
        unannotated = (
            f"<h3 style='margin:18px 0 6px;font-size:15px'>Not yet annotated "
            f"({summary.unannotated_count:,})</h3><ul>{items}</ul>{more}"
        )

    # Identity conditions worth surfacing rather than hiding.
    notices = ""
    if summary.unresolved:
        items = "".join(
            f"<li>{_e(item.filename)} &mdash; <span class='muted'>{_e(item.reason)}</span></li>"
            for item in summary.unresolved[:20]
        )
        notices += (
            '<div class="sug critical"><div class="t">'
            f"{len(summary.unresolved):,} image(s) have no establishable identity</div>"
            '<div class="d">These cannot be annotated: without a content hash an annotation could '
            "only be attached to a guess, and a wrong diagnosis is worse than a missing one. "
            f"Make the files readable and re-run.</div><ul>{items}</ul></div>"
        )
    if summary.relocated:
        notices += (
            '<div class="sug"><div class="t">'
            f"{summary.relocated:,} annotation(s) followed a moved file</div>"
            '<div class="d">Matched by content identity even though the file is no longer at the '
            "path recorded when it was annotated - which is the point of hashing content rather "
            "than location.</div></div>"
        )
    if summary.migration is not None and summary.migration.ran:
        notices += (
            '<div class="sug"><div class="t">Identity migration ran</div>'
            f'<div class="d">{_e(summary.migration.render())}</div></div>'
        )
    if summary.pending_migration:
        notices += (
            '<div class="sug warning"><div class="t">'
            f"{len(summary.pending_migration):,} older path-keyed annotation(s) not yet re-keyed</div>"
            '<div class="d">Their files were not found, so they are kept as-is and retried on every '
            "run. Nothing has been lost.</div></div>"
        )

    footer = (
        f'<p class="sub" style="margin-top:18px">Database: <code>{_e(summary.database_path)}</code>. '
        "Annotations are keyed on the image's content hash, so a diagnosis follows the image "
        "through renames, reorganisation and moves between drives. They are recorded by you, never "
        "generated, and never affect any metric in this report.</p>"
    )
    return (
        cards
        + notices
        + legacy_note
        + frequencies
        + combinations
        + recent
        + unannotated
        + footer
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
    errors = result.errors

    # Every thumbnail in the report prefers its detector-box overlay when one
    # was rendered, computed once here so every section below (error tables,
    # annotation panels) reads a plain dict and needs no overlay awareness of
    # its own.
    plain_thumbs = thumbs or {}
    thumbs = {path: (_annotated_thumb(result, path) or plain) for path, plain in plain_thumbs.items()}
    has_any_overlay = any(record.boxes for record in (result.detections or {}).values())

    sections = [
        _section("Recommendations", _suggestions(result), f"{len(result.suggestions)} item(s)"),
        _section("Confusion matrix", _confusion(result)),
        _section("Charts", _charts(result, config.charts_dir, output_dir)),
        _section("All metrics", _metrics_table(result), f"{len(result.metrics.values)} metrics", collapsed=True),
        _section("Threshold analysis", _thresholds(result), f"optimising {config.optimize_for}", collapsed=True),
        _section(
            "False negatives - annotate why they were missed",
            _annotation_panels(
                errors.false_negatives,
                result,
                thumbs,
                scope="false_negative",
                decision_label="you kept it",
                empty_message="No false negatives - nothing to annotate.",
            ),
            f"{len(errors.false_negatives)} shown",
        ),
        _section(
            "False negative summary",
            _annotation_summary(
                result.annotation_summary,
                category_label="False negatives",
                section_hint="False negatives",
            ),
            (
                f"{result.annotation_summary.annotated} annotated"
                if result.annotation_summary
                else "no database"
            ),
        ),
        _section(
            "False positives - annotate why they were kept",
            _annotation_panels(
                errors.false_positives,
                result,
                thumbs,
                scope="false_positive",
                decision_label="you rejected it",
                empty_message="No false positives - nothing to annotate.",
            ),
            f"{len(errors.false_positives)} shown",
        ),
        _section(
            "False positive summary",
            _annotation_summary(
                result.fp_annotation_summary,
                category_label="False positives",
                section_hint="False positives",
            ),
            (
                f"{result.fp_annotation_summary.annotated} annotated"
                if result.fp_annotation_summary
                else "no database"
            ),
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
            f'<li><a href="{_e(asset_url(p, output_dir))}" target="_blank">{_e(p.stem)}</a></li>'
            for p in sheets
            if asset_url(p, output_dir)
        )
        sections.append(_section("Contact sheets", f"<ul>{links}</ul>", f"{len(sheets)} sheet(s)", collapsed=True))

    # Inlined so a report opened from disk still shows the annotations that
    # existed when it was written; the served page refreshes from the API.
    inlined_annotations = json.dumps(
        {path: annotation.as_dict() for path, annotation in result.annotations.items()}
    )

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
{DETECTOR_BOX_LEGEND if has_any_overlay else ''}
{''.join(sections)}
<footer>Generated by picklikeme analyze &middot; read-only: no model, cache or source image was modified
&middot; annotations are yours, never generated, and never affect any metric</footer>
</div><script>window.PLM_ANNOTATIONS={inlined_annotations};</script>
<script>{JS}</script></body></html>"""


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
