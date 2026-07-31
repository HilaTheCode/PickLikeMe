"""The review page: one self-contained document, generated per request.

Follows the same constraints as the analysis report (`analyzer/reports/html.py`):
no CDN, no framework, all CSS and JS inline, every fetch URL relative so the
page works on whatever port the server happened to get.

Unlike the report, this page is generated on demand rather than written to
disk. There is nothing to keep: the gallery is a live view of a folder, a
ranking and a database, and writing a stale copy into the photographer's shoot
would be a liability rather than an artifact.

The document ships empty and fills itself from `api/review/state`, so the
initial response is instant however many thousands of images the folder holds;
thumbnails then load lazily as the browser scrolls them into view.

UX model (see session.py for the backend side of this): the AI ranking is
read-only metadata, shown as a small "AI Keep/Reject" suggestion chip. The
photographer's own review status is always exactly one of Keep, Reject or
Neutral, set explicitly - never inferred from the ranking, never a toggle
that silently falls back to whatever the model would have picked.
"""

from __future__ import annotations

import json

from ..analyzer.annotations import (
    REVIEW_REASON_BAD_QUALITY,
    REVIEW_REASON_CLEAR_EYES_SEEN,
    REVIEW_REASON_EYES_NOT_SEEN,
    REVIEW_REASON_GOOD_QUALITY,
    REVIEW_REASON_OTHER,
)
from .session import REVIEW_STATUS_KEEP, REVIEW_STATUS_NEUTRAL, REVIEW_STATUS_REJECT

# Keep-percentage presets. 25 is DEFAULT_SELECTION_PERCENTAGE, so the default
# is always one of the buttons rather than an invisible custom value.
KEEP_PRESETS = (5, 10, 20, 25, 35)

# Badge/filter text for the photographer's own review status - always exactly
# these three words, everywhere the status appears (card badge, filter
# buttons, bulk action bar, Lightbox). One vocabulary, never "Selected" in one
# place and "Keep" in another.
REVIEW_STATUS_LABELS = {
    REVIEW_STATUS_KEEP: "Keep",
    REVIEW_STATUS_REJECT: "Reject",
    REVIEW_STATUS_NEUTRAL: "Neutral",
}

# Why a Keep/Reject overrides the model - optional, shown as a dropdown next
# to Keep/Reject/Neutral in the Lightbox. Fixed, like the status itself: see
# REVIEW_REASONS in analyzer/annotations.py for why this isn't config-driven.
# Meaningless for Neutral - "no opinion" needs no justification.
REASON_LABELS = {
    REVIEW_REASON_EYES_NOT_SEEN: "Eyes not seen",
    REVIEW_REASON_CLEAR_EYES_SEEN: "Clear Eyes Seen",
    REVIEW_REASON_GOOD_QUALITY: "Overall good image quality",
    REVIEW_REASON_BAD_QUALITY: "Overall bad image quality",
    REVIEW_REASON_OTHER: "Other",
}

CSS = """
*{box-sizing:border-box}
:root{--bg:#f8fafc;--panel:#fff;--panel-2:#f1f5f9;--text:#0f172a;--muted:#64748b;
--border:#e2e8f0;--accent:#2563eb;--good:#10b981;--bad:#ef4444;--warn:#f59e0b;--shadow:rgba(15,23,42,.08);
--overlay-bg:rgba(241,245,249,.7)}
:root[data-theme="dark"]{--bg:#0a0f28;--panel:#131a3a;--panel-2:#19214a;--text:#e2e8f0;
--muted:#94a3b8;--border:#293567;--accent:#60a5fa;--good:#34d399;--bad:#f87171;--warn:#fbbf24;--shadow:rgba(0,0,0,.4);
--overlay-bg:rgba(10,15,40,.55)}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:20;background:var(--panel);border-bottom:1px solid var(--border);
padding:10px 20px;box-shadow:0 1px 3px var(--shadow)}

/* One wrapping row of labelled groups - Folder / AI Suggests / Tools - plus
   Statistics and Appearance pinned to the right. View/Filter/Sort live in the
   collapsible side panel instead (see .panel below), keeping this row short
   even as the app grows. A thin vertical rule between groups substitutes for
   the visual weight a heading would otherwise need, without spending a whole
   line on it (see the toolbar-redesign notes in this module's docstring: no
   app/page name is shown here at all - the window/tab title already carries
   that). */
.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.toolbar-shell{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.toolbar-menu{position:relative}
.toolbar-menu>summary{list-style:none;cursor:pointer;padding:7px 12px;border-radius:999px;
border:1px solid var(--border);background:linear-gradient(180deg,var(--panel),var(--panel-2));color:var(--text);font-size:12.5px;
font-weight:700;display:inline-flex;align-items:center;gap:6px;letter-spacing:.01em}
.toolbar-menu>summary::after{content:"▾";font-size:10px;opacity:.7}
.toolbar-menu[open]>summary{background:var(--accent);color:#fff;border-color:var(--accent)}
.toolbar-menu[open]>summary::after{content:"▴";opacity:1}
.menu-panel{position:absolute;top:calc(100% + 8px);left:0;display:flex;flex-direction:column;gap:8px;
padding:10px;border:1px solid var(--border);border-radius:12px;background:var(--panel);
box-shadow:0 12px 32px var(--shadow);min-width:280px;z-index:30}
.menu-panel .group{padding:6px 4px;border-radius:8px;background:rgba(148,163,184,.06)}
.menu-panel .glabel{display:block;margin-bottom:2px}
.group{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.glabel{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-right:2px}
.divider{width:1px;align-self:stretch;min-height:22px;background:var(--border)}
.folder-name{font-size:12.5px;color:var(--muted);font-family:ui-monospace,Consolas,monospace;
max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.folder-name.missing{color:var(--bad);font-weight:600}
button{font:inherit;font-size:13px;padding:6px 12px;border-radius:7px;border:1px solid var(--border);
background:var(--panel-2);color:var(--text);cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
button:disabled{opacity:.5;cursor:not-allowed}
input[type=number]{font:inherit;font-size:13px;width:60px;padding:6px 8px;border-radius:7px;
border:1px solid var(--border);background:var(--panel);color:var(--text)}
.gunit{font-size:12.5px;color:var(--muted)}

/* Statistics: plain numbers, not buttons - nothing here is clickable, so
   nothing here looks clickable. Pinned right by the toolbar's own wrapping;
   Appearance (the theme toggle) sits just past it, the last thing in the row. */
.stats{display:flex;gap:14px;margin-left:auto;font-size:13px;flex-wrap:wrap}
.stat{display:flex;gap:5px;align-items:baseline}
.stat b{font-variant-numeric:tabular-nums;font-size:15px}
.stat.keep b{color:var(--good)}.stat.reject b{color:var(--bad)}.stat.neutral b{color:var(--muted)}

/* Multi-select bulk action bar (Phase 2): entirely absent from the layout -
   not merely disabled - whenever nothing is picked, so it never sits there
   as dead chrome. Its own accent border marks it as contextual/temporary,
   distinct from the always-present toolbar above it. */
.bulkbar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-top:10px;
padding:8px 16px;border-radius:10px;border:1px solid var(--accent);background:var(--panel-2)}
.bulkcount{font-size:13px;font-weight:600}
.bulkacts{display:flex;gap:8px}
.bk{display:flex;align-items:center;gap:5px}
.bk .ic{font-size:14px;line-height:1}
.bk.keep .ic{color:var(--good)}.bk.reject .ic{color:var(--bad)}.bk.neutral .ic{color:var(--muted)}

.notice{margin:12px 20px 0;padding:10px 14px;border-radius:8px;border:1px dashed var(--warn);
background:var(--panel-2);font-size:13px}
.statusbar{margin-top:8px}
.status{font-size:12.5px;color:var(--muted)}
.status.error{color:var(--bad)}

/* One small rotating ring, reused everywhere something is working: the
   folder-load overlay below, and the Arrange dialog's own in-progress state
   - a single spinner style rather than a bespoke one per feature. */
@keyframes plmSpin{to{transform:rotate(360deg)}}
.spinner{display:inline-block;width:16px;height:16px;border-radius:50%;
border:2px solid var(--border);border-top-color:var(--accent);
animation:plmSpin .7s linear infinite;flex-shrink:0}
.spinner.big{width:34px;height:34px;border-width:3px}

/* Covers the gallery while a folder is (re)loading - opening a different
   folder, relocating one, or the very first fetch on page load. The
   previous gallery content stays in the DOM underneath (dimmed, not wiped),
   so switching back to a fast, already-loaded folder never produces a
   flash of empty content. */
.loading-overlay{position:fixed;inset:0;z-index:4;display:none;
align-items:center;justify-content:center;gap:12px;flex-direction:column;
background:var(--overlay-bg);color:var(--text);font-size:13.5px}

/* Filters, sorting and view options live here rather than in the toolbar
   (Phase 7 of the redesign) - contextual to the gallery it controls, and
   collapsible so a photographer who wants every last pixel for the grid can
   tuck it away; the toggle's own state persists like the theme does. */
.layout{display:flex;align-items:flex-start;gap:0}
.grid{flex:1;min-width:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
gap:12px;padding:16px 20px 60px}
.panel{width:250px;flex-shrink:0;padding:16px 20px 20px 4px;position:sticky;top:0;
max-height:100vh;overflow-y:auto}
.panel.collapsed{display:none}
.panel-section{margin-bottom:20px}
.panel-section h3{font-size:11px;margin:0 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.panel-filters{display:flex;flex-direction:column;gap:5px}
.panel-filters button{text-align:left;justify-content:flex-start}
.panel-row{display:flex;align-items:center;gap:8px;margin-top:10px}
.panel-hint{font-size:12px;color:var(--muted)}
.panel select{font:inherit;font-size:13px;padding:6px 8px;border-radius:7px;
border:1px solid var(--border);background:var(--panel);color:var(--text);flex:1}
.agreement-stats{display:grid;grid-template-columns:1fr auto;gap:4px 10px;font-size:12.5px}
.agreement-stats b{font-variant-numeric:tabular-nums}
.agreement-stats b.agree{color:var(--good)}
.agreement-stats b.disagree{color:var(--bad)}
.card{position:relative;background:var(--panel);border:1px solid var(--border);border-radius:11px;overflow:hidden;
display:flex;flex-direction:column;box-shadow:0 1px 2px var(--shadow)}
.card.keep{border-color:var(--good);border-width:2px}
.card.reject{border-color:var(--bad);border-width:2px}
.card.neutral{border-style:dashed}
.card.picked{outline:3px solid var(--accent);outline-offset:-3px}
.pick{position:absolute;top:8px;left:8px;z-index:2;width:24px;height:24px;display:flex;
align-items:center;justify-content:center;background:rgba(15,23,42,.55);border-radius:6px;cursor:pointer}
.pick input{width:16px;height:16px;margin:0;cursor:pointer;accent-color:var(--accent)}
.thumb-link{display:block;position:relative;cursor:zoom-in}
.thumb-link:hover .thumb{filter:brightness(1.06)}
.thumb{width:100%;aspect-ratio:1;background:var(--panel-2);object-fit:cover;display:block}
.ph{width:100%;aspect-ratio:1;background:var(--panel-2);display:flex;align-items:center;
justify-content:center;color:var(--muted);font-size:12.5px;text-align:center;padding:12px}
.meta{padding:8px 10px;display:flex;flex-direction:column;gap:4px;flex:1}
.name{font-size:12.5px;font-family:ui-monospace,Consolas,monospace;word-break:break-all}
.nums{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;
display:flex;align-items:center;flex-wrap:wrap;gap:6px}
/* The AI's suggestion, not the photographer's status - kept visually
   distinct (its own accent-outlined chip, never green/red/grey) so the two
   can never be mistaken for each other at a glance. */
.ai-chip{font-size:10.5px;padding:1px 7px;border-radius:20px;border:1px solid var(--accent);
color:var(--accent);white-space:nowrap}
/* Structured subject metadata (bird/mammal/human/...), not a judgement of
   any kind - its own neutral, muted style so it is never mistaken for
   either the AI-suggestion chip or the card's own colored border
   (.card.keep/.reject/.neutral) that shows review_status. */
.category-chip{font-size:10.5px;padding:1px 7px;border-radius:20px;border:1px solid var(--muted);
color:var(--muted);white-space:nowrap}
.acts{display:flex;gap:6px;padding:0 10px 10px}
.acts button{flex:1;padding:6px 4px;font-size:15px;line-height:1}
.acts .a-keep.on{background:var(--good);border-color:var(--good);color:#fff}
.acts .a-reject.on{background:var(--bad);border-color:var(--bad);color:#fff}
.acts .a-neutral.on{background:var(--muted);border-color:var(--muted);color:#fff}
dialog{border:1px solid var(--border);border-radius:12px;background:var(--panel);color:var(--text);
padding:0;max-width:520px;width:92%;box-shadow:0 12px 40px var(--shadow)}
dialog::backdrop{background:rgba(2,6,23,.55)}
.dlg{padding:20px}
.dlg h2{margin:0 0 4px;font-size:16px}
.dlg .sub{color:var(--muted);font-size:13px;margin-bottom:14px}
.plan{display:grid;grid-template-columns:auto 1fr;gap:6px 16px;font-size:13.5px;margin-bottom:16px}
.plan b{font-variant-numeric:tabular-nums;text-align:right}
.plan .dest{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;color:var(--muted)}
.dlg-acts{display:flex;gap:8px;justify-content:flex-end}
.dlg-progress{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);margin-bottom:14px}
.empty{padding:60px 20px;text-align:center;color:var(--muted)}

/* Lightbox: full-screen in-app viewer opened by clicking a card's thumbnail.
   Deliberately always dark regardless of the page's light/dark theme - a
   review loupe, like Lightroom's, is dark so the photo itself reads correctly
   without a bright surround fighting it. */
#lightbox{border:none;padding:0;width:100vw;height:100vh;max-width:100vw;max-height:100vh;
background:#05070d;color:#fff;overflow:hidden}
#lightbox::backdrop{background:rgba(2,6,23,.9)}
#lightbox[open]{animation:lbFadeIn .15s ease}
#lightbox.closing{animation:lbFadeOut .12s ease forwards}
@keyframes lbFadeIn{from{opacity:0}to{opacity:1}}
@keyframes lbFadeOut{to{opacity:0}}
.lb-stage{position:relative;width:100%;height:100%;display:flex;align-items:center;justify-content:center}
.lb-img-wrap{position:relative;max-width:97vw;max-height:86vh;display:flex}
.lb-img{max-width:97vw;max-height:86vh;object-fit:contain;user-select:none;-webkit-user-drag:none;
transform-origin:center center;display:block;transition:transform .08s ease-out}
.lb-img.dragging{transition:none}
.lb-img.grabbable{cursor:grab}.lb-img.grabbing{cursor:grabbing;transition:none}
.lb-crop-overlay{position:absolute;inset:0;pointer-events:none;display:none;overflow:hidden}
.lb-crop-box{position:absolute;border:2px solid #fbbf24;box-shadow:0 0 0 9999px rgba(2,6,23,.38);border-radius:4px;background:rgba(251,191,36,.12)}
.lb-crop-box::before{content:"Auto crop";position:absolute;top:-24px;left:0;padding:2px 6px;border-radius:4px;background:#fbbf24;color:#111827;font-size:11px;font-weight:700;white-space:nowrap}
.lb-crop-toggle{background:rgba(0,0,0,.06);border:1px solid rgba(0,0,0,.2);color:#000;border-radius:7px;padding:4px 10px;font-size:12px;cursor:pointer}
.lb-crop-toggle.on{background:#fbbf24;color:#111827;border-color:#f59e0b}
.lb-spinner{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
color:#cbd5e1;font-size:13px;text-align:center;padding:20px}
.lb-ai{font-size:11.5px;padding:2px 9px;border-radius:20px;border:1px solid #60a5fa;color:#93c5fd}
/* z-index:5 on every piece of overlay chrome (close/nav/bottom bar):
   a CSS transform doesn't resize .lb-img-wrap's own layout box, so once the
   image is zoomed past fit it can paint outside that box - and img-wrap
   comes after this chrome in the markup, so without an explicit z-index the
   zoomed image would paint over it and hide it. */
.lb-close,.lb-nav{position:absolute;z-index:5;border-radius:50%;background:rgba(255,255,255,.08);
border:1px solid rgba(255,255,255,.18);color:#fff;display:flex;align-items:center;justify-content:center;
cursor:pointer}
.lb-close:hover,.lb-nav:hover{background:rgba(255,255,255,.2)}
.lb-nav:disabled{opacity:.2;cursor:not-allowed;background:rgba(255,255,255,.08)}
.lb-close{top:16px;right:20px;width:38px;height:38px;font-size:20px;line-height:1}
.lb-nav{top:50%;margin-top:-26px;width:52px;height:52px;font-size:24px}
.lb-nav.prev{left:16px}.lb-nav.next{right:16px}
.lb-bottom{position:absolute;z-index:5;left:0;right:0;bottom:0;display:flex;flex-direction:column;
align-items:center;gap:10px;padding:12px 16px 18px;background:linear-gradient(to top,rgba(0,0,0,.5),transparent)}
/* One box holding everything (info, exposure, Save JPEG, Keep/Reject/Neutral,
   status), centred above the film strip by .lb-bottom's own
   align-items:center - no extra wrapper needed for that. It is opaque and
   light so black text has something to contrast against; the dark viewer
   behind it is what light-on-dark text used to rely on instead. */
.lb-info{display:flex;justify-content:center;align-items:center;gap:14px;flex-wrap:wrap;
font-size:12.5px;color:#000;font-variant-numeric:tabular-nums;
background:rgba(255,255,255,.92);border-radius:20px;padding:8px 18px;
box-shadow:0 2px 10px rgba(0,0,0,.35)}
.lb-exp{display:flex;align-items:center;gap:6px}
.lb-exp button{width:22px;height:22px;padding:0;border-radius:50%;font-size:14px;line-height:1;
background:rgba(0,0,0,.06);border:1px solid rgba(0,0,0,.2);color:#000;cursor:pointer}
.lb-exp button:hover{background:rgba(0,0,0,.14)}
.lb-exp .val{min-width:56px;text-align:center}
.lb-save{background:rgba(0,0,0,.06);border:1px solid rgba(0,0,0,.2);color:#000;
border-radius:7px;padding:4px 10px;font-size:12px;cursor:pointer}
.lb-save:hover{background:rgba(0,0,0,.14)}
#lb-reason{font:inherit;font-size:12px;color:#000;background:rgba(0,0,0,.06);
border:1px solid rgba(0,0,0,.2);border-radius:7px;padding:4px 8px;cursor:pointer}
#lb-reason-note{font:inherit;font-size:12px;color:#000;background:rgba(0,0,0,.06);
border:1px solid rgba(0,0,0,.2);border-radius:7px;padding:4px 8px;width:160px}
.lb-acts{display:flex;gap:8px}
.lb-acts button{font-size:13px;padding:7px 16px;border-radius:10px;font-weight:650;
background:rgba(0,0,0,.04);border:1px solid rgba(0,0,0,.15);color:#000}
.lb-acts .keep{border-color:var(--good);color:#065f46}
.lb-acts .keep.on{background:var(--good);color:#06281f}
.lb-acts .reject{border-color:var(--bad);color:#991b1b}
.lb-acts .reject.on{background:var(--bad);color:#2a0a0a}
.lb-acts .neutral{border-color:var(--muted);color:#334155}
.lb-acts .neutral.on{background:var(--muted);color:#0b1220}
.lb-status{font-size:12px;color:#334155}
.lb-status.error{color:var(--bad)}
.lb-film{display:flex;gap:6px;max-width:96vw;overflow-x:auto;padding:2px}
.lb-film img{width:50px;height:50px;object-fit:cover;border-radius:6px;opacity:.5;cursor:pointer;
border:2px solid transparent;flex-shrink:0}
.lb-film img:hover{opacity:.85}
.lb-film img.current{opacity:1;border-color:#fff}
"""

JS_TEMPLATE = """
const PLM = {
  state: null,
  boxes: false,
  busy: false,
  // 'all' | 'keep' | 'reject' | 'neutral' | 'ai_keep' | 'ai_reject' |
  // 'ai_keep_user_reject' | 'ai_reject_user_keep' - client-side only,
  // survives state refreshes.
  filter: 'all',
  sort: {key: 'score', dir: 'desc'},  // key: 'score'|'name'|'date' - 'score'/'desc' matches the server's own default order
  picked: new Set(),      // image_path set, for the multi-select bulk actions bar
  labels: __STATUS_LABELS__,
  reasons: __REASON_LABELS__,
  relocateDismissed: false,  // "Not Now" on the folder-missing dialog; a folder change resets this
  lastFolderSeen: undefined,
};

// Filter, then sort - "Select All Visible" and the Lightbox's own next/
// previous both read this same list, so "visible" always means exactly what
// the photographer is currently looking at under both.
function visibleImages(){
  const images = (PLM.state && PLM.state.images) || [];
  return sortImages(filterImages(images));
}

function filterImages(images){
  if(PLM.filter === 'all') return images;
  if(PLM.filter === 'ai_keep') return images.filter(i => i.ai_suggestion === 'keep');
  if(PLM.filter === 'ai_reject') return images.filter(i => i.ai_suggestion === 'reject');
  if(PLM.filter === 'ai_keep_user_reject') {
    return images.filter(i => i.ai_suggestion === 'keep' && i.review_status === 'reject');
  }
  if(PLM.filter === 'ai_reject_user_keep') {
    return images.filter(i => i.ai_suggestion === 'reject' && i.review_status === 'keep');
  }
  // Subject category (bird/mammal/human/...) - one dynamic filter per
  // category actually present in this folder, see renderCategoryFilters().
  if(PLM.filter.indexOf('category:') === 0){
    const category = PLM.filter.slice('category:'.length);
    return images.filter(i => i.detected_category === category);
  }
  return images.filter(i => i.review_status === PLM.filter);
}

// The panel's Subject filters are data-driven, not a fixed list: only
// categories actually present in this folder get a button, so the section
// simply grows (or shows nothing at all, on a folder with no recorded
// detections) as the detector's own coverage grows, with no UI code change.
function renderCategoryFilters(images){
  const section = q('#category-section');
  const present = Array.from(new Set(images.map(i => i.detected_category).filter(Boolean))).sort();
  if(!present.length){ section.style.display = 'none'; return; }
  section.style.display = '';
  const container = q('#category-filters');
  container.innerHTML = present.map(category =>
    '<button class="filter' + (PLM.filter === 'category:' + category ? ' on' : '') + '" data-filter="'
      + esc('category:' + category) + '">' + esc(cap(category)) + '</button>'
  ).join('');
  container.querySelectorAll('.filter').forEach(b =>
    b.addEventListener('click', () => setFilter(b.dataset.filter)));
}

// A missing sort value (no score, no capture date) always sorts last,
// regardless of direction - there is no meaningful position for "unknown"
// within an ascending or descending order, in either direction.
function sortImages(images){
  const key = PLM.sort.key, factor = PLM.sort.dir === 'desc' ? -1 : 1;
  const valueOf = image => key === 'name' ? image.filename.toLowerCase()
    : key === 'date' ? image.captured_at
    : image.score;
  return images.slice().sort((a, b) => {
    const va = valueOf(a), vb = valueOf(b);
    const missingA = va == null, missingB = vb == null;
    if(missingA && missingB) return 0;  // Array.sort is stable - keeps the server's own tiebreak order
    if(missingA) return 1;
    if(missingB) return -1;
    if(va < vb) return -1 * factor;
    if(va > vb) return 1 * factor;
    return 0;
  });
}

const esc = s => { const d = document.createElement('div'); d.textContent = s == null ? '' : s; return d.innerHTML; };
const q = s => document.querySelector(s);
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

function setTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  try{ localStorage.setItem('plm-theme', t) }catch(e){}
  // Guarded: this runs before DOMContentLoaded so the theme is applied without
  // a flash of the wrong one, which means the button may not be parsed yet.
  // Nothing cosmetic is allowed to throw here - an exception at this point
  // would abort the whole script and the gallery would never load at all.
  const button = q('#theme');
  if(button) button.textContent = t === 'dark' ? 'Light Mode' : 'Dark Mode';
}
(function(){
  let t; try{ t = localStorage.getItem('plm-theme') }catch(e){}
  if(!t) t = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  setTheme(t);
})();

async function api(path, options){
  const r = await fetch(path, Object.assign({cache:'no-store'}, options || {}));
  const j = await r.json();
  if(!r.ok || j.error){ throw new Error(j.error || ('HTTP ' + r.status)); }
  return j;
}

// Mirrors into the lightbox's own status line too: its <dialog> backdrop
// covers the whole page while open, so the header's #status is invisible for
// as long as a photographer is reviewing without ever leaving the viewer.
function say(message, isError){
  const cls = 'status' + (isError ? ' error' : '');
  [q('#status'), q('#lb-status')].forEach(el => {
    if(!el) return;
    el.textContent = message || '';
    el.className = cls.replace('status', el.id === 'lb-status' ? 'lb-status' : 'status');
  });
}

// One shared "this could take a while" overlay for every gallery-replacing
// server round trip (initial load, switching folders, relocating one) -
// large folders are real disk I/O, not instant, and the previous gallery
// content otherwise just sits there unchanged with no sign anything is
// happening beyond the easy-to-miss status line. The overlay dims the
// existing gallery rather than clearing it, so a fast reload never flashes
// empty content first.
function setLoading(active, message){
  q('#loading-overlay').style.display = active ? 'flex' : 'none';
  if(active) q('#loading-overlay-message').textContent = message || 'Loading…';
}

// The gallery is rebuilt from state on every change. At a few thousand cards
// this is comfortably fast, and every card's markup then comes from exactly
// one place; `loading="lazy"` keeps the browser from fetching a thumbnail
// until it is actually scrolled into view.
function render(){
  const s = PLM.state;
  if(!s) return;
  const loading = s.loading || null;
  const stageText = loading && !loading.complete ? (loading.message || 'Loading…') : '';
  if(stageText){ say(stageText); } else if(!q('#status').textContent) { say('PeakPic workflow ready'); }
  const hasFolder = !!s.input_folder;
  const folderEl = q('#folder');
  folderEl.textContent = s.input_folder || 'No folder open';
  folderEl.title = s.input_folder || '';
  folderEl.classList.toggle('missing', !!s.folder_missing);
  q('#c-total').textContent = s.counts.total.toLocaleString();
  q('#c-keep').textContent = s.counts.keep.toLocaleString();
  q('#c-reject').textContent = s.counts.reject.toLocaleString();
  q('#c-neutral').textContent = s.counts.neutral.toLocaleString();
  const workflow = s.workflow || {};
  const workflowStage = q('#workflow-stage');
  const workflowStatus = q('#workflow-status');
  if(workflowStage) workflowStage.textContent = workflow.stage ? ('Stage: ' + workflow.stage) : 'Stage: ready';
  if(workflowStatus) workflowStatus.textContent = 'Ranked ' + (workflow.ranked ? '✓' : '—') + ' | Reviewed ' + (workflow.reviewed ? '✓' : '—') + ' | Imported ' + (workflow.imported ? '✓' : '—');

  document.querySelectorAll('.preset').forEach(b => {
    b.classList.toggle('on', Number(b.dataset.percent) === s.keep_percent);
  });
  q('#percent').value = s.keep_percent;
  // The AI Suggests group (threshold + Apply Suggestions) has nothing to act
  // on without a ranking at all - hidden rather than shown inert, per the
  // same "disappear until needed" rule the bulk bar follows.
  q('#ai-group').style.display = s.has_ranking ? '' : 'none';
  if(loading && !loading.complete){
    q('#apply-ai').disabled = true;
    q('#percent').disabled = true;
  } else {
    q('#percent').disabled = false;
  }
  q('#apply-ai').disabled = !s.has_ranking || s.counts.neutral === 0;

  const notice = q('#notice');
  if(s.warnings && s.warnings.length){
    notice.innerHTML = s.warnings.map(esc).join('<br>');
    notice.style.display = '';
  } else {
    notice.style.display = 'none';
  }

  // The folder this session thinks it has can no longer be found - moved,
  // renamed, or a changed drive letter. Auto-prompt once per such episode:
  // "Not Now" is remembered until either the folder changes again or it
  // relocates successfully, so it does not nag on every re-render.
  if(s.input_folder !== PLM.lastFolderSeen) PLM.relocateDismissed = false;
  PLM.lastFolderSeen = s.input_folder;
  if(s.folder_missing && !PLM.relocateDismissed && !q('#relocate-dlg').open){
    q('#relocate-dlg').showModal();
  }

  document.querySelectorAll('.filter').forEach(b => {
    b.classList.toggle('on', b.dataset.filter === PLM.filter);
  });
  renderCategoryFilters(s.images);
  q('#sort-key').value = PLM.sort.key;
  q('#sort-dir').textContent = PLM.sort.dir === 'desc' ? '↓' : '↑';
  q('#sort-dir').title = PLM.sort.dir === 'desc' ? 'Descending (click for ascending)' : 'Ascending (click for descending)';

  // Neither has anywhere to point without a real, present folder -
  // "Reveal" would otherwise send the literal string "null" to the server
  // (see openFolder) or fail outright on one that cannot be found, and
  // Arrange has nothing to file.
  q('#open').disabled = !hasFolder || s.folder_missing;
  q('#go').disabled = !hasFolder || s.folder_missing || !!(loading && !loading.complete);

  // A pick can outlive the image it names - arranging or switching folders
  // both replace s.images wholesale (arrange repoints paths, a new folder is
  // a different set of files entirely) - so anything no longer present is
  // dropped rather than left to bulk-decide a path that isn't in this
  // gallery any more.
  const present = new Set(s.images.map(i => i.image_path));
  for(const path of PLM.picked){ if(!present.has(path)) PLM.picked.delete(path); }
  updateBulkBar();
  renderAgreementStats(s.agreement);

  const grid = q('#grid');
  if(!s.images.length){
    q('#visible-count').textContent = '0 of 0 visible';
    grid.innerHTML = hasFolder
      ? '<div class="empty">No images found in this folder.</div>'
      : '<div class="empty">No folder open yet. Click &ldquo;Open Folder&hellip;&rdquo; above to choose one.</div>';
    return;
  }
  // Index into the filtered, displayed list - not s.images - so the lightbox
  // opened from a filtered view navigates only what is actually on screen.
  const visible = visibleImages();
  q('#visible-count').textContent = visible.length.toLocaleString() + ' of ' + s.images.length.toLocaleString() + ' visible';
  if(!visible.length){
    grid.innerHTML = '<div class="empty">No images match this filter.</div>';
    return;
  }
  grid.innerHTML = visible.map((image, i) => card(image, i)).join('');
  grid.querySelectorAll('button[data-status]').forEach(b => {
    b.addEventListener('click', () => setStatus(b.dataset.path, b.dataset.status));
  });
  grid.querySelectorAll('input[data-pick]').forEach(cb => {
    cb.addEventListener('change', () => {
      const path = cb.dataset.pick;
      if(cb.checked) PLM.picked.add(path);
      else PLM.picked.delete(path);
      cb.closest('.card').classList.toggle('picked', cb.checked);
      updateBulkBar();
    });
  });
  // Index into s.images, not image.rank: the lightbox navigates the gallery's
  // own displayed order (best-first, unranked last), which is what "next" and
  // "previous" should mean regardless of what the ranking says.
  grid.querySelectorAll('.thumb-link[data-index]').forEach(el => {
    el.addEventListener('click', () => Lightbox.open(Number(el.dataset.index)));
  });
}

// Purely informational (see session.py's agreement_stats): never read by
// review_status, ai_suggestion, or arrange() - just a way to see, over time,
// how often the model's own opinion matches what the photographer actually
// decided, and where it doesn't.
function renderAgreementStats(agreement){
  const section = q('#agreement-section');
  if(!agreement || !agreement.compared){
    section.style.display = 'none';
    return;
  }
  section.style.display = '';
  q('#agreement-stats').innerHTML =
    '<span>AI agrees</span><b class="agree">' + agreement.agree_percent.toFixed(1) + '%</b>' +
    '<span>AI disagrees</span><b class="disagree">' + agreement.disagree_percent.toFixed(1) + '%</b>' +
    '<span>AI Keep / You Reject</span><b>' + agreement.ai_keep_user_reject.toLocaleString() + '</b>' +
    '<span>AI Reject / You Keep</span><b>' + agreement.ai_reject_user_keep.toLocaleString() + '</b>';
}

function card(image, index){
  const status = image.review_status;
  const picked = PLM.picked.has(image.image_path);
  const url = 'thumb?path=' + encodeURIComponent(image.image_path) + (PLM.boxes ? '&boxes=1' : '');
  // Opens the in-app Lightbox rather than navigating anywhere - see Lightbox
  // below. Not an <a> any more: this is a click handler, not a link.
  const visual = image.missing_file
    ? '<div class="ph">File not found<br>(listed in the ranking)</div>'
    : '<div class="thumb-link" data-index="' + index + '" title="Open in viewer">' +
        '<img class="thumb" loading="lazy" src="' + esc(url) + '" alt="' + esc(image.filename) + '"' +
        ' onerror="this.outerHTML=\\'<div class=ph>No preview available</div>\\'">' +
      '</div>';
  const score = image.score == null
    ? 'No score'
    : 'Score ' + image.score.toFixed(4) + (image.rank ? ' &middot; Rank ' + image.rank.toLocaleString() : '');
  // The AI's own suggestion, kept visually separate from review_status (see
  // .ai-chip) - it is a hint, never the photographer's actual decision.
  const aiChip = image.ai_suggestion
    ? '<span class="ai-chip">AI ' + cap(image.ai_suggestion) + '</span>'
    : '';
  // The recorded subject category - structured metadata, not a judgement -
  // kept visually distinct from both the AI-suggestion chip and the review
  // status badge (its own neutral-outlined style, see .category-chip).
  const categoryChip = image.detected_category
    ? '<span class="category-chip">' + esc(cap(image.detected_category)) + '</span>'
    : '';
  // A sibling of .thumb-link, not inside it - a click on the checkbox must
  // never also open the lightbox, and its own click listener is bound
  // straight to .thumb-link (see render()), so a sibling target simply never
  // reaches it; no stopPropagation needed.
  const pick = '<label class="pick" title="Select for a bulk action">' +
      '<input type="checkbox" data-pick="' + esc(image.image_path) + '"' + (picked ? ' checked' : '') + '>' +
    '</label>';
  // No text label for review_status any more - the card's own colored
  // border (.card.keep/.reject/.neutral) already communicates it, and
  // repeating it in text only added clutter without room for slightly
  // larger thumbnails. The status is still on the card's own title
  // attribute, so it is a hover away, not gone entirely.
  return '<div class="card ' + status + (picked ? ' picked' : '') + '" title="' +
      esc(PLM.labels[status] || status) + '">' + pick + visual +
    '<div class="meta">' +
      '<div class="name" title="' + esc(image.image_path) + '">' + esc(image.filename) + '</div>' +
      '<div class="nums">' + score + aiChip + categoryChip + '</div>' +
    '</div>' +
    '<div class="acts">' +
      '<button class="a-keep' + (status === 'keep' ? ' on' : '') + '"' +
        ' data-status="keep" data-path="' + esc(image.image_path) + '" title="Keep">&#10003;</button>' +
      '<button class="a-reject' + (status === 'reject' ? ' on' : '') + '"' +
        ' data-status="reject" data-path="' + esc(image.image_path) + '" title="Reject">&#10005;</button>' +
      '<button class="a-neutral' + (status === 'neutral' ? ' on' : '') + '"' +
        ' data-status="neutral" data-path="' + esc(image.image_path) + '" title="Neutral">&#9675;</button>' +
    '</div></div>';
}

// Shared by every caller that sets a review status - a gallery card button,
// the Lightbox's Keep/Reject/Neutral row, and its reason dropdown/note
// (which must save a new reason for the SAME status without re-triggering
// Lightbox's advance-to-next behaviour). Always sets the exact status given;
// there is no toggle any more; PLM.labels'/the card's own "on" class is what
// shows the CURRENT status, so a second click on the same button is simply a
// no-op re-save, never a hidden "clear" gesture. reason_note only ever
// reaches the server alongside reason === 'other', mirroring what the store
// itself enforces (annotations.py's set_review_decision).
async function setStatus(path, status, reason, reason_note){
  try{
    const j = await api('api/review/status', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        image_path: path,
        status: status,
        reason: status !== 'neutral' ? (reason || null) : null,
        reason_note: status !== 'neutral' && reason === 'other' ? (reason_note || null) : null,
      }),
    });
    PLM.state = j.state;
    render();
    say('Marked ' + (PLM.labels[status] || status) + '.');
  }catch(e){ say('Could not save: ' + e.message, true); }
}

// ===========================================================================
// Lightbox: full-screen in-app viewer, opened by clicking a card's thumbnail.
//
// Self-contained on purpose - it reads the gallery's image list off
// PLM.state and reuses setStatus() to persist Keep/Reject/Neutral, but owns
// all of its own state (current index, zoom, pan, a small preload cache)
// behind this one object. Nothing outside this block touches that state
// directly, so the viewer can be extended - or lifted into its own file, if
// this ever needs to be reused elsewhere - without touching the gallery code
// above it.
// ===========================================================================
const Lightbox = (function(){
  const MIN_SCALE = 1;        // 1 == fit-to-screen, what every image opens at
  const MAX_SCALE = 8;
  const PRELOAD_RADIUS = 2;   // neighbours on each side kept decoded and warm
  const FILM_RADIUS = 8;      // neighbours on each side shown in the filmstrip

  const EXPOSURE_STEP = 1 / 3;  // EV per click
  const EXPOSURE_MIN_STEPS = -9;  // -3.0 EV
  const EXPOSURE_MAX_STEPS = 9;   // +3.0 EV

  let index = -1;
  let scale = 1;
  let pan = {x: 0, y: 0};
  let showCropOverlay = false;
  let drag = null;            // {startX,startY,panX,panY,moved} while dragging
  let suppressClick = false;  // true for the click a drag-release also fires
  // Steps, not EV directly: an integer avoids float drift across repeated
  // +-1/3 additions. Deliberately NOT reset by open()/go()/resetView() - it
  // is meant to stay applied while browsing, like FastRawViewer's exposure
  // preview, until the reviewer dials it back down themselves. Display only:
  // never read by setStatus()/arrange(), never sent to the server, never
  // touches the RAW file.
  let exposureSteps = 0;
  // path -> blob object URL for the full-size decode. Bounded to a small
  // window around the current index and revoked on eviction, so a session of
  // thousands of images cannot leak memory.
  const cache = new Map();
  const cropCache = new Map();
  // path -> in-flight Promise<url>, so two callers requesting the SAME image
  // before either has resolved (renderImage() for the now-current image,
  // preloadAround() for a neighbour that becomes current a moment later)
  // share one fetch instead of firing a duplicate network request and a
  // duplicate server-side decode. Profiling the "Lightbox gets slower while
  // browsing quickly" report traced it here and to review/server.py's
  // `_serve_preview`: /preview re-ran a full RAW decode + JPEG re-encode on
  // every request (it is `Cache-Control: no-store`, correctly, for the
  // analysis report's own long-lived use of the same endpoint - see
  // thumbnails.review_preview for why the review app can safely cache it
  // instead), and rapid navigation was requesting the same handful of
  // images over and over as the reviewer flipped back and forth.
  const pending = new Map();

  // The same filtered list the gallery is currently showing - see
  // visibleImages() - so opening from a filtered view and pressing
  // next/previous stays inside what the reviewer is actually looking at.
  function images(){ return visibleImages(); }
  function current(){ return images()[index]; }
  function previewUrl(path){ return 'preview?path=' + encodeURIComponent(path); }

  function open(startIndex){
    index = startIndex;
    showCropOverlay = false;
    q('#lightbox').showModal();
    resetView();
    renderAll();
    preloadAround();
  }

  function close(){
    const dlg = q('#lightbox');
    dlg.classList.add('closing');
    setTimeout(() => { dlg.classList.remove('closing'); dlg.close(); }, 110);
  }

  function resetView(){
    scale = 1; pan = {x: 0, y: 0};
  }

  // Navigating within an already-open lightbox keeps the current zoom level
  // deliberately - a photographer zoomed in to check focus/eyes wants that
  // same zoom on the next frame too, not to re-zoom after every arrow key.
  // Only pan resets: a pan position is a spot on ONE photo's composition, and
  // has no meaning carried over to a different one. Opening the lightbox
  // fresh (open(), above) still starts every browsing session at Fit.
  function go(newIndex){
    if(newIndex < 0 || newIndex >= images().length) return;
    index = newIndex;
    pan = {x: 0, y: 0};
    renderAll();
    preloadAround();
  }
  function next(){ go(index + 1); }
  function prev(){ go(index - 1); }

  function renderAll(){
    const image = current();
    if(!image) return;
    q('#lb-crop-toggle').classList.toggle('on', showCropOverlay);
    q('#lb-prev').disabled = index <= 0;
    q('#lb-next').disabled = index >= images().length - 1;
    q('#lb-counter').textContent = (index + 1) + ' / ' + images().length;
    q('#lb-filename').textContent = image.filename;
    q('#lb-score').textContent = image.score == null ? 'No score'
      : 'Score ' + image.score.toFixed(4) + (image.rank ? ' · Rank ' + image.rank.toLocaleString() : '');
    q('#lb-ai').textContent = image.ai_suggestion ? 'AI suggests ' + cap(image.ai_suggestion) : '';
    q('#lb-ai').style.display = image.ai_suggestion ? '' : 'none';
    updateZoomIndicator();
    updateStatusButtons(image);
    updateSaveButton(image);
    updateReasonSelect(image);
    renderImage(image);
    updateCropOverlay(image);
    renderFilmstrip();
  }

  function updateSaveButton(image){
    q('#lb-save-jpeg').disabled = !image || !!image.missing_file;
  }

  // Reset to this image's own stored reason (and note, and note visibility)
  // on every navigation, so a reason picked for one image can never leak
  // onto the next one it was never actually saved against. Meaningless (and
  // disabled) for a Neutral image - there is no override to justify.
  function updateReasonSelect(image){
    const reason = (image && image.reason) || '';
    const disabled = !image || !!image.missing_file || (image && image.review_status === 'neutral');
    q('#lb-reason').value = reason;
    q('#lb-reason').disabled = disabled;
    const note = q('#lb-reason-note');
    note.value = (image && image.reason_note) || '';
    note.style.display = reason === 'other' ? '' : 'none';
    note.disabled = disabled;
  }

  function updateStatusButtons(image){
    q('#lb-keep').classList.toggle('on', image.review_status === 'keep');
    q('#lb-reject').classList.toggle('on', image.review_status === 'reject');
    q('#lb-neutral').classList.toggle('on', image.review_status === 'neutral');
  }

  function updateZoomIndicator(){
    const el = q('#lb-zoom');
    if(scale <= 1.001){ el.textContent = 'Fit'; return; }
    const img = q('#lb-img');
    if(img.naturalWidth && img.offsetWidth){
      el.textContent = Math.round((scale / hundredPercentScale(img)) * 100) + '%';
    } else {
      el.textContent = Math.round(scale * 100) + '%';
    }
  }

  // The scale (relative to fit) at which the image renders at its own native
  // pixel size. offsetWidth is the fit-rendered, untransformed layout width -
  // a CSS transform never changes it, so this stays correct at any zoom.
  function hundredPercentScale(img){
    return img.naturalWidth / img.offsetWidth;
  }

  function applyTransform(){
    const transform = 'translate(' + pan.x + 'px,' + pan.y + 'px) scale(' + scale + ')';
    q('#lb-img').style.transform = transform;
    q('#lb-crop-overlay').style.transform = transform;
  }

  async function fetchCropOverlay(image){
    if(!image || image.missing_file || !showCropOverlay) return null;
    const cached = cropCache.get(image.image_path);
    if(cached !== undefined) return cached;
    try{
      const j = await api('api/review/crop-data?path=' + encodeURIComponent(image.image_path));
      const crop = j && j.crop ? j.crop : null;
      cropCache.set(image.image_path, crop);
      return crop;
    }catch(e){
      cropCache.set(image.image_path, null);
      return null;
    }
  }

  function updateCropOverlay(image){
    const overlay = q('#lb-crop-overlay');
    const box = q('#lb-crop-box');
    const img = q('#lb-img');
    if(!image || !showCropOverlay || image.missing_file){ overlay.style.display = 'none'; return; }
    if(!img.naturalWidth || !img.naturalHeight){ overlay.style.display = 'none'; return; }
    const overlayW = img.clientWidth || img.offsetWidth || img.naturalWidth;
    const overlayH = img.clientHeight || img.offsetHeight || img.naturalHeight;
    if(!overlayW || !overlayH){ overlay.style.display = 'none'; return; }
    overlay.style.display = 'block';
    overlay.style.width = overlayW + 'px';
    overlay.style.height = overlayH + 'px';
    overlay.style.left = '0px';
    overlay.style.top = '0px';
    if(!img.naturalWidth || !img.naturalHeight){ box.style.display = 'none'; return; }
    fetchCropOverlay(image).then(crop => {
      if(current() !== image) return;
      if(!crop){ overlay.style.display = 'none'; return; }
      const left = Math.max(0, Math.min(1, crop.left || 0));
      const top = Math.max(0, Math.min(1, crop.top || 0));
      const right = Math.max(left, Math.min(1, crop.right || 1));
      const bottom = Math.max(top, Math.min(1, crop.bottom || 1));
      const width = Math.max(4, (right - left) * overlayW);
      const height = Math.max(4, (bottom - top) * overlayH);
      box.style.left = (left * overlayW) + 'px';
      box.style.top = (top * overlayH) + 'px';
      box.style.width = width + 'px';
      box.style.height = height + 'px';
      box.style.display = 'block';
      overlay.style.display = 'block';
    });
  }

  function renderImage(image){
    const img = q('#lb-img');
    const spinner = q('#lb-spinner');
    img.style.transform = 'none';
    img.classList.toggle('grabbable', false);
    // The zoom indicator's % reading depends on THIS image's own
    // naturalWidth/offsetWidth (see hundredPercentScale) - a carried-over
    // zoom level applied to a differently-sized photo needs those measured
    // fresh once it has actually decoded and laid out, not the outgoing
    // image's, which is all that is on screen the instant renderAll() calls
    // updateZoomIndicator() above.
    img.onload = () => { updateZoomIndicator(); updateCropOverlay(image); };
    if(image.missing_file){
      img.removeAttribute('src');
      spinner.textContent = 'File not found (has it moved?)';
      spinner.style.display = '';
      return;
    }
    const cached = cache.get(image.image_path);
    if(cached){
      spinner.style.display = 'none';
      img.src = cached;
      applyTransform();
      updateCropOverlay(image);
      return;
    }
    spinner.textContent = 'Loading full size…';
    spinner.style.display = '';
    fetchPreview(image.image_path).then(url => {
      // The photographer may already have moved on by the time this
      // resolves; only apply it if we are still looking at that image.
      if(current() !== image) return;
      spinner.style.display = 'none';
      img.src = url;
      applyTransform();
      updateCropOverlay(image);
    }).catch(() => {
      if(current() !== image) return;
      spinner.textContent = 'Could not load a full-size preview';
    });
  }

  function fetchPreview(path){
    const existing = cache.get(path);
    if(existing) return Promise.resolve(existing);
    const inFlight = pending.get(path);
    if(inFlight) return inFlight;

    // No {cache: 'no-store'} here any more: review/server.py's
    // _serve_preview override sends a real Cache-Control for THIS app's own
    // /preview, so the browser's native HTTP cache is a second, free layer
    // underneath this Map - a blob URL evicted from here (see
    // evictFarFromCurrent) can still come back without a server round trip
    // at all if the browser still has it, let alone without re-decoding it.
    const promise = fetch(previewUrl(path))
      .then(r => { if(!r.ok) throw new Error('HTTP ' + r.status); return r.blob(); })
      .then(blob => {
        const url = URL.createObjectURL(blob);
        pending.delete(path);
        cache.set(path, url);
        evictFarFromCurrent();
        return url;
      })
      .catch(err => { pending.delete(path); throw err; });
    pending.set(path, promise);
    return promise;
  }

  function evictFarFromCurrent(){
    const imgs = images();
    const keep = new Set();
    for(let i = Math.max(0, index - PRELOAD_RADIUS); i <= Math.min(imgs.length - 1, index + PRELOAD_RADIUS); i++){
      keep.add(imgs[i].image_path);
    }
    for(const [path, url] of cache){
      if(!keep.has(path)){ URL.revokeObjectURL(url); cache.delete(path); }
    }
  }

  function preloadAround(){
    const imgs = images();
    for(let offset = 1; offset <= PRELOAD_RADIUS; offset++){
      const before = imgs[index - offset];
      const after = imgs[index + offset];
      if(before && !before.missing_file) fetchPreview(before.image_path).catch(() => {});
      if(after && !after.missing_file) fetchPreview(after.image_path).catch(() => {});
    }
  }

  function renderFilmstrip(){
    const imgs = images();
    const start = Math.max(0, index - FILM_RADIUS);
    const end = Math.min(imgs.length - 1, index + FILM_RADIUS);
    let html = '';
    for(let i = start; i <= end; i++){
      html += '<img class="' + (i === index ? 'current' : '') + '" data-index="' + i + '" loading="lazy" ' +
        'src="' + esc('thumb?path=' + encodeURIComponent(imgs[i].image_path)) + '" alt="">';
    }
    const strip = q('#lb-film');
    strip.innerHTML = html;
    strip.querySelectorAll('img').forEach(el =>
      el.addEventListener('click', () => go(Number(el.dataset.index))));
    const activeEl = strip.querySelector('.current');
    if(activeEl && activeEl.scrollIntoView) activeEl.scrollIntoView({inline: 'center', block: 'nearest'});
  }

  // Reuses setStatus() rather than posting again: same validation, same
  // PLM.state refresh, same status reporting - the lightbox just also
  // advances afterwards, which is the one thing the gallery view never needs.
  // Whatever the reason dropdown (and, for "Other", its note) currently show
  // travel with this decision - that is the only place a fresh reason ever
  // comes from, and only for Keep/Reject (see setStatus).
  async function decideAndAdvance(status){
    const image = current();
    if(!image) return;
    const wasLast = index >= images().length - 1;
    const reason = q('#lb-reason').value || null;
    const reasonNote = reason === 'other' ? (q('#lb-reason-note').value || null) : null;
    await setStatus(image.image_path, status, reason, reasonNote);
    updateStatusButtons(current());
    updateReasonSelect(current());
    if(!wasLast) next();
  }

  // Lets the reason be changed (or added) for an image that already carries a
  // Keep/Reject, without re-clicking either - setStatus() directly, with the
  // SAME status the image already has, rather than decideAndAdvance()'s
  // advance-to-next behaviour. Picking "Other" only reveals the note field
  // and waits - see onReasonNoteCommit - rather than saving an empty
  // explanation the instant it's chosen.
  async function onReasonChange(){
    const image = current();
    const reason = q('#lb-reason').value || null;
    q('#lb-reason-note').style.display = reason === 'other' ? '' : 'none';
    if(!image || image.review_status === 'neutral') return;
    if(reason === 'other'){ q('#lb-reason-note').focus(); return; }
    await setStatus(image.image_path, image.review_status, reason, null);
    updateReasonSelect(current());
  }

  // Commits the free-text note for "Other" - on blur or Enter, not on every
  // keystroke, so choosing a few words doesn't fire a save per character.
  async function onReasonNoteCommit(){
    const image = current();
    if(!image || image.review_status === 'neutral' || q('#lb-reason').value !== 'other') return;
    await setStatus(image.image_path, image.review_status, 'other', q('#lb-reason-note').value || null);
    updateReasonSelect(current());
  }

  // -- exposure (display only) -----------------------------------------------
  // A CSS brightness() filter on the <img> itself - nothing is decoded,
  // re-rendered or written anywhere. brightness(2^EV) approximates a stop of
  // exposure well enough to judge a dark or bright frame at a glance, which
  // is the only job this has: it is not colour-accurate RAW development.

  function applyExposure(){
    const ev = exposureSteps * EXPOSURE_STEP;
    q('#lb-img').style.filter = exposureSteps === 0 ? '' : 'brightness(' + Math.pow(2, ev).toFixed(4) + ')';
    q('#lb-exp-value').textContent = (ev >= 0 ? '+' : '') + ev.toFixed(1) + ' EV';
    q('#lb-exp-down').disabled = exposureSteps <= EXPOSURE_MIN_STEPS;
    q('#lb-exp-up').disabled = exposureSteps >= EXPOSURE_MAX_STEPS;
  }

  function adjustExposure(direction){
    exposureSteps = Math.max(EXPOSURE_MIN_STEPS, Math.min(EXPOSURE_MAX_STEPS, exposureSteps + direction));
    applyExposure();
  }

  // -- save as JPEG -----------------------------------------------------------
  // A real navigation to a same-origin URL, not a fetch: the server answers
  // with Content-Disposition: attachment (see review/server.py's
  // _serve_save_jpeg), so the browser's own download handling takes over -
  // its own Save As dialog or download folder, whichever the browser is
  // configured for. Deliberately does not bake the exposure preview above
  // into the file: that is a display-only adjustment, and a save is meant to
  // be the camera's own rendering, not an edit.

  function saveJpeg(){
    const image = current();
    if(!image || image.missing_file) return;
    const a = document.createElement('a');
    a.href = 'save-jpeg?path=' + encodeURIComponent(image.image_path);
    a.download = '';
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // -- zoom & pan -----------------------------------------------------------

  function setScale(newScale, anchor){
    const old = scale;
    newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));
    if(anchor){
      // Zoom toward the cursor (or double-click point), not the image centre:
      // whatever image-space point was under the cursor stays there. transform
      // is `translate(pan) scale(s)` with a centred origin, so translate is
      // already in unscaled/screen pixels - this is the standard derivation
      // for keeping a point fixed under those semantics.
      const stage = q('#lb-stage').getBoundingClientRect();
      const relX = anchor.x - (stage.left + stage.width / 2);
      const relY = anchor.y - (stage.top + stage.height / 2);
      pan.x = relX - (relX - pan.x) * (newScale / old);
      pan.y = relY - (relY - pan.y) * (newScale / old);
    }
    scale = newScale <= 1.001 ? 1 : newScale;
    if(scale === 1) pan = {x: 0, y: 0};
    clampPan();
    q('#lb-img').classList.toggle('grabbable', scale > 1);
    applyTransform();
    updateZoomIndicator();
  }

  function clampPan(){
    const img = q('#lb-img');
    const stage = q('#lb-stage');
    if(!img.offsetWidth) return;
    const scaledW = img.offsetWidth * scale, scaledH = img.offsetHeight * scale;
    const maxX = Math.max(0, (scaledW - stage.clientWidth) / 2);
    const maxY = Math.max(0, (scaledH - stage.clientHeight) / 2);
    pan.x = Math.max(-maxX, Math.min(maxX, pan.x));
    pan.y = Math.max(-maxY, Math.min(maxY, pan.y));
  }

  function onWheel(e){
    if(!current()) return;
    e.preventDefault();
    // A plain wheel steps through images - the fast, default gesture for
    // browsing a shoot without leaving the mouse. Zooming is the deliberate,
    // less frequent action, so it only happens with Ctrl held (which also
    // stops the browser's own Ctrl+wheel page-zoom, courtesy of the
    // preventDefault above).
    if(!e.ctrlKey){
      if(e.deltaY > 0) next();
      else if(e.deltaY < 0) prev();
      return;
    }
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setScale(scale * factor, {x: e.clientX, y: e.clientY});
  }

  function onDblClick(e){
    if(!current()) return;
    const img = q('#lb-img');
    if(scale > 1){ setScale(1); return; }
    if(!img.naturalWidth) return;
    setScale(hundredPercentScale(img), {x: e.clientX, y: e.clientY});
  }

  function onPointerDown(e){
    if(scale <= 1) return;
    drag = {startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y, moved: false};
    q('#lb-img').classList.add('grabbing');
    e.preventDefault();
  }
  function onPointerMove(e){
    if(!drag) return;
    const dx = e.clientX - drag.startX, dy = e.clientY - drag.startY;
    if(Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    pan.x = drag.panX + dx; pan.y = drag.panY + dy;
    clampPan();
    applyTransform();
  }
  function onPointerUp(){
    if(drag && drag.moved) suppressClick = true;
    drag = null;
    q('#lb-img').classList.remove('grabbing');
  }

  // Clicking the stage's own background - not the image, not a button, not
  // the top/bottom bars - closes the viewer. A drag that ends over that same
  // background must not also close it, hence suppressClick.
  function onStageClick(e){
    if(suppressClick){ suppressClick = false; return; }
    if(e.target === q('#lb-stage')) close();
  }

  // <select> counts as a typing target too - its own arrow-key/letter
  // handling (opening it, jumping to an option) must not be fought with next
  // in the same keystroke.
  function isTypingTarget(target){
    const tag = target && target.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || !!(target && target.isContentEditable);
  }

  function onKeyDown(e){
    if(!q('#lightbox').open) return;
    if(isTypingTarget(e.target)) return;  // never hijack the reason dropdown/note field
    if(e.key === 'ArrowRight') next();
    else if(e.key === 'ArrowLeft') prev();
    else{
      // 5/0 mirror a lot of RAW viewers' star-rating keys (5 stars = keep,
      // 0 stars = reject); k/r are the mnemonic pair for the same two
      // actions, so either hand position works. u is "unflag"/undecided -
      // the same convention Lightroom's own U key uses - for Neutral.
      const key = e.key.toLowerCase();
      if(key === '5' || key === 'k') decideAndAdvance('keep');
      else if(key === '0' || key === 'r') decideAndAdvance('reject');
      else if(key === 'u') decideAndAdvance('neutral');
    }
    // Escape is native <dialog> behaviour - see the 'cancel' listener below,
    // which only exists to route it through the same fade-out as every other
    // close, not to implement the key itself.
  }

  function bind(){
    q('#lb-close').addEventListener('click', close);
    q('#lb-prev').addEventListener('click', prev);
    q('#lb-next').addEventListener('click', next);
    q('#lb-keep').addEventListener('click', () => decideAndAdvance('keep'));
    q('#lb-reject').addEventListener('click', () => decideAndAdvance('reject'));
    q('#lb-neutral').addEventListener('click', () => decideAndAdvance('neutral'));
    q('#lb-exp-down').addEventListener('click', () => adjustExposure(-1));
    q('#lb-exp-up').addEventListener('click', () => adjustExposure(1));
    q('#lb-save-jpeg').addEventListener('click', saveJpeg);
    q('#lb-reason').addEventListener('change', onReasonChange);
    q('#lb-reason-note').addEventListener('blur', onReasonNoteCommit);
    q('#lb-reason-note').addEventListener('keydown', e => { if(e.key === 'Enter') e.target.blur(); });
    applyExposure();  // paints the initial "+0.0 EV" label and button state once
    q('#lb-stage').addEventListener('click', onStageClick);
    q('#lb-stage').addEventListener('wheel', onWheel, {passive: false});
    q('#lb-img').addEventListener('dblclick', onDblClick);
    q('#lb-img').addEventListener('mousedown', onPointerDown);
    q('#lb-crop-toggle').addEventListener('click', () => {
      showCropOverlay = !showCropOverlay;
      q('#lb-crop-toggle').classList.toggle('on', showCropOverlay);
      updateCropOverlay(current());
    });
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);
    window.addEventListener('resize', () => { clampPan(); applyTransform(); updateCropOverlay(current()); });
    document.addEventListener('keydown', onKeyDown);
    q('#lightbox').addEventListener('cancel', e => { e.preventDefault(); close(); });
    q('#lightbox').addEventListener('close', () => { index = -1; });
  }

  return {bind, open};
})();

async function setPercent(value){
  try{
    PLM.state = (await api('api/review/keep-percent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({keep_percent: Number(value)}),
    })).state;
    render();
    say('');
  }catch(e){ say('Could not change the AI suggestion threshold: ' + e.message, true); }
}

// Client-side only - it doesn't round-trip through the server, so switching
// filters is instant and never disturbs the underlying review status.
function setFilter(name){
  PLM.filter = name;
  render();
}

function setSortKey(key){
  PLM.sort.key = key;
  render();
}

function toggleSortDir(){
  PLM.sort.dir = PLM.sort.dir === 'desc' ? 'asc' : 'desc';
  render();
}

// Collapsible per Phase 7 of the redesign - persisted like the theme, so a
// photographer who tucks it away to maximise the grid does not have to redo
// that every time the page loads.
function setPanelOpen(open){
  q('#side-panel').classList.toggle('collapsed', !open);
  q('#panel-toggle').classList.toggle('on', open);
  try{ localStorage.setItem('plm-panel-open', open ? '1' : '0') }catch(e){}
}
function togglePanel(){
  setPanelOpen(q('#side-panel').classList.contains('collapsed'));
}

// -- one generic confirmation dialog, reused by every action below that
// needs "are you sure?" before it touches many images or the disk at once
// (multi-select bulk actions, Apply AI Suggestions) - Arrange keeps its own
// dialog, since that one shows a real computed plan (directories, counts),
// not just a yes/no question. -----------------------------------------------

let pendingConfirm = null;  // () => Promise<void> | void - staged until Confirm is actually clicked
function askConfirm(title, body, action){
  pendingConfirm = action;
  q('#confirm-title').textContent = title;
  q('#confirm-body').textContent = body;
  q('#confirm-dlg').showModal();
}
async function runPendingConfirm(){
  q('#confirm-dlg').close();
  const action = pendingConfirm;
  pendingConfirm = null;
  if(action) await action();
}

// -- multi-select bulk actions ---------------------------------------------
// Picking images is client-side only (see PLM.picked); nothing is sent until
// the confirmation dialog above is confirmed, and even then it is one
// request, not one per image - a slow network shouldn't mean a photographer
// watches N separate saves for one gesture.

function updateBulkBar(){
  const n = PLM.picked.size;
  const bar = q('#bulk-bar');
  // Gone from the layout entirely with nothing picked - not merely
  // disabled - so it never sits there as dead chrome (Phase 2).
  if(n === 0){ bar.style.display = 'none'; return; }
  bar.style.display = '';
  q('#bulk-count').textContent = n === 1 ? '1 image selected' : n.toLocaleString() + ' images selected';
}

function clearPicked(){
  PLM.picked.clear();
  render();
}

// The primary way to start a bulk selection (Phase 1 of this round): every
// image currently on screen - after BOTH the active filter and the active
// sort - joins the pick set in one click, exactly matching what "visible"
// means to the gallery and the Lightbox alike (see visibleImages()).
function selectAllVisible(){
  for(const image of visibleImages()){ PLM.picked.add(image.image_path); }
  render();
}

function confirmBulkStatus(status){
  if(!PLM.picked.size) return;
  const n = PLM.picked.size;
  const label = PLM.labels[status] || status;
  askConfirm(
    'Mark ' + n + (n === 1 ? ' image ' : ' images ') + label + '?',
    status === 'neutral'
      ? 'This clears the review status on these images - each becomes Neutral. Nothing is moved or deleted.'
      : 'This sets the review status on these images to ' + label + ', overriding whatever each currently has.',
    () => runBulkStatus(status)
  );
}

async function runBulkStatus(status){
  const paths = Array.from(PLM.picked);
  try{
    const j = await api('api/review/bulk-status', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image_paths: paths, status: status}),
    });
    PLM.picked.clear();
    PLM.state = j.state;
    render();
    say('Updated ' + j.applied.toLocaleString() + ' image(s).'
      + (j.failed.length ? ' ' + j.failed.length.toLocaleString() + ' could not be found (moved or deleted?).' : ''));
  }catch(e){ say('Could not apply the bulk action: ' + e.message, true); }
}

// -- AI suggestions (still entirely read-only until this runs, and even
// then never overrides a photographer's own Keep/Reject without asking
// first) --------------------------------------------------------------

// A Neutral image has nothing manual at risk, so applying the AI's current
// suggestion to it happens immediately - no confirmation needed. Only if
// that leaves some already-decided images disagreeing with the AI does a
// second, explicit question follow, since THAT step would overwrite real
// manual work.
async function applyAiSuggestions(){
  try{
    const j = await api('api/review/apply-ai-suggestions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({include_decided: false}),
    });
    PLM.state = j.state;
    render();
    if(j.conflicts > 0){
      askConfirm(
        "Override " + j.conflicts + " manually-marked image(s)?",
        (j.applied ? j.applied + " Neutral image(s) were just updated automatically. " : "")
          + j.conflicts + " image(s) you already marked Keep or Reject differ from the AI's current suggestion. "
          + "Override those too, to match the AI?",
        applyAiSuggestionsToDecided
      );
    } else {
      say('Applied the AI suggestion to ' + j.applied.toLocaleString() + ' image(s).');
    }
  }catch(e){ say('Could not apply AI suggestions: ' + e.message, true); }
}

async function applyAiSuggestionsToDecided(){
  try{
    const j = await api('api/review/apply-ai-suggestions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({include_decided: true}),
    });
    PLM.state = j.state;
    render();
    say('Overrode ' + j.overridden.toLocaleString() + ' manually-marked image(s) to match the AI.');
  }catch(e){ say('Could not override: ' + e.message, true); }
}

async function openFolder(){
  if(!PLM.state.input_folder) return;  // nothing open - #open is disabled too, but belt and suspenders
  try{
    await api('open-folder?path=' + encodeURIComponent(PLM.state.input_folder));
  }catch(e){ say('Could not open the folder: ' + e.message, true); }
}

async function runAutoCrop(){
  if(PLM.busy) return;
  PLM.busy = true;
  q('#auto-crop').disabled = true;
  say('Starting auto crop for Lightroom…');
  try{
    const j = await api('api/review/auto-crop', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({}),
    });
    say(j.message || 'Auto crop started.');
  }catch(e){ say('Could not start auto crop: ' + e.message, true); }
  finally{ PLM.busy = false; q('#auto-crop').disabled = false; }
}

// Shows the OS's own folder-browser dialog (the server bridges it - a
// browser cannot show one itself, see os_actions.py's choose_folder) and, if
// the photographer picked something, switches the whole review to it. Meant
// for a folder that was never ranked at all: it still loads, every image
// Neutral, so Keep/Reject/Neutral is the only way to sort it. PLM.busy
// guards against the dialog (which blocks the request on the server) being
// opened twice from a second click.
async function importSelected(){
  if(PLM.busy) return;
  PLM.busy = true;
  q('#import-selected').disabled = true;
  try{
    const j = await api('api/review/import-selected', {});
    if(j.ok){
      say('Imported selected images into the destination folder.', false);
    } else {
      say(j.error || 'Import failed', true);
    }
  } catch(e){ say('Import failed: ' + e.message, true); }
  finally{ PLM.busy = false; q('#import-selected').disabled = false; }
}

async function switchFolder(){
  if(PLM.busy) return;
  PLM.busy = true;
  q('#switch-folder').disabled = true;
  say('Choose a folder…');
  // Covers both the native folder-picker dialog and, once one is chosen, the
  // actual load - a single awaited request from here, with no way to tell
  // the two phases apart from outside it. The OS dialog sits above this
  // overlay regardless (a separate top-level window), so showing it early
  // is harmless.
  setLoading(true, 'Opening folder…');
  try{
    const j = await api('api/review/open-folder', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({}),
    });
    if(j.cancelled){ say(''); return; }
    PLM.state = j.state;
    render();
    // The folder path itself is not repeated here - the toolbar's own
    // #folder chip already shows it persistently, right above this message.
    say('Folder opened.' + (j.recovered ? ' ' + j.recovered + ' decision(s) recovered.' : ''));
  }catch(e){ say('Could not open a folder: ' + e.message, true); }
  finally{ PLM.busy = false; q('#switch-folder').disabled = false; setLoading(false); }
}

// The folder this session was reviewing can no longer be found - moved,
// renamed, or a changed drive letter (see the relocate-dlg auto-shown by
// render()). Reuses the same native folder picker as switchFolder(), but
// every stored path (the ranking, any review decisions) is repointed at the
// new location automatically - see ReviewSession.relocate_folder - rather
// than starting the folder over as if it had never been reviewed.
async function relocateFolder(){
  if(PLM.busy) return;
  PLM.busy = true;
  say("Choose the folder's new location…");
  setLoading(true, 'Relocating folder…');
  try{
    const j = await api('api/review/relocate-folder', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({}),
    });
    if(j.cancelled){ say(''); return; }
    PLM.state = j.state;
    render();
    say('Folder relocated: ' + j.relocated.toLocaleString() + ' path(s) updated'
      + (j.recovered ? ', ' + j.recovered.toLocaleString() + ' decision(s) recovered' : '') + '.');
  }catch(e){ say('Could not relocate the folder: ' + e.message, true); }
  finally{ PLM.busy = false; setLoading(false); }
}

// Nothing moves until the photographer has seen the exact plan, so the dialog
// is populated from a real dry run rather than from the counters on screen.
async function confirmArrange(){
  try{
    const plan = await api('api/review/arrange', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({dry_run: true}),
    });
    q('#p-keep').textContent = plan.result.selected.toLocaleString();
    q('#p-reject').textContent = plan.result.rejected.toLocaleString();
    q('#p-neutral').textContent = PLM.state.counts.neutral.toLocaleString();
    q('#p-keep-dir').textContent = plan.result.selected_dir;
    q('#p-reject-dir').textContent = plan.result.rejected_dir;
    q('#dlg').showModal();
  }catch(e){ say('Could not prepare the plan: ' + e.message, true); }
}

async function doArrange(){
  if(PLM.busy) return;
  PLM.busy = true;
  q('#go').disabled = true;
  // Moving thousands of files is real, visible work - a busy cursor, the
  // dialog's own spinner in place of its buttons, and neither button
  // clickable meanwhile (unlike before: a second click, or Cancel, while a
  // move was already in flight was never actually guarded against here).
  document.body.style.cursor = 'wait';
  q('#dlg-cancel').disabled = true;
  q('#dlg-go').disabled = true;
  q('#dlg-progress').style.display = 'flex';
  say('Moving files...');
  try{
    const j = await api('api/review/arrange', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({dry_run: false}),
    });
    PLM.state = j.state;
    render();
    const r = j.result;
    let message = 'Arranged: ' + r.moved.toLocaleString() + ' moved';
    if(r.skipped) message += ', ' + r.skipped.toLocaleString() + ' skipped';
    if(r.renamed) message += ', ' + r.renamed.toLocaleString() + ' renamed to avoid overwriting';
    if(r.errors) message += ', ' + r.errors.toLocaleString() + ' failed';
    say(message, r.errors > 0);
  }catch(e){ say('Arrange failed: ' + e.message, true); }
  finally{
    PLM.busy = false;
    q('#go').disabled = false;
    document.body.style.cursor = '';
    q('#dlg-cancel').disabled = false;
    q('#dlg-go').disabled = false;
    q('#dlg-progress').style.display = 'none';
    q('#dlg').close();
  }
}

// A real navigation, exactly like Lightbox's saveJpeg: the server answers
// with Content-Disposition: attachment (see server.py's
// _serve_evaluation_report), so the browser's own download handling takes
// over rather than this page trying to save the file itself.
function exportEvaluationReport(fmt){
  const a = document.createElement('a');
  a.href = 'evaluation-report.' + fmt;
  a.download = '';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function boot(){
  const theme = q('#theme');
  // The button exists (see build_page), and setTheme has already labelled it
  // by now; both are written defensively so a markup change can never leave
  // the gallery unable to load.
  if(theme){
    theme.textContent = document.documentElement.getAttribute('data-theme') === 'dark'
      ? 'Light Mode' : 'Dark Mode';
    theme.addEventListener('click', () =>
      setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'));
  }
  document.querySelectorAll('.preset').forEach(b =>
    b.addEventListener('click', () => setPercent(b.dataset.percent)));
  document.querySelectorAll('.filter').forEach(b =>
    b.addEventListener('click', () => setFilter(b.dataset.filter)));
  q('#percent').addEventListener('change', e => setPercent(e.target.value));
  q('#boxes').addEventListener('click', () => {
    PLM.boxes = !PLM.boxes;
    q('#boxes').classList.toggle('on', PLM.boxes);
    render();
  });
  q('#sort-key').addEventListener('change', e => setSortKey(e.target.value));
  q('#sort-dir').addEventListener('click', toggleSortDir);
  q('#select-all-visible').addEventListener('click', selectAllVisible);
  q('#panel-toggle').addEventListener('click', togglePanel);
  q('#auto-crop').addEventListener('click', runAutoCrop);
  q('#import-selected').addEventListener('click', importSelected);
  let panelOpen = true;
  try{ panelOpen = localStorage.getItem('plm-panel-open') !== '0' }catch(e){}
  setPanelOpen(panelOpen);
  q('#open').addEventListener('click', openFolder);
  q('#switch-folder').addEventListener('click', switchFolder);
  q('#relocate-go').addEventListener('click', () => { q('#relocate-dlg').close(); relocateFolder(); });
  q('#relocate-later').addEventListener('click', () => { PLM.relocateDismissed = true; q('#relocate-dlg').close(); });
  q('#apply-ai').addEventListener('click', applyAiSuggestions);
  q('#export-report').addEventListener('click', () => exportEvaluationReport('html'));
  q('#export-report-csv').addEventListener('click', () => exportEvaluationReport('csv'));
  q('#go').addEventListener('click', confirmArrange);
  q('#dlg-cancel').addEventListener('click', () => q('#dlg').close());
  q('#dlg-go').addEventListener('click', doArrange);
  q('#bulk-keep').addEventListener('click', () => confirmBulkStatus('keep'));
  q('#bulk-reject').addEventListener('click', () => confirmBulkStatus('reject'));
  q('#bulk-neutral').addEventListener('click', () => confirmBulkStatus('neutral'));
  q('#bulk-clear-sel').addEventListener('click', clearPicked);
  q('#confirm-cancel').addEventListener('click', () => { pendingConfirm = null; q('#confirm-dlg').close(); });
  q('#confirm-go').addEventListener('click', runPendingConfirm);
  Lightbox.bind();

  setLoading(true, 'Loading folder…');
  try{
    // Every endpoint answers {ok, ..., state}, not the gallery state itself -
    // arrange and reconcile carry a sibling (result, recovered) alongside it,
    // which is why api() hands back the whole envelope rather than unwrapping
    // it for every caller. Every other call site extracts .state at the point
    // of use (see setStatus/setPercent/doArrange above); this is the same thing.
    PLM.state = (await api('api/review/state')).state;
    render();
  }catch(e){
    q('#grid').innerHTML = '<div class="empty">Could not load this review: ' + esc(e.message) + '</div>';
  } finally{ setLoading(false); }

  // Decisions for images that moved since they were reviewed are recovered by
  // content identity in the background, so the gallery is never held up by it.
  try{
    const j = await api('api/review/reconcile', {method:'POST'});
    if(j.recovered){ PLM.state = j.state; render(); say(j.recovered + ' decision(s) recovered for moved files.'); }
  }catch(e){ /* best effort - never blocks the review */ }
}

document.addEventListener('DOMContentLoaded', boot);
"""


def _presets_html() -> str:
    return "".join(
        f'<button class="preset" data-percent="{percent}">{percent}%</button>' for percent in KEEP_PRESETS
    )


def _reason_options_html() -> str:
    options = "".join(f'<option value="{value}">{label}</option>' for value, label in REASON_LABELS.items())
    # Blank first option: a reason is optional, so "nothing selected" must be
    # a real, distinct choice rather than defaulting to one of the two.
    return '<option value="">Reason (optional)</option>' + options


def build_js() -> str:
    """JS_TEMPLATE with its config-driven constants filled in.

    Plain substitution rather than an f-string: the template is full of literal
    braces (object literals, arrow functions), which an f-string would require
    escaping almost everywhere.
    """
    return JS_TEMPLATE.replace("__STATUS_LABELS__", json.dumps(REVIEW_STATUS_LABELS)).replace(
        "__REASON_LABELS__", json.dumps(REASON_LABELS)
    )


def build_page(title: str = "PeakPic") -> str:
    """The whole document. Empty of data - it fetches its own state on load.

    `title` only ever reaches the browser's own tab/window chrome (the
    <title> tag) - the toolbar itself never repeats the application's name,
    since that chrome already identifies it.
    """
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{CSS}</style></head><body>
<header>
  <div class="toolbar">
    <div class="toolbar-shell">
      <details class="toolbar-menu">
        <summary>▣ Folder</summary>
        <div class="menu-panel">
          <div class="group" data-group="folder">
            <span class="glabel">Folder</span>
            <span class="folder-name" id="folder">No folder open</span>
            <button id="switch-folder">Open&hellip;</button>
            <button id="open" title="Reveal the current folder in the OS file manager">Reveal</button>
          </div>
        </div>
      </details>
      <details class="toolbar-menu">
        <summary>◎ Review</summary>
        <div class="menu-panel">
          <div class="group" id="ai-group" data-group="ai">
            <span class="glabel">AI Suggests</span>
            {_presets_html()}
            <input type="number" id="percent" min="0" max="100" step="1" aria-label="AI suggestion threshold, percent">
            <span class="gunit">%</span>
            <button id="apply-ai" title="Set every Neutral, ranked image to the AI's current suggestion">Apply Suggestions</button>
          </div>
          <div class="group" data-group="review-actions">
            <span class="glabel">Review</span>
            <button id="go" class="primary" title="Move Keep to _Selected and Reject to _Rejected. Neutral is never moved.">Arrange Files</button>
            <button id="import-selected" title="Copy the current folder's Selected images into a destination folder">Import Selected</button>
          </div>
        </div>
      </details>
      <details class="toolbar-menu">
        <summary>⚙ Tools</summary>
        <div class="menu-panel">
          <div class="group" data-group="tools">
            <span class="glabel">Tools</span>
            <button id="auto-crop" title="Generate Lightroom crop metadata from the current folder's RAW images">Auto Crop for Lightroom</button>
            <button id="panel-toggle" class="on" title="Show/hide filters, sorting and view options">Filters &amp; Sorting</button>
          </div>
        </div>
      </details>
    </div>
    <div class="stats" id="stats">
      <span class="stat"><b id="c-total">0</b><span class="glabel">Total</span></span>
      <span class="stat keep"><b id="c-keep">0</b><span class="glabel">Keep</span></span>
      <span class="stat reject"><b id="c-reject">0</b><span class="glabel">Reject</span></span>
      <span class="stat neutral"><b id="c-neutral">0</b><span class="glabel">Neutral</span></span>
    </div>
  </div>

  <div class="bulkbar" id="bulk-bar" style="display:none">
    <span class="bulkcount" id="bulk-count">0 selected</span>
    <div class="bulkacts">
      <button class="bk keep" id="bulk-keep"><span class="ic">&#10003;</span> Keep</button>
      <button class="bk reject" id="bulk-reject"><span class="ic">&#10005;</span> Reject</button>
      <button class="bk neutral" id="bulk-neutral"><span class="ic">&#9675;</span> Neutral</button>
    </div>
    <div class="bulkacts">
      <button id="bulk-clear-sel">Clear Selection</button>
    </div>
  </div>

  <div class="statusbar">
    <span class="status" id="status">PeakPic workflow ready</span>
    <span class="status" id="workflow-stage" style="margin-left:10px">Stage: ready</span>
    <span class="status" id="workflow-status" style="margin-left:8px">Ranked — | Reviewed — | Imported —</span>
  </div>
</header>
<div class="notice" id="notice" style="display:none"></div>

<div class="loading-overlay" id="loading-overlay" style="display:none">
  <span class="spinner big"></span>
  <span id="loading-overlay-message">Loading&hellip;</span>
</div>

<div class="layout">
  <div class="grid" id="grid"><div class="empty">Loading...</div></div>

  <aside class="panel" id="side-panel">
    <div class="panel-section">
      <h3>View</h3>
      <div class="panel-filters">
        <button class="filter on" data-filter="all">All</button>
        <button class="filter" data-filter="keep">Keep</button>
        <button class="filter" data-filter="reject">Reject</button>
        <button class="filter" data-filter="neutral">Neutral</button>
        <button class="filter" data-filter="ai_keep">AI Keep</button>
        <button class="filter" data-filter="ai_reject">AI Reject</button>
        <button class="filter" data-filter="ai_keep_user_reject" title="AI Keep / You Reject">AI Keep / You Reject</button>
        <button class="filter" data-filter="ai_reject_user_keep" title="AI Reject / You Keep">AI Reject / You Keep</button>
      </div>
      <div class="panel-row">
        <button id="boxes" title="Show/hide the detector's bounding boxes on thumbnails">Detector Boxes</button>
        <button id="theme">Dark Mode</button>
      </div>
    </div>

    <div class="panel-section" id="category-section" style="display:none">
      <h3>Subject</h3>
      <div class="panel-filters" id="category-filters"></div>
    </div>

    <div class="panel-section">
      <h3>Sort</h3>
      <div class="panel-row">
        <select id="sort-key" aria-label="Sort by">
          <option value="score">AI Score</option>
          <option value="name">File Name</option>
          <option value="date">Capture Date</option>
        </select>
        <button id="sort-dir" aria-label="Toggle ascending/descending" title="Descending (click for ascending)">&darr;</button>
      </div>
    </div>

    <div class="panel-section">
      <h3>Selection</h3>
      <button id="select-all-visible" title="Select every image currently shown, after the active filter and sort">Select All Visible</button>
      <div class="panel-row"><span class="panel-hint" id="visible-count">0 of 0 visible</span></div>
    </div>

    <div class="panel-section" id="agreement-section" style="display:none">
      <h3>AI Agreement</h3>
      <div class="agreement-stats" id="agreement-stats"></div>
      <div class="panel-row">
        <button id="export-report" title="A standalone HTML report - agreement, confusion matrix, precision/recall/F1 and every disagreement - to archive and compare across model versions">Export Evaluation Report</button>
        <button id="export-report-csv" title="Just the per-image differences table, as CSV">CSV</button>
      </div>
    </div>
  </aside>
</div>

<dialog id="relocate-dlg"><div class="dlg">
  <h2>Folder Not Found</h2>
  <div class="sub">This folder could not be found - it may have moved, been renamed, or its drive letter
  may have changed. Please select its new location; every stored path (the ranking, any review
  decisions) will be updated automatically.</div>
  <div class="dlg-acts">
    <button id="relocate-later">Not Now</button>
    <button id="relocate-go" class="primary">Locate Folder&hellip;</button>
  </div>
</div></dialog>

<dialog id="dlg"><div class="dlg">
  <h2>Arrange Files</h2>
  <div class="sub">Files are moved, not copied. Nothing is ever overwritten - a name
  collision gets a numbered suffix. Neutral images are never touched.</div>
  <div class="plan">
    <b id="p-keep">0</b><span>Keep &rarr; <span class="dest" id="p-keep-dir"></span></span>
    <b id="p-reject">0</b><span>Reject &rarr; <span class="dest" id="p-reject-dir"></span></span>
    <b id="p-neutral">0</b><span>Neutral &rarr; <span class="dest">left where they are</span></span>
  </div>
  <div class="dlg-progress" id="dlg-progress" style="display:none">
    <span class="spinner"></span> Moving files&hellip;
  </div>
  <div class="dlg-acts">
    <button id="dlg-cancel">Cancel</button>
    <button id="dlg-go" class="primary">Arrange</button>
  </div>
</div></dialog>

<dialog id="confirm-dlg"><div class="dlg">
  <h2 id="confirm-title">Confirm</h2>
  <div class="sub" id="confirm-body"></div>
  <div class="dlg-acts">
    <button id="confirm-cancel">Cancel</button>
    <button id="confirm-go" class="primary">Confirm</button>
  </div>
</div></dialog>

<dialog id="lightbox">
  <div class="lb-stage" id="lb-stage">
    <button class="lb-close" id="lb-close" aria-label="Close" title="Close (Esc)">&times;</button>
    <button class="lb-nav prev" id="lb-prev" aria-label="Previous image" title="Previous">&#8249;</button>
    <button class="lb-nav next" id="lb-next" aria-label="Next image" title="Next">&#8250;</button>
    <div class="lb-img-wrap">
      <img class="lb-img" id="lb-img" alt="" draggable="false">
      <div class="lb-crop-overlay" id="lb-crop-overlay" style="display:none">
        <div class="lb-crop-box" id="lb-crop-box" style="display:none"></div>
      </div>
      <div class="lb-spinner" id="lb-spinner" style="display:none"></div>
    </div>
    <div class="lb-bottom">
      <div class="lb-info">
        <span id="lb-counter"></span>
        <span id="lb-filename"></span>
        <span id="lb-score"></span>
        <span class="lb-ai" id="lb-ai" style="display:none"></span>
        <span id="lb-zoom">Fit</span>
        <span class="lb-exp" id="lb-exp" title="Display only - never written to the RAW file or saved anywhere">
          <span aria-hidden="true">&#9728;</span>
          <button id="lb-exp-down" type="button" aria-label="Decrease exposure">&minus;</button>
          <span class="val" id="lb-exp-value">+0.0 EV</span>
          <button id="lb-exp-up" type="button" aria-label="Increase exposure">+</button>
        </span>
        <button class="lb-save" id="lb-save-jpeg" type="button" title="Save the camera's own JPEG for sharing">Save JPEG</button>
        <button class="lb-crop-toggle" id="lb-crop-toggle" type="button" title="Show the auto-crop rectangle over the photo">Crop Overlay</button>
        <select id="lb-reason" aria-label="Reason for overriding the model" title="Why you kept or rejected this, if you want to record it">
          {_reason_options_html()}
        </select>
        <input type="text" id="lb-reason-note" style="display:none" placeholder="Describe why..." aria-label="Describe the reason">
        <div class="lb-acts">
          <button class="keep" id="lb-keep" title="Keep (5 or K)">Keep</button>
          <button class="reject" id="lb-reject" title="Reject (0 or R)">Reject</button>
          <button class="neutral" id="lb-neutral" title="Neutral (U)">Neutral</button>
        </div>
        <div class="lb-status" id="lb-status"></div>
      </div>
      <div class="lb-film" id="lb-film"></div>
    </div>
  </div>
</dialog>

<script>{build_js()}</script>
</body></html>"""
