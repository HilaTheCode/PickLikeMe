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
"""

from __future__ import annotations

import json

from .session import (
    STATE_AUTO_REJECTED,
    STATE_AUTO_SELECTED,
    STATE_MANUAL_KEEP,
    STATE_MANUAL_REJECT,
    STATE_UNRANKED,
)

# Keep-percentage presets. 25 is DEFAULT_SELECTION_PERCENTAGE, so the default
# is always one of the buttons rather than an invisible custom value.
KEEP_PRESETS = (5, 10, 20, 25, 35)

# Badge text per state. The wording says what happened and who decided it, so
# the photographer never has to work out why an image sits where it does.
STATE_LABELS = {
    STATE_MANUAL_KEEP: "You kept this",
    STATE_MANUAL_REJECT: "You rejected this",
    STATE_AUTO_SELECTED: "Selected by model",
    STATE_AUTO_REJECTED: "Rejected by model",
    STATE_UNRANKED: "No ranking",
}

CSS = """
*{box-sizing:border-box}
:root{--bg:#f8fafc;--panel:#fff;--panel-2:#f1f5f9;--text:#0f172a;--muted:#64748b;
--border:#e2e8f0;--accent:#2563eb;--good:#10b981;--bad:#ef4444;--warn:#f59e0b;--shadow:rgba(15,23,42,.08)}
:root[data-theme="dark"]{--bg:#0b1020;--panel:#141a2e;--panel-2:#1b2338;--text:#e2e8f0;
--muted:#94a3b8;--border:#26314d;--accent:#60a5fa;--good:#34d399;--bad:#f87171;--warn:#fbbf24;--shadow:rgba(0,0,0,.4)}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:20;background:var(--panel);border-bottom:1px solid var(--border);
padding:12px 20px;box-shadow:0 1px 3px var(--shadow)}
.title{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.title #theme{margin-left:auto;flex-shrink:0}
h1{font-size:17px;margin:0;font-weight:650}
.folder{font-size:12.5px;color:var(--muted);font-family:ui-monospace,Consolas,monospace}
.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:10px}
.group{display:flex;gap:6px;align-items:center}
.group>.lbl{font-size:12.5px;color:var(--muted);margin-right:2px}
button{font:inherit;font-size:13px;padding:6px 12px;border-radius:7px;border:1px solid var(--border);
background:var(--panel-2);color:var(--text);cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
button:disabled{opacity:.5;cursor:not-allowed}
input[type=number]{font:inherit;font-size:13px;width:64px;padding:6px 8px;border-radius:7px;
border:1px solid var(--border);background:var(--panel);color:var(--text)}
.counts{display:flex;gap:14px;margin-left:auto;font-size:13px;flex-wrap:wrap}
.count{display:flex;gap:5px;align-items:baseline}
.count b{font-variant-numeric:tabular-nums;font-size:15px}
.count.sel b{color:var(--good)}.count.rej b{color:var(--bad)}.count.unt b{color:var(--warn)}
.notice{margin:12px 20px 0;padding:10px 14px;border-radius:8px;border:1px dashed var(--warn);
background:var(--panel-2);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(232px,1fr));gap:14px;padding:16px 20px 60px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:11px;overflow:hidden;
display:flex;flex-direction:column;box-shadow:0 1px 2px var(--shadow)}
.card.sel{border-color:var(--good);border-width:2px}
.card.rej{border-color:var(--bad);border-width:2px}
.card.unt{border-style:dashed;border-color:var(--warn)}
.thumb-link{display:block;position:relative;cursor:zoom-in}
.thumb-link:hover .thumb{filter:brightness(1.06)}
.thumb{width:100%;aspect-ratio:1;background:var(--panel-2);object-fit:cover;display:block}
.ph{width:100%;aspect-ratio:1;background:var(--panel-2);display:flex;align-items:center;
justify-content:center;color:var(--muted);font-size:12.5px;text-align:center;padding:12px}
.meta{padding:9px 11px;display:flex;flex-direction:column;gap:5px;flex:1}
.name{font-size:12.5px;font-family:ui-monospace,Consolas,monospace;word-break:break-all}
.nums{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.badge{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:20px;
border:1px solid var(--border);background:var(--panel-2);width:fit-content}
.badge.manual_keep{background:var(--good);color:#fff;border-color:var(--good)}
.badge.manual_reject{background:var(--bad);color:#fff;border-color:var(--bad)}
.badge.auto_selected{color:var(--good);border-color:var(--good)}
.badge.auto_rejected{color:var(--muted)}
.badge.unranked{color:var(--warn);border-color:var(--warn)}
.acts{display:flex;gap:6px;padding:0 11px 11px}
.acts button{flex:1;padding:6px 4px}
.acts .keep.on{background:var(--good);border-color:var(--good);color:#fff}
.acts .rej.on{background:var(--bad);border-color:var(--bad);color:#fff}
.status{font-size:12.5px;color:var(--muted)}
.status.error{color:var(--bad)}
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
.lb-img-wrap{position:relative;max-width:92vw;max-height:74vh;display:flex}
.lb-img{max-width:92vw;max-height:74vh;object-fit:contain;user-select:none;-webkit-user-drag:none;
transform-origin:center center;display:block;transition:transform .08s ease-out}
.lb-img.dragging{transition:none}
.lb-img.grabbable{cursor:grab}.lb-img.grabbing{cursor:grabbing;transition:none}
.lb-spinner{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
color:#cbd5e1;font-size:13px;text-align:center;padding:20px}
.lb-badge{position:absolute;top:10px;left:10px;font-size:12px;padding:4px 10px;border-radius:20px;
font-weight:650;pointer-events:none}
.lb-badge.manual_keep{background:var(--good);color:#06281f}
.lb-badge.manual_reject{background:var(--bad);color:#2a0a0a}
.lb-badge.auto_selected{background:rgba(16,185,129,.2);color:#6ee7b7;border:1px solid var(--good)}
.lb-badge.auto_rejected{background:rgba(148,163,184,.15);color:#cbd5e1;border:1px solid #475569}
.lb-badge.unranked{background:rgba(245,158,11,.18);color:#fcd34d;border:1px solid var(--warn)}
.lb-close,.lb-nav{position:absolute;border-radius:50%;background:rgba(255,255,255,.08);
border:1px solid rgba(255,255,255,.18);color:#fff;display:flex;align-items:center;justify-content:center;
cursor:pointer}
.lb-close:hover,.lb-nav:hover{background:rgba(255,255,255,.2)}
.lb-nav:disabled{opacity:.2;cursor:not-allowed;background:rgba(255,255,255,.08)}
.lb-close{top:16px;right:20px;width:38px;height:38px;font-size:20px;line-height:1}
.lb-nav{top:50%;margin-top:-26px;width:52px;height:52px;font-size:24px}
.lb-nav.prev{left:16px}.lb-nav.next{right:16px}
.lb-top{position:absolute;top:0;left:70px;right:70px;padding:16px 0;display:flex;justify-content:center;
gap:16px;font-size:12.5px;color:#cbd5e1;flex-wrap:wrap;font-variant-numeric:tabular-nums}
.lb-bottom{position:absolute;left:0;right:0;bottom:0;display:flex;flex-direction:column;
align-items:center;gap:10px;padding:12px 16px 18px;background:linear-gradient(to top,rgba(0,0,0,.5),transparent)}
.lb-acts{display:flex;gap:12px}
.lb-acts button{font-size:15px;padding:11px 28px;border-radius:10px;font-weight:650;
background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.25);color:#e2e8f0}
.lb-acts .keep{border-color:var(--good);color:#6ee7b7}
.lb-acts .keep.on{background:var(--good);color:#06281f}
.lb-acts .rej{border-color:var(--bad);color:#fca5a5}
.lb-acts .rej.on{background:var(--bad);color:#2a0a0a}
.lb-status{font-size:12px;color:#cbd5e1;min-height:15px}
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
  labels: __STATE_LABELS__,
};

const esc = s => { const d = document.createElement('div'); d.textContent = s == null ? '' : s; return d.innerHTML; };
const q = s => document.querySelector(s);

function setTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  try{ localStorage.setItem('plm-theme', t) }catch(e){}
  // Guarded: this runs before DOMContentLoaded so the theme is applied without
  // a flash of the wrong one, which means the button may not be parsed yet.
  // Nothing cosmetic is allowed to throw here - an exception at this point
  // would abort the whole script and the gallery would never load at all.
  const button = q('#theme');
  if(button) button.textContent = t === 'dark' ? 'Light mode' : 'Dark mode';
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

// The gallery is rebuilt from state on every change. At a few thousand cards
// this is comfortably fast, and every card's markup then comes from exactly
// one place; `loading="lazy"` keeps the browser from fetching a thumbnail
// until it is actually scrolled into view.
function render(){
  const s = PLM.state;
  if(!s) return;
  q('#folder').textContent = s.input_folder;
  q('#c-total').textContent = s.counts.total.toLocaleString();
  q('#c-sel').textContent = s.counts.selected.toLocaleString();
  q('#c-rej').textContent = s.counts.rejected.toLocaleString();
  q('#c-unt').textContent = s.counts.untouched.toLocaleString();

  document.querySelectorAll('.preset').forEach(b => {
    b.classList.toggle('on', Number(b.dataset.percent) === s.keep_percent);
  });
  q('#percent').value = s.keep_percent;

  const notice = q('#notice');
  if(s.warnings && s.warnings.length){
    notice.innerHTML = s.warnings.map(esc).join('<br>');
    notice.style.display = '';
  } else {
    notice.style.display = 'none';
  }

  const grid = q('#grid');
  if(!s.images.length){
    grid.innerHTML = '<div class="empty">No images found in this folder.</div>';
    return;
  }
  grid.innerHTML = s.images.map((image, i) => card(image, i)).join('');
  grid.querySelectorAll('button[data-act]').forEach(b => {
    b.addEventListener('click', () => decide(b.dataset.path, b.dataset.act));
  });
  // Index into s.images, not image.rank: the lightbox navigates the gallery's
  // own displayed order (best-first, unranked last), which is what "next" and
  // "previous" should mean regardless of what the ranking says.
  grid.querySelectorAll('.thumb-link[data-index]').forEach(el => {
    el.addEventListener('click', () => Lightbox.open(Number(el.dataset.index)));
  });
}

function card(image, index){
  const cls = image.state === 'manual_keep' || image.state === 'auto_selected' ? 'sel'
            : image.state === 'unranked' ? 'unt' : 'rej';
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
    ? 'no score'
    : 'score ' + image.score.toFixed(4) + (image.rank ? ' &middot; rank ' + image.rank.toLocaleString() : '');
  return '<div class="card ' + cls + '">' + visual +
    '<div class="meta">' +
      '<div class="name" title="' + esc(image.image_path) + '">' + esc(image.filename) + '</div>' +
      '<div class="nums">' + score + '</div>' +
      '<span class="badge ' + esc(image.state) + '">' + esc(PLM.labels[image.state] || image.state) + '</span>' +
    '</div>' +
    '<div class="acts">' +
      '<button class="keep' + (image.decision === 'keep' ? ' on' : '') + '"' +
        ' data-act="keep" data-path="' + esc(image.image_path) + '">Keep</button>' +
      '<button class="rej' + (image.decision === 'reject' ? ' on' : '') + '"' +
        ' data-act="reject" data-path="' + esc(image.image_path) + '">Reject</button>' +
    '</div></div>';
}

// Clicking the button an image already carries clears the override, so the
// image returns to whatever the threshold says. That is the only way back to
// automatic, so it must be the obvious one.
async function decide(path, action){
  const image = PLM.state.images.find(i => i.image_path === path);
  const decision = image && image.decision === action ? null : action;
  try{
    const j = await api('api/review/decision', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image_path: path, decision: decision}),
    });
    PLM.state = j.state;
    render();
    say(decision ? 'Saved.' : 'Back to the model\\'s decision.');
  }catch(e){ say('Could not save: ' + e.message, true); }
}

// ===========================================================================
// Lightbox: full-screen in-app viewer, opened by clicking a card's thumbnail.
//
// Self-contained on purpose - it reads the gallery's image list off
// PLM.state and reuses decide() to persist Keep/Reject, but owns all of its
// own state (current index, zoom, pan, a small preload cache) behind this one
// object. Nothing outside this block touches that state directly, so the
// viewer can be extended - or lifted into its own file, if this ever needs to
// be reused elsewhere - without touching the gallery code above it.
// ===========================================================================
const Lightbox = (function(){
  const MIN_SCALE = 1;        // 1 == fit-to-screen, what every image opens at
  const MAX_SCALE = 8;
  const PRELOAD_RADIUS = 2;   // neighbours on each side kept decoded and warm
  const FILM_RADIUS = 8;      // neighbours on each side shown in the filmstrip

  let index = -1;
  let scale = 1;
  let pan = {x: 0, y: 0};
  let drag = null;            // {startX,startY,panX,panY,moved} while dragging
  let suppressClick = false;  // true for the click a drag-release also fires
  // path -> blob object URL for the full-size decode. Bounded to a small
  // window around the current index and revoked on eviction, so a session of
  // thousands of images cannot leak memory. This exists at all because
  // /preview is Cache-Control: no-store (shared with the analysis report's
  // own use of it), so the browser's own HTTP cache cannot do the job.
  const cache = new Map();

  function images(){ return (PLM.state && PLM.state.images) || []; }
  function current(){ return images()[index]; }
  function previewUrl(path){ return 'preview?path=' + encodeURIComponent(path); }

  function open(startIndex){
    index = startIndex;
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

  function go(newIndex){
    if(newIndex < 0 || newIndex >= images().length) return;
    index = newIndex;
    resetView();
    renderAll();
    preloadAround();
  }
  function next(){ go(index + 1); }
  function prev(){ go(index - 1); }

  function renderAll(){
    const image = current();
    if(!image) return;
    q('#lb-prev').disabled = index <= 0;
    q('#lb-next').disabled = index >= images().length - 1;
    q('#lb-counter').textContent = (index + 1) + ' / ' + images().length;
    q('#lb-filename').textContent = image.filename;
    q('#lb-score').textContent = image.score == null ? 'no score'
      : 'score ' + image.score.toFixed(4) + (image.rank ? ' · rank ' + image.rank.toLocaleString() : '');
    updateZoomIndicator();
    updateDecisionButtons(image);
    updateBadge(image);
    renderImage(image);
    renderFilmstrip();
  }

  function updateBadge(image){
    const badge = q('#lb-badge');
    badge.className = 'lb-badge ' + image.state;
    badge.textContent = PLM.labels[image.state] || image.state;
  }

  function updateDecisionButtons(image){
    q('#lb-keep').classList.toggle('on', image.decision === 'keep');
    q('#lb-reject').classList.toggle('on', image.decision === 'reject');
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
    q('#lb-img').style.transform = 'translate(' + pan.x + 'px,' + pan.y + 'px) scale(' + scale + ')';
  }

  function renderImage(image){
    const img = q('#lb-img');
    const spinner = q('#lb-spinner');
    img.style.transform = 'none';
    img.classList.toggle('grabbable', false);
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
    }).catch(() => {
      if(current() !== image) return;
      spinner.textContent = 'Could not load a full-size preview';
    });
  }

  function fetchPreview(path){
    const existing = cache.get(path);
    if(existing) return Promise.resolve(existing);
    return fetch(previewUrl(path), {cache: 'no-store'})
      .then(r => { if(!r.ok) throw new Error('HTTP ' + r.status); return r.blob(); })
      .then(blob => {
        const url = URL.createObjectURL(blob);
        cache.set(path, url);
        evictFarFromCurrent();
        return url;
      });
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

  // Reuses decide() rather than posting again: same validation, same
  // PLM.state refresh, same status reporting - the lightbox just also
  // advances afterwards, which is the one thing the gallery view never needs.
  async function decideAndAdvance(action){
    const image = current();
    if(!image) return;
    const wasLast = index >= images().length - 1;
    await decide(image.image_path, action);
    updateDecisionButtons(current());
    updateBadge(current());
    if(!wasLast) next();
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

  function onKeyDown(e){
    if(!q('#lightbox').open) return;
    if(e.key === 'ArrowRight') next();
    else if(e.key === 'ArrowLeft') prev();
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
    q('#lb-stage').addEventListener('click', onStageClick);
    q('#lb-stage').addEventListener('wheel', onWheel, {passive: false});
    q('#lb-img').addEventListener('dblclick', onDblClick);
    q('#lb-img').addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);
    window.addEventListener('resize', () => { clampPan(); applyTransform(); });
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
  }catch(e){ say('Could not change the percentage: ' + e.message, true); }
}

async function openFolder(){
  try{
    await api('open-folder?path=' + encodeURIComponent(PLM.state.input_folder));
  }catch(e){ say('Could not open the folder: ' + e.message, true); }
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
    q('#p-sel').textContent = plan.result.selected.toLocaleString();
    q('#p-rej').textContent = plan.result.rejected.toLocaleString();
    q('#p-unt').textContent = PLM.state.counts.untouched.toLocaleString();
    q('#p-sel-dir').textContent = plan.result.selected_dir;
    q('#p-rej-dir').textContent = plan.result.rejected_dir;
    q('#dlg').showModal();
  }catch(e){ say('Could not prepare the plan: ' + e.message, true); }
}

async function doArrange(){
  if(PLM.busy) return;
  PLM.busy = true;
  q('#go').disabled = true;
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
  finally{ PLM.busy = false; q('#go').disabled = false; q('#dlg').close(); }
}

async function boot(){
  const theme = q('#theme');
  // The button exists (see build_page), and setTheme has already labelled it
  // by now; both are written defensively so a markup change can never leave
  // the gallery unable to load.
  if(theme){
    theme.textContent = document.documentElement.getAttribute('data-theme') === 'dark'
      ? 'Light mode' : 'Dark mode';
    theme.addEventListener('click', () =>
      setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'));
  }
  document.querySelectorAll('.preset').forEach(b =>
    b.addEventListener('click', () => setPercent(b.dataset.percent)));
  q('#percent').addEventListener('change', e => setPercent(e.target.value));
  q('#boxes').addEventListener('click', () => {
    PLM.boxes = !PLM.boxes;
    q('#boxes').classList.toggle('on', PLM.boxes);
    q('#boxes').textContent = PLM.boxes ? 'Hide detector boxes' : 'Show detector boxes';
    render();
  });
  q('#open').addEventListener('click', openFolder);
  q('#go').addEventListener('click', confirmArrange);
  q('#dlg-cancel').addEventListener('click', () => q('#dlg').close());
  q('#dlg-go').addEventListener('click', doArrange);
  Lightbox.bind();

  try{
    // Every endpoint answers {ok, ..., state}, not the gallery state itself -
    // arrange and reconcile carry a sibling (result, recovered) alongside it,
    // which is why api() hands back the whole envelope rather than unwrapping
    // it for every caller. Every other call site extracts .state at the point
    // of use (see decide/setPercent/doArrange below); this is the same thing.
    PLM.state = (await api('api/review/state')).state;
    render();
  }catch(e){
    q('#grid').innerHTML = '<div class="empty">Could not load this review: ' + esc(e.message) + '</div>';
  }

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


def build_js() -> str:
    """JS_TEMPLATE with its config-driven constant filled in.

    Plain substitution rather than an f-string: the template is full of literal
    braces (object literals, arrow functions), which an f-string would require
    escaping almost everywhere.
    """
    return JS_TEMPLATE.replace("__STATE_LABELS__", json.dumps(STATE_LABELS))


def build_page(title: str = "PickLikeMe review") -> str:
    """The whole document. Empty of data - it fetches its own state on load."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{CSS}</style></head><body>
<header>
  <div class="title">
    <h1>Review</h1>
    <span class="folder" id="folder"></span>
    <button id="theme">Dark mode</button>
  </div>
  <div class="bar">
    <div class="group">
      <span class="lbl">Keep</span>
      {_presets_html()}
      <input type="number" id="percent" min="0" max="100" step="1" aria-label="Keep percentage">
      <span class="lbl">%</span>
    </div>
    <div class="group">
      <button id="boxes">Show detector boxes</button>
      <button id="open">Open Folder</button>
      <button id="go" class="primary">Arrange Files On Disk</button>
    </div>
    <div class="counts">
      <span class="count"><span class="lbl">Total</span><b id="c-total">0</b></span>
      <span class="count sel"><span class="lbl">Selected</span><b id="c-sel">0</b></span>
      <span class="count rej"><span class="lbl">Rejected</span><b id="c-rej">0</b></span>
      <span class="count unt"><span class="lbl">No ranking</span><b id="c-unt">0</b></span>
    </div>
  </div>
  <div class="bar"><span class="status" id="status"></span></div>
</header>
<div class="notice" id="notice" style="display:none"></div>
<div class="grid" id="grid"><div class="empty">Loading...</div></div>

<dialog id="dlg"><div class="dlg">
  <h2>Arrange files on disk</h2>
  <div class="sub">Files are moved, not copied. Nothing is ever overwritten - a name
  collision gets a numbered suffix.</div>
  <div class="plan">
    <b id="p-sel">0</b><span>Selected &rarr; <span class="dest" id="p-sel-dir"></span></span>
    <b id="p-rej">0</b><span>Rejected &rarr; <span class="dest" id="p-rej-dir"></span></span>
    <b id="p-unt">0</b><span>No ranking &rarr; <span class="dest">left where they are</span></span>
  </div>
  <div class="dlg-acts">
    <button id="dlg-cancel">Cancel</button>
    <button id="dlg-go" class="primary">Arrange</button>
  </div>
</div></dialog>

<dialog id="lightbox">
  <div class="lb-stage" id="lb-stage">
    <div class="lb-top">
      <span id="lb-counter"></span>
      <span id="lb-filename"></span>
      <span id="lb-score"></span>
      <span id="lb-zoom">Fit</span>
    </div>
    <button class="lb-close" id="lb-close" aria-label="Close" title="Close (Esc)">&times;</button>
    <button class="lb-nav prev" id="lb-prev" aria-label="Previous image" title="Previous">&#8249;</button>
    <button class="lb-nav next" id="lb-next" aria-label="Next image" title="Next">&#8250;</button>
    <div class="lb-img-wrap">
      <img class="lb-img" id="lb-img" alt="" draggable="false">
      <span class="lb-badge" id="lb-badge"></span>
      <div class="lb-spinner" id="lb-spinner" style="display:none"></div>
    </div>
    <div class="lb-bottom">
      <div class="lb-film" id="lb-film"></div>
      <div class="lb-acts">
        <button class="keep" id="lb-keep">Keep</button>
        <button class="rej" id="lb-reject">Reject</button>
      </div>
      <div class="lb-status" id="lb-status"></div>
    </div>
  </div>
</dialog>

<script>{build_js()}</script>
</body></html>"""
