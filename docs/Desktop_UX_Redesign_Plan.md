# PeakPic Desktop Redesign Plan

## Current State Assessment

### What Exists (and Works)

The desktop application has **solid architectural foundations** that should be preserved:

1. **ReviewService** (services.py) - Cleanly exposes the backend
   - Wraps ReviewSession (the core review state machine)
   - Handles folder opening, ranking, status updates, arrangement
   - Provides thumbnail/preview paths
   - Ready to use; no rewrite needed

2. **ReviewSession** (review/session.py) - Core decision engine
   - Image gallery state (union of disk + ranking)
   - AI metadata (score, rank, suggestion)
   - Photographer's review status (Keep/Reject/Neutral)
   - Filtering, cutoff logic, arrangement
   - Thread-safe; mature and reliable

3. **Background Jobs** (core/jobs.py)
   - Production-grade thread pool
   - JobManager + JobSignal (Qt signals)
   - Cancellation support
   - No issues; ready for use

4. **Image Model** (models/image_model.py, models/image_item.py)
   - Qt Model/View compliant
   - Lazy thumbnail loading
   - Supports filtering by review status
   - Clean and simple; no rewrite needed

5. **Loupe Dialog** (dialogs/loupe_dialog.py)
   - Zoom/pan with mouse wheel
   - Per-image Keep/Reject/Neutral with reasons
   - Navigation (prev/next)
   - Needs UI polish but logic is sound

### What Doesn't Work (and Needs Redesign)

1. **Gallery View (GalleryView, MainWindow)**
   - Uses basic QListView.IconMode
   - Placeholder-level aesthetics
   - Thumbnail display shows only filename + status text
   - No visual hierarchy; text-heavy
   - Inefficient use of space
   - No clear visual feedback for AI suggestions
   - Does not match web UI density or clarity

2. **MainWindow Layout**
   - Central widget is text editor placeholder
   - Inspector panel adds clutter without clear purpose
   - Toolbar not designed for professional workflow
   - Menu bar is functional but uninspiring
   - No coherent visual design system

3. **Toolbar / Commands**
   - Commands exist but are not well organized
   - No professional icon set
   - No clear workflow progression
   - Similar to web menu but not optimized for keyboard/mouse mix

4. **Loupe Dialog UI**
   - Minimal design; works but doesn't inspire
   - No exposure control UI (though crop embed exists)
   - No visual indicator of image position in sequence
   - No clear visual hierarchy for score/rank/decision

5. **Theme / Visual Design**
   - No consistent semantic colors
   - No theme system (light/dark)
   - Placeholder default appearance
   - Typography not polished

6. **Keyboard Workflow**
   - Not optimized
   - No clear shortcut system
   - Professional users would find it frustrating

---

## Comparison with Web UI (Reference Images)

### Web Gallery (Image 1)
- **High thumbnail density** - compact grid, maximum image count visible
- **Clear information hierarchy** - filename, AI score, rank visible at glance
- **Visual decision indicators** - color dots/labels showing review status
- **AI suggestion clarity** - "AI suggests Reject" or "AI suggests Keep" is prominent
- **Hover states** - clear feedback on interaction
- **Fast review workflow** - mouse/keyboard combined, minimal clicks

### Web Loupe (Image 2)
- **Toolbar overlay on image** - Keep/Reject/Neutral buttons, reason dropdown
- **Minimal chrome** - image dominates, controls overlay gracefully
- **Quick navigation** - keyboard and mouse both supported
- **Score/rank visible** - always visible while reviewing
- **Fit-to-window default** - image fills screen
- **Scroll/zoom controls** - intuitive and responsive

### Desktop Prototype (Images 3-5)
- **Placeholder text editor** - "PeakPic Desktop / Open a folder to begin"
- **No visual design** - bare Qt widgets
- **Desktop panel at bottom** - image grid with basic thumbnails
- **Inspector panel** - shows minimal info
- **Loupe in new window** - workflow broken by mode switch

---

## Problem Analysis

### The Core Issue
The **presentation layer** (UI) doesn't match the **backend capability** or the **web reference**.

The backend is mature:
- ✅ Review session state machine works correctly
- ✅ Services layer is clean
- ✅ Background jobs are production-ready
- ✅ Thumbnail generation is fast
- ✅ AI ranking is working

But the UI is proto-tier:
- ❌ Visual hierarchy is poor
- ❌ Information density is too low
- ❌ Workflow is clumsy (mode-switching via dialogs)
- ❌ No design system (colors, typography, icons)
- ❌ Doesn't compete with professional tools

**Result:** A photographer would use the web interface instead, even on their local desktop.

---

## Design Goals

Photographer-centric, not widget-centric:

1. **The gallery is the workspace**
   - Occupies most of the window
   - Shows many images at once
   - Decision state is visually obvious
   - Quick Keep/Reject is 1-2 keypresses

2. **Loupe is fast and full-screen**
   - Minimal toolbar
   - Image-centric (no clutter)
   - Zoom/pan/navigation are smooth
   - Preserve all UI state (zoom level, pan position, scroll in gallery)

3. **Keyboard-first workflow**
   - K = Keep, R = Reject, N = Neutral
   - Arrow keys navigate gallery
   - Ctrl+Wheel = zoom in Loupe
   - Mouse wheel = next/prev in Loupe
   - No reaching for the mouse unless you want to

4. **Professional appearance**
   - Semantic color system (green/red/gray/blue)
   - Dark theme by default (photographers prefer it)
   - Polished typography
   - Proper spacing and alignment
   - Professional icon set

5. **Workflow clarity**
   - Gallery shows exactly what the user needs
   - Loupe review is uncluttered
   - Progress is always visible
   - No confusion about what happens next

---

## Proposed Architecture Changes (Minimal)

### Do NOT Change
- ReviewSession (core logic)
- ReviewService (backend wrapper)
- JobManager (background execution)
- ImageModel (Qt model)
- Existing services/infrastructure

### Improve Only (Priority Order)
1. **Gallery widget** - replace text rendering with styled thumbnail cards and better layout
2. **Thumbnail cards** - display filename, score, rank, review status with proper hierarchy
3. **Loupe dialog** - polish UI, minimize chrome, improve image-centric layout
4. **Toolbar** - organize commands logically, add icons, clarify workflow
5. **Color system** - add semantic palette (keep/reject/neutral/selection)

### No Backend Changes
- AI ranking stays as-is
- ReviewSession logic is untouched
- Sidecar/annotation storage unchanged
- Export/organize logic unchanged

---

## Phase 1: Desktop Presentation Layer

### Milestone 1: Gallery Layout Redesign
**Goal:** Create a professional gallery workspace with high thumbnail density

- Redesign MainWindow central area
- Remove placeholder text editor
- Implement efficient grid layout for thumbnails
- Add proper spacing and margins
- Support high-density display (40+ thumbnails without scrolling)
- Maintain filter state and scrolling position
- Keyboard navigation support (arrow keys, Enter for Loupe)

### Milestone 2: Thumbnail Card Redesign
**Goal:** Display rich information in compact, scannable format

- Design thumbnail card widget
- Display: image thumbnail, filename, AI score, rank, review status
- Visual feedback for hover/selection states
- Clear indication of AI suggestion (Keep vs Reject)
- Proper typography and spacing
- Lazy loading for performance

### Milestone 3: Loupe Redesign
**Goal:** Create minimal, image-centric review experience

- Minimize toolbar chrome
- Make image the focal point (maximize viewing area)
- Show score/rank/status overlay (non-intrusive)
- Improve decision controls (buttons + reason dropdown)
- Smooth zoom/pan behavior
- Preserve state on close

### Milestone 4: Toolbar Redesign
**Goal:** Organize commands for efficient workflow

- Group related commands logically
- Add/improve icons for clarity
- Clarify workflow progression (Open → Rank → Review → Organize)
- Keyboard shortcut hints
- Status indicators (GPU, progress)

### Milestone 5: Theme System
**Goal:** Support light/dark modes with semantic colors

- Define semantic color palette (green/red/gray/blue)
- Implement theme switching (dark by default)
- Persist theme preference
- Apply to all widgets consistently
- Support both light and dark variants

---

## Success Criteria

After Phase 1, the desktop app should:

1. **Gallery Experience**
   - ✅ Display 40+ thumbnails without scrolling on 1920x1080
   - ✅ Show filename, score, rank, review status at glance
   - ✅ Support K/R/N keyboard shortcuts
   - ✅ Navigate with arrow keys smoothly
   - ✅ Open Loupe with Enter or double-click

2. **Loupe Experience**
   - ✅ Image fills most of the window (minimal chrome)
   - ✅ Zoom with Ctrl+MouseWheel
   - ✅ Navigate with arrow keys or mouse wheel
   - ✅ Preserve zoom level and pan position
   - ✅ Show image index, score, rank, status

3. **Performance**
   - ✅ Scrolling gallery is smooth (60 FPS)
   - ✅ Thumbnail loading doesn't stutter
   - ✅ Loupe zoom is responsive
   - ✅ Decision persistence is instant

4. **Workflow**
   - ✅ Keyboard-first (K/R/N/arrows)
   - ✅ Minimal mode-switching (Loupe is modal, Esc returns)
   - ✅ No confusion about next action
   - ✅ Professional appearance throughout

---

## Implementation Strategy

1. **Work in phase order**
   - Gallery layout first (biggest structural change)
   - Thumbnails next (visual iteration on gallery)
   - Loupe (isolated improvement)
   - Toolbar (navigation/workflow)
   - Theme (crosses all)

2. **Commit each milestone separately**
   - Incremental, testable changes
   - Easy to revert if issues arise
   - Clear history for future reference

3. **Validate at each step**
   - Run regression tests after each milestone
   - Test keyboard shortcuts
   - Verify performance (no stuttering/lag)
   - Check both Linux/macOS/Windows if available

4. **Preserve backward compatibility**
   - No API changes
   - Database schema untouched
   - Export format unchanged
   - Command-line interface stable

---

## Why This Approach Works

**Minimal Risk:** Architecture is untouched; only the presentation layer changes.

**Maximum Impact:** Photographers will immediately notice improved gallery workflow.

**Sustainable:** Each improvement is independent; easy to debug and validate.

**User-Centric:** Design follows photographer workflow, not widget limitations.

**Maintainable:** Clear structure makes future improvements straightforward.

---

## Progress Log

### Phase 1 — complete
1. ✅ Milestone 1: Gallery Layout Redesign — gallery fills the window, placeholder panels removed.
2. ✅ Milestone 2: Thumbnail Card Redesign — custom-painted cards (filename, score, rank, status, AI suggestion).
3. ✅ Milestone 3: Loupe Redesign — single dark overlay bar, Ctrl+wheel zoom / wheel navigate, persisted zoom mode, EV preview.
4. ✅ Milestone 4: Toolbar Redesign — shared QActions (menu+toolbar), icons, workflow-ordered groups, GPU status.
5. ✅ Milestone 5: Theme System — dark/light semantic palette, switchable from the View menu, persisted, no restart required.

### Phase 2 — complete
6. ✅ Milestone 6: Keyboard-First Workflow — multi-select bulk Keep/Reject/Neutral, Space-to-toggle-selection, rank-display bug fix, status bar Keep/Reject/Neutral breakdown, contextual empty-state messaging.

### Phase 3 — complete (scoped)
7. ✅ Milestone 7: Typography & Spacing Polish — fixed a real thumbnail-card overflow bug (badge line was painting past the card's bottom edge), bold filename, dynamic AI-badge positioning.
   - Deliberately deferred: bespoke icon set (asset-design decision for the user), further large-folder perf work (no evidence of a problem after review), cancel buttons on backend operations that don't support cooperative cancellation (would be misleading UI).

### Phase 4 — complete: Visual QA pass
Every prior milestone was validated with offscreen smoke tests that checked for crashes and inspected object state, but never actually looked at rendered pixels. Capturing and reading real screenshots (native Qt rendering, Fusion style, not the font-less offscreen QPA platform) surfaced six real bugs no logic-only test could have caught:

8. ✅ Milestone 8: Visual QA Pass — (1) Windows' native style ignored QSS on QMenuBar/QToolBar/QStatusBar, so theme switching barely changed the chrome — fixed by switching to the Fusion style. (2) The status bar's folder label was created and never updated anywhere — pre-existing bug, now fixed. (3) Gallery card status-badge text overlapped the score/rank line above it (mixing rect-based and baseline-based `drawText` calls) — fixed to use one consistent API. (4) The Loupe rendered images tiny in a mostly-empty viewport because `fitInView()` ran before the dialog had its final layout size — fixed with a `resizeEvent` override that re-fits. Also widened the dark theme's Keep/Reject/Neutral background separation, which was clustering into the same luminance band.
9. ✅ Milestone 9: Real Preferences Dialog — replaced the "will be implemented in a later phase" stub with a functional dialog (theme + default species language). Caught and fixed a Qt radio-button auto-grouping bug in the same pass (see commit).
10. ✅ Milestone 10: RankDialog Width Fix — the checkpoint path field was narrow enough to show "nodel_checkpoint.pt" (a truncated/scrolled "model_checkpoint.pt"), reading like a typo. Fixed with a minimum dialog width.
11. ✅ Milestone 11: Selection Visibility & Progress Bar Theming — multi-select was functionally correct (confirmed by precise pixel sampling after an eyeballed screenshot first suggested a false alarm) but too visually weak in a dense grid; strengthened to a 3px border plus a translucent selection wash. `QProgressBar` had no theme.py rule at all, so its unfilled portion was stark white against the dark theme — added one.

**Known gap, not fixed (out of scope for this redesign):** `MainWindow`'s `QSettings("PeakPic", "PeakPicDesktop")` has no test/dev isolation — every construction (real app, pytest, or an ad-hoc script) reads/writes the same real per-machine settings store. Screenshot scripts during this phase, and apparently the pre-existing pytest suite too, accumulated test-path artifacts in real user settings; cleaned up manually after each test run. Worth a dedicated test-isolation fix (e.g. `QSettings::IniFormat` pointed at a tmp path in tests) at some point, but that's test infrastructure, not desktop presentation-layer work.

Every milestone was validated against the existing desktop regression suite (30 tests) plus either an offscreen-Qt smoke test or, from Phase 4 onward, actual captured screenshots read and inspected before considering a fix complete - and committed separately per the user's instruction.

---

## Stabilization & Performance Sprint (post-redesign)

Scope: fix regressions only. No architecture changes, no ReviewSession/ReviewService/JobManager/AI-pipeline rewrites.

### Priority 1 — Open Folder blocks on a modal "Loading categories…" dialog

**Symptom:** opening a ~4,400-image folder shows the gallery almost immediately, but a *modal* `QProgressDialog` stays up ("Loading categories…", ~1%/min or slower), blocking all interaction even though the gallery is already usable. The web app opens the same folder fast with no equivalent block.

**Method:** traced Desktop's and Web's open-folder code paths line-by-line (not guessed), then benchmarked the backend in isolation to separate "shared backend cost" from "desktop-only UI behavior."

**Desktop vs Web comparison** (both call the identical `ReviewSession.open_folder()` → `.load()`):

| Stage | Web | Desktop (before fix) | Blocking? | Execution time (4,400 imgs) | Purpose |
|---|---|---|---|---|---|
| Folder enumeration + ranked-file merge | ✅ | ✅ | Both: yes, briefly | ~0.26s | List files, merge any existing ranking CSV |
| `ReviewSession.load()` background thread spawn | ✅ | ✅ | Neither (backgrounded) | n/a (async) | Kicks off per-image metadata fill |
| Bounded 0.15s join on that thread | ✅ | ✅ | Both: yes, capped at 150ms | ≤0.15s | Lets a small/fast folder finish inline |
| Initial HTTP/IPC response → gallery renders | ✅ (`setLoading` off after `await api()`) | ✅ (`_refresh_from_state`) | Both: no, once response lands | fast | Gallery becomes usable |
| Remaining `captured_at` / `detected_category` backfill (per-image `AnnotationStore` SQLite lookups) | ✅ runs on | ✅ runs on | **Web: no** (status line only) / **Desktop (before): yes** (modal stays open) | 2.3–3.3s cold (0.53–0.75ms/img), ~0.1ms/img warm | Fills fields the gallery doesn't even display |
| Surfacing ongoing background progress | Plain non-blocking status line (`page.py` `say(stageText)`) | Modal `QProgressDialog`, `Qt.WindowModal`, closed only on full completion | **Desktop was the outlier** | — | — |

**Root cause:** Desktop-specific UI code, not the shared backend. `_start_open_folder()` kept the modal `QProgressDialog` open until `ReviewSession`'s background thread fully finished (`_finish_open_folder`, driven by a timer polling `loading_state()`), instead of closing it once the gallery already had real data — mirroring the web app's `setLoading(false)` immediately after its initial request, well before its own background categorization finishes (`page.py`'s `switchFolder()` / `render()`'s `stageText` status line).

A secondary, real cost was also measured and is worth recording even though it wasn't the primary regression: `AnnotationStore._cached_per_file()` commits every row individually (`with self._conn: INSERT OR REPLACE ...`) instead of batching. Isolated benchmark: 4,400 individual commits = 6.3s (1.43ms/commit) vs. one batched commit for the same rows = 0.005s — a 1327x difference. This cost is paid identically by Web and Desktop, so it is *not* "the" desktop regression, and per the "do not rewrite the backend" constraint it was **not touched**. Flagging it here as a follow-up candidate: it likely explains a meaningful share of the real-world "~1%/minute" the user observed on actual RAW files (the synthetic-JPEG benchmark above shows the commit overhead in isolation but can't reproduce real RAW decode cost or antivirus/disk interference on the user's machine, so it should be read as a lower bound, not the full explanation).

**Fix (main_window.py):** decoupled the modal dialog's lifetime from full background completion.
- Added `_background_load_active`, tracked separately from `_open_folder_in_progress`. The former stays true only long enough to keep the status-bar progress ticking; the latter now clears the moment the gallery has data.
- `_start_open_folder()` now closes the modal dialog and clears `_open_folder_in_progress` right after `_refresh_from_state(result)` — as soon as scores, ranks, thumbnails, and review status are populated — instead of waiting for `captured_at`/`detected_category` (fields the gallery never displays) to finish loading.
- Remaining background progress now only updates the status bar (`_loading_progress` bar, previously created but never made visible — a second, smaller pre-existing bug fixed along the way — plus `_set_status()` text), matching the web app's non-blocking status line exactly.
- `_poll_loading_state()`'s guard switched from `_open_folder_in_progress` to `_background_load_active` so status-bar polling keeps running for the remainder of the background pass without a modal gating it.

**Bug found and fixed while implementing this:** `QProgressDialog.close()`/`.cancel()` emits its own `canceled` signal whenever the dialog hasn't reached its maximum value — true for every one of these now-earlier programmatic closes. Left connected, that signal re-entered `_cancel_open_folder()`, which would incorrectly restore the *previous* session right after a folder had just opened successfully (and, in a narrower ordering, threw `AttributeError` from re-entrant `deleteLater()` on an already-cleared dialog reference). Fixed by disconnecting `canceled` and clearing `self._folder_load_dialog` *before* calling `.close()`/`.deleteLater()` in `_hide_folder_load_dialog()`, so any re-entrant call sees `None` and no-ops, and `close()` can no longer trigger a cancel at all.

**Result:** time-to-usable-gallery is unchanged (it was already fast); time-to-interactive now matches it, instead of waiting for the full background metadata pass. Desktop's behavior now matches Web's: initial render blocks briefly, background categorization never blocks again.

**Test fallout:** `test_open_folder_failure_restores_previous_session` (tests/test_desktop_shell.py) started hanging under the offscreen Qt test platform once `_open_folder_in_progress` began clearing synchronously instead of only via the timer — a second, unmonkeypatched `_start_open_folder()` call in that test used to be silently no-op'd by the (accidental, timing-dependent) "already loading" guard, and now genuinely reaches `_handle_open_folder_failure()`'s `QMessageBox.warning()`, which blocks forever with no one to click it under `QT_QPA_PLATFORM=offscreen`. Fixed by stubbing `QMessageBox.warning` in that test, the same pattern `test_desktop_workflow.py` already uses for `loupe_dialog`'s message boxes. All 30 desktop tests pass.

### Priority 2 — Gallery grid left-aligned with wasted space on the right

**Symptom:** the thumbnail grid packed against the viewport's left edge, leaving a large unused strip on the right in a wide window.

**Root cause:** `QListView` in icon mode always starts each row at the viewport's left edge; a viewport wider than an exact multiple of the 220px card cell leaves the remainder as dead space on the right only. There is no built-in "center the grid" option.

**Fix (gallery_view.py):** `GalleryView._center_grid()`, driven from `resizeEvent`, computes how many whole card columns fit the viewport and adds equal left/right `setViewportMargins()` to absorb the remainder symmetrically — the row/column wrapping and scrolling logic itself is untouched.

**Two real bugs found and fixed while implementing this** (both confirmed by direct measurement — offscreen screenshots plus `visualRect()`, not guessed):
- `getViewportMargins()` doesn't exist on this PySide6 build; the resulting unhandled `AttributeError`, raised inside a Qt virtual-function override (`resizeEvent`), crashed the process (`Segmentation fault`) instead of raising a normal Python exception. Switched to `viewportMargins()`.
- Driving the recentering off `viewportEvent` instead of `resizeEvent` set up an unbounded resize↔margin feedback loop (`setViewportMargins` resizes the viewport, which re-fires `viewportEvent`) and stack-overflowed. `resizeEvent` — this widget's own geometry, never touched by `setViewportMargins` — has no such loop.
- Shrinking the viewport to *exactly* `columns * item_width` made `QListView`'s own grid layout drop a column right at that exact-multiple boundary (measured: a 1384px-wide viewport that should fit 6 220px columns only rendered 5 once margins made it exactly 1320px). Leaving 1px of slack unclaimed avoids the boundary.

**Result:** verified across five widths (900–1600px) via `visualRect()` measurement — margins symmetric to within 1px at every width, column pitch a constant 220px. Screenshots confirm the grid centers correctly with no visual regression to wrapping or scrolling.

### Priority 3 — Rank by AI: "No RAW images found" on an already-organized folder

**Symptom:** running Rank by AI again on a folder that had already been arranged (via Organize) reported "No RAW images found," even though the folder's images were still there, just moved into `_Selected`/`_Rejected`.

**Root cause:** recursive search was never the problem — `UnlabeledImageDataset.from_folder()` already walks with `rglob()`. `from_folder()` deliberately excludes `_Selected`/`_Rejected` (so a second run never re-ranks images a previous Organize already filed) and `.picklikeme` (the ranking sidecar). Once every RAW in a folder has been moved into `_Selected`/`_Rejected`, a later Rank by AI run on that same folder legitimately finds zero *un-arranged* images — the generic "no RAW images found" message was technically true but misleading, since the images are right there, just already handled.

**Fix (rank.py):** `_folder_already_organized()` reruns the same `from_folder()` scan without the exclusions, only on that error path (rare, so the extra walk is cheap), to tell "genuinely empty" apart from "already organized," and raises a message that says which one it is. Applied to both `rank_folder()` (the desktop entry point) and the CLI's `main()`.

**Verified directly** (not just by reading the code) with three scenarios: a genuinely empty folder still gets the original message; a folder with everything moved into `_Selected`/`_Rejected` gets the new, accurate message; and a folder with RAWs nested in a subfolder passes the raw-image check and proceeds into real ranking work — confirming recursive search was never actually broken.

### Priority 4 — Detector Boxes don't appear anywhere

**Symptom:** reported as boxes never appearing in the Gallery at all (Loupe's overlay was to be added to match).

**Investigation, not assumption:** rather than starting from "the overlay must be broken," this was verified end-to-end first. A direct reproduction — a real `MainWindow` and a real `LoupeDialog`, a synthetic image with a manually-recorded detection written the same way `rank_folder`'s preprocessing would, offscreen screenshots — showed the Gallery delegate, the toggle wiring (`MainWindow._show_detector_boxes`, cache keyed by `(path, with_boxes)`), and the Loupe's `QGraphicsScene` overlay with its coordinate transform **all draw correctly** once given real detection data. Both already shared `_show_detector_boxes` as one global toggle, exactly as requested — no UI defect existed to fix.

**Actual root cause, found by tracing the data the UI was reading:** `preprocess.build_cache()`'s incremental-skip check (the crop-building step `rank_folder` calls before scoring) only tested whether an image's crop PNG already existed before deciding to skip re-detecting it entirely. `save_detections()` — which writes the sidecar `review`'s `DetectionCache` reads for the overlay — was added to the codebase after this crop-cache format existed. Any crop built before then (or any sidecar lost for any other reason while its crop survived) is treated as fully cached forever: no re-run of Rank by AI could ever fix it, because the crop existing was the *only* thing checked. That silently explains a folder where boxes never appear on any image, with no error surfaced anywhere.

**Fix (preprocess.py):** `_refill()` now also requires the detections sidecar to exist before treating an image as already cached, so a normal incremental Rank by AI re-run heals exactly the images missing it — without forcing an expensive full rebuild of an otherwise fully-cached folder via `--force-preprocess`.

**Verified two ways:** two new tests in `test_preprocess_pipeline.py` pin both directions — a crop with no sidecar is reprocessed and gets one; a normal cache hit (both crop and sidecar present) still skips exactly as before, so the documented idempotent behavior for the healthy case is unchanged. And the original visual repro (screenshots of both Gallery and Loupe drawing the green/amber boxes correctly) stands as evidence the painting side was already correct.

### Final validation

All four priorities fixed and committed separately. Full desktop regression suite plus the directly-affected suites (`test_preprocess_pipeline.py`, `test_fn_overlay.py`, `test_rank.py`) pass — 100+ tests, zero failures. Time-to-usable-gallery on Desktop now matches Web's responsiveness (Priority 1); the remaining fixes (2–4) were UI/data correctness issues independent of load performance.
