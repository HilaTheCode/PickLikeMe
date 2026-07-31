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
