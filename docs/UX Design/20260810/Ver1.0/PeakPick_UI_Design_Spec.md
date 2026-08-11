# PeakPick — UI/UX Design Specification v1

## Purpose
This package is a complete visual direction for a first-pass redesign of PeakPick's three principal screens. It is intentionally implementation-oriented: use the SVGs as visual reference and this document as the behavioral/source-of-truth specification.

The design should be implemented as a coherent system, not as isolated incremental tweaks.

## Design principles
1. The photograph is always the dominant content.
2. Information hierarchy is explicit: primary actions > contextual controls > metadata.
3. Dense information is grouped into panels rather than scattered across the screen.
4. Algorithm results are explicit and traceable to an algorithm's latest successful run.
5. Grid is optimized for fast scanning; Loupe is optimized for deep review; Dashboard is optimized for understanding performance.
6. Avoid visual noise, tiny text, excessive borders, and duplicated information.
7. Every control must have a clear active/inactive/disabled state.
8. The UI must work well at approximately 1470px MacBook screen width.

## Visual language
- Theme: dark, professional, wildlife-photography workstation.
- Background: #0B1014.
- Primary panel: #141B21.
- Secondary panel/control: #1B242C.
- Divider: #2B3740.
- Main text: #F2F5F7.
- Secondary text: #9AA8B2.
- Accent / active selection: #F5C542.
- Keep: #42CC8E.
- Review: #F5C542.
- Reject: #EF4444.
- Filtered Out: #8B95A0.
- Skipped: #9B6BDB.
- Secondary blue information accent: #5AA7FF.
- Typography: Inter if available; otherwise a clean system sans-serif.
- Use semibold/bold for headings and primary values; regular for metadata.
- Rounded corners 7–12px. Avoid excessive pill-shaped UI.
- Primary controls ~44px high; secondary controls ~36px high.
- Use consistent 8px spacing rhythm.

## 1. Grid — main library
Reference: `01_Grid.svg`.

Purpose: scan hundreds of photographs rapidly, sort/filter/color by the selected algorithm source, and make Keep/Reject decisions.

### Layout
- Top toolbar: primary actions.
- Secondary toolbar: filter/sort/color/domain/search/view.
- Left sidebar: Recent Folders and Collections.
- Main area: 5-column thumbnail grid at 1470px-class screen.
- Bottom legend: Keep / Review / Reject / Filtered Out / Skipped.

### Primary toolbar
Keep large/high-value actions:
- Rank
- Apply Cutoff
- Keep
- Reject
- Clear Selection
- Export

Secondary controls:
- Algorithm Ran Last / selected Color Source
- Sort
- Filter
- Domain
- Search
- Burst mode
- View

Do not let the toolbar become a long undifferentiated row.

### Thumbnail anatomy
Each card should show:
1. rank position
2. favorite toggle
3. image
4. selected-source score, normalized as 0.xxx
5. filename
6. capture time
7. decision/status color
8. optional domain indicator
9. optional rating indicator

Do not show every algorithm score on each thumbnail.

Color must be controlled by the selected Color Source / selected latest run.

Categories:
- Keep
- Review
- Reject
- Filtered Out
- Skipped

"Skipped" is intentionally distinct from Reject.

## 2. Loupe — detailed image review
Reference: `02_Loupe.svg`.

Purpose: inspect one image deeply, compare algorithm results, inspect Elements, and make a manual decision.

### Layout
- Top: previous / image index / next.
- Left: Algorithm Results and metadata.
- Center: photograph.
- Right: Elements Source, detection confidence and technical information.
- Bottom: manual decisions, rating, zoom, brightness, tools.

### Algorithm Results
Show all available latest algorithm scores for the current image simultaneously.

Example:
- Bird Fusion — 0.946
- SuperAnimal Bird — 0.912
- Classic Vision — 0.873
- EyePose-v0 — 0.821
- Mammal Fusion — 0.421

The selected algorithm is highlighted.

Selecting an algorithm should also select it as the Elements source.

Do not use a separate hidden "last detector" state.

### Elements
The Elements source is the selected algorithm/run.

Display:
- Head confidence
- Left Eye confidence
- Right Eye confidence

Render the actual geometry produced by that selected algorithm.

Changing algorithm selection must change the overlays to that algorithm's own results.

Boxes and Elements may both be active.

### Score
Primary image score is prominent and formatted `0.xxx`.

Grid shows one selected-source score; Loupe shows all latest algorithm scores.

### Zoom
Support:
- trackpad pinch
- +/- keyboard
- mouse wheel if reliable
- presets: 100%, 150%, 200%, 300%

Zoom persists across image navigation.

### Keyboard
- Left/Right: navigate.
- K: Keep.
- R: Reject.
- Hebrew layout must recognize the physical K/R keys.
- +/-: zoom.
- Ctrl +/-: brightness.

### Important independence rule
Loupe must open a valid image even when no algorithm has detected anything. Detection is an optional overlay, never a prerequisite for image viewing.

## 3. Analytics Dashboard
Reference: `03_Analytics_Dashboard.svg`.

Purpose: understand what PeakPick is doing, compare algorithms/domains, and understand the library.

### Header
Tabs:
- Overview
- Algorithms
- Domains
- Trends
- Quality
- Export

Scope selectors:
- current folder
- date range

### KPI row
- Total Images
- Keep
- Reject
- Review
- Filtered Out
- Skipped

### Charts
Recommended panels:
- Score Distribution
- Score by Domain
- Top Algorithms (Average Score)
- Detection Success Rate
- Keep Rate by Algorithm
- Domain Breakdown
- Score Over Time
- Insights

Charts should answer questions, not merely decorate the screen.

### Insights
Use short natural-language findings, e.g.:
- which algorithm performs best in the current domain
- percentage with eye detections
- percentage kept
- number requiring manual review

## Algorithm/run source-of-truth
The UI must never infer results from whichever cache entry happened to be written last.

For every algorithm:
- retain latest successfully completed run only
- algorithm-specific object detection/crop/head/eye/score results remain logically associated with that algorithm
- a new run is promoted to latest only after successful completion
- old artifacts may then be removed if not shared
- no unlimited historical cache

## Design implementation rule
Do not implement the three screens as independent one-off layouts.

Create reusable components:
- PrimaryButton
- SecondaryButton
- ComboControl
- FilterChip
- ThumbnailCard
- StatusLegend
- AlgorithmResultRow
- ConfidenceBar
- Panel
- KPI Card
- Chart Card
- Navigation Control

Use one spacing, typography, color, radius and state system throughout.

## Do not
- preserve bad legacy layout just because it already exists
- add more controls to compensate for unclear hierarchy
- shrink fonts to make everything fit
- duplicate information
- make Loupe dependent on detection
- create a second sorting mechanism
- make cache ownership global per image
