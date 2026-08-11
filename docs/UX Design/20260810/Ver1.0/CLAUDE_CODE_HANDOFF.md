# Claude Code Handoff — PeakPick Visual Redesign

Implement the attached PeakPick design package as a coherent redesign, not as incremental cosmetic patches.

INPUTS:
- 01_Grid.svg
- 02_Loupe.svg
- 03_Analytics_Dashboard.svg
- 04_Toolbar.svg
- 05_Thumbnail.svg
- PeakPick_UI_Design_Spec.md

PRIORITY:
1. Preserve all working functionality.
2. Use the SVGs as visual/layout references.
3. Use the specification as behavioral and component guidance.
4. Build reusable UI components rather than hardcoding each screen separately.
5. Do not change ranking/scoring algorithms merely to implement the UI.
6. Do not break current Burst sorting.
7. Do not make Loupe dependent on detection.
8. Keep Grid score limited to the selected Color Source.
9. Show all latest algorithm scores in Loupe.
10. Selecting an algorithm in Loupe must select the same algorithm's Elements/detection results.
11. Respect algorithm-specific cache/result ownership and latest-run-only retention.

IMPORTANT:
The SVGs are design references, not literal HTML/SVG widgets to embed into the application. Recreate the UI using the project's native PyQt architecture and reusable widgets/components.

At the end:
- run focused UI tests
- run the full suite where practical
- report any visual compromises caused by the existing framework
- do not commit or push unless explicitly instructed.
