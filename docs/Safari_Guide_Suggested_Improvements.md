# Safari Guide — Suggested Future Improvements

While writing `docs/Safari_Operational_Guide_Mac.md` (a field-operation
guide for photographers, not developers), several operational rough
edges and UX gaps became apparent from actually walking through the
application's real behavior end to end. None of these were fixed, and
**the operational guide itself describes PeakPic exactly as it works
today** — it was not softened, rewritten, or worked around to avoid
these issues. This document exists purely to record them separately, as
requested, for whoever picks up desktop-app development work next.

These are UX/product observations, not bug reports in the "something is
broken" sense — everything described below works as designed; the
concern is that the design has rough edges for a non-technical user
relying on it unattended in the field, over a multi-week trip, with no
one to ask.

---

## 1. Organize by Species has no confirmation step before moving files

Unlike **Organize** (which shows exact counts and destination paths and
asks Yes/No before moving anything — see `main_window.py`'s `_organize`)
and **Apply Cutoff** (which asks before overriding manual decisions),
**Organize by Species** starts moving files immediately after the user
clicks OK in the language/backend dialog — no "this will move N images
into species folders, continue?" step exists. Its progress dialog also
has no Cancel button, so once started, a run on a large folder can't be
interrupted.

**Why this matters in the field:** a photographer working quickly at the
end of a long day, muscle-memory-clicking through dialogs the way they
click through the Apply Cutoff / Organize confirmations, could click OK
on Organize by Species expecting a similar "are you sure" prompt that
isn't actually there — and the files start moving immediately.

**Suggested fix:** add a lightweight preview/confirm step (even just "N
images will be classified and moved into species folders inside
{folder}. Continue?"), matching the pattern already established by
`_organize()` and `_apply_cutoff()` elsewhere in the same file.

## 2. Organize by Species has no awareness of Keep/Reject/Neutral status

`species/arrange.py`'s `arrange_by_species()` moves **every** image in
the currently-open folder, regardless of review status — it has no
concept of "only move my Keep-marked images." The only way to
species-sort just the keepers is the two-step workflow this guide
documents (Organize first, then reopen `_Selected` and run Organize by
Species on that) — and forgetting that order silently species-sorts
rejected images too.

**Suggested fix:** either an optional "Keep-marked images only" checkbox
in the `SpeciesLanguageDialog`, or — more consistent with the rest of the
app's design — a clear inline note in that same dialog stating which
folder is about to be affected and that it operates on everything in it
regardless of decision, so a photographer who skipped this guide's
Section 2/3 notes still gets the warning at the point of action rather
than only in documentation.

## 3. No species filter in the main Review Window grid

Species handling is entirely file-system-based (Organize by Species
physically moves files into subfolders); there's no way to filter the
gallery grid itself to "just today's Lilac-breasted Rollers" without
either already having run Organize by Species (a move operation) or
manually browsing folders in Finder. The desktop app's Analytics
Dashboard *does* have richer, filterable, non-destructive per-species
browsing (via its Image Explorer), but that tool is dashboard-only and
not part of the main review flow this guide recommends for daily use.

**Suggested fix:** a lightweight, non-destructive "Species" filter option
in the main window's existing Filter dropdown (Section 4 of the
operational guide), reading from the same species-cache data the
Analytics Dashboard already has, would let a photographer review by
species without moving any files first.

## 4. Loupe's landmark/confidence detail requires switching to a different, more complex tool

The Loupe's **Boxes** overlay is intentionally simple (a detection box,
a runner-up box, an eye box, two eye position dots — no text, no
confidence numbers). The fuller picture — every measured landmark with
its own confidence value, and a full breakdown of exactly how a score
was calculated — only exists in the Analytics Dashboard's Image Explorer
(Visual Debug + Score Explanation), a substantially more complex,
filter-heavy, analyst-oriented tool.

**Why this matters in the field:** a photographer in the Loupe, looking
at one specific difficult image and wanting to understand *why* it
scored low, currently has to leave the Loupe, open a completely
different dialog with a different mental model (experiment browser,
tabs, an 11-checkbox overlay system), find the same image again in a
filterable list, and only then get the detail — a large amount of extra
navigation for what is fundamentally a "tell me more about the image
I'm already looking at" request.

**Suggested fix:** a lightweight "Why this score?" popover or expandable
panel directly in the Loupe — even just the per-metric raw/normalized/
weight/contribution breakdown that Score Explanation already computes,
attached to the image currently on screen — would close a real gap
between "quick field review" and "deep investigation" without requiring
the photographer to learn the Analytics Dashboard's more complex UI.

## 5. No at-a-glance model-readiness / offline-readiness indicator

Section 1 of the operational guide requires the user to open Terminal
and run `ls` commands against `cache/eye_models/` and
`~/.cache/huggingface/hub/` to confirm every model is actually
downloaded before departure — necessary because nothing in the
application's own UI currently answers "are all my models ready to work
offline?" in one place.

**Suggested fix:** a simple "Model Status" section (in Preferences, or a
new small dialog reachable from Help) listing each model this project
uses (subject detector, EyePose-v0/SuperAnimal-Bird, species classifier,
AI ranking backbone) with a clear Ready/Not Downloaded indicator per
model. This would let this guide's entire Section 1.2/1.4 collapse into
"open this one screen and confirm every row says Ready" — removing the
only place in the whole operational guide that currently requires
Terminal use at all.

## 6. No simple per-day summary export for a trip journal

The Analytics Dashboard's Run Summary/Species Analysis tabs show rich
per-day numbers, but there's no one-click way to export "today's
numbers" as a simple text/PDF summary a photographer could keep as a
running trip journal. Currently the only way to preserve a day's
statistics outside the app is to manually copy numbers by hand from the
Dashboard's tables.

**Suggested fix:** a plain-text or Markdown "Export Day Summary" action
on the Run Summary tab, reusing data already computed and displayed
there.

## 7. Ground Truth content-matching can silently produce zero matches, with no explanation surfaced at the point of failure

`Set User Decisions by Subfolders` matches images by content hash, not
filename or path (documented, deliberate behavior — see the tool's own
intro text). This means a re-exported or re-processed copy of an image
(a JPEG derived from a RAW, for instance) will not match its original and
silently produces a Neutral/no-match result rather than an error. The
tool's own preview step does show a count, so a mismatch is *visible* as
an unexpectedly low match count — but nothing actively explains *why* the
count is low in that moment.

**Suggested fix:** when the preview scan finds a meaningful number of
images in a selected folder that could not be matched to anything in the
currently-open Root Folder at all (not merely "already Neutral"), surface
that explicitly ("N image(s) in the folders you selected could not be
matched by content to any image in the Root Folder — if these are
copies/exports rather than the original files, that's why") rather than
leaving the photographer to work this out from documentation.

## 8. Recent Folders can't distinguish "a day's shoot" from "a subfolder PeakPic created"

The Recent Folders list (File menu) faithfully remembers every folder
explicitly opened via File → Open Folder — which, over a multi-week
trip, ends up including not just each day's top-level shoot folder but
also every `_Selected`, `_Rejected`, or species subfolder the
photographer happened to open directly at some point. There's no visual
distinction between the two kinds of entry in the list.

**Suggested fix:** no change needed to the *correctness* of this list
(it already behaves exactly as documented — see the operational guide's
own Section 9 entry and the desktop app's existing self-healing
behavior), but a visual indicator (e.g. an indented/greyed sub-entry for
a folder that lives inside another recently-opened one) would make a
long multi-week Recent Folders list easier to scan quickly.
