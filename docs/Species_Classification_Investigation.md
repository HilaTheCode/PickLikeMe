# Species Classification Investigation

**Date:** 2026-08-03. **Scope:** Forensic investigation only, per explicit instruction - no product code was changed, no thresholds were tuned, no species list was edited. Tool used: `tools/debug_species_pipeline.py` (built for this investigation, modeled on `tools/debug_eye_pipeline.py` - reuses real production functions end to end, reimplements nothing, not imported by the product).

**Sample:** 23 real photos, drawn from the same `Test data for PickLikeMe` folder used in the EyePose investigation. Convenient timing: "Organize by Species" had already been run against this exact folder in production before this investigation started, so every image's **current folder location already records production's real prediction** - this report cites that as-found state directly rather than only the debug tool's own re-run, and cross-checks both agree (the debug tool's `predicted_species` column reproduces the folder name in every case checked). 12 of the 23 images have an independently-known true species - 3 from this project's own EyePose Investigation Phase 1 report (which named species while describing test poses), 9 from direct visual inspection of `01_original.jpg`/`03_selected_crop.jpg` during this investigation. The remaining 11 are reported with their prediction and margin data but not claimed as verified.

---

## Critical question 1: does BioCLIP classify the selected crop, the full frame, or something else?

**Answer: (C) the full original image. Proven by reading the code, not assumed.**

Traced the exact call chain from "Organize by Species" to the classifier:

```
desktop/main_window.py: _organize_by_species()
  -> desktop/services.py: ReviewService.organize_by_species()
    -> species/arrange.py: arrange_by_species()
      -> species/cache.py: SpeciesCache.get_or_classify(image_path, classifier)
        -> analyzer/contactsheets.py: load_source_image(image_path)   # <-- Stage 1
        -> species/bioclip_classifier.py: BioClipSpeciesClassifier.classify(image)
```

`species/cache.py:107` reads:

```python
image = load_source_image(str(image_path))
prediction = classifier.classify(image)
```

And `load_source_image`'s own docstring (`analyzer/contactsheets.py:104-115`) states, verbatim:

> "The whole original frame, as cheaply as the format allows. **Never the cached bird crop.**"

`bird_crop` (the subject detector/crop cache) is never imported by `species/arrange.py`, `species/cache.py`, or `species/bioclip_classifier.py` - confirmed by reading all three files' imports directly, not by absence-of-evidence. `species/arrange.py`'s own module docstring says the same thing independently: "never imports anything from bird_crop, rank, train, or review."

For a RAW file (every image in the sample), `load_source_image` returns the camera's own **embedded JPEG preview** (full frame, full resolution, no crop) - not even a full demosaic, chosen for speed ("costs milliseconds, where a demosaic costs about a second"). BioCLIP's own `_preprocess` then does `Resize(224, bicubic) -> CenterCrop(224, 224) -> Normalize` on that whole frame (confirmed by inspecting the live `preprocess` object BioCLIP itself constructs). So the actual input tensor the model ever sees is: **the center 224x224 region of a Resize-to-224-on-the-short-side version of the entire original photo** - background, sky, water, reeds and all, with the subject's actual size in that tensor exactly as small (or off-center) as it was in the original frame.

**Confidence level:** Proven. Read the real call chain and the real docstring, not inferred from behavior.

---

## Critical question 2: is the EyePose/ranking crop policy also optimal for species classification?

**Do not implement anything - answer only, with measured data.**

The debug tool ran BioCLIP's own real preprocessing/model/text-embeddings on both inputs for every one of the 23 images: (a) the real production input (full frame) and (b) `bird_crop.build_crop`'s output (the same crop policy EyePose/ranking uses). Full comparison in `00_batch_summary.csv`; highlights:

| Image | True species (if known) | Full-frame prediction | Crop prediction | What changed |
|---|---|---|---|---|
| 032A2780 | Pied Kingfisher (visually confirmed) | Red-tailed Hawk, 22.3% (below 0.5 -> **Unknown**, safe) | Osprey, **84.6%** (confidently accepted, wrong) | Cropping turned a safe "don't know" into a confident wrong answer |
| DSC_1179 | Black-winged Kite (visually confirmed) | White-winged Black Tern, 54.7% (wrong) | **Lion**, 49.4% (wrong, worse - not even the right animal class) | Cropping did not help; produced a different, more severe error |
| DSC_1184 | Black-winged Kite (visually confirmed) | Common Tern, 66.7% (wrong) | **Lion**, 42.3% (wrong, and now below threshold) | Same kite burst, same "Lion" failure mode on crop |
| DSC_4538 | Unverified | Colobus Monkey, 64.8% | Colobus Monkey, **97.0%** | Agreed, crop raised confidence a lot |
| 032A4476 | Tern (burst context) | White-winged Black Tern, 99.5% | White-winged Black Tern, **99.99%** | Agreed, crop raised confidence slightly |
| DSC_4600 | Unverified | Osprey, 69.0% | Golden Eagle, 75.6% | Disagreed - crop flipped the species entirely |
| DSC_4264 | Black Kite (per EyePose Phase 1) | Red-tailed Hawk, 66.1% | Golden Eagle, 70.8% | Disagreed - wrong both ways, on different wrong species |
| DSC_3909 | Unverified (tern-burst context) | White-winged Black Tern, 66.7% | Kingfisher, 57.4% | Disagreed - crop flipped the species |

Across all 23 images, full-frame and crop predictions **disagreed on the top-1 species in 9 of 23 cases (39%)** - not a small effect. Where they agreed, cropping usually raised confidence (sometimes sharply: DSC_4538 64.8%->97.0%, 032A2018 stayed ~99.6%/99.3%). Where a real species is missing from the vocabulary (see Q4/finding 1 below), cropping did **not** reliably fix the wrong answer, and in the two clearest test cases (both a photographed Black-winged Kite) it made the answer *worse*, not better, landing on "Lion" - a result outside any bird category at all.

**Assessment: cropping is not a clean win by itself.** The evidence supports two separate conclusions:
1. When the correct species **is** in the vocabulary, a tighter crop reliably increases confidence (the model has more signal, less background) - the 032A2780 Pied Kingfisher case shows this dramatically (22%->85%), it's just landing on the *wrong* species there because "Pied Kingfisher" isn't itself an available label.
2. When the correct species is **not** in the vocabulary, no crop policy can fix that - the model is answering "which of these 55 things is this closest to," and a better crop just makes it *more confident* in whatever wrong answer it lands on, which is arguably worse for a photographer trusting the result, not better.

This means crop margin/framing is a secondary lever, not the primary one. If crop-based classification were adopted, it should NOT reuse EyePose's exact margin/aspect policy - EyePose's crop is optimized to keep a bird's *eye* comfortably inside frame for a sharpness measurement (see `bird_crop.py`'s own docstring on `margin_frac`), a different goal from "maximize how much of the 224x224 CLIP input is filled by identifying plumage/shape detail." A species-classification-specific crop would likely want a *tighter* margin than EyePose's, but this needs its own controlled test, not a shared policy - explicitly not implemented here per instruction.

**Confidence level:** Proven that disagreement is real and frequent (measured, not estimated). Hypothesis, moderately supported, on *why* cropping doesn't reliably help (small sample, and the "Lion" failure mode is not fully explained - see finding 3 below).

---

## Findings, with evidence

### Finding 1 (primary): the 55-species vocabulary is North-American-centric and is missing many of the photographer's actual species

`species/bioclip_classifier.py`'s `DEFAULT_SPECIES_LIST` has exactly 55 entries. Checked directly against the sample: **"Bee-eater" and "Kite"/"Black Kite" do not appear anywhere in the list.** Neither does "Pied Kingfisher," "Darter"/"Anhinga"/"Cormorant," or "Langur" (only the generic "Kingfisher," and only "Baboon"/"Colobus Monkey"/"Chimpanzee"/"Gorilla" among primates).

This alone explains the majority of verified errors in the sample - BioCLIP is a **closed-set, forced-choice** classifier here (softmax over exactly `len(species_list)` options, not open-vocabulary retrieval), so when the true species is absent, the model is structurally guaranteed to answer with *something else*, regardless of image quality:

- Real European Bee-eater (`032A6869`, visually confirmed - beak, plumage, and literally photographed *eating a bee*) -> predicted **Kingfisher**, 53.8% probability, **accepted** by production (above the 0.5 threshold).
- Second real Bee-eater (`032A7114`, per EyePose Phase 1's own description) -> predicted Kingfisher, 46.6% - just below threshold, correctly fell to Unknown.
- Real Black-winged Kite, photographed twice in the same session (`DSC_1179`, `DSC_1184`, both visually confirmed - grey/white plumage, black shoulder patches, red eye, one holding prey) -> predicted **White-winged Black Tern** (54.7%) and **Common Tern** (66.7%) respectively, both **accepted**.
- Real Black Kite (`DSC_4264`, per EyePose Phase 1) -> predicted **Red-tailed Hawk**, 66.1%, accepted.
- Real Pied Kingfisher (`032A2780`, visually confirmed - unmistakable black/white speckled plumage) -> full frame correctly fell to Unknown (22.3%, image is small in frame - see finding 2), but cropped tightly gives **Osprey** at 84.6%, confidently wrong.
- Real Little/Great Egret (`032A2530`, per EyePose Phase 1 - "egret") -> predicted **Snowy Egret**, a North American species not found in Israel; "Great Egret" (also in the list, and the closer real candidate) scored second at 22.7% - so even where the *family* is right, the *specific* species named is very likely wrong.
- Real Langur monkey (`DSC_5110`, visually confirmed by long tail, black face, grey body - not a Baboon's stockier build/snout) -> predicted **Baboon** at 97.7%/99.5% confidence, wrong specific species but correct broad category.

**Corroborating evidence, independent of any image at all:** `species/translations.py`'s `ENGLISH_TO_HEBREW` table (built specifically for "species a wildlife photographer is most likely to encounter" - its own docstring) lists 19 species. Checked programmatically against `DEFAULT_SPECIES_LIST`: **18 of 19 translation entries can never fire in production**, because the species they translate (Common Kingfisher, Grey Heron, White Stork, Eurasian Kestrel, Common Buzzard, House Sparrow, European Robin, Great Tit, Blue Tit, Common Blackbird, Barn Swallow, Common Starling, Rock Dove, Eurasian Collared Dove, Hooded Crow, Eurasian Magpie, Common Chaffinch, Goldfinch) simply are not in the classifier's own vocabulary. Only "Mallard" appears in both lists. This is strong, code-level evidence that the translation table was built against the *actual* regional species this photographer needs, but was never reconciled with `DEFAULT_SPECIES_LIST` - the two were developed independently and never wired together.

A second, separate wiring gap: `species/cli.py` already supports `--species-list`/`--species-list-path` for a custom vocabulary, but `desktop/services.py`'s `organize_by_species()` never accepts or forwards one - Desktop always uses the hardcoded 55-species default, with no way to override it from the UI even though the underlying mechanism to fix this already exists in the codebase.

### Finding 2: full-frame classification loses most of a small or off-center subject to background

Even where the correct species *is* in the vocabulary, image framing matters. `032A2780`'s Pied Kingfisher occupies only **0.4% of the full frame's area** (`bird_area_frac_of_frame` in the tool's own crop-analysis output) - a small, distant, in-flight bird against open sky. Full-frame classification's top guess scored only 22.3%, correctly parked as Unknown; not a wrong answer, but also not a usable one. This is architecturally expected given Q1's finding: `Resize(224) -> CenterCrop(224,224)` compresses the *entire* frame - sky, reeds, water - into the same 224x224 budget a subject-filling photo would get, so a small subject arrives at the model as a handful of blurred pixels.

### Finding 3: cropping tightly can trigger a different, sometimes worse, misclassification - not fully explained

The two Black-winged Kite images (`DSC_1179`, `DSC_1184`) are the clearest evidence: full-frame gives a wrong-but-at-least-a-bird answer (tern species); the crop - a sharp, unambiguous, textbook photo of the kite with visible red eye and prey - gives **"Lion"** both times (49.4% and 42.3%). This is not a vocabulary-gap explanation alone (Lion is also the wrong *category*, not just the wrong species) and is not explained by crop quality (the crop is excellent). This looks like a genuine CLIP zero-shot embedding-space quirk specific to this composition (pale grey/white bird, plain sky background, warm-toned prey in talons) - documented here as an observed, reproduced (2/2) failure mode, not a fully diagnosed one.

### Finding 4: some "wrong-looking" folder names are correct or defensible

Not every non-bird or unexpected-looking folder is an error. `DSC_7235.NEF` filed under "Hippopotamus" at 61.3%/73.0% (full/crop) is very likely a genuine hippopotamus photo (this test-data folder was confirmed elsewhere to include non-bird wildlife samples, not exclusively birds) - included here as a reminder that this investigation should not assume every unusual-sounding folder is a bug without checking. `DSC09855.ARW` -> "Sandhill Crane" at 99.9% confidence, both full-frame and cropped, is plausibly the closest available label for a Common Crane (a genuine winter visitor to Israel, visually similar to Sandhill Crane) rather than a random error.

---

## Where does the observed error originate? (per-image attribution, the 12 independently-checkable images)

| Image | True species | Stage the error (if any) originates at | Category |
|---|---|---|---|
| `032A2018` | Common Tern | - | Correct |
| `032A1560` | Kingfisher | - | Correct |
| `DSC03129` | Kingfisher | - | Correct (post EyePose-crop-fix) |
| `032A2530` | Egret (Little/Great, not Snowy) | Species list (missing regional species) | Partial - right family |
| `032A6869` | European Bee-eater | Species list (missing entirely) | Wrong, confidently accepted |
| `032A7114` | European Bee-eater | Species list (missing entirely) | Safely fell to Unknown |
| `DSC_4264` | Black Kite | Species list (missing entirely) | Wrong, confidently accepted |
| `DSC_1179` | Black-winged Kite | Species list + unexplained crop behavior | Wrong, confidently accepted |
| `DSC_1184` | Black-winged Kite | Species list | Wrong, confidently accepted |
| `032A2780` | Pied Kingfisher | Species list + small-in-frame (full); species list (crop) | Safely Unknown (full-frame, as production runs it) |
| `DSC_1022` | Darter (hard, occluded pose) | Species list + genuinely hard pose | Safely fell to Unknown |
| `DSC_5110` | Langur (not Baboon) | Species list (missing entirely) | Wrong, right category only |

Of these 12: **3 correct, 1 partially correct, 5 confidently wrong, 3 safely deferred to Unknown.** Every single non-correct case traces to the same root cause: the true species is absent from `DEFAULT_SPECIES_LIST`. Not one of the 12 verified errors was caused by a detection failure, a wrong crop, or a BioCLIP encoder mistake on an in-vocabulary species.

---

## Final report

**1. Where do most classification errors originate?**
The species vocabulary (`DEFAULT_SPECIES_LIST`), not the model, not detection, not cropping. Every verified error in this sample (8 of 12 checkable images) involves a true species that simply is not one of BioCLIP's 55 available answers. This is corroborated independently by the translations.py finding (18 of 19 real regional species names can never even be produced, because they were never added to the classifier's vocabulary in the first place).

**2. How many errors are caused before BioCLIP is even called?**
None, in the "bad input" sense the eye-pipeline investigation found (there, a wrong crop selection was the root cause). Here, BioCLIP is always given the real, intended production input (`load_source_image`'s full frame) - there is no equivalent "wrong image sent to the model" bug. However, one upstream, structural cause counts as "before the model is called" in a different sense: the vocabulary passed to `create_model_and_transforms`'s text encoder at construction time. That happens once, at startup, before any image is classified - it is a configuration defect, not a per-image bug, but it is upstream of every single classification.

**3. How many appear to be genuine BioCLIP encoder mistakes?**
At most 1 of 12 verified cases (`DSC_1179`/`DSC_1184`'s crop-only "Lion" result) looks like a genuine model behavior worth investigating on its own merits, independent of vocabulary - and even that might resolve itself once the correct species is added as an available answer (a Kite embedding would then have to actually compete against a plausible right answer, not just against 55 wrong ones). Zero cases in this sample show BioCLIP failing on a species that was actually present in its vocabulary and reasonably well photographed (the 3 correct cases and the false-negative-turned-Unknown cases all behave sensibly given what the model was allowed to say).

**4. Would another crop policy likely improve accuracy?**
Partially, and only for images already limited by frame size (Finding 2) - not for the vocabulary-gap cases, which dominate this sample. Measured disagreement between full-frame and crop classification was 39% (9/23), and cropping was not reliably an improvement even when it changed the answer (see Finding 3's "Lion" cases, and 032A2780's confident-wrong-Osprey vs. safe-Unknown result). A species-specific crop policy is worth a dedicated, controlled follow-up, but it is not the highest-leverage fix available (see Q5) and should not simply reuse EyePose's existing margin/aspect settings, which were tuned for a different goal.

**5. Three highest-impact improvements, ordered by expected benefit:**

1. **Expand/replace `DEFAULT_SPECIES_LIST` with a region-appropriate vocabulary**, reconciled against the species already present in `translations.py`'s Hebrew table (which already encodes exactly the right species for this photographer, just was never connected to the classifier). This is a data change, not a model or architecture change, and the evidence above suggests it alone would fix the majority of observed errors. The `--species-list-path` mechanism to do this already exists in `species/cli.py`.
2. **Wire `species_list_path` through to the Desktop UI** (`desktop/services.py`'s `organize_by_species`, currently hardcoded to the default list with no override) - without this, fix #1 only helps CLI users, not the actual Desktop workflow this investigation was triggered from.
3. **A dedicated, controlled test of a tighter, species-classification-specific crop**, run only after #1 - so any crop-vs-full-frame comparison is measured against a vocabulary that can actually contain the right answer, rather than being confounded by Finding 1 the way this investigation's own comparison necessarily was.

---

## Remaining uncertainty

- 11 of the 23 sampled images have no independently confirmed true species; their folder placement and confidence data are reported as-is, not validated. A larger, independently-labeled benchmark (ideally with the photographer confirming species for a held-out sample) would firm up the error-rate estimates above into precise numbers rather than a 12-image directional read.
- Finding 3 (the "Lion" crop failure) is observed and reproduced but not diagnosed to a specific mechanism - flagged as a candidate for its own narrow follow-up if pursued.
- This investigation used `bird_crop.BirdDetector`'s existing detector/crop policy as the "cropped" comparison arm throughout; a species-classification-purpose-built crop (tighter margin, different aspect handling) was explicitly not tried, per the instruction not to implement anything yet.

## Appendix: artifacts produced

- `tools/debug_species_pipeline.py` - the investigation tool itself (new this investigation, not part of the product).
- Per-image debug bundles (`01_original.jpg` through `08_final_decision.md`, plus `04b`/`05b` crop-comparison artifacts) for all 23 sampled images - available in the debug tool's own output directory, not committed (real photos from the user's archive).
- `00_batch_summary.csv` - the full 23-image comparison table this report's tables are drawn from.
