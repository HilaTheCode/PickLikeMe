# BioCLIP / BioCLIP 2 Multi-Backend Architecture Review

**Date:** 2026-08-03. **Scope:** Formal review of the multi-backend species-classification work completed earlier this session, before further development. Every claim below is either a direct code citation (file:line, quoted or paraphrased from the actual current file) or an official-documentation citation (URL + quoted text), or is explicitly marked as unverified/unknown. Nothing here is asserted from memory or general expectation about how BioCLIP "probably" works - see §1-2 for how each fact was obtained.

---

## 1. Verify the actual models

Two independent sources were checked for each model and cross-referenced against each other: (a) the official Hugging Face model card, and (b) the actual `open_clip_config.json` inside the model weights downloaded onto this machine (i.e., the literal bytes `open_clip.create_model_and_transforms()` loads) - not a summary of either, the file itself.

### BioCLIP (v1) - registered as `"bioclip"`

| Field | Value | Source |
|---|---|---|
| HuggingFace model ID | `imageomics/bioclip` | `species/bioclip_classifier.py:78`, `BIOCLIP_V1_MODEL_ID` |
| `open_clip` loading string | `hf-hub:imageomics/bioclip` | Same |
| Source repository | `github.com/Imageomics/bioclip` ("This is the repository for the BioCLIP model and the TreeOfLife-10M dataset [CVPR'24 Oral, Best Student Paper]") | GitHub search, confirmed title |
| Resolved commit (this machine) | `ce901ab3c6a913f9e9ef94ce6d27761069f4f01c` (`main` branch HEAD) | `~/.cache/huggingface/hub/models--imageomics--bioclip/refs/main`, read directly |
| Model architecture | CLIP (contrastive image-text), fine-tuned from OpenAI CLIP ViT-B/16 | HF model card: "Fine-tuned from model: OpenAI CLIP, ViT-B/16" |
| Vision encoder | ViT-B/16 - **confirmed in the actual downloaded config**: `vision_cfg: {layers: 12, width: 768, patch_size: 16}` (patch 16, width 768, 12 layers = the standard ViT-B/16 shape) | `open_clip_config.json` read directly from `~/.cache/huggingface/hub/models--imageomics--bioclip/snapshots/ce901ab3.../open_clip_config.json` |
| Text encoder | Transformer, `text_cfg: {width: 512, heads: 8, layers: 12, context_length: 77}` | Same file |
| Embedding dimension | **512** (`model_cfg.embed_dim: 512`) - also independently confirmed by actually loading the model and calling `encode_text`, which returned a tensor of shape `[1, 512]` | Config file, and a real local run this session |
| Input image size | 224x224 (`vision_cfg.image_size: 224`) | Same config file |
| Training dataset | TreeOfLife-10M | HF model card: "TreeOfLife-10M" |
| Number of training images | ~10 million | HF model card |
| Number of taxa | "454K different taxa" (also stated as "over 450K taxa") | HF model card |
| License | MIT | HF model card: `"license": "mit"` |
| Paper | Stevens et al., "BioCLIP: A Vision Foundation Model for the Tree of Life," CVPR 2024 | HF model card citation block |

### BioCLIP 2 - registered as `"bioclip2"` (existing default, unchanged)

| Field | Value | Source |
|---|---|---|
| HuggingFace model ID | `imageomics/bioclip-2` | `species/bioclip_classifier.py:73`, `DEFAULT_MODEL_ID` |
| `open_clip` loading string | `hf-hub:imageomics/bioclip-2` | Same |
| Source repository | `github.com/Imageomics/bioclip-2` ("BioCLIP 2 is a biological foundation model trained on TreeOfLife-200M... [NeurIPS'25 Spotlight]") | GitHub search, confirmed title |
| Resolved commit (this machine) | `2957b322090f9cb17ae72c71981c7218a28d81e0` (`main` branch HEAD) | `~/.cache/huggingface/hub/models--imageomics--bioclip-2/refs/main`, read directly |
| Model architecture | CLIP, fine-tuned from CLIP pre-trained on LAION-2B, ViT-L/14 | HF model card: "The base checkpoint was CLIP pre-trained on LAION-2B, ViT-L/14" |
| Vision encoder | ViT-L/14 - **confirmed in the actual downloaded config**: `vision_cfg: {layers: 24, width: 1024, patch_size: 14}` (patch 14, width 1024, 24 layers = the standard ViT-L/14 shape) | `open_clip_config.json` read directly from `~/.cache/huggingface/hub/models--imageomics--bioclip-2/snapshots/2957b322.../open_clip_config.json` |
| Text encoder | "masked self-attention Transformer," `text_cfg: {width: 768, heads: 12, layers: 12, context_length: 77}` | HF model card + config file |
| Embedding dimension | **768** (`model_cfg.embed_dim: 768`) | Config file |
| Input image size | 224x224 (`vision_cfg.image_size: 224`) - **identical to v1**, confirmed both from the config file and by directly inspecting the live `preprocess` transform object for both models this session (`Resize(224, bicubic) -> CenterCrop(224,224)`, byte-identical `Compose` repr for both) | Config file + direct local inspection (this session, both models) |
| Training dataset | TreeOfLife-200M | HF model card |
| Number of training images | "nearly 214M images" | HF model card |
| Number of taxa | "952K taxa" | HF model card |
| License | MIT | HF model card |
| Paper | "BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning," NeurIPS 2025 (arXiv:2505.23883) | HF model card |

**Confidence level:** Proven for every field above - each has either a direct config-file citation, a direct model-card quote, or both, plus (for image size, embedding dim, and preprocessing) independent confirmation from actually loading both models locally this session.

---

## 2. Prove this is really BioCLIP v1

Checked, explicitly, not assumed:

1. **It is the official `imageomics` organization's repository, not a fork or mirror.** `github.com/Imageomics/bioclip` is returned as the canonical repo by GitHub's own search, described by its own README title as "the repository for the BioCLIP model and the TreeOfLife-10M dataset [CVPR'24 Oral, Best Student Paper]" - the same paper cited on the model card.
2. **It is not an experimental or intermediate checkpoint.** The resolved commit (`ce901ab3...`) is the `main` branch's current HEAD - the same reference `hf-hub:imageomics/bioclip` (no pinned revision) resolves to by default. There is no separate "stable" vs. "experimental" branch on this repository; `main` is what the model card itself documents and what every official usage example loads.
3. **It is not merely another BioCLIP-2 variant under a confusing name.** The architecture recovered directly from the downloaded weights (ViT-B/16, embed_dim 512, TreeOfLife-10M-shaped text vocabulary) is categorically different from BioCLIP 2's (ViT-L/14, embed_dim 768) - these are not two names for the same checkpoint; the config files prove two structurally different models. `imageomics/bioclip` and `imageomics/bioclip-2` are also two separate, distinctly-named GitHub repositories with two separate papers (CVPR'24 vs. NeurIPS'25), not two tags of one repository.
4. **A real forward pass was run and produced dimensionally-consistent output.** `encode_text` on a real prompt returned a `[1, 512]` tensor - exactly matching the config's stated `embed_dim: 512` for v1, and different from v2's stated 768. This is an executable, falsifiable check, not documentation-reading alone: if the registered "bioclip" id actually pointed at BioCLIP 2 by mistake, this would have returned `[1, 768]` instead.

**One thing NOT verified, stated explicitly:** whether the exact weight *values* inside `open_clip_pytorch_model.bin` are bit-identical to whatever the original CVPR 2024 authors trained (i.e., that Hugging Face's copy hasn't silently drifted from the paper's own checkpoint). This would require an independent checksum against a source outside Hugging Face Hub, which was not attempted - reported as unknown rather than assumed true.

---

## 3. Architecture review

**Is `SpeciesClassifier` fully open for future backends?** Yes, for anything that can express its answer as "one image in, one species-name-plus-confidence out." The Protocol (`species/classifier.py:46-60`) requires only `classifier_id: str` and `classify(image: Image.Image) -> SpeciesPrediction`. Every current caller (`species/arrange.py`'s `arrange_by_species`, `species/cache.py`'s `get_or_classify`, `desktop/services.py`'s `organize_by_species`) depends only on this Protocol - confirmed by grep (§4): no caller anywhere imports `BioClipSpeciesClassifier` directly except `species/classifier.py`'s own registry.

**Can a completely different model (BirdNet, BirderEU, Merlin) be added without modifying Desktop code?** Partially yes, with real caveats found by this review:

- **Yes:** registering a new class in `build_classifier()`'s dict (`species/classifier.py:120-123`) and adding one `ClassifierInfo` entry to `AVAILABLE_CLASSIFIERS` (`species/classifier.py:79-90`) is sufficient for it to appear in both Desktop dialogs automatically (`workflow_dialogs.py`'s `SpeciesLanguageDialog`/`PreferencesDialog` both iterate `available_classifiers()` - confirmed by reading the code added this session, no per-backend UI code exists).
- **Not fully, for two concrete reasons found during this review:**
  1. **The kwargs contract is implicit, not typed.** `build_classifier(name, **kwargs)` forwards whatever kwargs the caller passes straight to the constructor. Every current caller (`desktop/services.py:206`, `species/cli.py:89-94`) hardcodes the same three kwarg names: `min_confidence`, `device`, `species_list_path`. These are `BioClipSpeciesClassifier`-shaped concepts (a float confidence threshold; a torch device string; a text-file vocabulary). A genuinely different model - e.g., BirdNet, which is audio-first and location/date-aware, not confidence-threshold-and-device shaped in the same way - would either have to accept-and-ignore these three specific kwargs, or the caller code in `services.py`/`cli.py` would need editing to pass different kwargs for different backends. **This is real, current coupling, not hypothetical.**
  2. **The species-vocabulary mechanism assumes an open-vocabulary, zero-shot, text-prompted model.** `species_list_path`/`DEFAULT_SPECIES_LIST` work by turning species names into CLIP text prompts (`"a photo of a {}"`) and comparing embeddings - this is BioCLIP-family-specific by construction. A closed-set classifier like BirdNet has its own fixed, pretrained output vocabulary (thousands of species baked into the model's final layer) - it does not take an external species list the same way at all. `species_list_path` would simply be meaningless to it. Nothing in the current architecture prevents a new backend from ignoring this kwarg, but the *concept* "the species list is external, editable, and shared across backends" (stated in `bioclip_classifier.py`'s own module docstring as a design goal) does not actually hold for a closed-set model - this needs to be an explicit, documented exception per backend, not assumed universal.

**What is still coupled?** Summarized: the `SpeciesClassifier` Protocol itself is clean; the *calling convention* around it (which kwargs a caller passes, and the assumption that "species list" is a universal, backend-agnostic concept) is not yet abstracted, because there has only ever been one family of backend (BioCLIP-shaped) to abstract from. See §10 for a concrete recommendation.

---

## 4. Registry review

Searched the entire `src/` tree (not just the species package) for two patterns: (a) branching on a classifier/backend name string, (b) importing the concrete `BioClipSpeciesClassifier` class anywhere outside its own registry.

**Pattern (a) - `classifier == ...` / `backend == ...` / `name == "bioclip..."` branching logic:** Zero occurrences anywhere in `src/`.

**Pattern (b) - direct imports of `BioClipSpeciesClassifier`:** Exactly one call site, its own registry:

```
src/picklikeme/species/classifier.py:118:    from .bioclip_classifier import BIOCLIP_V1_MODEL_ID, BioClipSpeciesClassifier
```

Every other match of the string `BioClipSpeciesClassifier` in `src/` is either the class's own definition (`bioclip_classifier.py:115`) or a docstring/comment reference (`classifier.py:103,111`, `eyes/superanimal_bird.py:358` - a comment drawing an analogy to a different subsystem, not an import).

**The literal string `"bioclip2"` (the default backend id) does appear in seven places**, all as a *default parameter value or fallback return*, never as a conditional:

```
desktop/main_window.py:1340, 1396          - QSettings default value
desktop/dialogs/workflow_dialogs.py:220    - PreferencesDialog default parameter
desktop/dialogs/workflow_dialogs.py:203,293 - unreachable fallback return (one radio is always checked)
desktop/services.py:193                    - organize_by_species's own default parameter
species/cli.py:38                          - argparse --classifier default
```

**Risk, not a violation of the "no hardcoded conditionals" requirement, but worth flagging:** this default is duplicated as a literal string seven times rather than one shared named constant (e.g., a `DEFAULT_CLASSIFIER_ID` exported from `species/classifier.py`). If the project default ever changes, all seven sites need editing in lockstep; nothing enforces that today. Recommended, not implemented (§10).

**Conclusion: no hardcoded backend branching and no concrete-class coupling outside the registry exist anywhere in the repository.** The one remaining gap is the *default-value* duplication above, which is a maintainability risk, not an architectural violation of "depend only on the abstraction."

---

## 5. Cache review

Four caches exist in this project. Each is examined for identity, invalidation, backend-change survival, and coexistence safety - proven from the actual schema/code, not inferred.

### Crop cache (`bird_crop.py`)

- **Identifies an entry:** the source image's path, sharded by a content hash (`crop_cache_path`).
- **Invalidates:** a *whole-cache* version check - `build_cache` compares the stored `CropParams` (including `version`) against the requested one; any mismatch refuses the entire cache (`CropCacheVersionMismatch`, fixed earlier this session) unless `force=True`.
- **Survives backend changes:** N/A in the multi-classifier sense - this cache has no concept of "backend," only one active detector configuration per `crop_cache_dir` at a time, by design.
- **Coexistence:** two different `CropParams` configurations cannot coexist in the same `crop_cache_dir` - this is intentional (see this session's earlier `CropCacheVersionMismatch` work) and orthogonal to species classification.

### Eye cache (`eyes/cache.py`)

- **Identifies an entry:** the source image's path (same sharded scheme as the crop cache).
- **Invalidates on read:** only `EYE_CACHE_VERSION` (a payload-shape version) is checked in `read_eye_detection` - **`detector_id` is stored in the payload but is never compared against a caller-supplied value on read.** `read_eye_detection(cache_dir, source_path)` takes no `detector_id` parameter at all.
- **Survives backend changes:** **No, and unlike `SpeciesCache`, this is not even detected.** If a folder was processed with SuperAnimal-Bird, then re-processed with EyePose-v0, `save_eye_detection` overwrites the same file (same path, same name); reading it back afterward returns whichever detector wrote last, silently, with no cache-miss signal for the mismatch the way `SpeciesCache.get()` provides.
- **Coexistence:** No - one file per image, last writer wins, and (unlike SpeciesCache) nothing downstream is warned about it.
- **Practical severity, verified rather than assumed:** this cache is **write-for-display, not read-to-skip-recomputation**. `ranking/classic.py`'s `rank_folder()` calls `self._detector.detect(candidate.subject_crop)` unconditionally for every image, every run (confirmed in the ranking loop added/reviewed this session) - it never reads `eyes.cache` first to decide whether to skip inference. `eyes.cache` exists purely so the Gallery/Loupe debugging overlay has something to show. **Consequence:** switching eye-detector backends never produces a wrong *ranking* (eye detection is always freshly computed), but the debugging overlay can show a stale, wrong-detector eye position for any image not yet re-ranked under the new backend, with nothing in the UI indicating this. Lower severity than the SpeciesCache finding below, but a real, previously-unflagged gap.

### Species cache (`species/cache.py`) - **the significant finding of this review**

The schema, quoted directly:

```sql
CREATE TABLE IF NOT EXISTS species_cache (
    image_hash    TEXT PRIMARY KEY,
    species       TEXT NOT NULL,
    confidence    REAL,
    classifier_id TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
)
```

- **Identifies an entry:** `image_hash` (content identity) **alone** - `classifier_id` is a stored column, not part of the primary key.
- **Invalidates on read:** `get()` does correctly check `row["classifier_id"] != classifier_id` and returns a cache miss on mismatch - reads are safe; a caller is never served a different backend's answer without knowing it.
- **Write behavior - the bug:** `store()` executes `INSERT OR REPLACE INTO species_cache(image_hash, species, confidence, classifier_id) VALUES (...)`. Because `image_hash` is the sole primary key, writing a new classifier's result for an image **overwrites and destroys** whatever the other classifier previously stored for that same image. There is no `(image_hash, classifier_id)` composite key.
- **Can two classifier backends safely coexist? No - proven, not inferred.** Classifying a folder with BioCLIP 2, then the same folder with BioCLIP v1, causes every image's BioCLIP-2 row to be replaced by BioCLIP-v1's answer. Switching back to BioCLIP 2 afterward is now *also* a cache miss (the stored `classifier_id` no longer matches), forcing a full recompute that again overwrites the row.
- **Can results from BioCLIP and BioCLIP 2 overwrite each other? Yes - proven directly from the `INSERT OR REPLACE` statement and the single-column primary key.**

**This directly contradicts the "benchmark support... avoid preventing it" goal from the original multi-backend request.** The read path is safe (never shows wrong data), but the storage layer cannot hold two backends' answers for the same image at once, which is exactly what a same-folder, both-backends comparison needs. This was not caught during the original implementation and is the most important finding in this review. See §10 for the recommended fix (not implemented here, per instruction to review first).

### Analytics store (`analytics/store.py`, built earlier this session)

- **Identifies an entry:** a generated UUID `run_id`, unique per call to `record_run` - never derived from image identity or backend name.
- **Invalidates:** never, by design - it is a history log, not a cache; every run is kept.
- **Survives backend changes:** trivially yes - `runs.strategy_id` records which algorithm produced each run, and every run gets its own row regardless of how many other runs exist.
- **Coexistence:** Safe by construction - there is no shared key two different backends' runs could collide on. (This store is currently wired to the *ranking* pipeline, not species classification - noted for completeness since the question was asked generically across "every cache.")

---

## 6. Species vocabulary

- **Where is the vocabulary loaded?** `BioClipSpeciesClassifier.__init__` (`bioclip_classifier.py:132-136` in the constructor): `species_list = _read_species_list(species_list_path)` if a path was given, else `DEFAULT_SPECIES_LIST` (the built-in 55-name tuple).
- **When is it loaded?** Once, at classifier construction - not per image. The resulting species names are immediately turned into CLIP text prompts and encoded **once**, also at construction: `tokens = tokenizer(prompts).to(device); text_features = self._model.encode_text(tokens)` (`bioclip_classifier.py`, inside `__init__`). `classify()` never re-touches the species list or re-runs the text encoder.
- **How many species can realistically be supported?** No hard-coded ceiling exists anywhere in the code - `_read_species_list` just splits a text file into lines, and `tokenizer(prompts)` accepts an arbitrary-length list. Practically bounded only by GPU/CPU memory and construction-time latency (see §7).
- **Is loading 1000 species expected to work?** Yes, based on the code structure - there is no algorithmic barrier. **Not tested in this review** (explicitly not benchmarked, per instruction); this is a structural, not empirical, conclusion.
- **Is there any algorithmic limitation?** One found: **no batching/chunking**. All N prompts are tokenized and encoded in a single forward pass (`tokenizer(prompts)` then one `encode_text` call, no loop, no batch-size cap). For very large N this is a single large batch, not several small ones - a memory-usage pattern that has not been tested at scale (see §7's explicit uncertainty on this point).
- **Is runtime linear?**
  - **Construction (one-time, per classifier instance):** roughly linear in N - one forward pass through the text encoder over a batch of N prompts.
  - **Per-image classification (`classify()`, the repeated cost):** **not** meaningfully linear in N for any realistic N. The only N-dependent step is `image_features @ text_features.T` - a `(1, D) x (D, N)` matrix multiply, `O(N*D)` floating-point operations. For N=2000, D=768, that is ~1.5M multiply-adds - negligible next to the vision encoder's own forward pass (a fixed cost per image, independent of N, and the dominant cost of `classify()` by orders of magnitude). **Complexity summary: construction is O(N); classification is O(1) in practice with respect to species count, dominated by a constant-cost vision forward pass, plus a negligible O(N·D) similarity step.**

---

## 7. Performance analysis (derived from the implementation, not benchmarked)

Per the instruction, no benchmark was run. The estimates below are derived purely from the code structure in §6 and the measured architecture in §1, with explicit uncertainty noted. **None of the absolute numbers below have been measured on this machine or any other - they are order-of-magnitude reasoning from known model sizes, not results.**

| Species count | One-time construction cost | Per-image classification cost | Additional GPU memory (text embeddings) | Additional RAM |
|---|---|---|---|---|
| 55 (current default) | Already measured indirectly: negligible, sub-second in every run this session | Unaffected - dominated by vision encoder | ~55 x 768 floats (v2) = ~170KB | Negligible |
| 200 | Expected to still be sub-second (one batched forward pass, ~4x more prompts than today) | Unaffected | ~200 x 768 floats ≈ 615KB | Negligible |
| 500 | Expected low seconds at most - **not measured** | Unaffected | ~500 x 768 floats ≈ 1.5MB | Negligible |
| 1000 | Expected low-to-several seconds - **not measured**; this is the point where "one unchunked batch" starts to be worth verifying rather than assuming | Unaffected | ~1000 x 768 floats ≈ 3MB | Negligible |
| 2000 | **Unknown with confidence** - plausibly still fine (text-embedding memory stays in the single-digit MB range even here), but this is exactly the range where an untested, unchunked batched forward pass deserves an actual measurement before being relied on | Unaffected | ~2000 x 768 floats ≈ 6MB | Negligible |

**Why per-image cost is listed as "unaffected" at every row:** this is the one number in this table backed by more than order-of-magnitude reasoning - §6 already showed the per-image similarity computation is `O(N·D)` against a vision-encoder forward pass that is `O(1)` in N and orders of magnitude larger. Species count essentially does not move the per-image number.

**Why the text-embedding memory column is small even at 2000:** a `(2000, 768)` float32 tensor is `2000 * 768 * 4 bytes ≈ 6.1MB` - trivial next to the vision model itself (BioCLIP 2's weights alone are 1.6GB on disk). This is a real calculation from the measured `embed_dim`, not a guess.

**What is genuinely unknown, stated explicitly rather than estimated:** the one-time construction latency at 1000-2000 species, because it depends on the text encoder's actual per-batch throughput on the specific GPU/CPU in use, which was not measured. The architecture gives high confidence this is *possible*; it gives no confidence about *how many seconds* without running it.

**A separate, concrete finding directly relevant to this section:** `desktop/services.py:196`'s `organize_by_species` defaults to `device="cpu"`, and `desktop/main_window.py`'s `_organize_by_species` (the only Desktop caller) never overrides it - confirmed by reading the current call site, which passes only `backend`, `language`, `on_progress`. **Every performance number above is dramatically worse on CPU than the GPU this review's other validation used.** This is a pre-existing gap (not introduced by this session's multi-backend work) but is squarely relevant to any real performance conversation about scaling species count, and is flagged here as a concrete near-term risk independent of vocabulary size.

---

## 8. BioCLIP 2 recommendation - reconsidered

The earlier recommendation ("BioCLIP 2 as default") was based on one image. That is correctly identified as insufficient evidence for a general recommendation, and it is withdrawn as stated.

**Revised recommendation: keep BioCLIP 2 as the current default until a statistically meaningful benchmark (§9) has been run - not because there is evidence it is wrong, but because there is not yet evidence it is right at any scale beyond n=1.**

This is not a purely defensive non-answer; it is supported by what *is* known without further testing:

- BioCLIP 2 is the architecturally larger, more recently trained model (20x more training images, ~2x more taxa, per §1's cited official figures) - a reasonable prior in its favor, not proof of better real-world accuracy on this photographer's specific species mix.
- BioCLIP 2 is also the **official default** - the BioCLIP 2 GitHub repository states "BioCLIP 2 is set as the default model for pybioclip" as of July 2025 (confirmed via the official repository this session) - so keeping it as this project's default aligns with the model authors' own current guidance, independent of this project's own testing.
- Against switching defaults on n=1 evidence: the one data point available (a real Kingfisher photo) showed a large gap (93.8% vs. an Unknown-triggering 37.2%), but a single image cannot establish whether that gap is typical, an outlier, or specific to that species/pose/lighting - exactly the concern raised.

**No default change is recommended at this time, in either direction.** Continuing to default to BioCLIP 2 is the status quo, not a new claim requiring new evidence; actively recommending it *because* of last week's single-image test would be the overreach being corrected here.

---

## 9. Benchmark plan (proposed, not implemented)

**Sample size:** a minimum of 200-300 images for species-level Top-1/Top-3/Top-5 numbers to have usable confidence intervals per species bucket; fewer than ~30 images per species subgroup makes per-species accuracy numbers too noisy to act on. Given this project's real archive is a few thousand images across a few dozen species (per this session's earlier species investigation), a realistic first benchmark is the **entire already-reviewed Keep folder** (ground truth already exists implicitly - see below) rather than a hand-picked small sample.

**Species coverage:** should include every species that already appears meaningfully often in the photographer's real archive (from this session's earlier investigation: kingfisher, bee-eater, egret, tern, kite, and others), not just the 55-name `DEFAULT_SPECIES_LIST` - the earlier species investigation (`docs/Species_Classification_Investigation.md`) already found several real species entirely absent from that list, which is itself a variable the benchmark should hold constant or explicitly test (i.e., run the benchmark once against the current default list, and once against an expanded list, to separate "is BioCLIP 2 vs. v1 better" from "is the vocabulary the real bottleneck," which the earlier investigation already suggested it is).

**Ground truth:** the most reliable source is the photographer's own species identification, not this reviewer's visual guesses (the earlier species investigation explicitly flagged its own visual IDs as non-authoritative for exactly this reason). Two practical options, in order of reliability: (a) species already encoded in existing folder names/keywords/XMP metadata if the photographer has already organized by species manually for some portion of the archive; (b) a short manual labeling pass by the photographer over the benchmark sample before running either backend, so the labels are never influenced by seeing a model's guess first (avoiding confirmation bias).

**Metrics to collect, per the request:**
- **Top-1 / Top-3 / Top-5 accuracy** - requires `classify()`'s current single-answer return to be extended to expose the full ranked list (already demonstrated possible without touching production code, via the same instance-introspection pattern `tools/debug_species_pipeline.py` used in the earlier species investigation).
- **Unknown rate** - the fraction of images where confidence never clears `min_confidence`, tracked per backend, since a lower Unknown rate is not automatically better (a confidently wrong answer is worse than an honest Unknown - this exact distinction was the finding of the earlier species investigation).
- **Runtime** - both the one-time construction cost and the steady-state per-image cost, measured separately (§7 already shows these have very different scaling behavior and must not be conflated into one number).
- **GPU memory** - peak allocated, via `torch.cuda.max_memory_allocated()`, reset between backends so one model's memory footprint is never attributed to the other.
- **Agreement rate** - how often the two backends produce the same Top-1 answer, independent of whether either is correct - a cheap, ground-truth-free signal that can be computed on the *entire* archive, not just the labeled benchmark subset.
- **Precision / Recall / confusion matrix** - per-species, requires the ground-truth labels above; the confusion matrix is likely the single most actionable artifact, since it would show *which* species get confused for *which* others (directly extending the earlier species investigation's per-image error attribution into an aggregate view).

**What this benchmark should NOT be:** a single aggregate accuracy number. Given this session's own finding that the *vocabulary* (not necessarily the model) is the dominant error source for this photographer's real species mix, an aggregate "BioCLIP 2 got 80% vs. BioCLIP got 75%" number would be close to meaningless without also reporting how many of those errors were vocabulary-gap errors common to *both* backends versus genuine model-quality differences.

---

## 10. Future architecture - recommendations before continuing

Imagining BirdNet, BirderEU, Merlin, and future foundation models alongside the two BioCLIP versions one year from now, three concrete, scoped changes are recommended - **not implemented in this review**, per instruction:

1. **Fix `SpeciesCache`'s primary key to `(image_hash, classifier_id)`.** This is the one finding in this review that is already actively wrong today, with only two backends - it will not get better by waiting, and every additional backend makes the "last write wins, destroying every other backend's cached answer for that image" problem worse, not just more theoretical. This is the highest-priority recommendation in this document.
2. **Formalize the per-backend configuration contract**, the same way ranking strategies already formalize theirs via `ParamSpec`/a params dataclass (`ranking/classic.py`'s `ClassicVisionParams`, etc.) rather than untyped `**kwargs`. A closed-set model like BirdNet needs a genuinely different parameter shape (no `species_list_path` concept, possibly location/date-aware) than an open-vocabulary CLIP-family model - today's `build_classifier(name, **kwargs)` silently assumes every backend wants the same three kwargs, which is only true because only BioCLIP-family backends exist so far.
3. **Decide, explicitly, what "species list" means for a closed-set backend before one is added**, rather than discovering the mismatch mid-implementation. Options include: (a) closed-set backends simply ignore `species_list_path` (documented, not silent), or (b) `species_list_path` becomes an optional *filter* applied after a closed-set model's own prediction (map its native output down to the requested subset) rather than an input to the model itself. This is a design decision with real consequences for accuracy reporting and should be made deliberately, not organically.

A smaller, related note: centralize the seven-times-duplicated `"bioclip2"` default-value literal (§4) into one named constant, e.g. `species/classifier.py`'s `DEFAULT_CLASSIFIER_ID`, so a future default change is a one-line edit instead of a grep-and-verify exercise.

---

## 11. Deliverables summary

1. **Architecture review:** §3, §10 above.
2. **Evidence for every answer:** inline throughout, distinguishing direct code citation, direct official-documentation citation, and this-session local verification (running real code, not reading about it) in every section.
3. **Code references:** file:line citations given throughout, most concretely in §4 (registry grep results) and §5 (cache schemas, quoted verbatim).
4. **Official documentation references:** §1's two tables (Hugging Face model cards, GitHub repositories, both accessed and quoted this session); §8's BioCLIP 2 GitHub repository citation for the "official default" claim.
5. **Risks:**
   - **High:** `SpeciesCache`'s single-column primary key silently destroys cross-backend cached results (§5) - directly undermines the benchmark-support goal from the original request.
   - **Medium:** `eyes.cache` does not validate `detector_id` on read, though its write-for-display (not read-to-skip) usage limits the practical impact to the debugging overlay, never ranking correctness (§5).
   - **Medium:** Desktop's species classification silently runs on CPU by default regardless of GPU availability (§7) - a pre-existing gap, newly relevant now that performance/scale is under discussion.
   - **Low:** the `"bioclip2"` default-value literal is duplicated seven times with no single source of truth (§4).
   - **Informational:** a newer model, "BioCLIP 2.5 Huge" (`imageomics/bioclip-2.5-vith14`, ViT-H/14, released February 2026), already exists upstream and is not covered by either registered backend - not a defect, but relevant context for any "which model should be default" discussion going forward, and for §10's future-architecture planning.
6. **Recommended next steps, in priority order:**
   1. Fix the `SpeciesCache` primary key (§10, item 1) - small, contained, and the one finding that is already actively harmful today.
   2. Decide whether to fix Desktop's CPU-default for species classification now or defer it (§7) - independent of the multi-backend work, but newly visible because of it.
   3. Design and run the benchmark in §9 before making any default-backend decision, including evaluating whether to add BioCLIP 2.5 as a third comparison point given its now-confirmed existence.
   4. Only after the above: consider the two architecture changes in §10 (items 2-3) if/when a genuinely different (non-BioCLIP-family) backend is actually being integrated - not speculatively now, since their right shape depends on which real backend arrives first.
