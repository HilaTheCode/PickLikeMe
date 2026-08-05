# EyePose Investigation – Phase 1

**Date:** 2026-08-01/02 (investigation), 2026-08-02 (implementation - see "Phase 2: Implementation" at the end of this document)
**Scope:** Forensic investigation only. No production code was changed, no thresholds were tuned, no model was replaced. Tool used: `tools/debug_eye_pipeline.py` (built earlier tonight), plus several one-off scratch scripts for controlled experiments (listed inline, not committed - pure investigation, not product code).

**Sample:** 10 real photos from the user's own archive - `032A2530.CR3` (the egret used in the original bug report) plus 9 images drawn from `Test data for PickLikeMe\_Selected` and `Test data for PickLikeMe\_Rejected` (a mix of CR3/ARW/NEF, covering egret, kingfisher, tern, kite, and bee-eater, in poses ranging from calm profile stands to mid-flight, wing-raised, and head-down-with-prey). This is a **small, non-random sample** - see "Remaining uncertainty" throughout. It was not selected to prove any particular conclusion; it was the full set of images readily available under the user's own `Test data for PickLikeMe` folder plus a handful of additional real photos chosen for pose diversity.

---

## Executive summary

The single most important finding of this investigation: **the eye overlay a photographer sees in Gallery/Loupe is produced by a coordinate-projection bug in `ranking/classic.py` and `review/thumbnails.py`, not by a defect in EyePose's model, ONNX export, or keypoint decoding.** The bug uses the *tight* bird-detection box to map a crop-space eye coordinate back onto the full photo, when the crop was actually built from a *larger, margin-expanded* box. This is proven (not hypothesized) with an exact arithmetic match against measured data (see Q1). Critically, this bug **only affects the on-screen overlay** - the actual eye-sharpness score used for ranking/keep-reject decisions is computed entirely in crop-space and never touches the buggy projection (see Q7). So: the ranking has likely been fine; the diagnostic picture the user has been looking at has not.

Separately, one clear case of "an eye reported where none should be" was found and root-caused: it is a **subject-selection bug in `bird_crop.select_best_detection`**, not an EyePose bug - see Q1's DSC03129 case study.

Across 9 of 10 images where the crop actually contained the bird's head, EyePose's own crop-space keypoint decode was accurate, including on several genuinely hard poses (wing raised over the head, head pointing straight down with prey in talons, distant birds in flight). See Q4-Q6.

---

## Q1. Does the standalone debugger produce exactly the same eye coordinates as the Desktop application?

**Question:** If not, where do the two code paths diverge? Show the first different function.

**Evidence:**

Ran the real production pipeline end-to-end on `032A2530.CR3` (via `preprocess.build_cache` → `ranking.classic.ClassicVisionEyePoseStrategy._load_candidate` → `ranking.filters.EyeFilter` → `eyes.cache.save_eye_detection` → `review.thumbnails.eye_keypoints_for`, the literal function Desktop's Gallery/Loupe overlay calls), and compared its output against `debug_eye_pipeline.py`'s own fresh computation for the same image:

| | Standalone debugger | Production (`eye_keypoints_for`, what Desktop shows) |
|---|---|---|
| left eye (full frame) | (3933.09, 1573.31) | (3976.99, 1659.65) |
| eye box (full frame) | (3847.3, 1487.6, 4018.8, 1659.1) | (3899.0, 1581.7, 4054.9, 1737.6) |
| confidence | 0.9998587 | 0.9998549 |

Box position differs by **44-86 pixels**, both x and y. Confidence differs by ~4×10⁻⁶ (floating-point noise, not meaningful).

Traced the divergence to its exact source. `bird_crop.build_crop` produces two different rectangles for one detection:
- `detection.box` - the *tight* Faster R-CNN box, e.g. `(3443.83, 1449.70, 5392.53, 3598.13)`
- `expanded_box` - the tight box grown by `margin_frac` (default 0.05, i.e. 5% of the box's own width/height on each side) and clamped to the image, e.g. `(3346, 1342, 5490, 3706)`

**The crop image is generated from `expanded_box`** (`crop_to_box(image, expanded)` - `bird_crop.py:780`). But `ranking/classic.py`'s `_load_candidate` (line 661/671) reads `candidate.subject_box` from `selected["box"]` - the **tight** box - never touching `expanded_box`, even though it is right there in the same JSON record (`bird_crop.py:452` writes it, nothing in the read path ever reads it back). `review/thumbnails.py`'s `eye_keypoints_for` has the same problem one level removed: it sources `record.selected` from `analyzer.detections.DetectionCache`, which is also built purely from the tight `selected`/`others` boxes (`analyzer/detections.py:121-137` never references `expanded_box` either).

Both `ranking.debug._projected_eye_box` (the debug-image tool) and `review.thumbnails.eye_keypoints_for` (the Desktop overlay) then compute:
```
scale_x = (subject_box.x2 - subject_box.x1) / crop_width
frame_x = subject_box.x1 + crop_x * scale_x
```
using the **wrong** `subject_box` (tight, not expanded) against a crop that was actually sized to `expanded_box`.

**Proof this fully explains the measured divergence** (not just "a plausible cause"): computed the projection by hand with the tight box's own numbers:
```
scale_x_wrong = (5392.53 - 3443.83) / 2144 = 0.9089   (should be 1.0 - crop was 1:1 with expanded_box)
frame_x = 3443.83 + 587.09 * 0.9089 = 3977.51
frame_y = 1449.70 + 231.31 * 0.9087 = 1659.90
```
This matches the measured production output (3976.99, 1659.65) to within the same ~1px explained by the separate JPEG-cache-roundtrip effect (below) - i.e. the arithmetic **fully reconstructs** the bug from first principles. Also confirmed the scale error's magnitude is exactly what a symmetric 5%-per-side margin predicts: `1/(1+2×0.05) = 0.909`, matching the measured 0.9089 scale factor.

**First function where the two paths diverge:** `ranking/classic.py`'s `ClassicVisionStrategy._load_candidate` (line 671, `candidate.subject_box = tuple(float(v) for v in box)` where `box = selected["box"]`) is the first place production code picks the wrong rectangle. `review/thumbnails.eye_keypoints_for` inherits the same wrong rectangle via a separate path (`analyzer.detections.DetectionCache`), so the bug is not confined to one call site - it's present everywhere `subject_box`/`record.selected` is used as if it were the crop's own rectangle.

**Secondary, smaller effect measured along the way:** the Vision Cache stores crops as JPEG q98 (not the original array). A controlled single-process A/B test (`detect()` on the in-memory crop vs. the same crop after a JPEG q98 round-trip) showed this changes the decoded eye keypoint by only ~0.3-0.5px and confidence by ~4×10⁻⁶ - negligible, and *not* the cause of the 44-86px divergence above.

**Conclusion:** The two paths do **not** produce identical coordinates. The divergence is fully explained and reproduced by one specific, provable bug: production uses the tight detection box instead of the margin-expanded crop box when projecting a crop-space eye coordinate back onto the full photo. This is a real defect in `ranking/classic.py` / `review/thumbnails.py` / `analyzer/detections.py` (the "expanded_box vs. tight box" mismatch), not in EyePose.

**Confidence level:** Proven. Exact arithmetic reconstruction from measured inputs, not inference from correlation.

**Remaining uncertainty:** None on the mechanism itself. Not yet measured: how this scales across the full sample (only algebraically verified on this one image, though the formula is general and detector-agnostic - it would apply identically to SuperAnimal-Bird's overlay). Not yet checked whether any other call site reads `subject_box` this way for a purpose other than the eye overlay (worth a follow-up grep before fixing).

---

## Q2. Is every coordinate transformation correct? (Original → Crop → Model input → Decoded keypoint → EyeRecord → Overlay)

**Evidence:** Each stage checked against its own real production function, using the same 032A2530.CR3 trace:

| Stage | Function | Status | Evidence |
|---|---|---|---|
| Original → Crop | `bird_crop.expand_and_clamp_box` + `crop_to_box` | **Correct** | `expanded_box=(3346,1342,5490,3706)` reproduced bit-identically across 5 repeated runs (see Q8); crop shape (2364,2144) matches `expanded_box`'s own span exactly |
| Crop → Model input | `eyepose_v0._letterbox_forward` | **Correct** | Matches Ultralytics' own `LetterBox(auto=False)` transform, including its `-0.1/+0.1` odd-padding tie-break; previously verified byte-for-byte against the original PyTorch checkpoint's raw forward pass (`docs/eyepose_v0_validation.md`, prior work) |
| Model input → Decoded keypoint | `eyepose_v0._letterbox_inverse` | **Correct** | Exact algebraic inverse of `_letterbox_forward` (`(x-pad_x)/scale`); confirmed by construction and by prior round-trip validation |
| Decoded keypoint → EyeRecord | `eyes.cache.save_eye_detection`/`EyeRecord` | **Correct** | Straight field copy, no transform - read the dataclass and the write function directly, no arithmetic happens here at all |
| EyeRecord → Overlay | `ranking.debug._projected_eye_box` / `review.thumbnails.eye_keypoints_for` | **Incorrect** | See Q1 - uses the wrong source rectangle |

**Conclusion:** 4 of 5 transformation stages are correct and provably so. The final stage (crop-space → full-frame, for display) is broken, per Q1.

**Confidence level:** Proven for all 5 stages (4 correct, 1 incorrect) - each checked against real code and/or repeated-run evidence, not assumed.

**Remaining uncertainty:** None identified for these 5 stages specifically.

---

## Q3. Is the semantic mapping correct? (0=beak, 1=left_eye, 2=right_eye, 3=head_top, 4=left_shoulder, 5=right_shoulder)

**Evidence:** Two independent checks, not one:

1. **The ONNX file's own embedded metadata.** Loaded `cache/eye_models/eye_pose_v0.onnx` and read `session.get_modelmeta().custom_metadata_map` directly:
   ```
   kpt_names: {0: ['beak', 'left_eye', 'right_eye', 'head_top', 'left_shoulder', 'right_shoulder']}
   kpt_shape: [6, 3]
   description: Ultralytics YOLO11n-pose model trained on training\configs\wildlife_bird.yaml
   ```
   This is embedded by Ultralytics' own export pipeline, sourced from the training config that actually produced the model - **not** a comment or a guess. It matches the hardcoded `KPT_NAMES` tuple in `eyepose_v0.py` **exactly**, in the same order.

2. **Independent behavioral check.** Across the 10-image sample, the keypoint labelled `left_eye`/`right_eye` visually landed on the bird's actual eye in 9 of 10 images (see Q4-Q6); `beak` landed at or near the beak tip in every image with a visible beak; `head_top`/`left_shoulder`/`right_shoulder` consistently landed near the top of the head and the wing bases respectively. If the mapping were wrong (e.g. index 1 actually meaning something else), a labelled "left_eye" circle would not visibly land on an eye - it does.

**Conclusion:** The semantic mapping in code is correct, and provably sourced from the model's own conversion metadata rather than only a hopeful comment.

**Confidence level:** Proven with high confidence. One honest caveat: this confirms the *code's* `KPT_NAMES` matches what Ultralytics' export process *recorded* as the training config's own keypoint order. It does not independently re-verify against the original raw annotation files (which this project does not have access to) that Ultralytics' export process itself extracted the right order from - but that would require re-deriving the upstream model from scratch, which is out of scope, and the behavioral cross-check (item 2) independently corroborates the same conclusion via a completely different method.

**Remaining uncertainty:** None material.

---

## Q4. Why are the beak and shoulder landmarks frequently anatomically incorrect? Expected, or a decoding problem?

**Evidence:** Across the 10-image sample, beak and shoulder placement was reviewed visually stage-by-stage (`06_all_keypoints.jpg` for each image):

- 8 of 10 images: beak landed at/near the beak tip, shoulders landed at/near the wing bases - anatomically reasonable.
- 1 of 10 images (032A2018.CR3, a tern in flight with one wing raised, largely covering the head): **every** landmark, not just beak/shoulder, clustered on the wing feathers rather than the actual head (visible at the frame edge). Confidence for the primary eye channel was 0.4535 - just below the 0.5 accept threshold, and the detection was correctly rejected.
- No image showed beak/shoulder confidently mislocalized while eye/head_top were correctly localized - when the model missed, it missed on the *whole* head region at once, not on isolated landmarks.

**Conclusion:** In this sample, beak/shoulder inaccuracy was not a systematic, independent decoding problem - it co-occurred with a genuinely hard, atypical pose (near-total head occlusion by a raised wing), and correlated with low confidence on every landmark in that same frame, not just beak/shoulder. This looks like legitimate model uncertainty on a hard case, not a coordinate-mapping bug (which Q3 already rules out for the *label* assignment, and Q2 rules out for the *transform* arithmetic).

**Confidence level:** Hypothesis, moderately supported. One clear example, consistent with "hard pose, not a bug" - but a sample of 1 failure is not enough to rule out a systematic beak/shoulder-specific bias with certainty.

**Remaining uncertainty:** This needs a larger, purpose-built sample specifically of images with wings raised, birds preening, or heads turned away, to see whether beak/shoulder degrade *faster* than eye/head_top under occlusion (which would suggest a genuine per-landmark weakness) or *together* (supporting "hard pose" over "landmark-specific bug"). Not measured quantitatively - this is a visual read, not pixel error numbers (see Q5's own limitation).

---

## Q5. Does the model systematically miss the eye by a consistent offset? Measure localization error vs. the visible pupil.

**Evidence:** Visual inspection (not sub-pixel manual annotation - see limitation below) of the primary-eye keypoint against the visible pupil center, across every image in the sample where the eye was visible and the crop was correct:

- 032A2530 (egret): keypoint visibly centered on the pupil.
- 032A1560 (kingfisher): keypoint visibly centered on the eye.
- DSC_1179 (kite, head down): keypoint visibly on the eye.
- DSC_4264 (black kite, flight): keypoint precisely on the pupil.
- 032A6869 (bee-eater): keypoint precisely on the eye.
- 032A7114 (bee-eater, close-up): keypoint precisely on the pupil - the clearest, highest-resolution example in the sample.
- 032A2780, 032A4476 (small/distant birds in flight): keypoints appear correctly placed on the head region, though the birds are small enough in frame that sub-pixel confirmation isn't meaningful.

No image in the sample showed the kind of displacement described in the original bug report (eye reported toward cheek, neck, or shoulder) **when the crop actually contained the bird's head**. The one case that superficially resembles that description (032A2018) was correctly rejected (Q4).

**Conclusion:** No systematic directional bias was found in this sample. This directly contradicts the "almost never centered" impression from the original bug report - but see Q1's finding: the original bug report was very likely describing the **on-screen overlay**, which Q1 proves is broken independent of the model's own crop-space accuracy. It is plausible - not yet proven - that fixing the Q1 bug alone resolves most or all of what prompted this investigation.

**Confidence level:** Hypothesis for "no systematic bias," moderately-to-well supported by 8-9 consistent data points, but explicitly **not** a rigorous quantitative measurement.

**Remaining uncertainty:** This is the weakest-evidenced answer in this report by design of tonight's time-box. I did not hand-annotate exact pupil pixel coordinates and compute a numeric pixel-error distribution - I visually assessed "on target" vs. "off target." A rigorous version of this answer needs a proper benchmark: 20-50 images with hand-labeled ground-truth pupil coordinates, giving a real mean/median/max pixel error and a bias vector, not a qualitative impression. Recommended as the first concrete follow-up task.

---

## Q6. Does EyePose correctly distinguish visible vs. hidden eye? Is confidence behavior consistent?

**Evidence:** Confidence values for the *non-primary* (usually hidden, profile-view) eye channel across the sample:

| Image | Visible eye confidence | Hidden eye confidence |
|---|---|---|
| 032A2530 (egret) | 0.9999 | 0.0071 |
| DSC03129 (wrong crop - see Q1) | 0.9709 | 0.0200 |
| 032A2018 (wing covers head) | 0.4535 (rejected) | 0.0071 |
| DSC_4264 (black kite) | 0.9998 | 0.0100 |
| 032A6869 (bee-eater) | 0.9988 | 0.0021 |

The pattern is consistent and clean: the anatomically-hidden eye channel scores near zero (0.002-0.02) in **every** image checked, regardless of whether the visible eye's own confidence was high or borderline. This is exactly the behavior a well-calibrated model should show, and it held even in the one hard-pose failure case (032A2018) - the model correctly reported near-zero confidence for the hidden eye even while spatially misplacing the "visible" one.

**Conclusion:** Confidence behaves consistently and correctly for the visible-vs-hidden distinction across every test performed.

**Confidence level:** Proven for this sample (5/5 clean, consistent separation). Not proven as a universal guarantee.

**Remaining uncertainty:** Small sample. Also untested: front-facing poses where *both* eyes are genuinely visible at once (the whole sample happened to be profile-ish views) - confidence behavior there is unverified.

---

## Q7. Is localization accuracy sufficient for measuring eye sharpness specifically (not just "can it find the eye")?

**Evidence:** Traced exactly which coordinates the sharpness metric actually uses. `ranking/classic.py`'s `measure()` function (line 336):
```python
eye_sharpness=region_focus_measure(candidate.subject_crop, candidate.eye.box)
```
`candidate.eye.box` is the **crop-space** box from `derive_eye_box` - it is never projected to full-frame coordinates for scoring. This means:

- **The Q1 bug does not affect scoring at all.** It only affects the diagnostic overlay drawn for a human to look at. Ranking/keep-reject decisions never go through the broken projection.
- Crop-space keypoint accuracy (Q5) is what actually matters for sharpness measurement, and that has held up well in this sample.
- `derive_eye_box`'s box size (`eye_box_frac=0.08` of the crop's shorter side, floored at 12px) is generously sized relative to a real pupil (which is typically a handful of pixels) - meaning even a several-pixel keypoint error would very likely still leave the true eye inside the measured region, not outside it.

**Conclusion:** For the sharpness metric specifically, current localization precision looks adequate, *and* it is architecturally insulated from the Q1 overlay bug. This is a meaningfully different (better) answer than "can it find the eye" would suggest on its own.

**Confidence level:** Proven that scoring uses crop-space coordinates only (read directly from the code, not inferred). The adequacy judgment itself inherits Q5's "hypothesis, moderately supported" confidence level, since it depends on how accurate the crop-space keypoint truly is at scale.

**Remaining uncertainty:** Same as Q5 - a rigorous pixel-error benchmark would firm this up. Also not tested: whether the eye-box's fixed 8%-of-crop sizing is appropriately scaled across very different subject-to-frame ratios (a bird filling the whole frame vs. a small distant one) - both were present in the sample but not compared quantitatively for this specific question.

---

## Q8. Is the pipeline fully deterministic? Prove it.

**Evidence:** Ran the complete pipeline (RAW decode → bird detection → crop → EyePose inference) **5 times in one process**, same image, same loaded models:

```
Run  expanded_box                    eye_box(crop-space)                          confidence
0 (3346, 1342, 5490, 3706) (501.327903, 145.550836, 672.847903, 317.070836) 0.9998587369918823
1 (3346, 1342, 5490, 3706) (501.327903, 145.550836, 672.847903, 317.070836) 0.9998587369918823
2 (3346, 1342, 5490, 3706) (501.327903, 145.550836, 672.847903, 317.070836) 0.9998587369918823
3 (3346, 1342, 5490, 3706) (501.327903, 145.550836, 672.847903, 317.070836) 0.9998587369918823
4 (3346, 1342, 5490, 3706) (501.327903, 145.550836, 672.847903, 317.070836) 0.9998587369918823

expanded_box identical across all 5 runs: True
eye box (crop space) identical across all 5 runs: True
confidence identical across all 5 runs: True
```
Bit-for-bit identical, including the RAW decode and the GPU (CUDA) bird-detection forward pass.

One separate, smaller observation from Q1's investigation: running the pipeline in **two separate process launches** (not repeats within one process) showed a confidence difference of ~4×10⁻⁶ - consistent with known GPU-library (cuDNN/cuBLAS) kernel-selection nondeterminism across process starts, not a bug, and far too small to explain any of this investigation's findings (the 44-86px divergence in Q1 was fully explained by the box-mismatch bug, not by this).

**Conclusion:** The pipeline is deterministic within a process, proven directly (5/5 identical runs, exact bit-for-bit match on every intermediate value checked). Across separate process launches there is a theoretical, negligible (~1e-6) floating-point wobble typical of GPU inference, with no observed practical consequence.

**Confidence level:** Proven.

**Remaining uncertainty:** None for the within-process case. Cross-process wobble was observed but not exhaustively characterized (only two data points) - it is far too small to matter for this investigation's purposes.

---

## Q9. Does any post-processing modify the model output after inference?

**Evidence:** Read every line between the raw ONNX tensor and the final `EyeDetection` in `eyepose_v0.py`, `detector.py`, and `eyes/cache.py`, specifically searching for smoothing/averaging/clamping/rounding beyond documented coordinate transforms:

- `_decode_best`: a single `argmax` over the 8400 anchor columns' confidence row - no NMS (explicitly: `model.export(..., nms=False)`, confirmed in both the export call and the ONNX file's own `end2end: 'False'` metadata), no thresholding, no averaging across anchors.
- `_letterbox_inverse`: pure coordinate transform (documented, proven correct in Q2), not a value modification.
- `derive_eye_box`: clamps the *derived box's edges* to stay inside the crop - this affects the box's boundary only, never the underlying keypoint/center coordinate used for scoring or display.
- `_point_to_segment_distance` (the anatomical-plausibility gate): computes a distance for an accept/reject decision - does not alter the point's own coordinates, whichever way the gate decides.
- No exponential moving average, no confidence recalibration, no multi-frame temporal smoothing anywhere in this path (this pipeline processes one image at a time, no burst/temporal context reaches EyePose at all).

**Conclusion:** No hidden post-processing modifies the model's predicted location. Every transform between the raw tensor and what gets displayed is one of the five stages already itemized and checked in Q2 - four correct, one (the final projection) proven broken.

**Confidence level:** Proven - based on reading the complete, actual code path, not sampling behavior.

**Remaining uncertainty:** None identified.

---

# Part 2: Alternative models benchmark (research spike - nothing integrated)

Five fundamentally different open-source approaches to bird eye/keypoint localization, covering pose estimation, heatmap-based landmark detection, general-purpose animal pose frameworks, and zero-shot object detection:

| | **EyePose-v0 (current)** | **SuperAnimal-Bird (already integrated)** | **SLEAP** | **MMPose (OpenMMLab)** | **Grounding DINO / YOLO-World** |
|---|---|---|---|---|---|
| Approach | Single-shot landmark regression (YOLO11n-pose) | Heatmap-based top-down pose estimation | Heatmap or top-down/bottom-up multi-animal pose | Heatmap/transformer landmark detection (HRNet/ViTPose/RTMPose) | Zero-shot, text-prompted object detection |
| Paper/project | Ultralytics YOLO-pose; `synthet/eye-pose-v0` checkpoint | Ye et al., "SuperAnimal pretrained pose estimation models," 2023 (arXiv:2203.07436) | Pereira et al., "SLEAP: A deep learning system for multi-animal pose tracking," Nature Methods 2022 | AP-10K: Yu et al., NeurIPS 2021 Datasets & Benchmarks | Grounding DINO (IDEA Research, 2023) / YOLO-World (Tencent AI Lab, 2024) |
| Repository | `synthet/eye-pose-v0` (Hugging Face) | `DeepLabCut/DeepLabCut` + `DeepLabCutModelZoo-SuperAnimal-Bird` | `talmolab/sleap` | `open-mmlab/mmpose` | `IDEA-Research/GroundingDINO`, `AILab-CVC/YOLO-World` |
| License | MIT (weights); Ultralytics/AGPL-3.0 avoided at runtime (see `eyepose_v0.py`) | Code: LGPLv3. **Weights: research/non-commercial use only** | BSD-3-Clause-Clear | Apache 2.0 | Apache 2.0 |
| Last activity | Active (Ultralytics ecosystem) | Active (Mathis Lab; bird detector/model updated recently) | Active (Talmo Lab, Salk Institute) | Active (OpenMMLab) | Active (both) |
| Bird-specific? | Yes (CUB-200 fine-tune) | Yes (dedicated SuperAnimal-Bird dataset/model) | No - general framework, no pretrained bird-eye model | **No pretrained bird model** - AP-10K is mammal-only (23 families, 54 species, zero birds) | No - fully generic, zero-shot |
| Expected accuracy | Good on correctly-cropped subjects (this investigation) | Good on correctly-cropped subjects (this investigation) | Unknown - would need training on a new bird-eye dataset | Unknown for birds - would need retraining, no shortcut | Documented weakness on **small objects** (an eye is a small object) |
| Runtime | Fast (ONNX Runtime, single shot) | Slower (heatmap CNN, still real-time) | Real-time capable | Varies by backbone (HRNet moderate, RTMPose fast) | Slower than either (transformer-based for Grounding DINO; YOLO-World faster but less accurate on small objects) |
| GPU support | Yes (CUDA via onnxruntime-gpu) | Yes (torch) | Yes (torch/TF) | Yes (torch) | Yes (torch) |
| Install complexity | Already integrated | **Already integrated** | Moderate (own GUI app + Python API) | Moderate-high (OpenMMLab dependency stack: mmcv, mmengine, mmpose) | Moderate (transformers/detectron2-adjacent stacks) |
| Maturity for this use case | Already in production | Already in production, alternate backend | Would require building a labeled bird-eye dataset from scratch | Would require building a labeled bird-eye dataset from scratch (same effort as SLEAP, different architecture) | Usable immediately, zero training - but the small-object weakness is a direct, known mismatch with our exact requirement |
| Advantages | Fast, already validated tonight | Fundamentally different architecture family (heatmap vs. regression) - useful cross-check; already zero-cost to run | Very actively maintained, purpose-built for exactly this kind of problem (ecology/behavior tracking), strong tooling for building a *new* training set if ever needed | Enormous ecosystem/model zoo, very active; RTMPose variants are fast | Zero training data needed; could sanity-check "is there an eye-shaped thing here at all" independent of any bird-specific model |
| Disadvantages | N/A (baseline) | Non-commercial weights license is a real constraint for a distributed app (already the reason it isn't the sole backend) | No pretrained bird-eye model - this is a "build from scratch" option, not a drop-in alternative | Same - no pretrained bird-eye model; AP-10K's mammal-only keypoint scheme doesn't even transfer conceptually well to a bird's anatomy | Known weak on small objects (exactly our failure mode of concern); doesn't give a calibrated per-eye confidence the way a keypoint model does; heavier runtime |

**Selected candidate to actually run tonight: SuperAnimal-Bird.** Reasoning: it is the only candidate that (a) is a genuinely different architecture family (heatmap-based top-down pose estimation vs. EyePose's single-shot regression - directly satisfying the "heatmap-based landmark detection" category asked for), (b) has a bird-specific pretrained model (no training gap), and (c) required zero new integration risk since it's already a registered `EyeDetector` backend in this codebase - allowing a true same-image, same-infrastructure comparison rather than a rough approximation. The other four candidates were researched in full but not run tonight, since each would require either building a new labeled dataset (SLEAP, MMPose) or accepting a documented small-object weakness with no calibrated confidence signal (Grounding DINO/YOLO-World) - all reasonable *next* investigations, not tonight's scope.

**Result, same egret crop, same expanded_box:**

| | EyePose-v0 | SuperAnimal-Bird |
|---|---|---|
| Primary eye (crop space) | (587.1, 231.3) | (594.1, 204.4) |
| Confidence | 0.9999 | 0.9240 |
| Accepted | True | True |
| Visual verdict | Precisely on the pupil | Also correctly on the pupil (slightly less tight than EyePose, ~27px higher) |

Both models localize the eye correctly on this image. Neither shows the kind of gross mislocalization described in the original bug report. This is a second, independent piece of evidence (alongside Q1-Q9) that the *model* is not where this investigation's root cause lives.

**Can another modern model localize the eye more accurately than EyePose on this image?** No clear win either way on this single test - both are visually accurate, EyePose scored higher confidence and landed marginally closer to the pupil's exact center. A real accuracy comparison would need the same rigorous pixel-error benchmark flagged as missing in Q5.

---

## Overall recommendation

**B. Fix integration issues.**

Evidence-based reasoning, not speculative:

- The one fully-proven, high-confidence root cause found tonight (Q1/Q2) is a coordinate-projection bug in the **integration layer** (`ranking/classic.py`, `review/thumbnails.py`, `analyzer/detections.py`) - using the tight detection box instead of the margin-expanded crop box. This is a small, well-localized, mechanical fix (read `expanded_box` instead of `selected.box` at the identified call sites), not a model problem.
- That bug affects **only the diagnostic overlay** - the actual ranking/scoring math never goes through it (Q7), which is good news for confidence in past ranking runs, but also means the visual symptom the user has been seeing was likely worse-looking than the underlying data.
- Across a real, diverse sample - including several genuinely hard poses - EyePose's own crop-space keypoint decode, semantic mapping, coordinate transforms, and determinism all check out (Q2, Q3, Q8, Q9).
- The one clean "eye reported where none should be" case found (DSC03129) was root-caused to a **separate, also-fixable integration bug**: `bird_crop.select_best_detection`'s area-dominant policy choosing an incorrect, low-confidence "cow" detection over a correct, high-confidence bird detection, because the false positive's box happened to be larger. Also not a model problem, and also not something replacing EyePose would fix.
- SuperAnimal-Bird (a fundamentally different architecture, already available) shows comparable accuracy on the one test performed tonight - no evidence surfaced that swapping the model would resolve anything the two integration bugs above don't already explain.

**D (continue benchmarking) is a reasonable secondary track**, not urgent: the Q5 pixel-error benchmark is the most valuable next step regardless of which model is used, since it would firm up several "hypothesis, moderately supported" answers above into proven ones, and would make any future model comparison actually rigorous rather than visual.

**A (continue investing in EyePose) and C (replace EyePose)** are not supported by tonight's evidence either way - no model-level defect was found to justify replacement, and no investigation was performed into whether EyePose's *training data or checkpoint itself* could be improved (out of scope tonight, and not indicated as needed by anything found).

---

## Appendix: artifacts produced tonight

- `tools/debug_eye_pipeline.py` - the forensic tool itself (already existed at investigation start, built earlier this session).
- Debug output for 10 images (`01_original.jpg` through `07_final_overlay.jpg`, `05_raw_output.json`, `report.md` each) - available on request, not committed to the repo (real, private photos from the user's archive).
- Three scratch investigation scripts (Q1 parity check with custom cache dir, Q1 parity check with default cache dir, JPEG round-trip A/B test, determinism proof, SuperAnimal-Bird comparison) - one-off, not committed; logic summarized inline above.

---

# Phase 2: Implementation

Everything below implements the recommendation ("B - fix integration issues") against the findings above. Scope was deliberately bounded to what the investigation actually evidenced - no unrelated refactoring, no model changes, no new UI.

## 1. Projection fix (Q1)

**Change:** the crop's own rectangle (`bird_crop.CropResult.expanded_box` - the tight detection box grown by `CropParams.margin_frac`, which is what the cached crop's pixels actually span) is now threaded through and used everywhere a crop-space coordinate is mapped back onto the full frame. Previously, `ranking.classic._load_candidate` and `review.thumbnails.eye_keypoints_for` both used the *tight* detection box for this - a real rectangle mismatch present on every image with a margin > 0 (the default).

**Files changed:**
- `analyzer/detections.py` - `DetectionRecord` gained an `expanded_box` field, read from the same JSON `bird_crop.save_detections` already wrote (no cache rebuild needed for this field specifically - it was already on disk, just never read back). The rare backfill path (`DetectionCache._detect`, used only for images preprocessing never recorded at all) now computes a self-consistent `expanded_box` itself via `bird_crop.expand_and_clamp_box`.
- `ranking/filters.py` - `FilterCandidate` gained a `crop_box` field, documented as distinct from `subject_box` (the tight box, still correctly used for the subject-size metric and the subject-box overlay - that one was never wrong).
- `ranking/classic.py` - `_load_candidate` now populates `crop_box` from the detection record's `expanded_box`.
- `ranking/debug.py` - `_projected_eye_box` now scales against `candidate.crop_box`, not `candidate.subject_box`.
- `review/thumbnails.py` - `eye_keypoints_for` now scales against `record.expanded_box`, not `record.selected`.

**Verified NOT changed:** the subject-size scoring metric and the green subject-box overlay both still correctly use the tight box (`subject_box`) - only the eye-projection math moved to `crop_box`/`expanded_box`.

## 2. Detection selection fix (Q1, DSC03129)

**Change:** `bird_crop.select_best_detection`'s policy inverted - confidence now dominates, area only breaks a tie between similarly-confident detections (within `confidence_tie_frac`, default 10%, as a fraction of the winning confidence). Previously area dominated and confidence only broke a tie between similarly-*sized* detections, which is exactly what let a low-confidence false positive with a large box beat a correct, high-confidence detection.

Renamed `CropParams.area_tie_frac` → `confidence_tie_frac` (semantics inverted along with the algorithm - keeping the old name with new meaning would have been actively misleading). `CROP_CACHE_VERSION` bumped `v5` → `v6`, so every existing cached crop is recognized as built under the old policy and gets rebuilt automatically on next use - no silent reuse of a crop that might have come from the wrong subject.

Deliberately generic: `select_best_detection` never reads a detection's `label`, so the new policy applies identically to birds, mammals, and any future catalogued class - confirmed by `test_a_real_false_positive_no_longer_beats_a_confident_true_positive` using the exact DSC03129 numbers (a bird at 0.998 vs. a "cow"-labelled false positive at 0.458).

**Files changed:** `bird_crop.py` (the algorithm, `CropParams`, `BirdDetector`), `preprocess.py` and `inspect_crops.py` (CLI flag `--area-tie-frac` → `--confidence-tie-frac`, printed summary text).

**A real bug found and fixed while validating this change:** `bird_crop.read_crop_params` used to unpack a stored `crop_params.json` directly into `CropParams(**data)`. A real v5 cache file has the old `area_tie_frac` key, which crashed with `TypeError` before `build_cache`'s own version-mismatch check ever ran - discovered by actually re-running the production path against this repository's own real `cache/crops` during validation, not found by unit tests alone. Fixed by having `read_crop_params` drop unrecognised keys before constructing `CropParams`; the stored `version` field is still read normally, so the *version* mismatch (the thing that actually matters) is still caught exactly as before, with the same "pass --force to rebuild" message instead of a stack trace. Covered by a new test using a literal v5-shaped payload.

## 3. Eye confidence filtering (Q3)

**Finding on arrival:** this was already almost entirely implemented. `EyePoseV0EyeDetector` already kept only the higher-confidence eye channel as `EyeDetection`'s primary box (the weaker channel is retained in `EyeDetection.left`/`.right` for the debugging overlay only, never for scoring), and `EyeFilter` already skipped a rejected image out of `measure()` entirely.

**Change made:** `ClassicVisionEyePoseParams.min_eye_confidence` renamed to `eye_confidence_threshold`, matching this report's own Part 3 naming exactly, so it reads as a first-class, named concept rather than an implementation detail. (`EyePoseV0EyeDetector`'s own constructor kwarg stays `min_confidence` - matching `SuperAnimalBirdEyeDetector`'s identical kwarg name; only the outer, photographer-facing params field changed.) `ranking.filters.REJECT_REASON_LABELS[NO_VISIBLE_EYE]` changed from "No visible eye" to "No reliable visible eye", matching this report's own Part 3 phrasing - the old label undersold what the gate actually checks (confidence AND anatomical plausibility, not just "was any eye channel present at all").

## 4. Head confidence investigation (Q4/Part 4)

**Decision: not implemented**, per the explicit instruction to leave behaviour unchanged and explain why if the data doesn't support it. Pulled the real `head_top` confidence values out of all 10 investigation images' saved raw output:

| Image | Eye confidence | head_top confidence |
|---|---|---|
| 032A2530 (good) | 0.9999 | 0.9991 |
| 032A1560 (good) | 1.0000 | 0.9973 |
| DSC_1179 (good) | 0.9941 | 0.9896 |
| DSC_4264 (good) | 0.9998 | 0.9982 |
| 032A2780 (good) | 0.9990 | 0.9972 |
| 032A4476 (good) | 0.9999 | 0.9998 |
| 032A6869 (good) | 0.9988 | 0.9987 |
| 032A7114 (good) | 0.9993 | 0.9846 |
| 032A2018 (correctly rejected - hard pose) | 0.4535 | 0.2303 |
| DSC03129 (wrongly accepted - wrong crop, since fixed) | 0.9709 | 0.9159 |

Two things this rules out cleanly:
- In the one case confidence alone correctly rejected (032A2018), head_top was *also* low (0.23) - combining would not have changed an already-correct decision.
- In the one case confidence alone wrongly accepted (DSC03129), head_top was *also* high (0.92) - combining would **not** have caught that failure. Both channels were confidently fooled together, because the actual problem was upstream (the crop itself, fixed in §2) - not something either keypoint's own confidence could diagnose.

Across every sample, head_top confidence tracked eye confidence rather than adding independent signal. This reasoning (with the same data) is written directly into `eyes.eyepose_v0.accepts_eye`'s own docstring, so it stays next to the code it explains rather than only in this report.

## 5. Cache architecture (Q5/Part 5)

**Finding on arrival:** the architecture the report asked to confirm was already almost entirely in place. The Crop Cache (`bird_crop`) and EyePose's own results (`eyes.cache.EyeRecord`) are already two independent, separately-versioned caches; `eye_confidence_threshold` has never affected the crop cache (it is not a `CropParams` field, never was) - so "changing thresholds never requires rebuilding the crop cache" was already true for that threshold before tonight.

**Change made:** the accept/reject decision (confidence threshold + anatomical plausibility) was pulled out of `EyePoseV0EyeDetector.detect()` into a standalone pure function, `eyes.eyepose_v0.accepts_eye(primary, landmarks, *, min_confidence, max_head_distance_ratio)` - no model, no I/O, a function of already-decoded keypoints and two numbers. `detect()` calls it directly rather than duplicating the logic; behaviour is unchanged (pinned by the existing `detect()`-level tests plus new direct tests of `accepts_eye` itself). This is the "Decision Engine" the report's diagram asked for, made real and reusable rather than only conceptual.

**Explicitly not done, and why:** wiring `rank_folder` to skip re-running EyePose entirely when a fresh cached `EyeRecord` already exists (making a threshold change literally free rather than merely "no crop rebuild") was considered and deliberately deferred. `EyeRecord` currently persists the primary eye's `confidence` but not `beak`/`head_top` (needed to re-check anatomical plausibility, the other half of the gate, without re-running inference) - and adding EyePose-specific fields to `EyeRecord`/`EyeDetection` would compromise their deliberate genericness across detector backends (SuperAnimal-Bird has no `beak`/`head_top` concept at all). Given Part 4's finding that combining head confidence isn't needed, extending the cache schema for it isn't justified either. This is flagged as the natural next step if EyePose inference time itself becomes the bottleneck (it wasn't measured as one tonight); `accepts_eye`'s own docstring names this seam explicitly.

**Important distinction documented, not just implemented:** `detection_confidence_threshold`/`confidence_tie_frac` (§2/§6) legitimately **do** require a crop-cache rebuild when changed - they can select a different subject, so a different crop. This is not a violation of the "thresholds are free" principle; it's a different, upstream kind of parameter, and `CropParams`'s own existing version-mismatch mechanism already enforces the rebuild correctly (no new code needed for that half).

## 6. Configuration (Part 6)

Both Classic Vision backends' params dataclasses (`ClassicVisionParams`, `ClassicVisionEyePoseParams`) gained `detection_confidence_threshold` and `confidence_tie_frac` fields with `ParamSpec` entries (factored into a shared `_detection_specs()` so the two backends can never drift apart on what these mean), following the existing self-generating-dialog pattern exactly - no UI code was written, matching the "configuration only" instruction. `rank_folder` now builds `CropParams(conf_threshold=params.detection_confidence_threshold, confidence_tie_frac=params.confidence_tie_frac)` instead of a bare `CropParams()`, so these are no longer silently hardcoded. `eye_confidence_threshold` (§3) was already dialog-ready as `min_eye_confidence`; only its name changed. No `head_confidence_threshold` parameter was added, per §4's finding.

## 7. Validation

Re-ran `tools/debug_eye_pipeline.py` on all 10 investigation images after the fix.

**Projection fix, egret (`032A2530.CR3`)** - compared the standalone debugger's own output against the real production path (`build_cache` → `_load_candidate` → `EyeFilter` → `save_eye_detection` → `eye_keypoints_for`, the literal function the Desktop overlay calls):

| | Before | After |
|---|---|---|
| Eye box divergence (debugger vs. production) | 44-86 px | ≤0.5 px |

The residual ≤0.5px is not new - it is the already-identified, negligible JPEG q98 crop-cache round-trip effect from the original Q1 investigation, confirmed again by a controlled single-process A/B test at the time. The 44-86px *systematic* divergence is gone.

**Detection selection fix, full 9-image batch** - re-ran every image from the original investigation sample (not just DSC03129) and diffed the selected detection and eye result against the pre-fix run:

| Image | Selected detection | Eye result |
|---|---|---|
| 032A1560, DSC_1179, DSC_4264, 032A2018, 032A2780, 032A4476, 032A6869, 032A7114 (8 images) | **Unchanged** | **Unchanged** |
| DSC03129 | **Changed**: 0.998-confidence bird (was: 0.458-confidence "cow" false positive) | **Changed**: eye now correctly on the bird (was: on the metal post the wrong crop happened to contain) |

Visual confirmation (`07_final_overlay.jpg`, before vs. after): before, the green subject box spanned a dark post, a truck mirror, and only incidentally the bird, with the pink eye box on a rusty spot on the post; after, the green box tightly bounds the actual bird and the pink eye box sits precisely on its visible eye.

This is the result the fix should produce: the one image that was actually broken is fixed, and nothing that was already correct changed - the new confidence-dominant policy agrees with the old area-dominant one whenever (as in 8 of these 9 real photos) there was only one real candidate to begin with, and differs exactly where a false positive was previously able to win on size alone.

**Low-confidence rejection (032A2018)** - `accepted=False`, confidence 0.4535, unchanged before/after (this image's correct behaviour was never affected by either bug - included as a control, not a fix).

**Full test suite:** 1084 tests passed after all changes (3 pre-existing tests updated for the intentional behaviour changes: two pinned `CROP_CACHE_VERSION == "v5"`, now `"v6"`; one pinned the old reject-reason label text).

## 8. Remaining limitations

- The Q5/Q7 pixel-error benchmark this report already flagged as the most valuable next step (hand-labelled ground-truth pupil coordinates across a larger sample) is still not done - nothing in tonight's implementation depended on it, but it remains the way to turn several "hypothesis, moderately supported" findings into proven ones.
- `review.thumbnails.eye_keypoints_for`'s `crop_cache_dir` parameter still does not reach the internal `_detections()` singleton (a real, separate bug found but not fixed during the investigation, since it only affects a non-default cache directory - not how Desktop actually calls it). Left as documented, out of scope.
- The "free threshold changes" architecture in §5 covers `eye_confidence_threshold` conceptually (via `accepts_eye`) but not yet operationally (`rank_folder` still re-runs EyePose on every call, regardless of whether a fresh cache entry exists) - deferred, with the reasoning above, pending evidence that inference time is actually a bottleneck.
- Validation tonight re-ran the same 10-image sample from the original investigation, not a fresh, independent set - appropriate for confirming the two specific bugs found are fixed and nothing regressed, but not a substitute for the larger benchmark above.

---

# Phase 3: Decision Engine (absolute-confidence selection + independent head-confidence gate)

**Date:** 2026-08-02. **Trigger:** after reviewing Phase 2's §2 fix, two design objections were raised before the architecture was allowed to freeze - not new bugs, but a challenge to whether §2's *policy* (confidence-dominant, area as tie-break) was the right one long-term, plus a correction to what §4's "head confidence" question actually needed to mean. Analysis was done and approved before any code changed; this section documents the resulting implementation and its validation.

## 1. The two design questions

**1a. Is confidence-dominant selection (§2's v6 fix) actually correct?** Objection, from project history: confidence-dominant selection was tried before and abandoned, because it picked small, distant, high-confidence detections over the actual photographic subject - which is exactly why the project moved to area-dominant in the first place. DSC03129 (§2) then proved pure area-dominance is *also* wrong. Neither extreme is correct on its own; the hypothesis was that the selection score should combine both confidence (reliability) and relative size (intent) - a weighted score, a normalized score, or some other mathematically justified combination.

**1b. Can we tell whether the bird's head is actually visible and correctly localized, independent of the eye's own confidence?** A correction to §4's question, not a repeat of it: §4 asked whether the `head_top` *landmark's* confidence correlates with the eye's confidence (it found: no independent signal - both channels get fooled together). The real question is different: can the pipeline detect when the head is hidden, facing away, outside the crop, or otherwise not really there *before* trusting whatever the eye channel reports - so the system rejects uncertain cases rather than confidently ranking on an unreliable eye location.

## 2. Why a linear combination of confidence and area doesn't work

Tested algebraically against two real failure modes, using a linear score `S = α·confidence + β·area_frac` (area normalized to some reference, weights nonnegative):

- **DSC03129 (§2, measured):** a correct bird at confidence 0.998 with a smaller box must beat an incorrect "cow" false positive at confidence 0.458 with a *larger* box. This requires `α·0.998 + β·area_bird > α·0.458 + β·area_cow`, i.e. `α·0.540 > β·(area_cow − area_bird)` - satisfied only if `α` is weighted heavily enough relative to `β` given how much larger the false positive's box was.
- **The historical failure mode that originally killed confidence-dominant selection:** a small, distant, spuriously high-confidence detection (score close to 1.0, tiny box) must *lose* to the true, larger photographic subject at a lower-but-still-legitimate confidence (e.g. 0.7-0.9). This requires the opposite: `β` weighted heavily enough that a large area difference overcomes even a substantial confidence gap.

These two constraints pull `α`/`β` in opposite directions - there is no single fixed weighting that satisfies both real cases at once, because they are not the same failure mode measured on the same axis. One is "is this detection trustworthy at all" (a reliability question, answered by confidence, with a natural absolute threshold - a detection is either believable or it is noise); the other is "which trustworthy detection is the actual subject" (an intent question, answered by size, once trustworthiness is no longer in doubt). A weighted sum forces these two different questions to trade off against each other on one shared scale, which is exactly what produces contradictory weight requirements. The fix is to keep them as two sequential, independent decisions rather than one combined score - never asking area to compensate for a detection that shouldn't be trusted, and never asking confidence to arbitrate between two detections that are already both trusted.

## 3. v7 selection policy: absolute gate, then area among survivors

**Change:** `bird_crop.select_best_detection` replaced again (v6 → v7, `CROP_CACHE_VERSION` bumped accordingly). Candidates are first filtered to those clearing a configurable **absolute** reliability floor, `CropParams.min_crop_confidence` (default `0.6`, informed by - not exhaustively validated against - Phase 1's sample: every real, class-eligible detection observed was ≥ 0.94; the one confirmed false positive topped out at 0.458). Area then selects the subject *only among survivors* - never used to rescue a detection that failed the floor, never competing against confidence on a shared scale.

This resolves both §2 cases without contradiction, because they're no longer forced onto one axis:
- DSC03129's 0.458 "cow" is excluded by the floor outright (0.458 < 0.6) - it never even reaches the area comparison, regardless of its box size.
- A small, spuriously-confident distant detection no longer auto-wins just for edging out a legitimate large subject on confidence alone - if both clear the reliability floor, the larger (true subject) box wins, exactly like the original area-dominant policy this project started with, restored now that low-confidence noise can no longer sneak into that comparison.

If no candidate clears the floor, `select_best_detection` returns `None` (not a forced "least-bad" pick) - `build_crop`'s existing full-frame fallback and `SubjectFilter`'s `NO_SUBJECT` rejection handle that downstream, consistent with "prefer honest rejection over confident-but-wrong."

**Files changed:** `bird_crop.py` (`DEFAULT_MIN_CROP_CONFIDENCE`, `CropParams.min_crop_confidence`, `select_best_detection`, `BirdDetector`), `ranking/classic.py` (`crop_confidence_threshold` param/spec, replacing `confidence_tie_frac`), `preprocess.py`/`inspect_crops.py` (CLI flag `--confidence-tie-frac` → `--min-crop-confidence`).

## 4. Independent head-confidence gate

**The signal:** EyePose-v0's decoder (`_decode_best`) reads a per-anchor "is a real bird-head instance here at all" score (`predictions[4, :]`) before ever looking at individual keypoints - the winning anchor's own pre-decode confidence, now surfaced as `detection_confidence`. This is genuinely independent of any single landmark's confidence (including the primary eye's own), because it answers a different question: not "where is the eye, and how sure is that placement" but "is there a real head instance in this crop at all."

**Why it's needed, with real measured data:** on DSC03129's *original*, mis-selected crop (before §2's fix - see §2's case study), the eye landmark itself reported a misleading 0.97 confidence, yet `detection_confidence` measured **0.026** - against 0.82-0.92 on every real head in the sample. A crop containing no real bird head can still produce a confident-looking guess for where "the eye" would be; per-landmark confidence alone cannot catch this, because the model is answering "if there were an eye here, where would it be" even when there is no head to have one. This is the exact scenario §2's selection fix already prevents at the *selection* stage - the head-confidence gate is a second, independent line of defense for the same failure mode, active even if a future false positive were ever confident enough to clear §3's selection floor. (This value is pinned permanently as a regression test: `tests/test_eyepose_v0.py::test_a_low_head_confidence_is_rejected_independent_of_eye_confidence`.)

**Implementation:** `detection_confidence` is now propagated end to end - `EyePoseV0EyeDetector.detect()` returns it on `EyeDetection.head_confidence`, alongside a derived `EyeDetection.head_visible: bool` (whether it clears a configurable `min_head_confidence`, default `DEFAULT_MIN_HEAD_CONFIDENCE = 0.5`). Both fields default to backend-agnostic no-op values (`None`/`True`) on the generic `EyeDetection` dataclass, so SuperAnimal-Bird (which has no equivalent single scalar) is completely unaffected. `eyes.cache.EyeRecord`/`save_eye_detection`/`read_eye_detection` persist `head_confidence` alongside the rest of the cached result (`EYE_CACHE_VERSION` bumped `1` → `2`). `ranking.classic.ClassicVisionEyePoseParams` gained `detection_head_confidence_threshold` (dialog-configurable, matching every other threshold in this pipeline).

## 5. Decision Engine: three independent gates, one detector call

`ranking.filters.EyeFilter.check()` now evaluates three genuinely independent questions in sequence, each with its own rejection reason, from a **single** `detector.detect()` call (no duplicate inference):

1. **`UNSUPPORTED_SUBJECT`** - is this subject a class the eye detector even covers (unchanged from Phase 2 §3).
2. **`LOW_HEAD_CONFIDENCE`** (new) - does `EyeDetection.head_visible` hold - is a real head instance present at all (§4).
3. **`NO_VISIBLE_EYE`** - does `EyeDetection.accepted` hold - given a head is there, is *this* eye trustworthy (confidence + anatomical plausibility, unchanged from Phase 2 §3).

`accepted` and `head_visible` are deliberately never merged into one flag - a caller (or a future debugging overlay) can always tell *which* of the three questions failed, not just that something did.

## 6. Validation

Re-ran `tools/debug_eye_pipeline.py` (Stage 7a extended to also print `head_confidence`/`head_visible`) on the same 10-image sample, on top of the real, current `CropParams()` (now `min_crop_confidence=0.6`, `version='v7'`):

| Image | Selected detection confidence | Candidates considered | `head_confidence` | `head_visible` | Eye `accepted` | Eye confidence |
|---|---|---|---|---|---|---|
| 032A2530 (egret) | 0.9990 | 1 | 0.828 | True | True | 0.9999 |
| 032A1560 (kingfisher) | 0.9969 | 1 | 0.915 | True | True | 1.0000 |
| DSC03129 (§2 case, now correctly selected) | 0.9984 | 6 (1 bird ≥ floor; the 0.458 "cow" and 4 others excluded by the floor) | 0.912 | True | True | 0.9999 |
| DSC_1179 (kite, head down) | 0.9985 | 1 | 0.830 | True | True | 0.9941 |
| DSC_4264 (black kite) | 0.9981 | 1 | 0.907 | True | True | 0.9998 |
| 032A2018 (wing covers head) | 0.9988 | 1 | 0.903 | True | **False** (0.4674 < 0.5) | 0.4674 |
| 032A2780 (distant, in-flight) | 0.9868 | 1 | 0.920 | True | True | 0.9990 |
| 032A4476 (distant, in-flight) | 0.9951 | 1 | 0.901 | True | True | 0.9999 |
| 032A6869 (bee-eater) | 0.9903 | 2 (0.990 and 0.328; only 0.990 ≥ floor) | 0.819 | True | True | 0.9988 |
| 032A7114 (bee-eater close-up) | 0.9424 | 2 (0.942 and 0.818; both ≥ floor) | 0.829 | True | True | 0.9993 |

**Which images changed, and why:** none, at the level of final crop-target selection or accept/reject outcome - every image's outcome matches the post-§2 (v6) state exactly. This is expected, not a null result: this specific 10-image sample contains no case where v6 and v7 disagree.
- Where only one bird-labelled candidate ever existed (8 of 10 images), v6 and v7 trivially agree - there was nothing to select between either way.
- DSC03129's false positive (0.458) was already below v6's confidence-dominant winner by a wide margin, so v6 already picked the correct bird; v7's floor (0.6) also excludes it, for the same underlying reason via a different mechanism (an absolute cutoff vs. a comparative one) - v7 does not depend on the false positive happening to lose a head-to-head comparison, which is the actual point of the redesign (§2).
- 032A7114 is this sample's only case where v7's "area among survivors" logic is actually exercised (two candidates, 0.942 and 0.818, both clearing the 0.6 floor): the larger box (0.942, area ≈ 999k px²) beats the smaller one (0.818, area ≈ 472k px²) - and also happens to be the higher-confidence one here, so v6 would have picked it too. No image in this sample has two floor-clearing candidates where the *larger* one is *not* also the more confident one - the scenario that motivated §1's redesign (a small, highly-confident false positive vs. a large, legitimately-confident true subject) simply isn't present in this specific sample. That scenario is precisely why §2's proof was done algebraically (§2 above) rather than only empirically: this sample cannot exercise it, but the DSC03129 case (a related but different shape of the same underlying problem: confidence and size disagreeing about which detection to trust) is real, measured evidence that the failure mode exists.
- The head-confidence gate never fires in this sample either: every accepted image's `head_confidence` falls in 0.819-0.920 (comfortably above the 0.5 floor), and the one rejected image (032A2018) was already correctly rejected by the pre-existing eye-confidence gate (0.4674 < 0.5), independent of `head_visible` (which is `True` here - the head genuinely is visible in that crop; the *eye* channel is what the model was unsure about, a different, correctly-still-firing gate). The gate's real motivating evidence remains DSC03129's *original*, pre-fix crop (§4's 0.026 measurement) - not reproducible now that §3 prevents that crop from ever being selected again, which is itself the intended outcome: the two layers now overlap in coverage by design.

**Is the new behaviour actually better?** On this sample: no regression (every previously-correct outcome is unchanged) plus two new, currently-dormant protections whose necessity is demonstrated by real measured data from this investigation rather than by this sample alone - §2's algebraic proof (the small-confident-detection case) and §4's 0.026 measurement (the confident-eye-on-no-head case) are both real, previously-observed failure shapes that v6/pre-gate code had no defense against. Confirming they *actually* trigger correctly on a live false positive would need a sample containing one (see "Remaining limitations" below).

**Full test suite:** 1095 tests passed (11 new tests added this round: `EyeFilter`'s `LOW_HEAD_CONFIDENCE` path and its "no signal, never gated" counterpart; `head_visible()`'s own direct unit tests; `eyes/cache.py` `head_confidence` round-trip coverage - previously 1084 in Phase 2, plus tests already added during this round's own implementation).

## 7. Remaining limitations

- Validation (§6) confirms **no regression** on the known 10-image sample, but that sample happens not to contain a live example of either failure mode this redesign targets (a small-confident/large-legitimate selection conflict, or a confident false positive that clears the new selection floor). The evidence that these failure modes are real comes from §2's algebraic argument and §4's DSC03129 pre-fix measurement, not from a live positive hit in this validation run. A dedicated follow-up sample specifically containing small-and-confident vs. large-and-legitimate detection pairs (and, if one can be found or synthesized safely, a high-confidence non-bird false positive) would let both new mechanisms be validated by direct observation rather than by proof and historical measurement.
- `min_crop_confidence` (0.6) and `min_head_confidence` (0.5) are both informed by, but not exhaustively tuned against, this project's still-small sample - same caveat Phase 2 already carried for its own thresholds.
