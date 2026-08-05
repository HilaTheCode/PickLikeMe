# PickLikeMe version roadmap

PickLikeMe is a personal preference model: it learns to imitate my own select/reject
decisions on wildlife and bird photography RAW files. It is not a generic image-quality
ranker. The dataset is ~50,000 RAW images (~10,000 selected, ~40,000 rejected), mostly
organized in bursts.

This roadmap improves the system scientifically: **each version changes exactly one
major component** so that every improvement can be measured in isolation against the
previous version.

> This roadmap covers the **AI Model** strategy only. **Classic Vision Ranking** - a
> separate, deterministic, non-learned strategy (subject/eye detection + sharpness/size
> metrics, no training) - ships alongside it and is versioned independently; see
> README.md's "Analysis modules (AI Model, Classic Vision)" section. It is intentionally
> absent from the V1-V9 sequence below, since it never changes "one major component" of
> this model - it is not a version of it.

## Rules

1. One major change per version. Never combine two improvements.
2. Every version is trained and evaluated with the same protocol (see below) so that
   metric deltas are attributable to that version's single change.
3. Backward compatibility is preserved whenever possible: checkpoints, CLI flags, and
   the single-preference-score output contract stay stable.
4. A version is only accepted if it improves agreement with my decisions on the
   held-out set (or is explicitly accepted as neutral for other reasons, e.g. speed).

## Fixed evaluation protocol (applies to all versions)

- One held-out split, created once and frozen: split **by burst**, never by image, so
  no burst has frames in both train and test. Store the split file in `data/` and
  reuse it for every version.
- Primary metric: **Top-1 burst accuracy** — for each held-out burst, does the model's
  highest-scored frame match one of my selected frames?
- Secondary metrics: Top-3 burst accuracy, image-level ROC AUC, precision/recall at a
  fixed threshold.
- Fixed random seed per run; report each version as mean over the same seeds.
- Results for each version are recorded in `docs/results/` as `vN_results.md`
  (metrics table + training config + comparison against the previous version).

---

## V1 — Baseline (current implementation, frozen)

Keep unchanged. This is the reference point.

- RAW loading via rawpy (`src/picklikeme/raw_io.py`)
- Stretch-resize to 384×384 (`RawImageLoader.load_image`)
- Small custom CNN backbone + linear head (`src/picklikeme/model.py`)
- Single preference score output
- MSE loss on select(1)/reject(0) labels (`src/picklikeme/train.py`)

Action: tag the current state (e.g. git tag `v1-baseline`) and run the evaluation
protocol once to establish baseline numbers.

## V2 — Aspect-ratio preserving preprocessing

**Single change:** replace the stretch-resize in `RawImageLoader.load_image` with a
letterbox resize.

- Resize the longest side to 384, preserving aspect ratio, then pad the shorter side
  (constant pad, e.g. black or mean color) to reach 384×384.
- No stretching, no cropping, no bird detection. The full frame is always preserved.
- Network input size is unchanged, so `PreferenceHead` and training code are untouched.

Rationale: stretching distorts subject geometry (wing shape, head proportions), which
are likely inputs to my real decisions.

Files: `src/picklikeme/raw_io.py` only.

## V3 — Pretrained vision backbone

**Single change:** replace the custom CNN backbone with a modern pretrained backbone.

Preferred order:

1. **DINOv3** (self-supervised ViT features; strong for fine-grained visual judgments)
2. **SigLIP2** (vision-language pretraining; strong semantic features)
3. **ConvNeXt** (fallback; well-supported convolutional option)

Design constraints:

- The backbone becomes a swappable feature extractor; the `PreferenceHead` remains a
  small MLP/linear head on top of the backbone embedding, keeping the checkpoint and
  output contract compatible.
- Start with the backbone frozen (linear probe) as the primary V3 experiment; full or
  partial fine-tuning is a recorded sub-variant (V3a/V3b), not a separate version.
- Everything else (preprocessing from V2, MSE loss, training loop) stays unchanged.
- Pick the largest DINOv3 variant that comfortably fits the training GPU's VRAM as a
  frozen backbone; on this project's 12GB card that's `vit_huge_plus_patch16_dinov3`
  (~840M params, ~4.4GB peak at batch size 16). `vit_7b_patch16_dinov3` (~27GB) does
  not fit and is out of scope without multi-GPU/offload infrastructure.

Files: `src/picklikeme/model.py` (+ backbone weights dependency).

## V4 — Ranking loss

**Single change:** replace MSELoss with a ranking-based loss.

Options, in suggested order of experimentation:

- Pairwise logistic ranking loss (RankNet-style) — recommended first
- Margin ranking loss (`nn.MarginRankingLoss`)
- Triplet loss

Objective becomes **selected image > rejected image** rather than regressing 0/1.
Pairs are sampled selected-vs-rejected; at this stage pairs may come from anywhere in
the dataset (burst-restricted pairing is deliberately deferred to V5 so its effect is
measured separately).

Model output is still a single scalar score, so inference and ranking code are
unchanged.

Files: `src/picklikeme/train.py` (loss + pair sampling in the dataset/dataloader).

## V5 — Burst-aware learning

**Single change:** make burst membership drive training pair construction.

- Pairs/triplets are sampled preferentially **within the same burst**: my selected
  frame vs. its rejected siblings.
- Same-burst pairs get higher sampling weight than cross-burst pairs (keep some
  cross-burst pairs for global calibration).
- Requires burst IDs in the training data — the ingest pipeline
  (`src/picklikeme/ingest/`) already detects bursts; wire `burst_id` through
  `dataset.py` into pair sampling.

Rationale: my real decision is comparative within a burst. Frames from the same burst
differ only in the factors I actually judge (pose, eye, sharpness moment), making them
far more informative training signal than unrelated image pairs.

Files: `src/picklikeme/dataset.py`, `src/picklikeme/train.py` (sampler only — loss
stays as chosen in V4).

## V6 — Input resolution study

**Single change:** input resolution. Train identical configurations at:

- 384×384 (current)
- 512×512
- 640×640

For each, measure and record:

- GPU memory usage (peak)
- Training time per epoch
- All protocol metrics

Deliverable: a results table and a recommendation for the best quality/cost
compromise. The winning resolution becomes the default for later versions.

Rationale: sharpness and eye detail — likely key selection factors — may be invisible
at 384px on a downscaled full frame.

Files: `src/picklikeme/config.py`, `src/picklikeme/raw_io.py` (size parameter only).

## V7 — Data augmentation

**Single change:** add training-time augmentation, restricted to transforms that do
not alter the qualities I judge.

Allowed:

- horizontal flip
- slight brightness variation
- slight color/white-balance variation
- slight rotation (small angles, letterbox-safe)

Forbidden — anything touching the quality signals the model must learn to judge:

- blur / sharpen
- motion blur simulation
- noise injection
- focus/defocus effects
- aggressive crops

Augmentation applies to training only; evaluation always uses clean images.

Files: `src/picklikeme/raw_io.py` or a new transform step in the training dataset.

## V8 — Hard negative mining

**Single change:** prioritize difficult training pairs.

- Track per-pair loss (or score margin) during training.
- Oversample pairs the model currently gets wrong or barely right — especially
  near-duplicate same-burst pairs where a selected and rejected frame are visually
  almost identical.
- Implement as a sampling-weight update (e.g. periodically re-scoring the training
  set), not as a change to the loss function.

Rationale: near-identical burst frames encode exactly the fine distinctions
(micro-sharpness, eye state, wing position) that define my preferences; easy pairs
stop teaching the model anything.

Files: `src/picklikeme/train.py` (sampler).

## V9 — Evaluation and reporting

**Single change:** evaluation quality (no model/training change; model metrics should
be identical to V8 — this version improves measurement, not the model).

Generate proper evaluation reports containing:

- Top-1 burst accuracy
- Top-3 burst accuracy
- Precision / Recall
- Confusion matrix
- ROC AUC

Plus a per-burst comparison report: for every held-out burst, show **my selection vs.
the model's selection** side by side (thumbnails + scores), so disagreements can be
inspected visually.

Files: new `src/picklikeme/evaluate.py` + report output (HTML or Markdown per burst).

## V10 — Explainability heads

**Single change:** add auxiliary diagnostic outputs alongside the preference score.

Internal quality indicators estimated by small auxiliary heads on the shared backbone:

- sharpness confidence
- exposure quality
- motion blur confidence
- eye visibility confidence
- bird size confidence

Constraints:

- These outputs are **for analysis and debugging only**. The final model output
  remains a single preference score, and the preference training objective is
  unchanged.
- Auxiliary heads must not degrade the primary metric; if they do (e.g. via shared
  gradient interference), detach them from the backbone gradient.

Files: `src/picklikeme/model.py` (auxiliary heads), `src/picklikeme/evaluate.py`
(include indicators in reports).

---

## Version comparison summary

| Version | Component changed | What stays fixed |
|---------|-------------------|------------------|
| V1 | — (baseline) | everything |
| V2 | preprocessing (letterbox) | model, loss, data, resolution |
| V3 | backbone (pretrained) | preprocessing, loss, data, resolution |
| V4 | loss (ranking) | preprocessing, backbone, data, resolution |
| V5 | pair sampling (burst-aware) | preprocessing, backbone, loss, resolution |
| V6 | input resolution | preprocessing, backbone, loss, sampling |
| V7 | augmentation | everything else |
| V8 | hard negative mining | everything else |
| V9 | evaluation/reporting | model and training untouched |
| V10 | explainability heads | preference objective and output contract |

Each version builds on the accepted configuration of the previous one. If a version
does not improve the primary metric, record the result, keep the previous
configuration, and move on — a documented negative result is still progress.
