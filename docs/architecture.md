# Pick Like Me architecture

> **Scope note:** everything below is about the *learned* ranking strategy ("AI Model"). PeakPic also ships a second, independent, non-learned strategy - **Classic Vision Ranking** (`src/picklikeme/ranking/classic.py`) - a deterministic pipeline of subject/eye detection plus three sharpness/size metrics, with no training and no checkpoint. Classic Vision is itself a framework of interchangeable eye-localisation backends (currently EyePose-v0 and SuperAnimal-Bird, each its own selectable strategy with its own coexisting results) rather than one fixed algorithm - see `src/picklikeme/eyes/detector.py`'s module docstring for that backend boundary. It exists alongside the AI Model rather than instead of it: see README.md's "Analysis modules (AI Model, Classic Vision)" section for what it does and why. The "Classical handcrafted feature engineering" rejection below is about the *main*, learned architecture only - it does not apply to Classic Vision, which is deliberately hand-crafted and deterministic by design (a debuggable baseline/cross-check, not a competitor to the learned model).

## Problem framing

This project is not an objective image-quality ranking problem. It is a personal preference modeling problem: given a RAW wildlife image or burst, predict whether I would keep it.

The central modeling target is:

$$
P(y=1 \mid x, \text{context})
$$

where $y=1$ means "keep", $x$ is the input image or burst representation, and context may include capture timestamp, burst membership, camera metadata, and prior editing history.

## Key design decision: use a learned visual model, not hand-crafted scoring

A hand-crafted system based on sharpness, exposure, or composition heuristics is unlikely to capture the real decision boundary. My own decisions are shaped by a mixture of:

- subject size and placement
- eye visibility
- pose and posture
- background simplicity
- motion, action, and narrative
- visual balance and impact
- burst-level comparison effects

These factors interact in ways that are hard to codify explicitly. A learned model is more appropriate because it can discover latent, non-linear combinations of those cues directly from historical keep/reject labels.

## Recommended architecture

### Primary recommendation: burst-aware vision transformer with a pairwise or listwise ranking objective

The best starting point is a vision transformer backbone such as ViT or Swin Transformer operating on a rendered RAW preview, with a burst-aware head that aggregates multiple frames from the same burst.

Why this is the most suitable starting point:

1. Strong visual representation power
   - Transformers are effective at modeling global context, spatial relationships, and composition.
   - They are especially good when the important signal is a combination of subject presence, pose, and surrounding context.

2. Burst awareness is central
   - A burst is not just a set of independent images; the decision is often comparative.
   - A model that can process multiple frames jointly can learn relative selection signals better than a per-image classifier.

3. The problem is fundamentally ranking-like
   - The output does not need to be a binary label only; it needs a ranked list.
   - A ranking objective is more faithful than pure classification because it teaches the model which images are more likely to be preferred over others.

4. The project is personal and data-rich
   - With tens of thousands of historical decisions, a modern vision backbone is justified.
   - The model can improve continuously as new corrections arrive.

## Alternative approaches and why they are less ideal

### 1. CNN-based classifier

A ResNet or EfficientNet classifier is a strong baseline and much easier to train and deploy.

Advantages:

- Simpler and more interpretable
- Lower compute cost
- Proven robustness for image classification tasks

Disadvantages:

- Weaker at modeling long-range composition relationships than transformers
- Less natural for burst-level reasoning without extra architectural complexity
- Less flexible for future extension into pairwise ranking and multi-image comparison

Recommendation:

Use a CNN only as a baseline or early-stage benchmark. It is a good engineering baseline, but not the best long-term architecture for this problem.

### 2. Siamese or triplet ranking network

This approach learns embeddings such that preferred images are closer together or ranked according to similarity to a reference concept.

Advantages:

- Natural for comparing images directly
- Useful when the target is relative preference rather than absolute score

Disadvantages:

- Usually requires carefully curated positive/negative pairs or triplets
- More brittle when labels are noisy or inconsistent
- Less direct for learning from historical keep/reject decisions if we want a calibrated probability estimate

Recommendation:

Useful as a later refinement, especially if we want to model pairwise relative preference more explicitly. It is less suitable as the first main architecture because it demands more careful supervision design.

### 3. Classical handcrafted feature engineering + logistic regression

This would use manually extracted features such as sharpness, exposure, face/eye detection, or saliency statistics.

Advantages:

- Very interpretable
- Cheap to train
- Easy to debug

Disadvantages:

- Poor fit for the actual problem statement
- Likely to miss the hidden, high-level cues that govern personal selection
- Not aligned with the stated goal of learning from historical decisions rather than designing scoring heuristics

Recommendation:

Reject for the main architecture. This is the wrong modeling philosophy for this project.

## Recommended modeling objective

### Best practical choice: two-stage training

1. Stage 1: image-level binary classification
   - Train a model to predict $P(\text{keep} \mid \text{image})$
   - This provides a strong initial representation and allows fast experimentation

2. Stage 2: burst-aware ranking refinement
   - Extend the model to process a burst as a set or sequence of images
   - Train with a ranking loss such as pairwise logistic ranking loss or listwise softmax ranking loss

This staged approach is preferable because:

- it gives a stable initial training signal
- it makes debugging easier
- it allows the training pipeline to start before fully solving burst modeling
- it supports incremental development

## Recommended input representation

### RAW ingestion

The pipeline should read RAW files directly using rawpy or a similar library.

The model should preferably use:

- a demosaiced and tone-mapped preview for the vision backbone
- optionally a small set of metadata features such as ISO, shutter speed, aperture, focal length, timestamp, and burst index

JPEG previews should only be used as a temporary convenience for early experiments, not as the core input path.

## Why not start with a pure classifier on single images?

A single-image classifier is simpler, but it misses the core structure of the problem. Wildlife selection is often comparative within a burst, and a single frame is sometimes only meaningful when seen alongside other frames taken in the same moment.

For that reason, the first production architecture should be burst-aware, even if initial experiments start at the image level.

## Suggested architecture stack

### Phase 1: baseline

- Backbone: ResNet-50 or EfficientNet-B0
- Input: single RAW-derived RGB preview
- Objective: binary cross-entropy for keep/reject
- Output: calibrated probability

This phase is good for establishing reliable data pipelines and validating that the labels carry signal.

### Phase 2: improved visual backbone

- Backbone: Swin Transformer Tiny or ViT-Small
- Input: single RAW-derived RGB preview
- Objective: binary cross-entropy with class-weighting and calibration

This phase should improve representation quality and make the model more sensitive to subtle visual factors.

### Phase 3: burst-aware production model

- Backbone: Swin Transformer or ViT-Small as frame encoder
- Burst aggregation: transformer or attention pooling over frames within a burst
- Objective: pairwise or listwise ranking loss over burst members
- Output: ranked keep probabilities per frame

This is the architecture I recommend for the long-term project.

## Why this is scientifically sound

The project is best framed as a preference-learning problem from historical decisions. That is a supervised learning task with noisy labels and strong contextual structure. A burst-aware ranking model aligns closely with that formulation.

It is also a good engineering choice because it supports incremental development:

- start with simple single-image training
- validate the data pipeline
- add burst structure
- add ranking objectives
- continue improving as the dataset grows

## Recommended implementation plan

1. Build a robust RAW ingestion and preprocessing pipeline.
2. Create a dataset abstraction that can represent both single images and bursts.
3. Train a baseline model on single images to confirm the signal.
4. Add a burst aggregation module and train with ranking losses.
5. Evaluate with held-out historical decisions and calibration metrics.
6. Iterate using new editing corrections as additional training data.

## Final recommendation

The most appropriate modern architecture is a burst-aware vision transformer trained with a ranking objective, starting from a simpler single-image baseline.

This choice is justified because the task is not generic image quality assessment. It is personal preference learning, and that makes burst structure, global composition, and relative decision context essential.
