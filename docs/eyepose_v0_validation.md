# EyePose-v0 validation notes

Companion to `eyes/eyepose_v0.py`'s module docstring. Two things are
recorded here: the coordinate-transform verification that had to hold
*before* the model's own predictions were worth judging at all, and a real
(if small) side-by-side comparison against SuperAnimal-Bird.

## 1. Coordinate transform - verified before trusting any prediction

Per the reported task's own instruction: "Do not assume the model is wrong
until coordinate mapping has been verified." Two independent checks, both
against the real `eye_pose_v0.pt`/`eye_pose_v0.onnx`, before writing
`EyePoseV0EyeDetector`:

1. **ONNX ≡ PyTorch.** The exact same static 640x640 letterboxed input
   tensor (`Ultralytics`' own `LetterBox(auto=False)`, reproduced in plain
   numpy/cv2 as `_letterbox_forward`) was run through (a) the exported ONNX
   graph via `onnxruntime` and (b) the original checkpoint's raw
   `model.model(tensor)` forward pass in PyTorch. Both produced the
   **identical winning anchor index, identical detection confidence
   (0.8422279), and identical all 6 keypoints to 3 decimal places.** This
   proves the ONNX export - the thing this project actually ships and
   runs - computes exactly what the published checkpoint computes; nothing
   was lost or altered in conversion.
2. **End-to-end visual verification on real photos.** `eyes.inspect_eyepose`
   (`python -m picklikeme.eyes.inspect_eyepose --input-folder ...`) runs the
   complete pipeline - subject detection, crop, letterbox, model, decode,
   inverse transform, projection back to the original frame - and saves all
   six stages as separate images/text per the reported task's own checklist.
   Run against three real photos (see below), the eye box and all six
   landmarks landed exactly where visual inspection says they should, in
   **both** the crop-space overlay and the full-original-frame projected
   overlay - confirming the forward (crop -> 640-space) and inverse
   (640-space -> crop -> full frame) transforms are both correct, not just
   internally self-consistent.

Both checks are also pinned as regression tests
(`tests/test_eyepose_v0.py`'s `TestLetterboxForward`/`TestLetterboxInverse`,
including the exact hand-verified numbers from check 1 above).

## 2. EyePose-v0 vs SuperAnimal-Bird - a real, small-sample comparison

**Sample size: 3 real photographs**, sourced from Wikimedia Commons (this
sandboxed environment has no access to the project's own RAW archive) -
chosen to cover a clear portrait, a dynamic wing-spread action shot (the
same category as SuperAnimal-Bird's own documented "soaring bird" false
positive), and a deliberately hard, near-top-down foreshortened head angle.
This is **not** a substitute for SuperAnimal-Bird's own 30-image
hand-adjudicated study (`eyes/superanimal_bird.py`'s module docstring) -
that took a real photographer's own archive and verdicts, neither of which
this sandboxed environment has access to. Treat this as directional
evidence pending that same rigor on PeakPic's actual archive, not a final
verdict - see "What this does NOT establish" below.

| Image | Pose | EyePose-v0 | SuperAnimal-Bird |
| --- | --- | --- | --- |
| Indian White-eye portrait | head-on, eye clearly visible | Correct, confident (left_eye 1.00), **accepted** | Box lands on cheek feathers, off the eye; confidence 1.00 but **rejected** by its own left/right disagreement gate |
| White Stork in flight | dynamic, wings spread, profile, eye visible | Correct, confident (1.00), accepted | Correct, confident (0.93), accepted |
| Black Skimmer skimming | extreme head-down foreshortening, eye not clearly resolvable | Plausible head-region placement, close to beak/head_top axis; confidence 0.95 | Plausible head-region placement, slightly further from the beak; confidence 0.80, accepted |

Artifacts backing this table (not committed - `cache/` is gitignored, and a
`.pt`/`.onnx` download plus two extra photos is not something to add to a
diff): `eyes.inspect_eyepose`'s own six-artifact output per image, plus a
direct-comparison overlay for SuperAnimal-Bird on the identical crops.
Reproducible with:

```bash
python -m picklikeme.eyes.inspect_eyepose --input-folder <folder-of-real-photos>
```

### Reading the results

- On the one image where the two disagreed outright (the portrait),
  EyePose-v0 was correct and SuperAnimal-Bird's own gate correctly caught
  its own mistake - a wrong answer that got filtered, not a wrong answer
  that scored. This is the SuperAnimal-Bird gate design working as
  documented, not a SuperAnimal-Bird failure exactly, but it does mean that
  image would have been reported `NO_VISIBLE_EYE` under SuperAnimal-Bird
  and correctly scored under EyePose-v0.
- On the dynamic action shot, both were correct and confident - EyePose-v0
  is not "better on hard poses" in general; this one just wasn't hard for
  either.
- On the genuinely hard foreshortened shot, **neither** model hallucinated
  onto an unrelated body part (SuperAnimal-Bird's own documented failure
  mode, e.g. the soaring-vulture case landing on wing roots) - both stayed
  in the general head region. EyePose-v0's `min_confidence`/
  `max_head_distance_ratio` gate (see `eyes/eyepose_v0.py`'s "Accept/reject
  gate") is a **starting default, not yet empirically tuned** the way
  SuperAnimal-Bird's 0.80/0.50 were - this image is a concrete example of
  exactly the "confidently in the right region, not exactly on a resolvable
  iris" case that tuning should target.

### What this does NOT establish

- **Not a verdict on which backend is "better."** Three photos, a different
  distribution (wild internet photos vs PeakPic's own archive), and no
  independent human adjudication (unlike SuperAnimal-Bird's 30-image study)
  is not enough evidence for that claim, and none is made here.
- **No false-positive/false-negative rate.** That needs the same
  adjudicated-sample methodology `eyes/superanimal_bird.py` used, run
  against PeakPic's own Selected/Rejected archive - infeasible in this
  sandboxed session (no access to that archive), and squarely the kind of
  follow-up `eyes.inspect_eyepose` exists to make easy to run later.
- **`min_confidence`/`max_head_distance_ratio` are not tuned.** They are
  reasonable starting points (see `eyes/eyepose_v0.py`), not validated
  thresholds. Tune them the same way SuperAnimal-Bird's were: run
  `eyes.inspect_eyepose` (or Classic Vision's own debug mode - see
  `ranking.classic`'s `debug_dir`) against a real sample, adjudicate by eye,
  sweep the threshold.

### Recommendation

EyePose-v0 is registered as the recommended default for new Classic Vision
analyses (see README.md) on the strength of: correct, verified coordinate
math; the model's own strong published validation metrics (mAP50 0.994 on
CUB-200-2011); an MIT-licensed, ONNX-only runtime with no AGPL exposure; and
this session's own small sample showing at-least-comparable, and in one case
better, real-world localisation than SuperAnimal-Bird. SuperAnimal-Bird
remains fully available, selectable, and unmodified - both backends'
results coexist per folder for exactly this kind of ongoing comparison.
