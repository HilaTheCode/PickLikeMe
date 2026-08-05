# Eye detector evaluation sample

The 30 crops (`00.jpg`-`29.jpg`) and `records.json` in this directory are the fixed
adjudication sample referenced in `src/picklikeme/eyes/superanimal_bird.py`'s module
docstring, which is the canonical, detailed writeup - this file is just an index.

## What it is

30 subject crops drawn with a fixed random seed from this project's own crop cache,
stratified 15/15 by the photographer's own Selected/Rejected verdict, birds only, then
each one adjudicated by eye against a 6x zoom of the eye box `SuperAnimalBirdEyeDetector`
predicted for it.

`records.json` has one entry per crop: its index (matching `NN.jpg`), the cached crop
path and original source RAW path at the time the sample was drawn, which stratum it
came from (`SELECTED`/`rejected`), and the detector's own `confidence`/`left`/`right`
scores for that crop.

## Result

|                                       | count |
| ------------------------------------- | ----- |
| box lands on a real eye               | 14/30 |
| box on nape, wing, or background      | 7/30  |
| head too small/blurred/dark to judge  | 8/30  |
| correctly filtered (no eye visible)   | 1/30  |

Every correct detection scored confidence >= 0.89; six of the seven wrong ones scored
below 0.80. That separation is why `DEFAULT_MIN_CONFIDENCE = 0.80` in
`superanimal_bird.py`, and the left/right eye-channel disagreement check
(`DEFAULT_MAX_EYE_DISAGREEMENT`) was validated against this same sample - see the
module docstring for the full methodology, the rejected alternative hypotheses, and
known failure modes.
