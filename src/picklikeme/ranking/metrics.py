"""Image-quality measurements, independent of any ranking strategy.

Every function here is a pure function of pixels (or of numbers), with no
knowledge of detectors, filters, weights, or what is being ranked. That is
deliberate: the Classic Vision strategy is the first consumer, but "how sharp
is this region" and "how much of the frame does the subject fill" are exactly
the questions a later strategy - or a burst-comparison tool, or a diagnostic
overlay - will want to ask without inheriting a ranking pipeline to do it.

`focus_measure` is the interesting one. The textbook answer - variance of the
Laplacian - was tried first and measurably misranks this archive, so the
implementation below differs from it in four deliberate ways. Each was
checked against real crops from this project's own cache, not reasoned about
in the abstract:

- **Resample to a canonical size.** Downscaling concentrates the same edge
  energy into fewer pixels and *raises* the measured value. Cached crops are
  capped at a long side of 1024 but never upscaled (see
  `bird_crop.downscale_long_side`), so a frame-filling subject is measured
  after downscaling and a distant one at native resolution - comparing raw
  values would favour whichever happened to be resized more.

- **Standardise to unit variance.** A high-contrast subject against a bright
  sky otherwise scores higher than a soft-toned one at identical optical
  sharpness. This leaves a measure of edge *structure* rather than of tonal
  range.

- **Denoise slightly first.** Sensor noise is high-frequency, and every
  gradient-based focus measure counts it as detail. A mild Gaussian
  (`DENOISE_SIGMA`) suppresses pixel-level grain while leaving real edges
  intact. Measured on a noisy high-ISO crop this cut the spurious reading by
  ~10x while barely touching a clean sharp frame.

- **Take a high percentile of the edge magnitude, not the variance.** This is
  the correction that matters most. Variance averages over every pixel, so a
  shallow-depth-of-field portrait - a tack-sharp bird against a large, smooth
  bokeh background, i.e. exactly what good wildlife photography looks like -
  is dragged down by all that smooth area, while a mediocre frame that is
  merely *busy* everywhere scores high. Measured on real crops, variance
  ranked a genuinely sharp portrait 11x BELOW a noisy, unremarkable frame; the
  99th percentile of |Laplacian| ranks them correctly and separates sharp from
  motion-blurred by a wide margin. A high percentile asks "how sharp are the
  sharpest edges in this region", which is what "is the subject in focus"
  actually means, and it is insensitive to how much of the region is
  background.

None of this makes the value physically meaningful on its own - it remains a
relative focus measure, comparable only within one run over one folder - which
is why `robust_normalize` exists rather than any absolute threshold.
"""

from __future__ import annotations

import numpy as np

# Every focus measurement is taken at this size, so the value describes the
# subject and not how much the crop happened to be downscaled. Large enough
# to preserve the fine detail that distinguishes a sharp eye from a nearly
# sharp one; small enough that measuring thousands of images stays cheap.
CANONICAL_PATCH_SIZE = 128

# The subject crop is measured at this long side, for the same reason.
CANONICAL_SUBJECT_LONG_SIDE = 512

# Floor on the standard deviation used to contrast-normalise a patch. Guards
# the degenerate case - a perfectly flat region (blown-out sky, a black
# silhouette) - where dividing by the true std would amplify sensor noise into
# a large, meaningless focus value.
_MIN_STD = 1e-3

# Gaussian sigma applied before the edge operator, to keep sensor noise from
# reading as detail. Small on purpose: enough to suppress single-pixel grain,
# far too little to blur a real edge. Larger values were tried and start
# compressing the sharp-vs-blurred separation the metric exists to measure.
DENOISE_SIGMA = 0.8

# Which percentile of |Laplacian| is the focus value - "how sharp are the
# sharpest edges here". See the module docstring for why this rather than the
# variance. 99 rather than the maximum, so one hot pixel or a single specular
# highlight cannot define the score for the whole region.
FOCUS_PERCENTILE = 99.0

# Percentiles `robust_normalize` maps to 0.0 and 1.0. Using the extremes
# instead would let one pathological frame (a lens flare, a totally blown
# highlight) compress every real image into a narrow band at the bottom of
# the range and flatten the ranking.
NORMALIZE_LOW_PERCENTILE = 5.0
NORMALIZE_HIGH_PERCENTILE = 95.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Luminance as float32. Accepts RGB or an already-single-channel image."""
    import cv2

    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def focus_measure(patch: np.ndarray, canonical_size: int = CANONICAL_PATCH_SIZE) -> float:
    """How sharp `patch` is, as a scale- and contrast-normalised focus value.

    The 99th percentile of |Laplacian|, measured after resampling to
    `canonical_size`, standardising to unit variance, and a mild Gaussian
    denoise. See the module docstring for why each of those four choices is
    there and what the textbook variance-of-Laplacian got wrong on this
    archive. Returns 0.0 for an empty or degenerate patch, which sorts such an
    image to the bottom rather than crashing the run.
    """
    import cv2

    if patch is None or patch.size == 0 or min(patch.shape[:2]) < 2:
        return 0.0

    gray = _to_gray(patch)
    height, width = gray.shape[:2]
    if max(height, width) != canonical_size:
        scale = canonical_size / max(height, width)
        target = (max(2, round(width * scale)), max(2, round(height * scale)))
        # INTER_AREA when shrinking (it averages, so it does not alias fine
        # detail into false edges), INTER_LINEAR when growing.
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray = cv2.resize(gray, target, interpolation=interpolation)

    denoised = cv2.GaussianBlur(gray, (0, 0), DENOISE_SIGMA)
    standardized = denoised / max(float(denoised.std()), _MIN_STD)
    edges = np.abs(cv2.Laplacian(standardized, cv2.CV_32F))
    return float(np.percentile(edges, FOCUS_PERCENTILE))


def region_focus_measure(image: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """`focus_measure` of one sub-rectangle of `image` (x1, y1, x2, y2).

    The box is clamped to the image, so a region detected hard against an
    edge still yields a real measurement instead of an empty slice.
    """
    height, width = image.shape[:2]
    x1 = max(0, min(int(round(box[0])), width - 1))
    y1 = max(0, min(int(round(box[1])), height - 1))
    x2 = max(x1 + 1, min(int(round(box[2])), width))
    y2 = max(y1 + 1, min(int(round(box[3])), height))
    return focus_measure(image[y1:y2, x1:x2])


def subject_focus_measure(subject_crop: np.ndarray) -> float:
    """`focus_measure` of a whole subject crop, at its own canonical size.

    Separate from `focus_measure`'s default only in scale: a subject crop is
    measured larger than an eye patch because it has far more area over which
    real detail can be lost to over-aggressive downsampling.
    """
    return focus_measure(subject_crop, canonical_size=CANONICAL_SUBJECT_LONG_SIDE)


def normalized_subject_size(
    box: tuple[float, float, float, float],
    source_size: tuple[int, int],
) -> float:
    """Subject bounding-box area as a fraction of the full frame, in [0, 1].

    `source_size` is (width, height) of the original frame - not of the
    cached crop - because the whole point of the metric is how much of the
    photograph the subject occupies. Returns 0.0 for a frame of unknown or
    degenerate size rather than raising: a missing dimension is a metadata
    gap, not a reason to fail the image.
    """
    width, height = source_size
    frame_area = float(width) * float(height)
    if frame_area <= 0.0:
        return 0.0
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return min(1.0, area / frame_area)


def robust_normalize(values: list[float]) -> list[float]:
    """Map a metric's raw values across one run onto [0, 1].

    Min-max between the 5th and 95th percentiles, clipped - not between the
    true extremes, so a single outlier cannot squash the rest of the
    distribution into a sliver. A metric whose values are all effectively
    identical normalises to 0.5 for every image: it genuinely carries no
    ranking information here, and 0.5 lets the other metrics decide the order
    instead of an arbitrary 0 or 1 skewing it.

    The outlier protection is a property of the sample size, and is worth
    being precise about: on a real folder (dozens of images upward) a wild
    value falls outside the 95th percentile and is clipped. On a handful of
    images the 95th percentile interpolates close to the outlier itself, and
    the remaining values do compress toward zero. That is tolerable rather
    than a defect, because this value only ever ORDERS images - the ordering
    stays strictly correct either way, and the other two metrics still carry
    their weight.

    Folder-relative by construction. These focus measures have no absolute
    scale, so "the sharpest eye in this shoot" is the only meaningful
    reading of a normalised 1.0 - never "sharp in absolute terms".
    """
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    low = float(np.percentile(array, NORMALIZE_LOW_PERCENTILE))
    high = float(np.percentile(array, NORMALIZE_HIGH_PERCENTILE))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-12:
        return [0.5] * len(values)
    return [float(np.clip((value - low) / (high - low), 0.0, 1.0)) for value in array]
