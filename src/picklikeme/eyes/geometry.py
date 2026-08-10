"""Model-agnostic geometric primitives for the eye-detection fusion layer.

Nothing here knows about EyePose-v0, SuperAnimal-Bird, or any specific
landmark schema - it operates on plain (x, y) points and a "head frame"
built from whichever two head landmarks a caller has available. That is
deliberate: `fusion.py` is meant to stay extensible to a third model with a
different landmark set without this module changing.

Head-relative coordinates
--------------------------
A bird's head rotates between frames (and, in a Burst, the camera itself
often reframes - see `burst_consistency.py`'s own module docstring). Two
raw pixel coordinates are therefore not comparable across frames, and even
within one frame a naive "how far apart are these two points in pixels"
number means nothing without knowing how big the head itself was in that
crop.

`HeadFrame` fixes both problems at once: it is a local 2D coordinate system
anchored to two reliably-detected head landmarks (EyePose-v0's
beak/head_top, SuperAnimal-Bird's crown/bill, or any future model's
equivalent pair) - not just a scale factor. Its origin is one landmark, its
u-axis points at the other (so `scale`, the distance between them, doubles
as the "how big is this head" reference every existing per-model accept/
reject gate already uses), and its v-axis is the perpendicular. Projecting
a point into (u, v) expresses it as "how far along the head axis, how far
off to the side, both in units of head size" - a description that rotates
and scales WITH the head, unlike a plain image-plane bounding-box
normalisation (`(x - box_left) / box_width`), which does not: a bird that
merely turned its head between Burst frames would register as a large
apparent position change under a bounding-box normalisation even though the
eye never moved relative to the head at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Floor on a head-scale reference distance, exactly like eyepose_v0's
# MIN_HEAD_SCALE_PX and superanimal_bird's own - guards the pathological
# case of the two reference landmarks coinciding, never actually reached on
# a real detection.
MIN_AXIS_SCALE_PX = 3.0


@dataclass(frozen=True)
class HeadFrame:
    """A local, head-anchored 2D coordinate system - see the module
    docstring. Build one with `build_head_frame`; use `project` to express
    any point in it."""

    origin_x: float
    origin_y: float
    axis_x: float  # unit vector: u-axis direction (origin -> axis landmark)
    axis_y: float
    scale: float  # origin<->axis-landmark distance, floored at MIN_AXIS_SCALE_PX

    def project(self, x: float, y: float) -> tuple[float, float]:
        """`(u, v)`: position relative to `origin`, in units of `scale`,
        resolved along the head axis (u) and across it (v). A point AT the
        axis landmark projects to `(1.0, 0.0)`; the origin itself to
        `(0.0, 0.0)`."""
        dx, dy = x - self.origin_x, y - self.origin_y
        u = (dx * self.axis_x + dy * self.axis_y) / self.scale
        perp_x, perp_y = -self.axis_y, self.axis_x
        v = (dx * perp_x + dy * perp_y) / self.scale
        return u, v


def build_head_frame(
    origin: tuple[float, float] | None, axis_point: tuple[float, float] | None
) -> HeadFrame | None:
    """A `HeadFrame` anchored at `origin` with its u-axis pointing at
    `axis_point` - or `None` if either landmark is missing, or the two
    coincide closely enough that no direction can be derived from them."""
    if origin is None or axis_point is None:
        return None
    dx, dy = axis_point[0] - origin[0], axis_point[1] - origin[1]
    raw_scale = math.hypot(dx, dy)
    if raw_scale <= 1e-6:
        return None
    scale = max(MIN_AXIS_SCALE_PX, raw_scale)
    return HeadFrame(origin_x=origin[0], origin_y=origin[1], axis_x=dx / raw_scale, axis_y=dy / raw_scale, scale=scale)


def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Shortest distance from `(px, py)` to the line SEGMENT a<->b (not the
    infinite line) - the standard clamped-projection formula. An independent
    copy of the same arithmetic `eyepose_v0._point_to_segment_distance`
    already uses privately for its own accept/reject gate: kept separate
    (not imported from there) so this module has no dependency on any one
    backend's internals, and a backend's own gate can change shape without
    this shared utility changing under it.
    """
    abx, aby = bx - ax, by - ay
    length_sq = abx * abx + aby * aby
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def normalized_distance(point_a: tuple[float, float], point_b: tuple[float, float], head_scale: float) -> float:
    """Distance between two points, in units of `head_scale` - the shared
    "how far apart, relative to how big the head is" measure `fusion.py`
    uses both for cross-model agreement and (via `burst_consistency.py`) for
    Burst-level consistency. `head_scale` is floored the same way
    `build_head_frame`'s own `scale` is, so a degenerate reference cannot
    turn this into a division blow-up."""
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]) / max(head_scale, MIN_AXIS_SCALE_PX)


def point_in_bounds(point: tuple[float, float], width: float, height: float, *, margin_frac: float = 0.15) -> bool:
    """Is `point` inside the crop, generously - allowing a small overshoot
    (`margin_frac` of the crop's own shorter side) rather than a hard
    `0 <= x < width` test, because a keypoint model regresses a coordinate
    rather than a clamped pixel index and can legitimately land a few
    pixels beyond an edge on an eye that is genuinely near the crop's own
    border. A point far outside the crop (well past the margin) is not a
    borderline eye - it is a landmark that latched onto something the crop
    does not even contain, which is exactly the kind of geometric
    implausibility this module exists to catch."""
    if width <= 0 or height <= 0:
        return False
    margin = margin_frac * min(width, height)
    x, y = point
    return (-margin <= x <= width + margin) and (-margin <= y <= height + margin)


def eye_pair_disagreement(
    left: tuple[float, float] | None, right: tuple[float, float] | None, head_scale: float
) -> float | None:
    """How far a single model's own left/right eye-channel predictions
    diverge, in head-scale units - `None` if either channel is missing.

    This is the same left/right agreement signal
    `superanimal_bird.SuperAnimalBirdEyeDetector.detect` already computes
    for itself (see that module's "Confidence is not enough" investigation)
    generalised into a model-agnostic utility, so `fusion.py` can apply the
    identical check to ANY sub-detector that exposes both channels -
    including EyePose-v0, which does not run this check internally today.
    Reusing SuperAnimal-Bird's own validated arithmetic here (not importing
    it - see `point_to_segment_distance`'s own docstring for why) rather
    than inventing a second formula.
    """
    if left is None or right is None:
        return None
    return normalized_distance(left, right, head_scale)
