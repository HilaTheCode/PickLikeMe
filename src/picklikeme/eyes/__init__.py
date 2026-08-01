"""Animal eye detection - see `detector.py` for the boundary every caller uses.

Only the lightweight boundary is re-exported here. The concrete detector is
reached through `build_eye_detector`, which imports it (and torch/timm) lazily.
"""

from .detector import EyeDetection, EyeDetector, EyeKeypoint, build_eye_detector

__all__ = ["EyeDetection", "EyeDetector", "EyeKeypoint", "build_eye_detector"]
