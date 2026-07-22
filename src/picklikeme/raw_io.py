from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import rawpy  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    rawpy = None


class RawImageLoader:
    RAW_EXTENSIONS = {".arw", ".cr2", ".dng", ".nef", ".orf", ".raw", ".rw2"}
    RESIZE_MODES = {"letterbox", "stretch"}

    def __init__(
        self,
        raw_root: str,
        output_size: tuple[int, int] = (384, 384),
        resize_mode: str = "letterbox",
    ):
        if resize_mode not in self.RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {sorted(self.RESIZE_MODES)}, got {resize_mode!r}")
        self.raw_root = Path(raw_root)
        self.output_size = output_size
        self.resize_mode = resize_mode

    def _resolve_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.raw_root / path
        return path

    def _read_raw_image(self, path: Path) -> np.ndarray:
        if rawpy is None:
            raise RuntimeError(
                "rawpy is not available. Install rawpy and a compatible libraw backend to load RAW files."
            )

        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess()
        return rgb

    def _read_standard_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unsupported or unreadable image file: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _letterbox(self, rgb: np.ndarray) -> np.ndarray:
        """Resize preserving aspect ratio, pad the remainder with black.

        The full frame is always kept (no cropping) and never distorted (no
        stretching): subject geometry like wing shape and head proportions is
        part of what the model must judge.
        """
        target_width, target_height = self.output_size
        height, width = rgb.shape[:2]
        scale = min(target_width / width, target_height / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (new_width, new_height), interpolation=interpolation)

        canvas = np.zeros((target_height, target_width, 3), dtype=resized.dtype)
        top = (target_height - new_height) // 2
        left = (target_width - new_width) // 2
        canvas[top : top + new_height, left : left + new_width] = resized
        return canvas

    def load_image(self, image_path: str) -> np.ndarray:
        path = self._resolve_path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in self.RAW_EXTENSIONS:
            rgb = self._read_raw_image(path)
        else:
            rgb = self._read_standard_image(path)

        if self.resize_mode == "letterbox":
            rgb = self._letterbox(rgb)
        else:
            rgb = cv2.resize(rgb, self.output_size, interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0
        return rgb

    def load_image_from_path(self, image_path: str) -> np.ndarray:
        return self.load_image(image_path)
