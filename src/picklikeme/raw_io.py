from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .bird_crop import crop_cache_path

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
        crop_cache_dir: str | None = None,
    ):
        if resize_mode not in self.RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {sorted(self.RESIZE_MODES)}, got {resize_mode!r}")
        self.raw_root = Path(raw_root)
        self.output_size = output_size
        self.resize_mode = resize_mode
        # When set, load_image reads pre-computed bird crops from this cache
        # (built by picklikeme.preprocess) instead of the full frame.
        self.crop_cache_dir = Path(crop_cache_dir) if crop_cache_dir is not None else None
        self._warned_missing_crop = False

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

    def _decode_full_frame(self, image_path: str) -> np.ndarray:
        """Decode the source image to a full-resolution RGB uint8 array (no
        resize). Used both by the normal load path and by the preprocessing
        pass that detects birds on the native-resolution frame."""
        path = self._resolve_path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.suffix.lower() in self.RAW_EXTENSIONS:
            return self._read_raw_image(path)
        return self._read_standard_image(path)

    def _source_pixels(self, image_path: str) -> np.ndarray:
        """Full-resolution RGB uint8 to feed the resize step.

        With a crop cache configured, reads the pre-computed bird crop; if a
        crop is missing (cache not built for this image) it falls back to the
        full frame and warns once, rather than loading a detector inside a
        DataLoader worker.
        """
        if self.crop_cache_dir is not None:
            cache_path = crop_cache_path(self.crop_cache_dir, self._resolve_path(image_path))
            if cache_path.exists():
                return self._read_standard_image(cache_path)
            if not self._warned_missing_crop:
                print(
                    f"[RawImageLoader] crop cache enabled but missing entry for {image_path}; "
                    "falling back to full frame. Run `python -m picklikeme.preprocess` to build crops."
                )
                self._warned_missing_crop = True
        return self._decode_full_frame(image_path)

    def load_image(self, image_path: str) -> np.ndarray:
        rgb = self._source_pixels(image_path)

        if self.resize_mode == "letterbox":
            rgb = self._letterbox(rgb)
        else:
            rgb = cv2.resize(rgb, self.output_size, interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0
        return rgb

    def load_image_from_path(self, image_path: str) -> np.ndarray:
        return self.load_image(image_path)
