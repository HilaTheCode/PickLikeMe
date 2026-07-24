"""Crop exporters: translate a generic NormalizedCrop into a specific photo
editor's format.

Detection and crop computation live in bird_crop.py (the single source of
truth). Exporters only convert the resulting NormalizedCrop, so supporting a
new editor (Capture One, etc.) means adding one class here and registering it
in EXPORTERS — nothing about detection or crop geometry changes.

LightroomExporter writes a Lightroom Classic crop:
- proprietary RAW (NEF/ARW/CR3/...) -> a `.xmp` sidecar next to the image
- DNG -> embedded into the DNG's own XMP via exiftool, because Lightroom reads
  a DNG's crop from inside the file and ignores a neighbouring sidecar.

Only crop fields are written; no develop settings, tone curves, or profiles.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from xml.sax.saxutils import quoteattr

from .bird_crop import NormalizedCrop

# Match the reference Lightroom Classic XMP the crop fields came from.
LIGHTROOM_CRS_VERSION = "18.4"
LIGHTROOM_PROCESS_VERSION = "15.4"


@dataclass
class ExportResult:
    output_path: Path
    action: str  # "written" | "embedded" | "skipped_exists"


def _lightroom_crop_fields(crop: NormalizedCrop, raw_filename: str) -> dict[str, str]:
    """The crop-only crs: fields, in the exact value format Lightroom uses
    (string booleans, numeric flags, 6-decimal coordinates)."""
    return {
        "Version": LIGHTROOM_CRS_VERSION,
        "ProcessVersion": LIGHTROOM_PROCESS_VERSION,
        "HasCrop": "True",
        "CropTop": f"{crop.top:.6f}",
        "CropLeft": f"{crop.left:.6f}",
        "CropBottom": f"{crop.bottom:.6f}",
        "CropRight": f"{crop.right:.6f}",
        "CropAngle": f"{crop.angle:g}",
        "CropConstrainToWarp": "0",
        "CropConstrainToUnitSquare": "1",
        "AlreadyApplied": "False",
        "RawFileName": raw_filename,
    }


def render_lightroom_xmp(crop: NormalizedCrop, raw_filename: str) -> str:
    """A minimal, crop-only Lightroom Classic XMP sidecar document."""
    fields = _lightroom_crop_fields(crop, raw_filename)
    attrs = "\n".join(f"   crs:{key}={quoteattr(value)}" for key, value in fields.items())
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="PickLikeMe auto_crop">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"\n'
        f"{attrs}>\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>\n'
    )


class CropExporter(Protocol):
    name: str

    def export(self, image_path: Path, crop: NormalizedCrop, *, overwrite: bool) -> ExportResult: ...


class LightroomExporter:
    name = "lightroom"

    def __init__(self, exiftool_path: str = "exiftool"):
        self.exiftool_path = exiftool_path

    def export(self, image_path: Path, crop: NormalizedCrop, *, overwrite: bool) -> ExportResult:
        image_path = Path(image_path)
        if image_path.suffix.lower() == ".dng":
            return self._embed_dng(image_path, crop, overwrite)
        return self._write_sidecar(image_path, crop, overwrite)

    def _write_sidecar(self, image_path: Path, crop: NormalizedCrop, overwrite: bool) -> ExportResult:
        sidecar = image_path.with_suffix(".xmp")
        if sidecar.exists() and not overwrite:
            return ExportResult(sidecar, "skipped_exists")
        xml = render_lightroom_xmp(crop, image_path.name)
        tmp = sidecar.with_name(sidecar.name + ".tmp")
        tmp.write_text(xml, encoding="utf-8")
        tmp.replace(sidecar)
        return ExportResult(sidecar, "written")

    def _embed_dng(self, image_path: Path, crop: NormalizedCrop, overwrite: bool) -> ExportResult:
        if not overwrite and self._dng_has_crop(image_path):
            return ExportResult(image_path, "skipped_exists")
        fields = _lightroom_crop_fields(crop, image_path.name)
        cmd = (
            [self.exiftool_path]
            + [f"-XMP-crs:{key}={value}" for key, value in fields.items()]
            + ["-overwrite_original", str(image_path)]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"exiftool failed to embed crop into {image_path}: {proc.stderr.strip()}")
        return ExportResult(image_path, "embedded")

    def _dng_has_crop(self, image_path: Path) -> bool:
        proc = subprocess.run(
            [self.exiftool_path, "-s3", "-XMP-crs:HasCrop", str(image_path)],
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() != ""


# Registry of available exporters. Add new editors here (e.g. "capture_one").
EXPORTERS: dict[str, type] = {
    "lightroom": LightroomExporter,
}
