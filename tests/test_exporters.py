import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import picklikeme.exporters as exporters_module
from picklikeme.bird_crop import NormalizedCrop
from picklikeme.exporters import LightroomExporter, render_lightroom_xmp

CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
CROP = NormalizedCrop(left=0.069099, top=0.051437, right=0.826688, bottom=0.809027)


class RenderXmpTests(unittest.TestCase):
    def _description(self, xml: str):
        # Parse just the xmpmeta element (skip the xpacket processing instructions).
        start = xml.index("<x:xmpmeta")
        end = xml.index("</x:xmpmeta>") + len("</x:xmpmeta>")
        root = ET.fromstring(xml[start:end])
        return root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")

    def test_crop_fields_present_and_correct(self):
        desc = self._description(render_lightroom_xmp(CROP, "DSC08239.ARW"))
        self.assertEqual(desc.get(f"{{{CRS}}}HasCrop"), "True")
        self.assertEqual(desc.get(f"{{{CRS}}}AlreadyApplied"), "False")
        self.assertEqual(desc.get(f"{{{CRS}}}CropAngle"), "0")
        self.assertEqual(desc.get(f"{{{CRS}}}CropConstrainToWarp"), "0")
        self.assertEqual(desc.get(f"{{{CRS}}}CropConstrainToUnitSquare"), "1")
        self.assertEqual(desc.get(f"{{{CRS}}}CropLeft"), "0.069099")
        self.assertEqual(desc.get(f"{{{CRS}}}CropRight"), "0.826688")
        self.assertEqual(desc.get(f"{{{CRS}}}RawFileName"), "DSC08239.ARW")

    def test_crop_only_no_develop_settings(self):
        xml = render_lightroom_xmp(CROP, "x.ARW")
        for leaked in ("Exposure2012", "Contrast2012", "ToneCurve", "crs:Look", "Vibrance"):
            self.assertNotIn(leaked, xml)


class SidecarTests(unittest.TestCase):
    def test_writes_sidecar_next_to_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "DSC1234.NEF"
            raw.write_bytes(b"raw")
            result = LightroomExporter().export(raw, CROP, overwrite=False)
            self.assertEqual(result.action, "written")
            self.assertEqual(result.output_path, raw.with_suffix(".xmp"))
            self.assertTrue(result.output_path.exists())

    def test_existing_sidecar_not_overwritten_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "DSC1234.ARW"
            raw.write_bytes(b"raw")
            sidecar = raw.with_suffix(".xmp")
            sidecar.write_text("ORIGINAL", encoding="utf-8")

            result = LightroomExporter().export(raw, CROP, overwrite=False)
            self.assertEqual(result.action, "skipped_exists")
            self.assertEqual(sidecar.read_text(encoding="utf-8"), "ORIGINAL")

            result2 = LightroomExporter().export(raw, CROP, overwrite=True)
            self.assertEqual(result2.action, "written")
            self.assertIn("HasCrop", sidecar.read_text(encoding="utf-8"))


class DngEmbedTests(unittest.TestCase):
    def test_dng_routes_to_exiftool_embed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dng = Path(tmp) / "DSC1234.DNG"
            dng.write_bytes(b"dng")
            calls = []

            def fake_run(cmd, capture_output, text):
                calls.append(cmd)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(exporters_module.subprocess, "run", side_effect=fake_run):
                result = LightroomExporter(exiftool_path="exiftool").export(dng, CROP, overwrite=True)

            self.assertEqual(result.action, "embedded")
            self.assertEqual(result.output_path, dng)
            # overwrite=True skips the HasCrop probe, so exactly one exiftool call (the write)
            self.assertEqual(len(calls), 1)
            embed_cmd = calls[0]
            self.assertIn("-XMP-crs:HasCrop=True", embed_cmd)
            self.assertIn("-XMP-crs:CropLeft=0.069099", embed_cmd)
            self.assertIn("-overwrite_original", embed_cmd)
            # DNG gets embedded, not a sidecar
            self.assertFalse(dng.with_suffix(".xmp").exists())

    def test_existing_dng_crop_skipped_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            dng = Path(tmp) / "DSC1234.DNG"
            dng.write_bytes(b"dng")

            def fake_run(cmd, capture_output, text):
                # The -s3 HasCrop probe reports an existing crop.
                if "-s3" in cmd:
                    return mock.Mock(returncode=0, stdout="True\n", stderr="")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(exporters_module.subprocess, "run", side_effect=fake_run):
                result = LightroomExporter().export(dng, CROP, overwrite=False)
            self.assertEqual(result.action, "skipped_exists")


if __name__ == "__main__":
    unittest.main()
