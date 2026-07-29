"""export_jpeg_bytes: the extraction "Save as JPEG" (review/server.py's
/save-jpeg) is built on. The RAW/embedded-thumbnail branch is exercised via a
mocked rawpy - there is no RAW fixture in this repo - everything else runs
against real files, the same way the rest of contactsheets.py is tested
indirectly through the report/server tests.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.contactsheets import export_jpeg_bytes


class ExportJpegBytesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_existing_jpeg_is_returned_byte_for_byte(self):
        """No decode, no re-encode: the fastest possible case, and the most
        common one (this project's own review fixtures are .jpg)."""
        source = self.root / "photo.jpg"
        Image.fromarray(np.full((20, 30, 3), 128, dtype=np.uint8)).save(source, quality=90)
        original = source.read_bytes()

        self.assertEqual(export_jpeg_bytes(str(source)), original)

    def test_a_non_jpeg_standard_image_is_converted(self):
        """PNG has no embedded-thumbnail shortcut to take - it goes through
        the same decode load_source_image already does, then a JPEG encode."""
        source = self.root / "photo.png"
        Image.fromarray(np.full((20, 30, 3), 64, dtype=np.uint8)).save(source)

        data = export_jpeg_bytes(str(source))

        decoded = Image.open(__import__("io").BytesIO(data))
        self.assertEqual(decoded.format, "JPEG")
        self.assertEqual(decoded.size, (30, 20))

    def test_a_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            export_jpeg_bytes(str(self.root / "nope.jpg"))

    def test_a_raws_embedded_jpeg_thumbnail_is_returned_untouched(self):
        """The fast path this function exists for: the camera's own bytes,
        not a decode-then-recompress of them. rawpy is mocked because there
        is no RAW fixture in this repo; the file only needs to exist and
        carry a RAW-like suffix so export_jpeg_bytes takes this branch."""
        import rawpy

        source = self.root / "photo.nef"
        source.write_bytes(b"not a real NEF - rawpy.imread is mocked below")
        camera_jpeg = b"\xff\xd8\xff\xe0 pretend this is the camera's own jpeg \xff\xd9"

        fake_thumb = mock.Mock(format=rawpy.ThumbFormat.JPEG, data=camera_jpeg)
        fake_raw = mock.MagicMock()
        fake_raw.__enter__.return_value = fake_raw
        fake_raw.extract_thumb.return_value = fake_thumb

        with mock.patch("rawpy.imread", return_value=fake_raw):
            result = export_jpeg_bytes(str(source))

        self.assertEqual(result, camera_jpeg, "must be the thumbnail's own bytes, not a re-encode")


if __name__ == "__main__":
    unittest.main()
