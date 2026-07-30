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

from picklikeme.analyzer.contactsheets import export_jpeg_bytes, read_capture_timestamp


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


class ReadCaptureTimestampTests(unittest.TestCase):
    """The review app's sort-by-capture-date feature - read as cheaply as
    the format allows: PIL's own EXIF for standard images, rawpy's parsed
    header (no demosaic) for RAW."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_exif_datetimeoriginal_from_a_standard_image(self):
        source = self.root / "photo.jpg"
        image = Image.new("RGB", (4, 4), color="red")
        exif = image.getexif()
        exif[36867] = "2024:06:15 10:30:00"  # DateTimeOriginal
        image.save(source, format="JPEG", exif=exif)

        self.assertEqual(read_capture_timestamp(str(source)), "2024-06-15T10:30:00")

    def test_falls_back_to_the_plain_datetime_tag(self):
        source = self.root / "photo.jpg"
        image = Image.new("RGB", (4, 4), color="red")
        exif = image.getexif()
        exif[306] = "2023:01:02 03:04:05"  # DateTime, no DateTimeOriginal present
        image.save(source, format="JPEG", exif=exif)

        self.assertEqual(read_capture_timestamp(str(source)), "2023-01-02T03:04:05")

    def test_a_standard_image_with_no_exif_has_no_capture_time(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (4, 4), color="red").save(source, format="JPEG")

        self.assertIsNone(read_capture_timestamp(str(source)))

    def test_a_missing_file_has_no_capture_time(self):
        self.assertIsNone(read_capture_timestamp(str(self.root / "gone.jpg")))

    def test_reads_a_raws_own_timestamp_via_rawpy_with_no_demosaic(self):
        """rawpy is mocked because there is no RAW fixture in this repo -
        the file only needs to exist and carry a RAW-like suffix so this
        takes the RAW branch, same convention as ExportJpegBytesTests'
        mocked-rawpy test above."""
        from datetime import datetime

        source = self.root / "photo.nef"
        source.write_bytes(b"not a real NEF - rawpy.imread is mocked below")
        moment = datetime(2024, 6, 15, 10, 30, 0)

        fake_raw = mock.MagicMock()
        fake_raw.__enter__.return_value = fake_raw
        fake_raw.other = mock.Mock(timestamp=moment.timestamp())

        with mock.patch("rawpy.imread", return_value=fake_raw):
            result = read_capture_timestamp(str(source))

        self.assertEqual(result, moment.isoformat(timespec="seconds"))

    def test_reads_a_raws_timestamp_when_rawpy_returns_a_datetime_directly(self):
        """Regression test: rawpy's docs describe other.timestamp as a Unix
        epoch, but some rawpy/LibRaw versions return a datetime.datetime
        instead - datetime.fromtimestamp() then raises TypeError. Must
        produce the same ISO string either way, without assuming either
        library version."""
        from datetime import datetime

        source = self.root / "photo.nef"
        source.write_bytes(b"not a real NEF - rawpy.imread is mocked below")
        moment = datetime(2024, 6, 15, 10, 30, 0)

        fake_raw = mock.MagicMock()
        fake_raw.__enter__.return_value = fake_raw
        fake_raw.other = mock.Mock(timestamp=moment)  # a datetime, not a float

        with mock.patch("rawpy.imread", return_value=fake_raw):
            result = read_capture_timestamp(str(source))

        self.assertEqual(result, moment.isoformat(timespec="seconds"))

    def test_a_raw_with_no_timestamp_has_no_capture_time(self):
        source = self.root / "photo.nef"
        source.write_bytes(b"not a real NEF")

        fake_raw = mock.MagicMock()
        fake_raw.__enter__.return_value = fake_raw
        fake_raw.other = mock.Mock(timestamp=0)  # LibRaw's own "unknown" convention

        with mock.patch("rawpy.imread", return_value=fake_raw):
            self.assertIsNone(read_capture_timestamp(str(source)))


if __name__ == "__main__":
    unittest.main()
