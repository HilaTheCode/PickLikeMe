import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.ingest.metadata import ExifToolNotFoundError, ensure_exiftool_available
from picklikeme.platform import collect_environment_status


class ExifToolResolutionTests(unittest.TestCase):
    def test_macos_error_message_mentions_homebrew(self):
        with mock.patch("picklikeme.ingest.metadata._resolve_exiftool_path", return_value=None), mock.patch(
            "sys.platform", "darwin"
        ):
            with self.assertRaises(ExifToolNotFoundError) as ctx:
                ensure_exiftool_available("exiftool")

        self.assertIn("brew install exiftool", str(ctx.exception))

    def test_environment_report_lists_exiftool_status(self):
        report = collect_environment_status()
        statuses = {item["name"]: item for item in report}

        self.assertIn("ExifTool", statuses)
        self.assertIn("Python", statuses)
        self.assertIn("PyTorch", statuses)
        self.assertIn("RawPy", statuses)
