import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.train import build_arg_parser


class EpochsCliTests(unittest.TestCase):
    def test_epochs_is_required(self):
        parser = build_arg_parser()
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            parser.parse_args(["--select-root", "S", "--reject-root", "R"])
        self.assertIn("--epochs", err.getvalue())

    def test_epochs_parsed_as_int(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--select-root", "S", "--reject-root", "R", "--epochs", "15"])
        self.assertEqual(args.epochs, 15)
        self.assertIsInstance(args.epochs, int)


if __name__ == "__main__":
    unittest.main()
