"""Part 3: filing images into _Selected / _Rejected.

These images are the only copy, so the safety properties are tested as hard as
the happy path: nothing is overwritten, a partial failure does not abort the
rest, and a re-run is a no-op rather than a shuffle.

Two entry points share one move loop - `organize_ranked_images` (a percentage
cut of a ranking) and `organize_by_decision` (explicit sets, which is what a
reviewed shoot produces) - so the safety properties are asserted against both.
"""

import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.dataset import UnlabeledImageDataset
from picklikeme.organize import (
    DEFAULT_SELECTION_PERCENTAGE,
    ORGANIZE_DIRNAMES,
    REJECTED_DIRNAME,
    SELECTED_DIRNAME,
    InvalidSelectionPercentage,
    organize_by_decision,
    organize_ranked_images,
    selection_count,
    unique_destination,
    validate_selection_percentage,
)
from picklikeme.rank import build_arg_parser


def make_ranked(root: Path, count: int = 8) -> list[str]:
    """`count` files in a folder, returned in ranking order (best first)."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        target = root / f"IMG_{index:04d}.NEF"
        target.write_bytes(f"frame {index}".encode())
        paths.append(str(target))
    return paths


class PercentageTests(unittest.TestCase):
    def test_valid_values_are_accepted(self):
        for value in (0, 0.0, 1, 25, 25.5, 50, 99.9, 100):
            self.assertEqual(validate_selection_percentage(value), float(value))

    def test_out_of_range_values_are_rejected_with_a_useful_message(self):
        for value in (-1, -0.01, 100.01, 101, 1000):
            with self.assertRaises(InvalidSelectionPercentage) as ctx:
                validate_selection_percentage(value)
            self.assertIn("between 0 and 100", str(ctx.exception))

    def test_non_numeric_values_are_rejected(self):
        for value in ("abc", None, [25]):
            with self.assertRaises(InvalidSelectionPercentage):
                validate_selection_percentage(value)

    def test_nan_is_rejected(self):
        with self.assertRaises(InvalidSelectionPercentage):
            validate_selection_percentage(float("nan"))

    def test_selection_count_at_the_documented_percentages(self):
        self.assertEqual(selection_count(100, 0), 0)
        self.assertEqual(selection_count(100, 25), 25)
        self.assertEqual(selection_count(100, 50), 50)
        self.assertEqual(selection_count(100, 100), 100)

    def test_endpoints_are_exact_regardless_of_rounding(self):
        for total in (1, 3, 7, 999):
            self.assertEqual(selection_count(total, 0), 0)
            self.assertEqual(selection_count(total, 100), total)

    def test_rounding_is_to_nearest(self):
        self.assertEqual(selection_count(10, 25), 2)   # 2.5 -> 2
        self.assertEqual(selection_count(8, 25), 2)
        self.assertEqual(selection_count(3, 50), 2)    # 1.5 -> 2
        self.assertEqual(selection_count(0, 25), 0)


class OrganizeTests(unittest.TestCase):
    def test_top_percentage_goes_to_selected_and_the_rest_to_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranked = make_ranked(root / "shoot", 8)
            result = organize_ranked_images(ranked, root / "shoot", 25)

            self.assertEqual((result.ranked, result.selected, result.rejected), (8, 2, 6))
            self.assertEqual(result.moved, 8)
            self.assertEqual(result.errors, 0)

            selected = sorted(p.name for p in (root / "shoot" / SELECTED_DIRNAME).iterdir())
            rejected = sorted(p.name for p in (root / "shoot" / REJECTED_DIRNAME).iterdir())
            # Ranking order decides: the first two entries are the selection.
            self.assertEqual(selected, ["IMG_0000.NEF", "IMG_0001.NEF"])
            self.assertEqual(len(rejected), 6)

    def test_zero_percent_selects_nothing_but_still_files_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranked = make_ranked(root / "shoot", 6)
            result = organize_ranked_images(ranked, root / "shoot", 0)

            self.assertEqual(result.selected, 0)
            self.assertEqual(result.rejected, 6)
            self.assertFalse(any((root / "shoot" / SELECTED_DIRNAME).iterdir()))
            self.assertEqual(len(list((root / "shoot" / REJECTED_DIRNAME).iterdir())), 6)

    def test_hundred_percent_selects_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranked = make_ranked(root / "shoot", 6)
            result = organize_ranked_images(ranked, root / "shoot", 100)

            self.assertEqual(result.selected, 6)
            self.assertEqual(len(list((root / "shoot" / SELECTED_DIRNAME).iterdir())), 6)
            self.assertFalse(any((root / "shoot" / REJECTED_DIRNAME).iterdir()))

    def test_fifty_percent_splits_evenly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranked = make_ranked(root / "shoot", 10)
            result = organize_ranked_images(ranked, root / "shoot", 50)
            self.assertEqual((result.selected, result.rejected), (5, 5))

    def test_destination_folders_are_created_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranked = make_ranked(root / "shoot", 4)
            destination = root / "does" / "not" / "exist"
            self.assertFalse(destination.exists())

            organize_ranked_images(ranked, destination, 25)
            self.assertTrue((destination / SELECTED_DIRNAME).is_dir())
            self.assertTrue((destination / REJECTED_DIRNAME).is_dir())

    def test_filenames_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranked = make_ranked(root / "shoot", 4)
            organize_ranked_images(ranked, root / "shoot", 50)
            filed = {
                p.name
                for folder in ORGANIZE_DIRNAMES
                for p in (root / "shoot" / folder).iterdir()
            }
            self.assertEqual(filed, {Path(p).name for p in ranked})

    def test_a_collision_never_overwrites_the_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shoot = root / "shoot"
            ranked = make_ranked(shoot, 2)
            # An unrelated file already occupies the destination name.
            occupied = shoot / SELECTED_DIRNAME
            occupied.mkdir(parents=True)
            existing = occupied / "IMG_0000.NEF"
            existing.write_bytes(b"PRECIOUS - must survive")

            result = organize_ranked_images(ranked, shoot, 50)

            self.assertEqual(existing.read_bytes(), b"PRECIOUS - must survive")
            self.assertEqual(result.renamed, 1)
            self.assertTrue((occupied / "IMG_0000_1.NEF").exists())
            self.assertEqual((occupied / "IMG_0000_1.NEF").read_bytes(), b"frame 0")

    def test_repeated_collisions_keep_finding_free_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.NEF"
            target.write_bytes(b"x")
            (root / "a_1.NEF").write_bytes(b"x")
            (root / "a_2.NEF").write_bytes(b"x")
            self.assertEqual(unique_destination(target).name, "a_3.NEF")

    def test_rerunning_is_a_no_op_rather_than_a_shuffle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shoot = root / "shoot"
            ranked = make_ranked(shoot, 6)
            first = organize_ranked_images(ranked, shoot, 50)
            self.assertEqual(first.moved, 6)

            # Second pass over the files at their NEW locations.
            relocated = [str(path) for path in first.moves.values()]
            second = organize_ranked_images(relocated, shoot, 50)
            self.assertEqual(second.moved, 0)
            self.assertEqual(second.skipped, 6)
            self.assertEqual(second.errors, 0)

    def test_a_missing_source_is_reported_and_the_rest_still_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shoot = root / "shoot"
            ranked = make_ranked(shoot, 5)
            Path(ranked[2]).unlink()

            result = organize_ranked_images(ranked, shoot, 40)
            self.assertEqual(result.moved, 4)
            self.assertEqual(result.skipped, 1)
            self.assertTrue(any("not found" in reason for _, reason in result.failures))

    def test_an_empty_ranking_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = organize_ranked_images([], root / "shoot", 25)
            self.assertEqual((result.ranked, result.moved), (0, 0))
            self.assertFalse((root / "shoot" / SELECTED_DIRNAME).exists())

    def test_dry_run_reports_without_touching_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shoot = root / "shoot"
            ranked = make_ranked(shoot, 4)
            result = organize_ranked_images(ranked, shoot, 25, dry_run=True)

            self.assertEqual(result.moved, 4)
            for path in ranked:
                self.assertTrue(Path(path).exists(), "dry run must not move files")

    def test_invalid_percentage_is_rejected_before_any_file_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shoot = root / "shoot"
            ranked = make_ranked(shoot, 4)
            with self.assertRaises(InvalidSelectionPercentage):
                organize_ranked_images(ranked, shoot, 150)
            for path in ranked:
                self.assertTrue(Path(path).exists())

    def test_summary_reports_every_required_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranked = make_ranked(root / "shoot", 4)
            text = organize_ranked_images(ranked, root / "shoot", 25).render()
            for label in ("Images ranked:", "Selected:", "Rejected:",
                          "Moved successfully:", "Skipped:", "Errors:"):
                self.assertIn(label, text)


class OrganizeByDecisionTests(unittest.TestCase):
    """The reviewed path: explicit sets, because a manual Keep on a low-scoring
    frame means the selection is no longer a prefix of the ranking."""

    def test_explicit_sets_are_filed_regardless_of_ranking_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            shoot = Path(tmp) / "shoot"
            ranked = make_ranked(shoot, 6)
            # The worst-ranked image kept, the best-ranked rejected - exactly
            # what a percentage cut could never express.
            result = organize_by_decision([ranked[5]], ranked[:5], shoot)

            self.assertEqual(result.selected, 1)
            self.assertEqual(result.rejected, 5)
            self.assertEqual(result.moved, 6)
            self.assertTrue((shoot / SELECTED_DIRNAME / "IMG_0005.NEF").exists())
            self.assertTrue((shoot / REJECTED_DIRNAME / "IMG_0000.NEF").exists())

    def test_an_image_in_neither_set_is_left_untouched(self):
        """How an unranked, undecided image stays put instead of being swept
        somewhere it was never judged to belong."""
        with tempfile.TemporaryDirectory() as tmp:
            shoot = Path(tmp) / "shoot"
            ranked = make_ranked(shoot, 4)
            untouched = Path(ranked[3])

            result = organize_by_decision(ranked[:2], ranked[2:3], shoot)

            self.assertEqual(result.ranked, 3, "the omitted image is not counted")
            self.assertTrue(untouched.exists(), "an omitted image must not move")
            self.assertEqual(untouched.parent, shoot)

    def test_nothing_is_overwritten_on_a_name_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            shoot = Path(tmp) / "shoot"
            ranked = make_ranked(shoot, 1)
            existing = shoot / SELECTED_DIRNAME
            existing.mkdir(parents=True)
            (existing / "IMG_0000.NEF").write_bytes(b"do not clobber me")

            result = organize_by_decision(ranked, [], shoot)

            self.assertEqual(result.renamed, 1)
            self.assertEqual((existing / "IMG_0000.NEF").read_bytes(), b"do not clobber me")
            self.assertTrue((existing / "IMG_0000_1.NEF").exists())

    def test_a_missing_source_is_recorded_and_the_rest_still_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            shoot = Path(tmp) / "shoot"
            ranked = make_ranked(shoot, 3)
            Path(ranked[0]).unlink()

            result = organize_by_decision(ranked[:2], ranked[2:], shoot)

            self.assertEqual(result.skipped, 1)
            self.assertEqual(result.moved, 2)
            self.assertEqual(result.failures[0][1], "source file not found")

    def test_rerunning_is_a_no_op_rather_than_a_shuffle(self):
        with tempfile.TemporaryDirectory() as tmp:
            shoot = Path(tmp) / "shoot"
            ranked = make_ranked(shoot, 4)
            first = organize_by_decision(ranked[:2], ranked[2:], shoot)

            selected_now = [str(p) for p in first.moves.values()][:2]
            rejected_now = [str(p) for p in first.moves.values()][2:]
            second = organize_by_decision(selected_now, rejected_now, shoot)

            self.assertEqual(second.moved, 0)
            self.assertEqual(second.skipped, 4)
            self.assertEqual(second.errors, 0)

    def test_dry_run_moves_nothing_but_reports_the_full_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            shoot = Path(tmp) / "shoot"
            ranked = make_ranked(shoot, 4)

            result = organize_by_decision(ranked[:1], ranked[1:], shoot, dry_run=True)

            self.assertEqual(result.moved, 4)
            self.assertEqual(len(result.moves), 4, "the plan is needed for the confirm dialog")
            for path in ranked:
                self.assertTrue(Path(path).exists(), "dry_run must not move a file")
            self.assertFalse((shoot / SELECTED_DIRNAME).exists(), "dry_run must not create folders")

    def test_empty_input_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            shoot = Path(tmp) / "shoot"
            shoot.mkdir(parents=True)
            result = organize_by_decision([], [], shoot)
            self.assertEqual(result.ranked, 0)
            self.assertFalse((shoot / SELECTED_DIRNAME).exists())


class EnumerationTests(unittest.TestCase):
    """A second ranking run must not re-rank its own output."""

    def test_organize_folders_are_excluded_from_enumeration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.NEF").write_bytes(b"x")
            for folder in (SELECTED_DIRNAME, REJECTED_DIRNAME):
                (root / folder).mkdir()
                (root / folder / f"filed_{folder}.NEF").write_bytes(b"x")

            everything = UnlabeledImageDataset.from_folder(root)
            self.assertEqual(len(everything), 3)

            fresh_only = UnlabeledImageDataset.from_folder(root, exclude_dirs=set(ORGANIZE_DIRNAMES))
            self.assertEqual([Path(i.image_path).name for i in fresh_only.items], ["a.NEF"])

    def test_exclusion_is_case_insensitive_and_covers_nesting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # A case variant of SELECTED_DIRNAME, nested: exclusion must match
            # on the folder name however it is spelled and at any depth.
            nested = root / "2026" / "_selected" / "deeper"
            nested.mkdir(parents=True)
            (nested / "filed.NEF").write_bytes(b"x")
            (root / "keep.NEF").write_bytes(b"x")

            dataset = UnlabeledImageDataset.from_folder(root, exclude_dirs=set(ORGANIZE_DIRNAMES))
            self.assertEqual([Path(i.image_path).name for i in dataset.items], ["keep.NEF"])


class CliTests(unittest.TestCase):
    def test_defaults_match_the_specification(self):
        args = build_arg_parser().parse_args(["--input", "x"])
        self.assertTrue(args.organize_output)
        self.assertEqual(args.selection_percentage, DEFAULT_SELECTION_PERCENTAGE)
        self.assertEqual(args.selection_percentage, 25)

    def test_true_and_false_are_both_accepted(self):
        for text, expected in (
            ("true", True), ("false", False), ("True", True), ("FALSE", False),
            ("yes", True), ("no", False), ("1", True), ("0", False),
        ):
            args = build_arg_parser().parse_args(["--input", "x", "--organize-output", text])
            self.assertIs(args.organize_output, expected, text)

    def test_flag_without_a_value_means_true(self):
        args = build_arg_parser().parse_args(["--input", "x", "--organize-output"])
        self.assertTrue(args.organize_output)

    def test_a_nonsense_value_is_rejected(self):
        with self.assertRaises(SystemExit):
            build_arg_parser().parse_args(["--input", "x", "--organize-output", "maybe"])

    def test_percentage_and_dir_are_parsed(self):
        args = build_arg_parser().parse_args(
            ["--input", "x", "--selection-percentage", "40", "--organize-dir", "out"]
        )
        self.assertEqual(args.selection_percentage, 40.0)
        self.assertEqual(args.organize_dir, "out")

    def test_existing_command_lines_still_work_unchanged(self):
        """Backward compatibility: the flags are optional and old invocations
        parse exactly as before."""
        args = build_arg_parser().parse_args(
            ["--input", "D:/shoot", "--checkpoint", "ckpt.pt", "--max-rows", "500"]
        )
        self.assertEqual(args.input, "D:/shoot")
        self.assertEqual(args.max_rows, 500)
        self.assertTrue(args.organize_output)  # new default, opt-out available


class IndependenceTests(unittest.TestCase):
    """The ranking and analyzer modules must stay decoupled."""

    @staticmethod
    def _imported_names(module_path: Path) -> set[str]:
        """Every module named in an import statement.

        Parsed with `ast` rather than grepped: the docstrings deliberately
        discuss the other module to explain the separation, and a substring
        search would flag that prose as a dependency.
        """
        import ast

        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(alias.name for alias in node.names)
        return names

    def test_organize_does_not_import_the_analyzer(self):
        source = Path(__file__).resolve().parents[1] / "src" / "picklikeme" / "organize.py"
        for name in self._imported_names(source):
            self.assertNotIn("analyzer", name, f"organize imports {name}")

    def test_the_analyzer_never_imports_organize(self):
        analyzer_dir = Path(__file__).resolve().parents[1] / "src" / "picklikeme" / "analyzer"
        for module in analyzer_dir.rglob("*.py"):
            for name in self._imported_names(module):
                self.assertNotIn("organize", name, f"{module.name} imports {name}")


if __name__ == "__main__":
    unittest.main()
