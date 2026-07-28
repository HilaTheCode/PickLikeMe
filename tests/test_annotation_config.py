"""load_annotation_fields(): parsing and validating config/annotations.yaml.

Every case here is something a photographer editing the YAML by hand could
get wrong, so each must raise AnnotationConfigError with a message that
names the offending field or value - not a bare traceback.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picklikeme.analyzer.annotation_config import (
    AnnotationConfigError,
    AnnotationFieldsConfig,
    load_annotation_fields,
)


def _write(tmp: Path, text: str) -> Path:
    path = tmp / "annotations.yaml"
    path.write_text(text, encoding="utf-8")
    return path


VALID = """
annotation_fields:
  crop_quality:
    label: Crop Quality
    values:
      - id: good
        label: Good
      - id: too_small
        label: Too Small
  agree_with_model_decision:
    label: Agree with Model Decision
    values:
      - id: "yes"
        label: "Yes"
      - id: "no"
        label: "No"
"""


class LoadValidConfigTests(unittest.TestCase):
    def test_loads_fields_in_file_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), VALID)
            config = load_annotation_fields(path)
            self.assertIsInstance(config, AnnotationFieldsConfig)
            self.assertEqual(config.field_ids, ("crop_quality", "agree_with_model_decision"))

    def test_values_preserve_file_order_and_ids_are_not_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_annotation_fields(_write(Path(tmp), VALID))
            crop = config.get("crop_quality")
            self.assertEqual(crop.value_ids, ("good", "too_small"))
            self.assertEqual(crop.values[1].label, "Too Small")

    def test_quoted_yes_no_load_as_strings_not_booleans(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_annotation_fields(_write(Path(tmp), VALID))
            agree = config.get("agree_with_model_decision")
            self.assertEqual(agree.value_ids, ("yes", "no"))
            self.assertEqual(agree.label_for("no"), "No")

    def test_get_returns_none_for_an_unconfigured_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_annotation_fields(_write(Path(tmp), VALID))
            self.assertIsNone(config.get("nonexistent"))

    def test_label_for_falls_back_to_the_raw_id_when_retired(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_annotation_fields(_write(Path(tmp), VALID))
            crop = config.get("crop_quality")
            self.assertEqual(crop.label_for("something_removed"), "something_removed")
            self.assertIsNone(crop.label_for(None))
            self.assertIsNone(crop.label_for(""))

    def test_the_real_shipped_config_loads(self):
        from picklikeme.analyzer.annotation_config import DEFAULT_ANNOTATIONS_CONFIG

        config = load_annotation_fields(DEFAULT_ANNOTATIONS_CONFIG)
        self.assertIn("crop_quality", config.field_ids)
        self.assertIn("image_quality", config.field_ids)
        self.assertIn("agree_with_model_decision", config.field_ids)


class InvalidConfigTests(unittest.TestCase):
    def _load(self, tmp: Path, text: str):
        return load_annotation_fields(_write(tmp, text))

    def test_missing_file_raises_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError) as ctx:
                load_annotation_fields(Path(tmp) / "does_not_exist.yaml")
            self.assertIn("not found", str(ctx.exception))

    def test_duplicate_value_id_within_a_field_is_rejected(self):
        text = """
annotation_fields:
  crop_quality:
    label: Crop Quality
    values:
      - id: good
        label: Good
      - id: good
        label: Also Good
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError) as ctx:
                self._load(Path(tmp), text)
            self.assertIn("duplicate value id", str(ctx.exception))
            self.assertIn("'good'", str(ctx.exception))

    def test_empty_value_id_is_rejected(self):
        text = """
annotation_fields:
  crop_quality:
    label: Crop Quality
    values:
      - id: ""
        label: Good
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError) as ctx:
                self._load(Path(tmp), text)
            self.assertIn("must not be empty", str(ctx.exception))

    def test_empty_value_label_is_rejected(self):
        text = """
annotation_fields:
  crop_quality:
    label: Crop Quality
    values:
      - id: good
        label: ""
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError) as ctx:
                self._load(Path(tmp), text)
            self.assertIn("must not be empty", str(ctx.exception))

    def test_empty_field_label_is_rejected(self):
        text = """
annotation_fields:
  crop_quality:
    label: ""
    values:
      - id: good
        label: Good
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError) as ctx:
                self._load(Path(tmp), text)
            self.assertIn("must not be empty", str(ctx.exception))

    def test_zero_values_for_a_field_is_rejected(self):
        text = """
annotation_fields:
  crop_quality:
    label: Crop Quality
    values: []
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError) as ctx:
                self._load(Path(tmp), text)
            self.assertIn("non-empty 'values'", str(ctx.exception))

    def test_zero_fields_defined_is_rejected(self):
        text = "annotation_fields: {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError):
                self._load(Path(tmp), text)

    def test_missing_top_level_key_is_rejected(self):
        text = "something_else: {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError) as ctx:
                self._load(Path(tmp), text)
            self.assertIn("annotation_fields", str(ctx.exception))

    def test_unquoted_yes_no_value_id_is_rejected_with_a_helpful_hint(self):
        """The classic YAML 1.1 footgun: unquoted `yes`/`no` parse as booleans,
        not strings - exactly the ids this field's config is likely to use."""
        text = """
annotation_fields:
  agree_with_model_decision:
    label: Agree with Model Decision
    values:
      - id: yes
        label: Yes
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError) as ctx:
                self._load(Path(tmp), text)
            self.assertIn("quote", str(ctx.exception))

    def test_unquoted_yes_label_is_also_rejected(self):
        text = """
annotation_fields:
  agree_with_model_decision:
    label: Agree with Model Decision
    values:
      - id: "yes"
        label: yes
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError):
                self._load(Path(tmp), text)

    def test_malformed_yaml_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationConfigError):
                self._load(Path(tmp), "annotation_fields: [this, is, not, a, mapping\n")


if __name__ == "__main__":
    unittest.main()
