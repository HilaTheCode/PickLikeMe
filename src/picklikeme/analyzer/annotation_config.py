"""The annotation field/value vocabulary, as data - not code.

`AnnotationStore` used to hardcode three fields and their fixed vocabularies
as Python constants. That made every new diagnostic value a source change.
This module loads the same information from `config/annotations.yaml`
instead, so adding a field or a value is an edit to that file, not to this
package - the analyzer just needs restarting.

Two failure modes are treated very differently:

- **A malformed config file** (duplicate ids, empty labels, bad YAML, ...) is
  a pure authoring bug with nothing sensible to fall back to, so
  `load_annotation_fields()` raises `AnnotationConfigError` and the analyzer
  refuses to start - the same "clear startup error" this project already uses
  for `AnalysisConfig.from_file`.
- **A config edit that removes an id already used in the database** is not
  caught here at all (this module has no database access) - see
  `AnnotationStore` for how that is detected and handled as a warning, not a
  crash, once the store is open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT

DEFAULT_ANNOTATIONS_CONFIG = PROJECT_ROOT / "config" / "annotations.yaml"


class AnnotationConfigError(ValueError):
    """The annotation config file is missing or invalid. Always includes
    enough detail (which field, which value, why) to fix without guessing."""


@dataclass(frozen=True)
class AnnotationValue:
    id: str
    label: str


@dataclass(frozen=True)
class AnnotationField:
    id: str
    label: str
    values: tuple[AnnotationValue, ...]

    @property
    def value_ids(self) -> tuple[str, ...]:
        return tuple(value.id for value in self.values)

    def has_value(self, value_id: str) -> bool:
        return value_id in self.value_ids

    def label_for(self, value_id: str | None) -> str | None:
        """The configured label for a value id, or the raw id itself if it is
        no longer configured (a retired value - see the module docstring).
        Never raises: a report must always be able to render historical data,
        even after a config edit that dropped the id."""
        if not value_id:
            return None
        for value in self.values:
            if value.id == value_id:
                return value.label
        return value_id


@dataclass(frozen=True)
class AnnotationFieldsConfig:
    fields: tuple[AnnotationField, ...]  # preserves config file order

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def get(self, field_id: str) -> AnnotationField | None:
        for field in self.fields:
            if field.id == field_id:
                return field
        return None

    @property
    def field_ids(self) -> tuple[str, ...]:
        return tuple(field.id for field in self.fields)


def _require_string(value: object, what: str) -> str:
    if not isinstance(value, str):
        hint = ""
        if isinstance(value, bool):
            # The classic trap: an unquoted `yes`/`no`/`on`/`off` id is parsed
            # as a YAML 1.1 boolean, not the string the author meant.
            hint = " - did you forget to quote it (e.g. \"yes\", not yes)?"
        raise AnnotationConfigError(f"{what} must be a string, got {value!r}{hint}")
    return value


def _parse_value(raw: object, *, field_id: str) -> AnnotationValue:
    if not isinstance(raw, dict):
        raise AnnotationConfigError(
            f"annotation field {field_id!r}: each value must be a mapping with 'id' and 'label', got {raw!r}"
        )
    value_id = _require_string(raw.get("id"), f"annotation field {field_id!r}: value id")
    if not value_id.strip():
        raise AnnotationConfigError(f"annotation field {field_id!r}: a value id must not be empty")
    label = _require_string(raw.get("label"), f"annotation field {field_id!r}: label for value {value_id!r}")
    if not label.strip():
        raise AnnotationConfigError(f"annotation field {field_id!r}: label for value {value_id!r} must not be empty")
    return AnnotationValue(id=value_id, label=label)


def _parse_field(field_id: object, raw: object) -> AnnotationField:
    field_id = _require_string(field_id, "annotation field id")
    if not field_id.strip():
        raise AnnotationConfigError("an annotation field id must not be empty")
    if not isinstance(raw, dict):
        raise AnnotationConfigError(f"annotation field {field_id!r} must be a mapping with 'label' and 'values'")

    label = _require_string(raw.get("label"), f"annotation field {field_id!r}: label")
    if not label.strip():
        raise AnnotationConfigError(f"annotation field {field_id!r}: label must not be empty")

    raw_values = raw.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        raise AnnotationConfigError(f"annotation field {field_id!r} must have a non-empty 'values' list")

    values = tuple(_parse_value(item, field_id=field_id) for item in raw_values)
    seen: set[str] = set()
    for value in values:
        if value.id in seen:
            raise AnnotationConfigError(
                f"annotation field {field_id!r}: duplicate value id {value.id!r} - ids must be unique within a field"
            )
        seen.add(value.id)

    return AnnotationField(id=field_id, label=label, values=values)


def load_annotation_fields(path: str | Path = DEFAULT_ANNOTATIONS_CONFIG) -> AnnotationFieldsConfig:
    """Load and validate the annotation field/value config.

    Raises `AnnotationConfigError` for anything a photographer editing this
    file by hand could get wrong: a missing file, bad YAML, an empty or
    duplicate id, an empty label, or zero fields defined.
    """
    path = Path(path)
    if not path.is_file():
        raise AnnotationConfigError(
            f"Annotation config not found at {path}. Restore config/annotations.yaml, or pass "
            "--annotations-config to point at a different file."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AnnotationConfigError(f"Could not parse {path}: {exc}") from exc

    if not isinstance(raw, dict) or "annotation_fields" not in raw:
        raise AnnotationConfigError(f"{path} must have a top-level 'annotation_fields' mapping")
    raw_fields = raw["annotation_fields"]
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise AnnotationConfigError(f"{path}: 'annotation_fields' must be a non-empty mapping")

    # A literal duplicate key in the YAML source is not detectable here - by
    # the time yaml.safe_load() returns a dict, PyYAML has already collapsed
    # it to one entry (last wins), same as Python dict-literal semantics.
    fields = tuple(_parse_field(field_id, body) for field_id, body in raw_fields.items())
    return AnnotationFieldsConfig(fields=fields)
