"""species.classifier.read_species_list - the public, backend-agnostic
species-list-file parser, and the fix for a real bug found while building
this: a species_list_path resolving to zero valid species used to be
silently swallowed by "species_list or DEFAULT_SPECIES_LIST" and replaced
with the 55-species built-in list, with no indication to the photographer
that their chosen file was effectively ignored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picklikeme.species.classifier import read_species_list


def test_reads_one_species_per_line(tmp_path: Path) -> None:
    path = tmp_path / "species.txt"
    path.write_text("Kingfisher\nOsprey\nCommon Tern\n", encoding="utf-8")
    assert read_species_list(path) == ("Kingfisher", "Osprey", "Common Tern")


def test_ignores_blank_lines_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "species.txt"
    path.write_text("Kingfisher\n\n# a comment\n  \nOsprey\n#another\n", encoding="utf-8")
    assert read_species_list(path) == ("Kingfisher", "Osprey")


def test_strips_surrounding_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "species.txt"
    path.write_text("  Kingfisher  \n\tOsprey\t\n", encoding="utf-8")
    assert read_species_list(path) == ("Kingfisher", "Osprey")


def test_a_file_of_only_blanks_and_comments_returns_empty_not_an_error(tmp_path: Path) -> None:
    """read_species_list itself never raises for this case - it is the
    caller's job to decide what "zero species" means (see
    BioClipSpeciesClassifier's own validation below)."""
    path = tmp_path / "species.txt"
    path.write_text("# nothing here\n\n   \n", encoding="utf-8")
    assert read_species_list(path) == ()


def test_a_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_species_list(tmp_path / "does_not_exist.txt")


class TestEmptySpeciesListFileIsARealError:
    """The bug: species_list_path -> zero species used to silently fall
    back to DEFAULT_SPECIES_LIST. Constructing a real BioClipSpeciesClassifier
    downloads a model, so these tests bypass __init__ for the parts that
    don't need it and only exercise the real constructor once, with the
    heavy model-loading calls stubbed - matching this project's own "stub
    the model, test the wiring" convention."""

    @staticmethod
    def _stub_open_clip(monkeypatch):
        import torch

        class _FakeTensor:
            def to(self, device):
                return self

            def norm(self, dim=-1, keepdim=True):
                return torch.ones(1)

            def __truediv__(self, other):
                return self

        class _FakeModel:
            def to(self, device):
                return self

            def eval(self):
                return self

            def encode_text(self, tokens):
                return _FakeTensor()

        import open_clip

        monkeypatch.setattr(
            open_clip, "create_model_and_transforms", lambda model_id: (_FakeModel(), None, lambda img: None)
        )
        monkeypatch.setattr(open_clip, "get_tokenizer", lambda model_id: lambda prompts: _FakeTensor())

    def test_an_empty_species_list_file_raises_a_clear_error(self, tmp_path: Path, monkeypatch) -> None:
        from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier

        self._stub_open_clip(monkeypatch)
        path = tmp_path / "empty_species.txt"
        path.write_text("# only comments\n\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no valid species"):
            BioClipSpeciesClassifier(species_list_path=path)

    def test_a_populated_species_list_file_is_used_not_the_default(self, tmp_path: Path, monkeypatch) -> None:
        from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier, DEFAULT_SPECIES_LIST

        self._stub_open_clip(monkeypatch)
        path = tmp_path / "species.txt"
        path.write_text("Kingfisher\nOsprey\n", encoding="utf-8")

        classifier = BioClipSpeciesClassifier(species_list_path=path)
        assert classifier.species_list == ("Kingfisher", "Osprey")
        assert classifier.species_list != DEFAULT_SPECIES_LIST

    def test_no_path_at_all_still_falls_back_to_the_default(self, monkeypatch) -> None:
        """Unrelated, pre-existing behaviour, confirmed unchanged: a caller
        that passes neither species_list nor species_list_path still gets
        the built-in default - only an explicitly-chosen, empty FILE is
        now an error."""
        from picklikeme.species.bioclip_classifier import BioClipSpeciesClassifier, DEFAULT_SPECIES_LIST

        self._stub_open_clip(monkeypatch)
        classifier = BioClipSpeciesClassifier()
        assert classifier.species_list == DEFAULT_SPECIES_LIST
