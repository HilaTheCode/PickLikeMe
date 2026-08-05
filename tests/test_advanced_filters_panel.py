"""desktop/widgets/advanced_filters_panel.py - the one Advanced Filters
widget shared by the Review Window and the Analytics Dashboard.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


def _panel():
    from picklikeme.desktop.widgets.advanced_filters_panel import AdvancedFiltersPanel

    QApplication.instance() or QApplication([])
    return AdvancedFiltersPanel()


def test_starts_collapsed_with_default_criteria():
    panel = _panel()
    assert panel._content.isHidden() is True
    assert panel.criteria.is_active() is False


def test_toggle_button_expands_and_collapses():
    panel = _panel()
    panel._toggle_button.setChecked(True)
    assert panel._content.isHidden() is False
    assert "▾" in panel._toggle_button.text()
    panel._toggle_button.setChecked(False)
    assert panel._content.isHidden() is True
    assert "▸" in panel._toggle_button.text()


def test_search_text_flows_into_criteria_and_emits_live():
    panel = _panel()
    received = []
    panel.criteriaChanged.connect(lambda: received.append(panel.criteria))

    panel._search_edit.setText("DSC1234")

    assert received[-1].search == "DSC1234"


def test_folder_and_species_options_populate_and_select():
    panel = _panel()
    panel.set_available_options(folders=["/shoot", "/shoot/_Selected"], species=["Kingfisher", "Osprey"])

    panel._folder_combo.setCurrentText("/shoot/_Selected")
    panel._species_combo.setCurrentText("Kingfisher")

    assert panel.criteria.folder == "/shoot/_Selected"
    assert panel.criteria.species == "Kingfisher"


def test_all_folders_and_all_species_map_to_none():
    panel = _panel()
    panel.set_available_options(folders=["/shoot"], species=["Kingfisher"])
    assert panel.criteria.folder is None
    assert panel.criteria.species is None


def test_burst_winners_and_losers():
    panel = _panel()
    panel._burst_combo.setCurrentText("Winners")
    assert panel.criteria.burst == "winners"
    panel._burst_combo.setCurrentText("Losers")
    assert panel.criteria.burst == "losers"
    panel._burst_combo.setCurrentText("All")
    assert panel.criteria.burst == "all"


def test_burst_rank_options_populate_from_max_burst_size():
    panel = _panel()
    panel.set_available_options(max_burst_size=3)
    items = [panel._burst_rank_combo.itemText(i) for i in range(panel._burst_rank_combo.count())]
    assert items == ["(Any)", "1", "2", "3"]

    panel._burst_rank_combo.setCurrentText("2")
    assert panel.criteria.burst_rank == 2


def test_user_and_algorithm_decision_lowercase_into_criteria():
    panel = _panel()
    panel._user_decision_combo.setCurrentText("Reject")
    panel._algorithm_decision_combo.setCurrentText("Keep")
    assert panel.criteria.user_decision == "reject"
    assert panel.criteria.algorithm_decision == "keep"


def test_conflict_type_maps_the_display_label_back_to_the_engines_key():
    from picklikeme.desktop.filtering import CONFLICT_FALSE_POSITIVE, CONFLICT_LABELS

    panel = _panel()
    panel._conflict_combo.setCurrentText(CONFLICT_LABELS[CONFLICT_FALSE_POSITIVE])
    assert panel.criteria.conflict_type == CONFLICT_FALSE_POSITIVE


def test_reject_reason_options_populate_and_map_label_to_code():
    panel = _panel()
    panel.set_available_options(reject_reasons=[("NO_VISIBLE_EYE", "No visible eye")])
    panel._reject_reason_combo.setCurrentText("No visible eye")
    assert panel.criteria.reject_reason == "NO_VISIBLE_EYE"


def test_range_filter_is_inactive_until_its_checkbox_is_checked():
    panel = _panel()
    panel._range_mins["score"].setValue(0.5)
    # Spinbox has a value, but the checkbox is still off - must not filter.
    assert panel.criteria.score_min is None
    assert panel.criteria.score_max is None

    panel._range_checks["score"].setChecked(True)
    assert panel.criteria.score_min == 0.5


def test_checking_a_range_checkbox_enables_its_spinboxes():
    panel = _panel()
    assert panel._range_mins["eye_confidence"].isEnabled() is False
    panel._range_checks["eye_confidence"].setChecked(True)
    assert panel._range_mins["eye_confidence"].isEnabled() is True
    assert panel._range_maxes["eye_confidence"].isEnabled() is True


def test_every_range_row_is_wired_to_its_own_criteria_field():
    panel = _panel()
    for field in (
        "score", "eye_confidence", "head_confidence", "subject_size",
        "eye_sharpness", "subject_sharpness", "species_confidence",
    ):
        panel._range_checks[field].setChecked(True)
        panel._range_mins[field].setValue(1.0)
        panel._range_maxes[field].setValue(2.0)
        criteria = panel.criteria
        assert getattr(criteria, f"{field}_min") == 1.0, field
        assert getattr(criteria, f"{field}_max") == 2.0, field
        panel._range_checks[field].setChecked(False)


def test_reset_clears_every_control_and_emits_once():
    panel = _panel()
    panel._search_edit.setText("something")
    panel._range_checks["score"].setChecked(True)
    panel._range_mins["score"].setValue(0.5)
    panel._user_decision_combo.setCurrentText("Reject")

    received = []
    panel.criteriaChanged.connect(lambda: received.append(True))
    panel.reset()

    assert panel.criteria.is_active() is False
    assert len(received) == 1


def test_set_available_options_preserves_a_still_valid_selection():
    panel = _panel()
    panel.set_available_options(species=["Kingfisher", "Osprey"])
    panel._species_combo.setCurrentText("Kingfisher")

    panel.set_available_options(species=["Kingfisher", "Osprey", "Fish Eagle"])

    assert panel._species_combo.currentText() == "Kingfisher"


def test_set_available_options_falls_back_to_all_when_selection_no_longer_exists():
    panel = _panel()
    panel.set_available_options(species=["Kingfisher"])
    panel._species_combo.setCurrentText("Kingfisher")

    panel.set_available_options(species=["Osprey"])  # Kingfisher no longer present

    assert panel.criteria.species is None
