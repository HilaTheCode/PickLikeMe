"""The Advanced Filters panel (Product Direction Phase A/B/C): one
collapsible widget, reused unchanged by both the main Review Window and
the Analytics Dashboard, so "use one shared filtering engine" is true at
the UI layer too, not only in `desktop.filtering`'s matching logic.

Every control writes into one `FilterCriteria` and emits `criteriaChanged`
immediately on any change - no Apply button, per the product direction's
own explicit requirement ("The main gallery should immediately update. No
additional Apply button."). A numeric range filter is only active while
its own checkbox is checked - checkbox state, not spinbox value, is what
tells `criteria` a range filter is "on", since a spinbox always holds some
number and using an extreme value as an implicit "off" sentinel would be
both fragile and invisible to the user.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..filtering import CONFLICT_LABELS, FilterCriteria

_ANY = "(Any)"
_ALL_FOLDERS = "All Folders"
_ALL_SPECIES = "All Species"

# (criteria field prefix, display label) for every ranged numeric filter -
# one shared definition the widget builds its rows from, so a field added
# to `filtering.py` later needs one new tuple here, not a new copy-pasted
# row block.
_RANGE_ROWS: tuple[tuple[str, str], ...] = (
    ("score", "Score"),
    ("eye_confidence", "Eye Confidence"),
    ("head_confidence", "Head Confidence"),
    ("subject_size", "Subject Size"),
    ("eye_sharpness", "Eye Sharpness"),
    ("subject_sharpness", "Subject Sharpness"),
    ("species_confidence", "Species Confidence"),
)


class AdvancedFiltersPanel(QWidget):
    """Collapsed by default (rarely-needed power tool, not the first thing
    a photographer sees) - click the header to expand. See this module's
    own docstring for why it emits live rather than needing an Apply
    button."""

    criteriaChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._toggle_button = QPushButton("▸ Advanced Filters", self)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(False)
        self._toggle_button.setFlat(True)
        self._toggle_button.setStyleSheet("text-align: left; border: none; padding: 2px 0; font-weight: 600;")
        self._toggle_button.toggled.connect(self._on_toggled)

        self._content = QWidget(self)
        self._content.setVisible(False)
        grid = QGridLayout(self._content)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        row = 0

        self._search_edit = QLineEdit(self._content)
        self._search_edit.setPlaceholderText("Search filename...")
        grid.addWidget(QLabel("Search", self._content), row, 0)
        grid.addWidget(self._search_edit, row, 1, 1, 3)
        row += 1

        self._folder_combo = QComboBox(self._content)
        self._species_combo = QComboBox(self._content)
        grid.addWidget(QLabel("Folder", self._content), row, 0)
        grid.addWidget(self._folder_combo, row, 1)
        grid.addWidget(QLabel("Species", self._content), row, 2)
        grid.addWidget(self._species_combo, row, 3)
        row += 1

        self._burst_combo = QComboBox(self._content)
        self._burst_combo.addItems(["All", "Winners", "Losers"])
        self._burst_rank_combo = QComboBox(self._content)
        grid.addWidget(QLabel("Burst", self._content), row, 0)
        grid.addWidget(self._burst_combo, row, 1)
        grid.addWidget(QLabel("Burst Rank", self._content), row, 2)
        grid.addWidget(self._burst_rank_combo, row, 3)
        row += 1

        self._user_decision_combo = QComboBox(self._content)
        self._user_decision_combo.addItems([_ANY, "Keep", "Reject", "Neutral"])
        self._algorithm_decision_combo = QComboBox(self._content)
        self._algorithm_decision_combo.addItems([_ANY, "Keep", "Reject"])
        grid.addWidget(QLabel("User Decision", self._content), row, 0)
        grid.addWidget(self._user_decision_combo, row, 1)
        grid.addWidget(QLabel("Algorithm Decision", self._content), row, 2)
        grid.addWidget(self._algorithm_decision_combo, row, 3)
        row += 1

        self._conflict_combo = QComboBox(self._content)
        self._conflict_combo.addItem(_ANY)
        for label in CONFLICT_LABELS.values():
            self._conflict_combo.addItem(label)
        self._reject_reason_combo = QComboBox(self._content)
        grid.addWidget(QLabel("Conflict Type", self._content), row, 0)
        grid.addWidget(self._conflict_combo, row, 1)
        grid.addWidget(QLabel("Reject Reason", self._content), row, 2)
        grid.addWidget(self._reject_reason_combo, row, 3)
        row += 1

        grid.addWidget(QLabel("Ranges", self._content), row, 0)
        row += 1

        self._range_checks: dict[str, QCheckBox] = {}
        self._range_mins: dict[str, QDoubleSpinBox] = {}
        self._range_maxes: dict[str, QDoubleSpinBox] = {}
        for field, label in _RANGE_ROWS:
            check = QCheckBox(label, self._content)
            check.toggled.connect(self._on_control_changed)
            spin_min = QDoubleSpinBox(self._content)
            spin_max = QDoubleSpinBox(self._content)
            for spin in (spin_min, spin_max):
                spin.setRange(-1_000_000.0, 1_000_000.0)
                spin.setDecimals(4)
                spin.setEnabled(False)
                spin.valueChanged.connect(self._on_control_changed)
            spin_max.setValue(1_000_000.0)
            check.toggled.connect(spin_min.setEnabled)
            check.toggled.connect(spin_max.setEnabled)
            self._range_checks[field] = check
            self._range_mins[field] = spin_min
            self._range_maxes[field] = spin_max

            range_row = QHBoxLayout()
            range_row.addWidget(spin_min)
            range_row.addWidget(QLabel("to", self._content))
            range_row.addWidget(spin_max)
            grid.addWidget(check, row, 0)
            grid.addLayout(range_row, row, 1, 1, 3)
            row += 1

        clear_button = QPushButton("Clear All Filters", self._content)
        clear_button.clicked.connect(self.reset)
        grid.addWidget(clear_button, row, 0, 1, 4)

        for combo in (
            self._folder_combo, self._species_combo, self._burst_combo, self._burst_rank_combo,
            self._user_decision_combo, self._algorithm_decision_combo, self._conflict_combo,
            self._reject_reason_combo,
        ):
            combo.currentIndexChanged.connect(self._on_control_changed)
        self._search_edit.textChanged.connect(self._on_control_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._content)

        self._label_to_conflict_key = {label: key for key, label in CONFLICT_LABELS.items()}

    # -- collapse/expand -----------------------------------------------------

    def _on_toggled(self, expanded: bool) -> None:
        self._content.setVisible(expanded)
        self._toggle_button.setText(("▾" if expanded else "▸") + " Advanced Filters")

    # -- populating dynamic option lists --------------------------------------

    def set_available_options(
        self, *, folders: list[str] | None = None, species: list[str] | None = None,
        reject_reasons: list[tuple[str, str]] | None = None, max_burst_size: int = 1,
    ) -> None:
        """Repopulate the dynamic dropdowns from the current candidate image
        set - called whenever that set changes (a folder opened, an
        experiment selected). Preserves the current selection when it is
        still one of the new options, so re-populating after a decision
        change doesn't silently reset a filter the photographer is mid-way
        through using.

        `reject_reasons` is `(code, label)` pairs - the code is what
        `FilterableRecord.reject_reason` actually holds, the label is what
        a photographer reads (see `ranking.filters.REJECT_REASON_LABELS`).
        """
        self._repopulate(self._folder_combo, [_ALL_FOLDERS] + sorted(folders or []))
        self._repopulate(self._species_combo, [_ALL_SPECIES] + sorted(species or []))
        self._reject_reason_by_label = {label: code for code, label in (reject_reasons or [])}
        self._repopulate(self._reject_reason_combo, [_ANY] + sorted(self._reject_reason_by_label))
        self._repopulate(self._burst_rank_combo, [_ANY] + [str(n) for n in range(1, max(1, max_burst_size) + 1)])

    @staticmethod
    def _repopulate(combo: QComboBox, options: list[str]) -> None:
        previous = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(options)
        restored = combo.findText(previous)
        combo.setCurrentIndex(restored if restored >= 0 else 0)
        combo.blockSignals(False)

    # -- reading/writing criteria ----------------------------------------------

    def _on_control_changed(self, *_args) -> None:
        self.criteriaChanged.emit()

    @property
    def criteria(self) -> FilterCriteria:
        def _optional(combo: QComboBox, placeholder: str) -> str | None:
            text = combo.currentText()
            return None if text in (placeholder, "") else text

        def _range(field: str) -> tuple[float | None, float | None]:
            if not self._range_checks[field].isChecked():
                return None, None
            return self._range_mins[field].value(), self._range_maxes[field].value()

        ranges = {field: _range(field) for field, _label in _RANGE_ROWS}
        burst_choice = self._burst_combo.currentText().lower()
        conflict_label = _optional(self._conflict_combo, _ANY)
        burst_rank_text = _optional(self._burst_rank_combo, _ANY)
        reject_reason_label = _optional(self._reject_reason_combo, _ANY)

        return FilterCriteria(
            search=self._search_edit.text().strip(),
            folder=_optional(self._folder_combo, _ALL_FOLDERS),
            species=_optional(self._species_combo, _ALL_SPECIES),
            burst=burst_choice if burst_choice in ("winners", "losers") else "all",
            burst_rank=int(burst_rank_text) if burst_rank_text else None,
            user_decision=(_optional(self._user_decision_combo, _ANY) or "").lower() or None,
            algorithm_decision=(_optional(self._algorithm_decision_combo, _ANY) or "").lower() or None,
            conflict_type=self._label_to_conflict_key.get(conflict_label) if conflict_label else None,
            reject_reason=self._reject_reason_by_label.get(reject_reason_label) if reject_reason_label else None,
            score_min=ranges["score"][0], score_max=ranges["score"][1],
            eye_confidence_min=ranges["eye_confidence"][0], eye_confidence_max=ranges["eye_confidence"][1],
            head_confidence_min=ranges["head_confidence"][0], head_confidence_max=ranges["head_confidence"][1],
            subject_size_min=ranges["subject_size"][0], subject_size_max=ranges["subject_size"][1],
            eye_sharpness_min=ranges["eye_sharpness"][0], eye_sharpness_max=ranges["eye_sharpness"][1],
            subject_sharpness_min=ranges["subject_sharpness"][0], subject_sharpness_max=ranges["subject_sharpness"][1],
            species_confidence_min=ranges["species_confidence"][0], species_confidence_max=ranges["species_confidence"][1],
        )

    def reset(self) -> None:
        """Back to "no filters active" - every control to its neutral
        value, one signal emitted at the end rather than one per control
        (each child widget's own signal is blocked individually - blocking
        `self`'s signals would do nothing, since `criteriaChanged` is only
        ever emitted from `_on_control_changed`, not by Qt automatically
        forwarding a blocked child's signal)."""
        controls = (
            self._search_edit, self._folder_combo, self._species_combo, self._user_decision_combo,
            self._algorithm_decision_combo, self._conflict_combo, self._reject_reason_combo,
            self._burst_rank_combo, self._burst_combo,
            *self._range_checks.values(), *self._range_mins.values(), *self._range_maxes.values(),
        )
        for control in controls:
            control.blockSignals(True)
        try:
            self._search_edit.clear()
            for combo in (
                self._folder_combo, self._species_combo, self._user_decision_combo,
                self._algorithm_decision_combo, self._conflict_combo, self._reject_reason_combo,
                self._burst_rank_combo,
            ):
                combo.setCurrentIndex(0)
            self._burst_combo.setCurrentIndex(0)
            for field, _label in _RANGE_ROWS:
                self._range_checks[field].setChecked(False)
                self._range_mins[field].setEnabled(False)
                self._range_maxes[field].setEnabled(False)
        finally:
            for control in controls:
                control.blockSignals(False)
        self.criteriaChanged.emit()
