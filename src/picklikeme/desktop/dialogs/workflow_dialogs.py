"""Small parameter dialogs for Rank by AI, Organize by Species, and Auto Crop."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from ...config import DEFAULT_CHECKPOINT_PATH


class RankDialog(QDialog):
    """Collects parameters for a "Rank by AI" run."""

    def __init__(self, *, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rank by AI")

        self._checkpoint_edit = QLineEdit(str(DEFAULT_CHECKPOINT_PATH), self)
        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._browse_checkpoint)
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self._checkpoint_edit, 1)
        checkpoint_row.addWidget(browse_btn)

        self._crop_birds_check = QCheckBox("Score bird crops (recommended)", self)
        self._crop_birds_check.setChecked(True)

        form = QFormLayout()
        form.addRow("Checkpoint:", checkpoint_row)
        form.addRow("", self._crop_birds_check)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose checkpoint", str(Path(self._checkpoint_edit.text()).parent), "Checkpoints (*.pt)")
        if path:
            self._checkpoint_edit.setText(path)

    def checkpoint_path(self) -> str:
        return self._checkpoint_edit.text().strip() or str(DEFAULT_CHECKPOINT_PATH)

    def crop_birds(self) -> bool:
        return self._crop_birds_check.isChecked()


class SpeciesLanguageDialog(QDialog):
    """Picks the language used for species subfolder names."""

    def __init__(self, *, default_language: str = "en", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Organize by Species")

        self._english_radio = QRadioButton("English species names", self)
        self._hebrew_radio = QRadioButton("Hebrew species names (falls back to English if untranslated)", self)
        if default_language == "he":
            self._hebrew_radio.setChecked(True)
        else:
            self._english_radio.setChecked(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._english_radio)
        layout.addWidget(self._hebrew_radio)
        layout.addWidget(buttons)

    def language(self) -> str:
        return "he" if self._hebrew_radio.isChecked() else "en"


class PreferencesDialog(QDialog):
    """Application preferences: theme and the default species-organization
    language. Both are settings that already exist and persist (theme.py /
    QSettings "review/species_language") - this just gives them one
    conventional home instead of a "not implemented yet" stub, and instead
    of only being reachable via the View menu or by running Organize by
    Species once."""

    def __init__(self, *, default_theme: str = "dark", default_language: str = "en", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")

        # Qt auto-groups sibling QRadioButtons that share the same parent
        # widget into one mutually-exclusive set, regardless of which
        # layout visually contains them - without these two explicit
        # QButtonGroups, checking "English" would silently uncheck "Dark"
        # (all four buttons are children of `self`, so they'd otherwise
        # all compete as a single group of four).
        self._dark_radio = QRadioButton("Dark", self)
        self._light_radio = QRadioButton("Light", self)
        theme_group = QButtonGroup(self)
        theme_group.addButton(self._dark_radio)
        theme_group.addButton(self._light_radio)
        if default_theme == "light":
            self._light_radio.setChecked(True)
        else:
            self._dark_radio.setChecked(True)
        theme_row = QHBoxLayout()
        theme_row.addWidget(self._dark_radio)
        theme_row.addWidget(self._light_radio)

        self._english_radio = QRadioButton("English", self)
        self._hebrew_radio = QRadioButton("Hebrew", self)
        language_group = QButtonGroup(self)
        language_group.addButton(self._english_radio)
        language_group.addButton(self._hebrew_radio)
        if default_language == "he":
            self._hebrew_radio.setChecked(True)
        else:
            self._english_radio.setChecked(True)
        language_row = QHBoxLayout()
        language_row.addWidget(self._english_radio)
        language_row.addWidget(self._hebrew_radio)

        form = QFormLayout()
        form.addRow("Theme:", theme_row)
        form.addRow("Default species language:", language_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def theme_name(self) -> str:
        return "light" if self._light_radio.isChecked() else "dark"

    def species_language(self) -> str:
        return "he" if self._hebrew_radio.isChecked() else "en"


class AutoCropDialog(QDialog):
    """Picks the crop margin percentage (preset or custom) for Auto Crop."""

    _PRESETS = (20.0, 30.0, 40.0)

    def __init__(self, *, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto Crop")

        self._preset_radios: list[QRadioButton] = []
        preset_row = QHBoxLayout()
        for preset in self._PRESETS:
            radio = QRadioButton(f"{preset:g}%", self)
            preset_row.addWidget(radio)
            self._preset_radios.append(radio)
        self._preset_radios[0].setChecked(True)

        self._custom_radio = QRadioButton("Custom:", self)
        self._custom_spin = QDoubleSpinBox(self)
        self._custom_spin.setRange(0.0, 90.0)
        self._custom_spin.setSuffix("%")
        self._custom_spin.setValue(20.0)
        custom_row = QHBoxLayout()
        custom_row.addWidget(self._custom_radio)
        custom_row.addWidget(self._custom_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(preset_row)
        layout.addLayout(custom_row)
        layout.addWidget(buttons)

    def margin_percent(self) -> float:
        if self._custom_radio.isChecked():
            return float(self._custom_spin.value())
        for preset, radio in zip(self._PRESETS, self._preset_radios):
            if radio.isChecked():
                return preset
        return self._PRESETS[0]
