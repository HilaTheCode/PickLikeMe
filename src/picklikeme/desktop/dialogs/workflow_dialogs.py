"""Small parameter dialogs for the ranking strategies, Organize by Species, and Auto Crop."""

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
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from ...config import DEFAULT_CHECKPOINT_PATH
from ...ranking import GROUP_WEIGHTS
from ...species.classifier import available_classifiers


class RankDialog(QDialog):
    """Collects parameters for a "Rank by AI" run."""

    def __init__(self, *, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rank by AI")
        # The default checkpoint path is a full project-relative path
        # (~60 chars); without a minimum width the dialog shrinks to fit
        # its labels/buttons and the path field scrolls to the cursor,
        # showing only its tail ("...odel_checkpoint.pt") instead of the
        # full path.
        self.setMinimumWidth(480)

        self._checkpoint_edit = QLineEdit(str(DEFAULT_CHECKPOINT_PATH), self)
        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._browse_checkpoint)
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self._checkpoint_edit, 1)
        checkpoint_row.addWidget(browse_btn)

        self._crop_birds_check = QCheckBox("Score subject crops (recommended)", self)
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


class AlgorithmParametersDialog(QDialog):
    """Parameters for a ranking strategy, built from the strategy's own specs.

    Nothing here knows what "eye sharpness" means, or that there are three
    weights rather than five. It reads `ParamSpec`s (see `ranking.base`) and
    lays out one spin box per parameter, grouped into weights and everything
    else. Adding a parameter to a strategy therefore adds a field to this
    dialog with no change to this file - which is what makes this an
    *algorithm parameters* dialog rather than a weights dialog that would
    need rewriting the first time a non-weight knob appeared.

    Weights are deliberately not constrained to sum to 100: any set of
    numbers is valid and is normalised before scoring (see
    `WeightedParams.normalized_weights`), so a photographer can type 5/3/2 or
    50/30/20 and mean the same thing.
    """

    def __init__(self, *, params_cls, title: str, initial=None, parent=None) -> None:
        super().__init__(parent)
        self._params_cls = params_cls
        self._specs = tuple(params_cls.specs())
        self.setWindowTitle(title)
        self.setMinimumWidth(420)

        initial = initial or params_cls()
        self._spins: dict[str, QDoubleSpinBox] = {}

        weights_form = QFormLayout()
        other_form = QFormLayout()
        for spec in self._specs:
            spin = QDoubleSpinBox(self)
            spin.setRange(spec.minimum, spec.maximum)
            spin.setDecimals(spec.decimals)
            spin.setSingleStep(1.0 if spec.decimals == 0 else 10.0**-spec.decimals)
            if spec.suffix:
                spin.setSuffix(spec.suffix)
            if spec.help:
                spin.setToolTip(spec.help)
            spin.setValue(float(getattr(initial, spec.name)))
            self._spins[spec.name] = spin
            (weights_form if spec.group == GROUP_WEIGHTS else other_form).addRow(
                f"{spec.label}:", spin
            )

        layout = QVBoxLayout(self)
        if weights_form.rowCount():
            heading = QLabel("Weights (normalized automatically — any values are valid)", self)
            heading.setWordWrap(True)
            layout.addWidget(heading)
            layout.addLayout(weights_form)
        if other_form.rowCount():
            layout.addWidget(QLabel("Thresholds", self))
            layout.addLayout(other_form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText("Reset to Defaults")
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self.reset_to_defaults
        )
        layout.addWidget(buttons)

    def reset_to_defaults(self) -> None:
        """Every parameter back to the value its own spec declares."""
        for spec in self._specs:
            self._spins[spec.name].setValue(float(spec.default))

    def parameters(self):
        """The chosen values, as the strategy's own params dataclass."""
        return self._params_cls.from_values(
            {name: float(spin.value()) for name, spin in self._spins.items()}
        )


class SpeciesLanguageDialog(QDialog):
    """Picks the language used for species subfolder names, and which
    species-classification backend runs the pass - one registered entry per
    `species.classifier.available_classifiers()`, so a future third backend
    appears here with no change to this file, exactly like
    `AlgorithmParametersDialog` picks up a new ranking strategy's params."""

    def __init__(self, *, default_language: str = "en", default_backend: str = "bioclip2", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Organize by Species")

        self._english_radio = QRadioButton("English species names", self)
        self._hebrew_radio = QRadioButton("Hebrew species names (falls back to English if untranslated)", self)
        if default_language == "he":
            self._hebrew_radio.setChecked(True)
        else:
            self._english_radio.setChecked(True)

        self._backend_radios: dict[str, QRadioButton] = {}
        backend_group = QButtonGroup(self)
        backend_column = QVBoxLayout()
        for info in available_classifiers():
            radio = QRadioButton(info.display_name, self)
            radio.setToolTip(info.description)
            backend_group.addButton(radio)
            backend_column.addWidget(radio)
            self._backend_radios[info.classifier_id] = radio
        checked = self._backend_radios.get(default_backend) or next(iter(self._backend_radios.values()))
        checked.setChecked(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._english_radio)
        layout.addWidget(self._hebrew_radio)
        layout.addWidget(QLabel("Classification backend:", self))
        layout.addLayout(backend_column)
        layout.addWidget(buttons)

    def language(self) -> str:
        return "he" if self._hebrew_radio.isChecked() else "en"

    def backend(self) -> str:
        for classifier_id, radio in self._backend_radios.items():
            if radio.isChecked():
                return classifier_id
        return "bioclip2"  # unreachable in practice - one radio is always checked


class PreferencesDialog(QDialog):
    """Application preferences: theme, the default species-organization
    language, and the default species-classification backend. All three are
    settings that already exist and persist (theme.py / QSettings
    "review/species_language" / "review/species_backend") - this just gives
    them one conventional home instead of a "not implemented yet" stub, and
    instead of only being reachable via the View menu or by running Organize
    by Species once."""

    def __init__(
        self,
        *,
        default_theme: str = "dark",
        default_language: str = "en",
        default_backend: str = "bioclip2",
        parent=None,
    ) -> None:
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

        self._backend_radios: dict[str, QRadioButton] = {}
        backend_group = QButtonGroup(self)
        backend_row = QHBoxLayout()
        for info in available_classifiers():
            radio = QRadioButton(info.display_name, self)
            radio.setToolTip(info.description)
            backend_group.addButton(radio)
            backend_row.addWidget(radio)
            self._backend_radios[info.classifier_id] = radio
        checked = self._backend_radios.get(default_backend) or next(iter(self._backend_radios.values()))
        checked.setChecked(True)

        form = QFormLayout()
        form.addRow("Theme:", theme_row)
        form.addRow("Default species language:", language_row)
        form.addRow("Default species backend:", backend_row)

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

    def species_backend(self) -> str:
        for classifier_id, radio in self._backend_radios.items():
            if radio.isChecked():
                return classifier_id
        return "bioclip2"  # unreachable in practice - one radio is always checked


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
