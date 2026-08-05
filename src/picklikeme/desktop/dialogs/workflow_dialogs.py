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
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ...config import DEFAULT_CHECKPOINT_PATH
from ...ranking import GROUP_WEIGHTS
from ...species.classifier import available_classifiers
from .dialog_utils import polish_dialog

# Phase 12 (General UX) - one shared margin/spacing convention for every
# dialog's top-level layout in this file, replacing Qt's bare defaults so
# spacing reads as a deliberate choice rather than whatever the platform
# style happens to default to.
_DIALOG_MARGIN = 12
_DIALOG_SPACING = 8


def _polish_layout(layout) -> None:
    layout.setContentsMargins(_DIALOG_MARGIN, _DIALOG_MARGIN, _DIALOG_MARGIN, _DIALOG_MARGIN)
    layout.setSpacing(_DIALOG_SPACING)


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
        _polish_layout(layout)
        layout.addLayout(form)
        layout.addWidget(buttons)
        polish_dialog(self, geometry_key="dialogs/rank_geometry")

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
        _polish_layout(layout)
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
        polish_dialog(self, geometry_key=f"dialogs/algorithm_parameters_{params_cls.__name__}_geometry")

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

    def __init__(
        self,
        *,
        default_language: str = "en",
        default_backend: str = "bioclip2",
        default_species_list_path: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Organize by Species")
        self.setMinimumWidth(420)

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

        # Species List: any external text file, one species per line - see
        # species.classifier.read_species_list. Never copied into the
        # project; the classifier reads it directly from wherever the
        # photographer keeps it (All_Birds.txt, Israel_Birds.txt, ...).
        self._species_list_path: str | None = None
        self._species_list_edit = QLineEdit(self)
        self._species_list_edit.setReadOnly(True)
        self._species_list_edit.setPlaceholderText("(built-in default species list)")
        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._browse_species_list)
        clear_btn = QPushButton("Clear", self)
        clear_btn.setToolTip("Go back to the built-in default species list")
        clear_btn.clicked.connect(self._clear_species_list)
        species_row = QHBoxLayout()
        species_row.addWidget(self._species_list_edit, 1)
        species_row.addWidget(browse_btn)
        species_row.addWidget(clear_btn)

        self._species_list_info = QLabel(self)
        self._species_list_info.setWordWrap(True)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        _polish_layout(layout)
        layout.addWidget(self._english_radio)
        layout.addWidget(self._hebrew_radio)
        layout.addWidget(QLabel("Classification backend:", self))
        layout.addLayout(backend_column)
        layout.addWidget(QLabel("Species List:", self))
        layout.addLayout(species_row)
        layout.addWidget(self._species_list_info)
        layout.addWidget(self._buttons)
        polish_dialog(self, geometry_key="dialogs/species_language_geometry")

        if default_species_list_path:
            self._set_species_list_path(default_species_list_path)

    def language(self) -> str:
        return "he" if self._hebrew_radio.isChecked() else "en"

    def backend(self) -> str:
        for classifier_id, radio in self._backend_radios.items():
            if radio.isChecked():
                return classifier_id
        return "bioclip2"  # unreachable in practice - one radio is always checked

    def species_list_path(self) -> str | None:
        """The chosen external species-list file, or None to use the
        classifier's own built-in default."""
        return self._species_list_path

    def _browse_species_list(self) -> None:
        start_dir = str(Path(self._species_list_path).parent) if self._species_list_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a species list", start_dir, "Text files (*.txt);;All files (*)"
        )
        if path:
            self._set_species_list_path(path)

    def _clear_species_list(self) -> None:
        self._species_list_path = None
        self._species_list_edit.clear()
        self._species_list_edit.setToolTip("")
        self._species_list_info.setText("")
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def _set_species_list_path(self, path: str) -> None:
        """Validates and previews `path` immediately (species count, or a
        clear error) - reading a text file is cheap, unlike constructing a
        classifier, so this happens the moment a file is chosen, not only
        when the dialog is accepted. An invalid file disables OK rather
        than being silently accepted and only failing later, mid-run."""
        from ...species.classifier import read_species_list
        from .. import theme

        self._species_list_path = path
        self._species_list_edit.setText(Path(path).name)
        self._species_list_edit.setToolTip(path)

        # Phase 12 (General UX): theme.current_palette()'s own reject_fg/
        # keep_fg, not a hardcoded hex - the same red/green the gallery
        # already uses, and one that stays legible if the palette (or the
        # active theme) ever changes.
        palette = theme.current_palette()
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        try:
            species = read_species_list(path)
        except OSError as exc:
            self._species_list_info.setText(f"Could not read this file: {exc}")
            self._species_list_info.setStyleSheet(f"color: {palette.reject_fg};")
            ok_button.setEnabled(False)
            return
        if not species:
            self._species_list_info.setText(
                "This file contains no valid species - every line is blank or a '#' comment."
            )
            self._species_list_info.setStyleSheet(f"color: {palette.reject_fg};")
            ok_button.setEnabled(False)
            return
        self._species_list_info.setText(f"{len(species)} species loaded")
        self._species_list_info.setStyleSheet(f"color: {palette.keep_fg};")
        ok_button.setEnabled(True)


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
        _polish_layout(layout)
        layout.addLayout(form)
        layout.addWidget(buttons)
        polish_dialog(self, geometry_key="dialogs/preferences_geometry")

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
        _polish_layout(layout)
        layout.addLayout(preset_row)
        layout.addLayout(custom_row)
        layout.addWidget(buttons)
        polish_dialog(self, geometry_key="dialogs/auto_crop_geometry")

    def margin_percent(self) -> float:
        if self._custom_radio.isChecked():
            return float(self._custom_spin.value())
        for preset, radio in zip(self._PRESETS, self._preset_radios):
            if radio.isChecked():
                return preset
        return self._PRESETS[0]


class SetUserDecisionsBySubfoldersDialog(QDialog):
    """"Set User Decisions by Subfolders..." - Version 2: Keep and Reject
    each accept MULTIPLE subfolders of the Root Folder (a photographer's
    own organisation is rarely a single folder - "Selected", "Favorites",
    "Portfolio" might all mean Keep). Neutral is never folder-selected here
    at all - see ground_truth.py's own module docstring for why one walk of
    the Root Folder is enough to infer it automatically.

    Collects parameters only, exactly like every other workflow dialog
    here - the actual preview/confirm/apply sequence runs afterward, in
    MainWindow, with a progress bar (walking+hashing thousands of images is
    not instant)."""

    def __init__(self, *, root_folder: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set User Decisions by Subfolders")
        self.setMinimumWidth(520)
        self._root_folder = root_folder

        intro = QLabel(
            "Scans the Root Folder below once and sets the User Decision for every image "
            "found - Keep or Reject for anything inside a folder selected below, Neutral "
            "automatically for everything else (no folder to pick for it). Images are "
            "matched by content, not just their path, so a copy or a rename still matches. "
            "Nothing is copied or moved - only the decision changes.",
            self,
        )
        intro.setWordWrap(True)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Root Folder:", self))
        root_label = QLabel(root_folder, self)
        root_label.setWordWrap(True)
        root_row.addWidget(root_label, 1)

        self._keep_list, keep_group = self._folder_list_group("Keep Folders")
        self._reject_list, reject_group = self._folder_list_group("Reject Folders")
        lists_row = QHBoxLayout()
        lists_row.addWidget(keep_group)
        lists_row.addWidget(reject_group)

        neutral_note = QLabel(
            "Neutral: every remaining image under the Root Folder that is not inside a "
            "Keep or Reject folder above - inferred automatically, nothing to select.",
            self,
        )
        neutral_note.setWordWrap(True)
        neutral_note.setStyleSheet("color: palette(mid);")

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Preview…")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        _polish_layout(layout)
        layout.addWidget(intro)
        layout.addLayout(root_row)
        layout.addLayout(lists_row)
        layout.addWidget(neutral_note)
        layout.addWidget(self._buttons)
        polish_dialog(self, geometry_key="dialogs/set_user_decisions_by_subfolders_geometry")

    def _folder_list_group(self, title: str) -> tuple[QListWidget, QWidget]:
        list_widget = QListWidget(self)
        list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        add_button = QPushButton("Add Folder…", self)
        add_button.clicked.connect(lambda: self._add_folder(list_widget))
        remove_button = QPushButton("Remove Selected", self)
        remove_button.clicked.connect(lambda: self._remove_selected(list_widget))
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(add_button)
        buttons_row.addWidget(remove_button)

        group = QWidget(self)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(QLabel(title, self))
        group_layout.addWidget(list_widget)
        group_layout.addLayout(buttons_row)
        return list_widget, group

    def _add_folder(self, list_widget: QListWidget) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder", self._root_folder)
        if not path:
            return
        existing = {list_widget.item(row).text() for row in range(list_widget.count())}
        if path not in existing:
            item = QListWidgetItem(path)
            item.setToolTip(path)
            list_widget.addItem(item)

    @staticmethod
    def _remove_selected(list_widget: QListWidget) -> None:
        for item in list_widget.selectedItems():
            list_widget.takeItem(list_widget.row(item))

    @staticmethod
    def _folders(list_widget: QListWidget) -> list[str]:
        return [list_widget.item(row).text() for row in range(list_widget.count())]

    def root_folder(self) -> str:
        return self._root_folder

    def keep_folders(self) -> list[str]:
        return self._folders(self._keep_list)

    def reject_folders(self) -> list[str]:
        return self._folders(self._reject_list)

    def neutral_folder(self) -> str | None:
        return self._neutral_edit.text() or None
