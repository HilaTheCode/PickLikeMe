import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image

from picklikeme.desktop.application import ApplicationState, WorkerManager
from picklikeme.desktop.services import ReviewService
from picklikeme.desktop.settings import DesktopSettings
from picklikeme.species.translations import localized_species_name

try:
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]
    QCoreApplication = None  # type: ignore[assignment]
    QPixmap = None  # type: ignore[assignment]


def _make_jpeg(path) -> None:
    Image.new("RGB", (16, 16), color="blue").save(path, format="JPEG")


# ---------------------------------------------------------------------------
# ReviewService workflow methods
# ---------------------------------------------------------------------------


def test_import_selected_copies_files_and_updates_state(tmp_path) -> None:
    source = tmp_path / "card"
    selected = source / "_Selected"
    selected.mkdir(parents=True)
    _make_jpeg(selected / "a.jpg")
    _make_jpeg(selected / "b.jpg")

    destination = tmp_path / "library"
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(source)

    result = service.import_selected(destination)

    assert result["copied"] == 2
    assert (destination / "a.jpg").is_file()
    assert (destination / "b.jpg").is_file()
    assert "state" in result
    service.close()


def test_import_selected_requires_open_folder(tmp_path) -> None:
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    with pytest.raises(ValueError):
        service.import_selected(tmp_path / "dest")
    service.close()


def test_preview_ground_truth_import_reports_counts_without_writing(tmp_path) -> None:
    root = tmp_path / "Shoot"
    keep_folder = root / "Selected"
    keep_folder.mkdir(parents=True)
    _make_jpeg(keep_folder / "a.jpg")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    result = service.preview_ground_truth_import(root_folder=root, keep_folders=[keep_folder])

    assert result["totals"]["keep"] == 1
    assert result["totals"]["will_change"] == 1
    assert result["keep"]["will_change"] == 1
    assert service.store.review_decision_count() == 0  # preview never writes
    service.close()


def test_preview_ground_truth_import_infers_neutral_with_no_folder_selection(tmp_path) -> None:
    """Version 2 workflow: Neutral is never folder-selected - everything
    under root_folder not inside a selected Keep/Reject subfolder is
    automatically Neutral."""
    root = tmp_path / "Shoot"
    keep_folder = root / "Selected"
    keep_folder.mkdir(parents=True)
    _make_jpeg(keep_folder / "a.jpg")
    _make_jpeg(root / "b.jpg")  # not under any selected folder

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    result = service.preview_ground_truth_import(root_folder=root, keep_folders=[keep_folder])

    assert result["totals"]["keep"] == 1
    assert result["totals"]["neutral"] == 1
    service.close()


def test_apply_ground_truth_import_writes_the_previewed_plan_and_refreshes_state(tmp_path) -> None:
    from picklikeme.analyzer.annotations import REVIEW_KEEP

    root = tmp_path / "Shoot"
    keep_folder = root / "Selected"
    keep_folder.mkdir(parents=True)
    _make_jpeg(keep_folder / "a.jpg")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.preview_ground_truth_import(root_folder=root, keep_folders=[keep_folder])
    result = service.apply_ground_truth_import()

    assert result["updated_keep"] == 1
    assert result["skipped"] == []
    assert result["conflicts"] == []
    assert "state" in result
    decisions = {row["image_path"]: row["decision"] for row in service.store.review_decisions()}
    assert decisions[str(keep_folder / "a.jpg")] == REVIEW_KEEP
    service.close()


def test_apply_ground_truth_import_requires_a_preview_first(tmp_path) -> None:
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    with pytest.raises(ValueError):
        service.apply_ground_truth_import()
    service.close()


def test_rank_folder_delegates_and_refreshes_session(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    calls = {}

    def fake_rank_folder(input_folder, **kwargs):
        calls["input_folder"] = input_folder
        calls["kwargs"] = kwargs
        return {"output_csv": "ranking.csv", "image_count": 1, "top": []}

    # Patched at picklikeme.rank, the module that actually owns the AI ranking
    # entry point: ReviewService now dispatches through the ranking-strategy
    # registry, and AIModelStrategy imports rank_folder from there. Same
    # function, same arguments - only the seam this test observes moved.
    monkeypatch.setattr("picklikeme.rank.rank_folder", fake_rank_folder)

    result = service.rank_folder(checkpoint="dummy.pt", crop_birds=False)

    assert calls["input_folder"] == service.session.input_folder
    assert calls["kwargs"]["checkpoint"] == "dummy.pt"
    assert calls["kwargs"]["crop_birds"] is False
    assert "state" in result
    # The AI model scores everything it is given, so a strategy-aware caller
    # can rely on it never reporting filtered-out images.
    assert result["strategy"] == "ai-model"
    assert result["filtered"] == {}
    service.close()


def test_rank_folder_requires_open_folder(tmp_path) -> None:
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    with pytest.raises(ValueError):
        service.rank_folder()
    service.close()


def test_auto_crop_delegates_to_backend(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    calls = {}

    def fake_generate_lightroom_crops(input_folder, **kwargs):
        calls["input_folder"] = input_folder
        calls["margin_percent"] = kwargs.get("margin_percent")
        return {"processed": 1, "message": "ok", "stats": {}, "details": []}

    monkeypatch.setattr("picklikeme.desktop.services.generate_lightroom_crops", fake_generate_lightroom_crops)

    result = service.auto_crop(margin_percent=30.0)

    assert calls["margin_percent"] == 30.0
    assert result["processed"] == 1
    service.close()


def test_organize_by_species_uses_language_transform(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    class FakeCache:
        def close(self) -> None:
            pass

    captured = {}

    def fake_build_classifier(name, **kwargs):
        return object()

    def fake_run_with_analytics(
        input_folder, classifier, backend, cache, *, on_progress=None, folder_name_fn=None,
        species_list_path=None, analytics_db=None,
    ):
        captured["folder_name_fn"] = folder_name_fn
        captured["input_folder"] = input_folder
        captured["backend"] = backend

        class Result:
            total = 1
            classified = 1
            moved = 1
            skipped = 0
            errors = 0
            species_counts = {"House Sparrow": 1}

        assert folder_name_fn("House Sparrow") == localized_species_name("House Sparrow", language="he")
        return Result(), "fake-run-id", object()

    monkeypatch.setattr("picklikeme.species.classifier.build_classifier", fake_build_classifier)
    monkeypatch.setattr("picklikeme.species.cache.SpeciesCache", lambda *a, **k: FakeCache())
    monkeypatch.setattr("picklikeme.species.experiment_capture.run_with_analytics", fake_run_with_analytics)

    result = service.organize_by_species(language="he")

    assert result["moved"] == 1
    assert result["species_counts"] == {"House Sparrow": 1}
    assert result["experiment_id"] == "fake-run-id"
    assert captured["backend"] == "bioclip2"
    service.close()


def test_organize_by_species_forwards_species_list_path(tmp_path, monkeypatch) -> None:
    """An external species list must reach BOTH the classifier construction
    (which actually reads the file) and the analytics experiment metadata
    (species_list_filename/species_count/species_list_hash) - see
    species.experiment.build_experiment_metadata."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")
    species_list = tmp_path / "Israel_Birds.txt"
    species_list.write_text("Kingfisher\nEuropean Bee-eater\n", encoding="utf-8")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    class FakeCache:
        def close(self) -> None:
            pass

    captured = {}

    def fake_build_classifier(name, **kwargs):
        captured["build_classifier_species_list_path"] = kwargs.get("species_list_path")
        return object()

    def fake_run_with_analytics(
        input_folder, classifier, backend, cache, *, on_progress=None, folder_name_fn=None,
        species_list_path=None, analytics_db=None,
    ):
        captured["run_with_analytics_species_list_path"] = species_list_path

        class Result:
            total = 1
            classified = 1
            moved = 1
            skipped = 0
            errors = 0
            species_counts = {}

        return Result(), "fake-run-id", object()

    monkeypatch.setattr("picklikeme.species.classifier.build_classifier", fake_build_classifier)
    monkeypatch.setattr("picklikeme.species.cache.SpeciesCache", lambda *a, **k: FakeCache())
    monkeypatch.setattr("picklikeme.species.experiment_capture.run_with_analytics", fake_run_with_analytics)

    service.organize_by_species(species_list_path=str(species_list))

    assert captured["build_classifier_species_list_path"] == str(species_list)
    assert captured["run_with_analytics_species_list_path"] == str(species_list)
    service.close()


def test_organize_by_species_with_no_species_list_path_passes_none(tmp_path, monkeypatch) -> None:
    """The built-in-default case must reach the classifier as None, never
    an empty string - BioClipSpeciesClassifier treats `species_list_path
    is not None` as "an explicit file was chosen"."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    class FakeCache:
        def close(self) -> None:
            pass

    captured = {}

    def fake_build_classifier(name, **kwargs):
        captured["species_list_path"] = kwargs.get("species_list_path")
        return object()

    def fake_run_with_analytics(input_folder, classifier, backend, cache, **kwargs):
        class Result:
            total = 0
            classified = 0
            moved = 0
            skipped = 0
            errors = 0
            species_counts = {}

        return Result(), None, object()

    monkeypatch.setattr("picklikeme.species.classifier.build_classifier", fake_build_classifier)
    monkeypatch.setattr("picklikeme.species.cache.SpeciesCache", lambda *a, **k: FakeCache())
    monkeypatch.setattr("picklikeme.species.experiment_capture.run_with_analytics", fake_run_with_analytics)

    service.organize_by_species()

    assert captured["species_list_path"] is None
    service.close()


def test_organize_by_species_requires_open_folder(tmp_path) -> None:
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    with pytest.raises(ValueError):
        service.organize_by_species()
    service.close()


def test_set_keep_percent_updates_session(tmp_path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    _make_jpeg(folder / "a.jpg")
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    state = service.set_keep_percent(35.0)

    assert service.session.keep_percent == 35.0
    assert "images" in state
    service.close()


def test_save_jpeg_writes_file(tmp_path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    image_path = folder / "a.jpg"
    _make_jpeg(image_path)
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    destination = tmp_path / "share.jpg"
    result = service.save_jpeg(str(image_path), destination)

    assert result == destination
    assert destination.is_file()
    service.close()


# ---------------------------------------------------------------------------
# Species translation helper
# ---------------------------------------------------------------------------


def test_localized_species_name_falls_back_to_english() -> None:
    assert localized_species_name("Totally Unknown Bird", language="he") == "Totally Unknown Bird"
    assert localized_species_name("House Sparrow", language="he") == "דרור הבית"
    assert localized_species_name("House Sparrow", language="en") == "House Sparrow"


# ---------------------------------------------------------------------------
# Desktop dialogs (Qt required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_rank_dialog_returns_selected_values() -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import RankDialog

    app = QApplication.instance() or QApplication([])
    dialog = RankDialog()
    dialog._checkpoint_edit.setText("my_checkpoint.pt")
    dialog._crop_birds_check.setChecked(False)

    assert dialog.checkpoint_path() == "my_checkpoint.pt"
    assert dialog.crop_birds() is False
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_ground_truth_dialog_shows_root_and_starts_with_empty_keep_reject_lists(tmp_path) -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import SetUserDecisionsBySubfoldersDialog

    app = QApplication.instance() or QApplication([])
    root = tmp_path / "Shoot"
    dialog = SetUserDecisionsBySubfoldersDialog(root_folder=str(root))

    assert dialog.root_folder() == str(root)
    assert dialog.keep_folders() == []
    assert dialog.reject_folders() == []
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_ground_truth_dialog_add_and_remove_multiple_keep_and_reject_folders(tmp_path) -> None:
    from PySide6.QtWidgets import QListWidgetItem

    from picklikeme.desktop.dialogs.workflow_dialogs import SetUserDecisionsBySubfoldersDialog

    app = QApplication.instance() or QApplication([])
    root = tmp_path / "Shoot"
    favorites = root / "Favorites"
    portfolio = root / "Portfolio"
    rejected = root / "Rejected"
    for folder in (favorites, portfolio, rejected):
        folder.mkdir(parents=True)

    dialog = SetUserDecisionsBySubfoldersDialog(root_folder=str(root))

    dialog._keep_list.addItem(QListWidgetItem(str(favorites)))
    dialog._keep_list.addItem(QListWidgetItem(str(portfolio)))
    dialog._reject_list.addItem(QListWidgetItem(str(rejected)))

    assert dialog.keep_folders() == [str(favorites), str(portfolio)]
    assert dialog.reject_folders() == [str(rejected)]

    dialog._keep_list.item(0).setSelected(True)
    dialog._remove_selected(dialog._keep_list)

    assert dialog.keep_folders() == [str(portfolio)]
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_ground_truth_dialog_add_folder_browses_and_deduplicates(tmp_path, monkeypatch) -> None:
    from picklikeme.desktop.dialogs import workflow_dialogs as workflow_dialogs_module
    from picklikeme.desktop.dialogs.workflow_dialogs import SetUserDecisionsBySubfoldersDialog

    app = QApplication.instance() or QApplication([])
    root = tmp_path / "Shoot"
    favorites = root / "Favorites"
    favorites.mkdir(parents=True)

    monkeypatch.setattr(
        workflow_dialogs_module.QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: str(favorites)),
    )

    dialog = SetUserDecisionsBySubfoldersDialog(root_folder=str(root))
    dialog._add_folder(dialog._keep_list)  # real Browse… flow, via the monkeypatched file dialog
    dialog._add_folder(dialog._keep_list)  # picking the same folder again must not duplicate it

    assert dialog.keep_folders() == [str(favorites)]
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_species_language_dialog_defaults_and_selection() -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import SpeciesLanguageDialog

    app = QApplication.instance() or QApplication([])
    dialog = SpeciesLanguageDialog(default_language="he")

    assert dialog.language() == "he"
    dialog._english_radio.setChecked(True)
    assert dialog.language() == "en"
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_species_language_dialog_defaults_to_no_species_list(tmp_path) -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import SpeciesLanguageDialog

    app = QApplication.instance() or QApplication([])
    dialog = SpeciesLanguageDialog()
    assert dialog.species_list_path() is None
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_species_language_dialog_preselects_and_validates_a_species_list(tmp_path) -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import SpeciesLanguageDialog

    app = QApplication.instance() or QApplication([])
    species_list = tmp_path / "Israel_Birds.txt"
    species_list.write_text("Kingfisher\nEuropean Bee-eater\nBlack Kite\n", encoding="utf-8")

    dialog = SpeciesLanguageDialog(default_species_list_path=str(species_list))

    assert dialog.species_list_path() == str(species_list)
    assert "3 species loaded" in dialog._species_list_info.text()
    assert dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled()
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_species_language_dialog_shows_a_clear_error_for_an_empty_file(tmp_path) -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import SpeciesLanguageDialog

    app = QApplication.instance() or QApplication([])
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("# nothing here\n\n", encoding="utf-8")

    dialog = SpeciesLanguageDialog()
    dialog._set_species_list_path(str(empty_file))

    assert "no valid species" in dialog._species_list_info.text()
    assert not dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled()
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_species_language_dialog_shows_a_clear_error_for_a_missing_file(tmp_path) -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import SpeciesLanguageDialog

    app = QApplication.instance() or QApplication([])
    dialog = SpeciesLanguageDialog()
    dialog._set_species_list_path(str(tmp_path / "does_not_exist.txt"))

    assert "Could not read this file" in dialog._species_list_info.text()
    assert not dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled()
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_species_language_dialog_clear_button_reverts_to_built_in(tmp_path) -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import SpeciesLanguageDialog

    app = QApplication.instance() or QApplication([])
    species_list = tmp_path / "species.txt"
    species_list.write_text("Kingfisher\n", encoding="utf-8")
    dialog = SpeciesLanguageDialog(default_species_list_path=str(species_list))

    dialog._clear_species_list()

    assert dialog.species_list_path() is None
    assert dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled()
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_auto_crop_dialog_presets_and_custom() -> None:
    from picklikeme.desktop.dialogs.workflow_dialogs import AutoCropDialog

    app = QApplication.instance() or QApplication([])
    dialog = AutoCropDialog()

    assert dialog.margin_percent() == 20.0

    dialog._custom_radio.setChecked(True)
    dialog._custom_spin.setValue(37.5)
    assert dialog.margin_percent() == 37.5
    dialog.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_loupe_dialog_keep_reject_and_navigation(tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "shoot"
    folder.mkdir()
    path_a = folder / "a.jpg"
    path_b = folder / "b.jpg"
    _make_jpeg(path_a)
    _make_jpeg(path_b)

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    dialog = LoupeDialog(service=service, image_paths=[str(path_a), str(path_b)], start_index=0)
    dialog._apply_status("keep")

    assert dialog.index == 1
    images = {img["image_path"]: img for img in service.load_session()["images"]}
    assert images[str(path_a.resolve())]["review_status"] == "keep"

    dialog._go_prev()
    assert dialog.index == 0

    dialog.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_loupe_overlay_boxes_are_drawn_at_the_thickened_pen_widths(tmp_path) -> None:
    """The Loupe's own vector overlay (a QGraphicsScene, separate code from
    the Gallery's raster one - see contactsheets.annotate_thumbnail's own
    thickness test) needed its pen widths multiplied by the same ~5x the
    overlay-as-primary-debugging-tool pass applied there. Untested until
    now - the first test to actually inspect what set_detection_overlay
    draws rather than just that it doesn't crash."""
    from PySide6.QtWidgets import QGraphicsRectItem

    from picklikeme.desktop.dialogs.loupe_dialog import (
        BOX_PEN_WIDTH_EYE,
        BOX_PEN_WIDTH_OTHER,
        BOX_PEN_WIDTH_SELECTED,
        LoupeDialog,
    )

    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "shoot"
    folder.mkdir()
    path_a = folder / "a.jpg"
    _make_jpeg(path_a)

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    dialog = LoupeDialog(service=service, image_paths=[str(path_a)], start_index=0)
    view = dialog._view
    view.set_pixmap(QPixmap(100, 100))
    boxes = {
        "source_size": (100, 100),
        "selected": {"box": (10.0, 10.0, 40.0, 40.0)},
        "others": [{"box": (50.0, 50.0, 70.0, 70.0)}],
    }
    eye = {"source_size": (100, 100), "accepted": True, "box": (5.0, 5.0, 15.0, 15.0), "left": None, "right": None}

    view.set_detection_overlay(boxes, eye)

    rects_by_z = {item.zValue(): item for item in view._overlay_items if isinstance(item, QGraphicsRectItem)}
    assert rects_by_z[10].pen().widthF() == BOX_PEN_WIDTH_OTHER  # runner-up box
    assert rects_by_z[11].pen().widthF() == BOX_PEN_WIDTH_SELECTED  # the chosen subject box
    assert rects_by_z[12].pen().widthF() == BOX_PEN_WIDTH_EYE  # the eye box
    # ~5x the pre-existing 4/3px widths, not merely "thicker than zero".
    assert BOX_PEN_WIDTH_SELECTED >= 15
    assert BOX_PEN_WIDTH_EYE >= 15

    dialog.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_loupe_dialog_save_jpeg(tmp_path, monkeypatch) -> None:
    from picklikeme.desktop.dialogs import loupe_dialog as loupe_dialog_module

    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "shoot"
    folder.mkdir()
    path_a = folder / "a.jpg"
    _make_jpeg(path_a)

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    dialog = loupe_dialog_module.LoupeDialog(service=service, image_paths=[str(path_a)], start_index=0)

    destination = tmp_path / "shared.jpg"
    monkeypatch.setattr(
        loupe_dialog_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(destination), "")),
    )
    monkeypatch.setattr(loupe_dialog_module.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(loupe_dialog_module.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    dialog._save_jpeg()

    assert destination.is_file()
    dialog.close()
    service.close()
    app.quit()


# ---------------------------------------------------------------------------
# Loupe burst navigation order + burst info display
#
# Burst ORDER is decided entirely by MainWindow (see BURST_SORT_* and the
# View menu's "Burst Order" submenu in main_window.py, and test_burst_ui.py's
# own coverage of that decision) before a LoupeDialog is ever constructed -
# the Loupe itself just navigates exactly the `items`/`image_paths` it was
# handed, in that exact order, with no sort control and no re-sorting. These
# tests exercise LoupeDialog directly (as this file already does throughout)
# to prove that half of the contract: given an already-ordered burst, does
# it display/navigate/report on it correctly, without ever reordering it
# itself.
# ---------------------------------------------------------------------------


def _make_burst_items(folder, *, order: tuple[str, str, str] = ("a", "b", "c")) -> list:
    """Three members of one burst: burst_rank (score-descending, what
    burst_analysis.analyze_bursts already produces) deliberately disagrees
    with captured_at order, so a test can tell whichever order it was
    actually given apart from the other. `order` controls the order the
    returned list itself is in - the Loupe navigates exactly that list, so
    callers pass the sequence they want to verify is preserved."""
    from picklikeme.desktop.models.image_item import ImageItem

    by_name = {
        "a": ImageItem(
            path=str(folder / "a.jpg"), file_name="a.jpg", captured_at="2026-01-01T10:00:02",
            burst_id="burst-0018", burst_size=3, burst_rank=2, burst_best=False,
            ranking_results={"ai-model": {"score": 0.700, "rank": 2}},
        ),
        "b": ImageItem(
            path=str(folder / "b.jpg"), file_name="b.jpg", captured_at="2026-01-01T10:00:00",
            burst_id="burst-0018", burst_size=3, burst_rank=1, burst_best=True,
            ranking_results={"ai-model": {"score": 0.948, "rank": 1}},
        ),
        "c": ImageItem(
            path=str(folder / "c.jpg"), file_name="c.jpg", captured_at="2026-01-01T10:00:01",
            burst_id="burst-0018", burst_size=3, burst_rank=3, burst_best=False,
            ranking_results={"ai-model": {"score": 0.512, "rank": 3}},
        ),
    }
    for name in ("a", "b", "c"):
        _make_jpeg(folder / f"{name}.jpg")
    return [by_name[name] for name in order]


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_loupe_navigates_burst_members_in_exactly_the_order_it_was_given(tmp_path) -> None:
    """The core contract: whatever order the caller (MainWindow) sorted the
    burst into - Score order, Capture Time order, or anything else - the
    Loupe uses it verbatim. Checked against two different orderings of the
    SAME three members to prove the Loupe has no sort preference of its own
    baked in; there is also no sort combo to switch between them."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    app = QApplication.instance() or QApplication([])
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")

    score_order_folder = tmp_path / "score_order"
    score_order_folder.mkdir()
    score_items = _make_burst_items(score_order_folder, order=("b", "a", "c"))  # burst_rank order
    dialog = LoupeDialog(
        service=service, image_paths=[i.path for i in score_items], items=list(score_items),
        start_index=0, burst_scoped=True,
    )
    assert [i.file_name for i in dialog.items] == ["b.jpg", "a.jpg", "c.jpg"]
    assert not hasattr(dialog, "_burst_sort_combo")
    dialog.close()

    capture_time_folder = tmp_path / "capture_time_order"
    capture_time_folder.mkdir()
    capture_items = _make_burst_items(capture_time_folder, order=("a", "b", "c"))  # captured_at order
    dialog2 = LoupeDialog(
        service=service, image_paths=[i.path for i in capture_items], items=list(capture_items),
        start_index=0, burst_scoped=True,
    )
    assert [i.file_name for i in dialog2.items] == ["a.jpg", "b.jpg", "c.jpg"]
    dialog2.close()

    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_loupe_burst_info_labels_show_id_rank_best_and_score(tmp_path) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "shoot"
    folder.mkdir()
    # Handed in already in burst-score order (b, a, c) - MainWindow's own
    # job (see test_burst_ui.py), not the Loupe's.
    items = _make_burst_items(folder, order=("b", "a", "c"))

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    dialog = LoupeDialog(
        service=service, image_paths=[i.path for i in items], items=list(items),
        start_index=0, burst_scoped=True,
    )

    # Position 0 is "b.jpg", rank #1 of 3, best, score 0.948.
    assert dialog._burst_id_label.text() == "Burst 18"
    assert dialog._burst_rank_label.text() == "Burst Rank #1 of 3"
    assert dialog._burst_best_label.text() == "Best Image: Yes"
    assert dialog._burst_score_label.text() == "Score 94.8"

    dialog._go_next()
    assert dialog._burst_rank_label.text() == "Burst Rank #2 of 3"
    assert dialog._burst_best_label.text() == "Best Image: No"

    dialog.close()
    service.close()
    app.quit()


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_loupe_non_burst_session_shows_no_burst_sort_ui(tmp_path) -> None:
    """A Loupe session opened outside Collapse Bursts (burst_scoped=False,
    the default) must not grow any burst info row - every ImageItem still
    carries burst_id/burst_rank ("a burst of one"), so this only works if
    the UI keys off the explicit flag, not inferred data. There is no sort
    combo at all any more, burst-scoped or not."""
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "shoot"
    folder.mkdir()
    path_a = folder / "a.jpg"
    _make_jpeg(path_a)

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    dialog = LoupeDialog(service=service, image_paths=[str(path_a)], start_index=0)

    assert not hasattr(dialog, "_burst_sort_combo")
    assert dialog._burst_id_label.text() == ""

    dialog.close()
    service.close()
    app.quit()


# ---------------------------------------------------------------------------
# Background progress worker
# ---------------------------------------------------------------------------


@pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")
def test_run_in_background_reports_progress_and_result() -> None:
    from picklikeme.desktop.core.jobs import run_in_background

    app = QApplication.instance() or QApplication([])
    events: dict[str, object] = {}

    def work(on_progress=None, on_stage=None):
        if on_stage is not None:
            on_stage("working")
        if on_progress is not None:
            on_progress(1, 2)
            on_progress(2, 2)
        return "done"

    def on_finished(result):
        events["result"] = result

    thread = run_in_background(
        None,
        work,
        on_progress=lambda done, total: events.setdefault("progress", []).append((done, total)),
        on_stage=lambda message: events.setdefault("stage", message),
        on_finished=on_finished,
    )

    for _ in range(200):
        QCoreApplication.processEvents()
        if "result" in events:
            break
        thread.wait(10)

    assert events.get("result") == "done"
    assert events.get("stage") == "working"
    assert events.get("progress") == [(1, 2), (2, 2)]
