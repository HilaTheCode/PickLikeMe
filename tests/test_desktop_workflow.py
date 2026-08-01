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

    def fake_arrange_by_species(input_folder, classifier, cache, *, on_progress=None, folder_name_fn=None):
        captured["folder_name_fn"] = folder_name_fn
        captured["input_folder"] = input_folder

        class Result:
            total = 1
            classified = 1
            moved = 1
            skipped = 0
            errors = 0
            species_counts = {"House Sparrow": 1}

        assert folder_name_fn("House Sparrow") == localized_species_name("House Sparrow", language="he")
        return Result()

    monkeypatch.setattr("picklikeme.species.classifier.build_classifier", fake_build_classifier)
    monkeypatch.setattr("picklikeme.species.cache.SpeciesCache", lambda *a, **k: FakeCache())
    monkeypatch.setattr("picklikeme.species.arrange.arrange_by_species", fake_arrange_by_species)

    result = service.organize_by_species(language="he")

    assert result["moved"] == 1
    assert result["species_counts"] == {"House Sparrow": 1}
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
