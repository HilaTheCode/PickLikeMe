"""The desktop side of the ranking framework: the Rank menu and the generated
Algorithm Parameters dialog.

The point of both is that they are *derived*, not written per strategy - the
menu from the registry, the dialog from the strategy's own `ParamSpec`s. So
these tests mostly check that derivation, plus the one strategy-specific
branch that survives (the AI model's hand-written RankDialog).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from picklikeme.ranking import ClassicVisionParams, available_strategies
from picklikeme.ranking.base import GROUP_THRESHOLDS, GROUP_WEIGHTS, ParamSpec, WeightedParams

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _dialog(initial=None):
    from picklikeme.desktop.dialogs.workflow_dialogs import AlgorithmParametersDialog

    return AlgorithmParametersDialog(
        params_cls=ClassicVisionParams, title="Classic Vision — Parameters", initial=initial
    )


def test_the_dialog_builds_one_field_per_declared_parameter(app) -> None:
    dialog = _dialog()
    assert set(dialog._spins) == {spec.name for spec in ClassicVisionParams.specs()}
    dialog.close()


def test_the_dialog_opens_on_the_defaults_and_returns_them_unchanged(app) -> None:
    dialog = _dialog()
    assert dialog.parameters() == ClassicVisionParams()
    dialog.close()


def test_editing_a_field_is_reflected_in_the_returned_parameters(app) -> None:
    dialog = _dialog()
    dialog._spins["eye_sharpness_weight"].setValue(80.0)
    dialog._spins["subject_size_weight"].setValue(5.0)

    params = dialog.parameters()
    assert params.eye_sharpness_weight == 80.0
    assert params.subject_size_weight == 5.0
    # The weights are normalised at scoring time, not clamped in the dialog -
    # 80/30/5 is a perfectly valid thing to type.
    assert sum(params.normalized_weights().values()) == pytest.approx(1.0)
    dialog.close()


def test_reset_to_defaults_restores_50_30_20(app) -> None:
    dialog = _dialog()
    for name in dialog._spins:
        dialog._spins[name].setValue(1.0)
    dialog.reset_to_defaults()

    assert dialog.parameters() == ClassicVisionParams()
    assert dialog.parameters().normalized_weights() == pytest.approx(
        {
            "eye_sharpness_weight": 0.5,
            "subject_sharpness_weight": 0.3,
            "subject_size_weight": 0.2,
        }
    )
    dialog.close()


def test_the_dialog_reopens_on_the_previously_used_values(app) -> None:
    """Re-running a strategy with a tweak starts from what was used last."""
    previous = ClassicVisionParams(70, 20, 10, 0.45)
    dialog = _dialog(initial=previous)
    assert dialog.parameters() == previous
    dialog.close()


def test_a_threshold_keeps_its_declared_precision(app) -> None:
    """min_eye_confidence is a 0-1 value declared with 2 decimals; a spin box
    rounding it to whole numbers would make it unusable."""
    dialog = _dialog()
    dialog._spins["min_eye_confidence"].setValue(0.45)
    assert dialog.parameters().min_eye_confidence == pytest.approx(0.45)
    dialog.close()


def test_the_dialog_is_generated_so_a_new_parameter_needs_no_ui_change(app) -> None:
    """The extensibility claim, exercised: a params class this file has never
    seen gets a working dialog with no code here changing."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _FutureParams(WeightedParams):
        composition_weight: float = 40.0
        noise_penalty: float = 60.0
        max_iso: float = 6400.0

        @classmethod
        def specs(cls):
            return (
                ParamSpec("composition_weight", "Composition", 40.0, 0.0, 100.0, GROUP_WEIGHTS),
                ParamSpec("noise_penalty", "Noise penalty", 60.0, 0.0, 100.0, GROUP_WEIGHTS),
                ParamSpec("max_iso", "Max ISO", 6400.0, 100.0, 102400.0, GROUP_THRESHOLDS),
            )

    from picklikeme.desktop.dialogs.workflow_dialogs import AlgorithmParametersDialog

    dialog = AlgorithmParametersDialog(params_cls=_FutureParams, title="Future — Parameters")
    assert set(dialog._spins) == {"composition_weight", "noise_penalty", "max_iso"}
    dialog._spins["composition_weight"].setValue(10.0)
    params = dialog.parameters()
    assert params.composition_weight == 10.0
    assert params.normalized_weights() == pytest.approx(
        {"composition_weight": 0.142857, "noise_penalty": 0.857143}, rel=1e-4
    )
    # The threshold is not folded into the weight normalisation.
    assert "max_iso" not in params.normalized_weights()
    dialog.close()


def test_the_rank_menu_offers_every_registered_strategy(app, tmp_path) -> None:
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(
        state=ApplicationState(),
        settings=DesktopSettings(),
        service=service,
        worker_manager=WorkerManager(),
    )
    try:
        labels = [action.text() for action in window._rank_menu.actions()]
        for info in available_strategies():
            assert any(info.display_name in label for label in labels), info.display_name
        assert set(window._rank_strategy_actions) == {i.strategy_id for i in available_strategies()}
    finally:
        window.close()
        service.close()


def test_the_toolbar_rank_button_is_a_split_button_with_a_dropdown(app, tmp_path) -> None:
    """Single click runs the default strategy; the arrow offers the others.

    Asserted through the style option rather than `QToolButton.menu()`, which
    returns None here by design: QToolBar builds the button with
    setDefaultAction(), and the button paints/popups the *action's* menu while
    `menu()` only ever reports an explicitly-set one.
    """
    from PySide6.QtWidgets import QStyleOptionToolButton, QToolBar, QToolButton

    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(
        state=ApplicationState(),
        settings=DesktopSettings(),
        service=service,
        worker_manager=WorkerManager(),
    )
    try:
        window.initialize()
        button = window.findChild(QToolBar, "main_toolbar").widgetForAction(window._rank_action)
        assert isinstance(button, QToolButton)
        assert button.popupMode() == QToolButton.ToolButtonPopupMode.MenuButtonPopup

        option = QStyleOptionToolButton()
        button.initStyleOption(option)
        features = QStyleOptionToolButton.ToolButtonFeature
        assert option.features & features.HasMenu, "no dropdown arrow would be drawn"
        assert option.features & features.MenuButtonPopup
        assert window._rank_action.menu() is window._rank_menu
    finally:
        window.close()
        service.close()


def test_the_gallery_card_shows_one_row_per_analysis_module(app) -> None:
    """Both scores side by side, neither overwriting the other, driven by the
    data rather than by naming the two modules that exist today."""
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    item = ImageItem(
        path="/x/a.nef", file_name="a.nef",
        ranking_results={
            "ai-model": {"score": 0.75, "rank": 3},
            "classic-vision": {"score": 0.42, "rank": 11},
        },
    )
    # score/rank are properties derived from ranking_results, not separate
    # fields - so they must agree with what was passed in.
    assert item.score == pytest.approx(0.75)
    assert item.rank == 3
    rows = ThumbnailCardDelegate._score_rows(item)
    assert [label for label, _ in rows] == ["AI", "Classic (SuperAnimal)"]  # AI first
    assert rows[0][1] == "0.7500 · #3"
    assert rows[1][1] == "0.4200 · #11"


def test_a_card_shows_the_module_it_has_and_stays_quiet_about_the_rest(app) -> None:
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    only_classic = ImageItem(
        path="/x/b.nef", file_name="b.nef",
        ranking_results={"classic-vision": {"score": 0.9, "rank": 1}},
    )
    assert only_classic.score is None  # no "ai-model" entry - not the same as unranked-by-everyone
    assert ThumbnailCardDelegate._score_rows(only_classic) == [("Classic (SuperAnimal)", "0.9000 · #1")]

    unscored = ImageItem(path="/x/c.nef", file_name="c.nef")
    assert ThumbnailCardDelegate._score_rows(unscored) == [("", "Unranked")]


def test_an_unknown_module_is_labelled_by_its_id_rather_than_dropped(app) -> None:
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    item = ImageItem(
        path="/x/d.nef", file_name="d.nef",
        ranking_results={"burst-analysis": {"score": 0.5, "rank": 2}},
    )
    assert ThumbnailCardDelegate._score_rows(item) == [("burst-analysis", "0.5000 · #2")]


def test_the_card_reserves_room_for_every_score_row_it_will_draw(app) -> None:
    """Cards are a uniform grid, so the height must already account for the
    rows - otherwise a second module's score would overlap the buttons."""
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate as D

    assert D.MAX_SCORE_ROWS >= 2
    assert D.CARD_HEIGHT >= 252 + (D.MAX_SCORE_ROWS - 1) * D.SCORE_ROW_HEIGHT


def test_the_loupe_shows_every_module_score_on_one_line(app) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    text = LoupeDialog._scores_text({
        "ranking_results": {
            "classic-vision": {"score": 0.42, "rank": 11},
            "ai-model": {"score": 0.75, "rank": 3},
        }
    })
    assert "AI 0.7500 (#3)" in text
    assert "Classic (SuperAnimal) 0.4200 (#11)" in text
    assert text.index("AI") < text.index("Classic (SuperAnimal)")
    assert LoupeDialog._scores_text({}) == "Unranked"


def test_every_module_is_sortable(app) -> None:
    """Sorting options are generated from the registry, so a new module
    becomes sortable at the moment it becomes runnable - including a second
    Classic Vision backend, independently of the first."""
    from picklikeme.desktop.main_window import SORT_SCORE_PREFIX, sort_options

    fields = [field for field, _ in sort_options()]
    labels = [label for _, label in sort_options()]
    assert "score" in fields and "AI Score" in labels
    assert f"{SORT_SCORE_PREFIX}classic-vision" in fields
    assert f"{SORT_SCORE_PREFIX}classic-vision-eyepose-v0" in fields
    assert "Classic Vision Ranking (SuperAnimal) Score" in labels
    assert "Classic Vision Ranking (EyePose-v0, recommended) Score" in labels
    assert fields[-2:] == ["filename", "captured_at"]


def test_sorting_by_a_module_score_uses_that_module(app, tmp_path) -> None:
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    )
    try:
        # Deliberately opposite orderings, so the two sorts cannot agree.
        low_ai = ImageItem(path="/x/low.nef", file_name="low.nef",
                           ranking_results={"ai-model": {"score": 0.1}, "classic-vision": {"score": 0.9}})
        high_ai = ImageItem(path="/x/high.nef", file_name="high.nef",
                            ranking_results={"ai-model": {"score": 0.9}, "classic-vision": {"score": 0.1}})
        items = [low_ai, high_ai]

        window._sort_field = "score"
        assert window._sort_items(items)[0] is high_ai

        window._sort_field = "score:classic-vision"
        assert window._sort_items(items)[0] is low_ai

        # An image that module never scored sorts to the end, not to the top.
        unscored = ImageItem(path="/x/none.nef", file_name="none.nef")
        assert window._sort_items([unscored, low_ai])[-1] is unscored
    finally:
        window.close()
        service.close()


def test_the_status_line_reports_what_was_filtered_out(app) -> None:
    """A Classic Vision run skips images, and the photographer has to be told
    how many and why - the AI path reports no filtering at all."""
    from picklikeme.desktop.main_window import MainWindow

    plain = MainWindow._ranking_summary({"image_count": 12, "filtered": {}})
    assert plain == "Ranked 12 images"

    filtered = MainWindow._ranking_summary(
        {"image_count": 10, "filtered": {"NO_SUBJECT": 3, "NO_VISIBLE_EYE": 7}}
    )
    assert "Ranked 10 images" in filtered
    assert "skipped 10" in filtered
    assert "No visible eye: 7" in filtered
    assert "No subject detected: 3" in filtered


def test_a_ranking_run_clears_the_thumbnail_cache(app, tmp_path, monkeypatch) -> None:
    """Regression: the Gallery's in-memory thumbnail cache (cache_manager) is
    keyed by (path, with_boxes) and was never invalidated by a ranking run.

    The real-world sequence that broke: Detector Boxes gets toggled on while
    only the AI model has ranked a folder - no eye data exists yet, so the
    overlaid pixmap cached for (path, True) has no eye box. Running Classic
    Vision afterward computes and saves eye data on disk, and
    review_thumbnail() would happily build a fresh overlay for it - but the
    Gallery never asked again, because the stale pixmap was still sitting in
    cache_manager under the same key. The eye overlay silently never
    appeared, in the Gallery specifically (the Loupe has no such cache - it
    re-reads the eye record on every navigation).
    """
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop import main_window as main_window_module
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    folder = tmp_path / "shoot"
    folder.mkdir()
    image_path = str(folder / "a.jpg")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)
    monkeypatch.setattr(
        service, "rank_folder",
        lambda **kwargs: {"state": service.load_session(), "image_count": 0, "filtered": {}},
    )

    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    )
    try:
        window.state.current_folder = str(folder)
        stale_pixmap = object()
        window.cache_manager.put_thumbnail((image_path, True), stale_pixmap)
        assert window.cache_manager.get_thumbnail((image_path, True)) is stale_pixmap

        # Bypass parameter dialogs and the real background QThread: both are
        # irrelevant to what this test checks (the cache-clearing wiring),
        # and neither can run headless without its own modal event loop.
        monkeypatch.setattr(window, "_collect_ranking_parameters", lambda strategy_id, info: {})

        def fake_run_with_progress(parent, title, func, *, on_success, on_error=None):
            del parent, title, on_error
            on_success(func())

            class _FakeThread:
                class _Signal:
                    def connect(self, *_args, **_kwargs) -> None:
                        pass

                finished = _Signal()

            return _FakeThread()

        monkeypatch.setattr(main_window_module, "run_with_progress", fake_run_with_progress)

        window._rank_with_strategy("classic-vision")

        assert window.cache_manager.get_thumbnail((image_path, True)) is None
    finally:
        window.close()
        service.close()


# ---------------------------------------------------------------------------
# Color Source - which strategy's Keep/Reject-styled coloring is on screen.
#
# Before this, the Gallery always tinted a card green/red/neutral by
# review_status alone with no way to tell whether that reflected the AI
# model, Classic Vision, or the photographer's own decision - impossible to
# debug the two strategies disagreeing. color_source_options() lists an
# explicit choice per registered strategy (plus "Review Status", the old
# behavior, kept as the default); the delegate paints a low-to-high gradient
# between the same reject/keep colors for whichever one is picked, so a
# strategy's ranking can be scanned across a folder at a glance without
# sorting by it. Untested until now - these are the first tests this
# mechanism had.
# ---------------------------------------------------------------------------


def test_color_source_options_lists_review_status_and_every_strategy(app) -> None:
    from picklikeme.desktop.main_window import color_source_options

    options = color_source_options()
    assert options[0] == (None, "Review Status")
    ids = [source for source, _ in options[1:]]
    assert set(ids) == {info.strategy_id for info in available_strategies()}
    labels = dict(options)
    assert labels["ai-model"] == "AI Model Score"
    assert labels["classic-vision"] == "Classic Vision Ranking (SuperAnimal) Score"
    assert labels["classic-vision-eyepose-v0"] == "Classic Vision Ranking (EyePose-v0, recommended) Score"


def test_default_coloring_is_unchanged_when_no_color_source_is_set(app) -> None:
    """Backward compatibility: a delegate that never had set_color_source
    called behaves exactly as before this feature existed - plain
    review-status coloring, ignoring any strategy score on the item."""
    from picklikeme.desktop import theme
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    palette = theme.current_palette()
    keep_item = ImageItem(path="/x/a.nef", file_name="a.nef", review_status="keep",
                           ranking_results={"classic-vision": {"score": 0.1}})
    assert delegate._get_background_color(palette, keep_item, False).name() == theme_color(palette.keep_bg)


def theme_color(hex_value: str) -> str:
    from PySide6.QtGui import QColor

    return QColor(hex_value).name()


def test_color_source_tints_by_the_chosen_strategy_s_score(app) -> None:
    from picklikeme.desktop import theme
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    palette = theme.current_palette()
    delegate.set_color_source("classic-vision", (0.0, 10.0))

    lowest = ImageItem(path="/x/low.nef", file_name="low.nef", review_status="reject",
                        ranking_results={"classic-vision": {"score": 0.0}})
    highest = ImageItem(path="/x/high.nef", file_name="high.nef", review_status="keep",
                         ranking_results={"classic-vision": {"score": 10.0}})

    # The low end of the range must render as the reject color and the high
    # end as the keep color, regardless of that item's own review_status -
    # the whole point is that this coloring is independent of it.
    assert delegate._get_background_color(palette, lowest, False).name() == theme_color(palette.reject_bg)
    assert delegate._get_background_color(palette, highest, False).name() == theme_color(palette.keep_bg)


def test_an_image_the_chosen_strategy_never_scored_falls_back_to_neutral(app) -> None:
    """An image Classic Vision filtered out (or that only the AI model has
    scored) has no classic-vision score to tint by - it must fall back to
    the same plain color an unranked image gets by default, not be mistaken
    for a score of zero."""
    from picklikeme.desktop import theme
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    palette = theme.current_palette()
    delegate.set_color_source("classic-vision", (0.0, 10.0))

    unscored = ImageItem(path="/x/u.nef", file_name="u.nef", review_status="keep")
    assert delegate._get_background_color(palette, unscored, False).name() == theme_color(palette.neutral_bg)


def test_selecting_a_color_source_computes_the_range_from_visible_items_only(app, tmp_path) -> None:
    """The gradient must span only what a filter is currently showing, not
    every image the strategy ever scored - a folder with a wide overall
    range but a narrow filtered view should still spread across the full
    gradient for what is on screen."""
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    )
    try:
        window._color_source = "classic-vision"
        visible = [
            ImageItem(path="/x/a.nef", file_name="a.nef", ranking_results={"classic-vision": {"score": 2.0}}),
            ImageItem(path="/x/b.nef", file_name="b.nef", ranking_results={"classic-vision": {"score": 5.0}}),
            ImageItem(path="/x/c.nef", file_name="c.nef"),  # never scored by it - excluded from the range
        ]
        window._update_color_source(visible)
        assert window._gallery_view._delegate._color_source == "classic-vision"
        assert window._gallery_view._delegate._score_range == (2.0, 5.0)

        window._color_source = None
        window._update_color_source(visible)
        assert window._gallery_view._delegate._color_source is None
        assert window._gallery_view._delegate._score_range is None
    finally:
        window.close()
        service.close()
