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
    """One widget per spec - a spin box for a number, a checkbox for a
    boolean switch (see ParamSpec.is_boolean) - so `_spins | _checks`
    together cover every declared parameter exactly once."""
    dialog = _dialog()
    specs = ClassicVisionParams.specs()
    assert set(dialog._spins) == {spec.name for spec in specs if not spec.is_boolean}
    assert set(dialog._checks) == {spec.name for spec in specs if spec.is_boolean}
    assert set(dialog._spins) | set(dialog._checks) == {spec.name for spec in specs}
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


def test_reset_to_defaults_restores_70_10_20(app) -> None:
    dialog = _dialog()
    for name in dialog._spins:
        dialog._spins[name].setValue(1.0)
    dialog.reset_to_defaults()

    assert dialog.parameters() == ClassicVisionParams()
    assert dialog.parameters().normalized_weights() == pytest.approx(
        {
            "eye_sharpness_weight": 0.7,
            "subject_sharpness_weight": 0.1,
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


def test_the_rank_button_is_a_split_button_with_a_dropdown(app, tmp_path) -> None:
    """Single click runs the default strategy; the arrow offers the others.

    Asserted through the style option rather than `QToolButton.menu()`, which
    returns None here by design: the button is built with setDefaultAction(),
    and the button paints/popups the *action's* menu while `menu()` only ever
    reports an explicitly-set one.
    """
    from PySide6.QtWidgets import QStyleOptionToolButton, QToolButton

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
        # Rank lives on the redesigned primary toolbar (see
        # MainWindow._build_primary_bar) as a standalone QToolButton, not
        # inside a QToolBar - findable by its own objectName.
        button = window.findChild(QToolButton, "rankButton")
        assert button is not None
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


def test_the_gallery_card_shows_only_the_selected_color_sources_own_score(app) -> None:
    """Redesign rule (PeakPick_UI_Design_Spec.md): "do not show every
    algorithm score on each thumbnail" - the card shows exactly the
    currently selected Color Source's own score, normalized `0.xxx`, never
    every module's score stacked in rows."""
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    item = ImageItem(
        path="/x/a.nef", file_name="a.nef",
        ranking_results={
            "ai-model": {"score": 0.75, "rank": 3},
            "classic-vision": {"score": 0.42, "rank": 11},
        },
    )
    assert item.score == pytest.approx(0.75)
    assert item.rank == 3

    delegate = ThumbnailCardDelegate()
    delegate.set_color_source("ai-model")
    assert delegate._selected_score_text(item) == "0.750"
    delegate.set_color_source("classic-vision")
    assert delegate._selected_score_text(item) == "0.420"
    # "Review Status" (no Color Source selected) has no single algorithm to
    # draw a score from.
    delegate.set_color_source(None)
    assert delegate._selected_score_text(item) == "—"


def test_a_score_badge_reads_em_dash_for_a_strategy_that_never_scored_this_image(app) -> None:
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    unscored = ImageItem(path="/x/c.nef", file_name="c.nef")
    delegate = ThumbnailCardDelegate()
    delegate.set_color_source("ai-model")
    assert delegate._selected_score_text(unscored) == "—"


def test_the_card_reserves_room_for_the_image_metadata_and_button_row(app) -> None:
    """Cards are a uniform grid, so the height must already account for the
    thumbnail, the name/meta rows, and the button row - not overlap."""
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate as D

    minimum = D.PADDING + D.THUMBNAIL_HEIGHT + D.SPACING + D.NAME_ROW_HEIGHT + D.META_ROW_HEIGHT + D.SPACING + D.BUTTON_HEIGHT + D.PADDING
    assert D.CARD_HEIGHT == minimum


def test_the_loupe_shows_every_module_score_on_one_line(app) -> None:
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    text = LoupeDialog._scores_text({
        "ranking_results": {
            "classic-vision": {"score": 0.42, "rank": 11},
            "ai-model": {"score": 0.75, "rank": 3},
        }
    })
    assert "AI 0.750 (#3)" in text  # three decimals everywhere - design_system.SCORE_FORMAT
    assert "Classic (SuperAnimal) 0.420 (#11)" in text
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
    assert "No reliable visible eye: 7" in filtered
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


def test_a_crop_cache_mismatch_offers_to_rebuild_and_retries(app, tmp_path, monkeypatch) -> None:
    """The reported real-world block: a CropCacheVersionMismatch's dialog
    used to tell a photographer to "Pass --force to rebuild" - CLI language
    with no equivalent in the desktop UI, so the run was simply stuck. This
    pins the fix: on that specific error, desktop now offers to rebuild and
    retries the exact same run with force_preprocess=True through the same
    path, rather than leaving the photographer at a dead end.
    """
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop import main_window as main_window_module
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    folder = tmp_path / "shoot"
    folder.mkdir()

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    calls: list[bool] = []

    def fake_rank_folder(**kwargs):
        force = kwargs.get("force_preprocess", False)
        calls.append(force)
        if not force:
            raise RuntimeError(
                "Existing cache at cache\\crops was built with different parameters:\n"
                "  existing: CropParams(conf_threshold=0.3, min_crop_confidence=0.6, version='v7')\n"
                "  requested: CropParams(conf_threshold=0.8, min_crop_confidence=0.8, version='v7')\n"
                "Pass --force to rebuild, or delete the cache directory."
            )
        return {"state": service.load_session(), "image_count": 0, "filtered": {}}

    monkeypatch.setattr(service, "rank_folder", fake_rank_folder)

    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    )
    try:
        window.state.current_folder = str(folder)
        monkeypatch.setattr(window, "_collect_ranking_parameters", lambda strategy_id, info: {})

        def fake_run_with_progress(parent, title, func, *, on_success, on_error=None):
            del parent, title
            try:
                on_success(func())
            except Exception as exc:  # noqa: BLE001 - mirrors ProgressWorker's own except Exception
                if on_error is not None:
                    on_error(str(exc))

            class _FakeThread:
                class _Signal:
                    def connect(self, *_args, **_kwargs) -> None:
                        pass

                finished = _Signal()

            return _FakeThread()

        monkeypatch.setattr(main_window_module, "run_with_progress", fake_run_with_progress)
        # Simulate the photographer clicking "Yes" on the rebuild prompt.
        monkeypatch.setattr(
            main_window_module.QMessageBox, "question",
            lambda *args, **kwargs: main_window_module.QMessageBox.StandardButton.Yes,
        )

        window._rank_with_strategy("classic-vision")

        assert calls == [False, True], "must retry once, and the retry must pass force_preprocess=True"
    finally:
        window.close()
        service.close()


def test_declining_the_rebuild_prompt_shows_the_plain_error_instead(app, tmp_path, monkeypatch) -> None:
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop import main_window as main_window_module
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    folder = tmp_path / "shoot"
    folder.mkdir()
    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    service.open_folder(folder)

    calls: list[bool] = []

    def fake_rank_folder(**kwargs):
        calls.append(kwargs.get("force_preprocess", False))
        raise RuntimeError("Existing cache ... different parameters ...\nPass --force to rebuild, or delete the cache directory.")

    monkeypatch.setattr(service, "rank_folder", fake_rank_folder)

    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    )
    try:
        window.state.current_folder = str(folder)
        monkeypatch.setattr(window, "_collect_ranking_parameters", lambda strategy_id, info: {})

        def fake_run_with_progress(parent, title, func, *, on_success, on_error=None):
            del parent, title
            try:
                on_success(func())
            except Exception as exc:  # noqa: BLE001
                if on_error is not None:
                    on_error(str(exc))

            class _FakeThread:
                class _Signal:
                    def connect(self, *_args, **_kwargs) -> None:
                        pass

                finished = _Signal()

            return _FakeThread()

        monkeypatch.setattr(main_window_module, "run_with_progress", fake_run_with_progress)
        monkeypatch.setattr(
            main_window_module.QMessageBox, "question",
            lambda *args, **kwargs: main_window_module.QMessageBox.StandardButton.No,
        )
        warned = []
        monkeypatch.setattr(
            main_window_module.QMessageBox, "warning",
            lambda *args, **kwargs: warned.append(args),
        )

        window._rank_with_strategy("classic-vision")

        assert calls == [False]  # no retry
        assert len(warned) == 1  # the plain failure dialog was shown instead
    finally:
        window.close()
        service.close()


# ---------------------------------------------------------------------------
# Color Source - which strategy's Keep/Reject-styled coloring is on screen.
#
# The Color selector picks ONE of two independent modes, and that mode is
# the only input to a card's color (see design_system's own module-level
# comment on the two vocabularies).
#
# These tests used to encode the opposite rule: one blended five-value
# answer where the photographer's decision won and, failing that, the
# algorithm's binary keep/reject-at-a-threshold suggestion borrowed the very
# same Keep/Reject colors. That is what made the two kinds of information
# mutually contaminating - an algorithm-colored grid tinted by whatever had
# been reviewed, showing a threshold verdict rather than the score it
# claimed to show, and a User Decision-colored grid that could not be
# trusted to mean "I decided this". They now assert the separation.
# ---------------------------------------------------------------------------


def test_color_source_options_lists_algorithm_ran_last_review_status_and_every_strategy(app) -> None:
    from picklikeme.desktop.main_window import ALGORITHM_RAN_LAST, color_source_options

    options = color_source_options()
    assert options[0] == (ALGORITHM_RAN_LAST, "Algorithm Ran Last")
    assert options[1] == (None, "User Decision"), "renamed from 'Review Status'"
    ids = [source for source, _ in options[2:]]
    assert set(ids) == {info.strategy_id for info in available_strategies()}
    labels = dict(options)
    assert labels["ai-model"] == "AI Model Score"
    assert labels["classic-vision"] == "Classic Vision Ranking (SuperAnimal) Score"
    assert labels["classic-vision-eyepose-v0"] == "Classic Vision Ranking (EyePose-v0, recommended) Score"


def test_user_decision_mode_colors_only_by_the_users_own_decision(app) -> None:
    """"User Decision" as the Color mode (never called set_color_source, or
    explicitly None): Keep/Reject/Undecided from the photographer's own
    decision, ignoring every score, suggestion and filter verdict on the
    item."""
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    keep_item = ImageItem(path="/x/a.nef", file_name="a.nef", review_status="keep",
                           ranking_results={"classic-vision": {"score": 0.1}})
    assert delegate._resolve_status(keep_item) == "keep"
    reject_item = ImageItem(path="/x/c.nef", file_name="c.nef", review_status="reject")
    assert delegate._resolve_status(reject_item) == "reject"
    undecided_item = ImageItem(path="/x/b.nef", file_name="b.nef")
    assert delegate._resolve_status(undecided_item) == "undecided"


def test_user_decision_mode_ignores_ranking_results_and_suggestions(app) -> None:
    """THE regression: a top-scored, cutoff-suggested-Keep image that
    nobody has reviewed is Undecided in User Decision mode. Before the
    split, its algorithm suggestion painted it the Keep color, which is
    exactly how ~5,000 unreviewed images came to look reviewed."""
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()  # User Decision mode

    top_scored = ImageItem(
        path="/x/high.nef", file_name="high.nef",
        algorithm_suggestion="keep", ai_suggestion="keep", algorithm_decision="keep",
        ranking_results={"classic-vision": {"score": 0.99, "rank": 1}},
    )
    assert delegate._resolve_status(top_scored) == "undecided"


def test_algorithm_mode_ignores_the_users_own_decision(app) -> None:
    """The other direction: an algorithm mode reports what THAT strategy did
    with the image. A photographer's Keep/Reject on the same frame is a
    different fact and does not repaint it."""
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    delegate.set_color_source("classic-vision")

    user_kept = ImageItem(
        path="/x/a.nef", file_name="a.nef", review_status="keep",
        ranking_results={"classic-vision": {"score": 0.2}},
    )
    user_rejected = ImageItem(
        path="/x/b.nef", file_name="b.nef", review_status="reject",
        ranking_results={"classic-vision": {"score": 0.8}},
    )
    assert delegate._resolve_status(user_kept) == "scored"
    assert delegate._resolve_status(user_rejected) == "scored"


def test_an_image_the_chosen_strategy_explicitly_filtered_gets_filtered_out(app) -> None:
    """A strategy that recorded an explicit filter reason for this image DID
    examine it - "Filtered Out" (gray), distinct from "Skipped" (purple,
    below - never touched at all) and from plain "Review" (see the design
    spec's own "Skipped is intentionally different from Reject" rule,
    generalized to Filtered Out too)."""
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    delegate.set_color_source("classic-vision")

    filtered = ImageItem(
        path="/x/u.nef", file_name="u.nef", algorithm_suggestion=None,
        filter_reasons={"classic-vision": "NO_VISIBLE_EYE"},
    )
    assert delegate._resolve_status(filtered) == "filtered"


def test_an_image_the_chosen_strategy_never_touched_gets_skipped(app) -> None:
    """No ranking_result AND no filter_reasons entry for this strategy at
    all - never touched, not merely filtered - renders "Skipped", never
    silently identical to "Filtered Out" or plain "Review"."""
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    delegate.set_color_source("classic-vision")

    untouched = ImageItem(path="/x/v.nef", file_name="v.nef", algorithm_suggestion=None)
    assert delegate._resolve_status(untouched) == "skipped"


def test_an_image_the_chosen_strategy_scored_is_scored(app) -> None:
    """A real result for this image from this strategy - "Scored", whatever
    any threshold would say about it, and never "Skipped"/"Filtered Out"."""
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    delegate.set_color_source("classic-vision")

    scored = ImageItem(
        path="/x/w.nef", file_name="w.nef", algorithm_suggestion=None,
        ranking_results={"classic-vision": {"score": 0.5}},
    )
    assert delegate._resolve_status(scored) == "scored"


def test_algorithm_mode_colors_a_card_by_its_actual_score(app) -> None:
    """"The color must correspond to that score": across the visible range,
    the lowest-scoring card is painted at the bottom of the ramp and the
    highest at the top, with a mid-scoring card strictly between them."""
    from picklikeme.desktop import theme
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate
    from picklikeme.desktop.widgets.design_system import score_ramp_color

    delegate = ThumbnailCardDelegate()
    delegate.set_color_source("classic-vision")
    delegate.set_score_range((0.2, 0.8))
    palette = theme.current_palette()

    def color_of(score: float) -> str:
        item = ImageItem(path="/x/s.nef", file_name="s.nef",
                         ranking_results={"classic-vision": {"score": score}})
        return delegate._status_color(palette, item, delegate._resolve_status(item))

    assert color_of(0.2) == score_ramp_color(palette, 0.0)
    assert color_of(0.8) == score_ramp_color(palette, 1.0)
    assert color_of(0.5) == score_ramp_color(palette, 0.5)
    assert len({color_of(0.2), color_of(0.5), color_of(0.8)}) == 3


def test_selecting_a_color_source_propagates_to_the_gallery_delegate(app, tmp_path) -> None:
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
        visible = [ImageItem(path="/x/a.nef", file_name="a.nef", algorithm_suggestion="keep")]
        window._update_color_source(visible)
        assert window._gallery_view._delegate._color_source == "classic-vision"

        window._color_source = None
        window._update_color_source(visible)
        assert window._gallery_view._delegate._color_source is None
    finally:
        window.close()
        service.close()


# ---------------------------------------------------------------------------
# "Algorithm Ran Last" - the centralized, dynamically-resolving Color Source
# option (see main_window.ALGORITHM_RAN_LAST / MainWindow._resolve_color_source
# / ReviewSession.latest_run_strategy).
# ---------------------------------------------------------------------------


def _window_with_service(tmp_path):
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    )
    return window, service


def test_algorithm_ran_last_is_the_selected_combo_item_by_default(app, tmp_path) -> None:
    from picklikeme.desktop.main_window import ALGORITHM_RAN_LAST

    window, service = _window_with_service(tmp_path)
    try:
        assert window._color_source == ALGORITHM_RAN_LAST
        assert window._color_combo.currentData() == ALGORITHM_RAN_LAST
    finally:
        window.close()
        service.close()


def test_algorithm_ran_last_resolves_to_whichever_strategy_actually_ran(app, tmp_path) -> None:
    """The core requirement: selecting "Algorithm Ran Last" must track
    whichever strategy most recently completed a run - never a fixed
    default, never whatever happened to be first in the registry."""
    from picklikeme.sidecar import strategy_ranking_path, write_run_metadata

    from picklikeme.review.session import ReviewSession

    shoot = tmp_path / "shoot"
    shoot.mkdir()
    image = shoot / "a.jpg"
    image.write_bytes(b"frame")
    _write_csv(strategy_ranking_path(shoot, "classic-vision-fusion-mammals"), [(image, 0.7)])
    write_run_metadata(shoot, strategy="classic-vision-fusion-mammals")

    window, service = _window_with_service(tmp_path)
    try:
        service.open_folder(shoot)
        window._sync_color_source_from_session()

        from picklikeme.desktop.main_window import ALGORITHM_RAN_LAST

        assert window._color_source == ALGORITHM_RAN_LAST
        assert window._resolve_color_source() == "classic-vision-fusion-mammals"
    finally:
        window.close()
        service.close()


def _write_csv(target, entries) -> None:
    import csv

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow([])
        writer.writerow(["rank", "image_path", "score", "label"])
        for rank, (path, score) in enumerate(entries, start=1):
            writer.writerow([rank, str(path), f"{score:.6f}", 0])


def test_a_manually_selected_strategy_does_not_change_when_a_different_strategy_ranks(app, tmp_path) -> None:
    """Picking a specific strategy by name pins to it - only the
    ALGORITHM_RAN_LAST sentinel re-resolves dynamically."""
    window, service = _window_with_service(tmp_path)
    try:
        window._color_source = "classic-vision-eyepose-v0"
        assert window._resolve_color_source() == "classic-vision-eyepose-v0"
    finally:
        window.close()
        service.close()


# ---------------------------------------------------------------------------
# "Last Run Algorithm" -> displayed score -> rank -> sorting: one strategy.
#
# The card's score badge showed the SELECTED strategy's score while the rank
# prefix showed the AI model's rank and the default sort ordered by the AI
# model's score. On a folder ranked only by Crop Sharpness (no AI run at
# all) that meant Crop Sharpness numbers on cards that carried no rank and
# were not actually ordered by anything, under a Sort labelled "AI Score".
# ---------------------------------------------------------------------------


def test_the_default_sort_follows_the_selected_algorithm(app, tmp_path) -> None:
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import SORT_SELECTED_ALGORITHM, MainWindow
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    window = MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    )
    try:
        assert window._sort_field == SORT_SELECTED_ALGORITHM
        # Only "classic-vision" scored these - the AI model never ran.
        items = [
            ImageItem(path="/x/low.nef", file_name="low.nef",
                      ranking_results={"classic-vision": {"score": 0.1, "rank": 3}}),
            ImageItem(path="/x/high.nef", file_name="high.nef",
                      ranking_results={"classic-vision": {"score": 0.9, "rank": 1}}),
            ImageItem(path="/x/mid.nef", file_name="mid.nef",
                      ranking_results={"classic-vision": {"score": 0.5, "rank": 2}}),
        ]

        window._color_source = "classic-vision"
        ordered = [item.file_name for item in window._sort_items(items)]
        assert ordered == ["high.nef", "mid.nef", "low.nef"], "sorted by the strategy on screen"

        # The old default ("AI Score") has no value for any of them, so they
        # all fall through to the name ordering - the symptom this replaced.
        window._sort_field = "score"
        assert [i.file_name for i in window._sort_items(items)] == ["high.nef", "low.nef", "mid.nef"]
    finally:
        window.close()
        service.close()


def test_the_card_rank_comes_from_the_same_strategy_as_the_score_badge(app) -> None:
    from picklikeme.desktop.models.image_item import ImageItem
    from picklikeme.desktop.views.gallery.thumbnail_delegate import ThumbnailCardDelegate

    delegate = ThumbnailCardDelegate()
    item = ImageItem(
        path="/x/a.nef", file_name="a.nef",
        ranking_results={"ai-model": {"score": 0.2, "rank": 47},
                          "classic-vision": {"score": 0.9, "rank": 1}},
    )

    delegate.set_color_source("classic-vision")
    assert delegate._selected_score_text(item) == "0.900"
    assert item.rank_for("classic-vision") == 1, "the rank the card draws alongside that score"

    delegate.set_color_source("ai-model")
    assert delegate._selected_score_text(item) == "0.200"
    assert item.rank_for("ai-model") == 47


def test_the_score_range_is_measured_over_the_visible_set(app, tmp_path) -> None:
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
        visible = [
            ImageItem(path="/x/a.nef", file_name="a.nef",
                      ranking_results={"classic-vision": {"score": 0.25}}),
            ImageItem(path="/x/b.nef", file_name="b.nef",
                      ranking_results={"classic-vision": {"score": 0.75}}),
            ImageItem(path="/x/c.nef", file_name="c.nef"),  # unscored: not in the range
        ]
        assert window._score_range_for(visible, "classic-vision") == (0.25, 0.75)
        assert window._score_range_for(visible, "ai-model") is None, "nothing this strategy scored"
        assert window._score_range_for(visible, None) is None, "User Decision mode has no range"

        window._color_source = "classic-vision"
        window._update_color_source(visible)
        assert window._gallery_view._delegate._score_range == (0.25, 0.75)
    finally:
        window.close()
        service.close()
