"""Desktop wiring for Burst Analysis: the Collapse Bursts gallery toggle and
burst-scoped Loupe navigation.

Backend burst grouping/ranking is tested in isolation in
test_burst_analysis.py; ReviewSession's own wiring on top of it in
test_review_session.py's BurstInfoTests. This file covers only the desktop
layer built on top of both: MainWindow._apply_filter's collapsing,
ThumbnailCardDelegate's badge, and _open_loupe_for_item's burst-scoped
navigation.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _window(tmp_path):
    from picklikeme.desktop.application import ApplicationState, WorkerManager
    from picklikeme.desktop.main_window import MainWindow
    from picklikeme.desktop.services import ReviewService
    from picklikeme.desktop.settings import DesktopSettings

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    return MainWindow(
        state=ApplicationState(), settings=DesktopSettings(),
        service=service, worker_manager=WorkerManager(),
    ), service


def _burst_items():
    """Two bursts: "b1" (3 members, images[1] is burst_best) and "b2" (a
    singleton, always its own burst_best)."""
    from picklikeme.desktop.models.image_item import ImageItem

    return [
        ImageItem(path="/x/a.nef", file_name="a.nef", burst_id="b1", burst_size=3, burst_rank=2, burst_best=False),
        ImageItem(path="/x/b.nef", file_name="b.nef", burst_id="b1", burst_size=3, burst_rank=1, burst_best=True),
        ImageItem(path="/x/c.nef", file_name="c.nef", burst_id="b1", burst_size=3, burst_rank=3, burst_best=False),
        ImageItem(path="/x/d.nef", file_name="d.nef", burst_id="b2", burst_size=1, burst_rank=1, burst_best=True),
    ]


def test_the_gallery_is_unchanged_by_default(app, tmp_path) -> None:
    """Collapse Bursts starts off - every image shows individually, exactly
    as before Burst Analysis existed."""
    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._apply_filter()
        assert len(window._gallery_model.items()) == 4
        assert window._gallery_view._delegate._show_burst_badge is False
    finally:
        window.close()
        service.close()


def test_collapse_bursts_shows_only_the_best_of_each_burst(app, tmp_path) -> None:
    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._on_toggle_collapse_bursts(True)

        visible = {item.path for item in window._gallery_model.items()}
        assert visible == {"/x/b.nef", "/x/d.nef"}
        assert window._gallery_view._delegate._show_burst_badge is True

        window._on_toggle_collapse_bursts(False)
        assert len(window._gallery_model.items()) == 4
        assert window._gallery_view._delegate._show_burst_badge is False
    finally:
        window.close()
        service.close()


def test_collapse_bursts_action_label_always_names_the_next_click(app, tmp_path) -> None:
    """The action's own text is "Collapse Bursts" while expanded (a click
    collapses them) and "Uncollapse Bursts" once collapsed (a click expands
    them again) - never a static label that could disagree with what
    clicking it right now actually does. Same QAction is shared by the
    toolbar button and the View menu item, so this one label update covers
    both surfaces at once."""
    window, service = _window(tmp_path)
    try:
        assert window._collapse_bursts_action.text() == "Collapse Bursts"

        window._collapse_bursts_action.trigger()
        assert window._collapse_bursts is True
        assert window._collapse_bursts_action.text() == "Uncollapse Bursts"

        window._collapse_bursts_action.trigger()
        assert window._collapse_bursts is False
        assert window._collapse_bursts_action.text() == "Collapse Bursts"
    finally:
        window.close()
        service.close()


def test_burst_order_toolbar_combo_reuses_the_existing_sort_mechanism(app, tmp_path, monkeypatch) -> None:
    """The toolbar's own Burst Order control (item 1 of the redesign ask:
    "expose the existing sorting mode more clearly") is not a second
    sorting mechanism - it is a second UI surface over the exact same
    self._burst_sort_mode/_set_burst_sort_mode/BURST_SORT_* the View menu's
    "Burst Order" submenu already used (see test_burst_ui.py's other Burst
    Order tests). Both the combo and the menu actions stay in sync with
    each other because both ultimately call _set_burst_sort_mode - proven
    here by changing it from one surface and reading it back from the
    other."""
    from picklikeme.desktop.main_window import BURST_SORT_BURST_SCORE, BURST_SORT_CAPTURE_TIME

    _isolate_settings(monkeypatch, tmp_path)
    window, service = _window(tmp_path)
    try:
        # Present, visible, and reflecting the real (default) mode from the
        # moment the toolbar is built - not a separate combo that starts
        # out of sync with whatever was already selected.
        assert window._burst_sort_combo.currentData() == BURST_SORT_BURST_SCORE == window._burst_sort_mode

        # Changing it via the TOOLBAR combo (a real index change, not a
        # direct attribute set) must update self._burst_sort_mode AND the
        # View menu's own actions - the single source of truth invariant.
        index = window._burst_sort_combo.findData(BURST_SORT_CAPTURE_TIME)
        window._burst_sort_combo.setCurrentIndex(index)
        assert window._burst_sort_mode == BURST_SORT_CAPTURE_TIME
        assert window._burst_order_capture_time_action.isChecked()
        assert not window._burst_order_score_action.isChecked()

        # And the reverse direction: changing it via the View menu action
        # must update the toolbar combo back.
        window._burst_order_score_action.trigger()
        assert window._burst_sort_mode == BURST_SORT_BURST_SCORE
        assert window._burst_sort_combo.currentData() == BURST_SORT_BURST_SCORE

        # And it actually drives the real navigation order - the same
        # invariant test_burst_order_is_decided_by_the_main_grid_before_
        # the_loupe_ever_opens checks for the View menu path, exercised here
        # through the toolbar combo instead.
        captured = _spy_loupe_dialog(monkeypatch)
        window._all_items = _scored_disagreeing_burst(tmp_path)
        window._on_toggle_collapse_bursts(True)
        (visible,) = window._gallery_model.items()

        index = window._burst_sort_combo.findData(BURST_SORT_CAPTURE_TIME)
        window._burst_sort_combo.setCurrentIndex(index)
        window._open_loupe_for_item(visible)
        dialog = captured["dialog"]
        assert dialog.image_paths == [str(tmp_path / n) for n in ("a.jpg", "b.jpg", "c.jpg")]
        dialog.close()
    finally:
        window.close()
        service.close()


def test_opening_a_card_normally_scopes_the_loupe_to_the_visible_gallery(app, tmp_path, monkeypatch) -> None:
    from picklikeme.desktop import main_window as main_window_module

    captured = {}

    class _FakeLoupeDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def exec(self):
            return None

    monkeypatch.setattr(main_window_module, "LoupeDialog", _FakeLoupeDialog)

    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._apply_filter()  # collapse is off

        window._open_loupe_for_item(window._gallery_model.item_at(2))  # c.nef

        assert captured["image_paths"] == ["/x/a.nef", "/x/b.nef", "/x/c.nef", "/x/d.nef"]
        assert captured["start_index"] == 2
    finally:
        window.close()
        service.close()


def test_opening_a_collapsed_card_scopes_the_loupe_to_its_burst_in_rank_order(app, tmp_path, monkeypatch) -> None:
    """The reported requirement: selecting a burst opens the Loupe scoped to
    that burst's own members, ordered by burst_rank - not the whole
    collapsed gallery, and not left/right-file order.

    Uses items scored by the default burst_strategy (ai-model), not the
    plain _burst_items() fixture - with no score behind burst_rank at all,
    _burst_score_available correctly falls back to Capture Time (see
    test_an_unscored_burst_shows_the_no_effect_warning_... below), which
    would make this specific test's own rank-order assertion wrong for a
    reason unrelated to what it actually checks.
    """
    from picklikeme.desktop import main_window as main_window_module
    from picklikeme.desktop.models.image_item import ImageItem

    captured = {}

    class _FakeLoupeDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def exec(self):
            return None

    monkeypatch.setattr(main_window_module, "LoupeDialog", _FakeLoupeDialog)

    window, service = _window(tmp_path)
    try:
        window._all_items = [
            ImageItem(
                path="/x/a.nef", file_name="a.nef", burst_id="b1", burst_size=3, burst_rank=2, burst_best=False,
                ranking_results={"ai-model": {"score": 0.60, "rank": 2}},
            ),
            ImageItem(
                path="/x/b.nef", file_name="b.nef", burst_id="b1", burst_size=3, burst_rank=1, burst_best=True,
                ranking_results={"ai-model": {"score": 0.90, "rank": 1}},
            ),
            ImageItem(
                path="/x/c.nef", file_name="c.nef", burst_id="b1", burst_size=3, burst_rank=3, burst_best=False,
                ranking_results={"ai-model": {"score": 0.30, "rank": 3}},
            ),
            ImageItem(
                path="/x/d.nef", file_name="d.nef", burst_id="b2", burst_size=1, burst_rank=1, burst_best=True,
                ranking_results={"ai-model": {"score": 0.50, "rank": 1}},
            ),
        ]
        window._on_toggle_collapse_bursts(True)

        # Only b.nef (burst_best of "b1") is a visible row now.
        (visible_b1,) = [i for i in window._gallery_model.items() if i.burst_id == "b1"]
        window._open_loupe_for_item(visible_b1)

        # Ordered by burst_rank (b=1, a=2, c=3), not by burst_size membership
        # order or the original gallery order.
        assert captured["image_paths"] == ["/x/b.nef", "/x/a.nef", "/x/c.nef"]
        assert captured["start_index"] == 0
    finally:
        window.close()
        service.close()


def test_opening_a_collapsed_singleton_burst_shows_only_itself(app, tmp_path, monkeypatch) -> None:
    from picklikeme.desktop import main_window as main_window_module

    captured = {}

    class _FakeLoupeDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def exec(self):
            return None

    monkeypatch.setattr(main_window_module, "LoupeDialog", _FakeLoupeDialog)

    window, service = _window(tmp_path)
    try:
        window._all_items = _burst_items()
        window._on_toggle_collapse_bursts(True)

        (visible_b2,) = [i for i in window._gallery_model.items() if i.burst_id == "b2"]
        window._open_loupe_for_item(visible_b2)

        assert captured["image_paths"] == ["/x/d.nef"]
    finally:
        window.close()
        service.close()


def _isolate_settings(monkeypatch, tmp_path):
    """An isolated, file-backed QSettings - MainWindow always constructs its
    own QSettings("PeakPic", "PeakPicDesktop") internally (see __init__),
    which would otherwise read/write the real OS-level app settings during a
    test run."""
    from PySide6.QtCore import QSettings

    from picklikeme.desktop import main_window as main_window_module

    settings_path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        main_window_module, "QSettings",
        lambda *a, **k: QSettings(settings_path, QSettings.Format.IniFormat),
    )


def _spy_loupe_dialog(monkeypatch):
    """Patches MainWindow's LoupeDialog with a thin exec()-stubbing subclass
    of the REAL class (not a captured-kwargs stand-in), returning the dict
    the next _open_loupe_for_item call will populate under "dialog"."""
    from picklikeme.desktop import main_window as main_window_module
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    captured = {}

    class _SpyLoupeDialog(LoupeDialog):
        def exec(self):  # noqa: A003 - matches QDialog's own method name
            from PySide6.QtWidgets import QDialog

            captured["dialog"] = self
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(main_window_module, "LoupeDialog", _SpyLoupeDialog)
    return captured


def _scored_disagreeing_burst(tmp_path):
    """Three members of one burst whose burst_rank order (b, a, c - score-
    descending) and captured_at order (a, b, c) genuinely disagree, and
    which HAVE been scored by the default burst_strategy (ai-model) - so
    Burst Score is real signal, not a silent Capture Time fallback (see
    test_an_unscored_burst_falls_back_to_capture_time_and_sequences_coincide
    below for that case)."""
    from PIL import Image

    from picklikeme.desktop.models.image_item import ImageItem

    for name in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (16, 16), color="blue").save(tmp_path / name, format="JPEG")
    return [
        ImageItem(
            path=str(tmp_path / "a.jpg"), file_name="a.jpg", burst_id="b1", burst_size=3,
            burst_rank=2, burst_best=False, captured_at="2024-01-01T10:00:00",
            ranking_results={"ai-model": {"score": 0.60, "rank": 2}},
        ),
        ImageItem(
            path=str(tmp_path / "b.jpg"), file_name="b.jpg", burst_id="b1", burst_size=3,
            burst_rank=1, burst_best=True, captured_at="2024-01-01T10:00:01",
            ranking_results={"ai-model": {"score": 0.90, "rank": 1}},
        ),
        ImageItem(
            path=str(tmp_path / "c.jpg"), file_name="c.jpg", burst_id="b1", burst_size=3,
            burst_rank=3, burst_best=False, captured_at="2024-01-01T10:00:02",
            ranking_results={"ai-model": {"score": 0.30, "rank": 3}},
        ),
    ]


def test_burst_order_is_decided_by_the_main_grid_before_the_loupe_ever_opens(app, tmp_path, monkeypatch) -> None:
    """The architectural fix for a string of reported bugs where fixing
    Capture Time sorting broke Score sorting or vice versa, or a mode change
    inside the Loupe silently had no effect: there is now exactly ONE place
    burst order is decided - MainWindow._burst_sort_mode (set via the View
    menu's "Burst Order" submenu) - and the Loupe receives an already-
    ordered list it never re-sorts. Changing the mode only affects the NEXT
    Loupe session, not one already open - verified by opening twice.

    Goes through the real open_folder -> ReviewSession -> _refresh_from_state
    pipeline (like _real_burst_folder-based tests below), not a hand-built
    `window._all_items` - `_open_loupe` calls `_refresh_from_state(service.
    load_session())` once the (stubbed) dialog returns, which would silently
    wipe out a hand-assigned `_all_items` that the real session never knew
    about, and this test specifically needs a SECOND _open_loupe_for_item
    call after the first one closes.
    """
    from picklikeme.desktop.main_window import BURST_SORT_BURST_SCORE, BURST_SORT_CAPTURE_TIME
    from picklikeme.sidecar import AI_STRATEGY_ID

    _isolate_settings(monkeypatch, tmp_path)
    folder = _real_burst_folder(tmp_path, filenames_in_capture_order=["DSC_0001.jpg", "DSC_0002.jpg", "DSC_0003.jpg"])
    window, service = _window(tmp_path)
    try:
        service.open_folder(folder)
        scores = {"DSC_0001.jpg": 0.60, "DSC_0002.jpg": 0.90, "DSC_0003.jpg": 0.30}
        captured_at = {
            "DSC_0001.jpg": "2024-01-01T10:00:00",
            "DSC_0002.jpg": "2024-01-01T10:00:01",
            "DSC_0003.jpg": "2024-01-01T10:00:02",
        }
        for fname, score in scores.items():
            image = service.session._image_for(str(folder / fname))
            image.captured_at = captured_at[fname]
            image.ranking_results[AI_STRATEGY_ID] = {"score": score, "rank": None}

        window.state.current_folder = str(folder)
        window._refresh_from_state(service.load_session())
        window._on_toggle_collapse_bursts(True)
        (visible,) = window._gallery_model.items()

        # Default mode is Burst Score - matches score-descending order
        # (0002, 0001, 0003), NOT capture-time order (0001, 0002, 0003).
        assert window._burst_sort_mode == BURST_SORT_BURST_SCORE
        dialog = _open_burst_loupe(window, monkeypatch)
        assert [Path(p).name for p in dialog.image_paths] == ["DSC_0002.jpg", "DSC_0001.jpg", "DSC_0003.jpg"]
        # No sort control in the Loupe at all - see the module docstring.
        assert not hasattr(dialog, "_burst_sort_combo")
        # Navigation (the same path wheel/keyboard use) walks the order the
        # dialog was opened with.
        dialog.index = 0
        dialog._go_next()
        assert Path(dialog._current_path()).name == "DSC_0001.jpg"
        dialog.close()

        # Changing Burst Order in the Main Grid does NOT affect an already-
        # closed session retroactively, and requires a fresh open to apply -
        # exactly the "close, change, reopen" workflow this fix asks for.
        window._set_burst_sort_mode(BURST_SORT_CAPTURE_TIME)
        dialog2 = _open_burst_loupe(window, monkeypatch)
        assert [Path(p).name for p in dialog2.image_paths] == ["DSC_0001.jpg", "DSC_0002.jpg", "DSC_0003.jpg"]
        dialog2.index = 0
        assert Path(dialog2._current_path()).name == "DSC_0001.jpg"
        dialog2._go_next()
        assert Path(dialog2._current_path()).name == "DSC_0002.jpg"
        dialog2._go_next()
        assert Path(dialog2._current_path()).name == "DSC_0003.jpg"
        dialog2.close()
    finally:
        window.close()
        service.close()


def test_burst_order_menu_action_and_real_gallery_signals_drive_the_same_path(app, tmp_path, monkeypatch) -> None:
    """Re-verifies the fix through paths a unit test calling internal
    methods directly could still miss:

    - changing Burst Order via the REAL View-menu QAction.trigger() (what an
      actual menu click fires), not by setting window._burst_sort_mode
      directly.
    - opening the Loupe via the REAL gallery double-click *signal*
      (QAbstractItemView.doubleClicked.emit), not by calling
      _open_loupe_for_item directly.
    - advancing via _on_wheel_navigate (what _ZoomView.wheelEvent actually
      emits into) rather than only _go_next/_go_prev directly.
    """
    from PySide6.QtCore import Qt

    from picklikeme.desktop.main_window import BURST_SORT_CAPTURE_TIME

    _isolate_settings(monkeypatch, tmp_path)
    captured = _spy_loupe_dialog(monkeypatch)

    window, service = _window(tmp_path)
    try:
        window._all_items = _scored_disagreeing_burst(tmp_path)
        window._on_toggle_collapse_bursts(True)

        # A real menu-action trigger, not a direct attribute set.
        window._burst_order_capture_time_action.trigger()
        assert window._burst_sort_mode == BURST_SORT_CAPTURE_TIME
        assert window._burst_order_capture_time_action.isChecked()
        assert not window._burst_order_score_action.isChecked()

        # Real double-click *signal*, exactly what MainWindow.__init__ wires
        # the gallery view to.
        index = window._gallery_model.index(0, 0)
        window._gallery_view.doubleClicked.emit(index)

        dialog = captured["dialog"]
        assert dialog.image_paths == [str(tmp_path / n) for n in ("a.jpg", "b.jpg", "c.jpg")]
        assert dialog._burst_best_label.text() == "Best Image: No"  # a.jpg is rank #2

        # Wheel navigation - _ZoomView.wheelEvent emits navigateRequested,
        # which LoupeDialog connects straight to _on_wheel_navigate.
        dialog.index = 0
        assert dialog._current_path() == str(tmp_path / "a.jpg")
        dialog._on_wheel_navigate(1)  # +1 = next, the same signal a real wheel-forward sends
        assert dialog._current_path() == str(tmp_path / "b.jpg")
        assert dialog._burst_best_label.text() == "Best Image: Yes"  # b.jpg is rank #1
        dialog._on_wheel_navigate(1)
        assert dialog._current_path() == str(tmp_path / "c.jpg")
        dialog._on_wheel_navigate(-1)  # -1 = previous
        assert dialog._current_path() == str(tmp_path / "b.jpg")
        dialog.close()
    finally:
        window.close()
        service.close()


def test_changing_color_source_re_ranks_bursts_by_that_strategy(app, tmp_path, monkeypatch) -> None:
    """The reported requirement: burst_rank/burst_best follow the selected
    ranking strategy - here, the same Color Source selector Part 2 added."""
    from picklikeme.desktop.main_window import DEFAULT_STRATEGY_ID

    window, service = _window(tmp_path)
    try:
        calls = []
        monkeypatch.setattr(
            service, "set_burst_strategy",
            lambda strategy_id: (calls.append(strategy_id), service.load_session())[1],
        )

        window._color_combo.setCurrentIndex(0)  # "Review Status" (None)
        window._on_color_source_changed(0)
        assert calls[-1] == DEFAULT_STRATEGY_ID  # None falls back to the AI model

        classic_index = window._color_combo.findData("classic-vision")
        window._color_combo.setCurrentIndex(classic_index)
        window._on_color_source_changed(classic_index)
        assert calls[-1] == "classic-vision"
    finally:
        window.close()
        service.close()


def test_burst_sort_mode_is_remembered_via_qsettings_across_windows(app, tmp_path, monkeypatch) -> None:
    """Burst Order now lives on MainWindow (see BURST_SORT_SETTINGS_KEY) -
    persisted the moment it's changed (_set_burst_sort_mode), same as every
    other QSettings-backed preference, so a photographer only has to pick it
    once. A second MainWindow reading the same settings file starts with it
    already applied, no menu interaction required."""
    from PySide6.QtCore import QSettings

    from picklikeme.desktop.main_window import BURST_SORT_BURST_SCORE, BURST_SORT_CAPTURE_TIME, BURST_SORT_SETTINGS_KEY

    _isolate_settings(monkeypatch, tmp_path)

    window, service = _window(tmp_path)
    try:
        assert window._burst_sort_mode == BURST_SORT_BURST_SCORE  # the default
        window._burst_order_capture_time_action.trigger()
        assert window._burst_sort_mode == BURST_SORT_CAPTURE_TIME
    finally:
        window.close()
        service.close()

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    assert settings.value(BURST_SORT_SETTINGS_KEY) == BURST_SORT_CAPTURE_TIME

    window2, service2 = _window(tmp_path)
    try:
        assert window2._burst_sort_mode == BURST_SORT_CAPTURE_TIME
        assert window2._burst_order_capture_time_action.isChecked()
        assert not window2._burst_order_score_action.isChecked()
    finally:
        window2.close()
        service2.close()


# ---------------------------------------------------------------------------
# End-to-end through the REAL burst_analysis/ReviewSession pipeline, not
# hand-built ImageItems with burst_rank/captured_at already assigned by the
# test itself.
#
# Every test above (and every burst-sort test in test_desktop_workflow.py)
# constructs ImageItem objects directly, setting burst_rank/burst_best/
# captured_at to whatever values the test wants - which proves LoupeDialog's
# OWN sort logic is correct given some data, but never exercises whether
# burst_analysis.analyze_bursts (via ReviewSession.burst_info) actually
# PRODUCES a burst_rank that differs from capture order on real data. A
# reported "the toggle has no visible effect" turned out to be exactly that
# gap: when a burst's images were never scored by the ranking strategy
# ReviewSession.burst_strategy is currently keyed to (the gallery's Color
# Source selector - defaults to the AI model), analyze_bursts's stable sort
# falls back to whatever order burst.reconstruct_bursts already sorted the
# members into to detect the burst in the first place - which is capture-
# time order, because reconstruct_bursts sorts by timestamp before grouping
# (see burst.py). Burst Score then LOOKS identical to Capture Time - not a
# LoupeDialog defect, a data one: this burst has simply never been ranked by
# the active Color Source. These tests go through open_folder -> the real
# ReviewSession -> _refresh_from_state -> _open_loupe_for_item, exactly like
# production, to catch this class of gap the hand-built-ImageItem tests
# structurally cannot.
# ---------------------------------------------------------------------------


def _real_burst_folder(tmp_path, *, filenames_in_capture_order: list[str]):
    """Filenames deliberately NOT in capture order - DSC_0002 was shot
    first, DSC_0003 second, DSC_0001 last - so capture-time order and
    filename/insertion order are two clearly different, independently
    checkable sequences."""
    from PIL import Image

    folder = tmp_path / "shoot"
    folder.mkdir(exist_ok=True)
    for index, fname in enumerate(filenames_in_capture_order):
        Image.new("RGB", (16, 16), color=("red", "green", "blue")[index % 3]).save(folder / fname, format="JPEG")
    return folder


def _open_burst_loupe(window, monkeypatch):
    """Opens the Loupe on the one visible (collapsed) burst card through the
    real MainWindow._open_loupe_for_item, capturing the constructed
    LoupeDialog (a thin exec()-stubbing subclass of the real class, not a
    stand-in - see _spy_loupe_dialog)."""
    captured = _spy_loupe_dialog(monkeypatch)
    (visible,) = window._gallery_model.items()
    window._open_loupe_for_item(visible)
    return captured["dialog"]


def test_an_unscored_burst_falls_back_to_capture_time_and_sets_status(app, tmp_path, monkeypatch) -> None:
    """The confirmed real-world scenario: burst_strategy (defaults to the
    AI model) never scored these images - burst_rank degenerates to capture
    order (see burst_analysis.py), so the Main Grid's default Burst Score
    preference silently produces the SAME sequence Capture Time would, and
    _open_loupe_for_item surfaces the fallback as a status-bar message
    rather than opening a Loupe that quietly ignores the setting."""
    from picklikeme.desktop.main_window import BURST_SORT_BURST_SCORE, BURST_SORT_CAPTURE_TIME

    _isolate_settings(monkeypatch, tmp_path)
    folder = _real_burst_folder(tmp_path, filenames_in_capture_order=["DSC_0001.jpg", "DSC_0002.jpg", "DSC_0003.jpg"])
    window, service = _window(tmp_path)
    try:
        service.open_folder(folder)
        # captured_at deliberately scrambled relative to filename, so a
        # match against filename order (not capture order) would be visible.
        service.session._image_for(str(folder / "DSC_0001.jpg")).captured_at = "2024-01-01T10:00:02"
        service.session._image_for(str(folder / "DSC_0002.jpg")).captured_at = "2024-01-01T10:00:00"
        service.session._image_for(str(folder / "DSC_0003.jpg")).captured_at = "2024-01-01T10:00:01"
        # No ranking_results assigned anywhere - burst_strategy (AI model)
        # has never scored any of these three images.

        window.state.current_folder = str(folder)
        window._refresh_from_state(service.load_session())
        window._on_toggle_collapse_bursts(True)

        assert window._burst_sort_mode == BURST_SORT_BURST_SCORE  # the default, unchanged
        dialog = _open_burst_loupe(window, monkeypatch)
        try:
            assert dialog._burst_ranking_status_label.text() == "Ranking: Not available"
            assert "Burst Score unavailable" in window.state.status_message
            score_mode_sequence = [Path(p).name for p in dialog.image_paths]
            # This is the data limitation itself, printed and pinned so a
            # future change to analyze_bursts's tie-break is a deliberate,
            # visible decision rather than a silent behaviour change.
            assert score_mode_sequence == ["DSC_0002.jpg", "DSC_0003.jpg", "DSC_0001.jpg"]
        finally:
            dialog.close()

        # Explicitly requesting Capture Time (not relying on the fallback)
        # must produce the identical sequence - "both coincide" for this
        # unscored burst, matching the fallback's own logic.
        window._set_burst_sort_mode(BURST_SORT_CAPTURE_TIME)
        dialog2 = _open_burst_loupe(window, monkeypatch)
        try:
            assert [Path(p).name for p in dialog2.image_paths] == score_mode_sequence
        finally:
            dialog2.close()
    finally:
        window.close()
        service.close()


def test_a_scored_burst_shows_genuinely_different_sequences_per_mode(app, tmp_path, monkeypatch) -> None:
    """The contrasting case: once burst_strategy HAS scored these images,
    burst_rank carries real information and the two Burst Order modes
    produce genuinely different navigation orders, with no fallback status
    message - proving the Main Grid's sort logic itself is correct once the
    underlying data actually differs."""
    from picklikeme.desktop.main_window import BURST_SORT_CAPTURE_TIME
    from picklikeme.sidecar import AI_STRATEGY_ID

    _isolate_settings(monkeypatch, tmp_path)
    folder = _real_burst_folder(tmp_path, filenames_in_capture_order=["DSC_0001.jpg", "DSC_0002.jpg", "DSC_0003.jpg"])
    window, service = _window(tmp_path)
    try:
        service.open_folder(folder)
        scores = {"DSC_0001.jpg": 0.95, "DSC_0002.jpg": 0.40, "DSC_0003.jpg": 0.70}
        captured_at = {
            "DSC_0001.jpg": "2024-01-01T10:00:02",
            "DSC_0002.jpg": "2024-01-01T10:00:00",
            "DSC_0003.jpg": "2024-01-01T10:00:01",
        }
        for fname in scores:
            image = service.session._image_for(str(folder / fname))
            image.captured_at = captured_at[fname]
            image.ranking_results[AI_STRATEGY_ID] = {"score": scores[fname], "rank": None}

        window.state.current_folder = str(folder)
        window._refresh_from_state(service.load_session())
        window._on_toggle_collapse_bursts(True)
        window.state.status_message = ""

        # Default mode (Burst Score).
        dialog = _open_burst_loupe(window, monkeypatch)
        try:
            assert dialog._burst_ranking_status_label.text() == "Ranking: Available"
            assert window.state.status_message == ""  # no fallback needed
            burst_score_sequence = [Path(p).name for p in dialog.image_paths]
            assert burst_score_sequence == ["DSC_0001.jpg", "DSC_0003.jpg", "DSC_0002.jpg"]
        finally:
            dialog.close()

        window._set_burst_sort_mode(BURST_SORT_CAPTURE_TIME)
        dialog2 = _open_burst_loupe(window, monkeypatch)
        try:
            capture_time_sequence = [Path(p).name for p in dialog2.image_paths]
            assert capture_time_sequence == ["DSC_0002.jpg", "DSC_0003.jpg", "DSC_0001.jpg"]
            assert capture_time_sequence != burst_score_sequence
        finally:
            dialog2.close()
    finally:
        window.close()
        service.close()


def test_repeated_burst_order_switching_never_drifts_from_the_original_list(app, tmp_path, monkeypatch) -> None:
    """Regression test for a real bug found once Color Source matched the
    ranking strategy: Capture Time -> Burst Score -> Capture Time did not
    restore chronological order - it stayed in Burst Score order.

    Root cause (in the old Loupe-internal implementation): it re-sorted
    self.items IN PLACE on every mode change. list.sort() is stable, so
    re-sorting an already-Burst-Score-ordered list by captured_at only
    reorders images with genuinely different timestamps - two images
    sharing the same EXIF second (DSC_0002/DSC_0003 below - realistic for a
    fast burst, since DateTimeOriginal is only 1-second resolution) kept
    whatever relative order the PREVIOUS sort left them in, not the true
    original one.

    MainWindow._sort_burst_members is a `@staticmethod` that always derives
    fresh via `sorted()` from the list _open_loupe_for_item builds fresh
    from self._all_items on every call - there is no persisted "current
    order" for a later sort to accidentally build on top of, so this class
    of bug is structurally not reachable here. This test exercises the
    exact same tie the original regression used, now against three
    independently opened Loupe sessions instead of one Loupe's internal
    toggle.
    """
    from PIL import Image

    from picklikeme.desktop.main_window import BURST_SORT_BURST_SCORE, BURST_SORT_CAPTURE_TIME
    from picklikeme.sidecar import AI_STRATEGY_ID

    _isolate_settings(monkeypatch, tmp_path)
    folder = tmp_path / "shoot"
    folder.mkdir()
    filenames = ["DSC_0001.jpg", "DSC_0002.jpg", "DSC_0003.jpg", "DSC_0004.jpg"]
    for index, fname in enumerate(filenames):
        Image.new(
            "RGB", (16, 16), color=("red", "green", "blue", "yellow")[index]
        ).save(folder / fname, format="JPEG")

    window, service = _window(tmp_path)
    try:
        service.open_folder(folder)
        scores = {"DSC_0001.jpg": 0.95, "DSC_0002.jpg": 0.40, "DSC_0003.jpg": 0.70, "DSC_0004.jpg": 0.55}
        captured_at = {
            "DSC_0001.jpg": "2024-01-01T10:00:02",
            "DSC_0002.jpg": "2024-01-01T10:00:00",  # tied with DSC_0003 - the case that broke
            "DSC_0003.jpg": "2024-01-01T10:00:00",  # in-place re-sorting
            "DSC_0004.jpg": "2024-01-01T10:00:03",
        }
        for fname in filenames:
            image = service.session._image_for(str(folder / fname))
            image.captured_at = captured_at[fname]
            image.ranking_results[AI_STRATEGY_ID] = {"score": scores[fname], "rank": None}

        window.state.current_folder = str(folder)
        window._refresh_from_state(service.load_session())
        window._on_toggle_collapse_bursts(True)

        def sequence_for(mode):
            window._set_burst_sort_mode(mode)
            dialog = _open_burst_loupe(window, monkeypatch)
            try:
                return [Path(p).name for p in dialog.image_paths]
            finally:
                dialog.close()

        capture_time_first = sequence_for(BURST_SORT_CAPTURE_TIME)
        burst_score = sequence_for(BURST_SORT_BURST_SCORE)
        capture_time_again = sequence_for(BURST_SORT_CAPTURE_TIME)

        # DSC_0002/DSC_0003 tie on captured_at - Python's sorted() is stable,
        # so the tie resolves to whichever order they were already in within
        # self._all_items (filename order here) BEFORE this sort, same as
        # burst_score's own distinct-score ordering has no ties to resolve.
        assert capture_time_first == ["DSC_0002.jpg", "DSC_0003.jpg", "DSC_0001.jpg", "DSC_0004.jpg"]
        assert burst_score == ["DSC_0001.jpg", "DSC_0003.jpg", "DSC_0004.jpg", "DSC_0002.jpg"]
        # The actual regression: this must be byte-for-byte identical to
        # capture_time_first, not left in Burst Score's order.
        assert capture_time_again == capture_time_first
    finally:
        window.close()
        service.close()
