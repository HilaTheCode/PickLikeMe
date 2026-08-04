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
    collapsed gallery, and not left/right-file order."""
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


def test_burst_sort_toggle_actually_changes_loupe_navigation_order(app, tmp_path, monkeypatch) -> None:
    """Regression test for a reported bug: the Loupe's Capture Time / Burst
    Score combo appeared and Burst Rank displayed, but switching between
    the two modes never actually changed which image Next/Prev (or any
    other navigation) visited - the Loupe always stayed in capture-time
    order.

    Exercises the REAL LoupeDialog (a thin exec()-stubbing subclass, not a
    captured-kwargs stand-in like the other tests in this file), constructed
    through the REAL MainWindow._open_loupe_for_item code path, against a
    burst whose burst_rank order and captured_at order genuinely disagree -
    per the report's own instruction to validate with a burst where the two
    orderings differ.
    """
    from PIL import Image

    from picklikeme.desktop import main_window as main_window_module
    from picklikeme.desktop.dialogs.loupe_dialog import BURST_SORT_CAPTURE_TIME, LoupeDialog
    from picklikeme.desktop.models.image_item import ImageItem

    # An isolated, file-backed QSettings - MainWindow always constructs its
    # own QSettings("PeakPic", "PeakPicDesktop") internally (see __init__),
    # which would otherwise read/write the real OS-level app settings
    # during a test run.
    from PySide6.QtCore import QSettings

    settings_path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        main_window_module, "QSettings",
        lambda *a, **k: QSettings(settings_path, QSettings.Format.IniFormat),
    )

    captured = {}

    class _SpyLoupeDialog(LoupeDialog):
        def exec(self):  # noqa: A003 - matches QDialog's own method name
            from PySide6.QtWidgets import QDialog

            captured["dialog"] = self
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(main_window_module, "LoupeDialog", _SpyLoupeDialog)

    window, service = _window(tmp_path)
    try:
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            Image.new("RGB", (16, 16), color="blue").save(tmp_path / name, format="JPEG")

        # ranking_results matches burst_rank (rank #1 = highest score) - a
        # burst with burst_rank set but NO actual score behind it is exactly
        # the "unavailable" case _burst_score_available now detects (see
        # test_an_unscored_burst_shows_the_no_effect_warning_and_both_
        # sequences_coincide below); this test is specifically about a
        # burst that HAS been scored.
        window._all_items = [
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
        window._on_toggle_collapse_bursts(True)
        (visible,) = window._gallery_model.items()
        window._open_loupe_for_item(visible)

        dialog = captured["dialog"]
        # Default mode is Burst Score - matches burst_rank order (b, a, c),
        # NOT capture-time order (a, b, c).
        assert dialog.image_paths == [str(tmp_path / n) for n in ("b.jpg", "a.jpg", "c.jpg")]
        assert dialog._current_path() == str(tmp_path / "b.jpg")

        combo_index = dialog._burst_sort_combo.findData(BURST_SORT_CAPTURE_TIME)
        dialog._burst_sort_combo.setCurrentIndex(combo_index)

        # The reorder itself.
        assert dialog.image_paths == [str(tmp_path / n) for n in ("a.jpg", "b.jpg", "c.jpg")]
        # Navigation (Next/Prev, the same code path wheel/keyboard use) must
        # walk the NEW order, not the order the dialog was first opened
        # with - this is the actual bug: only image_paths/labels updated.
        dialog.index = 0
        assert dialog._current_path() == str(tmp_path / "a.jpg")
        dialog._go_next()
        assert dialog._current_path() == str(tmp_path / "b.jpg")
        dialog._go_next()
        assert dialog._current_path() == str(tmp_path / "c.jpg")
    finally:
        dialog_obj = captured.get("dialog")
        if dialog_obj is not None:
            dialog_obj.close()
        window.close()
        service.close()


def test_burst_sort_survives_real_gallery_click_combo_popup_and_wheel(app, tmp_path, monkeypatch) -> None:
    """The same regression as the test above, re-verified through paths a
    unit test calling internal methods directly could still miss:

    - opening the Loupe via the REAL gallery double-click *signal*
      (QAbstractItemView.doubleClicked.emit), not by calling
      _open_loupe_for_item directly - in case MainWindow wired the signal
      to something else.
    - picking "Capture Time" via a REAL simulated mouse click on the
      combo's own popup view (QTest.mouseClick), not
      combo.setCurrentIndex() - in case Qt's signal only fires for one and
      not the other.
    - advancing via _on_wheel_navigate (what _ZoomView.wheelEvent actually
      emits into) rather than only _go_next/_go_prev directly - the
      report's own explicit ask: "verify that mouse wheel... follow the
      selected sort mode."

    If this test and the one above both pass, the sort toggle's entire
    data flow - click to open, click to change sort, wheel to navigate -
    is proven correct, not merely each internal method in isolation.
    """
    from PIL import Image
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtTest import QTest

    from picklikeme.desktop import main_window as main_window_module
    from picklikeme.desktop.dialogs.loupe_dialog import BURST_SORT_CAPTURE_TIME, LoupeDialog
    from picklikeme.desktop.models.image_item import ImageItem

    settings_path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        main_window_module, "QSettings",
        lambda *a, **k: QSettings(settings_path, QSettings.Format.IniFormat),
    )

    captured = {}

    class _SpyLoupeDialog(LoupeDialog):
        def exec(self):  # noqa: A003 - matches QDialog's own method name
            from PySide6.QtWidgets import QDialog

            captured["dialog"] = self
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(main_window_module, "LoupeDialog", _SpyLoupeDialog)

    window, service = _window(tmp_path)
    try:
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            Image.new("RGB", (16, 16), color="blue").save(tmp_path / name, format="JPEG")

        # ranking_results matches burst_rank (rank #1 = highest score) - a
        # burst with burst_rank set but NO actual score behind it is exactly
        # the "unavailable" case _burst_score_available now detects (see
        # test_an_unscored_burst_shows_the_no_effect_warning_and_both_
        # sequences_coincide below); this test is specifically about a
        # burst that HAS been scored.
        window._all_items = [
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
        window._on_toggle_collapse_bursts(True)

        # Real double-click *signal*, exactly what MainWindow.__init__ wires
        # the gallery view to (self._gallery_view.doubleClicked.connect(
        # self._open_loupe_for_index)) - not a direct method call.
        index = window._gallery_model.index(0, 0)
        window._gallery_view.doubleClicked.emit(index)

        dialog = captured["dialog"]
        assert dialog.image_paths == [str(tmp_path / n) for n in ("b.jpg", "a.jpg", "c.jpg")]
        assert dialog._burst_best_label.text() == "Best Image: Yes"  # b.jpg, rank #1

        # A real simulated mouse click on the combo's own popup - not
        # setCurrentIndex().
        combo = dialog._burst_sort_combo
        combo.showPopup()
        item_index = combo.model().index(combo.findData(BURST_SORT_CAPTURE_TIME), 0)
        rect = combo.view().visualRect(item_index)
        QTest.mouseClick(combo.view().viewport(), Qt.MouseButton.LeftButton, pos=rect.center())

        assert combo.currentData() == BURST_SORT_CAPTURE_TIME
        assert dialog.image_paths == [str(tmp_path / n) for n in ("a.jpg", "b.jpg", "c.jpg")]
        # b.jpg (rank #1, still Best) is now at position 2 of 3 in
        # capture-time order - the label must reflect its NEW position,
        # not the position it had under the old sort.
        assert dialog._burst_rank_label.text() == "Burst Rank #1 of 3"
        assert dialog._burst_best_label.text() == "Best Image: Yes"

        # Wheel navigation - _ZoomView.wheelEvent emits navigateRequested,
        # which LoupeDialog connects straight to _on_wheel_navigate.
        dialog.index = 0
        assert dialog._current_path() == str(tmp_path / "a.jpg")
        dialog._on_wheel_navigate(1)  # +1 = next, the same signal a real wheel-forward sends
        assert dialog._current_path() == str(tmp_path / "b.jpg")
        assert dialog._burst_best_label.text() == "Best Image: Yes"
        dialog._on_wheel_navigate(1)
        assert dialog._current_path() == str(tmp_path / "c.jpg")
        assert dialog._burst_best_label.text() == "Best Image: No"
        dialog._on_wheel_navigate(-1)  # -1 = previous
        assert dialog._current_path() == str(tmp_path / "b.jpg")
    finally:
        dialog_obj = captured.get("dialog")
        if dialog_obj is not None:
            dialog_obj.close()
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
    stand-in - see test_burst_sort_toggle_actually_changes_loupe_navigation_
    order's own docstring for why)."""
    from picklikeme.desktop import main_window as main_window_module
    from picklikeme.desktop.dialogs.loupe_dialog import LoupeDialog

    captured = {}

    class _SpyLoupeDialog(LoupeDialog):
        def exec(self):  # noqa: A003 - matches QDialog's own method name
            from PySide6.QtWidgets import QDialog

            captured["dialog"] = self
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(main_window_module, "LoupeDialog", _SpyLoupeDialog)
    (visible,) = window._gallery_model.items()
    window._open_loupe_for_item(visible)
    return captured["dialog"]


def test_an_unscored_burst_shows_the_no_effect_warning_and_both_sequences_coincide(
    app, tmp_path, monkeypatch
) -> None:
    """The confirmed real-world scenario: burst_strategy (defaults to the
    AI model) never scored these images - burst_rank degenerates to capture
    order, so both sort modes produce the SAME sequence. LoupeDialog now
    surfaces this explicitly rather than silently looking broken."""
    from picklikeme.desktop.dialogs.loupe_dialog import BURST_SORT_CAPTURE_TIME, BURST_SORT_BURST_SCORE

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

        dialog = _open_burst_loupe(window, monkeypatch)
        try:
            assert "Burst Score unavailable" in dialog._burst_sort_warning_label.text()
            assert dialog._burst_ranking_status_label.text() == "Ranking: Not available"

            combo = dialog._burst_sort_combo
            combo.setCurrentIndex(combo.findData(BURST_SORT_CAPTURE_TIME))
            capture_time_sequence = [Path(p).name for p in dialog.image_paths]
            combo.setCurrentIndex(combo.findData(BURST_SORT_BURST_SCORE))
            burst_score_sequence = [Path(p).name for p in dialog.image_paths]

            # This is the data limitation itself, printed and pinned so a
            # future change to analyze_bursts's tie-break is a deliberate,
            # visible decision rather than a silent behaviour change.
            assert capture_time_sequence == burst_score_sequence == ["DSC_0002.jpg", "DSC_0003.jpg", "DSC_0001.jpg"]
        finally:
            dialog.close()
    finally:
        window.close()
        service.close()


def test_a_scored_burst_shows_no_warning_and_the_two_sequences_genuinely_differ(app, tmp_path, monkeypatch) -> None:
    """The contrasting case: once burst_strategy HAS scored these images,
    burst_rank carries real information and the two sort modes produce
    genuinely different navigation orders - proving LoupeDialog's toggle
    itself is correct once the underlying data actually differs."""
    from picklikeme.desktop.dialogs.loupe_dialog import BURST_SORT_CAPTURE_TIME, BURST_SORT_BURST_SCORE
    from picklikeme.sidecar import AI_STRATEGY_ID

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

        dialog = _open_burst_loupe(window, monkeypatch)
        try:
            assert dialog._burst_sort_warning_label.text() == ""

            combo = dialog._burst_sort_combo
            combo.setCurrentIndex(combo.findData(BURST_SORT_CAPTURE_TIME))
            capture_time_sequence = [Path(p).name for p in dialog.image_paths]
            combo.setCurrentIndex(combo.findData(BURST_SORT_BURST_SCORE))
            burst_score_sequence = [Path(p).name for p in dialog.image_paths]

            assert capture_time_sequence == ["DSC_0002.jpg", "DSC_0003.jpg", "DSC_0001.jpg"]
            assert burst_score_sequence == ["DSC_0001.jpg", "DSC_0003.jpg", "DSC_0002.jpg"]
            assert capture_time_sequence != burst_score_sequence
        finally:
            dialog.close()
    finally:
        window.close()
        service.close()


def test_switching_capture_time_burst_score_and_back_restores_the_exact_original_order(
    app, tmp_path, monkeypatch
) -> None:
    """Regression test for a real bug found once Color Source matched the
    ranking strategy: Capture Time -> Burst Score -> Capture Time did not
    restore chronological order - it stayed in Burst Score order.

    Root cause: the previous implementation re-sorted self.items IN PLACE
    on every mode change. list.sort() is stable, so re-sorting an
    already-Burst-Score-ordered list by captured_at only reorders images
    with genuinely different timestamps - two images sharing the same EXIF
    second (DSC_0002/DSC_0003 below - realistic for a fast burst, since
    DateTimeOriginal is only 1-second resolution) kept whatever relative
    order the PREVIOUS sort left them in, not the true original one. The
    fix rebuilds every mode fresh from an immutable `_burst_members_original`
    list, making each mode a pure function of fixed input regardless of
    what was selected in between - this test exercises exactly that tie.
    """
    from PIL import Image

    from picklikeme.desktop.dialogs.loupe_dialog import BURST_SORT_CAPTURE_TIME, BURST_SORT_BURST_SCORE
    from picklikeme.sidecar import AI_STRATEGY_ID

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

        dialog = _open_burst_loupe(window, monkeypatch)
        try:
            combo = dialog._burst_sort_combo

            def sequence_for(mode):
                combo.setCurrentIndex(combo.findData(mode))
                return [Path(p).name for p in dialog.image_paths]

            capture_time_first = sequence_for(BURST_SORT_CAPTURE_TIME)
            burst_score = sequence_for(BURST_SORT_BURST_SCORE)
            capture_time_again = sequence_for(BURST_SORT_CAPTURE_TIME)

            assert capture_time_first == ["DSC_0003.jpg", "DSC_0002.jpg", "DSC_0001.jpg", "DSC_0004.jpg"]
            assert burst_score == ["DSC_0001.jpg", "DSC_0003.jpg", "DSC_0004.jpg", "DSC_0002.jpg"]
            # The actual regression: this must be byte-for-byte identical to
            # capture_time_first, not left in Burst Score's order.
            assert capture_time_again == capture_time_first
        finally:
            dialog.close()
    finally:
        window.close()
        service.close()
