"""RecentItemsMenu: the reusable "recent items" QMenu behind File > Recent
Folders (and any future Recent Projects list built on the same class - see
desktop/widgets/recent_items.py's module docstring).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QMenu
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    QApplication = None  # type: ignore[assignment]

from picklikeme.desktop.widgets.recent_items import DEFAULT_RECENT_ITEMS_LIMIT, RecentItemsMenu

pytestmark = pytest.mark.skipif(QApplication is None, reason="PySide6 not installed")


def make_menu(tmp_path, *, settings_key="recent_folders", limit=DEFAULT_RECENT_ITEMS_LIMIT, on_select=None, is_valid=None):
    """A menu backed by a throwaway INI file, so tests never touch the real
    app's QSettings (registry on Windows) and never see state left over from
    other tests or runs."""
    QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    menu = QMenu()
    selected: list[str] = []
    recent = RecentItemsMenu(
        menu,
        settings,
        settings_key=settings_key,
        on_select=on_select or selected.append,
        empty_label="No recent folders yet",
        clear_label="Clear Recent Folders",
        limit=limit,
        is_valid=is_valid,
    )
    return recent, menu, settings, selected


def test_default_limit_is_five():
    assert DEFAULT_RECENT_ITEMS_LIMIT == 5


def test_starts_empty_with_disabled_menu_and_placeholder(tmp_path):
    recent, menu, _settings, _selected = make_menu(tmp_path)

    assert recent.items() == []
    assert menu.isEnabled() is False
    assert [a.text() for a in menu.actions()] == ["No recent folders yet"]


def test_remembering_items_is_most_recent_first(tmp_path):
    recent, *_ = make_menu(tmp_path)

    recent.remember("a")
    recent.remember("b")
    recent.remember("c")

    assert recent.items() == ["c", "b", "a"]


def test_reopening_an_existing_item_moves_it_to_the_top(tmp_path):
    recent, *_ = make_menu(tmp_path)

    recent.remember("a")
    recent.remember("b")
    recent.remember("c")
    recent.remember("a")

    assert recent.items() == ["a", "c", "b"]


def test_duplicates_are_never_stored_twice(tmp_path):
    recent, *_ = make_menu(tmp_path)

    recent.remember("a")
    recent.remember("a")
    recent.remember("a")

    assert recent.items() == ["a"]


def test_list_is_trimmed_to_the_configured_limit(tmp_path):
    recent, *_ = make_menu(tmp_path, limit=3)

    for item in ["a", "b", "c", "d", "e"]:
        recent.remember(item)

    assert recent.items() == ["e", "d", "c"]


def test_default_limit_applies_when_not_overridden(tmp_path):
    recent, *_ = make_menu(tmp_path)

    for i in range(8):
        recent.remember(f"folder-{i}")

    assert len(recent.items()) == DEFAULT_RECENT_ITEMS_LIMIT
    assert recent.items()[0] == "folder-7"


def test_persists_between_instances_via_qsettings(tmp_path):
    recent, _menu, settings, _selected = make_menu(tmp_path)
    recent.remember("a")
    recent.remember("b")

    menu2 = QMenu()
    reopened = RecentItemsMenu(menu2, settings, settings_key="recent_folders", on_select=lambda item: None)

    assert reopened.items() == ["b", "a"]


def test_selecting_a_menu_item_invokes_on_select_immediately(tmp_path):
    recent, menu, _settings, selected = make_menu(tmp_path)
    recent.remember(str(Path("some") / "folder"))

    action = menu.actions()[0]
    action.trigger()

    assert selected == [str(Path("some") / "folder")]


def test_full_path_is_always_available_as_a_tooltip(tmp_path):
    recent, menu, *_ = make_menu(tmp_path)
    long_path = str(Path("/very/deeply/nested") / ("segment_" * 40) / "photos")
    recent.remember(long_path)

    action = menu.actions()[0]
    assert action.toolTip() == long_path


def test_clear_action_is_last_and_empties_the_list(tmp_path):
    recent, menu, _settings, _selected = make_menu(tmp_path)
    recent.remember("a")
    recent.remember("b")

    actions = menu.actions()
    assert actions[-1].text() == "Clear Recent Folders"

    actions[-1].trigger()

    assert recent.items() == []
    assert menu.isEnabled() is False
    assert [a.text() for a in menu.actions()] == ["No recent folders yet"]


def test_remove_drops_a_single_item_without_touching_the_rest(tmp_path):
    recent, *_ = make_menu(tmp_path)
    recent.remember("a")
    recent.remember("b")
    recent.remember("c")

    recent.remove("b")

    assert recent.items() == ["c", "a"]


def test_remove_of_an_absent_item_is_a_no_op(tmp_path):
    recent, *_ = make_menu(tmp_path)
    recent.remember("a")

    recent.remove("not-there")

    assert recent.items() == ["a"]


def test_reload_reflects_settings_changed_by_someone_else(tmp_path):
    recent, _menu, settings, _selected = make_menu(tmp_path)
    recent.remember("a")

    settings.setValue("recent_folders", ["x", "y"])
    recent.reload()

    assert recent.items() == ["x", "y"]


def test_the_class_is_generic_enough_for_a_differently_keyed_list(tmp_path):
    """Not folder-specific: the same class, pointed at a different
    settings_key, is exactly what a future Recent Projects menu would reuse
    unchanged. Both instances share one QSettings store (same tmp_path), so
    this also proves the two lists never cross-talk."""
    projects, _menu, _settings, _selected = make_menu(tmp_path, settings_key="recent_projects")
    folders, *_ = make_menu(tmp_path, settings_key="recent_folders")

    projects.remember("project-a.peakpic")
    folders.reload()

    assert projects.items() == ["project-a.peakpic"]
    assert folders.items() == []  # independent keys, no cross-talk


# ---------------------------------------------------------------------------
# Self-healing (Manual QA Issue 1): a persisted entry that fails `is_valid`
# disappears on reload, and stays gone - it is pruned from QSettings itself,
# not just filtered out of this one in-memory view.
# ---------------------------------------------------------------------------


def test_an_invalid_persisted_entry_is_pruned_on_reload(tmp_path):
    real_dir = tmp_path / "real_folder"
    real_dir.mkdir()
    recent, _menu, settings, _selected = make_menu(tmp_path, is_valid=lambda p: Path(p).is_dir())
    # Simulate a leftover entry from before this validation existed, or a
    # folder that has since been deleted/renamed - written directly to
    # QSettings, bypassing remember() entirely, the same way stale state
    # from an older app version would already be sitting there.
    settings.setValue("recent_folders", [str(real_dir), str(tmp_path / "vanished_folder")])

    recent.reload()

    assert recent.items() == [str(real_dir)]


def test_the_pruned_list_is_what_gets_persisted_back(tmp_path):
    """The prune must be written back to QSettings immediately, not just
    held in memory - otherwise the invalid entry would keep reappearing on
    every future reload (e.g. every app launch)."""
    real_dir = tmp_path / "real_folder"
    real_dir.mkdir()
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    settings.setValue("recent_folders", [str(real_dir), str(tmp_path / "vanished_folder")])
    QApplication.instance() or QApplication([])
    menu = QMenu()
    RecentItemsMenu(
        menu, settings, settings_key="recent_folders", on_select=lambda item: None,
        is_valid=lambda p: Path(p).is_dir(),
    )

    persisted = settings.value("recent_folders")
    assert persisted == [str(real_dir)]


def test_no_is_valid_means_every_persisted_entry_is_kept(tmp_path):
    """Every use of this class before is_valid existed must behave exactly
    as before when the parameter is omitted - it defaults to None, which
    means "nothing gets pruned"."""
    recent, _menu, settings, _selected = make_menu(tmp_path)
    settings.setValue("recent_folders", [str(tmp_path / "does_not_exist")])

    recent.reload()

    assert recent.items() == [str(tmp_path / "does_not_exist")]
