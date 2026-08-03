"""MainWindow._filter_items's generalized "algorithm_*" conflict filters -
the same AI-vs-user conflict filters, generalized to compare the user's
decision against whichever strategy is currently selected (Color Source /
ReviewSession.burst_strategy), not only the AI model. `_filter_items` is a
staticmethod - tested directly against ImageItem instances, no QApplication
or window construction needed.
"""

from __future__ import annotations

import pytest

try:
    from picklikeme.desktop.main_window import FILTER_LABELS, FILTERS, MainWindow
except ImportError:  # pragma: no cover
    MainWindow = None

from picklikeme.desktop.models.image_item import ImageItem


def _item(path, *, ai="keep", algorithm="keep", status="neutral") -> ImageItem:
    return ImageItem(path=path, file_name=path, review_status=status, ai_suggestion=ai, algorithm_suggestion=algorithm)


@pytest.mark.skipif(MainWindow is None, reason="PySide6 not installed")
class TestGeneralizedConflictFilters:
    def test_algorithm_keep_reads_algorithm_suggestion_not_ai_suggestion(self):
        items = [
            _item("a.jpg", ai="keep", algorithm="reject"),
            _item("b.jpg", ai="reject", algorithm="keep"),
        ]
        result = MainWindow._filter_items(items, "algorithm_keep")
        assert [i.path for i in result] == ["b.jpg"]

    def test_algorithm_reject_reads_algorithm_suggestion(self):
        items = [
            _item("a.jpg", ai="keep", algorithm="reject"),
            _item("b.jpg", ai="reject", algorithm="keep"),
        ]
        result = MainWindow._filter_items(items, "algorithm_reject")
        assert [i.path for i in result] == ["a.jpg"]

    def test_algorithm_keep_user_reject_conflict(self):
        items = [
            _item("a.jpg", algorithm="keep", status="reject"),  # conflict
            _item("b.jpg", algorithm="keep", status="keep"),  # agreement, not a conflict
            _item("c.jpg", algorithm="reject", status="reject"),  # agreement
        ]
        result = MainWindow._filter_items(items, "algorithm_keep_user_reject")
        assert [i.path for i in result] == ["a.jpg"]

    def test_algorithm_reject_user_keep_conflict(self):
        items = [
            _item("a.jpg", algorithm="reject", status="keep"),  # conflict
            _item("b.jpg", algorithm="keep", status="keep"),  # agreement
        ]
        result = MainWindow._filter_items(items, "algorithm_reject_user_keep")
        assert [i.path for i in result] == ["a.jpg"]

    def test_ai_specific_filters_are_unaffected_by_algorithm_suggestion(self):
        """The existing "AI Keep/Reject" filters must always mean the AI
        model specifically, regardless of what Color Source is selected -
        see FILTER_LABELS's own comment on why these were deliberately
        left alone rather than redefined."""
        items = [_item("a.jpg", ai="keep", algorithm="reject")]
        assert [i.path for i in MainWindow._filter_items(items, "ai_keep")] == ["a.jpg"]
        assert MainWindow._filter_items(items, "algorithm_keep") == []

    def test_every_declared_filter_has_a_label(self):
        assert set(FILTERS) == set(FILTER_LABELS)

    def test_an_item_with_no_algorithm_suggestion_matches_neither_algorithm_filter(self):
        items = [_item("a.jpg", algorithm=None)]
        assert MainWindow._filter_items(items, "algorithm_keep") == []
        assert MainWindow._filter_items(items, "algorithm_reject") == []
