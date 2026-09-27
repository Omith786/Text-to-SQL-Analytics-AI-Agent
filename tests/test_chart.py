"""Tests for chart rendering."""

from text2sql.chart import is_chartable, render_bar_chart
from text2sql.db import QueryResult


def _result(columns, rows):
    return QueryResult(columns=columns, rows=rows, sql="SELECT ...")


def test_is_chartable_two_column_numeric():
    assert is_chartable(_result(["type", "n"], [("A", 3), ("B", 5)]))


def test_not_chartable_when_three_columns():
    assert not is_chartable(_result(["a", "b", "c"], [(1, 2, 3)]))


def test_not_chartable_when_second_column_non_numeric():
    assert not is_chartable(_result(["a", "b"], [("x", "y")]))


def test_not_chartable_when_empty():
    assert not is_chartable(_result(["a", "b"], []))


def test_render_bar_chart_writes_file(tmp_path):
    out = render_bar_chart(_result(["type", "n"], [("A", 3), ("B", 5)]), tmp_path / "c.png")
    assert out is not None and out.exists()


def test_render_bar_chart_returns_none_for_unchartable(tmp_path):
    assert render_bar_chart(_result(["a", "b", "c"], [(1, 2, 3)]), tmp_path / "c.png") is None
