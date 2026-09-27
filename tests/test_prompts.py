"""Tests for prompt construction and SQL extraction."""

import pytest

from text2sql.prompts import (
    STRATEGIES,
    build_correction_message,
    build_messages,
    extract_sql,
)


def test_zero_shot_omits_examples():
    messages = build_messages("How many vessels?", "TABLE vessels (mmsi BIGINT)", "zero_shot")
    user = messages[-1]["content"]
    assert "Examples:" not in user
    assert "How many vessels?" in user


def test_schema_examples_includes_examples():
    messages = build_messages("How many vessels?", "TABLE vessels (mmsi BIGINT)", "schema_examples")
    assert "Examples:" in messages[-1]["content"]


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        build_messages("q", "schema", "made_up_strategy")


def test_all_strategies_build():
    for strategy in STRATEGIES:
        assert build_messages("q", "schema", strategy)


@pytest.mark.parametrize("text,expected_start", [
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("Here you go:\n```\nSELECT 2\n```", "SELECT 2"),
    ("SELECT 3 FROM t", "SELECT 3"),
    ("WITH t AS (SELECT 1) SELECT * FROM t", "WITH t"),
])
def test_extract_sql(text, expected_start):
    assert extract_sql(text).startswith(expected_start)


def test_extract_sql_returns_empty_when_absent():
    assert extract_sql("I cannot help with that.") == ""


def test_correction_message_contains_error():
    msg = build_correction_message("Catalog Error: column x not found")
    assert "column x not found" in msg["content"]
    assert msg["role"] == "user"
