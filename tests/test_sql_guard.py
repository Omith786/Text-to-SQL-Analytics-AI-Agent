"""Tests for the read-only SQL guard."""

import pytest

from text2sql.sql_guard import UnsafeSQLError, enforce_limit, validate


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select * from vessels",
    "  SELECT name FROM ports;  ",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
])
def test_validate_accepts_read_only_selects(sql):
    assert validate(sql)


@pytest.mark.parametrize("sql", [
    "INSERT INTO vessels VALUES (1)",
    "UPDATE vessels SET name = 'x'",
    "DELETE FROM vessels",
    "DROP TABLE vessels",
    "CREATE TABLE t (a INT)",
    "ATTACH 'other.db'",
    "PRAGMA database_list",
    "COPY vessels TO 'out.csv'",
])
def test_validate_rejects_writes_and_side_effects(sql):
    with pytest.raises(UnsafeSQLError):
        validate(sql)


def test_validate_rejects_multiple_statements():
    with pytest.raises(UnsafeSQLError):
        validate("SELECT 1; SELECT 2")


def test_validate_rejects_empty():
    with pytest.raises(UnsafeSQLError):
        validate("   ")


def test_validate_ignores_keywords_hidden_in_comments():
    # A DELETE inside a comment must not trip the keyword check, and the real
    # statement is a plain SELECT.
    assert validate("SELECT 1 -- delete everything")


def test_validate_strips_trailing_semicolon():
    assert validate("SELECT 1;") == "SELECT 1"


def test_enforce_limit_appends_when_absent():
    out = enforce_limit("SELECT 1", 100)
    assert out.endswith("LIMIT 100")


def test_enforce_limit_respects_existing_limit():
    assert enforce_limit("SELECT 1 LIMIT 5", 100) == "SELECT 1 LIMIT 5"
