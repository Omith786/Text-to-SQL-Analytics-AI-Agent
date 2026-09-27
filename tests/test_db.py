"""Tests for the DuckDB access layer."""

import pytest

from text2sql.db import Database, QueryTimeout
from text2sql.sql_guard import UnsafeSQLError


def test_schema_text_lists_all_tables(db):
    schema = db.schema_text()
    for table in ("vessels", "ports", "port_calls", "energy_prices"):
        assert f"TABLE {table} (" in schema
    assert "gross_tonnage" in schema


def test_run_returns_rows_and_columns(db):
    result = db.run("SELECT COUNT(*) AS n FROM vessels", row_limit=10, timeout_s=5)
    assert result.columns == ["n"]
    assert result.row_count == 1
    assert result.rows[0][0] == 60


def test_run_applies_row_limit(db):
    result = db.run("SELECT * FROM port_calls", row_limit=5, timeout_s=5)
    assert result.row_count == 5
    assert "LIMIT 5" in result.sql


def test_run_rejects_writes(db):
    with pytest.raises(UnsafeSQLError):
        db.run("DELETE FROM vessels", row_limit=10, timeout_s=5)


def test_run_propagates_sql_errors_for_self_correction(db):
    # An unknown column should surface as a DuckDB error, not be swallowed.
    with pytest.raises(Exception) as exc:
        db.run("SELECT nonexistent_column FROM vessels", row_limit=10, timeout_s=5)
    assert not isinstance(exc.value, QueryTimeout)


def test_read_only_connection_blocks_engine_level_write(db):
    # Even bypassing the guard, the read-only engine must refuse a write.
    with pytest.raises(Exception):
        db._con.execute("CREATE TABLE t (a INT)")


def test_missing_database_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Database(tmp_path / "does-not-exist.duckdb")
