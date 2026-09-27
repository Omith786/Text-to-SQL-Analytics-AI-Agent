"""Tests for the sample database generator."""

from text2sql.db import Database
from text2sql.sample_data import build, ensure


def test_build_is_deterministic(tmp_path):
    p1 = build(tmp_path / "a.duckdb")
    p2 = build(tmp_path / "b.duckdb")
    with Database(p1) as d1, Database(p2) as d2:
        q = "SELECT SUM(cargo_tonnes) AS total FROM port_calls"
        assert d1.run(q, 10, 5).rows == d2.run(q, 10, 5).rows


def test_build_populates_expected_row_counts(tmp_path):
    path = build(tmp_path / "s.duckdb")
    with Database(path) as db:
        assert db.run("SELECT COUNT(*) FROM vessels", 10, 5).rows[0][0] == 60
        assert db.run("SELECT COUNT(*) FROM ports", 10, 5).rows[0][0] == 8
        assert db.run("SELECT COUNT(*) FROM port_calls", 10, 5).rows[0][0] == 900


def test_ensure_builds_only_when_missing(tmp_path):
    path = tmp_path / "e.duckdb"
    assert not path.exists()
    ensure(path)
    assert path.exists()
    mtime = path.stat().st_mtime_ns
    ensure(path)  # second call must not rebuild
    assert path.stat().st_mtime_ns == mtime
