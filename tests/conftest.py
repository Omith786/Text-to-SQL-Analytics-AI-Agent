"""Shared pytest fixtures.

Adds ``src`` to the import path (so tests run without installing the package)
and provides a session-scoped sample database built into a temporary directory.
No test contacts a network LLM: the agent tests use ``FakeChatClient``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text2sql.db import Database  # noqa: E402
from text2sql.sample_data import build  # noqa: E402


@pytest.fixture(scope="session")
def sample_db_path(tmp_path_factory) -> Path:
    """Build the seeded sample database once for the whole test session."""
    path = tmp_path_factory.mktemp("db") / "sample.duckdb"
    return build(path)


@pytest.fixture()
def db(sample_db_path) -> Database:
    """A read-only connection to the sample database, closed after each test."""
    connection = Database(sample_db_path)
    yield connection
    connection.close()
