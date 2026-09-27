"""Validation and normalisation of model-generated SQL before execution.

The agent runs SQL that a language model wrote, so every statement is treated as
untrusted. The guard enforces three properties: exactly one statement, that it is
read-only, and that its result set is bounded by a row limit. Timeouts are applied
separately at the connection level (see :mod:`text2sql.db`).
"""

from __future__ import annotations

import re

# Keywords that mutate data or schema, or otherwise escape a read-only session.
# Matched as whole words, case-insensitively.
_FORBIDDEN = {
    "insert", "update", "delete", "drop", "create", "alter", "truncate",
    "replace", "merge", "attach", "detach", "copy", "install", "load",
    "pragma", "set", "call", "export", "import", "vacuum", "checkpoint",
}

_WORD = re.compile(r"[a-zA-Z_]+")
_LIMIT_TAIL = re.compile(r"\blimit\s+\d+\s*$", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    """Raised when a statement fails a read-only or shape check."""


def _strip_comments(sql: str) -> str:
    """Remove line and block comments so keywords cannot hide inside them."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


def validate(sql: str) -> str:
    """Return the single statement stripped of a trailing semicolon, or raise.

    A statement is rejected if it is empty, contains more than one statement, does
    not begin with SELECT or WITH, or contains any forbidden keyword.
    """
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("Empty SQL statement.")

    # One statement only: any remaining semicolon implies a second statement.
    if ";" in cleaned:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")

    lowered_words = [w.lower() for w in _WORD.findall(cleaned)]
    if not lowered_words or lowered_words[0] not in {"select", "with"}:
        raise UnsafeSQLError("Only SELECT (or WITH ... SELECT) queries are allowed.")

    forbidden = _FORBIDDEN.intersection(lowered_words)
    if forbidden:
        raise UnsafeSQLError(f"Forbidden keyword(s) in query: {', '.join(sorted(forbidden))}.")

    return cleaned


def enforce_limit(sql: str, row_limit: int) -> str:
    """Append a ``LIMIT`` clause when the query does not already end with one.

    Queries that specify their own trailing limit are left untouched; the executor
    additionally fetches at most ``row_limit`` rows as a backstop.
    """
    if _LIMIT_TAIL.search(sql):
        return sql
    return f"{sql}\nLIMIT {row_limit}"
