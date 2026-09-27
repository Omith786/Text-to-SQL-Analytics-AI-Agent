"""DuckDB access layer: read-only connections, schema inspection and safe execution.

Queries are opened read-only so the engine itself rejects writes, giving a second
line of defence behind :mod:`text2sql.sql_guard`. Each query runs under a wall-clock
timeout enforced with DuckDB's interrupt mechanism, and results are capped at a
configurable row count.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .sql_guard import enforce_limit, validate


class QueryTimeout(RuntimeError):
    """Raised when a query exceeds its wall-clock budget."""


@dataclass
class QueryResult:
    """A tabular query result plus the exact SQL that produced it."""

    columns: list[str]
    rows: list[tuple]
    sql: str

    @property
    def row_count(self) -> int:
        return len(self.rows)


class Database:
    """A read-only DuckDB session over a single database file."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if not Path(self.db_path).exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. Build it with "
                "`python -m data.build_sample_db`."
            )
        # read_only prevents any accidental or malicious mutation at the engine level.
        self._con = duckdb.connect(self.db_path, read_only=True)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def schema_text(self) -> str:
        """Return a compact, LLM-friendly description of every table and column.

        The format lists each table with its columns and types, which is enough for
        a model to write correct joins without overwhelming a small context window.
        """
        rows = self._con.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()

        columns_by_table: dict[str, list[str]] = {}
        for table_name, column_name, data_type in rows:
            columns_by_table.setdefault(table_name, []).append(f"  {column_name} {data_type}")

        blocks = [
            "TABLE {} (\n{}\n)".format(table, ",\n".join(cols))
            for table, cols in columns_by_table.items()
        ]
        return "\n\n".join(blocks)

    def run(self, sql: str, row_limit: int, timeout_s: float) -> QueryResult:
        """Validate, bound and execute ``sql``, returning a :class:`QueryResult`.

        Raises :class:`~text2sql.sql_guard.UnsafeSQLError` for disallowed SQL,
        :class:`QueryTimeout` when the statement runs too long, and DuckDB errors
        (for example unknown columns) unchanged so the agent can self-correct.
        """
        safe_sql = enforce_limit(validate(sql), row_limit)

        # DuckDB has no statement_timeout setting, so a watchdog timer interrupts the
        # connection if the query overruns. interrupt() causes execute() to raise.
        timed_out = threading.Event()

        def _interrupt() -> None:
            timed_out.set()
            self._con.interrupt()

        watchdog = threading.Timer(timeout_s, _interrupt)
        watchdog.start()
        try:
            cursor = self._con.execute(safe_sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(row_limit)
        except duckdb.InterruptException as exc:  # pragma: no cover - timing dependent
            raise QueryTimeout(f"Query exceeded {timeout_s:.0f}s and was cancelled.") from exc
        except duckdb.Error:
            if timed_out.is_set():
                raise QueryTimeout(f"Query exceeded {timeout_s:.0f}s and was cancelled.")
            raise
        finally:
            watchdog.cancel()

        return QueryResult(columns=columns, rows=list(rows), sql=safe_sql)
