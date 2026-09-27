"""The tool-using agent loop.

The agent inspects the schema, asks the model for SQL, executes it against the
read-only database, and (when self-correction is enabled) feeds any execution
error back to the model for a bounded number of retries. Every run returns a
fully instrumented :class:`AgentResult` so the evaluation harness can measure
accuracy, retries, latency and token usage without re-running anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import AgentConfig
from .db import Database, QueryResult
from .llm import ChatClient, Message
from .prompts import (
    Strategy,
    build_correction_message,
    build_messages,
    extract_sql,
    to_messages,
)
from .sql_guard import UnsafeSQLError


@dataclass
class Attempt:
    """One model call plus the outcome of executing its SQL."""

    sql: str
    ok: bool
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class AgentResult:
    """Everything produced by a single question run."""

    question: str
    success: bool
    sql: str | None
    result: QueryResult | None
    attempts: list[Attempt] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def retries(self) -> int:
        """Number of extra attempts beyond the first."""
        return max(0, len(self.attempts) - 1)

    @property
    def total_tokens(self) -> int:
        return sum(a.prompt_tokens + a.completion_tokens for a in self.attempts)


class Agent:
    """Answers plain-English questions by generating and running SQL."""

    def __init__(
        self,
        db: Database,
        client: ChatClient,
        config: AgentConfig | None = None,
    ):
        self.db = db
        self.client = client
        self.config = config or AgentConfig()
        # Cached because the schema does not change within a session.
        self._schema = db.schema_text()

    def answer(
        self,
        question: str,
        strategy: Strategy = "schema_examples",
        self_correction: bool = True,
    ) -> AgentResult:
        """Run the loop for one question and return an instrumented result."""
        started = time.perf_counter()
        messages: list[Message] = to_messages(build_messages(question, self._schema, strategy))
        attempts: list[Attempt] = []

        # First attempt plus, when self-correction is on, up to max_retries more.
        max_attempts = 1 + (self.config.max_retries if self_correction else 0)

        final_result: QueryResult | None = None
        final_sql: str | None = None
        success = False

        for _ in range(max_attempts):
            response = self.client.chat(messages)
            sql = extract_sql(response.content)
            attempt = Attempt(
                sql=sql,
                ok=False,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )

            try:
                if not sql:
                    raise UnsafeSQLError("Model returned no SQL.")
                result = self.db.run(
                    sql,
                    row_limit=self.config.row_limit,
                    timeout_s=self.config.statement_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - errors feed the correction loop
                attempt.error = str(exc)
                attempts.append(attempt)
                # Record the failed turn so the model can see its own SQL, then the error.
                messages.append({"role": "assistant", "content": response.content})
                messages.append(build_correction_message(str(exc)))
                continue

            attempt.ok = True
            attempts.append(attempt)
            final_result, final_sql, success = result, result.sql, True
            break

        return AgentResult(
            question=question,
            success=success,
            sql=final_sql,
            result=final_result,
            attempts=attempts,
            latency_s=time.perf_counter() - started,
        )
