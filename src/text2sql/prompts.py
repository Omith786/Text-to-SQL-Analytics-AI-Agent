"""Prompt construction, few-shot examples and SQL extraction.

Two prompting strategies are supported and compared in the evaluation harness:

``zero_shot``
    The schema and the question only.
``schema_examples``
    The schema plus a handful of worked question/SQL pairs.

Whether the agent retries after an execution error (self-correction) is a property
of the loop, not the prompt, and is toggled separately.
"""

from __future__ import annotations

import re
from typing import Sequence

from .llm import Message

Strategy = str
STRATEGIES: tuple[Strategy, ...] = ("zero_shot", "schema_examples")

_SYSTEM = (
    "You are a careful data analyst that writes DuckDB SQL. "
    "Given a database schema and a question, respond with exactly one read-only "
    "SELECT query that answers it. Use only tables and columns from the schema. "
    "Return the SQL inside a single ```sql code block and nothing else."
)

# Worked examples used by the schema_examples strategy. They cover a join, an
# aggregate and a date filter so the model sees the main shapes it will need.
_FEW_SHOT: list[tuple[str, str]] = [
    (
        "How many vessels are there of each vessel type?",
        "SELECT vessel_type, COUNT(*) AS vessel_count\n"
        "FROM vessels\nGROUP BY vessel_type\nORDER BY vessel_count DESC",
    ),
    (
        "Which port had the most port calls?",
        "SELECT p.name, COUNT(*) AS calls\n"
        "FROM port_calls c\nJOIN ports p ON p.port_id = c.port_id\n"
        "GROUP BY p.name\nORDER BY calls DESC\nLIMIT 1",
    ),
    (
        "What was the average crude oil price in US Gulf Coast during 2024?",
        "SELECT AVG(price_usd) AS avg_price\n"
        "FROM energy_prices\n"
        "WHERE product = 'Crude Oil' AND region = 'US Gulf Coast'\n"
        "  AND price_date BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'",
    ),
]

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_START = re.compile(r"\b(select|with)\b", re.IGNORECASE)


def build_messages(question: str, schema: str, strategy: Strategy) -> list[Message]:
    """Assemble the chat messages for a first attempt at ``question``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from {STRATEGIES}.")

    user_parts = [f"Database schema:\n{schema}"]
    if strategy == "schema_examples":
        examples = "\n\n".join(
            f"Question: {q}\nSQL:\n{sql}" for q, sql in _FEW_SHOT
        )
        user_parts.append("Examples:\n" + examples)
    user_parts.append(f"Question: {question}\nSQL:")

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_correction_message(error: str) -> Message:
    """A user turn that hands the previous execution error back to the model."""
    return {
        "role": "user",
        "content": (
            f"That query failed with this error:\n{error}\n"
            "Rewrite the single SELECT query to fix it. "
            "Return only the corrected SQL in a ```sql code block."
        ),
    }


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply.

    Prefers a fenced ```sql block. If none is present, falls back to the text
    from the first SELECT or WITH keyword onwards, but only when that candidate
    actually contains a SELECT (so prose such as "I cannot help with that" is not
    mistaken for SQL). Returns an empty string when no SQL can be found, which the
    agent treats as a failed attempt.
    """
    match = _SQL_BLOCK.search(text)
    if match:
        return match.group(1).strip()

    start = _SQL_START.search(text)
    if start:
        candidate = text[start.start():].strip()
        if "select" in candidate.lower():
            return candidate
    return ""


def to_messages(history: Sequence[Message]) -> list[Message]:
    """Return a mutable copy of a message history."""
    return [dict(m) for m in history]
