"""Command-line interface for asking the agent a single question."""

from __future__ import annotations

import argparse
import sys

from .agent import Agent, AgentResult
from .chart import render_bar_chart
from .config import DEFAULT_DB_PATH, AgentConfig, LLMConfig
from .db import Database
from .llm import OpenAIChatClient
from .prompts import STRATEGIES
from .sample_data import ensure


def format_table(columns: list[str], rows: list[tuple], max_rows: int = 50) -> str:
    """Render a small result set as a fixed-width text table."""
    if not columns:
        return "(no columns)"
    shown = rows[:max_rows]
    widths = [len(c) for c in columns]
    for row in shown:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    def fmt(values) -> str:
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values))

    lines = [fmt(columns), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in shown)
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def _print_result(result: AgentResult, chart_path: str | None) -> None:
    if result.sql:
        print("\nSQL:\n" + result.sql)
    if not result.success:
        last_error = result.attempts[-1].error if result.attempts else "no attempts made"
        print(f"\nCould not answer the question. Last error: {last_error}")
        return

    assert result.result is not None
    print("\nResult:")
    print(format_table(result.result.columns, result.result.rows))
    print(
        f"\n{result.result.row_count} row(s), {result.retries} retr{'y' if result.retries == 1 else 'ies'}, "
        f"{result.total_tokens} tokens, {result.latency_s:.2f}s"
    )

    if chart_path:
        written = render_bar_chart(result.result, chart_path, title=result.question)
        print(f"\nChart written to {written}" if written else "\n(Result is not chartable.)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text2sql",
        description="Answer plain-English analytical questions over a DuckDB database.",
    )
    parser.add_argument("question", help="The question to answer, in plain English.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the DuckDB file.")
    parser.add_argument(
        "--strategy", choices=STRATEGIES, default="schema_examples",
        help="Prompting strategy (default: schema_examples).",
    )
    parser.add_argument(
        "--no-self-correction", action="store_true",
        help="Disable retrying after an execution error.",
    )
    parser.add_argument("--model", help="Override the model name (else T2SQL_MODEL / default).")
    parser.add_argument("--chart", metavar="PATH", help="Write a bar chart of the result to PATH.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    llm_config = LLMConfig.from_env()
    if args.model:
        llm_config = LLMConfig(**{**llm_config.__dict__, "model": args.model})

    # Build the bundled database on first use so the CLI works out of the box.
    if args.db == str(DEFAULT_DB_PATH):
        ensure()

    try:
        with Database(args.db) as db:
            agent = Agent(db, OpenAIChatClient(llm_config), AgentConfig())
            result = agent.answer(
                args.question,
                strategy=args.strategy,
                self_correction=not args.no_self_correction,
            )
            _print_result(result, args.chart)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
