"""Evaluation harness for the Text-to-SQL agent.

Loads a labelled question set (question, gold SQL, difficulty), runs the agent
under a matrix of prompting strategies and self-correction settings, and measures:

* execution accuracy: does the agent's result set match the gold query's result
  set (order-insensitive, with float rounding)?
* retries used, wall-clock latency and token usage per question.

Results are cached to JSON so re-plotting or aggregating never re-runs the model.
The comparison metric is execution accuracy, which is robust to a query being
written differently from the gold SQL as long as it returns the same rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# Allow running as `python eval/harness.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.agent import Agent  # noqa: E402
from text2sql.config import DEFAULT_DB_PATH, AgentConfig, LLMConfig  # noqa: E402
from text2sql.db import Database, QueryResult  # noqa: E402
from text2sql.llm import ChatClient, OpenAIChatClient  # noqa: E402
from text2sql.prompts import STRATEGIES  # noqa: E402
from text2sql.sample_data import ensure  # noqa: E402

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
RESULTS_DIR = Path(__file__).with_name("results")

# Rounding applied to numeric cells before comparison, to absorb floating-point
# noise between an agent query and the gold query.
FLOAT_DP = 2


@dataclass
class Question:
    id: str
    question: str
    gold_sql: str
    difficulty: str


@dataclass
class RunRecord:
    question_id: str
    difficulty: str
    strategy: str
    self_correction: bool
    correct: bool
    success: bool
    retries: int
    tokens: int
    latency_s: float


@dataclass
class Summary:
    strategy: str
    self_correction: bool
    n: int
    execution_accuracy: float
    mean_retries: float
    mean_latency_s: float
    mean_tokens: float
    by_difficulty: dict[str, float] = field(default_factory=dict)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    raw = yaml.safe_load(path.read_text())
    return [Question(**item) for item in raw]


def _normalise(result: QueryResult | None) -> set[tuple] | None:
    """Turn a result into an order-insensitive multiset of rounded rows."""
    if result is None:
        return None

    def cell(value):
        if isinstance(value, float):
            return round(value, FLOAT_DP)
        return value

    # Counter-style multiset via sorted tuple of (row, count) is awkward with
    # unhashable cells, so we stringify rows for a stable, comparable key.
    rows = sorted(tuple(cell(v) for v in row) for row in result.rows)
    return set((r, rows.count(r)) for r in rows)


def evaluate(
    client: ChatClient,
    db_path: str | Path = DEFAULT_DB_PATH,
    strategies: tuple[str, ...] = STRATEGIES,
    self_correction_modes: tuple[bool, ...] = (True, False),
    questions: list[Question] | None = None,
    limit: int | None = None,
) -> list[RunRecord]:
    """Run the full matrix and return one :class:`RunRecord` per cell."""
    questions = questions or load_questions()
    if limit is not None:
        questions = questions[:limit]

    records: list[RunRecord] = []
    with Database(db_path) as db:
        agent = Agent(db, client, AgentConfig())
        # Gold result sets are computed once and reused across every configuration.
        gold: dict[str, set | None] = {}
        for q in questions:
            gold_result = db.run(q.gold_sql, row_limit=AgentConfig().row_limit,
                                 timeout_s=AgentConfig().statement_timeout_s)
            gold[q.id] = _normalise(gold_result)

        for strategy in strategies:
            for self_correction in self_correction_modes:
                for q in questions:
                    res = agent.answer(q.question, strategy=strategy,
                                       self_correction=self_correction)
                    correct = res.success and _normalise(res.result) == gold[q.id]
                    records.append(RunRecord(
                        question_id=q.id,
                        difficulty=q.difficulty,
                        strategy=strategy,
                        self_correction=self_correction,
                        correct=correct,
                        success=res.success,
                        retries=res.retries,
                        tokens=res.total_tokens,
                        latency_s=round(res.latency_s, 3),
                    ))
    return records


def summarise(records: list[RunRecord]) -> list[Summary]:
    """Aggregate run records into per-configuration summaries."""
    summaries: list[Summary] = []
    configs = sorted({(r.strategy, r.self_correction) for r in records})
    for strategy, self_correction in configs:
        subset = [r for r in records if r.strategy == strategy and r.self_correction == self_correction]
        by_diff: dict[str, float] = {}
        for diff in sorted({r.difficulty for r in subset}):
            diff_rows = [r for r in subset if r.difficulty == diff]
            by_diff[diff] = round(sum(r.correct for r in diff_rows) / len(diff_rows), 4)
        summaries.append(Summary(
            strategy=strategy,
            self_correction=self_correction,
            n=len(subset),
            execution_accuracy=round(sum(r.correct for r in subset) / len(subset), 4),
            mean_retries=round(statistics.mean(r.retries for r in subset), 3),
            mean_latency_s=round(statistics.mean(r.latency_s for r in subset), 3),
            mean_tokens=round(statistics.mean(r.tokens for r in subset), 1),
            by_difficulty=by_diff,
        ))
    return summaries


def save_results(records: list[RunRecord], summaries: list[Summary], model: str,
                 out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "records": [asdict(r) for r in records],
        "summaries": [asdict(s) for s in summaries],
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def render_plots(summaries: list[Summary], out_dir: Path = RESULTS_DIR) -> Path | None:
    """Bar chart of execution accuracy per configuration. Returns the image path."""
    if not summaries:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{s.strategy}\n{'self-corr' if s.self_correction else 'single-shot'}" for s in summaries]
    values = [s.execution_accuracy for s in summaries]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#3b6ea5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Execution accuracy")
    ax.set_title("Text-to-SQL execution accuracy by configuration")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accuracy.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _print_summaries(summaries: list[Summary]) -> None:
    header = f"{'strategy':<16}{'self-corr':<11}{'acc':>7}{'retries':>9}{'latency':>9}{'tokens':>9}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s.strategy:<16}{str(s.self_correction):<11}{s.execution_accuracy:>7.2%}"
              f"{s.mean_retries:>9.2f}{s.mean_latency_s:>8.2f}s{s.mean_tokens:>9.0f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Text-to-SQL evaluation harness.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.db == str(DEFAULT_DB_PATH):
        ensure()

    config = LLMConfig.from_env()
    client = OpenAIChatClient(config)
    print(f"Evaluating model '{config.model}' at {config.base_url} ...")

    records = evaluate(client, db_path=args.db, limit=args.limit)
    summaries = summarise(records)
    _print_summaries(summaries)

    path = save_results(records, summaries, model=config.model)
    print(f"\nSaved results to {path}")
    if not args.no_plots:
        img = render_plots(summaries)
        if img:
            print(f"Saved plot to {img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
