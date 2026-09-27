"""Tests for the evaluation harness, using a fake LLM keyed to the gold SQL."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import harness  # noqa: E402
from text2sql.llm import FakeChatClient  # noqa: E402


def _oracle_client(questions):
    """A fake client that answers each question with its own gold SQL."""
    by_question = {q.question: q.gold_sql for q in questions}

    def responder(messages):
        user = messages[-1]["content"]
        for question, gold in by_question.items():
            if question in user:
                return f"```sql\n{gold}\n```"
        return "```sql\nSELECT 1\n```"

    return FakeChatClient(responder=responder)


def test_gold_questions_load():
    questions = harness.load_questions()
    assert len(questions) >= 50
    assert all(q.difficulty in {"easy", "medium", "hard"} for q in questions)
    assert len({q.id for q in questions}) == len(questions)  # ids are unique


def test_oracle_scores_full_accuracy(sample_db_path):
    questions = harness.load_questions()[:6]
    client = _oracle_client(questions)
    records = harness.evaluate(
        client,
        db_path=sample_db_path,
        strategies=("zero_shot",),
        self_correction_modes=(False,),
        questions=questions,
    )
    assert len(records) == len(questions)
    assert all(r.correct for r in records)


def test_summarise_reports_accuracy(sample_db_path):
    questions = harness.load_questions()[:4]
    client = _oracle_client(questions)
    records = harness.evaluate(
        client, db_path=sample_db_path,
        strategies=("zero_shot",), self_correction_modes=(False,), questions=questions,
    )
    summaries = harness.summarise(records)
    assert len(summaries) == 1
    assert summaries[0].execution_accuracy == 1.0
    assert set(summaries[0].by_difficulty).issubset({"easy", "medium", "hard"})


def test_wrong_answer_scored_incorrect(sample_db_path):
    questions = harness.load_questions()[:2]
    # Always returns a valid but wrong query.
    client = FakeChatClient(responder=lambda m: "```sql\nSELECT 999 AS n\n```")
    records = harness.evaluate(
        client, db_path=sample_db_path,
        strategies=("zero_shot",), self_correction_modes=(False,), questions=questions,
    )
    assert all(r.success for r in records)      # it executed
    assert not any(r.correct for r in records)  # but the answer is wrong
