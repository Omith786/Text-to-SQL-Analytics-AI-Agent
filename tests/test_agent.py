"""Tests for the agent loop, using a deterministic fake LLM (no network)."""

from text2sql.agent import Agent
from text2sql.config import AgentConfig
from text2sql.llm import FakeChatClient


def test_agent_answers_on_first_try(db):
    client = FakeChatClient(replies=["```sql\nSELECT COUNT(*) AS n FROM vessels\n```"])
    agent = Agent(db, client)
    result = agent.answer("How many vessels?", self_correction=False)

    assert result.success
    assert result.retries == 0
    assert result.result.rows[0][0] == 60
    assert result.total_tokens > 0


def test_agent_self_corrects_after_error(db):
    # First reply references a bad column; second reply is valid.
    client = FakeChatClient(replies=[
        "```sql\nSELECT bad_col FROM vessels\n```",
        "```sql\nSELECT COUNT(*) AS n FROM vessels\n```",
    ])
    agent = Agent(db, client, AgentConfig())
    result = agent.answer("How many vessels?", self_correction=True)

    assert result.success
    assert result.retries == 1
    assert len(result.attempts) == 2
    assert result.attempts[0].ok is False
    assert result.attempts[1].ok is True


def test_agent_gives_up_after_max_retries(db):
    config = AgentConfig()
    # Always return invalid SQL: attempts should equal 1 + max_retries.
    client = FakeChatClient(responder=lambda msgs: "```sql\nSELECT bad_col FROM vessels\n```")
    agent = Agent(db, client, config)
    result = agent.answer("anything", self_correction=True)

    assert result.success is False
    assert len(result.attempts) == 1 + config.max_retries


def test_agent_single_shot_does_not_retry(db):
    client = FakeChatClient(responder=lambda msgs: "```sql\nSELECT bad_col FROM vessels\n```")
    agent = Agent(db, client)
    result = agent.answer("anything", self_correction=False)

    assert result.success is False
    assert len(result.attempts) == 1


def test_agent_rejects_write_sql(db):
    client = FakeChatClient(replies=["```sql\nDELETE FROM vessels\n```"])
    agent = Agent(db, client)
    result = agent.answer("delete stuff", self_correction=False)

    assert result.success is False
    assert "SELECT" in result.attempts[0].error or "allowed" in result.attempts[0].error.lower()
