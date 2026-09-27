"""Tests for the LLM client abstraction (fake client only, no network)."""

import pytest

from text2sql.llm import FakeChatClient


def test_fake_returns_scripted_replies_in_order():
    client = FakeChatClient(replies=["first", "second"])
    assert client.chat([{"role": "user", "content": "a"}]).content == "first"
    assert client.chat([{"role": "user", "content": "b"}]).content == "second"


def test_fake_records_calls():
    client = FakeChatClient(replies=["x"])
    client.chat([{"role": "user", "content": "hello there"}])
    assert client.calls[-1][-1]["content"] == "hello there"


def test_fake_responder_takes_precedence():
    client = FakeChatClient(responder=lambda msgs: "computed")
    assert client.chat([{"role": "user", "content": "q"}]).content == "computed"


def test_fake_raises_when_out_of_replies():
    client = FakeChatClient(replies=["only"])
    client.chat([{"role": "user", "content": "1"}])
    with pytest.raises(AssertionError):
        client.chat([{"role": "user", "content": "2"}])


def test_token_counts_are_populated():
    client = FakeChatClient(replies=["one two three"])
    resp = client.chat([{"role": "user", "content": "a b"}])
    assert resp.prompt_tokens == 2
    assert resp.completion_tokens == 3
    assert resp.total_tokens == 5
